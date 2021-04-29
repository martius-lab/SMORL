import copy
import warnings

import numpy as np
from gym.spaces import Dict

import rlkit.torch.pytorch_util as ptu
from multiworld.core.multitask_env import MultitaskEnv
from rlkit.envs.wrappers import ProxyEnv


_GOAL_SAMPLING_MODES = ('oracle_prior', 'env_goal', 'custom_goal_sampler')
_REWARD_TYPES = ('state_distance', 'sparse', 'pos_sparse',
                 'state_distance_and_bounty')


class UnstructuredGTWrappedEnv(ProxyEnv, MultitaskEnv):
    """This class wraps environments with a GT stuctured representation, but
    outputs unstructured observations

    This class adheres to the "Silent Multitask Env" semantics: on reset,
    it resamples a goal.
    """
    def __init__(self,
                 wrapped_env,
                 reward_params=None,
                 goal_sampling_mode='oracle_prior',
                 custom_goal_sampler=None,
                 zero_goals_when_goal_sampler_none=False,
                 done_on_success=False,
                 done_on_episode_end=False,
                 max_episode_steps=None):
        if reward_params is None:
            reward_params = dict()
        super().__init__(wrapped_env)
        self.device = ptu.device

        orig_obs_space = self.wrapped_env.observation_space.spaces
        self.state_dim = orig_obs_space['state_observation'].shape[0]

        self.reward_params = reward_params
        self.reward_type = self.reward_params.get('type', 'state_distance')
        assert self.reward_type in _REWARD_TYPES, \
            'Unknown reward type {}'.format(self.reward_type)
        self.norm_order = self.reward_params.get('norm_order', 2)
        if self.reward_type in ('sparse', 'pos_sparse',
                                'state_distance_and_bounty'):
            self.success_threshold = self.reward_params.get('threshold')
            assert self.success_threshold is not None
        if self.reward_type == 'state_distance_and_bounty':
            self.bounty = self.reward_params.get('bounty')
            assert self.bounty is not None
        self.structured_rewards = self.reward_params.get('structured_rewards',
                                                         False)

        self.observation_space = Dict([
            ('state_observation', orig_obs_space['state_observation']),
            ('desired_goal', orig_obs_space['state_desired_goal'])
        ])

        self.desired_goal = None
        self._initial_obs = None

        assert goal_sampling_mode in _GOAL_SAMPLING_MODES
        self._goal_sampling_mode = goal_sampling_mode
        self._custom_goal_sampler = custom_goal_sampler
        self._zero_goals_when_goal_sampler_none = zero_goals_when_goal_sampler_none
        self._done_on_success = done_on_success
        self._done_on_episode_end = done_on_episode_end
        self._elapsed_steps = None
        self._max_episode_steps = max_episode_steps

        self.env_params = dict(reward_params=reward_params,
                               goal_sampling_mode=goal_sampling_mode,
                               custom_goal_sampler=custom_goal_sampler,
                               zero_goals_when_goal_sampler_none=zero_goals_when_goal_sampler_none)

    def reset(self):
        self._elapsed_steps = 0
        obs = self.wrapped_env.reset()

        goal = self.sample_goal()['desired_goal']
        self.set_goal(goal)
        self._initial_obs = obs

        return self._wrap_obs(obs)

    def sample_goals(self, batch_size):
        if self.goal_sampling_mode == 'custom_goal_sampler':
            goals = self._custom_goal_sampler(batch_size)
            if goals is None:
                if self._zero_goals_when_goal_sampler_none:
                    example_goal = self._sample_oracle_prior(batch_size)
                    goals = {
                        'desired_goal': np.zeros_like(example_goal)
                    }
                else:
                    raise ValueError('Custom goal sampler returned `None`')

            return goals
        elif self._goal_sampling_mode == 'oracle_prior':
            goals = self._sample_oracle_prior(batch_size)
        elif self._goal_sampling_mode == 'env_goal':
            assert batch_size == 1
            goals = self.wrapped_env.get_goal()['state_desired_goal'][np.newaxis]
        else:
            raise RuntimeError('Invalid: {}'.format(self._goal_sampling_mode))

        return {'desired_goal': goals}

    def _sample_oracle_prior(self, batch_size):
        space = self.wrapped_env.observation_space.spaces['state_desired_goal']
        return np.stack([space.sample() for _ in range(batch_size)])

    def step(self, action):
        self._elapsed_steps += 1
        obs, reward, done, info = self.wrapped_env.step(action)

        new_obs = self._wrap_obs(obs, action)
        info = self._wrap_info(info, new_obs)

        reward = self.compute_reward(action, new_obs)
        done = self.compute_done(done, reward)

        return new_obs, reward, done, info

    def _wrap_obs(self, obs, action=None):
        obs['desired_goal'] = self.desired_goal
        return obs

    def _wrap_info(self, info, obs):
        diff = obs['state_observation'] - obs['desired_goal']
        info['state_distance'] = np.linalg.norm(diff, ord=self.norm_order)

        return info

    """
    Multitask functions
    """
    def get_goal(self):
        return self.desired_goal

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_rewards(actions, next_obs)[0]

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_observation']
        desired_goals = obs['desired_goal']

        if not self.structured_rewards:
            dist = np.linalg.norm(desired_goals - achieved_goals,
                                  ord=self.norm_order, axis=1)
            if self.reward_type == 'state_distance':
                return -dist
            elif self.reward_type == 'sparse':
                return -1.0 * (dist >= self.success_threshold)
            elif self.reward_type == 'pos_sparse':
                return (dist < self.success_threshold).astype(np.float32)
            elif self.reward_type == 'state_distance_and_bounty':
                success = (dist < self.success_threshold).astype(np.float32)
                return -dist + self.bounty * success
        else:
            bs = achieved_goals.shape[0]
            achieved_goals = achieved_goals.reshape(bs, -1, 2)
            desired_goals = desired_goals.reshape(bs, -1, 2)
            dists = np.linalg.norm(desired_goals - achieved_goals,
                                   ord=self.norm_order, axis=-1)
            if self.reward_type == 'state_distance':
                return -np.sum(dist, axis=-1)
            elif self.reward_type == 'sparse':
                no_success = (dists >= self.success_threshold).astype(np.float32)
                return -no_success.mean(axis=-1)
            elif self.reward_type == 'pos_sparse':
                success = (dists < self.success_threshold).astype(np.float32)
                return success.mean(axis=-1)
            elif self.reward_type == 'state_distance_and_bounty':
                success = (dists < self.success_threshold).astype(np.float32)
                return (-np.sum(dists, axis=-1)
                        + self.bounty * success.sum(axis=-1))

    def compute_done(self, done, reward):
        if self._done_on_success:
            if self.reward_type == 'pos_sparse':
                return reward == 1.0 or done
            elif self.reward_type == 'sparse':
                return reward == 0.0 or done

        if (self._done_on_episode_end
                and self._elapsed_steps >= self._max_episode_steps):
            return True

        return done

    @property
    def goal_dim(self):
        return self.z_where_dim

    def set_goal(self, goal):
        """Assume goal contains both image_desired_goal and any goals required
        for wrapped envs

        :param goal:
        :return:
        """
        self.desired_goal = goal

    @property
    def custom_goal_sampler(self):
        return self._custom_goal_sampler

    @custom_goal_sampler.setter
    def custom_goal_sampler(self, new_custom_goal_sampler):
        assert self.custom_goal_sampler is None, (
            "Cannot override custom goal setter"
        )
        self._custom_goal_sampler = new_custom_goal_sampler

    @property
    def goal_sampling_mode(self):
        return self._goal_sampling_mode

    @goal_sampling_mode.setter
    def goal_sampling_mode(self, mode):
        assert mode in _GOAL_SAMPLING_MODES, \
            'Invalid env mode `{}`'.format(mode)
        self._goal_sampling_mode = mode

    """
    Other functions
    """
    def get_diagnostics(self, paths, **kwargs):
        return super().get_diagnostics()

    def __getstate__(self):
        """Custom state retrieval for pickling

        Needed because custom goal sampler may reference the replay buffer,
        which may contain multiprocessing objects, which can not be pickled.
        """
        state = super().__getstate__()
        state = copy.copy(state)
        state['_custom_goal_sampler'] = None
        warnings.warn(('UnstructuredGTWrappedEnv.custom_goal_sampler is '
                       'not saved.'))
        return state

    def __setstate__(self, state):
        warnings.warn(('UnstructuredGTWrappedEnv.custom_goal_sampler is '
                       'not saved.'))
        super().__setstate__(state)
