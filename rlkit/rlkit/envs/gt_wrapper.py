import collections
import copy
import random
import warnings

import torch
import copy
import cv2
import numpy as np
from gym.spaces import Box, Dict
import rlkit.torch.pytorch_util as ptu
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
from rlkit.envs.wrappers import ProxyEnv


class GTWrappedEnv(ProxyEnv, MultitaskEnv):
    """This class wraps environment with a GT stuctured representations.
    This class adheres to the "Silent Multitask Env" semantics: on reset,
    it resamples a goal.
    """
    def __init__(self,
                 wrapped_env,
                 z_where_dim,
                 z_depth_dim,
                 max_n_objects,
                 sub_task_horizon,
                 solved_goal_threshold=0.05,
                 sample_from_true_prior=True,
                 reward_params=None,
                 goal_sampling_mode="z_where_prior",
                 norm_order=2,
                 done_on_success=False,
                 track_success_rates=False,
                 goal_prioritization=False,
                 success_rate_coeff=0.95):
        if reward_params is None:
            reward_params = dict()
        super().__init__(wrapped_env)
        self.device = ptu.device
        self.max_n_objects = max_n_objects # used in representation shaping
        self.n_objects_max = self.wrapped_env.n_objects_max + 1 # from env
        self.z_what_dim = self.n_objects_max
        self.match_thresh = 0.5
        self.z_where_dim = z_where_dim
        self.z_depth_dim = z_depth_dim
        self.sub_task_horizon = sub_task_horizon
        self.sample_from_true_prior = sample_from_true_prior
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", 'object_distance')
        self.norm_order = self.reward_params.get("norm_order", norm_order)
        if self.reward_type in ('sparse', 'pos_sparse'):
            self.success_threshold = self.reward_params.get('threshold')
            solved_goal_threshold = self.success_threshold
            assert self.success_threshold is not None
        self.solved_goal_threshold = solved_goal_threshold
        self.reward_min_variance = self.reward_params.get("min_variance", 0)
        self.z_dim = (self.z_what_dim + self.z_where_dim + self.z_depth_dim)*self.max_n_objects + 1
        self.z_goal_dim = self.z_what_dim + self.z_where_dim + 1
        z_what_space = Box(
            -10 * np.ones(self.z_what_dim),
            10 * np.ones(self.z_what_dim),
            dtype=np.float32,
        )
        z_where_space = Box(
            -1 * np.ones(self.z_where_dim),
            1 * np.ones(self.z_where_dim),
            dtype=np.float32,
        )
        z_depth_space = Box(
            -1 * np.ones(self.z_depth_dim),
            1 * np.ones(self.z_depth_dim),
            dtype=np.float32,
        )
        z_space = Box(
            -10 * np.ones(self.z_dim),
            10 * np.ones(self.z_dim),
            dtype=np.float32,
        )
        z_goal_space = Box(
            -10 * np.ones(self.z_goal_dim),
            10 * np.ones(self.z_goal_dim),
            dtype=np.float32,
        )

        spaces = copy.copy(self.wrapped_env.observation_space.spaces)
        spaces['z_what'] = z_what_space
        spaces['z_where'] = z_where_space
        spaces['z_depth'] = z_depth_space
        spaces['latent_obs_vector'] = z_space
        spaces['goal_vector'] = z_goal_space
        spaces['desired_goal'] = z_where_space
        spaces['achieved_goal'] = z_where_space
        self.observation_space = Dict(spaces)
        self.observation_space.flat_dim = (self.z_what_dim + self.z_where_dim + self.z_depth_dim)*self.max_n_objects + self.z_what_dim + self.z_where_dim + 2
        self.desired_goal = None
        self.match_thresh = 0.01
        self._initial_obs = None
        self._custom_goal_sampler = None
        self._goal_sampling_mode = goal_sampling_mode
        self._done_on_success = done_on_success
        self._track_success_rates = track_success_rates
        if self._track_success_rates:
            self._success_rate_coeff = success_rate_coeff
            self._successes = {}
            self._attempts = {}
        self._goal_prioritization = goal_prioritization
        if self._goal_prioritization:
            assert self._track_success_rates

        self.t = 0
        self.reset_count = 0

        self.env_params = dict(z_where_dim=z_where_dim,
                               z_depth_dim=z_depth_dim,
                               max_n_objects=max_n_objects,
                               sub_task_horizon=sub_task_horizon,
                               sample_from_true_prior=sample_from_true_prior,
                               reward_params=reward_params,
                               goal_sampling_mode=goal_sampling_mode,
                               norm_order=norm_order,
                               solved_goal_threshold=solved_goal_threshold)

    def reset(self):
        self.reset_count += 1

        self.wrapped_env.reset()
        zero_action = np.zeros((2,))
        obs, _, _, _ = self.wrapped_env.step(zero_action)

        self.n_objects = self.wrapped_env.n_objects + 1
        self.obj_idx = np.random.choice(self.n_objects_max, self.n_objects, replace=False)

        goal, obs = self.sample_goal(obs)
        self.set_goal(goal)
        self._initial_obs = obs
        self.t = 0

        return self._update_obs(obs)

    def get_goal_from_gt(self, states, mode="z_where_prior"):
        states = states.reshape(-1, 2)[:self.n_objects]
        # z_what = np.eye(self.z_what_dim)[self.obj_idx]
        z_what = np.eye(self.z_what_dim)
        if mode == "z_where_prior":
            z_where = self._sample_z_where_prior(self.n_objects)
        else:
            z_where = states

        goal_vectors = np.concatenate((np.arange(self.n_objects)[:, None],
                                       z_what, z_where), axis=1)
        if (mode == "z_where_prior" and self._goal_prioritization
                and self.reset_count > 10):
            probs = self._get_sampling_probs(z_what)
            k = np.random.choice(self.n_objects, p=probs)
        else:
            k = np.random.randint(self.n_objects)
        # k = 1
        goal_vector = goal_vectors[k]
        z_what_k = z_what[k]
        z_where_k = z_where[k]
        goal = {"goal_vector": goal_vector,
                "goal_vectors": goal_vectors,
                "z_what_goals": z_what,
                "z_where_goals": z_where,
                "z_what_goal": z_what_k,
                "z_where_goal": z_where_k,
                "idx_goal": k}
        return goal

    def sample_goal(self, obs):
        if self._goal_sampling_mode == 'z_where_prior':
            latent_goal = self.get_goal_from_gt(obs['state_observation'])
        elif self._goal_sampling_mode == 'reset_of_env':
            goal_dict = self.wrapped_env.get_goal()
            states = goal_dict["state_desired_goal"]
            latent_goal = self.get_goal_from_gt(states, mode=self._goal_sampling_mode) # random_goal from goal state
        elif self._goal_sampling_mode == 'current_state':
            latent_goal = self.get_goal_from_gt(obs['state_observation'], mode=self._goal_sampling_mode)
        else:
            raise RuntimeError("Invalid: {}".format(self._goal_sampling_mode))
        return latent_goal, obs

    def sample_goals(self, batch_size, initial_goals):
        z_where_prior = self._sample_z_where_prior(batch_size)
        initial_goals[:, -self.z_where_dim:] = z_where_prior

        return initial_goals

    # def _sample_z_where_prior(self, batch_size):
    #     # table gt dist
    #     random = np.random.uniform(-0.1, 0.1, (batch_size, 2))
    #     random[:, 0] = random[:, 0] * 2
    #     n = np.array([[0.0, 0.6]]) + random
    #     return n

    def _sample_z_where_prior(self, batch_size):
        space = self.wrapped_env.observation_space.spaces['desired_goal']
        n = space.sample().reshape(-1, 2).shape[0]
        k = np.random.randint(n)
        z_wheres = np.stack([space.sample().reshape(-1, 2)[k]
                             for _ in range(batch_size)])
        return z_wheres

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        self.t += 1

        new_obs = self._update_obs(obs, action)
        self._update_info(info, new_obs)
        reward = self.compute_reward(
            action,
            {'latent_obs_vector': new_obs['latent_obs_vector'],
             'goal_vector': new_obs['goal_vector']}
        )
        if (self._track_success_rates and self.t % self.sub_task_horizon == 0):
            self._update_success_rates(new_obs, reward)

        if self._goal_sampling_mode == 'reset_of_env':
            # meta policy part
            if (self.t % self.sub_task_horizon == 0) and (self.t != 0):
                k_init = new_obs["idx_goal"]
                self.update_goal()
                new_obs = self._update_obs(obs, action)
                new_reward = self.compute_reward(
                    action,
                    {'latent_obs_vector': new_obs['latent_obs_vector'],
                     'goal_vector': new_obs['goal_vector']}
                )

                while self._compute_success_from_reward(new_reward):
                    self.update_goal()
                    new_obs = self._update_obs(obs, action)
                    new_reward = self.compute_reward(
                        action,
                        {'latent_obs_vector': new_obs['latent_obs_vector'],
                         'goal_vector': new_obs['goal_vector']}
                    )
                    if new_obs["idx_goal"] == k_init:
                        done = True
                        break
        else:
            done = self.compute_done(done, reward)

        return new_obs, reward, done, info

    def _compute_success_from_reward(self, reward):
        if self.reward_type == 'pos_sparse':
            return reward == 1
        elif self.reward_type == 'sparse':
            return reward == 0
        elif self.reward_type == 'object_distance':
            return (np.abs(reward) < self.solved_goal_threshold)

    def compute_done(self, done, reward):
        if self._done_on_success:
            if self.reward_type == 'pos_sparse':
                return reward == 1.0 or done
            elif self.reward_type == 'sparse':
                return reward == 0.0 or done

        return done

    def update_goal(self):
        goal = self.desired_goal
        goal_vectors = goal["goal_vectors"]
        z_where = goal["z_where_goals"]
        z_what = goal["z_what_goals"]
        n_objects = goal_vectors.shape[0]
        k = goal["idx_goal"]
        k = (k+1) % n_objects
        goal_vector = goal_vectors[k]
        z_what_k = z_what[k]
        z_where_k = z_where[k]
        goal = {"goal_vectors": goal_vectors,
                "z_what_goals": z_what,
                "z_where_goals": z_where,
                "goal_vector": goal_vector,
                "z_what_goal": z_what_k,
                "z_where_goal": z_where_k,
                "idx_goal": k}
        self.desired_goal = goal

    def _update_obs(self, obs, action=None):
        z_where = obs['state_observation'].reshape(-1, 2)
        z_where = z_where[:self.n_objects, :]
        # z_what = np.eye(self.z_what_dim)[self.obj_idx]
        z_what = np.eye(self.z_what_dim)
        z_depth = np.zeros((self.n_objects, 1))
        representation = dict(z_what=z_what, z_where=z_where, z_depth=z_depth)
        obs_vector = dict2vector(representation, self.max_n_objects)
        obs = {**obs, **representation, **self.desired_goal}
        obs["latent_obs_vector"] = obs_vector

        return obs

    def _update_info(self, info, obs):
        k = obs["idx_goal"]
        z_where = obs["z_where"][k]
        z_where_goal = obs["z_where_goal"]
        dist = z_where - z_where_goal
        info["z_where_dist"] = np.linalg.norm(dist, ord=self.norm_order)

    def _update_moving_average(self, key, success, coeff):
        successes = self._successes.get(key, 0)
        self._successes[key] = success + coeff * successes
        attempts = self._attempts.get(key, 0)
        self._attempts[key] = 1 + coeff * attempts

    def _update_success_rates(self, obs, reward):
        success = self._compute_success_from_reward(reward)
        z_what_goal = np.argmax(self.desired_goal["z_what_goal"])
        self._update_moving_average(z_what_goal,
                                    1.0 if success else 0.0,
                                    self._success_rate_coeff)

    def _get_sampling_probs(self, z_whats):
        rates = []
        for z_what in z_whats:
            i = np.argmax(z_what)
            if i not in self._successes:
                rates.append(0)
            else:
                rates.append(self._successes[i] / self._attempts[i])

        fail_rates = 1 - np.array(rates)
        probs = (fail_rates + 0.05) / np.sum(fail_rates + 0.05)

        return probs

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
        latent_obs_vector = obs['latent_obs_vector']
        goal_vector = obs['goal_vector']
        k = goal_vector[:, 0]
        desired_goals = goal_vector[:, -self.z_where_dim:]
        achieved_goals = get_z_where_from_obs(latent_obs_vector,
                                              k.astype(np.int),
                                              self.max_n_objects,
                                              self.z_what_dim,
                                              self.z_where_dim)
        dist = np.linalg.norm(desired_goals - achieved_goals,
                              ord=self.norm_order, axis=1)
        if self.reward_type == 'object_distance':
            return -dist
        elif self.reward_type == 'sparse':
            return -1.0 * (dist >= self.success_threshold)
        elif self.reward_type == 'pos_sparse':
            return (dist < self.success_threshold).astype(np.float32)
        else:
            raise NotImplementedError('reward_type {}'
                                      .format(self.reward_type))

    def match_goals(self, latent_obs, z_goal):
        match_idx = z_goal[:, 0].astype(np.int)
        if len(match_idx) == 1 and len(latent_obs) != 1:
            match_idx = np.tile(match_idx, (len(latent_obs), 1))

        match = np.ones_like(match_idx, dtype=np.bool)

        return match, match_idx

    def extract_achieved_goals(self, latent_obs_vector, obj_indices):
        bs = len(latent_obs_vector)
        latent_obs = latent_obs_vector[:, 1:]

        zs = latent_obs.reshape((bs, self.max_n_objects, -1))

        goal_objects = zs[np.arange(bs), obj_indices, :self.z_what_dim + self.z_where_dim]

        goals = np.concatenate((obj_indices[:, None], goal_objects), axis=1)

        return goals

    @property
    def goal_dim(self):
        return self.z_where_dim

    def set_goal(self, goal):
        """
        Assume goal contains both image_desired_goal and any goals required for wrapped envs

        :param goal:
        :return:
        """
        self.desired_goal = goal

    def get_diagnostics(self, paths, **kwargs):
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in ["z_where_dist"]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))

        if self._track_success_rates:
            for i in range(self.n_objects_max):
                attempts = self._attempts.get(i, 0)
                success_rate = self._successes.get(i, 0) / max(attempts, 1)
                statistics["success_rate_{}".format(i)] = success_rate

            sampling_probs = self._get_sampling_probs(np.eye(self.z_what_dim))
            for i, prob in enumerate(sampling_probs):
                statistics["goal_sampling_prob_{}".format(i)] = prob

        return statistics

    """
    Other functions
    """
    @property
    def goal_sampling_mode(self):
        return self._goal_sampling_mode

    @goal_sampling_mode.setter
    def goal_sampling_mode(self, mode):
        assert mode in [
            'z_where_prior',
            'env',
            'reset_of_env',
            'current_state'
        ], "Invalid env mode"
        self._goal_sampling_mode = mode


def dict2vector(representation, max_n_objects):
    n_objects = representation["z_where"].shape[-2]
    z_n = np.concatenate((representation["z_what"],
                          representation["z_where"],
                          representation["z_depth"]), axis=1)
    z = np.zeros((max_n_objects, z_n.shape[1]))
    z[:n_objects, :] = z_n
    z_vector = np.concatenate([np.array([n_objects]), z.flatten()])

    return z_vector


def get_z_where_from_obs(latent_obs, ks, max_n_objects, z_what_dim, z_where_dim):
    bs = len(latent_obs)

    zs = latent_obs[:, 1:].reshape((bs, max_n_objects, -1))
    z_wheres = zs[np.arange(bs), ks, z_what_dim:z_what_dim + z_where_dim]

    return z_wheres
