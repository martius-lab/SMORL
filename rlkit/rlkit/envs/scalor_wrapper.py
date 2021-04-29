import copy
import random
import warnings

import torch

import cv2
import numpy as np
import copy
from gym.spaces import Box, Dict
import rlkit.torch.pytorch_util as ptu
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
from rlkit.envs.wrappers import ProxyEnv
from einops import rearrange
import matplotlib.pyplot as plt
from rlkit.torch.scalor.utils import visualize_one_image

class SCALORWrappedEnv(ProxyEnv, MultitaskEnv):
    """This class wraps an image-based environment with a SCALOR.
    Assumes you get flattened (channels, 64, 64) observations from wrapped_env.
    This class adheres to the "Silent Multitask Env" semantics: on reset,
    it resamples a goal.
    """
    def __init__(self,
                 wrapped_env,
                 scalor,
                 z_what_dim,
                 z_where_dim,
                 z_depth_dim,
                 max_n_objects,
                 sub_task_horizon,
                 match_thresh,
                 max_dist,
                 done_on_success=False,
                 scalor_input_key_prefix='image',
                 solved_goal_threshold=0.2,
                 sample_from_true_prior=True,
                 decode_goals=False,
                 render_goals=False,
                 render_rollouts=False,
                 reward_params=None,
                 goal_sampling_mode="z_where_prior",
                 norm_scale=1,
                 imsize=64,
                 norm_order=2,
                 epsilon=20,
                 scalor_device=None):
        if reward_params is None:
            reward_params = dict()
        super().__init__(wrapped_env)
        self.scalor = scalor
        self.device = ptu.device # if scalor_device is None else scalor_device
        self.input_channels = 3
        self.norm_scale = norm_scale
        self.z_what_dim = z_what_dim
        self.z_where_dim = z_where_dim
        self.z_depth_dim = z_depth_dim
        self.max_n_objects = max_n_objects
        self.sub_task_horizon = sub_task_horizon
        self.sample_from_true_prior = sample_from_true_prior
        self._decode_goals = decode_goals
        self._done_on_success = done_on_success
        self.render_goals = render_goals
        self.render_rollouts = render_rollouts
        self.imsize = imsize
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", 'object_distance')
        self.norm_order = self.reward_params.get("norm_order", norm_order)
        self.epsilon = self.reward_params.get("epsilon", epsilon)
        self.reward_min_variance = self.reward_params.get("min_variance", 0)
        if self.reward_type in ('sparse', 'pos_sparse'):
            self.success_threshold = self.reward_params.get('threshold')
            solved_goal_threshold = self.success_threshold
            assert self.success_threshold is not None
        self.solved_goal_threshold = solved_goal_threshold
        self.latent_obj_dim = self.z_what_dim + self.z_where_dim + self.z_depth_dim
        self.z_dim = self.latent_obj_dim * self.max_n_objects + 1
        self.z_goal_dim = self.z_what_dim + self.z_where_dim + 1
        self.match_thresh = match_thresh / self.norm_scale
        self.max_dist = max_dist
        self.n_scalor_repeats = 5
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
        self.scalor_input_key_prefix = scalor_input_key_prefix
        assert scalor_input_key_prefix in {'image'}
        self.scalor_input_observation_key = scalor_input_key_prefix + '_observation'
        self.scalor_input_achieved_goal_key = scalor_input_key_prefix + '_achieved_goal'
        self.scalor_input_desired_goal_key = scalor_input_key_prefix + '_desired_goal'
        self._mode_map = {}
        self.desired_goal = None
        self._initial_obs = None
        self._custom_goal_sampler = None
        self._goal_sampling_mode = goal_sampling_mode
        self.t = 0
        self.debug = False
        self.k = 0
        self.n_objects = 0
        self._train_final_rewards = []
        self._eval_final_rewards = []
        self.render_bboxes = False

    def reset(self):
        self.scalor.reset()
        goal, obs = self.sample_goal()
        self.set_goal(goal)
        if self._goal_sampling_mode == 'reset_of_env':
            self.scalor.reset()
        self._initial_obs = obs
        self.t = 0

        return self._update_obs(obs)

    def sample_goal(self):
        self.wrapped_env.reset()
        zero_action = np.zeros((2,))
        obs, _, _, _ = self.wrapped_env.step(zero_action)
        if self._goal_sampling_mode == 'z_where_prior':
            x = obs['observation'].reshape(1, self.input_channels, self.imsize, self.imsize)
            x = torch.from_numpy(x).to(self.device, torch.float32)
            latent_goal = self.get_goal_from_image(x, self._goal_sampling_mode)
        elif self._goal_sampling_mode == 'reset_of_env':
            goals = self.wrapped_env.get_goal()
            goals_image = goals[self.scalor_input_desired_goal_key].reshape(1, self.input_channels, self.imsize, self.imsize)
            goals_image = torch.from_numpy(goals_image).to(self.device, torch.float32)
            latent_goal = self.get_goal_from_image(goals_image, mode=self._goal_sampling_mode) # random_goal from goal image
        else:
            raise RuntimeError("Invalid: {}".format(self._goal_sampling_mode))

        return latent_goal, obs

    def sample_goals(self, batch_size, initial_goals):
        z_where_prior = self._sample_z_where_prior(batch_size)
        initial_goals[:, -2:] = z_where_prior
        return initial_goals

    def visualize_goals(self, z_where, x):
        bbox = visualize_one_image(x, z_where)
        return bbox

    def get_goal_from_image(self, x, mode):
        for i in range(self.n_scalor_repeats):
            latent_representation = self._encode_one(x)
        n_objects = latent_representation["z_what"].size(-2)
        z_what = latent_representation["z_what"].detach().cpu().numpy() / self.norm_scale
        z_where = latent_representation["z_where"].detach().cpu().numpy()
        if mode == 'z_where_prior':
            z_where[:, 2:] = self._sample_z_where_prior(n_objects)
        goal_vectors = np.concatenate([np.array(range(n_objects))[..., None], z_what, z_where], axis=1)
        k = np.random.randint(n_objects)
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
        if self.render_bboxes:
            goal["image_desired_goal_bbox"] =  self.visualize_goals(torch.from_numpy(z_where), x)

        self.k = k
        self.n_objects = n_objects

        return goal

    def _sample_z_where_prior(self, batch_size):
        x = np.random.uniform(0.5, -0.25, (batch_size,))
        y = np.random.uniform(0.7, -0.7,  (batch_size,))
        z_where_pos = rearrange([x, y], "coords batch -> batch coords")
        return z_where_pos

    def update_goal(self, obs):
        goal = self.desired_goal
        goal_vectors = goal["goal_vectors"]
        z_where = goal["z_where_goals"]
        z_what = goal["z_what_goals"]
        n_objects = goal_vectors.shape[0]
        k = goal["idx_goal"]
        k = (k+1) % n_objects
        self.k = k
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
        if self.render_bboxes:
            goal["image_desired_goal_bbox"] =  self.desired_goal["image_desired_goal_bbox"]
        self.desired_goal = goal
        new_obs = dict(obs, **goal)
        return new_obs

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        self.t += 1

        new_obs = self._update_obs(obs, action)
        reward = self.compute_reward(
            action,
            {'latent_obs_vector': new_obs['latent_obs_vector'],
             'goal_vector': new_obs['goal_vector']}
        )

        if self._goal_sampling_mode == 'reset_of_env':
            if self.t == 1:
                self._eval_final_rewards.append([])
            if self.t % self.sub_task_horizon == 0:
                self._eval_final_rewards[-1].append(reward)

                k_init = new_obs["idx_goal"]
                new_obs = self.update_goal(new_obs)
                new_reward = self.compute_reward(
                    action,
                    {'latent_obs_vector': new_obs['latent_obs_vector'],
                     'goal_vector': new_obs['goal_vector']}
                )
                success = self.compute_success(new_reward)
                while success:
                    new_obs = self.update_goal(new_obs)
                    new_reward = self.compute_reward(
                        action,
                        {'latent_obs_vector': new_obs['latent_obs_vector'],
                         'goal_vector': new_obs['goal_vector']}
                    )
                    success = self.compute_success(new_reward)
                    if success and new_obs["idx_goal"] == k_init:
                        done = True
                        break
        else:
            done = self.compute_done(done, reward)
            if self.t == 1:
                self._train_final_rewards.append([])
            if self.t % self.sub_task_horizon == 0:
                self._train_final_rewards[-1].append(reward)

        return new_obs, reward, done, info

    def compute_success(self, reward):
        if self.reward_type == 'pos_sparse':
            success = reward == 1
        elif self.reward_type == 'sparse':
            success = reward == 0
        elif self.reward_type == 'object_distance':
            success = (np.abs(reward) < self.solved_goal_threshold)
        return success

    def compute_done(self, done, reward):
        if self._done_on_success:
            if self.reward_type == 'pos_sparse':
                return reward == 1.0 or done
            elif self.reward_type == 'sparse':
                return reward == 0.0 or done

        return done

    def _update_obs(self, obs, action=None):
        x = obs[self.scalor_input_observation_key].reshape(-1, self.input_channels, self.imsize, self.imsize)
        x = torch.from_numpy(x).to(self.device, torch.float32)

        if self.t == 0 and self.goal_sampling_mode == 'reset_of_env':
            # Encode initial image during evaluation several times to stabilize SCALOR
            for _ in range(self.n_scalor_repeats):
                _ = self._encode_one(x)
        latent_obs = self._encode_one(x, action)
        latent_obs["z_what"] = latent_obs["z_what"] / self.norm_scale
        latent_obs = {k: v.detach().cpu().numpy() for k, v in latent_obs.items()}
        latent_obs_vector = dict2vector(latent_obs,
                                        self.max_n_objects,
                                        self.z_what_dim,
                                        self.z_where_dim,
                                        self.z_depth_dim)
        obs = {**obs, **latent_obs, **self.desired_goal}
        if self.render_bboxes:
            obs["image_obs_bbox"] = self.visualize_goals(torch.from_numpy(latent_obs["z_where"]), x).numpy()
        obs["latent_obs_vector"] = latent_obs_vector
        return obs

    """
    Multitask functions
    """
    def get_goal(self):
        return self.desired_goal

    def _encode(self, x, action=None):
        return self._encode_one(x, action)

    def _encode_one(self, x, action=None):
        with torch.no_grad():
            if action is None:
                bs = x.size(0)
                action = torch.zeros(bs, 2).to(self.device)
            else:
                action = torch.from_numpy(action)[None].to(self.device)
            return self.scalor.encode(x, action)

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_rewards(actions, next_obs)[0]

    def compute_rewards(self, actions, obs):
        latent_obs_vector = obs['latent_obs_vector']
        goal_vector = obs['goal_vector']
        desired_goals_z_where = goal_vector[:, -self.z_where_dim:][:, 2:]

        achieved_z_where, match = self.get_matching_z_where(latent_obs_vector,
                                                            goal_vector)
        achieved_goals = achieved_z_where[:, 2:]

        dist = np.linalg.norm(desired_goals_z_where - achieved_goals,
                              ord=self.norm_order, axis=1)
        if self.reward_type == 'object_distance':
            return -(dist * match + self.max_dist * (1 - match))
        elif self.reward_type == 'sparse':
            return -1.0 * (dist >= self.success_threshold) * (1 - match)
        elif self.reward_type == 'pos_sparse':
            return (dist < self.success_threshold).astype(np.float32) * match
        else:
            raise NotImplementedError('reward_type {}'
                                      .format(self.reward_type))

    def get_matching_z_where(self, latent_obs, z_goal):
        bs = latent_obs.shape[0]

        zs = latent_obs[:, 1:].reshape((bs, self.max_n_objects, -1))

        match, match_idx = self.match_goals(latent_obs, z_goal)

        z_wheres = zs[np.arange(bs),
                      match_idx,
                      -(self.z_where_dim+self.z_depth_dim):-self.z_depth_dim]

        # Only z_wheres where `match == True` are valid!
        return z_wheres, match

    def match_goals(self, latent_obs, z_goal):
        """Get object in observation matching to goal

        Input:  latent_obs (batch, obs_dim)
                z_goal     ([batch or 1], goal_dim)
        Output: match      (batch,)
                match_idx  (batch,)
        where `match_idx` is the index of the first object that matched
        """
        bs = latent_obs.shape[0]
        assert z_goal.shape[0] == bs or z_goal.shape[0] == 1
        n_objects = latent_obs[:, 0]

        z = latent_obs[:, 1:].reshape((bs, self.max_n_objects, -1))
        z_whats = z[:, :, :self.z_what_dim]

        z_what_goal = z_goal[:, 1:1 + self.z_what_dim][:, None]

        dist = np.linalg.norm(z_whats - z_what_goal, ord=2, axis=-1)

        min_dist = np.min(dist, axis=1)
        match_idx = np.argmin(dist, axis=1)
        match = (min_dist < self.match_thresh) & (match_idx < n_objects)

        return match, match_idx

    def extract_achieved_goals(self, latent_obs, obj_indices):
        goal_obs = latent_obs[:, 1:]
        goal_obs = rearrange(goal_obs, "b (obj z) -> b obj z",
                             obj=self.max_n_objects, z=self.latent_obj_dim)

        bs = len(obj_indices)
        goal_objects = goal_obs[np.arange(bs), obj_indices, :-self.z_depth_dim]
        # goal_objects has shape (batch, z_what + z_where)

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

    def get_diagnostics(self, paths, phase, **kwargs):
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)

        if phase == 'evaluation':
            final_rewards = self._eval_final_rewards
            self._eval_final_rewards = []
        elif phase == 'exploration':
            final_rewards = self._train_final_rewards
            self._train_final_rewards = []
        else:
            final_rewards = []

        mean_rewards = [np.mean(r) for r in final_rewards]
        statistics["Reward Final Mean"] = np.mean(mean_rewards)
        statistics["Reward Final Std"] = np.std(mean_rewards)

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
            'reset_of_env'
        ], "Invalid env mode"
        self._goal_sampling_mode = mode


def dict2vector(representation, max_n_objects, z_what_dim, z_where_dim, z_depth_dim):
    z_n = np.concatenate([representation["z_what"],
                          representation["z_where"],
                          representation["z_depth"]], axis=1)
    z = np.zeros((max_n_objects, z_what_dim + z_where_dim + z_depth_dim))
    n_objects = z_n.shape[0]
    z[:n_objects, :] = z_n
    z_vector = np.concatenate([np.array([n_objects]), z.reshape((-1,))])
    return z_vector


def get_z_where(z, ks, max_n_objects, z_what_dim, z_where_dim, z_depth_dim):
    n_objects = z[:, 0]
    z = z[:, 1:]
    z = z.reshape((-1, max_n_objects, z_what_dim + z_where_dim + z_depth_dim))
    z_whats = z[:, :, :z_what_dim]
    z_wheres = z[:, :, -(z_where_dim+z_depth_dim):-z_depth_dim]
    z_where = np.asarray([z_where[int(k), :] for z_where, k in zip(z_wheres, ks)])
    return z_where
