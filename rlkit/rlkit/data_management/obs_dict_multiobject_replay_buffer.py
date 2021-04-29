import numpy as np
from einops import rearrange
from gym.spaces import Dict, Discrete

from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.data_management.obs_dict_replay_buffer import flatten_n, flatten_dict, preprocess_obs_dict, postprocess_obs_dict, normalize_image, unnormalize_image


class ObsDictRelabelingMultiObjectBuffer(ObsDictRelabelingBuffer):
    """
    Replay buffer for environment with multiple objects
    Only random_batch method changed
    """
    def __init__(self, *args, max_n_objects, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_n_objects = max_n_objects
        self._future_obs_idx_to_object_idx = np.array([-1] * len(self._idx_to_future_obs_idx))
        assert hasattr(self.env, 'match_goals')
        assert hasattr(self.env, 'extract_achieved_goals')

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        path_len = len(rewards)

        actions = flatten_n(actions)
        if isinstance(self.env.action_space, Discrete):
            actions = np.eye(self._action_dim)[actions].reshape((-1, self._action_dim))
        obs = flatten_dict(obs, self.ob_keys_to_save + self.internal_keys)
        next_obs = flatten_dict(next_obs, self.ob_keys_to_save + self.internal_keys)
        obs = preprocess_obs_dict(obs)
        next_obs = preprocess_obs_dict(next_obs)

        if self._top + path_len >= self.max_size:
            """
            All of this logic is to handle wrapping the pointer when the
            replay buffer gets full.
            """
            num_pre_wrap_steps = self.max_size - self._top
            # numpy slice
            pre_wrap_buffer_slice = np.s_[
                                    self._top:self._top + num_pre_wrap_steps, :
                                    ]
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, :]

            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._actions[buffer_slice] = actions[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                for key in self.ob_keys_to_save + self.internal_keys:
                    self._obs[key][buffer_slice] = obs[key][path_slice]
                    self._next_obs[key][buffer_slice] = next_obs[key][path_slice]

            rollout_goal = self._obs[self.desired_goal_key][self._top]
            matches, match_indices = self.env.match_goals(next_obs[self.observation_key],
                                                          rollout_goal[None])
            assert len(matches) == path_len

            post_wrap_indices = np.nonzero(matches[num_pre_wrap_steps:])[0]
            # Pointers from before the wrap
            for i in range(0, num_pre_wrap_steps):
                pre_wrap_indices = (self._top +
                                    np.nonzero(matches[i:num_pre_wrap_steps])[0] + i)
                self._idx_to_future_obs_idx[self._top + i] = np.concatenate((pre_wrap_indices,
                                                                             post_wrap_indices))
                match_idx = match_indices[i] if matches[i] else -1
                self._future_obs_idx_to_object_idx[self._top + i] = match_idx

            # Pointers after the wrap
            for i in range(num_pre_wrap_steps, path_len):
                post_wrap_idx = i - num_pre_wrap_steps
                indices = np.nonzero(matches[i:])[0] + post_wrap_idx
                self._idx_to_future_obs_idx[post_wrap_idx] = indices

                match_idx = match_indices[i] if matches[i] else -1
                self._future_obs_idx_to_object_idx[post_wrap_idx] = match_idx

        else:
            slc = np.s_[self._top:self._top + path_len, :]
            self._actions[slc] = actions
            self._terminals[slc] = terminals
            for key in self.ob_keys_to_save + self.internal_keys:
                self._obs[key][slc] = obs[key]
                self._next_obs[key][slc] = next_obs[key]

            rollout_goal = self._obs[self.desired_goal_key][self._top]
            matches, match_indices = self.env.match_goals(next_obs[self.observation_key],
                                                          rollout_goal[None])
            assert len(matches) == path_len

            for i in range(path_len):
                indices = self._top + np.nonzero(matches[i:])[0] + i
                self._idx_to_future_obs_idx[self._top + i] = indices

                match_idx = match_indices[i] if matches[i] else -1
                self._future_obs_idx_to_object_idx[self._top + i] = match_idx

        self._top = (self._top + path_len) % self.max_size
        self._size = min(self._size + path_len, self.max_size)

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        resampled_goals = self._next_obs[self.desired_goal_key][indices]

        num_env_goals = int(batch_size * self.fraction_goals_env_goals)
        num_rollout_goals = int(batch_size * self.fraction_goals_rollout_goals)
        num_future_goals = batch_size - (num_env_goals + num_rollout_goals)
        new_obs_dict = self._batch_obs_dict(indices)
        new_next_obs_dict = self._batch_next_obs_dict(indices)

        if num_env_goals > 0:
            last_env_goal_idx = num_rollout_goals + num_env_goals
            initial_goals = resampled_goals[num_rollout_goals:last_env_goal_idx]
            env_goals = self.env.sample_goals(num_env_goals, initial_goals)
            resampled_goals[num_rollout_goals:last_env_goal_idx] = env_goals
        if num_future_goals > 0:
            future_obs_idxs = []
            valid_idxs = []

            for j, i in zip(range(-num_future_goals, 0), indices[-num_future_goals:]):
                possible_future_obs_idxs = self._idx_to_future_obs_idx[i]
                # This is generally faster than random.choice. Makes you wonder what
                # random.choice is doing
                num_options = len(possible_future_obs_idxs)
                if num_options != 0:
                    next_obs_i = int(np.random.randint(0, num_options))
                    future_obs_idxs.append(possible_future_obs_idxs[next_obs_i])
                    valid_idxs.append(j)

            future_obs_idxs = np.array(future_obs_idxs)
            future_goals = self._next_obs[self.achieved_goal_key][future_obs_idxs]
            if "obs" in self.achieved_goal_key:
                object_indices = self._future_obs_idx_to_object_idx[future_obs_idxs]
                future_goals = self.env.extract_achieved_goals(future_goals,
                                                               object_indices)
            resampled_goals[valid_idxs] = future_goals
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][valid_idxs] = \
                    self._next_obs[goal_key][future_obs_idxs]
                new_next_obs_dict[goal_key][valid_idxs] = \
                    self._next_obs[goal_key][future_obs_idxs]

        new_obs_dict[self.desired_goal_key] = resampled_goals
        new_next_obs_dict[self.desired_goal_key] = resampled_goals
        new_obs_dict = postprocess_obs_dict(new_obs_dict)
        new_next_obs_dict = postprocess_obs_dict(new_next_obs_dict)
        # resampled_goals must be postprocessed as well
        resampled_goals = new_next_obs_dict[self.desired_goal_key]

        new_actions = self._actions[indices]
        """
        For example, the environments in this repo have batch-wise
        implementations of computing rewards:

        https://github.com/vitchyr/multiworld
        """

        if hasattr(self.env, 'compute_rewards'):
            new_rewards = self.env.compute_rewards(
                new_actions,
                new_next_obs_dict,
            )
        else:  # Assuming it's a (possibly wrapped) gym GoalEnv
            new_rewards = np.ones((batch_size, 1))
            for i in range(batch_size):
                new_rewards[i] = self.env.compute_reward(
                    new_next_obs_dict[self.achieved_goal_key][i],
                    new_next_obs_dict[self.desired_goal_key][i],
                    None
                )
        new_rewards = new_rewards.reshape(-1, 1)

        new_obs = new_obs_dict[self.observation_key]
        new_next_obs = new_next_obs_dict[self.observation_key]
        new_terminals = self._terminals[indices].copy()
        if self.terminate_episode_on_success:
            new_terminals[new_rewards == self.success_reward] = 1.0

        batch = {
            'observations': new_obs,
            'actions': new_actions,
            'rewards': new_rewards,
            'terminals': new_terminals,
            'next_observations': new_next_obs,
            'resampled_goals': resampled_goals,
            'indices': np.array(indices).reshape(-1, 1),
        }
        return batch
