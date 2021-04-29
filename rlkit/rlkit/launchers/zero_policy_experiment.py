import abc
import numpy as np

import gtimer as gt

from rlkit.samplers.data_collector.path_collector \
    import GoalConditionedPathCollector
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.core.trainer import Trainer
from rlkit.policies.simple import ZeroPolicy
from rlkit.samplers.data_collector import PathCollector
from rlkit.data_management.replay_buffer import ReplayBuffer


def get_envs(variant, eval_env=False):
    from rlkit.envs.unstructured_gt_wrapper import UnstructuredGTWrappedEnv
    reward_params = variant.get('reward_params', dict())

    if 'env_id' in variant:
        import gym
        import multiworld
        multiworld.register_all_envs()
        env = gym.make(variant['env_id'])
    else:
        env = variant['env_class'](**variant['env_kwargs'])

    if eval_env:
        goal_sampling_mode = variant.get('evaluation_goal_sampling_mode',
                                         'env_goal')
    else:
        goal_sampling_mode = variant.get('exploration_goal_sampling_mode',
                                         'oracle_prior')

    gt_env = UnstructuredGTWrappedEnv(env,
                                      reward_params=reward_params,
                                      goal_sampling_mode=goal_sampling_mode,
                                      **variant.get('wrapper_env_kwargs', {}))

    return gt_env


class DummyTrainer(Trainer):
    def train(self, data):
        pass


class DummyBuffer(ReplayBuffer):
    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        pass

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self, **kwargs):
        pass

    def add_path(self, path):
        pass

    def random_batch(self, batch_size):
        return None


class EvalRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            path_length,
            num_paths
    ):
        super().__init__(
            DummyTrainer(),
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            DummyBuffer(),
        )
        self.path_length = path_length
        self.num_paths = num_paths

    def _train(self):
        self.eval_data_collector.collect_new_paths(
            self.path_length,
            self.num_paths * self.path_length,
            discard_incomplete_paths=True,
        )
        gt.stamp('evaluation sampling')

        self.expl_data_collector.collect_new_paths(
            1,
            1,
            discard_incomplete_paths=False,
        )

        self._end_epoch(0)

    def training_mode(self, mode):
        pass


def zero_policy_experiment(variant):
    train_env = get_envs(variant, eval_env=True)
    eval_env = get_envs(variant, eval_env=True)

    observation_key = variant.get('observation_key', 'state_observation')
    desired_goal_key = variant.get('desired_goal_key', 'desired_goal')

    policy = ZeroPolicy(eval_env.action_space)

    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )

    expl_path_collector = GoalConditionedPathCollector(
        train_env,
        policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )

    algorithm = EvalRLAlgorithm(
        exploration_env=train_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        path_length=variant['max_path_length'],
        num_paths=variant['num_paths']
    )

    algorithm.train()
