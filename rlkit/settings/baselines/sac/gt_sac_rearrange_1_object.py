import numpy as np
import torch

from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
import rlkit.util.hyperparameter as hyp
from rlkit.launchers.gt_sac_experiment import gt_experiment
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch import pytorch_util as ptu

from path_length_settings import get_path_settings

experiment = gt_experiment

path_settings = get_path_settings('SAC', 'Rearrange', n_objects=1)
PATH_LENGTH = path_settings.train_path_length
PATH_LENGTH_EVAL = path_settings.eval_path_length

variant = dict(
        env_id='SawyerMultiobjectRearrangeEnv-OneObj-v0',
        init_camera=sawyer_init_camera_zoomed_in,
        save_video=False,

        reward_params=dict(type='state_distance'),
        exploration_goal_sampling_mode='oracle_prior',
        evaluation_goal_sampling_mode='env_goal',

        max_path_length=PATH_LENGTH,
        max_path_length_eval=PATH_LENGTH_EVAL,

        # Training
        algo_kwargs=dict(
            batch_size=2048,
            num_epochs=31,
            num_eval_steps_per_epoch=50 * PATH_LENGTH_EVAL,
            num_train_loops_per_epoch=600,
            num_expl_steps_per_train_loop=1 * PATH_LENGTH,
            num_trains_per_train_loop=50,
            min_num_steps_before_training=5000
        ),

        # Memory
        replay_buffer_kwargs=dict(
            max_size=250000,
            fraction_goals_rollout_goals=0.1,
            fraction_goals_env_goals=0.5
        ),

        # Networks
        normalize_obs=True,
        obs_normalizer_kwargs=dict(eps=1e-6),
        qf_kwargs=dict(
            hidden_sizes=[256, 256, 256],
            hidden_activation=torch.relu,
            hidden_init=torch.nn.init.orthogonal_,
            b_init_value=0,
            last_layer_init_w=torch.nn.init.orthogonal_,
            last_layer_init_b=torch.nn.init.zeros_
        ),
        policy_kwargs=dict(
            hidden_sizes=[128, 128, 128],
            hidden_activation=torch.relu,
            hidden_init=torch.nn.init.orthogonal_,
            b_init_value=0,
            last_layer_init_w=ptu.WeightInitializer(name='orthogonal',
                                                    gain=0.001),
            last_layer_init_b=torch.nn.init.zeros_,
            initial_log_std_offset=np.log(np.arctanh(0.2))
        ),

        # SAC & Optimization
        twin_sac_trainer_kwargs=dict(
            policy_lr=1e-3,
            qf_lr=1e-3,
            discount=0.95,
            reward_scale=1,
            soft_target_tau=1e-2,
            target_update_period=1,
            use_automatic_entropy_tuning=True
        ),

        observation_key='state_observation',
        desired_goal_key='desired_goal',
        achieved_goal_key='state_observation'
    )

if __name__ == "__main__":
    exp_prefix = 'gt-sac-rearrange-1-obj'

    run_experiment(
        gt_experiment,
        exp_prefix=exp_prefix,
        mode='local',
        variant=variant,
        use_gpu=True)
