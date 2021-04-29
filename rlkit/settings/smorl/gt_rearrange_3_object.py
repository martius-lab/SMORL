import torch
import numpy as np

import rlkit.util.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.launchers.gt_smorl_experiment import gt_experiment
from rlkit.torch import pytorch_util as ptu

from path_length_settings import get_path_settings

path_settings = get_path_settings('MOURL', 'Rearrange', n_objects=3)
PATH_LENGTH = path_settings.subtask_length
N_META = path_settings.attempts
PATH_LENGTH_EVAL = path_settings.eval_path_length

experiment = gt_experiment
variant=dict(
    algorithm='GT_MOURL',
    double_algo=False,
    imsize=64,
    init_camera=sawyer_init_camera_zoomed_in,
    env_id='SawyerMultiobjectRearrangeEnv-ThreeObj-v0',
    save_video=False,
    wrapped_env_kwargs=dict(track_success_rates=True),
    qf_kwargs=dict(
        hidden_sizes=[256, 256, 256],
        hidden_init=torch.nn.init.xavier_uniform_,
        b_init_value=0,
        last_layer_init_w=torch.nn.init.xavier_uniform_,
        last_layer_init_b=torch.nn.init.zeros_
    ),
    policy_kwargs=dict(
        hidden_sizes=[256, 256, 256],
        hidden_init=torch.nn.init.xavier_uniform_,
        b_init_value=0,
        last_layer_init_w=ptu.WeightInitializer(name='xavier_uniform',
                                                gain=1),
        last_layer_init_b=torch.nn.init.zeros_,
        initial_log_std_offset=np.log(np.arctanh(0.5))
    ),
    z_where_dim=2,
    z_depth_dim=1,
    max_n_objects=5,
    max_path_length=PATH_LENGTH,
    n_meta=N_META,
    algo_kwargs=dict(
        batch_size=2048,
        min_num_steps_before_training=4800,
        num_epochs=20,
        num_train_loops_per_epoch=150,
        num_expl_steps_per_train_loop=200,
        num_trains_per_train_loop=200,
        num_eval_steps_per_epoch=50 * PATH_LENGTH_EVAL
    ),
    twin_sac_trainer_kwargs=dict(
        policy_lr=1e-3,
        qf_lr=1e-3,
        discount=0.95,
        reward_scale=1,
        soft_target_tau=0.05,
        target_update_period=1,
        use_automatic_entropy_tuning=True,
    ),
    replay_buffer_kwargs=dict(
        max_size=250000,
        fraction_goals_rollout_goals=0.1,
        fraction_goals_env_goals=0.5,
        max_n_objects=5
    ),
    attention_key_query_size=16,
    exploration_goal_sampling_mode='z_where_prior',
    evaluation_goal_sampling_mode='reset_of_env',
    normalize=False,
    render=False,
    exploration_noise=0.0,
    exploration_type='ou',
    training_mode='train',
    testing_mode='test',
    reward_params=dict(
        type='object_distance',
    ),
    observation_key='latent_obs_vector',
    desired_goal_key='goal_vector',
    achieved_goal_key='latent_obs_vector',

    hp_options={"activation": "relu",
                  "initialization": "xavier_uniform",
                  "policy_last_layer_scale": 1,
                  "policy_initial_std": 0.2,
                  "qf_depth": 3,
                  "qf_width": 128,
                  "policy_depth": 2,
                  "policy_width": 128,
                  "lr": 0.001,
                  }
)

if __name__ == "__main__":
    exp_prefix = 'smorl-gt-rearrange-3-objects'

    run_experiment(
        gt_experiment,
        exp_prefix=exp_prefix,
        mode='local',
        variant=variant,
        use_gpu=True)
