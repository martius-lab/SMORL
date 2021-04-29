import numpy as np
import torch

from multiworld.envs.mujoco.cameras import sawyer_init_camera_multiobject_push
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.skewfit_experiments import skewfit_full_experiment
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.vae.conv_vae import imsize48_architecture_2
import rlkit.torch.vae.vae_schedules as vae_schedules

from path_length_settings import get_path_settings

experiment = skewfit_full_experiment

path_settings = get_path_settings('RIG', 'Push', n_objects=2)
PATH_LENGTH = path_settings.train_path_length
PATH_LENGTH_EVAL = path_settings.eval_path_length

vae_architecture = imsize48_architecture_2.copy()

variant = dict(
    algorithm='Skew-Fit',
    double_algo=False,
    online_vae_exploration=False,
    imsize=48,
    init_camera=sawyer_init_camera_multiobject_push,
    env_id='SawyerMultiobjectPushEnv-TwoObj-v2',
    skewfit_variant=dict(
        save_video=True,
        custom_goal_sampler=None,
        online_vae_trainer_kwargs=dict(
            beta=0.2,
            lr=1e-3,
            batch_size=128,
            start_skew_epoch=100000,
            log_interval=1000,
            img_dump_interval=1
        ),
        save_video_period=100,
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
            hidden_activation=torch.relu,
            hidden_init=torch.nn.init.orthogonal_,
            b_init_value=0,
            last_layer_init_w=torch.nn.init.orthogonal_,
            last_layer_init_b=torch.nn.init.zeros_
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
            hidden_activation=torch.relu,
            hidden_init=torch.nn.init.orthogonal_,
            b_init_value=0,
            last_layer_init_w=ptu.WeightInitializer(name='orthogonal',
                                                    gain=0.001),
            last_layer_init_b=torch.nn.init.zeros_,
            initial_log_std_offset=np.log(np.arctanh(0.2))
        ),
        max_path_length=PATH_LENGTH,
        max_path_length_eval=PATH_LENGTH_EVAL,
        algo_kwargs=dict(
            batch_size=2048,
            num_epochs=31,
            num_eval_steps_per_epoch=50 * PATH_LENGTH_EVAL,
            num_train_loops_per_epoch=60,
            num_expl_steps_per_train_loop=10 * PATH_LENGTH,
            num_trains_per_train_loop=15 * PATH_LENGTH,
            min_num_steps_before_training=10000,
            vae_training_schedule=vae_schedules.pretrain_40k,
            oracle_data=False,
            vae_save_period=25,
            parallel_vae_train=False,
            pretrain_vae=True
        ),
        twin_sac_trainer_kwargs=dict(
            policy_lr=1e-3,
            qf_lr=1e-3,
            discount=0.95,
            reward_scale=1,
            soft_target_tau=1e-2,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        replay_buffer_kwargs=dict(
            start_skew_epoch=100000,
            max_size=250000,
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
            exploration_rewards_type='None',
            vae_priority_type='None',
            power=0,
            relabeling_goal_sampling_mode='vae_prior'
        ),
        exploration_goal_sampling_mode='vae_prior',
        evaluation_goal_sampling_mode='reset_of_env',
        normalize=False,
        render=False,
        exploration_noise=0.0,
        exploration_type='ou',
        training_mode='train',
        testing_mode='test',
        reward_params=dict(
            type='latent_distance',
        ),
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        vae_wrapped_env_kwargs=dict(
            sample_from_true_prior=True,
        ),
    ),
    train_vae_variant=dict(
        representation_size=16,
        beta=0.2,
        num_epochs=0,
        dump_skew_debug_plots=False,
        decoder_activation='gaussian',
        generate_vae_dataset_kwargs=dict(
            N=50,
            test_p=.1,
            use_cached=False,
            show=False,
            oracle_dataset=True,
            oracle_dataset_using_set_to_goal=True,
            n_random_steps=100,
            non_presampled_goal_img_is_garbage=True,
        ),
        vae_kwargs=dict(
            input_channels=3,
            architecture=vae_architecture,
            decoder_distribution='gaussian_identity_variance',
            hidden_init=torch.nn.init.orthogonal_,
            mse_reduction='sum',
            normalize_logprob=True
        ),
        algo_kwargs=dict(
            start_skew_epoch=100000,
            is_auto_encoder=False,
            batch_size=128,
            lr=1e-3,
            use_parallel_dataloading=False,
            img_dump_interval=25
        ),

        save_period=25,
    ),
)

if __name__ == "__main__":
    exp_prefix = 'rig-push-2-obj'

    run_experiment(
        skewfit_full_experiment,
        exp_prefix=exp_prefix,
        mode='local',
        variant=variant,
        use_gpu=True)
