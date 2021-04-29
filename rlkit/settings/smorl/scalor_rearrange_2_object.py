from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.smorl_experiment import smorl_experiment

from path_length_settings import get_path_settings

path_settings = get_path_settings('SCALOR', 'Rearrange', n_objects=2)
PATH_LENGTH = path_settings.subtask_length
N_META = path_settings.attempts
PATH_LENGTH_EVAL = path_settings.eval_path_length

experiment = smorl_experiment
variant=dict(
        algorithm='SCALOR_MOURL',
        double_algo=False,
        imsize=64,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerMultiobjectRearrangeEnv-TwoObj-v0',
        scalor_path="./rlkit/data/scalor_training_checkpoints/scalor_2_objects_env.pth",
        representation_params=dict(
            n_itr=100000,
            lr=0.0001,
            batch_size=11,
            num_cell_h=4,
            num_cell_w=4,
            max_num_obj=5,
            explained_ratio_threshold=0.1,
            sigma=0.1,
            ratio_anc=1.0,
            var_anc=0.3,
            size_anc=0.22,
            var_s=0.12,
            z_pres_anneal_end_value=0.0001,
        ),
        save_video=False,
        qf_kwargs=dict(
            hidden_sizes=[128, 128, 128],
            attention_kwargs=dict(num_heads=1,
                                  uncond_attention=True,
                                  num_uncond_queries=3,
                                  num_uncond_heads=1)
        ),
        policy_kwargs=dict(
            hidden_sizes=[128, 128, 128],
            attention_kwargs=dict(num_heads=1,
                                  uncond_attention=True,
                                  num_uncond_queries=3,
                                  num_uncond_heads=1)
        ),
        z_what_dim=4,
        z_where_dim=4,
        z_depth_dim=1,
        max_n_objects=6,
        max_path_length=PATH_LENGTH,
        n_meta=N_META,
        algo_kwargs=dict(
            batch_size=2048,
            num_epochs=30,
            num_train_loops_per_epoch=150,
            num_eval_steps_per_epoch=50 * PATH_LENGTH_EVAL,
            num_expl_steps_per_train_loop=200,
            num_trains_per_train_loop=200,
            min_num_steps_before_training=5000,
        ),
        twin_sac_trainer_kwargs=dict(
            discount=0.925,
            reward_scale=1,
            soft_target_tau=0.05,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        replay_buffer_kwargs=dict(
            max_size=100000,
            fraction_goals_rollout_goals=0.1,
            fraction_goals_env_goals=0.5,
            max_n_objects=6
        ),
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

        norm_scale=10,
        match_thresh=13,
        max_dist=1.5,
        attention_key_query_size=32,

        hp_options={"initialization": "xavier_uniform",
                    "activation": "relu",
                    "policy_last_layer_scale": 1,
                    "policy_initial_std": 0.2,
                    "qf_depth": 3,
                    "qf_width": 128,
                    "policy_depth": 3,
                    "policy_width": 128,
                    "lr": 0.0005
                    }

    )

from rlkit.torch.scalor import common
common.z_what_dim = variant["z_what_dim"]

if __name__ == "__main__":
    assert variant["max_n_objects"] == variant["replay_buffer_kwargs"]["max_n_objects"]
    exp_prefix = 'smorl-scalor-rearrange-2-objects'

    run_experiment(
        smorl_experiment,
        exp_prefix=exp_prefix,
        mode='local',
        variant=variant,
        use_gpu=True)
