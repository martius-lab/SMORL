from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.smorl_experiment import smorl_experiment

from path_length_settings import get_path_settings
path_settings = get_path_settings('SCALOR', 'Push', n_objects=1)
PATH_LENGTH = path_settings.subtask_length
N_META = path_settings.attempts
PATH_LENGTH_EVAL = path_settings.eval_path_length

experiment = smorl_experiment
variant=dict(
        algorithm='SCALOR_MOURL',
        double_algo=False,
        imsize=64,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerMultiobjectPushEnv-OneObj-v2',
        scalor_path="./rlkit/data/scalor_training_checkpoints/scalor_1_object_env.pth",
        representation_params=dict(
            n_itr=100000,
            lr=0.0001,
            batch_size=11,
            num_cell_h=4,
            num_cell_w=4,
            max_num_obj=10,
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
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        vf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        z_what_dim=8,
        z_where_dim=4,
        z_depth_dim=1,
        max_n_objects=11,
        max_path_length=PATH_LENGTH,
        n_meta=N_META,
        algo_kwargs=dict(
            batch_size=2048,
            num_epochs=20,
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
            max_n_objects=11
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

        attention_key_query_size=16,
        match_thresh=12,
        norm_scale=10,
        max_dist=0.75,

        hp_options={"activation": "relu",
                    "initialization": "xavier_uniform",
                    "policy_last_layer_scale": 1,
                    "policy_initial_std": 0.2,
                    "qf_depth": 3,
                    "qf_width": 256,
                    "policy_depth": 2,
                    "policy_width": 128,
                    "lr": 0.001}

    )

from rlkit.torch.scalor import common
common.z_what_dim = variant["z_what_dim"]

if __name__ == "__main__":
    assert variant["max_n_objects"] == variant["replay_buffer_kwargs"]["max_n_objects"]
    exp_prefix = 'smorl-scalor-push-1-object'

    run_experiment(
        smorl_experiment,
        exp_prefix=exp_prefix,
        mode='local',
        variant=variant,
        use_gpu=True)
