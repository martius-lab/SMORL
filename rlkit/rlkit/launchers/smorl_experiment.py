import numpy as np

from rlkit.core import logger
from rlkit.samplers.data_collector.scalor_env import WrappedEnvPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.launchers.launcher_util import apply_hp_options

def get_envs(variant):
    from multiworld.core.image_env import ImageEnv
    from rlkit.envs.scalor_wrapper import SCALORWrappedEnv
    from rlkit.torch.scalor.scalor import SCALOR
    import rlkit.torch.pytorch_util as ptu
    render = variant.get('render', False)
    scalor_path = variant.get("scalor_path", None)
    reward_params = variant.get("reward_params", dict())
    init_camera = variant.get("init_camera", None)
    do_state_exp = variant.get("do_state_exp", False)
    representation_params = variant.get("representation_params", dict())
    z_what_dim = variant.get("z_what_dim", 32)
    z_where_dim = variant.get("z_where_dim", 4)
    z_depth_dim = variant.get("z_depth_dim", 1)
    max_n_objects = variant.get("max_n_objects", 5)
    match_thresh = variant.get("match_thresh", 25)
    norm_scale = variant.get("norm_scale", 1)
    max_dist = variant.get("max_dist", 2)
    scalor = SCALOR(**representation_params)
    scalor_path = variant.get("scalor_path")
    scalor.load(scalor_path)
    if 'env_id' in variant:
        import gym
        import multiworld
        multiworld.register_all_envs()
        if 'one_color' in variant['env_id']:
            from gym.envs.registration import register
            from multiworld.envs.mujoco.sawyer_xyz.multiobject.sawyer_push_multiobj import COLOR_SET_1
            register(id='SawyerMultiobjectPushEnv-OneObj-one_color-v0',
                     entry_point='multiworld.envs.mujoco.sawyer_xyz.multiobject'
                                 '.sawyer_push_multiobj:SawyerMultiobjectPushEnv',
                     kwargs=dict(n_objects=1, color_set=COLOR_SET_1))
            COLOR_SET_2 = np.array([(0, 0, 1), (0, 1, 0)])
            register(id='SawyerMultiobjectPushEnv-TwoObj-one_color-v0',
                     entry_point='multiworld.envs.mujoco.sawyer_xyz.multiobject'
                                 '.sawyer_push_multiobj:SawyerMultiobjectPushEnv',
                     kwargs=dict(n_objects=2, color_set=COLOR_SET_2))
        env = gym.make(variant['env_id'])
    else:
        env = variant["env_class"](**variant['env_kwargs'])
    if not do_state_exp:
        if isinstance(env, ImageEnv):
            image_env = env
        else:
            image_env = ImageEnv(
                env,
                variant.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
            )

        scalor_env = SCALORWrappedEnv(
            image_env,
            scalor,
            imsize=image_env.imsize,
            z_what_dim=z_what_dim,
            z_where_dim=z_where_dim,
            z_depth_dim=z_depth_dim,
            max_n_objects=max_n_objects,
            match_thresh=match_thresh,
            max_dist=max_dist,
            sub_task_horizon=variant['max_path_length'],
            decode_goals=render,
            render_goals=render,
            render_rollouts=render,
            reward_params=reward_params,
            norm_scale=norm_scale,
            **variant.get('scalor_wrapped_env_kwargs', {})
        )

        env = scalor_env

    return env


def smorl_preprocess_variant(variant):
    if variant.get("do_state_exp", False):
        variant['observation_key'] = 'latent_obs_vector'
        variant['desired_goal_key'] = 'goal_vector'
        variant['achieved_goal_key'] = 'latent_obs_vector'


def smorl_experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from rlkit.data_management.shared_obs_dict_replay_buffer import SharedObsDictRelabelingMultiObjectBuffer
    from rlkit.torch.networks import AttentionMlp
    from rlkit.torch.sac.policies import AttentionTanhGaussianPolicy
    apply_hp_options(variant)
    smorl_preprocess_variant(variant)
    env = get_envs(variant)

    observation_key = variant.get('observation_key', 'latent_obs_vector')
    desired_goal_key = variant.get('desired_goal_key', 'goal_vector')
    achieved_goal_key = variant.get('achieved_goal_key', 'latent_obs_vector')
    action_dim = env.action_space.low.size
    z_what_dim = variant.get("z_what_dim")
    z_where_dim = variant.get("z_where_dim")
    z_depth_dim = variant.get("z_depth_dim")
    max_n_objects = env.max_n_objects
    max_n_frames = env.n_frames if hasattr(env, 'n_frames') else None
    z_time_id_dim = env.z_time_id_dim if hasattr(env, 'z_time_id_dim') else 0

    if 'embedding_dim' in variant:
        embed_dim = variant["embedding_dim"]
    else:
        embed_dim = variant["attention_key_query_size"]

    qf_class = variant.get("qf_class", AttentionMlp)
    qf1 = qf_class(
                        embed_dim=embed_dim,
                        z_goal_size=z_what_dim + z_where_dim,
                        z_size=z_what_dim + z_where_dim + z_depth_dim + z_time_id_dim,
                        action_size=action_dim,
                        max_objects=max_n_objects,
                        output_size=1,
                        n_frames=max_n_frames,
                        **variant.get('qf_kwargs', {})
    )

    qf2 = qf_class(
                        embed_dim=embed_dim,
                        z_goal_size=z_what_dim + z_where_dim,
                        z_size=z_what_dim + z_where_dim + z_depth_dim + z_time_id_dim,
                        action_size=action_dim,
                        max_objects=max_n_objects,
                        output_size=1,
                        n_frames=max_n_frames,
                        **variant.get('qf_kwargs', {})
    )

    target_qf1 = qf_class(
                        embed_dim=embed_dim,
                        z_goal_size=z_what_dim + z_where_dim,
                        z_size=z_what_dim + z_where_dim + z_depth_dim + z_time_id_dim,
                        action_size=action_dim,
                        max_objects=max_n_objects,
                        output_size=1,
                        n_frames=max_n_frames,
                        **variant.get('qf_kwargs', {})
    )

    target_qf2 = qf_class(
                        embed_dim=embed_dim,
                        z_goal_size=z_what_dim + z_where_dim,
                        z_size=z_what_dim + z_where_dim + z_depth_dim + z_time_id_dim,
                        action_size=action_dim,
                        max_objects=max_n_objects,
                        output_size=1,
                        n_frames=max_n_frames,
                        **variant.get('qf_kwargs', {})
    )

    policy_class = variant.get("policy_class", AttentionTanhGaussianPolicy)
    policy = policy_class(
                        embed_dim=embed_dim,
                        z_size=z_what_dim + z_where_dim + z_depth_dim + z_time_id_dim,
                        z_goal_size=z_what_dim + z_where_dim,
                        max_objects=max_n_objects,
                        action_dim=action_dim,
                        n_frames=max_n_frames,
                        **variant.get('policy_kwargs', {})
    )

    scalor = env.scalor

    if 'max_n_objects' not in variant['replay_buffer_kwargs']:
        variant['replay_buffer_kwargs']['max_n_objects'] = env.max_n_objects

    replay_buffer = SharedObsDictRelabelingMultiObjectBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    max_path_length = variant.get("max_path_length")
    n_meta = variant.get("n_meta")
    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['twin_sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = WrappedEnvPathCollector(
        variant['evaluation_goal_sampling_mode'],
        env,
        MakeDeterministic(policy),
        n_meta * max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = WrappedEnvPathCollector(
        variant['exploration_goal_sampling_mode'],
        env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        max_path_length_eval=max_path_length * n_meta,
        **variant['algo_kwargs']
    )

    snapshot = algorithm._get_snapshot()
    logger.save_itr_params(-1, snapshot)

    algorithm.to(ptu.device)
    scalor.to(ptu.device)
    algorithm.train()
