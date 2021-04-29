from rlkit.core import logger
from rlkit.samplers.data_collector.scalor_env import WrappedEnvPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.launchers.launcher_util import apply_hp_options


def get_envs(variant):
    from rlkit.envs.gt_wrapper import GTWrappedEnv
    reward_params = variant.get("reward_params", dict())
    z_where_dim = variant.get("z_where_dim", 2)
    z_depth_dim = variant.get("z_depth_dim", 1)
    max_n_objects = variant.get("max_n_objects", 5)
    done_on_success = variant.get("done_on_success", False)

    if 'env_id' in variant:
        import gym
        import multiworld
        multiworld.register_all_envs()
        env = gym.make(variant['env_id'])
    else:
        env = variant["env_class"](**variant['env_kwargs'])
    variant["z_what_dim"] = env.n_objects_max + 1
    gt_env = GTWrappedEnv(
        env,
        z_where_dim=z_where_dim,
        z_depth_dim=z_depth_dim,
        max_n_objects=max_n_objects,
        sub_task_horizon=variant['max_path_length'],
        reward_params=reward_params,
        done_on_success=done_on_success,
        **variant.get('wrapped_env_kwargs', {})
    )

    return gt_env


def smorl_preprocess_variant(variant):
    if variant.get("do_state_exp", False):
        variant['observation_key'] = 'latent_obs_vector'
        variant['desired_goal_key'] = 'goal_vector'
        variant['achieved_goal_key'] = 'latent_obs_vector'


def gt_experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from rlkit.data_management.shared_obs_dict_replay_buffer import SharedObsDictRelabelingMultiObjectBuffer
    from rlkit.torch.networks import AttentionMlp
    from rlkit.torch.sac.policies import AttentionTanhGaussianPolicy

    apply_hp_options(variant)
    smorl_preprocess_variant(variant)
    train_env = get_envs(variant)
    eval_env = get_envs(variant)

    observation_key = variant.get('observation_key', 'latent_obs_vector')
    desired_goal_key = variant.get('desired_goal_key', 'goal_vector')
    achieved_goal_key = variant.get('achieved_goal_key', 'latent_obs_vector')
    action_dim = train_env.action_space.low.size
    z_what_dim = variant.get("z_what_dim")
    z_where_dim = variant.get("z_where_dim", 2)
    z_depth_dim = variant.get("z_depth_dim", 1)
    max_n_objects = variant.get("max_n_objects")
    n_meta = variant.get("n_meta")

    if 'embedding_dim' in variant:
        embed_dim = variant["embedding_dim"]
    else:
        embed_dim = variant["attention_key_query_size"]

    qf_class = variant.get("qf_class", AttentionMlp)
    qf1 = qf_class(
                        embed_dim=embed_dim,
                        z_goal_size=z_what_dim + z_where_dim,
                        z_size=z_what_dim + z_where_dim + z_depth_dim,
                        action_size=action_dim,
                        max_objects=max_n_objects,
                        output_size=1,
                        **variant.get('qf_kwargs', {})
    )

    qf2 = qf_class(
                        embed_dim=embed_dim,
                        z_goal_size=z_what_dim + z_where_dim,
                        z_size=z_what_dim + z_where_dim + z_depth_dim,
                        action_size=action_dim,
                        max_objects=max_n_objects,
                        output_size=1,
                        **variant.get('qf_kwargs', {})
    )

    target_qf1 = qf_class(
                        embed_dim=embed_dim,
                        z_goal_size=z_what_dim + z_where_dim,
                        z_size=z_what_dim + z_where_dim + z_depth_dim,
                        action_size=action_dim,
                        max_objects=max_n_objects,
                        output_size=1,
                        **variant.get('qf_kwargs', {})
    )

    target_qf2 = qf_class(
                        embed_dim=embed_dim,
                        z_goal_size=z_what_dim + z_where_dim,
                        z_size=z_what_dim + z_where_dim + z_depth_dim,
                        action_size=action_dim,
                        max_objects=max_n_objects,
                        output_size=1,
                        **variant.get('qf_kwargs', {})
    )

    policy_class = variant.get("policy_class", AttentionTanhGaussianPolicy)
    policy = policy_class(
                        embed_dim=embed_dim,
                        z_size=z_what_dim + z_where_dim + z_depth_dim,
                        z_goal_size=z_what_dim + z_where_dim,
                        max_objects=max_n_objects,
                        action_dim=action_dim,
                        **variant.get('policy_kwargs', {})
    )

    replay_buffer = SharedObsDictRelabelingMultiObjectBuffer(
        env=train_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    max_path_length = variant['max_path_length']

    trainer = SACTrainer(
        env=train_env,
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
        eval_env,
        MakeDeterministic(policy),
        n_meta * max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = WrappedEnvPathCollector(
        variant['exploration_goal_sampling_mode'],
        train_env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=train_env,
        evaluation_env=eval_env,
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
    algorithm.train()
