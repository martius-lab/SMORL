import numpy as np

from rlkit.core import logger
from rlkit.samplers.data_collector.path_collector \
    import GoalConditionedPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer


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


def apply_hp_options(variant):
    """Apply options that have more complex effects

    Needed for hyperparameter optimization
    """
    import torch
    import rlkit.torch.pytorch_util as ptu
    from rlkit.core import logger

    if 'hp_options' not in variant:
        return variant

    def set_and_log(dict_key, key, value):
        logger.log('Setting {}.{}={}'.format(dict_key, key, value))
        variant.setdefault(dict_key, {})[key] = value

    options = variant['hp_options']

    if 'lr' in options:
        set_and_log('twin_sac_trainer_kwargs', 'qf_lr', options['lr'])
        set_and_log('twin_sac_trainer_kwargs', 'policy_lr', options['lr'])

    if 'activation' in options:
        fn = {
            'relu': torch.nn.functional.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'elu': torch.nn.functional.elu,
            'swish': ptu.swish_activation
        }[options['activation']]
        set_and_log('qf_kwargs', 'hidden_activation', fn)
        set_and_log('policy_kwargs', 'hidden_activation', fn)

    if 'initialization' in options:
        init = {
            'orthogonal': torch.nn.init.orthogonal_,
            'lecun_uniform': ptu.fanin_init,
            'xavier_uniform': torch.nn.init.xavier_uniform_,
            'kaiming_uniform': torch.nn.init.kaiming_uniform_,
        }[options['initialization']]

        if 'policy_last_layer_scale' in options:
            scale = options['policy_last_layer_scale']
            p_init = ptu.WeightInitializer(fn=init, scaling=scale)
        else:
            p_init = init

        set_and_log('qf_kwargs', 'hidden_init', init)
        set_and_log('qf_kwargs', 'last_layer_init_w', init)
        set_and_log('policy_kwargs', 'hidden_init', init)
        set_and_log('policy_kwargs', 'last_layer_init_w', p_init)

    if 'policy_initial_std' in options:
        value = np.log(np.arctanh(options['policy_initial_std']))
        set_and_log('policy_kwargs', 'initial_log_std_offset', value)

    for name in ('policy', 'qf'):
        if ('{}_depth'.format(name) in options
                and '{}_width'.format(name) in options):
            depth = options['{}_depth'.format(name)]
            width = options['{}_width'.format(name)]
            sizes = [width] * depth
            set_and_log('{}_kwargs'.format(name), 'hidden_sizes', sizes)

    return variant


def gt_experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.torch.data_management.normalizer import TorchNormalizer
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from rlkit.data_management.shared_obs_dict_replay_buffer \
        import SharedObsDictRelabelingBuffer
    from rlkit.torch.networks import FlattenMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy

    train_env = get_envs(variant)
    eval_env = get_envs(variant, eval_env=True)

    observation_key = variant.get('observation_key', 'state_observation')
    desired_goal_key = variant.get('desired_goal_key', 'desired_goal')
    achieved_goal_key = variant.get('achieved_goal_key', 'state_observation')
    obs_dim = (train_env.observation_space.spaces[observation_key].shape[0]
               + train_env.observation_space.spaces[desired_goal_key].shape[0])
    action_dim = train_env.action_space.low.size

    obs_normalizer = None
    if variant.get('normalize_obs', False):
        kwargs = variant.get('obs_normalizer_kwargs', {})
        obs_normalizer = TorchNormalizer(size=obs_dim, **kwargs)

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant.get('qf_kwargs', {})
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant.get('qf_kwargs', {})
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant.get('qf_kwargs', {})
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant.get('qf_kwargs', {})
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        normalizer=obs_normalizer,
        **variant.get('policy_kwargs', {})
    )

    replay_buffer = SharedObsDictRelabelingBuffer(
        env=train_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    if variant.get('custom_goal_sampler') == 'replay_buffer':
        train_env.custom_goal_sampler = replay_buffer.sample_buffer_goals

    max_path_length = variant['max_path_length']
    max_path_length_eval = variant.get('max_path_length_eval', max_path_length)

    trainer = SACTrainer(
        env=train_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        obs_normalizer=obs_normalizer,
        **variant['twin_sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)

    n_eval_paths = (variant['algo_kwargs']['num_eval_steps_per_epoch']
                    // max_path_length_eval)
    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        MakeDeterministic(policy),
        max_num_epoch_paths_saved=n_eval_paths,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )

    n_train_paths = (variant['algo_kwargs']['num_train_loops_per_epoch']
                     * variant['algo_kwargs']['num_expl_steps_per_train_loop']
                     // max_path_length)
    expl_path_collector = GoalConditionedPathCollector(
        train_env,
        policy,
        max_num_epoch_paths_saved=n_train_paths,
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
        max_path_length_eval=max_path_length_eval,
        **variant['algo_kwargs']
    )

    snapshot = algorithm._get_snapshot()
    logger.save_itr_params(-1, snapshot)

    algorithm.to(ptu.device)
    algorithm.train()
