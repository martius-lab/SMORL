import time
from multiworld.core.image_env import ImageEnv, unormalize_image, normalize_image
from rlkit.core import logger

import cv2
import numpy as np
import os.path as osp

from rlkit.samplers.data_collector.scalor_env import WrappedEnvPathCollector as SCALORWrappedEnvPathCollector
from rlkit.torch.scalor.scalor import SCALOR
from rlkit.util.video import dump_video
from rlkit.util.io import load_local_or_remote_file
import rlkit.torch.pytorch_util as ptu
import gym
import multiworld

def generate_scalor_dataset(variant):
    env_kwargs = variant.get('env_kwargs', None)
    env_id = variant.get('env_id', None)
    N = variant.get('N', 100)
    rollout_length = variant.get('rollout_length', 100)
    test_p = variant.get('test_p', 0.9)
    use_cached = variant.get('use_cached', True)
    imsize = variant.get('imsize', 64)
    num_channels = variant.get('num_channels', 3)
    show = variant.get('show', False)
    init_camera = variant.get('init_camera', None)
    dataset_path = variant.get('dataset_path', None)
    oracle_dataset_using_set_to_goal = variant.get(
        'oracle_dataset_using_set_to_goal', False)
    random_rollout_data = variant.get('random_rollout_data', False)
    random_and_oracle_policy_data = variant.get('random_and_oracle_policy_data',
                                                False)
    random_and_oracle_policy_data_split = variant.get(
        'random_and_oracle_policy_data_split', 0)
    policy_file = variant.get('policy_file', None)
    n_random_steps = 1
    scalor_dataset_specific_env_kwargs = variant.get(
        'scalor_dataset_specific_env_kwargs', None)
    save_file_prefix = variant.get('save_file_prefix', None)
    tag = variant.get('tag', '')

    if env_kwargs is None:
        env_kwargs = {}
    if save_file_prefix is None:
        save_file_prefix = env_id
    filename = "./data/tmp/{}_N{}_rollout_length{}_imsize{}_{}{}.npz".format(
        save_file_prefix,
        str(N),
        str(rollout_length),
        init_camera.__name__ if init_camera else '',
        imsize,
        tag,
    )
    import os
    if not osp.exists('./data/tmp/'):
        os.makedirs('./data/tmp/')
    info = {}
    import os
    if not os.path.exists("./data/tmp/"):
        os.makedirs("./data/tmp/")
    if use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        print("loaded data from saved file", filename)
    else:
        now = time.time()
        multiworld.register_all_envs()
        env = gym.make(env_id)
        if not isinstance(env, ImageEnv):
            env = ImageEnv(
                env,
                imsize,
                init_camera=init_camera,
                transpose=True,
                normalize=True,
                non_presampled_goal_img_is_garbage=True,
            )
        env.reset()
        
        act_dim = env.action_space.low.size
        info['env'] = env
        imgs = np.zeros((N, rollout_length, imsize * imsize * num_channels),
                dtype=np.uint8)
        actions = np.zeros((N, rollout_length, act_dim))
        for i in range(N):
            env.reset()
            for j in range(rollout_length):
                action = env.action_space.sample()
                obs = env.step(action)[0]
                img = obs['image_observation']
                imgs[i, j, :] = unormalize_image(img)
                actions[i,j, :] = action
                if show:
                    img = img.reshape(3, imsize, imsize).transpose()
                    img = img[::-1, :, ::-1]
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
        print("done making training data", filename, time.time() - now)
        dataset = {"imgs": imgs, "actions": actions}
        print(imgs.shape)
        # np.savez(filename, **dataset)

    return dataset, info

def scalor_training(variant):
    scalor_params = variant.get("scalor_params", dict())
    scalor_params["logdir"] = logger.get_snapshot_dir()
    scalor = SCALOR(**scalor_params)
    data, info = generate_scalor_dataset(variant['generate_scalor_dataset_kwargs'])
    imgs, actions = data["imgs"], data["actions"]
    imgs = normalize_image(imgs)
    scalor.train(imgs=imgs, actions=actions)