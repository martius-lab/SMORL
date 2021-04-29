import argparse
import torch

from rlkit.core import logger
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu
from rlkit.envs.render_wrapper import get_render_wrapper
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
import os
import json

def simulate_policy(args):
    if args.gpu:
        ptu.set_gpu_mode(True)
    else:
        ptu.set_gpu_mode(False)
    data = torch.load(args.file, map_location=ptu.device)
    dir_path = os.path.dirname(args.file)
    with open(os.path.join(dir_path, "variant.json")) as f:
        variant = json.load(f)
    variant["init_camera"] = sawyer_init_camera_zoomed_in
    policy = data['evaluation/policy']
    policy.device = ptu.device
    algorithm = variant.get("algorithm", None)
    if algorithm is not None:
        if algorithm == "GT_MOURL":
            from rlkit.launchers.gt_smorl_experiment import get_envs
            env = get_envs(variant)
            wrapped_env = env.wrapped_env
            if hasattr(env, 'env_params'):
                env_params = env.env_params
            else:
                env_params = dict(z_where_dim=4,
                                  z_depth_dim=1,
                                  max_n_objects=5,
                                  sub_task_horizon=20)
        elif algorithm == "SCALOR_MOURL" or algorithm == "MOURL":
            from rlkit.launchers.smorl_experiment import get_envs
            env = get_envs(variant)
            scalor = env.scalor
            wrapped_env = env.wrapped_env
            if hasattr(env, 'env_params'):
                env_params = env.env_params
            else:
                env_params = dict(scalor=scalor,
                                  z_what_dim=variant["z_what_dim"],
                                  z_where_dim=variant["z_where_dim"],
                                  z_depth_dim=variant["z_depth_dim"],
                                  max_n_objects=variant["max_n_objects"],
                                  max_dist=variant["max_dist"],
                                  sub_task_horizon=variant['max_path_length'],
                                  match_thresh=variant['match_thresh'],
                                  norm_scale=variant["norm_scale"])
    else:
        env = data['evaluation/env']
        if hasattr(env, 'env_params'):
            env_params = env.env_params
    wrapped_env = env.wrapped_env
    if args.make_video:
        video_params = {"path": os.path.join(os.getcwd(), "policy_video", dir_path.split("/")[-1]), "video_episodes": args.video_episodes, "one_video": args.one_video, "only_video": args.only_video}
    else:
        video_params = None
    render_env = get_render_wrapper(env, wrapped_env, video_params, render_bboxes=True,
                                    env_params=env_params)
    print("Policy and environment loaded")
    policy.to(ptu.device)
    paths = []
    while True:
        paths.append(multitask_rollout(
            render_env,
            policy,
            max_path_length=args.H,
            render=not args.hide,
            observation_key=variant["observation_key"],
            desired_goal_key=variant["desired_goal_key"],
        ))
        if hasattr(render_env, "log_diagnostics"):
            render_env.log_diagnostics(paths)
        if hasattr(render_env, "get_diagnostics"):
            for k, v in render_env.get_diagnostics(paths, phase='evaluation').items():
                logger.record_tabular(k, v)
        logger.dump_tabular()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--make_video', action='store_true')
    parser.add_argument('--save_path',  default=os.getcwd(), type=str)
    parser.add_argument('--video_episodes', type=float, default=10,
                        help='Number of episodes in video')
    parser.add_argument('--one_video', action='store_true')
    parser.add_argument('--only_video', action='store_true')
    parser.add_argument('--mode', default='video_env', type=str,
                        help='env mode')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
