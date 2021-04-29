import time
from argparse import ArgumentParser

import cv2
import gym
import numpy as np
from PIL import Image

import multiworld
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco import register_mujoco_envs
from multiworld.envs.mujoco.cameras import (sawyer_init_camera_zoomed_in,
                                            sawyer_init_camera_multiobject_push)
from multiworld.envs.mujoco.sawyer_xyz.multiobject.sawyer_push_multiobj import SawyerMultiobjectPushEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import \
    SawyerPickAndPlaceEnv

CAMERAS = {
    'zoomed_in': sawyer_init_camera_zoomed_in,
    'mo_push': sawyer_init_camera_multiobject_push
}

parser = ArgumentParser()
parser.add_argument('--no-render', action='store_true',
                    help='Do not render')
parser.add_argument('--record-stats', action='store_true',
                    help='Record state statistics')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Print more info')
parser.add_argument('-s', '--sleep', type=float, default=0.2,
                    help='Sleep time per step in seconds')
parser.add_argument('-i', '--images', action='store_true',
                    help='Run image based environment')
parser.add_argument('--camera', default='mo_push', choices=list(CAMERAS),
                    help='Camera to use')
parser.add_argument('--store-images', action='store_true',
                    help='Store images on disk')
parser.add_argument('--store-gif', action='store_true',
                    help='Make a gif of the episode')
parser.add_argument('--image-size', default=64, type=int, help='Output folder')
parser.add_argument('--out-dir', default='.', help='Output folder')
parser.add_argument('--env-name', help='Name of environment')
parser.add_argument('n_episodes', type=int, default=1,
                    help='Number of episodes')
parser.add_argument('n_steps', type=int, default=100,
                    help='Number of steps per episode')


def image_transform(img):
    img = np.transpose(img, (2, 1, 0))
    img = np.flip(img, axis=0)
    return img


def save_image(img, path):
    img = image_transform(img)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def save_gif(images, path):
    images = [image_transform(img) for img in images]
    images = [Image.fromarray((img * 255).astype(np.uint8))
              for img in images]
    images[0].save(path, format='GIF', save_all=True, append_images=images[1:],
                   duration=120, loop=0, optimize=False)


def show_image(img, title, scale=2):
    img = image_transform(img)[..., ::-1]
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    cv2.imshow(title, cv2.resize(img, dsize=(width, height)))
    cv2.waitKey(1)


args = parser.parse_args()

if args.env_name is not None:
    register_mujoco_envs()
    env = gym.make(args.env_name)    
else:
    env = SawyerMultiobjectPushEnv(n_objects=(1, 4))

if args.images:
    img_size = args.image_size
    env = ImageEnv(env, img_size,
                   init_camera=CAMERAS[args.camera],
                   transpose=True,
                   normalize=True)

action_space = env.action_space

if args.no_render and args.images and args.store_images:
    args.sleep = 0

if args.record_stats:
    states = []
    goals = []


for ep in range(args.n_episodes):
    if args.images and args.store_gif:
        images = []

    obs = env.reset()

    if args.images:
        obs_img = obs['observation'].reshape(3, img_size, img_size)
        goal_img = obs['desired_goal'].reshape(3, img_size, img_size)
        if args.store_images:
            save_image(obs_img, '{}/{:02d}_{:02d}_obs.png'.format(args.out_dir,
                                                                  ep, 0))
            save_image(goal_img, '{}/{:02d}_{:02d}_goal.png'.format(args.out_dir,
                                                                    ep, 0))
        else:
            show_image(goal_img, 'Goal')
            show_image(obs_img, 'Observation')
        if args.store_gif:
            images.append(obs_img)

    if args.record_stats:
        states.append(obs['state_observation'])

    if args.verbose:
        print(obs)

    if not args.no_render:
        env.render()
        env.viewer._render_every_frame = True

    for step in range(args.n_steps):
        done = False

        action = action_space.sample()
        obs, reward, done, info = env.step(action)

        if args.images:
            img = obs['observation'].reshape(3, img_size, img_size)
            if args.store_images:
                save_image(img, '{}/{:02d}_{:02d}_obs.png'.format(args.out_dir,
                                                                  ep, step))
            else:
                show_image(img, 'Observation')
            if args.store_gif:
                images.append(img)

        n_objects = obs['num_objects']
        print('Episode {}, Step {}, Objects {}, Reward {}'.format(ep,
                                                                  step,
                                                                  n_objects,
                                                                  reward))

        if args.verbose:
            print(obs)
            print(info)

        if args.record_stats:
            states.append(obs['state_observation'])
            goals.append(obs['state_desired_goal'])

        if not args.no_render:
            env.render()

        if args.sleep > 0:
            time.sleep(args.sleep)

        if done:
            break

    if args.images and args.store_gif:
        save_gif(images, '{}/{:02d}.gif'.format(args.out_dir, ep))

if args.record_stats:
    states = np.stack(states)
    print('Observation shapes,', states.shape)
    print('Min:', np.min(states, axis=0))
    print('Mean:', np.mean(states, axis=0))
    print('Max:', np.max(states, axis=0))
    goals = np.stack(goals)
    print('Goal shapes,', goals.shape)
    print('Min:', np.min(goals, axis=0))
    print('Mean:', np.mean(goals, axis=0))
    print('Max:', np.max(goals, axis=0))


