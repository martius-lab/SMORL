import cv2
import numpy as np
import os

import torch
from rlkit.envs.gt_wrapper import GTWrappedEnv
from rlkit.envs.unstructured_gt_wrapper import UnstructuredGTWrappedEnv
from rlkit.envs.scalor_wrapper import SCALORWrappedEnv
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
import einops


def get_render_wrapper(env, wrapped_env, video_params,  render_bboxes, env_params,
                       image_size=128,
                       camera=sawyer_init_camera_zoomed_in):
    if isinstance(env, GTWrappedEnv):
        super_cls = GTWrappedEnv
        goal_sampling_mode = 'reset_of_env'
        obs_image_key = "image_observation"
        goal_image_key = "image_desired_goal"
    elif isinstance(env, UnstructuredGTWrappedEnv):
        super_cls = UnstructuredGTWrappedEnv
        goal_sampling_mode = 'env_goal'
        obs_image_key = "image_observation"
        goal_image_key = "image_desired_goal"
    elif isinstance(env, SCALORWrappedEnv):
        super_cls = SCALORWrappedEnv
        goal_sampling_mode = 'reset_of_env'
        obs_image_key = "image_obs_bbox"
        goal_image_key = "image_desired_goal_bbox"
    else:
        raise ValueError('Unsupported env object {}'.format(env))

    wrapper_cls = type('RenderWrappedEnv', (super_cls,),
                       {'__init__': _init, '_render': _render, 'step': _step, 'reset': _reset})
    wrapper = wrapper_cls(wrapped_env, image_size, camera, video_params, render_bboxes, **env_params)
    wrapper._goal_sampling_mode = goal_sampling_mode
    wrapper._obs_image_key = obs_image_key
    wrapper._goal_image_key = goal_image_key

    return wrapper


def _init(self, wrapped_env, image_size, camera, video_params, render_bboxes, **kwargs):
    if isinstance(wrapped_env, ImageEnv):
        image_env = wrapped_env
    else:
        image_env = ImageEnv(wrapped_env, image_size,
                             init_camera=camera,
                             transpose=True,
                             normalize=True)
    self.video_params = video_params
    self.n_episodes = 0
    if self.video_params is not None:
        self.n_videos = -1
    super(self.__class__, self).__init__(image_env, **kwargs)
    self.render_bboxes = render_bboxes


def _transform_image(img, imsize):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    img = img.reshape(3, imsize, imsize).transpose()[..., ::-1]
    img = np.flip(img, axis=0)
    return img


def _reset(self):
    self.n_episodes += 1
    return super(self.__class__, self).reset()


def _render(self, obs, reward=None):
    img = _transform_image(obs[self._obs_image_key], self.imsize)
    scale = 4
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale) * 2
    dim = (width, height)

    goal = _transform_image(obs[self._goal_image_key], self.imsize)
    img_with_goal = einops.rearrange([img, goal], "b h w c -> (b h) w c")
    img_with_goal = (img_with_goal*255).astype(np.uint8)
    img_with_goal = cv2.resize(img_with_goal, dsize=dim)
    if reward is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 10)
        fontScale = 0.4
        fontColor = (255,255,255)
        lineType = 1
        cv2.putText(img_with_goal, "reward: {0:1.2} ".format(reward),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
    if self.video_params is not None:
        dir_path = os.path.join(self.video_params["path"], "policy_videos")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if self.n_videos == -1:
            self.n_videos = 0
            self.out = cv2.VideoWriter(os.path.join(dir_path, "{0}.avi".format(self.n_videos)),fourcc, 5.0, dim)
        self.out.write(img_with_goal)
        if self.n_episodes > self.video_params["video_episodes"]:
            self.out.release()
            self.n_videos += 1
            self.n_episodes = 0
            if self.video_params["one_video"]:
                self.video_params = None
            else:
                self.out = cv2.VideoWriter(os.path.join(dir_path, "{0}.avi".format(self.n_videos)), fourcc, 5.0, dim)
        if not self.video_params["only_video"]:
            cv2.imshow('Goal and observations', img_with_goal)
            cv2.waitKey(1)
    else:
        cv2.imshow('Goal and observations', img_with_goal)
        cv2.waitKey(1)



def _step(self, action):
    obs, reward, done, info = super(self.__class__, self).step(action)
    self._render(obs, reward)
    return obs, reward, done, info
