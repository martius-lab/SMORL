import copy
from collections import namedtuple

import mujoco_py
import numpy as np
from gym import spaces

import multiworld
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from multiworld.envs.mujoco.util.create_xml import (clean_xml,
                                                    create_object_xml,
                                                    create_root_xml)

Camera = namedtuple('Camera',
                    ['lookat_x', 'lookat_y', 'lookat_z',
                     'distance', 'elevation_angle', 'rotation_angle'])
ROBOT_VIEW = Camera(lookat_x=0, lookat_y=0.5, lookat_z=0.2,
                    distance=1, elevation_angle=-45, rotation_angle=90)
THIRD_PERSON_VIEW = Camera(lookat_x=0, lookat_y=1.0, lookat_z=0.5,
                           distance=0.3, elevation_angle=-45,
                           rotation_angle=270)
TOP_DOWN_VIEW = Camera(lookat_x=0, lookat_y=0, lookat_z=1.5,
                       distance=0.2, elevation_angle=-90, rotation_angle=0)

COLOR_SET_1 = np.array([(0, 0, 1)])
COLOR_SET_BGR = np.array([(0, 0, 1),
                          (0, 1, 0),
                          (1, 0, 0)])
# Generated from `np.array(seaborn.hls_palette(6, l=0.5))[[2, 4, 0]]`
COLOR_SET_3 = np.array([(0.17500000000000004, 0.825, 0.21400000000000008),
                        (0.21400000000000008, 0.17500000000000004, 0.825),
                        (0.825, 0.21400000000000002, 0.17500000000000004)])
# Generated from `np.array(seaborn.hls_palette(6, l=0.5))[[2, 4, 0, 3, 5, 1]]`
COLOR_SET_6 = np.array([(0.17500000000000004, 0.825, 0.21400000000000008),
                        (0.21400000000000008, 0.17500000000000004, 0.825),
                        (0.825, 0.21400000000000002, 0.17500000000000004),
                        (0.17500000000000004, 0.7859999999999998, 0.825),
                        (0.825, 0.17500000000000004, 0.7859999999999998),
                        (0.7859999999999998, 0.825, 0.17500000000000004)])
# Generated from `seaborn.color_palette('hls', 8)`
COLOR_SET_8 = np.array([(0.86, 0.3712, 0.33999999999999997),
                        (0.86, 0.7612000000000001, 0.33999999999999997),
                        (0.5688000000000001, 0.86, 0.33999999999999997),
                        (0.33999999999999997, 0.86, 0.5012000000000001),
                        (0.33999999999999997, 0.8287999999999999, 0.86),
                        (0.33999999999999997, 0.43879999999999986, 0.86),
                        (0.6311999999999998, 0.33999999999999997, 0.86),
                        (0.86, 0.33999999999999997, 0.6987999999999996)])
# Generated from `seaborn.color_palette('hls', 16)`
COLOR_SET_16 = np.array([(0.86, 0.3712, 0.33999999999999997),
                         (0.86, 0.5661999999999999, 0.33999999999999997),
                         (0.86, 0.7612000000000001, 0.33999999999999997),
                         (0.7638, 0.86, 0.33999999999999997),
                         (0.5688000000000001, 0.86, 0.33999999999999997),
                         (0.3738000000000001, 0.86, 0.33999999999999997),
                         (0.33999999999999997, 0.86, 0.5012000000000001),
                         (0.33999999999999997, 0.86, 0.6962000000000002),
                         (0.33999999999999997, 0.8287999999999999, 0.86),
                         (0.33999999999999997, 0.6337999999999998, 0.86),
                         (0.33999999999999997, 0.43879999999999986, 0.86),
                         (0.43619999999999975, 0.33999999999999997, 0.86),
                         (0.6311999999999998, 0.33999999999999997, 0.86),
                         (0.8261999999999998, 0.33999999999999997, 0.86),
                         (0.86, 0.33999999999999997, 0.6987999999999996),
                         (0.86, 0.33999999999999997, 0.5037999999999996)])

BASE_DIR = '/'.join(str.split(multiworld.__file__, '/')[:-2])
asset_base_path = BASE_DIR + '/multiworld/envs/assets/multi_object_sawyer_xyz/'


class SawyerMultiobjectPushEnv(MujocoEnv, Serializable, MultitaskEnv):
    ASSET_FILENAME = 'sawyer_push_multiobj.xml'

    OBJECT_CYLINDER_RADIUS = 0.05
    OBJECT_FRICTION_PARAMS = (0.1, 0.1, 0.02)

    # Hard bounds for objects and hand mocap
    OBJ_BOUNDS_LOW = np.array([-0.24, 0.37, 0.0])
    OBJ_BOUNDS_HIGH = np.array([0.24, 0.75, 0.5])
    MOCAP_BOUNDS_LOW = np.array([-0.245, 0.40, 0.0])
    MOCAP_BOUNDS_HIGH = np.array([0.245, 0.75, 0.5])

    CENTER_WORKSPACE_POS = np.array([0, 0.6])

    INIT_HAND_POS = np.array([0, 0.4, 0.02])
    OBJECT_MIN_INIT_DISTANCE = 2 * OBJECT_CYLINDER_RADIUS
    OBJECT_MIN_GOAL_DISTANCE = 2.5 * OBJECT_CYLINDER_RADIUS
    OBJECT_HAND_MIN_INIT_DISTANCE = 1.2 * OBJECT_CYLINDER_RADIUS
    OBJECT_HAND_MIN_GOAL_DISTANCE = 1.5 * OBJECT_CYLINDER_RADIUS

    OBJECT_Z = 0.02
    MOCAP_TARGET_Z = 0.02
    OFFSCREEN_POS = [-10, -10]

    def __init__(self,
                 reward_info=None,
                 camera=THIRD_PERSON_VIEW,
                 frame_skip=50,
                 pos_action_scale=10 / 100,
                 action_repeat=1,

                 randomize_object_colors=True,
                 color_set=COLOR_SET_16,

                 n_objects=(1, 1),
                 randomize_object_init=True,
                 object_fixed_init_pos=None,
                 objects_init_low=(-0.2, 0.45),
                 objects_init_high=(0.2, 0.7),

                 randomize_hand_goal=False,
                 hand_fixed_goal=None,
                 hand_goal_low=(-0.2, 0.45),
                 hand_goal_high=(0.2, 0.7),
                 n_objects_to_move=None,
                 objects_goal_low=(-0.2, 0.45),
                 objects_goal_high=(0.2, 0.7)):
        self.quick_init(locals())
        self.cam = camera
        self.reward_info = reward_info

        self.pos_action_scale = pos_action_scale
        self.action_repeat = action_repeat

        self.randomize_object_colors = randomize_object_colors
        self.color_set = color_set

        if isinstance(n_objects, int):
            n_objects = (n_objects, n_objects)
        self.n_objects_range = n_objects
        self.n_objects_max = max(self.n_objects_range)

        assert len(self.color_set) >= self.n_objects_max, \
            'Need at least {} colors in color set'.format(self.n_objects_max)

        self.randomize_object_init = randomize_object_init
        self.obj_fixed_init_pos = object_fixed_init_pos
        if self.obj_fixed_init_pos is not None:
            self.obj_fixed_init_pos = np.array(self.obj_fixed_init_pos)
            assert self.obj_fixed_init_pos.shape == (self.n_objects_max, 2), \
                'Number of positions needs to match maximum number of objects'

        if n_objects_to_move is None:
            # Always move all available objects
            self.n_objects_to_move = (self.n_objects_max, self.n_objects_max)
        elif isinstance(n_objects_to_move, int):
            n_objects_to_move = (n_objects_to_move, n_objects_to_move)
            self.n_objects_to_move = (min(n_objects_to_move[0],
                                          self.n_objects_range[0]),
                                      min(n_objects_to_move[1],
                                          self.n_objects_range[1]))

        # Generate XML
        self.obj_stat_prop = _create_object_xml(asset_base_path,
                                                self.n_objects_max,
                                                self.OBJECT_FRICTION_PARAMS,
                                                self.OBJECT_CYLINDER_RADIUS)
        gen_xml = create_root_xml(asset_base_path + self.ASSET_FILENAME)
        MujocoEnv.__init__(self, gen_xml, frame_skip=frame_skip)
        clean_xml(gen_xml)

        # We are using mujoco_py substeps instead of the one in MujocoEnv
        self.sim.nsubsteps = self.frame_skip

        b_low = (self.MOCAP_BOUNDS_LOW[:2],
                 np.tile(self.OBJ_BOUNDS_LOW[:2], self.n_objects_max))
        b_high = (self.MOCAP_BOUNDS_HIGH[:2],
                  np.tile(self.OBJ_BOUNDS_HIGH[:2], self.n_objects_max))
        self.obs_box = spaces.Box(np.concatenate(b_low),
                                  np.concatenate(b_high),
                                  dtype=np.float32)

        self.randomize_hand_goal = randomize_hand_goal
        self.hand_fixed_goal = None
        if hand_fixed_goal is not None:
            self.hand_fixed_goal = np.array(hand_fixed_goal)
            assert self.hand_fixed_goal.shape == (2,)

        self.hand_goal_low = np.array(hand_goal_low)
        assert self.hand_goal_low.shape == (2,)
        self.hand_goal_high = np.array(hand_goal_high)
        assert self.hand_goal_high.shape == (2,)

        def maybe_repeat(arr, n):
            if arr.ndim == 1:
                arr = np.tile(arr, (n, 1))
            return arr

        self.objects_goal_low = maybe_repeat(np.array(objects_goal_low),
                                             self.n_objects_max)
        assert self.objects_goal_low.shape == (self.n_objects_max, 2)
        self.objects_goal_high = maybe_repeat(np.array(objects_goal_high),
                                              self.n_objects_max)
        assert self.objects_goal_high.shape == (self.n_objects_max, 2)

        goal_space_low = np.concatenate((self.hand_goal_low,
                                         self.objects_goal_low.flat))
        goal_space_high = np.concatenate((self.hand_goal_high,
                                          self.objects_goal_high.flat))
        self.goal_box = spaces.Box(goal_space_low, goal_space_high,
                                   dtype=np.float32)

        self.observation_space = spaces.Dict([
            ('observation', self.obs_box),
            ('state_observation', self.obs_box),
            ('desired_goal', self.goal_box),
            ('state_desired_goal', self.goal_box),
            ('achieved_goal', self.obs_box),
            ('state_achieved_goal', self.obs_box),
            ('num_objects', spaces.Discrete(self.n_objects_max + 1))
        ])

        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

        self.objects_init_low = maybe_repeat(np.array(objects_init_low),
                                             self.n_objects_max)
        assert self.objects_init_low.shape == (self.n_objects_max, 2)
        self.objects_init_high = maybe_repeat(np.array(objects_init_high),
                                              self.n_objects_max)
        assert self.objects_init_high.shape == (self.n_objects_max, 2)

        self.state_goal = None  # Set by reset()

        self._simulator_initial_state = copy.deepcopy(self._env_setup())
        self._hand_initial_pos = self._get_endeff_pos()
        self.reset()

    @property
    def model_name(self):
        return asset_base_path + self.ASSET_FILENAME

    def viewer_setup(self):
        self.viewer.cam.lookat[0] = self.cam.lookat_x
        self.viewer.cam.lookat[1] = self.cam.lookat_y
        self.viewer.cam.lookat[2] = self.cam.lookat_z
        self.viewer.cam.distance = self.cam.distance
        self.viewer.cam.elevation = self.cam.elevation_angle
        self.viewer.cam.azimuth = self.cam.rotation_angle
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        action = np.clip(action, -1, 1)

        for _ in range(self.action_repeat):
            self._set_action(action)
            self.sim.step()

        self._ensure_objects_within_bounds()

        obs = self._get_obs()
        reward = self.compute_reward(action, obs)

        return obs, reward, False, self._compute_info_dict()

    def _set_action(self, action):
        mocap_delta_z = self.MOCAP_TARGET_Z - self.data.mocap_pos[0, 2]
        pos_delta = np.concatenate((action * self.pos_action_scale,
                                    [mocap_delta_z]))
        self._mocap_set_action(pos_delta)

    def _mocap_set_action(self, pos_delta):
        _reset_mocap2body_xpos(self.sim)
        new_mocap_pos = self.data.mocap_pos[0] + pos_delta
        new_mocap_pos = np.clip(new_mocap_pos,
                                self.MOCAP_BOUNDS_LOW, self.MOCAP_BOUNDS_HIGH)

        self.data.mocap_pos[0] = new_mocap_pos
        self.data.mocap_quat[0] = np.array([1, 0, 1, 0])

    def _ensure_objects_within_bounds(self):
        """Reset out-of-bounds objects to be within bounds"""
        for qpos_adr in self.mujoco_obj_qpos_adr[:self.n_objects]:
            qpos = self.data.qpos[qpos_adr:qpos_adr + 3]
            clipped_qpos = np.clip(qpos,
                                   self.OBJ_BOUNDS_LOW,
                                   self.OBJ_BOUNDS_HIGH)
            if np.any(qpos != clipped_qpos):
                self.data.qpos[qpos_adr:qpos_adr + 3] = clipped_qpos
                self.sim.forward()

    def _get_obs(self):
        endeff_pos = self._get_endeff_pos().reshape(1, 2)
        object_pos = self._get_used_obj_pos()

        n_objects_unused = self.n_objects_max - self.n_objects
        unused_object_pos = np.zeros((n_objects_unused, 2))

        state = np.concatenate((endeff_pos, object_pos, unused_object_pos))
        state = state.reshape(-1).astype(np.float32)

        goal = self.state_goal.copy().reshape(-1).astype(np.float32)

        return dict(observation=state,
                    state_observation=state,
                    desired_goal=goal,
                    state_desired_goal=goal,
                    achieved_goal=state,
                    state_achieved_goal=state,
                    num_objects=np.array(self.n_objects, dtype=np.float32))

    def _compute_info_dict(self):
        endeff_pos = self._get_endeff_pos()
        hand_distance = np.linalg.norm(self.state_goal[0] - endeff_pos)

        object_pos = self._get_used_obj_pos()
        object_goals = self.state_goal[1:self.n_objects + 1]

        object_dists = np.linalg.norm(object_pos - object_goals, axis=1)
        hand_object_dists = np.linalg.norm(endeff_pos[np.newaxis] - object_pos,
                                           axis=1)

        n_items = self.n_objects + 1
        avg_distance = (hand_distance + object_dists.sum()) / n_items
        avg_object_distance = object_dists.sum() / self.n_objects

        threshold = n_items * self.OBJECT_CYLINDER_RADIUS
        success = float(hand_distance + np.sum(object_dists) < threshold)

        info = dict(success=success,
                    avg_distance=avg_distance,
                    avg_object_distance=avg_object_distance,
                    hand_distance=hand_distance,
                    **{'object{}_distance'.format(idx): dist
                       for idx, dist in enumerate(object_dists)},
                    **{'object{}_distance'.format(idx): np.nan
                       for idx in range(self.n_objects, self.n_objects_max)},
                    **{'touch{}_distance'.format(idx): dist
                       for idx, dist in enumerate(hand_object_dists)},
                    **{'touch{}_distance'.format(idx): np.nan
                       for idx in range(self.n_objects, self.n_objects_max)})

        return info

    def _get_used_obj_pos(self):
        return self.data.body_xpos[self.mujoco_obj_ids[:self.n_objects]][:, :2]

    def _get_all_obj_pos(self):
        return self.data.body_xpos[self.mujoco_obj_ids][:, :2]

    def _get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id][:2]

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    def _sample_random_object_positions(self, n_objects):
        endeff_pos = self._get_endeff_pos()
        low = self.objects_init_low[:n_objects]
        high = self.objects_init_high[:n_objects]
        while True:
            obj_pos = np.random.uniform(low, high)

            obj_dists = pairwise_distances(obj_pos)
            obj_hand_dists = single_point_distance(obj_pos, endeff_pos)
            if (np.all(obj_hand_dists >= self.OBJECT_HAND_MIN_INIT_DISTANCE)
                    and np.all(obj_dists >= self.OBJECT_MIN_INIT_DISTANCE)):
                break

        return obj_pos

    def _set_initial_object_positions(self):
        if self.randomize_object_init:
            obj_pos = self._sample_random_object_positions(self.n_objects)
        else:
            obj_pos = self.obj_fixed_init_pos[:self.n_objects]

        if self.n_objects_max > self.n_objects:
            n_offscreen_obj = self.n_objects_max - self.n_objects
            offscreen_obj_pos = np.tile(self.OFFSCREEN_POS,
                                        (n_offscreen_obj, 1))
            obj_pos = np.concatenate((obj_pos, offscreen_obj_pos))

        for idx in range(self.n_objects_max):
            self.set_object_xy(idx, obj_pos[idx])

    def _set_object_colors(self):
        if self.randomize_object_colors:
            new_colors = self.color_set[np.random.choice(len(self.color_set),
                                                         self.n_objects,
                                                         replace=False)]
        else:
            new_colors = self.color_set[:self.n_objects]

        colors = self.model.geom_rgba[self.mujoco_obj_geom_ids]
        colors[:self.n_objects, :3] = new_colors
        colors[:self.n_objects, 3] = 1
        colors[self.n_objects:, 3] = 0  # Set alpha of unused objects to zero
        self.model.geom_rgba[self.mujoco_obj_geom_ids] = colors

    def reset(self):
        self.sim.set_state(self._simulator_initial_state)
        self.sim.forward()

        self.n_objects = np.random.randint(self.n_objects_range[0],
                                           self.n_objects_range[1] + 1)
        self._set_initial_object_positions()
        self._set_object_colors()
        self.sim.forward()

        self.state_goal = self._sample_goals()

        return self._get_obs()

    def _env_setup(self):
        """Create initial good state for environment"""
        self.mujoco_obj_ids = [self.model.body_names.index('object' + str(i))
                               for i in range(self.n_objects_max)]
        self.mujoco_obj_geom_ids = [self.model.geom_name2id('object' + str(i))
                                    for i in range(self.n_objects_max)]
        self.mujoco_obj_qpos_adr = []
        self.mujoco_obj_qvel_adr = []
        for obj_id in self.mujoco_obj_ids:
            assert self.model.body_jntnum[obj_id] == 1
            joint_id = self.model.body_jntadr[obj_id]
            self.mujoco_obj_qpos_adr.append(self.model.jnt_qposadr[joint_id])
            self.mujoco_obj_qvel_adr.append(self.model.jnt_dofadr[joint_id])

        # Bring arm into good starting position (hacky!)
        arm_qpos = np.array([1.78026069e+00, -6.84415781e-01, -1.54549231e-01,
                             2.30672090e+00, 1.93111471e+00,  1.27854012e-01,
                             1.49353907e+00])
        self.data.qpos[:7] = arm_qpos

        _reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move arm into initial position
        self.data.set_mocap_pos('mocap', self.INIT_HAND_POS)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        for _ in range(10):
            self.sim.step()

        # Set initial object configuration
        if self.randomize_object_init:
            obj_pos = self._sample_random_object_positions(self.n_objects_max)
        else:
            obj_pos = self.obj_fixed_init_pos[:self.n_objects_max]

        for idx, pos in enumerate(obj_pos):
            self.set_object_xy(idx, pos)

        _reset_mocap_welds(self.sim)
        self.sim.forward()

        return self.sim.get_state()

    def compute_rewards(self, action, obs, info=None):
        bs = len(obs['state_achieved_goal'])
        n_items = obs['num_objects'] + 1
        achieved = obs['state_achieved_goal'].reshape(bs, -1, 2).copy()
        desired = obs['state_desired_goal'].reshape(bs, -1, 2).copy()

        distances = np.linalg.norm(achieved - desired, axis=2)

        # Each element of the batch can have different number of objects,
        # that need to be summed up for the reward, so this is a bit ugly.
        rewards = np.zeros_like(distances[:, 0])
        for item in range(self.n_objects_max + 1):
            selection = n_items > item
            rewards[selection] += (-1) * distances[selection, item]

        return rewards

    def compute_reward(self, action, obs, info=None):
        n_items = int(obs['num_objects'] + 1)
        achieved = obs['state_achieved_goal'].reshape(-1, 2)[:n_items]
        desired = obs['state_desired_goal'].reshape(-1, 2)[:n_items]
        return -np.linalg.norm(achieved - desired, axis=1).sum()

    """
    Multitask functions
    """
    @property
    def goal_dim(self) -> int:
        return 2 * (self.n_objects_max + 1)

    def sample_goals(self, batch_size):
        # Potentially super slow for large batches
        goals = np.array([self._sample_goals() for _ in range(batch_size)])

        return {
            'num_objects': np.array(self.n_objects,
                                    dtype=np.float32).repeat(batch_size),
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def _sample_hand_goal(self):
        if self.randomize_hand_goal:
            hand_goal = np.random.uniform(self.hand_goal_low,
                                          self.hand_goal_high)
        elif self.hand_fixed_goal is not None:
            hand_goal = self.hand_fixed_goal
        else:
            hand_goal = self._hand_initial_pos

        return hand_goal

    def _sample_goals(self):
        min_objects_to_move = min(self.n_objects, self.n_objects_to_move[0])
        max_objects_to_move = min(self.n_objects, self.n_objects_to_move[1])
        n_objects_to_move = np.random.randint(min_objects_to_move,
                                              max_objects_to_move + 1)
        objects_to_keep = np.random.choice(self.n_objects,
                                           self.n_objects - n_objects_to_move,
                                           replace=False)
        cur_object_pos = self._get_all_obj_pos()

        low = self.objects_goal_low[:self.n_objects]
        high = self.objects_goal_high[:self.n_objects]

        while True:
            hand_goal = self._sample_hand_goal()
            obj_goals = np.random.uniform(low, high)
            obj_goals[objects_to_keep] = cur_object_pos[objects_to_keep]

            obj_dists = pairwise_distances(obj_goals)
            obj_hand_dists = single_point_distance(obj_goals, hand_goal)
            if (np.all(obj_hand_dists >= self.OBJECT_HAND_MIN_GOAL_DISTANCE)
                    and np.all(obj_dists >= self.OBJECT_MIN_GOAL_DISTANCE)):
                break

        n_objects_unused = self.n_objects_max - self.n_objects
        return np.concatenate((hand_goal[np.newaxis],
                               obj_goals,
                               np.zeros((n_objects_unused, 2))))

    def get_goal(self):
        goal = self.state_goal.copy().reshape(-1).astype(np.float32)
        return {
            'num_objects': np.array(self.n_objects, dtype=np.float32),
            'desired_goal': goal,
            'state_desired_goal': goal,
        }

    def set_goal(self, goal):
        assert goal['num_objects'] == self.n_objects, \
            ('Goal is incompatible with env: goal has {} objects, but env '
             'currently {}').format(goal['num_objects'], self.n_objects)
        self.state_goal = goal['state_desired_goal'].copy().reshape(-1, 2)

    def set_to_goal(self, goal):
        assert goal['num_objects'] == self.n_objects, \
            ('Goal is incompatible with env: goal has {} objects, but env '
             'currently {}').format(goal['num_objects'], self.n_objects)

        state_goal = goal['state_desired_goal'].reshape(-1, 2)
        self.set_hand_xy(state_goal[0])

        for idx, pos in enumerate(state_goal[1:self.n_objects + 1]):
            self.set_object_xy(idx, pos)

        _reset_mocap_welds(self.sim)
        self.sim.forward()

    def convert_obs_to_goals(self, obs):
        return obs

    def set_hand_xy(self, xy):
        self.data.set_mocap_pos('mocap', np.array([xy[0], xy[1],
                                                  self.MOCAP_TARGET_Z]))
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        for _ in range(10):
            self.sim.step()

    def set_object_xy(self, idx, pos):
        qpos_adr = self.mujoco_obj_qpos_adr[idx]
        qpos = np.array([pos[0], pos[1], self.OBJECT_Z, 1, 0, 0, 0])
        self.data.qpos[qpos_adr:qpos_adr + 7] = qpos
        qvel_adr = self.mujoco_obj_qvel_adr[idx]
        self.data.qvel[qvel_adr:qvel_adr + 6] = np.zeros((6,))

    def get_env_state(self):
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        state = joint_state, mocap_state
        return copy.deepcopy(state)

    def set_env_state(self, state):
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.data.set_mocap_pos('mocap', mocap_pos)
        self.data.set_mocap_quat('mocap', mocap_quat)
        self.sim.forward()


def single_point_distance(X, y, ord=2):
    """Compute euclidian distances between set of points and another point"""
    assert X.ndim == 2 and y.ndim == 1
    diffs = X - y[np.newaxis]

    return np.linalg.norm(diffs, ord=ord, axis=-1)


def pairwise_distances(X, ord=2):
    """Compute pairwise euclidian distances for set of points in R^D

    Returns only upper triangular of the distance matrix.
    """
    assert X.ndim == 2
    diffs = X[np.newaxis, :, :] - X[:, np.newaxis, :]
    dist = np.linalg.norm(diffs, ord=ord, axis=-1)

    return dist[np.triu_indices(dist.shape[0], k=1)]


def _create_object_xml(output_dir, num_objects, friction_params,
                       cylinder_radius):
    return create_object_xml(filename=output_dir,
                             num_objects=num_objects,
                             friction_params=friction_params,
                             cylinder_radius=0.04,
                             use_textures=False,
                             object_meshes=None,
                             finger_sensors=False,
                             sliding_joints=False,
                             load_dict_list=None,
                             object_mass=1,  # Unused by `create_object_xml`
                             maxlen=0.06,  # Not needed
                             minlen=0.01,  # Not needed
                             obj_classname=None,  # Not needed
                             block_height=0.02,  # Not needed
                             block_width=0.02)  # Not needed


def _reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.

    Adapted from gym.envs.robotics.utils
    """

    if (sim.model.eq_type is None or
            sim.model.eq_obj1id is None or
            sim.model.eq_obj2id is None):
        return
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert (mocap_id != -1)
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]


def _reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation.

    Adapted from gym.envs.robotics.utils
    """
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
    sim.forward()
