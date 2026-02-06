import threading
import sys
import rospy
import numpy as np
import shlex
import time
import signal
import subprocess
import cv2
import queue
import gymnasium as gym
from gymnasium import spaces
import pyrealsense2 as rs
from ultralytics import YOLO
import geometry_msgs.msg as geom_msg
from dynamic_reconfigure.client import Client
from scipy.spatial.transform import Rotation as R
import math
import tf
import tf.transformations

from Impedance_controller import *
from constants import *
from init_results_detection import detector, pose_calculation, start_cameras, camera_cleanup
from dual_camera_detection import *
from SafetyPolicy import SafetyPolicy


class FrankaRLEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, FLAGS=None):
        super(FrankaRLEnv, self).__init__()

        self.roscore_process = None
        self.impedence_controller = None
        self.imp_controller = None
        self.camera = DualRealsenseCamera()
        self.initialized = False
        self.control_rate = 30
        self.step_count = 0
        self.sp = SafetyPolicy()

        self.w1 = 20.0
        self.w2 = 1.0
        self.force_threshold = 2.0
        self.prev_z = 0.0

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]).astype(np.float32),
            high=np.array([1.0, 1.0, 1.0]).astype(np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            'robot_state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(7,),
                dtype=np.float32
            ),
            'object_detected': spaces.Discrete(2),
            'object_position': spaces.Box(
                low=np.array([-np.inf, -np.inf, -np.inf]),
                high=np.array([np.inf, np.inf, np.inf]),
                dtype=np.float32
            ),
            'object_rotation': spaces.Box(
                low=np.array([-np.pi, -np.pi, -np.pi]),
                high=np.array([np.pi, np.pi, np.pi]),
                dtype=np.float32
            ),
            'camera_source': spaces.Discrete(3)
        })

        if FLAGS is not None and not self.initialize(FLAGS):
            raise RuntimeError("Environment initialization failed")

    def initialize(self, FLAGS):
        try:
            if not self._is_roscore_running():
                self.roscore_process = subprocess.Popen('roscore')
                time.sleep(2)

            if not self.camera.start_cameras():
                raise RuntimeError("Failed to start cameras")

            self.impedence_controller = subprocess.Popen(
                ['roslaunch', 'serl_franka_controllers', 'impedance.launch',
                 f'robot_ip:={FLAGS.robot_ip}', f'load_gripper:={FLAGS.load_gripper}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(5)

            if not rospy.core.is_initialized():
                rospy.init_node('franka_rl_env', anonymous=True)

            self.imp_controller = ImpedencecontrolEnv()
            self.initialized = True
            return True

        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            self.cleanup()
            return False

    def step(self, action):
        if not self.initialized:
            raise RuntimeError("Environment not initialized")

        try:
            self.step_count += 1

            v_p_corr = action[0:2] * 0.002
            k_z_mod = action[2]

            curr_ft = self.imp_controller.get_current_ft()
            safe_v_p = self.sp.get_safe_velocity(v_p_corr, curr_ft)

            self.imp_controller.apply_augmented_tp(
                v_z=0.002,
                omega_z=0.1,
                v_xy_corr=safe_v_p,
                k_z_mod=k_z_mod
            )

            rospy.sleep(1.0 / self.control_rate)

            ft = self.imp_controller.get_current_ft()
            pose = self.imp_controller.get_current_pose()
            robot_obs_7d = np.concatenate([ft, [pose[2]]]).astype(np.float32)

            camera_obs = self.camera.get_latest_detection()

            curr_z = robot_obs_7d[-1]
            delta_h = curr_z - self.prev_z
            f_norm = np.linalg.norm(robot_obs_7d[:3])

            reward = (self.w1 * delta_h) - (self.w2 * max(0, f_norm - self.force_threshold))

            self.prev_z = curr_z
            observation = self._format_observation(robot_obs_7d, camera_obs)
            done = self._check_termination(observation)

            return observation, reward, done, {}

        except Exception as e:
            print(f"Step failed: {str(e)}")
            return None, 0, True, {'error': str(e)}

    def reset(self):
        if not self.initialized:
            raise RuntimeError("Environment not initialized")

        try:
            self.imp_controller.reset_arm()
            time.sleep(1)
            self.imp_controller.set_reference_limitation()
            time.sleep(1)

            self._open_gripper()
            time.sleep(2)

            ft = self.imp_controller.get_current_ft()
            pose = self.imp_controller.get_current_pose()
            robot_obs_7d = np.concatenate([ft, [pose[2]]]).astype(np.float32)
            self.prev_z = robot_obs_7d[-1]

            camera_obs = self.camera.get_latest_detection()
            observation = self._format_observation(robot_obs_7d, camera_obs)

            return observation

        except Exception as e:
            print(f"Reset failed: {str(e)}")
            return None

    def render(self, mode='human'):
        if mode == 'rgb_array':
            detection = self.camera.get_latest_detection()
            if detection and 'image' in detection:
                return detection['image']
            return None
        elif mode == 'human':
            pass

    def close(self):
        self.cleanup()

    def _format_observation(self, robot_obs_7d, camera_obs):
        return {
            'robot_state': robot_obs_7d,
            'object_detected': 1 if camera_obs and camera_obs['object_detected'] else 0,
            'object_position': camera_obs['position'] if camera_obs else np.zeros(3),
            'object_rotation': camera_obs['rotation'] if camera_obs else np.zeros(3),
            'camera_source': camera_obs['camera_source'] if camera_obs else 0
        }

    def _check_termination(self, observation):
        f_norm = np.linalg.norm(observation['robot_state'][:3])
        if f_norm > 45.0:
            return True
        if observation['robot_state'][6] > 0.15:
            return True
        return False

    def _is_roscore_running(self):
        try:
            master = rospy.get_master()
            return master.is_online()
        except:
            return False

    def cleanup(self):
        camera_cleanup()
        if self.impedence_controller:
            self.impedence_controller.terminate()
        if self.roscore_process:
            self.roscore_process.terminate()

    def _open_gripper(self):
        pass