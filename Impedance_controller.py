import os
import sys
import rospy
import csv
import numpy as np
import shlex
import time
import geometry_msgs.msg as geom_msg
import subprocess
from dynamic_reconfigure.client import Client
from absl import app, flags, logging
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import math
import tf
import tf.transformations
import re
import json
import threading
import franka_msgs.msg


class ImpedencecontrolEnv(gym.Env):
    def __init__(self):
        super(ImpedencecontrolEnv, self).__init__()
        self.eepub = rospy.Publisher('/cartesian_impedance_controller/equilibrium_pose',
                                     geom_msg.PoseStamped, queue_size=10)
        self.client = Client("/cartesian_impedance_controller/dynamic_reconfigure_compliance_param_node")

        self.franka_EE_trans = []
        self.franka_EE_quat = []
        self.F_T_EE = np.eye(4)
        self.K_F_ext_hat_K = np.zeros(6)
        self.force_history = []
        self.stop_flag = 0
        self.time_window = 5
        self._lock = threading.Lock()

        self.Fx = self.Fy = self.Fz = 0
        self.Tx = self.Ty = self.Tz = 0
        self.resultant_force = 0
        self.resultant_torque = 0

        rospy.Subscriber("/franka_state_controller/franka_states", franka_msgs.msg.FrankaState, self.franka_callback)
        rospy.Subscriber("/franka_state_controller/franka_states", franka_msgs.msg.FrankaState, self.GetEEforce)

    def set_reference_limitation(self):
        time.sleep(1)
        for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
            self.client.update_configuration({"translational_clip_" + direction: 0.01})
            self.client.update_configuration({"rotational_clip_" + direction: 0.04})
        time.sleep(1)

    def GetEEforce(self, data):
        self.forceandtorque = np.array(data.K_F_ext_hat_K)
        self.Fx, self.Fy, self.Fz = self.forceandtorque[0:3]
        self.Tx, self.Ty, self.Tz = self.forceandtorque[3:6]
        self.resultant_force = math.sqrt(self.Fx ** 2 + self.Fy ** 2 + self.Fz ** 2)
        self.resultant_torque = math.sqrt(self.Tx ** 2 + self.Ty ** 2 + self.Tz ** 2)
        return self.Fx, self.Fy, self.Fz, self.Tx, self.Ty, self.Tz, self.resultant_force, self.resultant_torque

    def get_current_ft(self):
        return np.array([self.Fx, self.Fy, self.Fz, self.Tx, self.Ty, self.Tz])

    def get_current_pose(self):
        pos = self.F_T_EE[:3, 3].flatten()
        quat = tf.transformations.quaternion_from_matrix(self.F_T_EE)
        euler = tf.transformations.euler_from_quaternion(quat)
        return np.concatenate([pos, euler])

    def apply_augmented_tp(self, v_z, omega_z, v_xy_corr, k_z_mod):
        base_kz = 1000.0
        target_kz = base_kz * (1.0 + k_z_mod)
        self.client.update_configuration({"translational_stiffness_z": target_kz})

        dt = 1.0 / 30.0
        curr_p = self.get_current_pose()
        new_pos = curr_p[:3] + np.array([v_xy_corr[0], v_xy_corr[1], v_z]) * dt
        new_yaw = curr_p[5] + omega_z * dt

        self.publish_direct_command(new_pos, [curr_p[3], curr_p[4], new_yaw])

    def publish_direct_command(self, pos, euler):
        msg = geom_msg.PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "panda_link0"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos
        quat = tf.transformations.quaternion_from_euler(euler[0], euler[1], euler[2])
        msg.pose.orientation = geom_msg.Quaternion(*quat)
        self.eepub.publish(msg)

    def monitor_force_change(self, force_history):
        self.stop_flag = 0
        force_history.append(self.resultant_force)
        if len(force_history) > self.time_window:
            force_history.pop(0)
        if len(force_history) == self.time_window:
            prev_avg = np.mean(force_history[:self.time_window // 2])
            curr_avg = np.mean(force_history[self.time_window // 2:])
            if np.abs(curr_avg - prev_avg) / (np.abs(prev_avg) + 1e-6) > 0.5:
                self.stop_flag = 1
        return self.stop_flag

    def initialrobot(self):
        fh = self.force_history
        while True:
            if self.monitor_force_change(fh) == 0:
                target0 = np.array([0.534, 0.002, 0.0547, np.pi, 0, 0])
                self.MovetoPoint(target0)
                curr_pose = self.get_current_pose()
                if np.linalg.norm(curr_pose[:3] - target0[:3]) <= 0.003:
                    break
            else:
                break

    def MovetoPoint(self, Target):
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "0"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(Target[0], Target[1], Target[2])
        quat = R.from_euler('xyz', [Target[3], Target[4], Target[5]]).as_quat()
        msg.pose.orientation = geom_msg.Quaternion(*quat)
        self.eepub.publish(msg)
        time.sleep(3)

    def franka_callback(self, data):
        self.O_T_EE = np.array(data.O_T_EE).reshape(4, 4).T
        f_t_ee_mat = np.array(data.F_T_EE).reshape(4, 4).T
        hand_tcp = np.array([[0.7071, 0.7071, 0, 0], [-0.7071, 0.7071, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]])
        self.F_T_EE = (self.O_T_EE @ np.linalg.inv(f_t_ee_mat)) @ hand_tcp
        self.quat_fl = tf.transformations.quaternion_from_matrix(self.F_T_EE)
        self.Euler_fl = tf.transformations.euler_from_quaternion(self.quat_fl)

    def reset_arm(self):
        time.sleep(1)
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "0"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(0.39, 0, 0.35)
        quat = R.from_euler('xyz', [np.pi, 0, 0]).as_quat()
        msg.pose.orientation = geom_msg.Quaternion(*quat)
        self.eepub.publish(msg)
        time.sleep(1)

    def robot_control_grasptarget(self, target):
        self.MovetoPoint(target + np.array([0, 0, 0.2, 0, 0, 0]))
        fh = []
        while True:
            if self.monitor_force_change(fh) == 0:
                self.MovetoPoint(target)
                curr_p = self.get_current_pose()
                if np.linalg.norm(curr_p[:3] - target[:3]) <= 0.003:
                    break
            else:
                break
        time.sleep(2)
        gripper_control(0)

    def robot_control_place(self, tz):
        target = np.array([0.422, 0.340, tz, np.pi, 0, 0])
        self.MovetoPoint(target + np.array([0, 0, 0.2, 0, 0, 0]))
        self.MovetoPoint(target)
        open_gripper()


def gripper_control(state):
    subprocess.Popen(['rosrun', 'franka_real_demo', 'gripper_run', str(state)])


def open_gripper():
    subprocess.Popen(['rosrun', 'franka_real_demo', 'gripper_run', '1'])


def close_gripper():
    subprocess.Popen(['rosrun', 'franka_real_demo', 'gripper_run', '0'])