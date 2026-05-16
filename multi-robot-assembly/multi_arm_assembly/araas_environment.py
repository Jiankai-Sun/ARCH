import pyatk
import pyaraas
from pyatk import Transform, Vector, Part, Actor, Gripper
import time
import numpy as np
import gym
from gym import spaces, register
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from typing import List, Any
import copy
import argparse
import math
import traceback
from my_path_planner import PathPlanner
import json
from gym.envs.registration import register
from multi_arm_assembly.utils import log, log_list, task_from_name, R0_POSE, R1_POSE, solve_motion_plan, \
    check_collision, WORLD_T_HOLE_START, ROOT_DIR, WORLD_T_ZIVID_HOME, set_seed, wxyz_to_xyzw, xyzw_to_wxyz
set_seed(123)
from multi_arm_assembly.utils import WORLD_T_FIXTURE_POSE as WORLD_T_FIXTURE_START
from multi_arm_assembly.utils import solve_ik_araas as solve_ik
import multi_arm_assembly.pb_utils as pbu
from multi_arm_assembly.utils import visualize_depth_image, OBJECT_DICT, split_obj_dict
import cv2
import os
import pickle
from datetime import datetime

import trio

def obj_to_str(d):
    return json.dumps(d)

def force_sample_to_str(fs):
    return obj_to_str({"force": fs["force"].get_values(), "moment": fs["moment"].get_values()})

async def move_robot(robot, goal, execution_time=10):
    # goal = robot.get_transform()
    # goal.position.z += 200
	robot.moving = True
	await robot.move_cartesian(goal, execution_time)
	robot.moving = False

async def open_gripper_func(gripper, open_width):
    # goal = robot.get_transform()
    # goal.position.z += 200
	gripper.moving = True
	await gripper.open(open_width, 1)
	gripper.moving = False


# async def move_camera(robot, goal, execution_time=10):
# 	# goal = robot.get_transform()
# 	# goal.position.z += 200
# 	robot.moving = True
# 	await robot.move_cartesian(goal, execution_time)
# 	robot.moving = False

async def sample_robot(robot, 
                       demo_json_path=os.path.join(ROOT_DIR, 'multi_arm_assembly/logs/demos_araas/', '{:04d}_{}_{}.pkl'.format(0, 0, 0, ))):
	# sample at 100 HZ
    start = time.time()
    while True:
        await trio.sleep(0.01)
        # print(time.time()-start)
        # print(robot.get_transform())
        robot_transform = [arr.tolist() for arr in APose.from_transform(robot.get_transform()).to_list()]
        # data = {'time': time.time()-start, 'ee_pose': robot_transform}
        data = robot_transform[0] + robot_transform[1]
        print(data)
        with open(demo_json_path, 'a+') as f:
            json.dump(data, f)  # Use indent for pretty printing
        if not robot.moving:
            break
    # return demo

async def sample_gripper_func(gripper):
	# sample at 100 HZ
	start = time.time()
	while True:
		await trio.sleep(0.01)
		print(time.time()-start)
		print(gripper.opening)
		if not gripper.moving:
			break
    # return demo

async def sample_camera(camera):
	# sample at 100 HZ
	start = time.time()
	while True:
		await trio.sleep(0.01)
		print(time.time()-start)
		print(camera.get_transform())
        # camera.capture('depth', 512, '{}.png'.format(time.time()), 1100)
		if not camera.moving:
			break


UNIT_T = Transform(np.array([0, 0, 0, 0, 0, 0]))
H = 10
UNIT_OFFSET = 1000
# OPEN_WIDTH = 40   # hexagon
# OPEN_WIDTH = 55
# WORKCELL_NAME = "MAR_PEG_85_long"
WORKCELL_NAME = "Assembly"


@dataclass
class APose():
    trans: np.array
    quat: np.array

    @staticmethod
    def from_transform(transform: Transform):
        pos = transform.position
        quat = transform.rotation
        pos_np = np.array([pos.x, pos.y, pos.z]) / UNIT_OFFSET
        quat_np = np.array([quat.x, quat.y, quat.z, quat.w])
        return APose(pos_np, quat_np)

    @property
    def rotm(self):
        return R.from_quat(self.quat).as_matrix()

    @property
    def euler(self):
        return R.from_quat(self.quat).as_euler("XYZ")

    def to_list(self):
        return [self.trans, self.quat]

    def to_transform(self):
        return Transform(
            np.concatenate([np.array(self.trans) * UNIT_OFFSET, R.from_quat(self.quat).as_euler("XYZ")], axis=0))


@dataclass
class RobotState:
    robot: Actor
    gripper: Gripper
    force_sensor: Any = None
    init_world_force: np.array = np.zeros(3)
    init_world_torque: np.array = np.zeros(3)
    tool_pose: np.array = np.zeros(6)
    tool_vel: np.array = np.zeros(6)
    force_history: np.array = np.zeros([H, 6])
    world_force: np.array = np.zeros(6)
    calibration_sample: int = 0

    def copy(self):
        return RobotState(
            self.robot,
            self.gripper,
            self.force_sensor,
            copy.deepcopy(self.init_world_force),
            copy.deepcopy(self.init_world_torque),
            copy.deepcopy(self.tool_pose),
            copy.deepcopy(self.tool_vel),
            copy.deepcopy(self.force_history),
            copy.deepcopy(self.world_force),
            copy.deepcopy(self.calibration_sample)
        )


# Controller parameters
M_DEFAULT_LIN = 20
M_DEFAULT_ANG = 20
KP_DEFAULT = 200
KD_DEFAULT_LIN = 2 * np.sqrt(np.multiply(M_DEFAULT_LIN, KP_DEFAULT))
KD_DEFAULT_ANG = 2 * np.sqrt(np.multiply(M_DEFAULT_ANG, KP_DEFAULT))
FORCE_LIMIT_TRANS = 100  # This is the force READING limit. Force readings get clipped by this
VELOCITY_LIMIT_TRANS = 20
WORLD_T_R0_HOME = Transform([436.46, 299.20, 533.14, 0, -np.pi, 0])
WORLD_T_R1_HOME = Transform([-436.46, 299.20, 533.14, 0, -np.pi, 0])
# self.r2.get_flange_transform()
# WORLD_T_ZIVID_HOME = Transform([203.014038,-784.345520,802.262268, -131.2885996, 0, 0])
# WORLD_T_ZIVID_HOME = Transform(151.870987,-595.301941,760.453064, -0.910941,0.013006,-0.008058,0.412253)
WORLD_T_ZIVID_HOME = APose(*WORLD_T_ZIVID_HOME).to_transform() # Transform([451.870987,-595.301941,760.453064, -131.2885996/180*np.pi, 0, 0])

@dataclass
class AdmittanceGains:
    M: np.array = np.array([M_DEFAULT_LIN, M_DEFAULT_LIN, M_DEFAULT_LIN, M_DEFAULT_ANG, M_DEFAULT_ANG, M_DEFAULT_ANG])
    # kp: np.array = np.array([KP_DEFAULT*1.5, KP_DEFAULT*1.5, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT])
    kp: np.array = np.array([KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT])
    kd: np.array = np.array(
        [KD_DEFAULT_LIN, KD_DEFAULT_LIN, KD_DEFAULT_LIN, KD_DEFAULT_ANG, KD_DEFAULT_ANG, KD_DEFAULT_ANG])
    force_limit: np.array = np.array([FORCE_LIMIT_TRANS, FORCE_LIMIT_TRANS, FORCE_LIMIT_TRANS, 1, 1, 1])


class AssemblyMultiArm(gym.Env):
    def __init__(self,
                 friction=0.02,
                 force_limit=FORCE_LIMIT_TRANS,
                 dt=1. / 125.0,
                 max_v=VELOCITY_LIMIT_TRANS,
                 max_rad=1,
                 mass=1,
                 verbose=False,
                 use_rotation_randomization=False,
                 zero_noise=True,
                 enable_hardware=False,
                 task_name=None,
                 **kwargs):

        self.mode = task_name
        self.init_cfg = task_from_name(task_name)
        self.friction = friction
        self.force_limit = force_limit
        self.torque_limit = 1
        self.zero_noise = zero_noise
        self.enable_hardware = enable_hardware
        self.teleport = not self.enable_hardware
        if int(self.init_cfg.selected_obj_id) in [0, ]:
            self.OPEN_WIDTH = 40
        elif int(self.init_cfg.selected_obj_id) in [1, 4, ]:
            self.OPEN_WIDTH = 43
        elif int(self.init_cfg.selected_obj_id) in [5, 8]:
            self.OPEN_WIDTH = 50
        else:  # [2, 3, 6, 7, ]
            self.OPEN_WIDTH = 58
        # self.teleport = False

        self.skip_setup = False
        self.debug_action = self.init_cfg.DEBUG_ACTION  # True

        self.dt = dt
        self.forces_zeroed = False
        self.enable_compliance = True
        self.max_v = max_v
        self.max_w = max_rad
        self.max_rad = max_rad
        self.mass = mass
        self.use_rotation_randomization = use_rotation_randomization
        self.verbose = verbose
        self.max_time_per_step = 0.5
        self.offset_to_tip = np.array([0, 0, 0.217])

        self.translation_limit = 0.05  # in m
        self.rotation_limit = 0.1  # 1 / 180 * np.pi  # in rad
        self.max_step = 2  # 100 #

        self.w = pyaraas.start(WORKCELL_NAME, enable_hardware=self.enable_hardware)
        self.path_planner = PathPlanner(self.w, num_workers=4, expand_roadmap=True)
        self.setup = False

        self.collisions_enabled = True
        if WORKCELL_NAME in ["Assembly"]:
            self.aruna_table = self.w.get_machine('Aruna Table-0')
            self.aruna_table.set_collision_model(False, concave=False)
            self.fixture_0 = self.w.get_peripheral('fixture-0')
            self.fixture_0.set_collision_model(True, concave=True)
        self.r0 = self.w.get_robot("UR10e-0")
        # self.r0.set_collision_model(self.collisions_enabled, self.collisions_enabled)
        # self.r0_plate = self.w.get_peripheral("UR10e-Mounting-Plate-ST-RB-001-0009-1")
        # self.r0_plate.set_collision_model(self.collisions_enabled, self.collisions_enabled)
        self.r1 = self.w.get_robot("UR10e-1")
        # self.r1.set_collision_model(self.collisions_enabled, self.collisions_enabled)
        # self.r1_plate = self.w.get_peripheral("UR10e-Mounting-Plate-ST-RB-001-0009-0")
        # self.r1_plate.set_collision_model(self.collisions_enabled, self.collisions_enabled)
        self.robot_list = [self.r0, self.r1]

        if WORKCELL_NAME in ["Assembly"]:
            self.r2 = self.w.get_robot("UR10-0")
            # self.r2.set_collision_model(self.collisions_enabled, self.collisions_enabled)
            # self.r2_plate = self.w.get_peripheral("VentionAssembly_273382_v22-0")
            # self.r2_plate.set_collision_model(self.collisions_enabled, self.collisions_enabled)
            self.robot_list.append(self.r2)
            print('--------------- Robot Pose ----------------')
            print('self.r0: {}\nself.r1: {}\nself.r2: {}'
                  .format(self.r0.get_base_transform(), self.r1.get_base_transform(), 
                          self.r2.get_base_transform()))
        self.r0_base_frame_apose = APose.from_transform(self.r0.get_base_transform())
        # elif WORKCELL_NAME in ["MAR_PEG_85_long", "MAR_PEG_85"]:
        #     self.block = self.w.get_peripheral("just_a_medium_cube-0")
        #     self.block.set_collision_model(self.collisions_enabled, self.collisions_enabled)

        self.f0 = self.w.get_force_sensor('UR10e-0')
        self.f1 = self.w.get_force_sensor('UR10e-1')

        self.g0 = self.w.get_gripper("Rq85-0")
        # self.g0.set_collision_model(self.collisions_enabled, self.collisions_enabled)
        self.g1 = self.w.get_gripper("Rq85-1")
        # self.g1.set_collision_model(self.collisions_enabled, self.collisions_enabled)
        self.c0 = self.w.get_camera('Zivid2+-0')
        self.zivid_capture_times = 1  # 3  # for smooth and reducing error
        if not self.enable_hardware:
            self.c0.fov = 0.3910485208034515
            # cx = sensor_dim/(2*tan(fov)) = 512 / (2*tan(0.39)) = 622.79
        else:
            # setting hardware camera parameters not allowed in sim mode
            # self.c0.set_hardware_parameters(os.path.join(ROOT_DIR, 'multi_arm_assembly/zivid_parameters.json'))
            pass
        print('init cx: {}, cy: {}, fx: {}, fy: {}, fov: {}, sensor_dim: {}, parameter_json: {}'
              .format(self.c0.cx, self.c0.cy, self.c0.fx, self.c0.fy, self.c0.fov, self.c0.sensor_dim, self.c0.parameter_json,))
        self.parts_list = {}
        self.tool_friction = friction
        for k, v in OBJECT_DICT.items():
            self.parts_list[k] = self.w.add_part(k, k, APose(*v['start_pose']).to_transform())
            self.parts_list[k].set_collision_model(True, True)  # collision, cancave
            # self.parts_list[k].set_friction(self.tool_friction)
            self.parts_list[k].set_margin(0.001)
        self.img_dir = os.path.join(ROOT_DIR, 'multi_arm_assembly/logs/images/')
        self.insert_approach_1 = Transform([0, 0, 280, 0, 0, 0])
        self.after_insert_approach_1 = Transform([0, 0, 180, 0, 0, 0])
        os.makedirs(self.img_dir, exist_ok=True)
        self.movement_type = 'cartesian'  # 'joint' #  'cartesian', 'trajectory',  space
        # self.execution_time = 20.
        self.collect_demo_flag = False # True
        self.demo = []
        self.demo_idx = 0
        self.demo_dir = os.path.join(ROOT_DIR, 'multi_arm_assembly/logs/demos_araas/')
        os.makedirs(self.demo_dir, exist_ok=True)
        self.key_pose_list = []

    def set_up_spaces(self):
        action_bound = 1
        # self.action_dim = 3 * self.init_cfg.robot_count  # Just translation for now
        self.action_dim = 3 * self.init_cfg.num_robots + (3 * int(self.init_cfg.allow_hole_rotation)) + (
                3 * int(self.init_cfg.allow_peg_rotation)) + 1 * self.init_cfg.num_robots
        action_high = np.array([action_bound] * self.action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        obs_bound = 10
        # obs_dim = 9 * self.init_cfg.robot_count  # Joint positions and velocities
        obs_dim = (9 + 6 * (
                    self.init_cfg.USE_FT_SENSOR) + 1 * self.init_cfg.GRIPPER_STATUS_OBS * self.init_cfg.allow_gripper_status
                                 + 6 * self.init_cfg.VELOCITY_OBS) * self.init_cfg.num_robots
        obs_high = np.array([obs_bound] * obs_dim)
        if self.init_cfg.USE_CAMERA:
            self.observation_space = spaces.Dict({"policy": spaces.Box(-obs_high, obs_high), 
            "image": gym.spaces.box.Box(0, 1, (self.init_cfg.IMAGE_WIDTH, self.init_cfg.IMAGE_HEIGHT, 1))})
        else:
            self.observation_space = spaces.Dict({"policy": spaces.Box(-obs_high, obs_high)})

    def close(self):
        print("closing")
        pyaraas.stop()

    def calibrate_force(self, rstate: RobotState):
        print("Calibrating force")
        new_rstate: RobotState = rstate.copy()
        if new_rstate.calibration_sample < new_rstate.force_history.shape[0]:
            if (self.enable_hardware):
                reading = new_rstate.force_sensor.sample()
                # log.info("[Force Calibration Reading]", format(force_sample_to_str(reading)))

                force = np.array([reading["force"].x, reading["force"].y, reading["force"].z])
                torque = np.array([reading["moment"].x, reading["moment"].y, reading["moment"].z])
                log_list("calibrate_force_reading", [force])
                log_list("calibrate_torque_reading", [torque])

            else:
                force = np.array([0, 0, 0])
                torque = np.array([0, 0, 0])

            init_world_force = new_rstate.tool_pose.rotm @ force
            init_world_torque = new_rstate.tool_pose.rotm @ torque
            new_rstate.force_history[new_rstate.calibration_sample, :3] = init_world_force
            new_rstate.force_history[new_rstate.calibration_sample, 3:] = init_world_torque
            new_rstate.calibration_sample += 1
        else:
            new_rstate.init_world_force = np.mean(new_rstate.force_history[:, :3], axis=0)
            new_rstate.init_world_torque = np.mean(new_rstate.force_history[:, 3:], axis=0)
            new_rstate.force_history[:, :3] = new_rstate.force_history[:, :3] - new_rstate.init_world_force
            new_rstate.force_history[:, 3:] = new_rstate.force_history[:, 3:] - new_rstate.init_world_torque
            self.forces_zeroed = True
        return new_rstate

    def plan_to_joint_angles_v0(self, robot, start_qs, target_q, step_through=False, ee_collisions=True):
        print("Start q: " + str(start_qs))
        print("Target q: " + str(target_q))
        robot_index = self.robot_list.index(robot)
        trajectory = solve_motion_plan(start_qs, target_q, target_robot=robot_index, step_through=step_through,
                                       ee_collisions=ee_collisions)
        # print('trajectory, start_qs, target_q: ', trajectory, start_qs, target_q, robot_index, step_through, ee_collisions)
        if not pbu.all_close(list(trajectory[0]), list(start_qs[robot_index])):
            trajectory = [start_qs[robot_index]] + trajectory

        if not pbu.all_close(list(trajectory[-1]), list(target_q)):
            trajectory = trajectory + [target_q]

        speed_factor = 0.1 if self.teleport else 1.0
        if (len(trajectory) > 2):
            try:
                time.sleep(0.5)
                pyaraas.run(pyaraas.Task(self.path_planner.run_trajectory, robot, trajectory, speed_factor))
            except Exception:
                traceback.print_exc()
                import sys
                sys.exit()

    def plan_to_joint_angles(self, robot, start_qs, target_q, **kwargs):
        print("Start q: "+str(start_qs))
        print("Target q: "+str(target_q))
        try:
            time.sleep(0.5)
            speed_factor = 0.1 if self.teleport else 1.0
            pyaraas.run(pyaraas.Task(self.path_planner.plan_and_execute_raw_joint_trajectory, robot, target_q, speed_factor))
        except Exception:
            traceback.print_exc()
            import sys
            sys.exit()

    def current_robot_qs(self):
        current_robot_qs_list = [self.r0.get_joint_angles(joint_space=self.enable_hardware),
                self.r1.get_joint_angles(joint_space=self.enable_hardware)]
        if WORKCELL_NAME in ["Assembly"]:
            current_robot_qs_list.append(self.r2.get_joint_angles(joint_space=self.enable_hardware))
        return current_robot_qs_list

    def plan_to_flange_transform(self, robot, world_T_flange, ee_collisions=True, step_through=False, include_zivid=False, execution_time=10):
        print("Planning to flange transform : " + str(world_T_flange))
        current_qs = self.current_robot_qs()  # joint angles / position
        target_poses = [None, None, None]
        ridx = self.robot_list.index(robot)
        # target_pose = APose.from_transform(world_T_flange).to_list()
        if self.movement_type == 'joint':
            target_poses[ridx] = APose.from_transform(world_T_flange).to_list()
            target_q = solve_ik(current_qs, target_poses, ee_collisions=ee_collisions, include_zivid=include_zivid)[ridx]
            if (target_q is None):
                solve_ik(current_qs, target_poses, ee_collisions=ee_collisions, step_through=True, include_zivid=include_zivid)[ridx]
                assert False
            self.plan_to_joint_angles(robot, current_qs, target_q, ee_collisions=ee_collisions, step_through=step_through)
        elif self.movement_type == 'cartesian':
            # print('world_T_flange: ', world_T_flange)
            pyaraas.run(pyaraas.Task(robot.move_cartesian, world_T_flange, execution_time))
        else:
            raise NotImplemented(f'{self.movement_type} is not implemented')
        # try:
        #     time.sleep(0.5)
        #     # pyatk.Transform(px, py, pz, qx, qy, qz, qw)
        #     target_pose = pyatk.Transform(target_pose[0][0], target_pose[0][1], target_pose[0][2], target_pose[1][0], target_pose[1][1], target_pose[1][2], target_pose[1][3])
        #     pyaraas.run(pyaraas.Task(self.path_planner.plan_and_execute_joint_trajectory, robot, target_pose))
        # except Exception:
        #     traceback.print_exc()
        #     import sys
        #     sys.exit()

    def render(self, **kwargs):
        pass

    def update_buffer(self, rstate: RobotState, dt: float, update_vel=True,
                      flange_transform: Transform = None) -> RobotState:
        new_rstate = rstate.copy()
        if (flange_transform is None):
            # In sim
            temp_pose = APose.from_transform(new_rstate.robot.get_flange_transform())
        else:
            # In real
            temp_pose = APose.from_transform(flange_transform)

        if update_vel:
            vel = np.zeros(6)
            vel[:3] = (temp_pose.trans - new_rstate.tool_pose.trans) / dt
            Rd = temp_pose.rotm @ new_rstate.tool_pose.rotm.T
            axis_angle_vel = R.from_matrix(Rd).as_rotvec() / dt
            vel[3:] = axis_angle_vel
            new_rstate.tool_vel = vel
        new_rstate.tool_pose = temp_pose
        return new_rstate

    def update_force_real(self, rstate: RobotState):
        if not self.forces_zeroed:
            new_rstate = self.calibrate_force(rstate)
            return new_rstate

        new_rstate = rstate.copy()
        if (self.enable_hardware):
            reading = new_rstate.force_sensor.sample()
            # log.info("[Force Reading]:{}".format(force_sample_to_str(reading)))
            force = np.array([reading["force"].x, reading["force"].y, reading["force"].z])
            torque = np.array([reading["moment"].x, reading["moment"].y, reading["moment"].z])
        else:
            force = np.array([0, 0, 0])
            torque = np.array([0, 0, 0])

        world_force = new_rstate.tool_pose.rotm @ force - new_rstate.init_world_force
        world_torque = new_rstate.tool_pose.rotm @ torque - new_rstate.init_world_torque

        H = new_rstate.force_history.shape[0]
        new_rstate.force_history[:H - 1, :] = new_rstate.force_history[1:H, :]
        new_rstate.force_history[-1, :3] = world_force
        new_rstate.force_history[-1, 3:] = world_torque

        # average within the sliding window to de-noise the FT reading
        new_rstate.world_force[:3] = np.mean(new_rstate.force_history[:, :3], axis=0)
        new_rstate.world_force[3:] = np.mean(new_rstate.force_history[:, 3:], axis=0)
        log_list("world_force", [new_rstate.world_force])
        return new_rstate

    def update_control(self,
                       control_cmd: List[float],
                       rstate: RobotState,
                       gains: AdmittanceGains):

        # Admittance control
        # Clip the cmd to be safe
        # control_cmd = self.apply_control_limit(control_cmd)

        d_pos = control_cmd[:3]  # in m
        d_quat = control_cmd[3:7]  # in quat
        d_rot = R.from_quat(d_quat).as_matrix()
        d_pos_vel = control_cmd[7:10]
        d_ori_vel = control_cmd[10:13]

        pos_error = rstate.tool_pose.trans[:3] - d_pos
        ori_error_rot = rstate.tool_pose.rotm @ d_rot.T
        # from rot(9 dim) to error (3 dim), need to use axis-angle representation
        ori_error = R.from_matrix(ori_error_rot).as_rotvec()
        error = np.hstack([pos_error, ori_error])

        pos_error_dot = rstate.tool_vel[:3] - d_pos_vel
        ori_error_dot = rstate.tool_vel[3:] - d_ori_vel
        error_dot = np.hstack([pos_error_dot, ori_error_dot])

        # Feedback law: M*acc = F - kp e - kd e_dot
        world_force_clip = np.array(rstate.world_force)
        # clamp the force and torque while maintaining the "direction"
        force_value = Vector(world_force_clip[0], world_force_clip[1], world_force_clip[2]).magnitude()
        if force_value > self.force_limit:
            world_force_clip[:3] = world_force_clip[:3] * self.force_limit / force_value
        torque_value = Vector(world_force_clip[3], world_force_clip[4], world_force_clip[5]).magnitude()
        if torque_value > self.torque_limit:
            world_force_clip[3:] = world_force_clip[3:] * self.torque_limit / torque_value
        if not self.enable_compliance or not self.forces_zeroed:
            # turn off compliance
            world_force_clip = np.zeros(6)

        log_list("error", [error])
        log_list("error_dot", [error_dot])
        log_list("world_force_clip", [world_force_clip])
        RHS = 1 * world_force_clip - np.multiply(gains.kp, error) - np.multiply(gains.kd, error_dot)
        log_list("RHS", [RHS])
        acc = np.divide(RHS, gains.M)
        world_vel_cmd = acc * self.time_step  # proper way is tyo add the prev velocity but that seems to make robot drift which means tune the gains higher to make positioning force stronger

        log_list("world_vel_cmd", [world_vel_cmd])
        world_vel_cmd[0:3] = world_vel_cmd[0:3] * UNIT_OFFSET  # mm to m
        # clamp the velocity and angular velocity command while maintaining "direction"
        vel_value = Vector(world_vel_cmd[0], world_vel_cmd[1], world_vel_cmd[2]).magnitude()
        if vel_value > self.max_v:
            world_vel_cmd[:3] = world_vel_cmd[:3] * self.max_v / vel_value
        w_value = Vector(world_vel_cmd[3], world_vel_cmd[4], world_vel_cmd[5]).magnitude()
        if w_value > self.max_w:
            world_vel_cmd[3:] = world_vel_cmd[3:] * self.max_w / w_value
        cmd = (np.block(
            [[rstate.tool_pose.rotm, np.zeros([3, 3])],
             [np.zeros([3, 3]), rstate.tool_pose.rotm]]).T @ world_vel_cmd).tolist()
        return cmd

    def get_RL_state(self):
        rstate_vecs = []

        for ridx, active in enumerate([self.init_cfg.holding_peg, self.init_cfg.holding_hole]):
            if (active):
                rstate = [self.r0_state, self.r1_state][ridx]
                gains = [self.r0_gains, self.r1_gains][ridx]
                p_pose = copy.deepcopy(rstate.tool_pose)  # APose
                if self.init_cfg.relative_pose_obs:
                    # print('absolute p_pose: ', p_pose)
                    world_T_tool = p_pose.to_list()
                    if self.init_cfg.TASK_NAME in ['AssemblyInsert-v0', 'AssemblyScriptedInsert-v0', ]:
                        des_world_T_tool = pbu.multiply(pbu.multiply(self.init_cfg.WORLD_T_HOLE_START, self.init_cfg.PEG_GOAL), self.init_cfg.PEG_T_TOOL)
                    elif self.init_cfg.TASK_NAME in ['MoveToFixture', 'MoveToFixtureHorizontal',
                                                            'AssemblyScriptedPlace-v0',
                                                            'AssemblyScriptedPlaceHorizontal-v0',
                                                            'AssemblyScriptedGrasp-v0',
                                                            'AssemblyScriptedGraspHorizontal-v0',
                                                            'AssemblyScriptedRegrasp-v0']:
                        des_world_T_tool = pbu.multiply(pbu.multiply(self.init_cfg.WORLD_T_FIXTURE_START, self.init_cfg.PEG_GOAL), self.init_cfg.PEG_T_TOOL)
                    else:
                        raise NotImplementedError(
                            '{} _get_observations relative_pose_obs is not Implemented'.format(
                                self.init_cfg.TASK_NAME))
                    p_pose = pbu.multiply(des_world_T_tool, pbu.invert(world_T_tool))
                    p_pose = APose(*p_pose)
                    # print('relative p_pose: ', p_pose)
                # print('******************', self.init_cfg.TASK_NAME, p_pose)
                world_force = rstate.world_force.copy()
                world_force = np.clip(world_force, -gains.force_limit, gains.force_limit)
                world_force = np.divide(world_force, gains.force_limit)
                rstate_vecs += [p_pose.trans, p_pose.rotm[:3, 0], p_pose.rotm[:3, 1]]
                if self.init_cfg.USE_FT_SENSOR:
                    rstate_vecs += [world_force]
                    print('world_force: ', world_force)

        policy_obs = np.concatenate(rstate_vecs)
        print("Step: " + str(self.steps), 'USE_CAMERA: ', self.init_cfg.USE_CAMERA)
        obs = {"policy": policy_obs}
        if False and self.init_cfg.USE_CAMERA:
            depth_vis_img_path = os.path.join(self.img_dir, '{:04d}_depth_vis.png'.format(self.steps))
            rgb_vis_img_path = os.path.join(self.img_dir, '{:04d}_rgb_vis.png'.format(self.steps))
            # obs['image'] = np.zeros((128, 128, 1))
            for capture_i in range(self.zivid_capture_times):
                img_path = os.path.join(self.img_dir, '{:04d}_{:02d}.png'.format(self.steps, capture_i))
                print('img_path: ', img_path)
                pyaraas.run(pyaraas.Task(self.c0.capture, 'depth', int(self.init_cfg.IMAGE_WIDTH_RAW), img_path, 1100))
            print('Taking picture from {} (should be {})'.format(self.c0.get_transform(), WORLD_T_ZIVID_HOME))
            print('get_RL_state cx: {}, cy: {}, fx: {}, fy: {}, fov: {}, sensor_dim: {}, parameter_json: {}'
                .format(self.c0.cx, self.c0.cy, self.c0.fx, self.c0.fy, self.c0.fov, self.c0.sensor_dim, self.c0.parameter_json,))
            img4pose_est = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            # print('img: ', img, img.shape, img.max(), img.min())  #  (128, 128, 1) 255 0
            if True or self.init_cfg.SAVE_IMG:
                colored_depth = visualize_depth_image(img4pose_est[:, :, -1:])
                cv2.imwrite(depth_vis_img_path, colored_depth)
                cv2.imwrite(rgb_vis_img_path, img4pose_est[:, :, :3])
            obs['image'] = cv2.resize(img4pose_est, (int(self.init_cfg.IMAGE_WIDTH), int(self.init_cfg.IMAGE_HEIGHT)))[:, :, -1:] / 65535. * 5000
            obs['image'] = obs['image'].astype(np.int16)

        log_list("araas_obs", [policy_obs])
        return obs

    def get_updated_gains(self, gains, action):
        # Get gains from action
        kp = (self.kp_high + self.kp_low) / 2 + np.multiply(action[6:12], (self.kp_high - self.kp_low) / 2)
        kd = 4 * np.sqrt(np.multiply(gains.M, kp))  # 4 to make thw system overdamped

        new_gains = AdmittanceGains(
            gains.M, kp, kd, gains.force_limit
        )
        return new_gains

    def get_control_command(self, rstate, action, allow_rotation=False):
        # Get delta_motion from action
        delta = np.zeros(6)
        delta[:3] = action[
                    :3] * self.translation_limit  # / (np.linalg.norm(action[:3], ord=2) + 1e-3) * self.translation_limit
        if (allow_rotation):
            delta[3:6] = action[3:6] * self.rotation_limit  # / (np.linalg.norm(action[3:6], ord=2) + 1e-3) * self.rotation_limit

        delta_rotm = R.from_euler('XYZ', delta[3:6]).as_matrix()
        log_list("araas_tool_quat", [R.from_matrix(rstate.tool_pose.rotm).as_quat()])
        control_cmd = np.zeros(13)
        control_cmd[:3] = rstate.tool_pose.trans[:3] + delta[:3]
        if self.debug_action:
            print(4, 'step: {}, rstate.tool_pose.trans[:3]: {}, delta[:3]: {}, delta[3:]: {}'.format(self.steps, rstate.tool_pose.trans[:3], delta[:3], delta[3:]))
        control_cmd[3:7] = R.from_matrix(rstate.tool_pose.rotm @ delta_rotm).as_quat()
        log_list("araas_cmd", [control_cmd])
        return control_cmd

    def get_image_for_pose_estimation(self):
        obs = dict()
        depth_vis_img_path = os.path.join(self.img_dir, '{:04d}_pose_depth_vis.png'.format(self.steps))
        rgb_vis_img_path = os.path.join(self.img_dir, '{:04d}_pose_rgb_vis.png'.format(self.steps))
        # obs['image'] = np.zeros((128, 128, 1))
        for capture_i in range(self.zivid_capture_times):
            img_path = os.path.join(self.img_dir, '{:04d}_{:02d}_pose.png'.format(self.steps, capture_i))
            print('Image saved to {}'.format(img_path))
            pyaraas.run(pyaraas.Task(self.c0.capture, 'depth', int(self.init_cfg.IMAGE_WIDTH_RAW), img_path, 1100))

        img4pose_est = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # print('img: ', img, img.shape, img.max(), img.min())  #  (128, 128, 1) 255 0
        if True or self.init_cfg.SAVE_IMG:
            colored_depth = visualize_depth_image(img4pose_est[:, :, -1:])
            cv2.imwrite(depth_vis_img_path, colored_depth)
            cv2.imwrite(rgb_vis_img_path, img4pose_est[:, :, :3])
        obs['image'] = cv2.resize(img4pose_est, (int(self.init_cfg.IMAGE_WIDTH), int(self.init_cfg.IMAGE_HEIGHT)))[
                       :, :, -1:] / 65535. * 5000
        obs['image_pose_estimation'] = img4pose_est
        return obs

    def get_high_level_policy_state(self, last_selected_primitive='', init_cfg=None, use_image=False):
        # self.demo = []
        # self.demo_idx += 1
        # self.current_datetime_str = datetime.now().strftime('%Y%m%d%H%M%S')

        self.steps = 0
        # Initialize gains
        self.r0_gains = AdmittanceGains()
        self.r1_gains = AdmittanceGains()

        # Create some buffer for peg pose, link_6 rot
        self.r0_state = RobotState(self.r0, self.g0, self.f0)
        self.r1_state = RobotState(self.r1, self.g1, self.f1)

        # Update the link6 pose from identity to what is in simulation
        self.r0_state = self.update_buffer(self.r0_state, self.dt, update_vel=False)
        self.r1_state = self.update_buffer(self.r1_state, self.dt, update_vel=False)

        obs = self.get_RL_state()

        # if last_selected_primitive in ['AssemblyScriptedInsert', 'AssemblyInsert']:
        #     # convert ee_pos to Hexagon
        #     print('current ee', obs['policy'])
        #     pass
        if use_image:
            depth_vis_img_path = os.path.join(self.img_dir, '{:04d}_depth_vis.png'.format(self.steps))
            rgb_vis_img_path = os.path.join(self.img_dir, '{:04d}_rgb_vis.png'.format(self.steps))
            # obs['image'] = np.zeros((128, 128, 1))
            for capture_i in range(self.zivid_capture_times):
                img_path = os.path.join(self.img_dir, '{:04d}_{:02d}.png'.format(self.steps, capture_i))
                print('img_path: ', img_path)
                pyaraas.run(pyaraas.Task(self.c0.capture, 'depth', int(self.init_cfg.IMAGE_WIDTH_RAW), img_path, 1100))
            print('Taking picture from {} (should be {})'.format(self.c0.get_transform(), WORLD_T_ZIVID_HOME))
            print('get_RL_state cx: {}, cy: {}, fx: {}, fy: {}, fov: {}, sensor_dim: {}, parameter_json: {}'
                .format(self.c0.cx, self.c0.cy, self.c0.fx, self.c0.fy, self.c0.fov, self.c0.sensor_dim, self.c0.parameter_json,))
            img4pose_est = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            # print('img: ', img, img.shape, img.max(), img.min())  #  (128, 128, 1) 255 0
            if True or self.init_cfg.SAVE_IMG:
                colored_depth = visualize_depth_image(img4pose_est[:, :, -1:])
                cv2.imwrite(depth_vis_img_path, colored_depth)
                cv2.imwrite(rgb_vis_img_path, img4pose_est[:, :, :3])
            obs['image'] = cv2.resize(img4pose_est, (int(self.init_cfg.IMAGE_WIDTH), int(self.init_cfg.IMAGE_HEIGHT)))[:, :, -1:] / 65535. * 5000
            obs['image_pose_estimation'] = img4pose_est
        
        # self.demo.append([obs, mode])
        return obs

    def rollback(self, ):
        # if self.init_cfg.TASK_NAME in ['AssemblyInsert']:
        #     pass
        for each_key_pose in self.key_pose_list[::-1]:
            if each_key_pose in ['open gripper']:
                # self.close_gripper()
                pass
            elif each_key_pose in ['close gripper']:
                self.open_gripper()
            else:
                pyaraas.run(pyaraas.Task(self.r0.move_cartesian, each_key_pose, 20))
        self.key_pose_list = []

    def open_gripper(self,):
        pyaraas.run(pyaraas.Task(self.g1.open, 58, 1))

    def close_gripper(self,):
        pyaraas.run(pyaraas.Task(self.g1.open, 0, 1))

    def set_mode(self, mode, init_cfg):
        self.setup = False
        self.mode = mode
        if int(init_cfg.selected_obj_id) in [0, ]:
            self.OPEN_WIDTH = 40
        elif int(init_cfg.selected_obj_id) in [1, 4, ]:
            self.OPEN_WIDTH = 45
        elif int(init_cfg.selected_obj_id) in [5, 8]:
            self.OPEN_WIDTH = 50
        else:  # [2, 3, 6, 7, ]
            self.OPEN_WIDTH = 58

        
        self.demo = []
        self.demo_idx += 1
        self.current_datetime_str = datetime.now().strftime('%Y%m%d%H%M%S')

        self.steps = 0
        if self.collect_demo_flag:
            # Initialize gains
            self.r0_gains = AdmittanceGains()
            self.r1_gains = AdmittanceGains()
            
            # Create some buffer for peg pose, link_6 rot
            self.r0_state = RobotState(self.r0, self.g0, self.f0)
            self.r1_state = RobotState(self.r1, self.g1, self.f1)

            # Update the link6 pose from identity to what is in simulation
            self.r0_state = self.update_buffer(self.r0_state, self.dt, update_vel=False)
            self.r1_state = self.update_buffer(self.r1_state, self.dt, update_vel=False)

            obs = self.get_RL_state()
            self.demo.append([obs, mode])

        # import pdb; pdb.set_trace()
        if init_cfg is None:
            return

        print("Setting mode: " + str(mode))

        # First, we need to make sure the robots are not in collision
        for offset in range(10):
            qs = self.current_robot_qs()
            starting_flanges = [self.r0.get_flange_transform(), self.r1.get_flange_transform()]
            print("Offset: " + str(offset))
            if (check_collision(*qs, include_zivid=(WORKCELL_NAME in ["Assembly"]))):
                if (not init_cfg.holding_peg):
                    pre = starting_flanges[0].multiply(Transform([0, 0, -10, 0, 0, 0]))
                    pyaraas.run(pyaraas.Task(self.r0.move_cartesian, pre, 0.2))
                if (not init_cfg.holding_hole):
                    pre = starting_flanges[1].multiply(Transform([0, 0, -10, 0, 0, 0]))
                    pyaraas.run(pyaraas.Task(self.r1.move_cartesian, pre, 0.2))
            else:
                break
            # pyaraas.Robot.execute_cartesian_trajectory(path)
            # [{'xform': (x,y,z,rx,ry,rz), 'tick':0.01}, {'xform': (x,y,z,rx,ry,rz), 'tick':0.01}, {'xform': (x,y,z,rx,ry,rz), 'tick':0.01}, {'xform': (x,y,z,rx,ry,rz), 'tick':0.01}]

        if (not init_cfg.holding_peg):
            # Move robot 0 out of the way
            pyaraas.run(pyaraas.Task(self.g1.open, self.OPEN_WIDTH, 1))
            self.plan_to_flange_transform(self.r0, WORLD_T_R0_HOME, step_through=False)

        if WORKCELL_NAME in ["Assembly"]:
            print('Moving camera to {} from {}'.format(WORLD_T_ZIVID_HOME, self.c0.get_transform()))
            pyaraas.run(pyaraas.Task(self.c0.move_cartesian, WORLD_T_ZIVID_HOME, 20))
            # pass

            # self.plan_to_flange_transform(self.r2, WORLD_T_R2_HOME, step_through=False, include_zivid=True)
        # if (not init_cfg.holding_hole):
        #     # Move robot 1 out of the way
        #     pyaraas.run(pyaraas.Task(self.g1.open, self.OPEN_WIDTH, 1))
        #     self.plan_to_flange_transform(self.r1, WORLD_T_R1_HOME)

        self.init_cfg = init_cfg
        if self.init_cfg.TASK_NAME in ['AssemblyInsert-v0']:
            self.max_step = 100

        self.set_up_spaces()

        if self.collect_demo_flag:
            obs = self.get_RL_state()
            self.demo.append([obs, mode])

    def scripted_step(self, selected_primitive):
        self.key_pose_list = []
        if self.collect_demo_flag:
            obs = self.get_RL_state()
            self.demo.append([obs, selected_primitive])
        target_poses = [None, None, None]
        if selected_primitive in ['AssemblyScriptedGrasp', 'AssemblyScriptedGraspHorizontal']:
            approach = Transform([0, 0, 100, 0, 0, 0])
            # print((self.init_cfg.holding_peg and self.init_cfg.WORLD_T_TOOL_PICK_PEG is not None), (self.init_cfg.holding_peg), (self.init_cfg.WORLD_T_TOOL_PICK_PEG is not None))
            # True, True, True
            # Robot 1 Grasp
            # if (self.init_cfg.holding_peg and self.init_cfg.WORLD_T_TOOL_PICK_PEG is not None):
            if self.init_cfg.desired_tool_T_world is not None:
                world_T_flange0_grasp = APose(*self.init_cfg.desired_tool_T_world).to_transform()
                print('desired_tool_T_world: {}'.format(self.init_cfg.desired_tool_T_world))
            else:
                world_T_flange0_grasp = APose(*self.init_cfg.WORLD_T_TOOL_PICK_PEG).to_transform()
                print('Scripted WORLD_T_TOOL_PICK_PEG: {}'.format(self.init_cfg.WORLD_T_TOOL_PICK_PEG))
            print('world_T_flange0_grasp: ', APose(*self.init_cfg.WORLD_T_TOOL_PICK_PEG).to_list())
            world_T_flange0_pregrasp = approach.multiply(world_T_flange0_grasp)
            self.key_pose_list.append('open gripper')
            pyaraas.run(pyaraas.Task(self.g1.open, self.OPEN_WIDTH, 1))
            self.key_pose_list.append(world_T_flange0_pregrasp)
            self.plan_to_flange_transform(self.r0, world_T_flange0_pregrasp, execution_time=10)
            self.key_pose_list.append(world_T_flange0_grasp)
            self.plan_to_flange_transform(self.r0, world_T_flange0_grasp, ee_collisions=False, execution_time=10)
            self.key_pose_list.append('close gripper')
            pyaraas.run(pyaraas.Task(self.g1.open, 0, 1))
            # self.plan_to_flange_transform(self.r0, world_T_flange0_pregrasp, ee_collisions=False, execution_time=10)
            target_poses[0] = pbu.multiply(pbu.multiply(WORLD_T_FIXTURE_START, self.init_cfg.PEG_GOAL), self.init_cfg.PEG_T_TOOL)
        elif selected_primitive in ['AssemblyScriptedInsert', 'AssemblyInsert', ]:
            target_poses[0] = pbu.multiply(pbu.multiply(self.init_cfg.WORLD_T_HOLE_START, self.init_cfg.PEG_GOAL),
                                           self.init_cfg.PEG_T_TOOL)
            world_T_flange0_pregrasp = self.insert_approach_1.multiply(APose(*target_poses[0]).to_transform())
            self.key_pose_list.append(world_T_flange0_pregrasp)
            self.plan_to_flange_transform(self.r0, world_T_flange0_pregrasp, execution_time=4)
            approach2 = Transform([0, 0, 52, 0, 0, 0])
            world_T_flange0_pregrasp2 = approach2.multiply(APose(*target_poses[0]).to_transform())
            self.key_pose_list.append(world_T_flange0_pregrasp2)
            self.plan_to_flange_transform(self.r0, world_T_flange0_pregrasp2, execution_time=10)
            # print('\n\n\n\n\n\n\n\n\n\n\n\ntarget_poses[0]: {}, WORLD_T_HOLE_START: {}, self.init_cfg.PEG_GOAL: {}, self.init_cfg.PEG_T_TOOL: {}'.format(target_poses[0], WORLD_T_HOLE_START, self.init_cfg.PEG_GOAL, self.init_cfg.PEG_T_TOOL))
        elif selected_primitive in ['AssemblyMoveToBoard', 'AssemblyMoveToBoardFromFixture', 'AssemblyMoveToBoardFromGrasp']:
            # approach = Transform([0, 0, 280, 0, 0, 0])
            # world_T_flange0_pregrasp = approach.multiply(APose(*self.init_cfg.POSE_AFTER).to_transform())
            # target_poses[0] = pbu.multiply(pbu.multiply(WORLD_T_HOLE_START, self.init_cfg.PEG_GOAL), self.init_cfg.PEG_T_TOOL)
            # self.plan_to_flange_transform(self.r0, world_T_flange0_pregrasp, execution_time=10)
            target_poses[0] = self.init_cfg.POSE_AFTER
        elif selected_primitive in ['AssemblyMoveToFixture', 'AssemblyMoveToFixtureHorizontal']:
            target_poses[0] = pbu.multiply(pbu.multiply(WORLD_T_FIXTURE_START, self.init_cfg.PEG_GOAL), self.init_cfg.PEG_T_TOOL)
        elif selected_primitive in ['AssemblyScriptedPlace', 'AssemblyScriptedPlaceHorizontal']:
            target_poses[0] = pbu.multiply(pbu.multiply(WORLD_T_FIXTURE_START, self.init_cfg.PEG_GOAL), self.init_cfg.PEG_T_TOOL)
        elif selected_primitive in ['AssemblyScriptedRegrasp']:
            grasping_pose = pbu.multiply(self.init_cfg.WORLD_T_PEG_START, self.init_cfg.PEG_T_TOOL)
            print('world_T_flange0_grasp: ', APose(*grasping_pose).to_list())
            if not APose(*grasping_pose).to_transform().is_equal(1., 0.01, self.r0.get_transform()):
                PRE_WORLD_T_PEG_START = ([0.3, 0.2, 0.22], [0.0, 0.0, -0.7071067690849304, 0.7071067690849304])
                target_poses[0] = pbu.multiply(PRE_WORLD_T_PEG_START, self.init_cfg.PEG_T_TOOL)
                target_poses_transform = APose(*target_poses[0]).to_transform()
                self.key_pose_list.append(target_poses_transform)
                pyaraas.run(pyaraas.Task(self.r0.move_cartesian, target_poses_transform, 10))
            self.key_pose_list.append('open gripper')
            pyaraas.run(pyaraas.Task(self.g1.open, self.OPEN_WIDTH, 1))
            target_poses[0] = grasping_pose
            target_poses_transform = APose(*target_poses[0]).to_transform()
            self.key_pose_list.append(target_poses_transform)
            pyaraas.run(pyaraas.Task(self.r0.move_cartesian, target_poses_transform, 10))
            self.key_pose_list.append('close gripper')
            pyaraas.run(pyaraas.Task(self.g1.open, 0, 1))
            target_poses[0] = pbu.multiply(pbu.multiply(WORLD_T_FIXTURE_START, self.init_cfg.PEG_GOAL), self.init_cfg.PEG_T_TOOL)

        if selected_primitive not in ['AssemblyInsert',]:
            if self.movement_type == 'joint':
                solutions = solve_ik(self.current_robot_qs(), target_poses)
                self.plan_to_joint_angles(self.r0, self.current_robot_qs(), solutions[0], step_through=True)
            elif self.movement_type == 'cartesian':
                print('target_pose[0]: ', target_poses[0])
                target_poses_transform = APose(*target_poses[0]).to_transform()
                self.key_pose_list.append(target_poses_transform)
                pyaraas.run(pyaraas.Task(self.r0.move_cartesian, target_poses_transform, 10))
            else:
                raise NotImplemented(f'{self.movement_type} is not implemented')

        if selected_primitive in ['AssemblyScriptedPlace', 'AssemblyScriptedPlaceHorizontal']:
            self.key_pose_list.append('open gripper')
            pyaraas.run(pyaraas.Task(self.g1.open, self.OPEN_WIDTH, 1))
        elif selected_primitive in ['AssemblyScriptedInsert']:
            self.key_pose_list.append('open gripper')
            pyaraas.run(pyaraas.Task(self.g1.open, self.OPEN_WIDTH, 1))
            safe_height_pose = self.after_insert_approach_1.multiply(APose(*target_poses[0]).to_transform())
            self.key_pose_list.append(safe_height_pose)
            pyaraas.run(pyaraas.Task(self.r0.move_cartesian,safe_height_pose, 10))
            # print('pose_after_insertion: ', APose.from_transform(safe_height_pose).to_list())
            # pose_after_insertion = APose(*APose.from_transform(safe_height_pose).to_list()).to_transform()
            pose_after_insertion = APose(*self.init_cfg.POSE_AFTER).to_transform()
            self.key_pose_list.append(pose_after_insertion)
            pyaraas.run(pyaraas.Task(self.r0.move_cartesian, pose_after_insertion, 4))

        if self.collect_demo_flag:
            obs = self.get_RL_state()
            self.demo.append([obs, selected_primitive])

            # demo_path = os.path.join(self.demo_dir, '{:04d}_{}_{}.pkl'.format(self.demo_idx, self.current_datetime_str, selected_primitive, ))
            # with open(demo_path, 'wb') as file:
            #     pickle.dump(self.demo, file)
            #     print('Save demo to {}'.format(demo_path))
        print("Process execution finished.")

    def finegrained_scripted_step(self, selected_primitive, finegrained=True):
        self.key_pose_list = []
        if self.collect_demo_flag:
            obs = self.get_RL_state()
            self.demo.append([obs, selected_primitive])
        target_poses = [None, None, None]
        if selected_primitive in ['AssemblyScriptedGrasp', 'AssemblyScriptedGraspHorizontal']:
            approach = Transform([0, 0, 100, 0, 0, 0])
            # print((self.init_cfg.holding_peg and self.init_cfg.WORLD_T_TOOL_PICK_PEG is not None), (self.init_cfg.holding_peg), (self.init_cfg.WORLD_T_TOOL_PICK_PEG is not None))
            # True, True, True
            # Robot 1 Grasp
            # if (self.init_cfg.holding_peg and self.init_cfg.WORLD_T_TOOL_PICK_PEG is not None):
            if self.init_cfg.desired_tool_T_world is not None:
                world_T_flange0_grasp = APose(*self.init_cfg.desired_tool_T_world).to_transform()
                print('desired_tool_T_world: {}'.format(self.init_cfg.desired_tool_T_world))
            else:
                world_T_flange0_grasp = APose(*self.init_cfg.WORLD_T_TOOL_PICK_PEG).to_transform()
                print('Scripted WORLD_T_TOOL_PICK_PEG: {}'.format(self.init_cfg.WORLD_T_TOOL_PICK_PEG))
            print('world_T_flange0_grasp: ', APose(*self.init_cfg.WORLD_T_TOOL_PICK_PEG).to_list())
            world_T_flange0_pregrasp = approach.multiply(world_T_flange0_grasp)
            self.key_pose_list.append('open gripper')
            if finegrained:
                # pyaraas.run([pyaraas.Task(open_gripper_func, self.g1, self.OPEN_WIDTH, 1), pyaraas.Task(sample_gripper_func, self.g1)])
                pass
            else:
                pyaraas.run(pyaraas.Task(self.g1.open, self.OPEN_WIDTH, 1))
            self.key_pose_list.append(world_T_flange0_pregrasp)
            if finegrained:
                pyaraas.run([pyaraas.Task(move_robot, self.r0, world_T_flange0_pregrasp, 10), pyaraas.Task(sample_robot, self.r0)])
            else:
                self.plan_to_flange_transform(self.r0, world_T_flange0_pregrasp, execution_time=10)
            self.key_pose_list.append(world_T_flange0_grasp)
            if finegrained:
                pyaraas.run([pyaraas.Task(move_robot, self.r0, world_T_flange0_grasp, 10), pyaraas.Task(sample_robot, self.r0)])
            else:
                self.plan_to_flange_transform(self.r0, world_T_flange0_grasp, ee_collisions=False, execution_time=10)
            self.key_pose_list.append('close gripper')
            pyaraas.run(pyaraas.Task(self.g1.open, 0, 1))
            # self.plan_to_flange_transform(self.r0, world_T_flange0_pregrasp, ee_collisions=False, execution_time=10)
            target_poses[0] = pbu.multiply(pbu.multiply(WORLD_T_FIXTURE_START, self.init_cfg.PEG_GOAL), self.init_cfg.PEG_T_TOOL)
        elif selected_primitive in ['AssemblyScriptedInsert', 'AssemblyInsert', ]:
            target_poses[0] = pbu.multiply(pbu.multiply(self.init_cfg.WORLD_T_HOLE_START, self.init_cfg.PEG_GOAL),
                                           self.init_cfg.PEG_T_TOOL)
            world_T_flange0_pregrasp = self.insert_approach_1.multiply(APose(*target_poses[0]).to_transform())
            self.key_pose_list.append(world_T_flange0_pregrasp)
            if finegrained:
                pyaraas.run([pyaraas.Task(move_robot, self.r0, world_T_flange0_pregrasp, 10), pyaraas.Task(sample_robot, self.r0)])
            else:
                self.plan_to_flange_transform(self.r0, world_T_flange0_pregrasp, execution_time=10)
            approach2 = Transform([0, 0, 52, 0, 0, 0])
            world_T_flange0_pregrasp2 = approach2.multiply(APose(*target_poses[0]).to_transform())
            self.key_pose_list.append(world_T_flange0_pregrasp2)
            if finegrained:
                pyaraas.run([pyaraas.Task(move_robot, self.r0, world_T_flange0_pregrasp2, 10), pyaraas.Task(sample_robot, self.r0)])
            else:
                self.plan_to_flange_transform(self.r0, world_T_flange0_pregrasp2, execution_time=10)
            # print('\n\n\n\n\n\n\n\n\n\n\n\ntarget_poses[0]: {}, WORLD_T_HOLE_START: {}, self.init_cfg.PEG_GOAL: {}, self.init_cfg.PEG_T_TOOL: {}'.format(target_poses[0], WORLD_T_HOLE_START, self.init_cfg.PEG_GOAL, self.init_cfg.PEG_T_TOOL))
        elif selected_primitive in ['AssemblyMoveToBoard', 'AssemblyMoveToBoardFromFixture', 'AssemblyMoveToBoardFromGrasp']:
            # approach = Transform([0, 0, 280, 0, 0, 0])
            # world_T_flange0_pregrasp = approach.multiply(APose(*self.init_cfg.POSE_AFTER).to_transform())
            # target_poses[0] = pbu.multiply(pbu.multiply(WORLD_T_HOLE_START, self.init_cfg.PEG_GOAL), self.init_cfg.PEG_T_TOOL)
            # self.plan_to_flange_transform(self.r0, world_T_flange0_pregrasp, execution_time=10)
            target_poses[0] = self.init_cfg.POSE_AFTER
        elif selected_primitive in ['AssemblyMoveToFixture', 'AssemblyMoveToFixtureHorizontal']:
            target_poses[0] = pbu.multiply(pbu.multiply(WORLD_T_FIXTURE_START, self.init_cfg.PEG_GOAL), self.init_cfg.PEG_T_TOOL)
        elif selected_primitive in ['AssemblyScriptedPlace', 'AssemblyScriptedPlaceHorizontal']:
            target_poses[0] = pbu.multiply(pbu.multiply(WORLD_T_FIXTURE_START, self.init_cfg.PEG_GOAL), self.init_cfg.PEG_T_TOOL)
        elif selected_primitive in ['AssemblyScriptedRegrasp']:
            grasping_pose = pbu.multiply(self.init_cfg.WORLD_T_PEG_START, self.init_cfg.PEG_T_TOOL)
            print('world_T_flange0_grasp: ', APose(*grasping_pose).to_list())
            if not APose(*grasping_pose).to_transform().is_equal(1., 0.01, self.r0.get_transform()):
                PRE_WORLD_T_PEG_START = ([0.3, 0.2, 0.22], [0.0, 0.0, -0.7071067690849304, 0.7071067690849304])
                target_poses[0] = pbu.multiply(PRE_WORLD_T_PEG_START, self.init_cfg.PEG_T_TOOL)
                target_poses_transform = APose(*target_poses[0]).to_transform()
                self.key_pose_list.append(target_poses_transform)
                if finegrained:
                    pyaraas.run([pyaraas.Task(move_robot, self.r0, target_poses_transform, 10),
                                 pyaraas.Task(sample_robot, self.r0)])
                else:
                    pyaraas.run(pyaraas.Task(self.r0.move_cartesian, target_poses_transform, 10))
            self.key_pose_list.append('open gripper')
            pyaraas.run(pyaraas.Task(self.g1.open, self.OPEN_WIDTH, 1))
            target_poses[0] = grasping_pose
            target_poses_transform = APose(*target_poses[0]).to_transform()
            self.key_pose_list.append(target_poses_transform)
            if finegrained:
                pyaraas.run([pyaraas.Task(move_robot, self.r0, target_poses_transform, 10),
                             pyaraas.Task(sample_robot, self.r0)])
            else:
                pyaraas.run(pyaraas.Task(self.r0.move_cartesian, target_poses_transform, 10))
            self.key_pose_list.append('close gripper')
            pyaraas.run(pyaraas.Task(self.g1.open, 0, 1))
            target_poses[0] = pbu.multiply(pbu.multiply(WORLD_T_FIXTURE_START, self.init_cfg.PEG_GOAL), self.init_cfg.PEG_T_TOOL)

        if selected_primitive not in ['AssemblyInsert',]:
            if self.movement_type == 'joint':
                solutions = solve_ik(self.current_robot_qs(), target_poses)
                self.plan_to_joint_angles(self.r0, self.current_robot_qs(), solutions[0], step_through=True)
            elif self.movement_type == 'cartesian':
                print('target_pose[0]: ', target_poses[0])
                target_poses_transform = APose(*target_poses[0]).to_transform()
                self.key_pose_list.append(target_poses_transform)
                if finegrained:
                    pyaraas.run([pyaraas.Task(move_robot, self.r0, target_poses_transform, 10),
                                 pyaraas.Task(sample_robot, self.r0)])
                else:
                    pyaraas.run(pyaraas.Task(self.r0.move_cartesian, target_poses_transform, 10))
            else:
                raise NotImplemented(f'{self.movement_type} is not implemented')

        if selected_primitive in ['AssemblyScriptedPlace', 'AssemblyScriptedPlaceHorizontal']:
            self.key_pose_list.append('open gripper')
            pyaraas.run(pyaraas.Task(self.g1.open, self.OPEN_WIDTH, 1))
        elif selected_primitive in ['AssemblyScriptedInsert']:
            self.key_pose_list.append('open gripper')
            pyaraas.run(pyaraas.Task(self.g1.open, self.OPEN_WIDTH, 1))
            safe_height_pose = self.after_insert_approach_1.multiply(APose(*target_poses[0]).to_transform())
            self.key_pose_list.append(safe_height_pose)
            if finegrained:
                pyaraas.run([pyaraas.Task(move_robot, self.r0, safe_height_pose, 10),
                             pyaraas.Task(sample_robot, self.r0)])
            else:
                pyaraas.run(pyaraas.Task(self.r0.move_cartesian,safe_height_pose, 10))
            # print('pose_after_insertion: ', APose.from_transform(safe_height_pose).to_list())
            # pose_after_insertion = APose(*APose.from_transform(safe_height_pose).to_list()).to_transform()
            pose_after_insertion = APose(*self.init_cfg.POSE_AFTER).to_transform()
            self.key_pose_list.append(pose_after_insertion)
            if finegrained:
                demo = pyaraas.run([pyaraas.Task(move_robot, self.r0, pose_after_insertion, 10),
                             pyaraas.Task(sample_robot, self.r0)])
            else:
                pyaraas.run(pyaraas.Task(self.r0.move_cartesian, pose_after_insertion, 10))

        if self.collect_demo_flag:
            obs = self.get_RL_state()
            self.demo.append([obs, selected_primitive])

            # demo_path = os.path.join(self.demo_dir, '{:04d}_{}_{}.pkl'.format(self.demo_idx, self.current_datetime_str, selected_primitive, ))
            # with open(demo_path, 'wb') as file:
            #     pickle.dump(self.demo, file)
            #     print('Save demo to {}'.format(demo_path))
        print("Process execution finished.")

    def step(self, action):

        log_list("araas_action", [action])
        # print('action, action.shape: ', action, action.shape, self.init_cfg.holding_peg, self.init_cfg.holding_hole)
        self.steps += 1         

        # Assuming actions are always split down the middle
        # r0_action_size = 3 * (int(self.init_cfg.holding_peg) + int(self.init_cfg.allow_peg_rotation))
        # r1_action_size = 3 * (int(self.init_cfg.holding_hole) + int(self.init_cfg.allow_hole_rotation))
        r0_action_size = 3 * self.init_cfg.num_robots + (3 * int(self.init_cfg.allow_hole_rotation)) + (
                3 * int(self.init_cfg.allow_peg_rotation)) + 1 * self.init_cfg.num_robots
        r1_action_size = 3 * self.init_cfg.num_robots + (3 * int(self.init_cfg.allow_hole_rotation)) + (
                3 * int(self.init_cfg.allow_peg_rotation)) + 1 * self.init_cfg.num_robots
        r0_action = action[:r0_action_size]
        r1_action = action[r0_action_size:(r0_action_size + r1_action_size)]
        
        # for debug
        # r0_action = np.zeros_like(r0_action)
        # r0_action[2] = -1 * 1000
        # print('r0_action: ', r0_action, 'r1_action: ', r1_action)
        if (self.init_cfg.holding_peg):
            r0_control_command = self.get_control_command(self.r0_state, r0_action,
                                                          allow_rotation=self.init_cfg.allow_peg_rotation)

        if (self.init_cfg.holding_hole):
            r1_control_command = self.get_control_command(self.r1_state, r1_action,
                                                          allow_rotation=self.init_cfg.allow_hole_rotation)

        # Real velocity controller
        init_time = time.time()
        while ((time.time() - init_time) < self.max_time_per_step):
            if (self.init_cfg.holding_peg):
                r0_vel_command = self.update_control(r0_control_command, self.r0_state, self.r0_gains)
                _, r0_flange_transform = self.r0_state.robot.set_cartesian_velocity(
                    Vector(r0_vel_command[0], r0_vel_command[1], r0_vel_command[2]),
                    Vector(r0_vel_command[3], r0_vel_command[4], r0_vel_command[5]),
                    self.time_step, self.force_limit, mass=1)

            if (self.init_cfg.holding_hole):
                r1_vel_command = self.update_control(r1_control_command, self.r1_state, self.r1_gains)
                _, r1_flange_transform = self.r1_state.robot.set_cartesian_velocity(
                    Vector(r1_vel_command[0], r1_vel_command[1], r1_vel_command[2]),
                    Vector(r1_vel_command[3], r1_vel_command[4], r1_vel_command[5]),
                    self.time_step, self.force_limit, mass=1)

            # Make sure we don't control too quickly
            time.sleep(self.time_step)

            if (self.init_cfg.holding_peg):
                self.r0_state = self.update_buffer(self.r0_state, self.time_step, flange_transform=r0_flange_transform)
                self.r0_state = self.update_force_real(self.r0_state)

            if (self.init_cfg.holding_hole):
                self.r1_state = self.update_buffer(self.r1_state, self.time_step, flange_transform=r1_flange_transform)
                self.r1_state = self.update_force_real(self.r1_state)

        state = self.get_RL_state()
        done = self.steps > self.max_step

        des_world_T_tool = pbu.multiply(pbu.multiply(self.init_cfg.WORLD_T_HOLE_START, self.init_cfg.PEG_GOAL), self.init_cfg.PEG_T_TOOL)
        robot_base_frame_quat = self.r0_base_frame_apose.to_list()
        tool_pose = self.r0_state.tool_pose.to_list()   # world frame
        world_T_tool = tool_pose  # pbu.multiply(robot_base_frame_quat, tool_pose)
        peg_distance = pbu.multiply(des_world_T_tool, pbu.invert(world_T_tool))
        # print('peg_distance: ', peg_distance)
        diff_pos_b = peg_distance[0]
        # diff_rot_b = axis_angle_from_quat(xyzw_to_wxyz(peg_distance[3:]))
        diff_rot_b = R.from_quat(peg_distance[1]).as_rotvec()
        diff_pose = np.concatenate([diff_pos_b, diff_rot_b], axis=0)
        distance = np.linalg.norm(diff_pose, axis=0)
        distance_pos = np.linalg.norm(diff_pos_b, axis=0)
        distance_rot = np.linalg.norm(diff_rot_b, axis=0)
        # print('des_world_T_tool: {}, world_T_tool: {}, peg_distance: {}, distance: {}'.format(des_world_T_tool,
        #                                                                                        world_T_tool,
        #                                                                                        peg_distance, distance))
        print('distance: {}, distance_pos: {}, distance_rot: {}'.format(distance, distance_pos, distance_rot))
        print('diff_pos_b: {}'.format(diff_pos_b))
        # distance: 0.04193487980548752, distance_pos: 0.028616156992033476, distance_rot: 0.030653706192041087 
        if distance < 0.035 or distance_pos < 0.022 or abs(diff_pos_b[2]) < 0.018 :
            done = True

        if self.collect_demo_flag:
            self.demo.append([state, self.mode])
            # demo_path = os.path.join(self.demo_dir, '{:04d}_{}_{}.pkl'.format(self.demo_idx, self.current_datetime_str, self.mode,))
            # with open(demo_path, 'wb') as file:
            #     pickle.dump(self.demo, file)
        if self.debug_action:
            print(1, 'step: {}, r0_control_command: {}'.format(self.steps, r0_control_command))
            print(2, 'step: {}, action: {}'.format(self.steps, action))
            print(3, 'step: {}, state: {}'.format(self.steps, state['policy'][0]))
        
        if done:
            pyaraas.run(pyaraas.Task(self.g1.open, self.OPEN_WIDTH, 1))
            safe_height_pose = self.after_insert_approach_1.multiply(APose(*des_world_T_tool).to_transform())
            pyaraas.run(pyaraas.Task(self.r0.move_cartesian,safe_height_pose, 10))
            # print('RL pose_after_insertion: ', APose.from_transform(safe_height_pose).to_list())
            # pose_after_insertion = APose(*APose.from_transform(safe_height_pose).to_list()).to_transform()
            pose_after_insertion = APose(*self.init_cfg.POSE_AFTER).to_transform()
            pyaraas.run(pyaraas.Task(self.r0.move_cartesian, pose_after_insertion, 4))
        return state, 0, done, {}

    def execute_low_level_action(self, action):

        log_list("araas_action", [action])
        # print('action, action.shape: ', action, action.shape, self.init_cfg.holding_peg, self.init_cfg.holding_hole)
        self.steps += 1

        # Assuming actions are always split down the middle
        # r0_action_size = 3 * (int(self.init_cfg.holding_peg) + int(self.init_cfg.allow_peg_rotation))
        # r1_action_size = 3 * (int(self.init_cfg.holding_hole) + int(self.init_cfg.allow_hole_rotation))
        r0_action_size = 3 * self.init_cfg.num_robots + (3 * int(self.init_cfg.allow_hole_rotation)) + (
                3 * int(self.init_cfg.allow_peg_rotation)) + 1 * self.init_cfg.num_robots
        r1_action_size = 3 * self.init_cfg.num_robots + (3 * int(self.init_cfg.allow_hole_rotation)) + (
                3 * int(self.init_cfg.allow_peg_rotation)) + 1 * self.init_cfg.num_robots
        r0_action = action[:r0_action_size]
        r1_action = action[r0_action_size:(r0_action_size + r1_action_size)]

        # for debug
        # r0_action = np.zeros_like(r0_action)
        # r0_action[2] = -1 * 1000
        # print('r0_action: ', r0_action, 'r1_action: ', r1_action)
        r0_action[:3] = r0_action[:3] /  self.translation_limit
        r1_action[:3] = r1_action[:3] / self.translation_limit
        if self.init_cfg.allow_peg_rotation:
            r0_action[3:6] = r0_action[3:6] / self.rotation_limit
            r1_action[3:6] = r1_action[3:6] / self.rotation_limit
        if (self.init_cfg.holding_peg):
            r0_control_command = self.get_control_command(self.r0_state, r0_action,
                                                          allow_rotation=self.init_cfg.allow_peg_rotation)

        if (self.init_cfg.holding_hole):
            r1_control_command = self.get_control_command(self.r1_state, r1_action,
                                                          allow_rotation=self.init_cfg.allow_hole_rotation)

        # Real velocity controller
        init_time = time.time()
        while ((time.time() - init_time) < self.max_time_per_step):
            if (self.init_cfg.holding_peg):
                r0_vel_command = self.update_control(r0_control_command, self.r0_state, self.r0_gains)
                _, r0_flange_transform = self.r0_state.robot.set_cartesian_velocity(
                    Vector(r0_vel_command[0], r0_vel_command[1], r0_vel_command[2]),
                    Vector(r0_vel_command[3], r0_vel_command[4], r0_vel_command[5]),
                    self.time_step, self.force_limit, mass=1)

            if (self.init_cfg.holding_hole):
                r1_vel_command = self.update_control(r1_control_command, self.r1_state, self.r1_gains)
                _, r1_flange_transform = self.r1_state.robot.set_cartesian_velocity(
                    Vector(r1_vel_command[0], r1_vel_command[1], r1_vel_command[2]),
                    Vector(r1_vel_command[3], r1_vel_command[4], r1_vel_command[5]),
                    self.time_step, self.force_limit, mass=1)

            # Make sure we don't control too quickly
            time.sleep(self.time_step)

            if (self.init_cfg.holding_peg):
                self.r0_state = self.update_buffer(self.r0_state, self.time_step, flange_transform=r0_flange_transform)
                self.r0_state = self.update_force_real(self.r0_state)

            if (self.init_cfg.holding_hole):
                self.r1_state = self.update_buffer(self.r1_state, self.time_step, flange_transform=r1_flange_transform)
                self.r1_state = self.update_force_real(self.r1_state)

        state = self.get_RL_state()
        done = self.steps > self.max_step

        des_world_T_tool = pbu.multiply(pbu.multiply(self.init_cfg.WORLD_T_HOLE_START, self.init_cfg.PEG_GOAL),
                                        self.init_cfg.PEG_T_TOOL)
        robot_base_frame_quat = self.r0_base_frame_apose.to_list()
        tool_pose = self.r0_state.tool_pose.to_list()  # world frame
        world_T_tool = tool_pose  # pbu.multiply(robot_base_frame_quat, tool_pose)
        peg_distance = pbu.multiply(des_world_T_tool, pbu.invert(world_T_tool))
        # print('peg_distance: ', peg_distance)
        diff_pos_b = peg_distance[0]
        # diff_rot_b = axis_angle_from_quat(xyzw_to_wxyz(peg_distance[3:]))
        diff_rot_b = R.from_quat(peg_distance[1]).as_rotvec()
        diff_pose = np.concatenate([diff_pos_b, diff_rot_b], axis=0)
        distance = np.linalg.norm(diff_pose, axis=0)
        distance_pos = np.linalg.norm(diff_pos_b, axis=0)
        distance_rot = np.linalg.norm(diff_rot_b, axis=0)
        # print('des_world_T_tool: {}, world_T_tool: {}, peg_distance: {}, distance: {}'.format(des_world_T_tool,
        #                                                                                        world_T_tool,
        #                                                                                        peg_distance, distance))
        print('distance: {}, distance_pos: {}, distance_rot: {}'.format(distance, distance_pos, distance_rot))
        print('diff_pos_b: {}'.format(diff_pos_b))
        # distance: 0.04193487980548752, distance_pos: 0.028616156992033476, distance_rot: 0.030653706192041087
        if distance < 0.035 or distance_pos < 0.022 or abs(diff_pos_b[2]) < 0.018:
            done = True

        if self.collect_demo_flag:
            self.demo.append([state, self.mode])
            # demo_path = os.path.join(self.demo_dir, '{:04d}_{}_{}.pkl'.format(self.demo_idx, self.current_datetime_str, self.mode,))
            # with open(demo_path, 'wb') as file:
            #     pickle.dump(self.demo, file)
        if self.debug_action:
            print(1, 'step: {}, r0_control_command: {}'.format(self.steps, r0_control_command))
            print(2, 'step: {}, action: {}'.format(self.steps, action))
            print(3, 'step: {}, state: {}'.format(self.steps, state['policy'][0]))

        if done:
            pyaraas.run(pyaraas.Task(self.g1.open, self.OPEN_WIDTH, 1))
            safe_height_pose = self.after_insert_approach_1.multiply(APose(*des_world_T_tool).to_transform())
            pyaraas.run(pyaraas.Task(self.r0.move_cartesian, safe_height_pose, 10))
            # print('RL pose_after_insertion: ', APose.from_transform(safe_height_pose).to_list())
            # pose_after_insertion = APose(*APose.from_transform(safe_height_pose).to_list()).to_transform()
            pose_after_insertion = APose(*self.init_cfg.POSE_AFTER).to_transform()
            pyaraas.run(pyaraas.Task(self.r0.move_cartesian, pose_after_insertion, 4))
        return state, 0, done, {}

    def get_number_of_agents(self):
        return 1

    def release_and_lift_up(self):
        pyaraas.run(pyaraas.Task(self.g0.open, 35, 1))
        self.plan_to_flange_transform(self.r0, Transform([0, 0, 60, 0, 0, 0]).multiply(self.r0.get_flange_transform()))

    def pick_up_and_move_start(self):

        approach = Transform([0, 0, 60, 0, 0, 0])

        # Robot 1 Grasp
        # if (self.init_cfg.holding_peg and self.init_cfg.WORLD_T_TOOL_PICK_PEG is not None):
        #     world_T_flange0_grasp = APose(*self.init_cfg.WORLD_T_TOOL_PICK_PEG).to_transform()
        #     world_T_flange0_pregrasp = approach.multiply(world_T_flange0_grasp)
        #     pyaraas.run(pyaraas.Task(self.g0.open, self.OPEN_WIDTH, 1))
        #     self.plan_to_flange_transform(self.r0, world_T_flange0_pregrasp)
        #     self.plan_to_flange_transform(self.r0, world_T_flange0_grasp, ee_collisions=False)
        #     pyaraas.run(pyaraas.Task(self.g0.open, 0, 1))
        #     self.plan_to_flange_transform(self.r0, world_T_flange0_pregrasp, ee_collisions=False)
        #
        # if (self.init_cfg.holding_hole and self.init_cfg.WORLD_T_TOOL_PICK_HOLE is not None):
        #     world_T_flange1_grasp = APose(*self.init_cfg.WORLD_T_TOOL_PICK_HOLE).to_transform()
        #     world_T_flange1_pregrasp = approach.multiply(world_T_flange1_grasp)
        #     pyaraas.run(pyaraas.Task(self.g1.open, self.OPEN_WIDTH, 1))
        #     self.plan_to_flange_transform(self.r1, world_T_flange1_pregrasp)
        #     self.plan_to_flange_transform(self.r1, world_T_flange1_grasp, ee_collisions=False)
        #     pyaraas.run(pyaraas.Task(self.g1.open, 0, 1))
        #     self.plan_to_flange_transform(self.r1, world_T_flange1_pregrasp, ee_collisions=False)
        
        base_poses = []
        target_poses = [None, None]

        if WORKCELL_NAME in ["Assembly"]:
            target_poses.append(None)

        step_through = False
        if (self.init_cfg.holding_peg):
            base_poses.append(R0_POSE)
            # target_poses[0] = self.init_cfg.WORLD_T_PEG_TOOL_START
            target_poses[0] = self.init_cfg.POSE_AFTER  # pose_after_insertion
        if (self.init_cfg.holding_hole):
            base_poses.append(R1_POSE)
            target_poses[1] = self.init_cfg.WORLD_T_HOLE_TOOL_START

        # if (self.init_cfg.PEG_IK_WORLD_T_TOOL is not None):
        #     target_poses[0] = self.init_cfg.PEG_IK_WORLD_T_TOOL
        #
        # if (self.init_cfg.HOLE_IK_WORLD_T_TOOL is not None):
        #     target_poses[1] = self.init_cfg.HOLE_IK_WORLD_T_TOOL
        
        # print(hasattr(self.init_cfg, 'better_tool_frame'), self.init_cfg.better_tool_frame)
        # breakpoint()
        # if hasattr(self.init_cfg, 'better_tool_frame') and self.init_cfg.better_tool_frame:
        #     target_poses[0] = pbu.multiply(target_poses[0], pbu.invert(([0, 0, 0], [0, 0, 0.7071068, 0.7071068])))
        if self.movement_type == 'joint':
            solutions = solve_ik(self.current_robot_qs(), target_poses)
            if (solutions is None):
                solve_ik(self.current_robot_qs(), target_poses, step_through=True)
                assert False

            if (self.init_cfg.holding_peg):
                # step_through = (self.mode == "bolt_in_strut")
                self.plan_to_joint_angles(self.r0, self.current_robot_qs(), solutions[0], step_through=step_through)

            if (self.init_cfg.holding_hole):
                self.plan_to_joint_angles(self.r1, self.current_robot_qs(), solutions[-1])
        elif self.movement_type == 'cartesian':
            print('[pick_up_and_move_start]: ', target_poses[0])
            if (self.init_cfg.holding_peg):
                pyaraas.run(pyaraas.Task(self.r0.move_cartesian, APose(*target_poses[0]).to_transform(), 10))
            if (self.init_cfg.holding_hole):
                pyaraas.run(pyaraas.Task(self.r1.move_cartesian, APose(*target_poses[1]).to_transform(), 10))
        else:
            raise NotImplemented(f'{self.movement_type} is not implemented')


        # if self.mode in ['AssemblyScriptedInsert'] or self.init_cfg.SUBTASK in ['MoveToBoardFromFixture']:
        #     target_poses[0] = pbu.multiply(pbu.multiply(self.init_cfg.WORLD_T_HOLE_START, self.init_cfg.PEG_GOAL), self.init_cfg.PEG_T_TOOL)
        # elif self.init_cfg.SUBTASK in ['MoveToFixture'] or self.mode in [ 'AssemblyScriptedPlace', 'AssemblyScriptedGrasp', 'AssemblyScriptedRegrasp']:
        #     # target_poses[0] = pbu.multiply(pbu.multiply(self.init_cfg.WORLD_T_FIXTURE_START, self.init_cfg.PEG_GOAL), pbu.multiply(self.init_cfg.PEG_T_TOOL, pbu.invert(([0, 0, 0], [0, 0, 0.7071068, 0.7071068]))))
        #     target_poses[0] = pbu.multiply(pbu.multiply(self.init_cfg.WORLD_T_FIXTURE_START, self.init_cfg.PEG_GOAL), self.init_cfg.PEG_T_TOOL)
        # else:
        #     print(f'{self.mode}, {self.init_cfg.SUBTASK} not implemented.')
        # # if hasattr(self.init_cfg, 'better_tool_frame') and self.init_cfg.better_tool_frame:
        # #     target_poses[0] = pbu.multiply(target_poses[0], pbu.invert(([0, 0, 0], [0, 0, 0.7071068, 0.7071068])))
        # if self.movement_type == 'joint':
        #     solutions = solve_ik(self.current_robot_qs(), target_poses)
        #     self.plan_to_joint_angles(self.r0, self.current_robot_qs(), solutions[0], step_through=step_through)
        # elif self.movement_type == 'cartesian':
        #     print('[pick_up_and_move_start] end: ', target_poses[0])
        #     pyaraas.run(pyaraas.Task(self.r0.move_cartesian, APose(*target_poses[0]).to_transform(), 10))
        # else:
        #     raise NotImplemented(f'{self.movement_type} is not implemented')


        # if(self.init_cfg.holding_peg):
        #     self.plan_to_flange_transform(self.r0, R0_POSE, APose(*self.init_cfg.WORLD_T_PEG_START).to_transform().multiply(APose(*self.init_cfg.PEG_T_TOOL).to_transform()))
        # if(self.init_cfg.holding_hole):
        #     self.plan_to_flange_transform(self.r1, R1_POSE, APose(*self.init_cfg.WORLD_T_HOLE_START).to_transform().multiply(APose(*self.init_cfg.HOLE_T_TOOL).to_transform()))

    def move_to_joint_positions(self, robot, joint_positions):
        current_joint_angles = robot.get_joint_angles(joint_space=self.enable_hardware)
        try:
            time.sleep(0.5)
            pyaraas.run(pyaraas.Task(self.path_planner.run_trajectory, robot, [current_joint_angles, joint_positions]))
        except Exception:
            traceback.print_exc()
            import sys
            sys.exit()

    def hardcode_screw_in_bolt(self, robot, gripper):
        NUM_TURNS = 9 * 4

        current_joint_angles = robot.get_joint_angles()

        for half_turn in range(NUM_TURNS):
            print("Executing half turn: " + str(half_turn))
            # Wrist joint to lower limit
            current_joint_angles[-1] = -np.pi / 4.0
            self.move_to_joint_positions(robot, current_joint_angles)

            # Ungrasp
            pyaraas.run(pyaraas.Task(gripper.open, self.OPEN_WIDTH, 1))

            # Wrist joint to upper limit
            current_joint_angles[-1] = np.pi / 4.0
            self.move_to_joint_positions(robot, current_joint_angles)

            # Grasp
            pyaraas.run(pyaraas.Task(gripper.open, 0, 1))

    def reset(self, seed=None, options=None):
        self.steps = 0
        if (not self.setup):
            self.setup = True

            # Dynamic parameters
            self.tool_friction = self.friction

            self.time_step = self.dt

            self.table = self.w.get_peripheral("rectangularTable-0")
            self.table.set_margin(0)
            self.table.set_collision_model(self.collisions_enabled, self.collisions_enabled)
            
            try:
                self.kit = self.w.get_peripheral("Kit - Stewart Platform v29-0")
                self.kit.set_collision_model(self.collisions_enabled, self.collisions_enabled)
                self.stand = self.w.get_peripheral("stand_for_stewart_platform_assembly_no_motor-0")
                self.stand.set_collision_model(self.collisions_enabled, self.collisions_enabled)
            except Exception as e:
                print(e)

            print('*******************', self.init_cfg.TASK_NAME, self.init_cfg.TASK_NAME in ['AssemblyReset-v0'])
            if (not self.skip_setup) and (self.init_cfg.TASK_NAME in ['AssemblyReset-v0']):
                if (self.mode == "screw_in_bolt"):
                    self.hardcode_screw_in_bolt(self.r1, self.g1)
                    return
                elif (self.mode == "lift_up"):
                    self.release_and_lift_up()
                    return
                else:
                    pyaraas.run(pyaraas.Task(self.g1.open, 58, 1))
                    self.pick_up_and_move_start()

            # Initialize gains
            self.r0_gains = AdmittanceGains()
            self.r1_gains = AdmittanceGains()

            # Create some buffer for peg pose, link_6 rot
            self.r0_state = RobotState(self.r0, self.g0, self.f0)
            self.r1_state = RobotState(self.r1, self.g1, self.f1)

            # Update the link6 pose from identity to what is in simulation
            self.r0_state = self.update_buffer(self.r0_state, self.time_step, update_vel=False)
            self.r1_state = self.update_buffer(self.r1_state, self.time_step, update_vel=False)

            # Save the initial state to use for later
            self.init_r0_state = self.r0_state.copy()
            self.init_r1_state = self.r1_state.copy()

        # Reset robot states
        self.r0_state = self.update_buffer(self.r0_state, self.time_step, update_vel=False)
        self.r1_state = self.update_buffer(self.r1_state, self.time_step, update_vel=False)

        self.r0_state.tool_vel = np.zeros(6)
        self.r0_state.world_force = np.zeros(6)
        self.r0_state.force_history = np.zeros([H, 6])

        self.r1_state.tool_vel = np.zeros(6)
        self.r1_state.world_force = np.zeros(6)
        self.r1_state.force_history = np.zeros([H, 6])

        # Reset gains
        self.r0_gains = AdmittanceGains()
        self.r1_gains = AdmittanceGains()
        return self.get_RL_state()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pyaraas",
        action="store_true",
        help="Directory where project assets are located.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Directory where project assets are located.",
    )
    parser.add_argument(
        "--is-remote",
        action="store_true",
        help="Directory where project assets are located.",
    )
    parser.add_argument(
        "--enable-hardware",
        action="store_true",
        help="Directory where project assets are located.",
    )

    args = parser.parse_args()
    return args

def script_run_assembly(args):
    env = AssemblyMultiArm(render=args.render, pyaraas=args.pyaraas, enable_hardware=args.enable_hardware)
    for _ in range(1):
        print("Resetting")
        env.reset()

        # time.sleep(100)
        for i in range(100):
            init_time = time.time()

            action = np.zeros(env.action_dim)
            action[2] = -1
            state, reward, done, info = env.step(action)


def create_assembly_multi_arm_env(**kwargs):
    def _init():
        return AssemblyMultiArm(**kwargs)

    return _init


def register_envs(**kwargs):
    register(
        id='AssemblyMultiArm-v0',  # Unique ID for your environment
        entry_point=create_assembly_multi_arm_env(**kwargs),  # 'filename:classname'
        # max_episode_steps=100,  # Optional: Max steps per episode
    )
    register(
        id='AssemblySciptedInsert-v0',  # Unique ID for your environment
        entry_point=create_assembly_multi_arm_env(**kwargs),  # 'filename:classname'
        # max_episode_steps=100,  # Optional: Max steps per episode
    )
    register(
        id='AssemblyScriptedGrasp-v0',  # Unique ID for your environment
        entry_point=create_assembly_multi_arm_env(**kwargs),  # 'filename:classname'
        # max_episode_steps=100,  # Optional: Max steps per episode
    )
    register(
        id='AssemblyMoveToFixture-v0',  # Unique ID for your environment
        entry_point=create_assembly_multi_arm_env(**kwargs),  # 'filename:classname'
        # max_episode_steps=100,  # Optional: Max steps per episode
    )
    register(
        id='AssemblyScriptedPlace-v0',  # Unique ID for your environment
        entry_point=create_assembly_multi_arm_env(**kwargs),  # 'filename:classname'
        # max_episode_steps=100,  # Optional: Max steps per episode
    )
    register(
        id='AssemblyScriptedRegrasp-v0',  # Unique ID for your environment
        entry_point=create_assembly_multi_arm_env(**kwargs),  # 'filename:classname'
        # max_episode_steps=100,  # Optional: Max steps per episode
    )
    register(
        id='AssemblyMoveToBoard-v0',  # Unique ID for your environment
        entry_point=create_assembly_multi_arm_env(**kwargs),  # 'filename:classname'
        # max_episode_steps=100,  # Optional: Max steps per episode
    )
    register(
        id='AssemblyScriptedInsert-v0',  # Unique ID for your environment
        entry_point=create_assembly_multi_arm_env(**kwargs),  # 'filename:classname'
        # max_episode_steps=100,  # Optional: Max steps per episode
    )

def policy_run_assembly(args):
    env = AssemblyMultiArm(render=args.render, enable_hardware=args.enable_hardware)


if __name__ == '__main__':
    # Cannot launch araas and use it in the same proces
    # file_dir = pathlib.Path(__file__).parent.resolve()
    # pyaraas.launch(os.path.join(file_dir, "apa_workcells"))
    args = parse_args()
    if (args.checkpoint is not None):
        policy_run_assembly(args)
    else:
        script_run_assembly(args)