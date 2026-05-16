import pyatk
import pyaraas
from pyatk import Transform, Vector, Part, Actor, Gripper
import time
import numpy as np
import gym
from gym import spaces, register
from scipy.spatial.transform import Rotation as R
import time
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
    check_collision
from multi_arm_assembly.utils import solve_ik_araas as solve_ik
import multi_arm_assembly.pb_utils as pbu


def obj_to_str(d):
    return json.dumps(d)


def force_sample_to_str(fs):
    return obj_to_str({"force": fs["force"].get_values(), "moment": fs["moment"].get_values()})


UNIT_T = Transform(np.array([0, 0, 0, 0, 0, 0]))
H = 10
UNIT_OFFSET = 1000
OPEN_WIDTH = 40
WORKCELL_NAME = "MAR_PEG_85"


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
        # self.teleport = False

        self.skip_setup = False

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
        self.rotation_limit = 1 / 180 * np.pi  # in rad
        self.max_step = 2

        self.w = pyaraas.start(WORKCELL_NAME, enable_hardware=self.enable_hardware)
        self.path_planner = PathPlanner(self.w, num_workers=4, expand_roadmap=True)
        self.setup = False

        self.collisions_enabled = True

        self.r0 = self.w.get_robot("UR10e-0")
        self.r0.set_collision_model(self.collisions_enabled, self.collisions_enabled)
        self.r0_plate = self.w.get_peripheral("UR10e-Mounting-Plate-ST-RB-001-0009-1")
        self.r0_plate.set_collision_model(self.collisions_enabled, self.collisions_enabled)
        self.r1 = self.w.get_robot("UR10e-1")
        self.r1.set_collision_model(self.collisions_enabled, self.collisions_enabled)
        self.r1_plate = self.w.get_peripheral("UR10e-Mounting-Plate-ST-RB-001-0009-0")

        self.block = self.w.get_peripheral("just_a_medium_cube-0")
        self.block.set_collision_model(self.collisions_enabled, self.collisions_enabled)

        self.g0 = self.w.get_gripper("Rq85-0")
        self.g0.set_collision_model(self.collisions_enabled, self.collisions_enabled)
        self.g1 = self.w.get_gripper("Rq85-1")
        self.g1.set_collision_model(self.collisions_enabled, self.collisions_enabled)

    def set_up_spaces(self):
        action_bound = 1
        self.action_dim = 3 * self.init_cfg.robot_count  # Just translation for now
        action_high = np.array([action_bound] * self.action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        obs_bound = 10
        obs_dim = 9 * self.init_cfg.robot_count  # Joint positions and velocities
        obs_high = np.array([obs_bound] * obs_dim)
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
        print("Taraget q: " + str(target_q))
        robot_index = [self.r0, self.r1].index(robot)
        trajectory = solve_motion_plan(start_qs, target_q, target_robot=robot_index, step_through=step_through,
                                       ee_collisions=ee_collisions)
        print('trajectory, start_qs, target_q: ', trajectory, start_qs, target_q, robot_index, step_through, ee_collisions)
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
        print("Taraget q: "+str(target_q))
        try:
            time.sleep(0.5)
            speed_factor = 0.1 if self.teleport else 1.0
            pyaraas.run(pyaraas.Task(self.path_planner.plan_and_execute_raw_joint_trajectory, robot, target_q, speed_factor))
        except Exception:
            traceback.print_exc()
            import sys
            sys.exit()

    def current_robot_qs(self):
        return [self.r0.get_joint_angles(joint_space=self.enable_hardware),
                self.r1.get_joint_angles(joint_space=self.enable_hardware)]

    def plan_to_flange_transform(self, robot, world_T_flange, ee_collisions=True, step_through=False):
        print("Planning to flange transform : " + str(world_T_flange))
        current_qs = self.current_robot_qs()
        target_poses = [None, None]
        ridx = [self.r0, self.r1].index(robot)
        # target_pose = APose.from_transform(world_T_flange).to_list()
        target_poses[ridx] = APose.from_transform(world_T_flange).to_list()
        target_q = solve_ik(current_qs, target_poses, ee_collisions=ee_collisions)[ridx]
        if (target_q is None):
            solve_ik(current_qs, target_poses, ee_collisions=ee_collisions, step_through=True)[ridx]
            assert False
        self.plan_to_joint_angles(robot, current_qs, target_q, ee_collisions=ee_collisions, step_through=step_through)
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
                p_pose = copy.deepcopy(rstate.tool_pose)
                world_force = rstate.world_force.copy()
                world_force = np.clip(world_force, -gains.force_limit, gains.force_limit)
                world_force = np.divide(world_force, gains.force_limit)
                rstate_vecs += [p_pose.trans, p_pose.rotm[:3, 0], p_pose.rotm[:3, 1]]

        obs = np.concatenate(rstate_vecs)
        print("Step: " + str(self.steps))

        log_list("araas_obs", [obs])
        return {"policy": obs}

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
            delta[3:6] = action[3:6]  # / (np.linalg.norm(action[3:6], ord=2) + 1e-3) * self.rotation_limit

        delta_rotm = R.from_euler('XYZ', delta[3:6]).as_matrix()
        log_list("araas_tool_quat", [R.from_matrix(rstate.tool_pose.rotm).as_quat()])
        control_cmd = np.zeros(13)
        control_cmd[:3] = rstate.tool_pose.trans[:3] + delta[:3]
        control_cmd[3:7] = R.from_matrix(rstate.tool_pose.rotm @ delta_rotm).as_quat()
        log_list("araas_cmd", [control_cmd])
        return control_cmd

    def set_mode(self, mode, init_cfg):
        self.setup = False
        self.mode = mode
        # import pdb; pdb.set_trace()
        if init_cfg is None:
            return

        print("Setting mode: " + str(mode))

        # First, we need to make sure the robots are not in collision
        for offset in range(10):
            qs = self.current_robot_qs()
            starting_flanges = [self.r0.get_flange_transform(), self.r1.get_flange_transform()]
            print("Offset: " + str(offset))
            if (check_collision(*qs)):
                if (not init_cfg.holding_peg):
                    pre = starting_flanges[0].multiply(Transform([0, 0, -10, 0, 0, 0]))
                    pyaraas.run(pyaraas.Task(self.r0.move_cartesian, pre, 0.2))
                if (not init_cfg.holding_hole):
                    pre = starting_flanges[1].multiply(Transform([0, 0, -10, 0, 0, 0]))
                    pyaraas.run(pyaraas.Task(self.r1.move_cartesian, pre, 0.2))
            else:
                break

        if (not init_cfg.holding_peg):
            # Move robot 2 out of the way
            pyaraas.run(pyaraas.Task(self.g0.open, OPEN_WIDTH, 1))
            self.plan_to_flange_transform(self.r0, WORLD_T_R0_HOME, step_through=False)

        if (not init_cfg.holding_hole):
            # Move robot 2 out of the way
            pyaraas.run(pyaraas.Task(self.g1.open, OPEN_WIDTH, 1))
            self.plan_to_flange_transform(self.r1, WORLD_T_R1_HOME)

        self.init_cfg = init_cfg

        self.set_up_spaces()

    def step(self, action):

        log_list("araas_action", [action])
        print(action)
        self.steps += 1

        # Assuming actions are always split down the middle
        r0_action_size = 3 * (int(self.init_cfg.holding_peg) + int(self.init_cfg.allow_peg_rotation))
        r1_action_size = 3 * (int(self.init_cfg.holding_hole) + int(self.init_cfg.allow_hole_rotation))

        r0_action = action[:r0_action_size]
        r1_action = action[r0_action_size:r0_action_size + r1_action_size]

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

        return state, 0, done, {}

    def get_number_of_agents(self):
        return 1

    def release_and_lift_up(self):
        pyaraas.run(pyaraas.Task(self.g0.open, 35, 1))
        self.plan_to_flange_transform(self.r0, Transform([0, 0, 60, 0, 0, 0]).multiply(self.r0.get_flange_transform()))

    def pick_up_and_move_start(self):

        approach = Transform([0, 0, 60, 0, 0, 0])

        # Robot 1 Grasp
        if (self.init_cfg.holding_peg and self.init_cfg.WORLD_T_TOOL_PICK_PEG is not None):
            world_T_flange0_grasp = APose(*self.init_cfg.WORLD_T_TOOL_PICK_PEG).to_transform()
            world_T_flange0_pregrasp = approach.multiply(world_T_flange0_grasp)
            pyaraas.run(pyaraas.Task(self.g0.open, OPEN_WIDTH, 1))
            self.plan_to_flange_transform(self.r0, world_T_flange0_pregrasp)
            self.plan_to_flange_transform(self.r0, world_T_flange0_grasp, ee_collisions=False)
            pyaraas.run(pyaraas.Task(self.g0.open, 0, 1))
            self.plan_to_flange_transform(self.r0, world_T_flange0_pregrasp, ee_collisions=False)

        if (self.init_cfg.holding_hole and self.init_cfg.WORLD_T_TOOL_PICK_HOLE is not None):
            world_T_flange1_grasp = APose(*self.init_cfg.WORLD_T_TOOL_PICK_HOLE).to_transform()
            world_T_flange1_pregrasp = approach.multiply(world_T_flange1_grasp)
            pyaraas.run(pyaraas.Task(self.g1.open, OPEN_WIDTH, 1))
            self.plan_to_flange_transform(self.r1, world_T_flange1_pregrasp)
            self.plan_to_flange_transform(self.r1, world_T_flange1_grasp, ee_collisions=False)
            pyaraas.run(pyaraas.Task(self.g1.open, 0, 1))
            self.plan_to_flange_transform(self.r1, world_T_flange1_pregrasp, ee_collisions=False)

        base_poses = []
        target_poses = [None, None]
        if (self.init_cfg.holding_peg):
            base_poses.append(R0_POSE)
            target_poses[0] = self.init_cfg.WORLD_T_PEG_TOOL_START
        if (self.init_cfg.holding_hole):
            base_poses.append(R1_POSE)
            target_poses[1] = self.init_cfg.WORLD_T_HOLE_TOOL_START

        if (self.init_cfg.PEG_IK_WORLD_T_TOOL is not None):
            target_poses[0] = self.init_cfg.PEG_IK_WORLD_T_TOOL

        if (self.init_cfg.HOLE_IK_WORLD_T_TOOL is not None):
            target_poses[1] = self.init_cfg.HOLE_IK_WORLD_T_TOOL

        solutions = solve_ik(self.current_robot_qs(), target_poses)
        if (solutions is None):
            solve_ik(self.current_robot_qs(), target_poses, step_through=True)
            assert False

        if (self.init_cfg.holding_peg):
            # step_through = (self.mode == "bolt_in_strut")
            step_through = False
            self.plan_to_joint_angles(self.r0, self.current_robot_qs(), solutions[0], step_through=step_through)

        if (self.init_cfg.holding_hole):
            self.plan_to_joint_angles(self.r1, self.current_robot_qs(), solutions[-1])

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
            pyaraas.run(pyaraas.Task(gripper.open, OPEN_WIDTH, 1))

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

            self.f0 = self.w.get_force_sensor('UR10e-0')
            self.f1 = self.w.get_force_sensor('UR10e-1')
            self.table = self.w.get_peripheral("rectangularTable-0")
            self.table.set_margin(0)
            self.table.set_collision_model(self.collisions_enabled, self.collisions_enabled)

            self.kit = self.w.get_peripheral("Kit - Stewart Platform v29-0")
            self.kit.set_collision_model(self.collisions_enabled, self.collisions_enabled)

            self.stand = self.w.get_peripheral("stand_for_stewart_platform_assembly_no_motor-0")
            self.stand.set_collision_model(self.collisions_enabled, self.collisions_enabled)

            if (not self.skip_setup):
                if (self.mode == "screw_in_bolt"):
                    self.hardcode_screw_in_bolt(self.r1, self.g1)
                    return
                elif (self.mode == "lift_up"):
                    self.release_and_lift_up()
                    return
                else:
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