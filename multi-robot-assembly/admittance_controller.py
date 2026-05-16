import numpy as np
from scipy.spatial.transform import Rotation as R

import pyaraas
from pyaraas import Task
from pyatk import Vector, Transform
import trio
import math

class admittance_controller():
    def __init__(self, workcell, robot, force_sensor, max_force=100, dt=1. / 125.0, max_v=20, max_w=0.1):
        self.w = workcell
        self.r = robot
        self.force_sensor = force_sensor
        self.world_T_flange = self.r.get_flange_transform()

        self.time_step = dt
        self.max_v = max_v  # max velocity
        self.max_w = max_w  # max angular velocity
        self.max_force = max_force
        self.M = np.array([40, 40, 40, 20, 20, 20])
        self.kp = np.array([400, 400, 400, 400, 400, 400])
        # critically damped gains (overdamping is also suggested but seems to make the motors grind)
        self.kd = 2 * np.sqrt(np.multiply(self.M, self.kp))
        self.update_gain(self.M, self.kp, self.kd)
        self.force_limit = 50
        self.torque_limit = 1

        # For unit change
        self.unit_offset = 1000
        # Create some buffer for peg pose, link_6 rot
        self.p_pose = np.zeros(6)
        self.p_vel = np.zeros(6)
        self.p_rot = np.zeros([3, 3])
        self.link6_rot = np.zeros([3, 3])
        self.world_force = np.zeros(6)
        self.world_force_offset = np.zeros(6)
        self.update_buffer(update_vel=False)
        self.init_p_pos = self.p_pose[0:3].copy()
        self.init_p_rot = self.p_rot.copy()

        # force calibration (when the robot moves, the force sensor
        # will detect the load*acc which affects the "zero" contact force reading)
        # we zero the force sensor during the first H+1 samples of motion assuming
        # contact doesn't occur before the sampling ends. 
        H = 10
        self.force_history = np.zeros([H, 6])
        self.init_world_force = np.zeros(3)
        self.init_world_torque = np.zeros(3)
        self.calibration_sample = 0
        self.forces_zeroed = False

        self.enable_compliance = True

    def calibrate_force(self):
        if self.calibration_sample < self.force_history.shape[0]:
            reading = self.force_sensor.sample()
            force = np.array([reading["force"].x, reading["force"].y, reading["force"].z])
            torque = np.array([reading["moment"].x, reading["moment"].y, reading["moment"].z])
            init_world_force = self.link6_rot @ force
            init_world_torque = self.link6_rot @ torque
            self.force_history[self.calibration_sample, :3] = init_world_force
            self.force_history[self.calibration_sample, 3:] = init_world_torque
            self.calibration_sample += 1
        else:
            self.init_world_force = np.mean(self.force_history[:, :3], axis=0)
            self.init_world_torque = np.mean(self.force_history[:, 3:], axis=0)
            self.force_history[:, :3] = self.force_history[:, :3] - self.init_world_force
            self.force_history[:, 3:] = self.force_history[:, 3:] - self.init_world_torque
            self.forces_zeroed = True

    def get_pose_from_transform(self, transform):
        pos = transform.position
        quat = transform.rotation
        pos_np = np.array([pos.x, pos.y, pos.z])
        quat_np = np.array([quat.x, quat.y, quat.z, quat.w])
        rot = R.from_quat(quat_np).as_matrix()
        pose = np.array(transform.get_values())
        pose[0:3] = pose[0:3] / self.unit_offset  # mm to m
        return pose, rot

    def update_buffer(self, update_vel=True):
        temp_p_pose, temp_p_rot = self.get_pose_from_transform(self.world_T_flange)
        _, self.link6_rot = self.get_pose_from_transform(self.world_T_flange)
        if update_vel:
            vel = np.zeros(6)
            vel[:3] = (temp_p_pose[:3] - self.p_pose[:3]) / self.time_step
            # Rd @ R0 = R1
            Rd = temp_p_rot @ self.p_rot.T
            axis_angle_vel = R.from_matrix(Rd).as_rotvec() / self.time_step
            vel[3:] = axis_angle_vel
            self.p_vel = vel
        self.p_pose = temp_p_pose
        self.p_rot = temp_p_rot

    def update_gain(self, M, kp, kd):
        self.M = M
        self.kp = kp
        self.kd = kd

    def update_force(self):
        if not self.forces_zeroed:
            self.calibrate_force()
            return
        reading = self.force_sensor.sample()
        force = np.array([reading["force"].x, reading["force"].y, reading["force"].z])
        torque = np.array([reading["moment"].x, reading["moment"].y, reading["moment"].z])
        world_force = self.link6_rot @ force - self.init_world_force
        world_torque = self.link6_rot @ torque - self.init_world_torque

        H = self.force_history.shape[0]
        self.force_history[:H - 1, :] = self.force_history[1:H, :]
        self.force_history[-1, :3] = world_force
        self.force_history[-1, 3:] = world_torque
        # average within the sliding window to de-noise the FT reading
        self.world_force[:3] = np.mean(self.force_history[:, :3], axis=0)
        self.world_force[3:] = np.mean(self.force_history[:, 3:], axis=0)

    def update_control(self, control_cmd):
        # Admittance control
        # Clip the cmd to be safe
        # control_cmd = self.apply_control_limit(control_cmd)
        d_pos = control_cmd[:3]  # in m
        d_quat = control_cmd[3:7]  # in quat
        d_rot = R.from_quat(d_quat).as_matrix()
        d_pos_vel = control_cmd[7:10]
        d_ori_vel = control_cmd[10:13]

        pos_error = self.p_pose[:3] - d_pos
        ori_error_rot = self.p_rot @ d_rot.T
        # from rot(9 dim) to error (3 dim), need to use axis-angle representation
        ori_error = R.from_matrix(ori_error_rot).as_rotvec()
        error = np.hstack([pos_error, ori_error])
        pos_error_dot = self.p_vel[:3] - d_pos_vel
        ori_error_dot = self.p_vel[3:] - d_ori_vel
        error_dot = np.hstack([pos_error_dot, ori_error_dot])

        # Feedback law: M*acc = F - kp e - kd e_dot
        world_force_clip = np.array(self.world_force)
        # clamp the force and torque while maintaining the "direction"
        force_value = Vector(world_force_clip[0],world_force_clip[1],world_force_clip[2]).magnitude()
        if force_value > self.force_limit:
            world_force_clip[:3] = world_force_clip[:3]*self.force_limit/force_value
        torque_value = Vector(world_force_clip[3],world_force_clip[4],world_force_clip[5]).magnitude()
        if torque_value > self.torque_limit:
            world_force_clip[3:] = world_force_clip[3:]*self.torque_limit/torque_value
        if not self.enable_compliance or not self.forces_zeroed:
            # turn off compliance
            world_force_clip = np.zeros(6)


        RHS = 1 * world_force_clip - np.multiply(self.kp, error) - np.multiply(self.kd, error_dot)
        acc = np.divide(RHS, self.M)
        world_vel_cmd = acc * self.time_step # proper way is tyo add the prev velocity but that seems to make robot drift which means tune the gains higher to make positioning force stronger
        world_vel_cmd[0:3] = world_vel_cmd[0:3] * self.unit_offset  # mm to m
        # clamp the velocity and angular velocity command while maintaining "direction"
        vel_value = Vector(world_vel_cmd[0],world_vel_cmd[1],world_vel_cmd[2]).magnitude()
        if vel_value > self.max_v:
            world_vel_cmd[:3] = world_vel_cmd[:3]*self.max_v/vel_value
        w_value = Vector(world_vel_cmd[3],world_vel_cmd[4],world_vel_cmd[5]).magnitude()
        if w_value > self.max_w:
            world_vel_cmd[3:] = world_vel_cmd[3:]*self.max_w/w_value
        cmd = (np.block(
            [[self.link6_rot, np.zeros([3, 3])], [np.zeros([3, 3]), self.link6_rot]]).T @ world_vel_cmd).tolist()
        return cmd

    async def goto(self, workcell_T_flange_goal, duration, disable_rotation = True, process_cb=None):
        print("GOTO ADMITTANCE GOAL")

        self.disable_rotation = disable_rotation
        # convert from araas to controller format (mm to meters)
        goal_pose = [
            workcell_T_flange_goal.position.x/1000.0,
            workcell_T_flange_goal.position.y/1000.0,
            workcell_T_flange_goal.position.z/1000.0,
            workcell_T_flange_goal.rotation.x,
            workcell_T_flange_goal.rotation.y,
            workcell_T_flange_goal.rotation.z,
            workcell_T_flange_goal.rotation.w
        ]

        # init control cmd (goal pose)      
        self.control_cmd = np.zeros(13)
        self.control_cmd[:3] = goal_pose[:3]
        self.control_cmd[3:7] = goal_pose[3:7]

        if process_cb:
            process_cb(self)

        # velocity action will be updated at the rate determined by self.time_step
        # keep the action around though by a factor of two 
        velocity_action_lifetime = self.time_step*2
        num_samples = int(duration/self.time_step)
        self.stop = False
        force_log = []
        for i in range(num_samples):
            cmd = self.update_control(self.control_cmd)
            if self.disable_rotation:
                cmd[3] = cmd[4] = cmd[5] = 0
            raw_states, self.world_T_flange = self.r.set_cartesian_velocity(Vector(cmd[0], cmd[1], cmd[2]), 
                                                       Vector(cmd[3], cmd[4], cmd[5]),
                                                       velocity_action_lifetime, self.max_force, mass=1)
            await trio.sleep(self.time_step)

            self.update_buffer()
            # self.update_force()

            data_log = np.copy(self.world_force)
            force_log.append(data_log)
            print(data_log)

            if process_cb:
                process_cb(self)

            if self.stop:
                break            


        # wait just a bit more to ensure the velocity action is finished 
        #(this is accounting for the action update rate as well)
        await trio.sleep(0.1)

        return force_log