import pyatk
import pyaraas
from pyatk import Transform, Vector, Part, Actor, Gripper
from pathlib import Path
import time
import numpy as np
import gym
from gym import spaces,register
from scipy.spatial.transform import Rotation as R
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Any
import copy
import argparse
import os
import pathlib
import math
from pyaraas.tools import PathPlanner
import traceback
import sys
import trio

UNIT_T = Transform(np.array([0, 0, 0, 0, 0, 0]))
H = 10
UNIT_OFFSET = 1000


@dataclass
class Pose():
    trans: np.array
    rotm: np.array
    euler: np.array

def get_pose_from_transform(transform:Transform):
    pos = transform.position
    quat = transform.rotation
    pos_np = np.array([pos.x, pos.y, pos.z])
    quat_np = np.array([quat.x, quat.y, quat.z, quat.w])
    rot = R.from_quat(quat_np).as_matrix()
    pose = np.array(transform.get_values())
    pose[0:3] = pose[0:3] / UNIT_OFFSET  # mm to m
    return Pose(pose[0:3], rot, pose[3:])

def pose_multiply(pose1, pose2):
    return np.array(Transform(pose1).multiply(Transform(pose2)).get_values())

def pose_invert(pose):
    return np.array(Transform(pose).invert().get_values())
    
@dataclass
class RobotState:
    robot: Actor
    gripper: Gripper
    tool_part: Part
    force_sensor: Any = None
    init_world_force: np.array = np.zeros(6)
    init_world_torque: np.array = np.zeros(6)
    tool_pose: np.array = np.zeros(6)
    tool_vel: np.array = np.zeros(6)
    force_history: np.array = np.zeros([H, 6])
    world_force: np.array = np.zeros(6)

    def copy(self):
        return RobotState(
            self.robot,
            self.gripper,
            self.tool_part,
            self.force_sensor,
            copy.deepcopy(self.init_world_force),
            copy.deepcopy(self.init_world_torque),
            copy.deepcopy(self.tool_pose),
            copy.deepcopy(self.tool_vel),
            copy.deepcopy(self.force_history),
            copy.deepcopy(self.world_force),
        )
        
M_DEFAULT_LIN = 20
M_DEFAULT_ANG = 20
KP_DEFAULT = 400
KD_DEFAULT_LIN = 2 * np.sqrt(np.multiply(M_DEFAULT_LIN, KP_DEFAULT))
KD_DEFAULT_ANG = 2 * np.sqrt(np.multiply(M_DEFAULT_ANG, KP_DEFAULT))
FORCE_LIMIT_TRANS = 50

@dataclass
class AdmittanceGains:
    M: np.array = np.array([M_DEFAULT_LIN, M_DEFAULT_LIN, M_DEFAULT_LIN, M_DEFAULT_ANG, M_DEFAULT_ANG, M_DEFAULT_ANG])
    kp: np.array = np.array([KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT])
    kd: np.array = np.array([KD_DEFAULT_LIN, KD_DEFAULT_LIN, KD_DEFAULT_LIN, KD_DEFAULT_ANG, KD_DEFAULT_ANG, KD_DEFAULT_ANG])
    force_limit: np.array = np.array([FORCE_LIMIT_TRANS, FORCE_LIMIT_TRANS, FORCE_LIMIT_TRANS, 1, 1, 1])

class AssemblyMultiArm(gym.Env):
    def __init__(self, friction=0.02, force_limit=50, dt=1./125.0, max_v=20, max_rad=1, mass=1, gap=10,
                        render=False, 
                        verbose=False, 
                        use_rotation_randomization=False, 
                        real_robot = False, 
                        is_remote = False,
                        **kwargs):
        self.friction = friction
        self.force_limit = force_limit
        self.torque_limit = 1
    
        self.dt = dt
        self.forces_zeroed = False
        self.enable_compliance = True
        self.max_v = max_v
        self.max_w = max_rad
        self.max_rad = max_rad
        self.mass = mass
        self.gap = gap
        self.use_rotation_randomization=use_rotation_randomization
        self.real_robot = real_robot
        self.verbose = verbose
        self.max_time_per_step = 0.5
        self.offset_to_tip = np.array([0,0,0.217])
        
        action_bound = 1
        action_dim = 12 * 2 # two robots
        action_high = np.array([action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        obs_bound = 1
        obs_dim = 18 * 2
        obs_high = np.array([obs_bound] * obs_dim)
        self.observation_space = spaces.Box(-obs_high, obs_high)
        
        
        ip = "10.140.68.92" if is_remote else "127.0.0.1"
        if(self.real_robot):
            self.w = pyaraas.start("MAR_PEG", enable_hardware=False)  
            self.path_planner = PathPlanner(self.w, num_workers = 4, roadmap_path = "./roadmap.pkl")
        else:
            # Init the pyatk
            if(render):
                pyatk.init(True, addr=ip)
            else:
                pyatk.init(False, addr=ip)
     
            # Set the dir
            current_dir = Path().resolve()
            pyatk_dir = str(current_dir.joinpath("apa_workcells/APA"))
            pyatk.set_project_dir(pyatk_dir)
            # Load workcell (UR10e + gripper)
            gravity = pyatk.Vector(0, 0, 0)
            self.w = pyatk.load_workcell("MAR_PEG", gravity=gravity)
        
        self.setup = False
       
       

    def calibrate_force(self, rstate:RobotState):
        new_rstate = rstate.copy()
        for _ in range(rstate.force_history.shape[0]):
            try:
                reading = new_rstate.force_sensor.sample()
                print("Force Reading: "+str(reading))
            except:
                print("Failed force reading. Using all zeros")
                reading = {
                    "force": Vector(0,0,0),
                    "moment": Vector(0,0,0)
                }
            force = np.array([reading["force"].x, reading["force"].y, reading["force"].z])
            torque = np.array([reading["moment"].x, reading["moment"].y, reading["moment"].z])
            init_world_force = new_rstate.tool_pose.rotm @ force
            init_world_torque = new_rstate.tool_pose.rotm @ torque
            new_rstate.force_history[_, :3] = init_world_force
            new_rstate.force_history[_, 3:] = init_world_torque
        new_rstate.init_world_force = np.mean(new_rstate.force_history[:, :3], axis=0)
        new_rstate.init_world_torque = np.mean(new_rstate.force_history[:, 3:], axis=0)
        new_rstate.force_history[:, :3] = new_rstate.force_history[:, :3] - new_rstate.init_world_force
        new_rstate.force_history[:, 3:] = new_rstate.force_history[:, 3:] - new_rstate.init_world_torque
        return new_rstate
    
    def move_to_flange_transform(self, robot, world_T_flange):
        if not robot.get_flange_transform().is_equal(1, 0.1, world_T_flange):
            time.sleep(1)
            pyaraas.run(pyaraas.Task(self.path_planner.plan_and_execute_joint_trajectory, robot, world_T_flange, 1))
                
    def setup_hole(self, gap:float):

        hole_size = 40+gap
        
        h = self.w.add_part("square_hole_part_40_120", "hole_4", Transform(np.array([0,50,-20,np.pi/2,0,0])))
        h.set_collision_model(True, True)  # collision, cancave
        if not self.real_robot:
            h.set_friction(self.tool_friction)
        h.set_margin(0.001)
        
        h1 = self.w.add_part("square_hole_part_40_120", "hole_0", Transform(np.array([hole_size,0,0,0,0,np.pi/2])))
        h1.set_collision_model(True, True)  # collision, cancave
        if not self.real_robot:
            h1.set_friction(self.tool_friction)
        h1.set_margin(0.001)
        
        
        h2 = self.w.add_part("square_hole_part_40_120", "hole_1",Transform(np.array([-hole_size,0,0,0,0,np.pi/2])))
        h2.set_collision_model(True, True)  # collision, cancave
        if not self.real_robot:
            h2.set_friction(self.tool_friction)
        h2.set_margin(0.001)
        
        
        h3 = self.w.add_part("square_peg_40", "hole_2", Transform(np.array([0,hole_size,0,0,0,0])))
        h3.set_collision_model(True, True)  # collision, cancave
        if not self.real_robot:
            h3.set_friction(self.tool_friction)
        h3.set_margin(0.001)
        
        h4 = self.w.add_part("square_peg_40", "hole_3", Transform(np.array([0,-hole_size,0,0,0,0])))
        h4.set_collision_model(True, True)  # collision, cancave
        if not self.real_robot:
            h4.set_friction(self.tool_friction)
        h4.set_margin(0.001)
        
        collection = [h1, h2, h3, h4]
        
        # Linking function is different between araas and atk for some reason
        if(self.real_robot):
            h.link(collection, True)
        else:
            for c in collection:
                h.link(c, True)

        return h
    
    def FT_calibration(self, initial_rstate:RobotState, current_rstate:RobotState, gains:AdmittanceGains, H=10):
        new_rstate = current_rstate.copy()
        FT_list = []
        for _ in range(H):
            control_cmd = np.zeros(13)
            control_cmd[:3] = initial_rstate.tool_pose.trans
            control_cmd[3:7] = R.from_matrix(initial_rstate.tool_pose.rotm).as_quat()
            new_rstate = self.update_control(control_cmd, new_rstate, gains)
            FT_list.append(copy.deepcopy(new_rstate.world_force))
        return new_rstate

    def render(self, **kwargs):
        pass
        
    def update_buffer(self, rstate:RobotState, dt:float, update_vel=True, flange_transform:Transform=None) -> RobotState:
        new_rstate = rstate.copy()
        if(flange_transform is None):
            # In sim
            temp_pose = get_pose_from_transform(new_rstate.robot.get_flange_transform())
        else:
            # In real
            temp_pose = get_pose_from_transform(flange_transform)
                    
        if update_vel:
            vel = np.zeros(6)
            vel[:3] = (temp_pose.trans - new_rstate.tool_pose.trans) / dt
            Rd = temp_pose.rotm @ new_rstate.tool_pose.rotm.T
            axis_angle_vel = R.from_matrix(Rd).as_rotvec() / dt
            vel[3:] = axis_angle_vel
            new_rstate.tool_vel = vel
        new_rstate.tool_pose = temp_pose
        return new_rstate

    def update_force_sim(self, rstate:RobotState, raw_states:List[Transform]):
        
        new_rstate = rstate.copy()
        force = np.array([raw_states[1].x, raw_states[1].y, raw_states[1].z])
        torque = np.array([raw_states[2].x, raw_states[2].y, raw_states[2].z])
        world_force = new_rstate.tool_pose.rotm @ force
        world_torque = new_rstate.tool_pose.rotm @ torque
        H = new_rstate.force_history.shape[0]
        new_rstate.force_history[:H-1, :] = new_rstate.force_history[1:H, :]
        new_rstate.force_history[-1, :3] = world_force
        new_rstate.force_history[-1, 3:] = world_torque
        
        # average within the sliding window to de-noise the FT reading
        new_rstate.world_force[:3] = np.mean(new_rstate.force_history[:, :3], axis=0)
        new_rstate.world_force[3:] = np.mean(new_rstate.force_history[:, 3:], axis=0)
        
        return new_rstate
    

    def update_force_real(self, rstate:RobotState):
        
        new_rstate = rstate.copy()
        try:
            reading = new_rstate.force_sensor.sample()
            print("Force Reading: "+str(reading))
        except:
            print("Failed force reading. Using all zeros")
            reading = {
                "force": Vector(0,0,0),
                "moment": Vector(0,0,0)
            }
        force = np.array([reading["force"].x, reading["force"].y, reading["force"].z])
        torque = np.array([reading["moment"].x, reading["moment"].y, reading["moment"].z])
        world_force = new_rstate.tool_pose.rotm @ force - new_rstate.init_world_force
        world_torque = new_rstate.tool_pose.rotm @ torque - new_rstate.init_world_torque

        H = new_rstate.force_history.shape[0]
        new_rstate.force_history[:H-1, :] = new_rstate.force_history[1:H, :]
        new_rstate.force_history[-1, :3] = world_force
        new_rstate.force_history[-1, 3:] = world_torque
        
        # average within the sliding window to de-noise the FT reading
        new_rstate.world_force[:3] = np.mean(new_rstate.force_history[:, :3], axis=0)
        new_rstate.world_force[3:] = np.mean(new_rstate.force_history[:, 3:], axis=0)
        
        return new_rstate    

    def update_control(self, 
                       control_cmd:List[float], 
                       rstate:RobotState, 
                       gains:AdmittanceGains):
    
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
        force_value = Vector(world_force_clip[0],world_force_clip[1],world_force_clip[2]).magnitude()
        if force_value > self.force_limit:
            world_force_clip[:3] = world_force_clip[:3]*self.force_limit/force_value
        torque_value = Vector(world_force_clip[3],world_force_clip[4],world_force_clip[5]).magnitude()
        if torque_value > self.torque_limit:
            world_force_clip[3:] = world_force_clip[3:]*self.torque_limit/torque_value
        if not self.enable_compliance or not self.forces_zeroed:
            # turn off compliance
            world_force_clip = np.zeros(6)


        RHS = 1 * world_force_clip - np.multiply(gains.kp, error) - np.multiply(gains.kd, error_dot)
        acc = np.divide(RHS, gains.M)
        world_vel_cmd = acc * self.time_step + rstate.tool_vel # proper way is tyo add the prev velocity but that seems to make robot drift which means tune the gains higher to make positioning force stronger
        world_vel_cmd[0:3] = world_vel_cmd[0:3] * UNIT_OFFSET  # mm to m
        # clamp the velocity and angular velocity command while maintaining "direction"
        vel_value = Vector(world_vel_cmd[0],world_vel_cmd[1],world_vel_cmd[2]).magnitude()
        if vel_value > self.max_v:
            world_vel_cmd[:3] = world_vel_cmd[:3]*self.max_v/vel_value
        w_value = Vector(world_vel_cmd[3],world_vel_cmd[4],world_vel_cmd[5]).magnitude()
        if w_value > self.max_w:
            world_vel_cmd[3:] = world_vel_cmd[3:]*self.max_w/w_value
        cmd = (np.block(
            [[rstate.tool_pose.rotm, np.zeros([3, 3])], [np.zeros([3, 3]), rstate.tool_pose.rotm]]).T @ world_vel_cmd).tolist()
        return cmd


    def get_RL_state(self):
        rstate_vecs = []
        for rstate, gains in [(self.r0_state, self.r0_gains), (self.r1_state, self.r1_gains)]:
            p_pose = copy.deepcopy(rstate.tool_pose)
            p_pose.trans[:3] = p_pose.trans[:3] - self.world_origin + self.nominal_hole_offset[:3]
            p_pose.euler = p_pose.euler + self.nominal_hole_offset[3:6]
            p_vel = rstate.tool_vel.copy()
            world_force = rstate.world_force.copy()
            world_force = np.clip(world_force, -gains.force_limit, gains.force_limit)
            world_force = np.divide(world_force, gains.force_limit)
            rstate_vecs+=[p_pose.trans, p_pose.euler, p_vel, world_force]
            
        return np.concatenate(rstate_vecs)

            
    def get_updated_gains(self, gains, action):
        # Get gains from action
        kp = (self.kp_high + self.kp_low) / 2 + np.multiply(action[6:12],(self.kp_high - self.kp_low) / 2 )
        kd = 4 * np.sqrt(np.multiply(gains.M, kp)) # 4 to make thw system overdamped
        
        new_gains = AdmittanceGains(
            gains.M, kp, kd, gains.force_limit
        )
        return new_gains
        
    def get_control_command(self, rstate, action):
        # Get delta_motion from action
        delta = np.zeros(6)
        delta[:3] = action[:3] / (np.linalg.norm(action[:3], ord=2) + 1e-3) * self.translation_limit
        delta[3:6] = action[3:6] / (np.linalg.norm(action[3:6], ord=2) + 1e-3) * self.rotation_limit
        delta_rotm = R.from_euler('xyz', delta[3:6]).as_matrix()
    
        control_cmd = np.zeros(13)
        control_cmd[:3] = rstate.tool_pose.trans[:3] + delta[:3]
        control_cmd[3:7] = R.from_matrix(rstate.tool_pose.rotm@delta_rotm).as_quat()
        return control_cmd

            
    def step(self, action):
        
        self.steps+=1
        
        r0_action = action[:12]
        r1_action = action[12:]
        
        self.r0_gains = self.get_updated_gains(self.r0_gains, r0_action)
        self.r1_gains = self.get_updated_gains(self.r1_gains, r1_action)

        r0_control_command = self.get_control_command(self.r0_state, r0_action)
        r1_control_command = self.get_control_command(self.r1_state, r1_action)
        
        if(not self.real_robot):

            # Simulated velocity controller
            for _ in range(int(self.step_time_step/self.time_step)):
                # control and step
                r0_vel_command = self.update_control(r0_control_command, self.r0_state, self.r0_gains)
                r1_vel_command = self.update_control(r1_control_command, self.r1_state, self.r1_gains)
                
                r0_vel_output = self.r0_state.tool_part.set_vel(
                    Vector(r0_vel_command[0], r0_vel_command[1], r0_vel_command[2]), 
                    Vector(r0_vel_command[3], r0_vel_command[4], r0_vel_command[5]), 
                    self.time_step)
                
                r1_vel_output = self.r1_state.tool_part.set_vel(
                    Vector(r1_vel_command[0], r1_vel_command[1], r1_vel_command[2]), 
                    Vector(r1_vel_command[3], r1_vel_command[4], r1_vel_command[5]), 
                    self.time_step)
                
                self.r0_state = self.update_buffer(self.r0_state, self.time_step)
                self.r1_state = self.update_buffer(self.r1_state, self.time_step)
                
                self.r0_state = self.update_force_sim(self.r0_state, r0_vel_output)
                self.r1_state = self.update_force_sim(self.r1_state, r1_vel_output)
        else:
            
            # Real velocity controller
            init_time = time.time()
            while((time.time() - init_time) < self.max_time_per_step):
                r0_vel_command = self.update_control(r0_control_command, self.r0_state, self.r0_gains)
                r1_vel_command = self.update_control(r1_control_command, self.r1_state, self.r1_gains)

                time_pre_control = time.time()
                # print("Velocity command: "+str(r0_vel_command))
                r0_vel_command[3] = r0_vel_command[4] = r0_vel_command[5] = 0
                _, r0_flange_transform = self.r0_state.robot.set_cartesian_velocity(
                    Vector(r0_vel_command[0], r0_vel_command[1], r0_vel_command[2]), 
                    Vector(r0_vel_command[3], r0_vel_command[4], r0_vel_command[5]), 
                    self.time_step, 100, mass=1)
                
                r1_vel_command[3] = r1_vel_command[4] = r1_vel_command[5] = 0
                _, r1_flange_transform = self.r1_state.robot.set_cartesian_velocity(
                    Vector(r1_vel_command[0], r1_vel_command[1], r1_vel_command[2]), 
                    Vector(r1_vel_command[3], r1_vel_command[4], r1_vel_command[5]), 
                    self.time_step, 100, mass=1)
    
                # time_so_far = time.time()-time_pre_control
                
                # Make sure we don't control too quickly
                time.sleep(self.time_step)
                
                self.r0_state = self.update_buffer(self.r0_state, self.time_step, flange_transform=r0_flange_transform)
                self.r1_state = self.update_buffer(self.r1_state, self.time_step, flange_transform=r1_flange_transform)
                
                self.r0_state = self.update_force_real(self.r0_state)
                self.r1_state = self.update_force_real(self.r1_state)
                
        state = self.get_RL_state()
        reward, done = self.compute_reward(state)
        info = {}

        if(self.steps>=100):
            done = True
            
        return state, reward, done, info
    
    def pose_diff(self, t1:Transform, t2:Transform, trans_scale=1.0, rot_scale = 0.01):
        t1_vals = np.array(t1.get_values())
        t2_vals = np.array(t2.get_values())
        
        # Position difference
        trans_diff = np.linalg.norm(t1_vals[:3]-t2_vals[:3])
        
        # Create Rotation objects from Euler angles
        rot1 = R.from_euler('zyx', t1_vals[3:])
        rot2 = R.from_euler('zyx', t2_vals[3:])
        
        # Calculate the relative rotation
        rel_rot = rot1.inv() * rot2
        
        # Extract the angle from the relative rotation
        angle_diff = rel_rot.magnitude()
        # print(angle_diff)
        
        return trans_scale*trans_diff #+ rot_scale*angle_diff

    def compute_reward(self, state):
        # Calculate distance to goal
        world_T_peg = self.p.get_transform()
        world_T_hole = self.h.get_transform()
        hole_T_peg = world_T_hole.invert().multiply(world_T_peg)
        dist = self.pose_diff(hole_T_peg, self.goal_hole_T_peg)
        reward = -dist/1000.0
        print("Reward: "+str(reward))
        return reward, False

    def get_number_of_agents(self):
        return 1

    def reset(self, seed=None, options=None):
        print("Restting")
        self.steps = 0
        
        if(not self.setup):
            self.setup = True
            # Init poses
            init_height = 400
            self.g0_T_part = Transform(np.array([0, 0, -50, 0, 0, 0]))
            self.g1_T_hole = Transform(np.array([0, -50, 0, -np.pi/2.0, 0, 0]))
            
            # Tool tip down
            self.initial_world_T_g0 = Transform(np.array([0.0, 0.0, init_height, 0, 0, 0]))
            
            # Sideways
            # self.initial_world_T_g0 = Transform(np.array([0.0, 0.0, init_height, 0, np.pi/2.0, 0]))
            self.initial_world_T_g1 = Transform(np.array([-230, 0.0, init_height, 0, -np.pi/2.0, 0]))
            
            # Dynamic parameters
            self.tool_friction = self.friction
            
            self.time_step = self.dt
            self.step_time_step = 50*self.time_step
            if self.step_time_step < self.time_step:
                self.step_time_step = self.time_step # just in case
                
            print("Here0")
            # Get all parts (robot, peg, hole)
            self.p = self.w.add_part("square_peg_40", "peg0", UNIT_T)
            print("Here1")
            if(not self.real_robot):
                self.p.set_friction(self.tool_friction)
                self.p.set_mass(self.mass)
                
            self.p.set_margin(0.001)
            print("Here")
            self.h = self.setup_hole(self.gap)
            
            self.r0 = self.w.get_robot("UR10e-0")
            self.r1 = self.w.get_robot("UR10e-1")
            self.g0 = self.w.get_gripper("Rq85-0")
            self.g1 = self.w.get_gripper("Rq140-1")
            
            if(self.real_robot):
                self.f0 = self.w.get_force_sensor('UR10e-0')
                self.f1 = self.w.get_force_sensor('UR10e-1')
            else:
                self.f0 = None
                self.f1 = None
            
            if(self.real_robot):
                # I think we cannot set the tcp pose and have IK called automatically in pyaraas,   
                # so we have to manually set up a solver here
                joint_offsets = [0, -math.pi * 0.5, 0, math.pi * 0.5, 0, 0]
                joint_flip = [False, True, False, False, True, False]
                # the numbers here are the D&H values for the UR10e (see the doc for the expected parameters)
                self.ik_solver = pyatk.URSolver(
                    180.7, 174.15, 119.85, 116.55, -612.7, -571.55, joint_offsets, joint_flip
                )
                
                # Get flange_T_gripper. This is probably not the best way
                self.g0_T_flange =  self.g0.get_tcp_transform().invert().multiply(self.r0.get_flange_transform())
                # self.g1_T_flange =  self.g1.get_tcp_transform().invert().multiply(self.r1.get_flange_transform())
                
                self.move_to_flange_transform(self.r0, self.initial_world_T_g0.multiply(self.g0_T_flange))
                # self.move_to_flange_transform(self.r1, self.initial_world_T_g1.multiply(self.g1_T_flange))
                
            else:
                self.g0.set_tcp_transform(self.initial_world_T_g0)
                self.g1.set_tcp_transform(self.initial_world_T_g1)
            print("here")
            # Attach peg to gripper
            world_T_part = self.initial_world_T_g0.multiply(self.g0_T_part)
            self.p.set_transform(world_T_part) # set peg close to gripper
            self.g0.attach(self.p, True) # then attach
            
            # Get part to flange transform
            # Attach hole to other gripper
            world_T_hole = self.initial_world_T_g1.multiply(self.g1_T_hole)
            # self.h.set_transform(world_T_hole) # set peg close to gripper
            # self.g1.attach(self.h, True) # then attach
            
            # Overriding for single-handed robot
            self.world_T_hole = Transform(np.array([0, 50, init_height - 220, np.pi/2.0, 0, 0]))
            self.h.set_transform(self.world_T_hole)
                
            self.goal_hole_T_peg = Transform(np.array([0, 34, 50, 0, 0, 0]))

            if(not self.real_robot):
                # Set-up vel control of the peg
                r0_ee_T_world = self.r0.get_transform().invert()
                self.r0_ee_T_tcp = r0_ee_T_world.multiply(self.g0.get_transform())
                r0_pos_offset = self.r0_ee_T_tcp.get_values()[:3]
                self.r0_ee_T_tcp.identity()
                self.r0_ee_T_tcp.set_position(pyatk.Vector(r0_pos_offset[0], r0_pos_offset[1], r0_pos_offset[2]+100)) # +100 to include peg length
                self.p.create_vel_controlled_tool(True, self.force_limit, self.r0_ee_T_tcp)
                
                # # Set-up vel control of the hole
                # r1_ee_T_world = self.r1.get_transform().invert()
                # self.r1_ee_T_tcp = r1_ee_T_world.multiply(self.g1.get_transform())
                # r1_pos_offset = self.r1_ee_T_tcp.get_values()[:3]
                # self.r1_ee_T_tcp.identity()
                # self.r1_ee_T_tcp.set_position(pyatk.Vector(r1_pos_offset[0], r1_pos_offset[1], r1_pos_offset[2]+100)) # +100 to include hole length
                # self.h.create_vel_controlled_tool(True, self.force_limit, self.r1_ee_T_tcp)
            

            # Initialize gains
            self.r0_gains = AdmittanceGains()
            self.r1_gains = AdmittanceGains()
        
            # Create some buffer for peg pose, link_6 rot
            self.r0_state = RobotState(self.r0, self.g0, self.p, self.f0)
            self.r1_state = RobotState(self.r1, self.g1, self.h, self.f1)
            
            # Update the link6 pose from identity to what is in simulation
            self.r0_state = self.update_buffer(self.r0_state, self.time_step, update_vel=False)
            self.r1_state = self.update_buffer(self.r1_state, self.time_step, update_vel=False)
            
            # Save the initial state to use for later
            self.init_r0_state = self.r0_state.copy()
            self.init_r1_state = self.r1_state.copy()

            # Calibrate FT sensors
            print("Calibrating force sensors...")
            self.r0_state = self.calibrate_force(self.r0_state)
            self.r1_state = self.calibrate_force(self.r1_state)
            print("Done calibrating force sensors")

            # RL setup
            self.world_origin = np.array([0.00, 0, 0.18])
            self.world_origin_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            self.translation_limit = 0.05 # in m
            self.rotation_limit = 3 / 180 *np.pi # in rad
            self.xyz_safe_upper_limit = self.world_origin + np.array([0.12, 0.12, 0.6])
            self.xyz_safe_lower_limit = self.world_origin + np.array([-0.12, -0.12, -0.18])
            self.rotation_safe_limit = 5 / 180 *np.pi

            self.kp_low = np.array([10, 10, 10, 10, 10, 10])
            self.kp_high = np.array([1000, 1000, 1000, 1000, 1000, 1000])


            # Domain-randomization
            self.rotation_randomization = self.use_rotation_randomization
            if self.rotation_randomization:
                self.reset_pos_noise = np.array([20,20,10,5/180*np.pi,5/180*np.pi,5/180*np.pi]) #in mm and rad
                self.pose_uncertainty_level = np.array([6, 6, 6, 0, 0, 0])/UNIT_OFFSET # in m and rad
                self.pose_uncertainty_level[3:6] = np.array([0/180*np.pi, 10/180*np.pi, 0/180*np.pi])
            else:
                self.reset_pos_noise = np.array([20,20,10,0,0,0]) #in mm
                self.pose_uncertainty_level = np.array([6,6,6,0,0,0])/UNIT_OFFSET # in m
            self.nominal_hole_offset = np.random.uniform(-self.pose_uncertainty_level, self.pose_uncertainty_level)
            self.max_step = 20 
            
        # Randomly sample init pose
        peg_noise = Transform(np.random.uniform(-self.reset_pos_noise, self.reset_pos_noise))
        self.nominal_hole_offset = np.random.uniform(-self.pose_uncertainty_level, self.pose_uncertainty_level)
        
        self.step(np.zeros(24))
        
        if(self.real_robot):
            self.move_to_flange_transform(self.r0, self.initial_world_T_g0.multiply(self.g0_T_flange))
            # self.move_to_flange_transform(self.r1, self.initial_world_T_g1.multiply(self.g1_T_flange))
        else:
            self.p.create_vel_controlled_tool(False) # clear control tool
            self.g0.attach(self.p, False)  # not to attach at first
            self.g0.set_tcp_transform(self.initial_world_T_g0)
            world_T_part = peg_noise.multiply(self.initial_world_T_g0.multiply(self.g0_T_part))
            self.p.set_transform(world_T_part)  # set peg close to gripper
            self.g0.attach(self.p, True)  # then attach
            self.p.create_vel_controlled_tool(True, self.force_limit, self.r0_ee_T_tcp) # set control tool
            
            # self.h.create_vel_controlled_tool(False) # clear control tool
            # self.g1.attach(self.h, False)  # not to attach at first
            # self.g1.set_tcp_transform(self.initial_world_T_g1)
            # world_T_hole = self.initial_world_T_g1.multiply(self.g1_T_hole)
            # self.h.set_transform(world_T_hole)  # set peg close to gripper
            # self.g1.attach(self.h, True)  # then attach
            # self.h.create_vel_controlled_tool(True, self.force_limit, self.r1_ee_T_tcp) # set control tool        
        
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

        # Step once to get state
        action = np.zeros(24)
        self.step(action)
        
        return self.get_RL_state()



# register(
#     id="multiarm_tip_assembly-v0",
#     entry_point=AssemblyMultiArm,
# )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-robot",
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

    args = parser.parse_args()
    return args
    
def run_assembly(args):
    env = AssemblyMultiArm(render=args.render, real_robot=args.real_robot, is_remote=args.is_remote)
    for _ in range(100):
        print("Resetting")
        env.reset()
        for i in range(100):
            init_time = time.time()
            action = np.zeros(24)
            action[2] = -0.1
            state, reward, done, info = env.step(action)
            
if __name__ == '__main__':
    # Cannot launch araas and use it in the same proces
    # file_dir = pathlib.Path(__file__).parent.resolve()
    # pyaraas.launch(os.path.join(file_dir, "apa_workcells"))
    args = parse_args()
    if(args.real_robot):
        run_assembly(args)
    else:
        run_assembly(args)