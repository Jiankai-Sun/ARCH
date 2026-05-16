import pyatk
from pyatk import Transform, Vector, Part, Actor
from pathlib import Path
import time
import numpy as np
import gym
from gym import spaces,register
from scipy.spatial.transform import Rotation as R
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple
import copy


UNIT_T = Transform(np.array([0, 0, 0, 0, 0, 0]))
H = 10
UNIT_OFFSET = 1000

def get_rotation_matrix(t:np.array):
    return R.from_euler("zyx", t[3:]).as_matrix()
    
    
def get_pose_from_transform(transform:Transform):
    pose = np.array(transform.get_values())
    pose[0:3] = pose[0:3] / UNIT_OFFSET # mm to m
    return pose
    
@dataclass
class RobotState:
    robot: Actor
    tool_part: Part
    tool_pose: np.array = np.zeros(6)
    tool_vel: np.array = np.zeros(6)
    link6_pose: np.array = np.zeros(6)
    force_history: np.array = np.zeros([H, 6])
    world_force: np.array = np.zeros(6)

    def copy(self):
        return RobotState(
            self.robot,
            self.tool_part,
            copy.deepcopy(self.tool_pose),
            copy.deepcopy(self.tool_vel),
            copy.deepcopy(self.link6_pose),
            copy.deepcopy(self.force_history),
            copy.deepcopy(self.world_force)
        )
        
    
@dataclass
class AdmittanceGains:
    M: np.array = np.array([10, 10, 10, 10, 10, 10])
    kp: np.array = np.array([100, 100, 100, 100, 100, 100])
    kd: np.array = np.array([100, 100, 100, 100, 100, 100])
    force_limit: np.array = np.array([50, 50, 50, 1, 1, 1])

class AssemblyMultiArm(gym.Env):
    def __init__(self, friction=0.2, max_force=100, dt=1./100.0, max_v=20, max_rad=1, mass=1, gap=10,
                 render=False, verbose=False, 
                 use_rotation_randomization=False, **kwargs):
        
        
        print("Initting pyatk")
        # Init the pyatk
        if(render):
            pyatk.init(True, addr="127.0.0.1")
            # pyatk.init(True, addr="10.140.68.92")
        else:
            pyatk.init(False, addr="127.0.0.1")
     
        # Set the dir
        current_dir = Path().resolve()
        pyatk_dir = str(current_dir.joinpath("DC_MA/Sample"))
        pyatk.set_project_dir(pyatk_dir)
        # Load workcell (UR10e + gripper)
        gravity = pyatk.Vector(0, 0, 0)
        self.w = pyatk.load_workcell("MultiTipAssembly", gravity=gravity)
        
        
        # Init poses
        init_height = 600
        self.g0_T_part = Transform(np.array([0, 0, 0, 0, 0, 0]))
        self.g1_T_hole = Transform(np.array([0, -50, 0, -np.pi/2.0, 0, 0]))
        self.initial_world_T_g0 = Transform(np.array([0.0, 0.0, init_height, 0, np.pi/2.0, 0]))
        self.initial_world_T_g1 = Transform(np.array([-230, 0.0, init_height, 0, -np.pi/2.0, 0]))
        
        # Dynamic parameters
        self.tool_friction = friction
        self.max_force = max_force
        self.max_v = np.array([max_v, max_v, max_v, max_rad, max_rad, max_rad])
        self.time_step = dt
        self.step_time_step = 30*self.time_step
        if self.step_time_step < self.time_step:
            self.step_time_step = self.time_step # just in case
            
        # Get all parts (robot, peg, hole)
        self.p = self.w.add_part("square_peg_40", "peg0", UNIT_T)
        self.p.set_friction(self.tool_friction)
        self.p.set_mass(mass)
        self.p.set_margin(0.001)
        
        self.h = self.setup_hole(gap)
        
        self.r0 = self.w.get_robot("UR10-0")
        self.r1 = self.w.get_robot("UR10-1")
        
        self.g0 = self.w.get_gripper("Rq85-0")
        self.g0.set_tcp_transform(self.initial_world_T_g0)
        
        self.g1 = self.w.get_gripper("Rq140-0")
        self.g1.set_tcp_transform(self.initial_world_T_g1)
        
        # Attach peg to gripper
        self.g0.attach(self.p, False) # not to attach at first
        world_T_part = self.initial_world_T_g0.multiply(self.g0_T_part)
        self.p.set_transform(world_T_part) # set peg close to gripper
        self.g0.attach(self.p, True) # then attach
        
        # Attach hole to other gripper
        self.g1.attach(self.h, False) # not to attach at first
        world_T_hole = self.initial_world_T_g1.multiply(self.g1_T_hole)
        self.h.set_transform(world_T_hole) # set peg close to gripper
        self.g1.attach(self.h, True) # then attach
            
        self.goal_hole_T_peg = Transform(np.array([0, 34, 50, 0, 0, 0]))

        # Set-up vel control of the peg
        r0_ee_T_world = self.r0.get_transform().invert()
        self.r0_ee_T_tcp = r0_ee_T_world.multiply(self.g0.get_transform())
        r0_pos_offset = self.r0_ee_T_tcp.get_values()[:3]
        self.r0_ee_T_tcp.identity()
        self.r0_ee_T_tcp.set_position(pyatk.Vector(r0_pos_offset[0], r0_pos_offset[1], r0_pos_offset[2]+100)) # +100 to include peg length
        self.p.create_vel_controlled_tool(True, self.max_force, self.r0_ee_T_tcp)
        
        # Set-up vel control of the hole
        r1_ee_T_world = self.r1.get_transform().invert()
        self.r1_ee_T_tcp = r1_ee_T_world.multiply(self.g1.get_transform())
        r1_pos_offset = self.r1_ee_T_tcp.get_values()[:3]
        self.r1_ee_T_tcp.identity()
        self.r1_ee_T_tcp.set_position(pyatk.Vector(r1_pos_offset[0], r1_pos_offset[1], r1_pos_offset[2]+100)) # +100 to include hole length
        self.h.create_vel_controlled_tool(True, self.max_force, self.r1_ee_T_tcp)

        # Initialize gains
        self.r0_gains = AdmittanceGains()
        self.r1_gains = AdmittanceGains()
    
        # Create some buffer for peg pose, link_6 rot
        self.r0_state = RobotState(self.r0, self.p)
        self.r1_state = RobotState(self.r1, self.h)
        
        # Update the link6 pose from identity to what is in simulation
        self.r0_state = self.update_buffer(self.r0_state, update_vel=False)
        self.r1_state = self.update_buffer(self.r1_state, update_vel=False)
        
        # Save the initial state to use for later
        self.init_r0_state = self.r0_state.copy()
        self.init_r1_state = self.r1_state.copy()

        # Calibrate FT sensor
        # self.r0_state = self.FT_calibration(self.init_r0_state, self.r0_state, self.r0_gains)
        # self.r1_state = self.FT_calibration(self.init_r1_state, self.r1_state, self.r1_gains)

        # RL setup
        self.world_origin = np.array([0.00, 0, 0.18])
        self.translation_limit = 0.05 # in m
        self.rotation_limit = 3 / 180 *np.pi # in rad
        self.xyz_safe_upper_limit = self.world_origin + np.array([0.12, 0.12, 0.6])
        self.xyz_safe_lower_limit = self.world_origin + np.array([-0.12, -0.12, -0.18])
        self.rotation_safe_limit = 5 / 180 *np.pi

        self.kp_low = np.array([10, 10, 10, 10, 10, 10])
        self.kp_high = np.array([1000, 1000, 1000, 1000, 1000, 1000])

        action_bound = 1
        action_dim = 12 * 2 # two robots
        action_high = np.array([action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        obs_bound = 1
        obs_dim = 18
        obs_high = np.array([obs_bound] * obs_dim)
        self.observation_space = spaces.Box(-obs_high, obs_high)
        self.verbose = verbose

        # Domain-randomization
        self.rotation_randomization = use_rotation_randomization
        if self.rotation_randomization:
            self.reset_pos_noise = np.array([20,20,10,5/180*np.pi,5/180*np.pi,5/180*np.pi]) #in mm and rad
            self.pose_uncertainty_level = np.array([6, 6, 6, 0, 0, 0])/UNIT_OFFSET # in m and rad
            self.pose_uncertainty_level[3:6] = np.array([0/180*np.pi, 10/180*np.pi, 0/180*np.pi])
        else:
            self.reset_pos_noise = np.array([20,20,10,0,0,0]) #in mm
            self.pose_uncertainty_level = np.array([6,6,6,0,0,0])/UNIT_OFFSET # in m
        self.nominal_hole_offset = np.random.uniform(-self.pose_uncertainty_level, self.pose_uncertainty_level)
        self.steps = 0
        self.max_step = 20

        self.reset()  

    def setup_hole(self, gap:float):

        hole_size = 40+gap
        
        h = self.w.add_part("square_hole_part_40_120", "hole_4", Transform(np.array([0,50,-20,np.pi/2,0,0])))
        h.set_collision_model(True, True)  # collision, cancave
        h.set_friction(self.tool_friction)
        h.set_margin(0.001)
        
        h1 = self.w.add_part("square_hole_part_40_120", "hole_0", Transform(np.array([hole_size,0,0,0,0,np.pi/2])))
        h1.set_collision_model(True, True)  # collision, cancave
        h1.set_friction(self.tool_friction)
        h1.set_margin(0.001)
        h.link(h1, True)
        
        h2 = self.w.add_part("square_hole_part_40_120", "hole_1",Transform(np.array([-hole_size,0,0,0,0,np.pi/2])))
        h2.set_collision_model(True, True)  # collision, cancave
        h2.set_friction(self.tool_friction)
        h2.set_margin(0.001)
        h.link(h2, True)
        
        h3 = self.w.add_part("square_peg_40", "hole_2", Transform(np.array([0,hole_size,0,0,0,0])))
        h3.set_collision_model(True, True)  # collision, cancave
        h3.set_friction(self.tool_friction)
        h3.set_margin(0.001)
        h.link(h3, True)
        
        h4 = self.w.add_part("square_peg_40", "hole_3", Transform(np.array([0,-hole_size,0,0,0,0])))
        h4.set_collision_model(True, True)  # collision, cancave
        h4.set_friction(self.tool_friction)
        h4.set_margin(0.001)
        h.link(h4, True)
        
        return h
    
    def FT_calibration(self, initial_rstate:RobotState, current_rstate:RobotState, gains:AdmittanceGains, H=10):
        new_rstate = current_rstate.copy()
        FT_list = []
        for _ in range(H):
            control_cmd = np.zeros(13)
            control_cmd[:3] = initial_rstate.tool_pose[:3]
            control_cmd[3:7] = R.from_matrix(get_rotation_matrix(initial_rstate.tool_pose)).as_quat()
            new_rstate = self.update_control(control_cmd, new_rstate, gains)
            FT_list.append(copy.deepcopy(new_rstate.world_force))
        return new_rstate

    def render(self, **kwargs):
        pass
        
    def update_buffer(self, rstate:RobotState, update_vel=True) -> RobotState:
        new_rstate = rstate.copy()
        temp_pose = get_pose_from_transform(new_rstate.tool_part.get_transform())
        new_rstate.link6_pose = get_pose_from_transform(new_rstate.robot.get_flange_transform())
        if update_vel:
            vel = np.zeros(6)
            vel[:3] = (temp_pose[:3] - rstate.tool_pose[:3]) / self.time_step
            Rd = get_rotation_matrix(temp_pose) @ get_rotation_matrix(rstate.tool_pose).T
            axis_angle_vel = R.from_matrix(Rd).as_rotvec()/self.time_step
            vel[3:] = axis_angle_vel
            new_rstate.tool_vel = vel
        new_rstate.tool_pose = temp_pose
        return new_rstate

    def update_force(self, rstate:RobotState, raw_states:List[Transform]):
        new_rstate = rstate.copy()
        force = np.array([raw_states[1].x, raw_states[1].y, raw_states[1].z])
        torque = np.array([raw_states[2].x, raw_states[2].y, raw_states[2].z])
        world_force = get_rotation_matrix(new_rstate.link6_pose) @ force
        world_torque = get_rotation_matrix(new_rstate.link6_pose) @ torque
        H = new_rstate.force_history.shape[0]
        new_rstate.force_history[:H-1, :] = new_rstate.force_history[1:H, :]
        new_rstate.force_history[-1, :3] = world_force
        new_rstate.force_history[-1, 3:] = world_torque
        
        # average within the sliding window to de-noise the FT reading
        new_rstate.world_force[:3] = np.mean(new_rstate.force_history[:, :3], axis=0)
        new_rstate.world_force[3:] = np.mean(new_rstate.force_history[:, 3:], axis=0)
        
        return new_rstate

    def apply_control_limit(self, control_cmd):
        clipped_cmd = np.zeros(13)
        clipped_cmd[:3] = np.clip(control_cmd[:3], self.xyz_safe_lower_limit, self.xyz_safe_upper_limit)
        # cmd_rot = R.from_quat(control_cmd[3:7]).as_rotvec()
        # cmd_rot_norm = np.linalg.norm(cmd_rot, ord=2) + 1e-3
        # clipped_cmd[3:7] = R.from_rotvec(cmd_rot/cmd_rot_norm*np.clip(cmd_rot_norm, -self.rotation_safe_limit, self.rotation_safe_limit)).as_quat()
        
        # Override rotation limits
        clipped_cmd[3:7] = control_cmd[3:7]
        return clipped_cmd

    def update_control(self, 
                       control_cmd:List[float], 
                       rstate:RobotState, 
                       gains:AdmittanceGains) -> RobotState:
        
        # Admittance control
        # Clip the cmd to be safe
        # control_cmd = self.apply_control_limit(control_cmd)
        
        d_pos = control_cmd[:3] # in m
        d_quat = control_cmd[3:7]  # in quat
        d_rot = R.from_quat(d_quat).as_matrix()
        d_pos_vel = control_cmd[7:10]
        d_ori_vel = control_cmd[10:13]

        pos_error = rstate.tool_pose[:3] - d_pos       
        ori_error_rot = get_rotation_matrix(rstate.tool_pose) @ d_rot.T

        # from rot(9 dim) to error (3 dim), need to use axis-angle representation
        ori_error = R.from_matrix(ori_error_rot).as_rotvec()
        error = np.hstack([pos_error, ori_error])
        pos_error_dot = rstate.tool_vel[:3] - d_pos_vel
        ori_error_dot = rstate.tool_vel[3:] - d_ori_vel
        error_dot = np.hstack([pos_error_dot, ori_error_dot])

        # Feedback law: M*acc = F - kp e - kd e_dot
        world_force_clip = np.clip(rstate.world_force, -gains.force_limit, gains.force_limit)
        RHS = 1*world_force_clip - np.multiply(gains.kp, error) - np.multiply(gains.kd, error_dot)
        acc = np.divide(RHS, gains.M)
        world_vel_cmd = acc * self.time_step + rstate.tool_vel # maybe
        world_vel_cmd[0:3] = world_vel_cmd[0:3] * UNIT_OFFSET # mm to m
        world_vel_cmd = np.clip(0.8*world_vel_cmd, -self.max_v, self.max_v) # 0.8 to reduce vel response to make simulation stable
        link6_rot = get_rotation_matrix(rstate.link6_pose)
        action = (np.block([[link6_rot, np.zeros([3,3])],[np.zeros([3,3]), link6_rot]]).T @ world_vel_cmd).tolist()
        raw_states = rstate.tool_part.set_vel(Vector(action[0], action[1], action[2]), Vector(action[3], action[4], action[5]),
                               self.time_step)
        new_rstate = self.update_buffer(rstate)
        new_rstate = self.update_force(new_rstate, raw_states)
        return new_rstate

    def get_RL_state(self):
        rstate_vecs = []
        for rstate, gains in [(self.r0_state, self.r0_gains), (self.r1_state, self.r1_gains)]:
            p_pose = copy.deepcopy(rstate.tool_pose)
            p_pose[:3] = p_pose[:3] - self.world_origin + self.nominal_hole_offset[:3]
            p_pose[3:6] = p_pose[3:6] + self.nominal_hole_offset[3:6]
            p_vel = rstate.tool_vel.copy()
            world_force = rstate.world_force.copy()
            world_force = np.clip(world_force, -gains.force_limit, gains.force_limit)
            world_force = np.divide(world_force, gains.force_limit)
            rstate_vecs+=[p_pose, p_vel, world_force]
            
        return np.concatenate(rstate_vecs)

    def step_single(self, rstate:RobotState, gains:AdmittanceGains, action: List[float]) -> Tuple[RobotState, AdmittanceGains]:

        # Get delta_motion from action
        delta = np.zeros(6)
        delta[:3] = action[:3] / (np.linalg.norm(action[:3], ord=2) + 1e-3) * self.translation_limit
        delta[3:6] = action[3:6] / (np.linalg.norm(action[3:6], ord=2) + 1e-3) * self.rotation_limit
        delta_rotm = R.from_euler('zyx', delta[3:6]).as_matrix()
        
        # Get gains from action
        kp = (self.kp_high + self.kp_low) / 2 + np.multiply(action[6:12],(self.kp_high - self.kp_low) / 2 )
        kd = 4 * np.sqrt(np.multiply(gains.M, kp)) # 4 to make thw system overdamped
        
        new_gains = AdmittanceGains(
            gains.M, kp, kd, gains.force_limit
        )
    
        # init control cmd
        control_cmd = np.zeros(13)
        control_cmd[:3] = rstate.tool_pose[:3] + delta[:3]
        control_cmd[3:7] = R.from_matrix(get_rotation_matrix(rstate.tool_pose)@delta_rotm).as_quat()
        
        # Control peg
        for _ in range(int(self.step_time_step/self.time_step)):
            rstate = self.update_control(control_cmd, rstate, new_gains)  # control and step
            
        return rstate, new_gains
            
            
    def step(self, action):
        
        self.steps+=1
        
        r0_action = action[:12]
        r1_action = action[12:]

        self.r0_state, self.r0_gains = self.step_single(self.r0_state, self.r0_gains, r0_action)
        self.r1_state, self.r1_gains = self.step_single(self.r1_state, self.r1_gains, r1_action)
        
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
        reward = -dist
        return reward, False

    def get_number_of_agents(self):
        return 1

    def reset(self, seed=None, options=None):
        # Randomly sample init pose
        peg_noise = Transform(np.random.uniform(-self.reset_pos_noise, self.reset_pos_noise))
        self.nominal_hole_offset = np.random.uniform(-self.pose_uncertainty_level, self.pose_uncertainty_level)
        
        # ignore for now
        self.step(np.zeros(24))
        
        self.p.create_vel_controlled_tool(False) # clear control tool
        self.g0.attach(self.p, False)  # not to attach at first
        self.g0.set_tcp_transform(self.initial_world_T_g0)
        world_T_part = peg_noise.multiply(self.initial_world_T_g0.multiply(self.g0_T_part))
        self.p.set_transform(world_T_part)  # set peg close to gripper
        self.g0.attach(self.p, True)  # then attach
        self.p.create_vel_controlled_tool(True, self.max_force, self.r0_ee_T_tcp) # set control tool
        
        # # Not sure why, but this causes segfault
        self.h.create_vel_controlled_tool(False) # clear control tool
        self.g1.attach(self.h, False)  # not to attach at first
        self.g1.set_tcp_transform(self.initial_world_T_g1)
        world_T_hole = self.initial_world_T_g1.multiply(self.g1_T_hole)
        self.h.set_transform(world_T_hole)  # set peg close to gripper
        self.g1.attach(self.h, True)  # then attach
        self.h.create_vel_controlled_tool(True, self.max_force, self.r1_ee_T_tcp) # set control tool        
                
        # Reset robot states
        self.r0_state = self.update_buffer(self.r0_state, update_vel=False)
        self.r1_state = self.update_buffer(self.r1_state, update_vel=False)
        
        
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
        self.steps = 0
        
        return self.get_RL_state()



register(
    id="multiarm_tip_assembly-v0",
    entry_point=AssemblyMultiArm,
)

if __name__ == '__main__':
    env = AssemblyMultiArm(render=True)

    for _ in range(100):
        print("resetting")
        env.reset()
        for i in range(100):
            init_time = time.time()
            action = np.zeros(24)  
            action[0] = 1
            action[12] = -1
            state, reward, done, info = env.step(action)

