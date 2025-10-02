import pyatk
from pyatk import Transform, Vector
from pathlib import Path
import time
import numpy as np
import gym
from gym import spaces,register
from scipy.spatial.transform import Rotation as R
import random
class Assembly(gym.Env):
    def __init__(self, friction=0.2, max_force=100, dt=1./100.0, max_v=20, max_rad=1, mass=1, gap=1.5,
                 render=False, verbose=False, 
                 use_rotation_randomization=False, **kwargs):
        # Init the pyatk

        if(render):
            pyatk.init(True, addr="10.140.68.92")
        else:
            pyatk.init(False, addr="127.0.0.1")
     
        # Set the dir
        current_dir = Path().resolve()
        pyatk_dir = str(current_dir.joinpath("DC_SA/Sample"))
        pyatk.set_project_dir(pyatk_dir)
        # Load workcell (UR10e + gripper)
        gravity = pyatk.Vector(0, 0, 0)
        self.w = pyatk.load_workcell("UR10e", gravity=gravity)
        # Init poses
        init_height = 300
        self.part_initial_pose = [520, 0.0, init_height - 101, 0, 0, 0]
        self.hole_initial_pose = [520, 0.0, init_height - 220, 0, 0, 0]
        self.robot_initial_pose = [520, 0.0, init_height, 0, 0, 0]
        # Dynamic parameters
        self.tool_friction = friction
        self.max_force = max_force
        self.max_v = np.array([max_v, max_v, max_v, max_rad, max_rad, max_rad])
        self.time_step = dt
        self.step_time_step = 30*self.time_step
        if self.step_time_step < self.time_step:
            self.step_time_step = self.time_step # just in case
        # Get all parts (robot, peg, hole)
        self.p = self.w.add_part("square_peg_40", "peg0", Transform(self.part_initial_pose))
        self.p.set_friction(self.tool_friction)
        self.p.set_mass(mass)
        self.p.set_margin(0.001)

        hole_size = 40+gap
        
        self.h1 = self.w.add_part("square_hole_part_40_120", "hole_0", Transform(self.hole_initial_pose+np.array([hole_size,0,0,0,0,np.pi/2])))
        self.h1.set_collision_model(True, True)  # collision, cancave
        self.h1.set_friction(self.tool_friction)
        self.h1.set_margin(0.001)
        self.h2 = self.w.add_part("square_hole_part_40_120", "hole_1", Transform(self.hole_initial_pose+np.array([-hole_size,0,0,0,0,np.pi/2])))
        self.h2.set_collision_model(True, True)  # collision, cancave
        self.h2.set_friction(self.tool_friction)
        self.h2.set_margin(0.001)
        self.h3 = self.w.add_part("square_peg_40", "hole_2", Transform(self.hole_initial_pose+np.array([0,hole_size,0,0,0,0])))
        self.h3.set_collision_model(True, True)  # collision, cancave
        self.h3.set_friction(self.tool_friction)
        self.h3.set_margin(0.001)
        self.h4 = self.w.add_part("square_peg_40", "hole_3", Transform(self.hole_initial_pose+np.array([0,-hole_size,0,0,0,0])))
        self.h4.set_collision_model(True, True)  # collision, cancave
        self.h4.set_friction(self.tool_friction)
        self.h4.set_margin(0.001)
        self.h5 = self.w.add_part("square_hole_part_40_120", "hole_4", Transform(self.hole_initial_pose+np.array([0,50,-20,np.pi/2,0,0])))
        self.h5.set_collision_model(True, True)  # collision, cancave
        self.h5.set_friction(self.tool_friction)
        self.h5.set_margin(0.001)

        self.r = self.w.get_robot("UR10e-0")

        self.g = self.w.get_gripper("Rq85-0")
        self.g.set_tcp_transform(Transform(self.robot_initial_pose))
        
        # Attach peg to hole
        self.g.attach(self.p, False) # not to attach at first
        self.p.set_transform(Transform(self.part_initial_pose)) # set peg close to gripper
        self.g.attach(self.p, True) # then attach

        # Set-up vel control of the peg
        ee_T_world = self.r.get_transform().invert()
        self.ee_T_tcp = ee_T_world.multiply(self.g.get_transform())
        pos_offset = self.ee_T_tcp.get_values()[:3]
        self.ee_T_tcp.identity()
        self.ee_T_tcp.set_position(pyatk.Vector(pos_offset[0], pos_offset[1], pos_offset[2]+100)) # +100 to include peg length
        self.p.create_vel_controlled_tool(True, self.max_force, self.ee_T_tcp)

        # Admittance control gains
        self.M = np.array([10, 10, 10, 10, 10, 10])
        self.kp = np.array([100, 100, 100, 100, 100, 100])
        self.kd = np.array([100, 100, 100, 100, 100, 100])
        self.force_limit = np.array([50, 50, 50, 1, 1, 1])
        # self.force_limit = np.array([30, 30, 30, 0.3, 0.3, 0.3])
        
        # For unit change
        self.unit_offset = 1000
        # Create some buffer for peg pose, link_6 rot
        self.p_pose = np.zeros(6)
        self.p_vel = np.zeros(6)
        self.p_rot = np.zeros([3,3])
        self.link6_rot = np.zeros([3,3])
        self.world_force = np.zeros(6)
        self.world_force_offset = np.zeros(6)
        self.update_buffer(update_vel=False)
        self.init_p_pos = self.p_pose[0:3].copy()
        self.init_p_rot = self.p_rot.copy()

        # Calibrate FT sensor
        H = 10
        self.force_history = np.zeros([H, 6])
        # self.FT_calibration()

        # RL setup
        self.world_origin = np.array([0.52, 0, 0.18])
        self.translation_limit = 0.05 # in m
        self.rotation_limit = 3 / 180 *np.pi # in rad
        self.xyz_safe_upper_limit = self.world_origin + np.array([0.06, 0.06, 0.2])
        self.xyz_safe_lower_limit = self.world_origin + np.array([-0.06, -0.06, -0.1])
        self.rotation_safe_limit = 5 / 180 *np.pi

        self.kp_low = np.array([10, 10, 10, 10, 10, 10])
        self.kp_high = np.array([1000, 1000, 1000, 1000, 1000, 1000])

        action_bound = 1
        action_dim = 12 # since we include both traj and the gains
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
            self.pose_uncertainty_level = np.array([6, 6, 6, 0, 0, 0])/self.unit_offset # in m and rad
            self.pose_uncertainty_level[3:6] = np.array([0/180*np.pi, 10/180*np.pi, 0/180*np.pi])
        else:
            self.reset_pos_noise = np.array([20,20,10,0,0,0]) #in mm
            self.pose_uncertainty_level = np.array([6,6,6,0,0,0])/self.unit_offset # in m
        self.nominal_hole_offset = np.random.uniform(-self.pose_uncertainty_level, self.pose_uncertainty_level)
        self.steps = 0
        self.max_step = 20

        self.reset()

    def FT_calibration(self, H=10):
        FT_list = []
        for _ in range(H):
            control_cmd = np.zeros(13)
            control_cmd[:3] = self.init_p_pos[0:3]
            control_cmd[3:7] = R.from_matrix(self.init_p_rot).as_quat()
            self.update_control(control_cmd)
            FT_list.append(self.world_force.copy())
        self.world_force_offset = np.average(np.array(FT_list), axis=0)

    def render(self, **kwargs):
        pass
        
    def get_pose_from_transform(self, transform):
        pos = transform.position
        quat = transform.rotation
        pos_np = np.array([pos.x, pos.y, pos.z])
        quat_np = np.array([quat.x, quat.y, quat.z, quat.w])
        rot = R.from_quat(quat_np).as_matrix()
        pose = np.array(transform.get_values())
        pose[0:3] = pose[0:3] / self.unit_offset # mm to m
        return pose, rot

    def update_buffer(self, update_vel=True):
        print("Updating tool pose: "+str(self.p.tool_part.get_transform().get_values()))
        temp_p_pose, temp_p_rot = self.get_pose_from_transform(self.p.get_transform())
        _, self.link6_rot = self.get_pose_from_transform(self.r.get_flange_transform())
        if update_vel:
            vel = np.zeros(6)
            vel[:3] = (temp_p_pose[:3] - self.p_pose[:3]) / self.time_step
            # Rd @ R0 = R1
            Rd = temp_p_rot @ self.p_rot.T
            axis_angle_vel = R.from_matrix(Rd).as_rotvec()/self.time_step
            vel[3:] = axis_angle_vel
            self.p_vel = vel
        self.p_pose = temp_p_pose
        self.p_rot = temp_p_rot

    def update_gain(self, M=10*np.ones(6), kp=100*np.ones(6), kd=100*np.ones(6)):
        self.M = M
        self.kp = kp
        self.kd = kd

    def update_force(self, raw_states):
        force = np.array([raw_states[1].x, raw_states[1].y, raw_states[1].z])
        torque = np.array([raw_states[2].x, raw_states[2].y, raw_states[2].z])
        world_force = self.link6_rot @ force
        world_torque = self.link6_rot @ torque
        H = self.force_history.shape[0]
        self.force_history[:H-1, :] = self.force_history[1:H, :]
        self.force_history[-1, :3] = world_force
        self.force_history[-1, 3:] = world_torque
        # average within the sliding window to de-noise the FT reading
        self.world_force[:3] = np.mean(self.force_history[:, :3], axis=0)
        self.world_force[3:] = np.mean(self.force_history[:, 3:], axis=0)
        


    def apply_control_limit(self, control_cmd):
        clipped_cmd = np.zeros(13)
        clipped_cmd[:3] = np.clip(control_cmd[:3], self.xyz_safe_lower_limit, self.xyz_safe_upper_limit)
        cmd_rot = R.from_quat(control_cmd[3:7]).as_rotvec()
        cmd_rot_norm = np.linalg.norm(cmd_rot, ord=2) + 1e-3
        clipped_cmd[3:7] = R.from_rotvec(cmd_rot/cmd_rot_norm*np.clip(cmd_rot_norm, -self.rotation_safe_limit, self.rotation_safe_limit)).as_quat()
        return clipped_cmd

    def update_control(self, control_cmd):
        # Admittance control
        # Clip the cmd to be safe
        control_cmd = self.apply_control_limit(control_cmd)
        d_pos = control_cmd[:3] # in m
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
        world_force_clip = np.clip(self.world_force, -self.force_limit, self.force_limit)
        # # same tests for physx
        RHS = 1*world_force_clip - np.multiply(self.kp, error) - np.multiply(self.kd, error_dot)
        acc = np.divide(RHS, self.M)
        world_vel_cmd = acc * self.time_step + self.p_vel # maybe
        world_vel_cmd[0:3] = world_vel_cmd[0:3] * self.unit_offset # mm to m
        # ori feedback testing
        # world_vel_cmd = np.array([0,0,-10,0,0,0])
        world_vel_cmd = np.clip(0.8*world_vel_cmd, -self.max_v, self.max_v) # 0.8 to reduce vel response to make simulation stable
        action = (np.block([[self.link6_rot, np.zeros([3,3])],[np.zeros([3,3]), self.link6_rot]]).T @ world_vel_cmd).tolist()
        # action = [0,0,10,0,0,0]
        # print("cmd:",world_vel_cmd)
        raw_states = self.p.set_vel(Vector(action[0], action[1], action[2]), Vector(action[3], action[4], action[5]),
                               self.time_step)
        # time.sleep(self.time_step)
        self.update_buffer()
        self.update_force(raw_states)

    def get_RL_state(self):
        p_pose = self.p_pose.copy()
        p_pose[:3] = p_pose[:3] - self.world_origin + self.nominal_hole_offset[:3]
        p_pose[3:6] = p_pose[3:6] + self.nominal_hole_offset[3:6]
        p_vel = self.p_vel.copy()
        world_force = self.world_force.copy()

        # normalize force
        world_force = np.clip(world_force, -self.force_limit, self.force_limit)
        world_force = np.divide(world_force, self.force_limit)
        # Domain randomization on state
        # Ignore for now
        state = np.concatenate([p_pose, p_vel, world_force])
        return state

    def step(self, action):
        self.steps+=1
        # Get delta_motion from action
        delta = np.zeros(6)
        delta[:3] = action[:3] / (np.linalg.norm(action[:3], ord=2) + 1e-3) * self.translation_limit
        delta[3:6] = action[3:6] / (np.linalg.norm(action[3:6], ord=2) + 1e-3) * self.rotation_limit
        delta_rotm = R.from_euler('xyz', delta[3:6]).as_matrix()

        # Get gains from action
        kp = (self.kp_high + self.kp_low) / 2 + np.multiply(action[6:12],(self.kp_high - self.kp_low) / 2 )
        kd = 4 * np.sqrt(np.multiply(self.M, kp)) # 4 to make thw system overdamped
        self.update_gain(self.M, kp, kd)

        # init control cmd
        control_cmd = np.zeros(13)
        control_cmd[:3] = self.p_pose[:3] + delta[:3]
        control_cmd[3:7] = R.from_matrix(self.p_rot@delta_rotm).as_quat()

        for _ in range(int(self.step_time_step/self.time_step)):
            self.update_control(control_cmd)  # control and step
        
        state = self.get_RL_state()
        reward, done = self.compute_reward(state, action)
        info = {}

        if(self.steps>=100):
            done = True

        terminated = done
        if self.verbose:
            print("----------------------------------")
            # print("offset", self.nominal_hole_offset)
            print("pos:", state[0:6])
            # print("vel:", state[6:9])
            print("force", np.multiply(state[12:18],self.force_limit))
            # print("reward", reward)
            print("delta",delta)
            print("kp", self.kp)
            time.sleep(0.4)
        return state, reward, done, info
    
    def compute_reward(self, state, action):
        # ignore this part for now
        # dist = np.linalg.norm(state[:3]-np.array([0,0,0.05]))
        dist = np.linalg.norm(state[:3] + np.array([0,0,0.1]))
        reward = -10*dist
        done = False
        if reward > -0.01:
            done = True
        return reward, done

    def get_number_of_agents(self):
        return 1

    def reset(self, seed=None, options=None):
        # Randomly sample init pose
        noise = np.random.uniform(-self.reset_pos_noise, self.reset_pos_noise)
        self.nominal_hole_offset = np.random.uniform(-self.pose_uncertainty_level, self.pose_uncertainty_level)
        # ignore for now
        self.step(np.zeros(12))
        self.p.create_vel_controlled_tool(False) # clear control tool

        self.g.attach(self.p, False)  # not to attach at first
        self.g.set_tcp_transform(Transform(self.robot_initial_pose+noise))
        self.p.set_transform(Transform(self.part_initial_pose+noise))  # set peg close to gripper
        self.g.attach(self.p, True)  # then attach

        self.p.create_vel_controlled_tool(True, self.max_force, self.ee_T_tcp) # set control tool

        # Clean buffers
        self.update_buffer(update_vel=False)
        self.p_vel = np.zeros(6)
        self.world_force = np.zeros(6)
        self.force_history = np.zeros([self.force_history.shape[0], 6])
        self.update_gain()

        # Step once to get state
        action = np.zeros(12)
        self.step(action)
        self.steps = 0
        return self.get_RL_state()



register(
    id="tip_assembly-v0",
    entry_point=Assembly,
)

if __name__ == '__main__':
    env = Assembly()
    # env2 = Assembly()
    # action = np.array([0, 0, -1, 0, 0, 0])
    # state, reward, terminated, done, info = env.step(action)
    env.reset()
    
    for i in range(20000):
        init_time = time.time()
        action = np.zeros(12)
        action[:6] = np.array([0, -0, 0, 1, 0, 0])
        state, reward, done, info = env.step(action)
        # state, reward, done, info = env2.step(action)
        # print("----------------------------------")
        # print("pos:", state[0:6])
        # print("vel:", state[6:9])
        # print("force", state[12:18])
        # print("reward", reward)
        # print(env.p_rot@env.init_p_rot.T)
        print(time.time() - init_time)
    # env.reset()
    # print("reset")
    # for i in range(100):
    #     action = np.array([0, 1, -1, 0, 0, 0])
    #     state, reward, done, info = env.step(action)
    #     print("----------------------------------")
    #     print("pos:", state[0:6])
    #     # print("vel:", state[6:9])
    #     # print("force", state[12:18])
    #     # print(env.p_rot@env.init_p_rot.T)