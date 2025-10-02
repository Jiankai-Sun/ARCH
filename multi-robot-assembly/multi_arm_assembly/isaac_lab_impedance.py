from __future__ import annotations

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different single-arm manipulators.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/arms.py

"""

"""Launch Isaac Sim Simulator first."""
import argparse
import os
from typing import List
from dataclasses import MISSING
import numpy as np
import gymnasium as gym

import omni.isaac.lab.utils.math as math_utils

from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.controllers import DifferentialIKControllerCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.assets.rigid_object import RigidObjectCfg
from multi_arm_assembly.utils import solve_ik
from omni.isaac.lab.controllers.differential_ik import DifferentialIKController
from omni.isaac.lab.controllers.joint_impedance import JointImpedanceController
"""Rest everything follows."""

import torch
import omni.isaac.core.utils.stage as stage_utils
import copy
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.actuators import ImplicitActuatorCfg, DCMotorCfg, IdealPDActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.math import compute_pose_error
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
import json
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
import math
from omni.isaac.lab.managers import EventTermCfg as EventTerm
import omni.isaac.lab.utils.string as string_utils
from typing import Sequence
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import logging
from omni.isaac.lab.sim import SimulationCfg

home_dir = os.getcwd() + '/../'


log = logging.getLogger()

def obj_to_str(d):
    return json.dumps(d)
   
NUM_ENVS = 1

r1_offset_pos = (0.8367000122070312, 0.6095999755859375, 0.02250)
r1_offset_rot = (0.7071068 , 0, 0, 0.7071068)

r2_offset_pos = (-0.8367000122070312, 0.6095999755859375, 0.02250)
r2_offset_rot = (0.7071068 , 0, 0, -0.7071068)

EFFORT_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
FINGER_JOINTS = [
    # "right_inner_finger_knuckle_joint",
    # "left_inner_finger_knuckle_joint",
    "left_outer_knuckle_joint",
]

M_DEFAULT_LIN = 20
M_DEFAULT_ANG = 20
KP_DEFAULT = 1000
KD_DEFAULT_LIN = 2 * np.sqrt(np.multiply(M_DEFAULT_LIN, KP_DEFAULT))
KD_DEFAULT_ANG = 2 * np.sqrt(np.multiply(M_DEFAULT_ANG, KP_DEFAULT))
FORCE_LIMIT_TRANS = 50
VELOCITY_LIMIT_TRANS = 20
VELOCITY_LIMIT_ROT = 1
TORQUE_LIMIT = 1
H = 10
SIM_DT = 1/120
SUBSTEPS = 1
DECIMATION = 5

@dataclass
class AdmittanceGains:
    M: np.array = torch.tensor([M_DEFAULT_LIN, M_DEFAULT_LIN, M_DEFAULT_LIN, M_DEFAULT_ANG, M_DEFAULT_ANG, M_DEFAULT_ANG])
    kp: np.array = torch.tensor([KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT])
    kd: np.array = torch.tensor([KD_DEFAULT_LIN, KD_DEFAULT_LIN, KD_DEFAULT_LIN, KD_DEFAULT_ANG, KD_DEFAULT_ANG, KD_DEFAULT_ANG])

    def to(self, device):
        return AdmittanceGains(
            self.M.to(device),
            self.kp.to(device),
            self.kd.to(device)
        )

CAPTURE_DEBUG_LOG = True
def log_list(name, vector:List[float]):
    if(CAPTURE_DEBUG_LOG):
        if(isinstance(vector, torch.Tensor)):
            vector = vector.cpu().detach().numpy().tolist()
        assert isinstance(vector, list)
        assert len(vector) == NUM_ENVS
        log.info(obj_to_str({name: vector}))
                
TASK = "medium_gear_insert"
# TASK = "large_gear_insert"
# TASK = "small_gear_insert"

def wxyz_to_xyzw(q):
    return [q[1], q[2], q[3], q[0]]

def xyzw_to_wxyz(q):
    return [q[3], q[0], q[1], q[2]]

@dataclass
class IPose():
    pos: torch.tensor
    quat: torch.tensor

    def multiply(self, p:IPose):
        return IPose(*math_utils.combine_frame_transforms(self.pos, self.quat, p.pos, p.quat))
    
    def invert(self):
        t12, q12 = math_utils.subtract_frame_transforms(self.pos, self.quat)
        return IPose(t12, q12)
    
    def to_vec(self):
        return torch.cat([self.pos, self.quat], dim=1)

    def to(self, device):
        return IPose(self.pos.to(device), self.quat.to(device))
WORLD_T_HOLE = IPose(pos=torch.Tensor([[0.150, 0.400, -0.027]]).cuda(), quat=torch.Tensor([[0.7071068,0.7071068,0,0]]).cuda())

quat = R.from_euler('zyx', [math.pi/2.0, 0, -math.pi][::-1]).as_quat()
quat2 = [quat[3], quat[0], quat[1], quat[2]] # xyzw to wxyz
goal_hole_T_med_gear_inv = IPose(torch.Tensor([[0.21718, 0.26470, 0.06166]]).cuda(), torch.Tensor([quat2]).cuda())
goal_hole_T_med_gear = goal_hole_T_med_gear_inv.invert()
WORLD_T_MEDGEAR_GOAL = WORLD_T_HOLE.multiply(goal_hole_T_med_gear)
WORLD_T_SMALLGEAR_GOAL = IPose(pos=torch.Tensor([[-0.03, 0, 0]]).cuda(), quat=torch.Tensor([[1, 0, 0, 0]]).cuda()).multiply(WORLD_T_MEDGEAR_GOAL)
WORLD_T_LARGEGEAR_GOAL = IPose(pos=torch.Tensor([[0.055, 0, 0]]).cuda(), quat=torch.Tensor([[1, 0, 0, 0]]).cuda()).multiply(WORLD_T_MEDGEAR_GOAL)
WORLD_T_GEAR_START = IPose(pos=torch.Tensor([[0, 0.01, 0.04]]).cuda(), quat=torch.Tensor([[1, 0, 0, 0]]).cuda()).multiply(WORLD_T_MEDGEAR_GOAL)

# Target grasp
OBJECT_T_FLANGE_QUAT = xyzw_to_wxyz(R.from_euler('xyz', [0, -np.pi/2.0, 0]).as_quat())
OBJECT_T_FLANGE = IPose(pos=torch.Tensor([[0, 0.0, -0.176]]).cuda(), quat=torch.Tensor([OBJECT_T_FLANGE_QUAT]).cuda())


@dataclass
class RobotState:
    init_world_force: torch.tensor = torch.zeros([NUM_ENVS, 3])
    init_world_torque: torch.tensor = torch.zeros([NUM_ENVS, 3])
    tool_pos: torch.tensor = torch.zeros([NUM_ENVS, 3])
    tool_quat: torch.tensor = torch.zeros([NUM_ENVS, 4])
    tool_vel: torch.tensor= torch.zeros([NUM_ENVS, 6])
    force_history: torch.tensor = torch.zeros([NUM_ENVS, H, 6])
    world_force: torch.tensor = torch.zeros([NUM_ENVS, 6])

    def copy(self):
        return RobotState(
            copy.deepcopy(self.init_world_force),
            copy.deepcopy(self.init_world_torque),
            copy.deepcopy(self.tool_pos),
            copy.deepcopy(self.tool_quat),
            copy.deepcopy(self.tool_vel),
            copy.deepcopy(self.force_history),
            copy.deepcopy(self.world_force)
        )
    
    def to(self, device):
        return RobotState(
            self.init_world_force.to(device),
            self.init_world_torque.to(device),
            self.tool_pos.to(device),
            self.tool_quat.to(device),
            self.tool_vel.to(device),
            self.force_history.to(device),
            self.world_force.to(device)
        )

@configclass
class DefaultSceneCfg(InteractiveSceneCfg):
    
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", 
            spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg  = AssetBaseCfg(prim_path="/World/Light", 
            spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)))
    
    replicate_physics=False

class MyAction(mdp.ActionTerm):
    
    _asset: Articulation

    def __init__(self, cfg: MyActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self.asset_name = self._asset.cfg.prim_path.split("/")[-1]
        
        self.translation_limit = 0.05*5 # in m
        self.rotation_limit = 3 / 180 *np.pi # in rad

        self.force_limit = FORCE_LIMIT_TRANS
        self.velocity_limit = VELOCITY_LIMIT_TRANS
        self.torque_limit = TORQUE_LIMIT
        
        self._scale = cfg.scale
        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._finger_joint_ids, self._finger_joint_names = self._asset.find_joints(self.cfg.finger_joint_names)
        self._num_joints = len(self._joint_ids)
        # parse the body index
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)

        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        # check if articulation is fixed-base
        # if fixed-base then the jacobian for the base is not computed
        # this means that number of bodies is one less than the articulation's number of bodies
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
        else:
            self._jacobi_body_idx = self._body_idx

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)


        self.reset()

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        
        # For debugging, go straight down
        actions = torch.zeros([NUM_ENVS, self.action_dim])
        # actions[:, 2] = -0.005 
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        self._processed_actions[:, 3:] = 0

        delta = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        delta[:, :3] = self._processed_actions[:, :3] / (torch.linalg.norm(self._processed_actions[:, :3], ord=2) + 1e-3) * self.translation_limit
        delta[:, 3:6] = self._processed_actions[:, 3:6] / (torch.linalg.norm(self._processed_actions[:, 3:6], ord=2) + 1e-3) * self.rotation_limit
        
        tool_trans, tool_quat = self._compute_frame_pose()
        delta_rotm = math_utils.matrix_from_euler(delta[:, 3:6], "XYZ")

        if(self.control_cmd is None):
            # Target absolute position
            log_list(f"{self.asset_name}_delta", delta)
            self.control_cmd = torch.zeros(self.num_envs, 13).to(self.device)
            self.control_cmd[:, :3] = tool_trans + delta[:, :3]

            delta_rotation = math_utils.quat_from_matrix(delta_rotm)
            self.control_cmd[:, 3:7] = math_utils.quat_mul(tool_quat, delta_rotation)

        



    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pose_w = self._asset.data.body_state_w[:, self._body_idx, :7]
        root_pose_w = self._asset.data.root_state_w[:, :7]
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        return ee_pose_b, ee_quat_b


    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._joint_ids]
        return jacobian

    def _apply_task_space_gains(self, delta_fingertip_pose,
                                fingertip_midpoint_linvel,
                                fingertip_midpoint_angvel):
        """Interpret PD gains as task-space gains. Apply to task-space error."""

        task_wrench = torch.zeros_like(delta_fingertip_pose)
        task_prop_gains = torch.ones((self.num_envs, 6)).cuda()*0.01
        task_deriv_gains = torch.ones((self.num_envs, 6)).cuda()*0.001

        # Apply gains to lin error components
        lin_error = delta_fingertip_pose[:, 0:3]
        task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * lin_error + \
                            task_deriv_gains[:, 0:3] * (0.0 - fingertip_midpoint_linvel)

        # Apply gains to rot error components
        rot_error = delta_fingertip_pose[:, 3:6]
        task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * rot_error + \
                            task_deriv_gains[:, 3:6] * (0.0 - fingertip_midpoint_angvel)

        return task_wrench

    def get_quat_diff(self, q1, q2):
        # Compute quat error (i.e., difference quat)
        # Reference: https://personal.utdallas.edu/~sxb027100/dock/quat.html
        current_quat_norm = math_utils.quat_mul(q1, math_utils.quat_conjugate(q1))[:, 0]  # scalar component
        current_quat_inv = math_utils.quat_conjugate(q1) / current_quat_norm.unsqueeze(-1)
        quat_error = math_utils.quat_mul(q2, current_quat_inv)
        return quat_error
    
    def get_pose_error(self, current_pos, current_quat, target_pos, target_quat):

        # Compute pos error
        pos_error = target_pos - current_pos

        # Convert to axis-angle error
        quat_error = self.get_quat_diff(current_quat, target_quat)
        axis_angle_error = math_utils.axis_angle_from_quat(quat_error)

        return pos_error, axis_angle_error


    def apply_actions(self):
        motion_control_axes = [1,1,1,1,1,1]
        if(self.step%SUBSTEPS == 0):

            # obtain quantities from simulation
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
            if(self.previous_ee_pos is None or self.previous_ee_quat is None):
                self.previous_ee_pos = ee_pos_curr
                self.previous_ee_quat = ee_quat_curr

            # You can technically get these from the sim, 
            # but they are inaccurate when not doing implicit actuator PD
            ee_linvel_curr = (ee_pos_curr-self.previous_ee_pos)/(SIM_DT*SUBSTEPS)

            quat_diff = self.get_quat_diff(self.previous_ee_quat, ee_quat_curr)
            ee_angvel_curr = math_utils.axis_angle_from_quat(quat_diff)/(SIM_DT*SUBSTEPS)

            # import pdb; pdb.set_trace()
            
            self.previous_ee_pos = ee_pos_curr
            self.previous_ee_quat = ee_quat_curr

            jacobian = self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._joint_ids]
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)


            task_wrench = torch.zeros((self.num_envs, 6), device=self.device)

            
            pos_error, axis_angle_error = self.get_pose_error(
                ee_pos_curr,
                ee_quat_curr,
                self.control_cmd[:, :3],
                self.control_cmd[:, 3:7]
            )


            delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

            # Set tau = k_p * task_pos_error - k_d * task_vel_error (building towards eq. 3.96-3.98)
            task_wrench_motion = self._apply_task_space_gains(delta_fingertip_pose=delta_fingertip_pose,
                                                              fingertip_midpoint_linvel=ee_linvel_curr,
                                                              fingertip_midpoint_angvel=ee_angvel_curr)
   
            task_wrench = task_wrench + torch.tensor(motion_control_axes, device=self.device).unsqueeze(0) * task_wrench_motion

            # Set tau = J^T * tau, i.e., map tau into joint space as desired
            dof_torque = torch.zeros((self.num_envs, 6), device=self.device)
            dof_torque[:, 0:6] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

            dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)

            print("Efforts: "+str(dof_torque))
            self._asset.set_joint_effort_target(dof_torque, joint_ids = self._joint_ids)
            # Keep fingers closed
            # current_finger_joint_pos = self._asset.data.joint_pos[:, self._finger_joint_ids]
            # self._asset.set_joint_position_target(current_finger_joint_pos, joint_ids=self._finger_joint_ids)
            # self._asset.set_joint_velocity_target(24, joint_ids=self._finger_joint_ids)

        self.step+=1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self.gains = AdmittanceGains().to(self.device)
        self.r_state = RobotState().to(self.device)
        self.initialized = False
        self.step = 0
        self.previous_ee_pos = None
        self.previous_ee_quat = None  
        self.control_cmd = None
    
    @property
    def action_dim(self) -> int:
        return 6
    
@configclass
class MyActionCfg(mdp.ActionTermCfg):
    """Configuration for the joint velocity action term.

    See :class:`JointVelocityAction` for more details.
    """
    class_type: type[mdp.ActionTerm] = MyAction
    joint_names: List[str] = EFFORT_JOINTS
    finger_joint_names: List[str] = FINGER_JOINTS
    body_name: str = "flange"
    scale: float = MISSING
    controller = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")

@configclass
class VelocityActionsCfg():
    """Action specifications for the environment."""
    robot1_joint_positions = MyActionCfg(asset_name="robot1", scale=1.0)
    # robot2_joint_positions = MyActionCfg(asset_name="robot2", scale=0.0)


ROBOT1_EFFORT_CFG = SceneEntityCfg("robot1", body_names=["flange"], joint_names=EFFORT_JOINTS)
# ROBOT2_EFFORT_CFG = SceneEntityCfg("robot2", body_names=["flange"], joint_names=EFFORT_JOINTS)

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        robot1_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ROBOT1_EFFORT_CFG})
        robot1_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": ROBOT1_EFFORT_CFG})

        # robot2_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ROBOT2_EFFORT_CFG})
        # robot2_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": ROBOT2_EFFORT_CFG})


        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

def compute_frame_pose(asset, body_idx, offset_pos, offset_rot) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the pose of the target frame in the root frame.

    Returns:
        A tuple of the body's position and orientation in the root frame.
    """
    # obtain quantities from simulation
    ee_pose_w = asset.data.body_state_w[:, body_idx, :7]
    root_pose_w = asset.data.root_state_w[:, :7]
    # compute the pose of the body in the root frame
    ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )

    ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
        ee_pose_b, ee_quat_b, offset_pos, offset_rot
    )
    return ee_pose_b, ee_quat_b

def insert_relative_pose(
    env: ManagerBasedRLEnv,
    robot_entity_cfg: SceneEntityCfg
) -> torch.Tensor:
    
    """Reward the agent for reaching the object using tanh-kernel."""
    object = env.scene["gear_medium"]
    des_pos_b = WORLD_T_MEDGEAR_GOAL.pos
    rel_locs = object.data.root_pos_w[:, :3] - env.scene.env_origins
    distance = torch.norm(des_pos_b-rel_locs, dim=1)
    return -distance

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: move the gripper
    
    ee_pos = RewTerm(
        func=insert_relative_pose,
        weight=1,
        params={"robot_entity_cfg": SceneEntityCfg("robot1", body_names=["flange"])},
    )
    

def grasp_gear(env: ManagerBasedEnv, env_ids: torch.Tensor):

    MAX = 35*SUBSTEPS
    print(f"Closing gripper...")
    for i in range(MAX):
        for robot_name in ["robot1"]:
            # Simulate the scene a little to velocity control the robot to grasp the object
            _finger_joint_ids, _ = env.scene[robot_name].find_joints(FINGER_JOINTS)
            _joint_ids, _ = env.scene[robot_name].find_joints(EFFORT_JOINTS)
            current_joint_pos = env.scene[robot_name].data.joint_pos[:, _joint_ids]

            current_joint_pos = env.scene[robot_name].data.joint_pos[:, _joint_ids]
            env.scene[robot_name].set_joint_position_target(current_joint_pos, joint_ids=_joint_ids)
            env.scene[robot_name].set_joint_velocity_target(0, joint_ids=_joint_ids)
            
            env.scene[robot_name].set_joint_position_target(current_joint_pos, joint_ids=_joint_ids)
            env.scene[robot_name].set_joint_velocity_target(24, joint_ids=_finger_joint_ids)
            # set actions into simulator
            env.scene.write_data_to_sim()
            # simulate
            env.sim.step(render=True)
            # update buffers at sim dt
            env.scene.update(dt=env.physics_dt)

            body_ids, body_names = env.scene[robot_name].find_bodies("flange")
            print(env.scene[robot_name].data.body_state_w[:, body_ids[0], 7:])

    
@configclass
class EventCfg:
    """Configuration for events."""
    # on startup
    # reset_everything = EventTerm(
    #     func=mdp.reset_scene_to_default,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot1", joint_names=EFFORT_JOINTS),
    #         "mass_distribution_params": (0.1, 0.5),
    #         "operation": "add",
    #     },
    # )

    # on reset
    reset_initial_positions = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    grasp_gear = EventTerm(
        func=grasp_gear,
        mode="reset",
    )


def araas_to_cfg(**kwargs):

    # To keep the fingers stiff without bending
    finger_stiffness = 100.0
    finger_damping = 100.0

    gripper_vel_limit = 1.0
    # gripper_effort = 5.0 # Close hard but requires 5 substeps
    gripper_effort = 1.0
    small_joint_damping = 0.0
    cfg = DefaultSceneCfg(**kwargs)
    cfg.robot1 = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=home_dir + "pls_updated_gripper.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=0.5,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=r1_offset_pos,
            rot=r1_offset_rot,
            joint_pos={
                "shoulder_pan_joint": -0.5144105923545090,
                "shoulder_lift_joint": -1.643108558589230,
                "elbow_joint": -2.105897059942356,
                "wrist_1_joint": -0.9535990805992884,
                "wrist_2_joint": 1.561290095498969,
                "wrist_3_joint": -0.5144860744476318,
                # "right_inner_finger_knuckle_joint": 0.0,
                # "left_inner_finger_knuckle_joint": 0.0,
            } | {k: 0 for k in FINGER_JOINTS},
            joint_vel={
               "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": 0.0,
                "elbow_joint": 0.0,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
                # "right_inner_finger_knuckle_joint": 0.0,
                # "left_inner_finger_knuckle_joint": 0.0, 
            } | {k: 0 for k in FINGER_JOINTS}
        ),
        actuators={
            
            "arm": IdealPDActuatorCfg(
                joint_names_expr=EFFORT_JOINTS,
                effort_limit=100,
                velocity_limit=21.0,
                stiffness=0,
                damping=0,
                friction=0.0,
            ),
            "left_outer_knuckle_joint": ImplicitActuatorCfg(
                joint_names_expr=["left_outer_knuckle_joint"],
                velocity_limit=gripper_vel_limit,
                effort_limit=gripper_effort,
                stiffness=0.0,
                damping=5000.0,
            ),
            "right_outer_knuckle_joint": ImplicitActuatorCfg(
                joint_names_expr=["right_outer_knuckle_joint"],
                velocity_limit=1.0,
                effort_limit=100,
                stiffness=0,
                damping=0.0,
            ),
            "right_outer_finger_joint": ImplicitActuatorCfg(
                joint_names_expr=["right_outer_finger_joint"],
                velocity_limit=1.0,
                effort_limit=100,
                stiffness=0,
                damping=finger_damping,
            ),
            "right_inner_finger_joint": ImplicitActuatorCfg(
                joint_names_expr=["right_inner_finger_joint"],
                velocity_limit=1.0,
                effort_limit=100,
                stiffness=0,
                damping=small_joint_damping,
            ),
            "right_inner_finger_knuckle_joint": ImplicitActuatorCfg(
                joint_names_expr=["right_inner_finger_knuckle_joint"],
                velocity_limit=1.0,
                effort_limit=100,
                stiffness=0,
                damping=0.0,
            ),
            "left_outer_finger_joint": ImplicitActuatorCfg(
                joint_names_expr=["left_outer_finger_joint"],
                velocity_limit=1.0,
                effort_limit=100,
                stiffness=finger_stiffness,
                damping=0,
            ),
            "left_inner_finger_knuckle_joint": ImplicitActuatorCfg(
                joint_names_expr=["left_inner_finger_knuckle_joint"],
                velocity_limit=1.0,
                effort_limit=100,
                stiffness=0,
                damping=0.0,
            ),
            "left_inner_finger_joint": ImplicitActuatorCfg(
                joint_names_expr=["left_inner_finger_joint"],
                velocity_limit=1.0,
                effort_limit=100,
                stiffness=0,
                damping=small_joint_damping,
            )
        },
    ).replace(prim_path="{ENV_REGEX_NS}/Robot1")

    # cfg.robot2 = ArticulationCfg(
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=home_dir + "pls_updated_gripper.usd",
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=True,
    #             max_depenetration_velocity=1.0,
    #         ),
    #         activate_contact_sensors=False,
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=r2_offset_pos,
    #         rot=r2_offset_rot,
    #         joint_pos={
    #             "shoulder_pan_joint": -0.9920667074360754,
    #             "shoulder_lift_joint": -1.107338840559089,
    #             "elbow_joint": -2.473368842697666,
    #             "wrist_1_joint": -1.128834397685560,
    #             "wrist_2_joint": 1.563770470523882,
    #             "wrist_3_joint": 0.01950128190219402,
    #         } | {k: 0 for k in FINGER_JOINTS},
    #         joint_vel={
    #            "shoulder_pan_joint": 0.0,
    #             "shoulder_lift_joint": 0.0,
    #             "elbow_joint": 0.0,
    #             "wrist_1_joint": 0.0,
    #             "wrist_2_joint": 0.0,
    #             "wrist_3_joint": 0.0,
    #         }| {k: 0 for k in FINGER_JOINTS}
    #     ),
    #     actuators={
    #         "arm": ImplicitActuatorCfg(
    #             joint_names_expr=EFFORT_JOINTS,
    #             velocity_limit=1.0,
    #             effort_limit=10.0,
    #             stiffness=200.0,
    #             damping=800.0,
    #         ),
    #         "left_outer_knuckle_joint": ImplicitActuatorCfg(
    #             joint_names_expr=["left_outer_knuckle_joint"],
    #             velocity_limit=1.0,
    #             effort_limit=gripper_effort,
    #             stiffness=0.0,
    #             damping=5000.0,
    #         ),
    #         "right_outer_knuckle_joint": ImplicitActuatorCfg(
    #             joint_names_expr=["right_outer_knuckle_joint"],
    #             velocity_limit=1.0,
    #             effort_limit=100,
    #             stiffness=0.0,
    #             damping=0.0,
    #         ),
    #         "right_outer_finger_joint": ImplicitActuatorCfg(
    #             joint_names_expr=["right_outer_finger_joint"],
    #             velocity_limit=1.0,
    #             effort_limit=100,
    #             stiffness=0.0,
    #             damping=finger_damping,
    #         ),
    #         "right_inner_finger_joint": ImplicitActuatorCfg(
    #             joint_names_expr=["right_inner_finger_joint"],
    #             velocity_limit=1.0,
    #             effort_limit=100,
    #             stiffness=0.0,
    #             damping=0.0,
    #         ),
          
    #         "right_inner_finger_knuckle_joint": ImplicitActuatorCfg(
    #             joint_names_expr=["right_inner_finger_knuckle_joint"],
    #             velocity_limit=1.0,
    #             effort_limit=100,
    #             stiffness=0.0,
    #             damping=0.0,
    #         ),
    #         "left_outer_finger_joint": ImplicitActuatorCfg(
    #             joint_names_expr=["left_outer_finger_joint"],
    #             velocity_limit=1.0,
    #             effort_limit=100,
    #             stiffness=finger_stiffness,
    #             damping=0.0,
    #         ),
    #         "left_inner_finger_knuckle_joint": ImplicitActuatorCfg(
    #             joint_names_expr=["left_inner_finger_knuckle_joint"],
    #             velocity_limit=1.0,
    #             effort_limit=100,
    #             stiffness=0.0,
    #             damping=0.0,
    #         ),
    #         "left_inner_finger_joint": ImplicitActuatorCfg(
    #             joint_names_expr=["left_inner_finger_joint"],
    #             velocity_limit=1.0,
    #             effort_limit=100,
    #             stiffness=0.0,
    #             damping=0.0,
    #         )
        
    #     },
    # ).replace(prim_path="{ENV_REGEX_NS}/Robot2")

    cfg.table = AssetBaseCfg(
        prim_path="/World/Origin2/Table",  
        spawn = sim_utils.UsdFileCfg(
            usd_path=home_dir + "table_top_collision.usd", 
        )
    ).replace(prim_path="{ENV_REGEX_NS}/Table")

    cfg.taskboard = AssetBaseCfg(
        prim_path="/World/Origin2/Taskboard",  
        spawn = sim_utils.UsdFileCfg(
            usd_path=home_dir + "taskboard/taskboard.usd", 
        ),
        
        init_state = AssetBaseCfg.InitialStateCfg(WORLD_T_HOLE.pos[0], WORLD_T_HOLE.quat[0])
    ).replace(prim_path="{ENV_REGEX_NS}/Taskboard")
    
    # cfg.gear_small = MISSING
    cfg.gear_medium = MISSING
    # cfg.gear_large = MISSING
    
    return cfg

@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()


@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""
    # Scene settings
    scene = araas_to_cfg(num_envs=NUM_ENVS, env_spacing=4)
    # Basic settingsVS, env_spacing=4)
    # Basic settings
    observations = ObservationsCfg()
    actions = VelocityActionsCfg()
    events = EventCfg()

    # RL Specific
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No command generator
    commands: CommandsCfg = CommandsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""

        # general settings
        self.decimation = DECIMATION*SUBSTEPS
        self.episode_length_s = 4
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = SIM_DT/SUBSTEPS
        
        self.sim.physx.enable_ccd = True


        # lift up the medium gear
        
        
        # self.scene.gear_small = RigidObjectCfg( 
        #     prim_path="{ENV_REGEX_NS}/GearSmall",  
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path = "/home/aidanc/multi-robot-assembly/taskboard/gear_small.usd",
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=WORLD_T_SMALLGEAR_GOAL.pos[0], 
        #                                               rot=WORLD_T_SMALLGEAR_GOAL.quat[0]),
        # )

        self.scene.gear_medium = RigidObjectCfg( 
            prim_path="{ENV_REGEX_NS}/GearMedium",  
            spawn=sim_utils.UsdFileCfg(
                usd_path = "/home/aidanc/multi-robot-assembly/taskboard/gear_medium.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True
                )
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=WORLD_T_GEAR_START.pos[0], 
                                                      rot=WORLD_T_GEAR_START.quat[0]),
        )

        # self.scene.gear_large = RigidObjectCfg( 
        #     prim_path="{ENV_REGEX_NS}/GearLarge",  
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path = "/home/aidanc/multi-robot-assembly/taskboard/gear_large.usd",
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=WORLD_T_LARGEGEAR_GOAL.pos[0], 
        #                                               rot=WORLD_T_LARGEGEAR_GOAL.quat[0]),
        # )

        
        if(TASK == "medium_gear_insert"):
            base_q = wxyz_to_xyzw(self.scene.robot1.init_state.rot) 


            # Initialize the gripper to be directly above the target object pose
            # This is where the object is initialized to be reset later inside the fingers
            world_T_flange = WORLD_T_GEAR_START.multiply(OBJECT_T_FLANGE)
            joint_angles = solve_ik(self.scene.robot1.init_state.pos, 
                                    base_q, 
                                    world_T_flange.pos[0].cpu(),
                                    wxyz_to_xyzw(world_T_flange.quat[0].cpu()))
            assert len(joint_angles) == len(EFFORT_JOINTS)
            for joint_name, joint_angle in zip(EFFORT_JOINTS, joint_angles):
                self.scene.robot1.init_state.joint_pos[joint_name] = joint_angle
        else:
            raise NotImplementedError
        
        
gym.register(
    id="APAImpedance-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MyEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:myenv_cfg.yaml",
    },
)
