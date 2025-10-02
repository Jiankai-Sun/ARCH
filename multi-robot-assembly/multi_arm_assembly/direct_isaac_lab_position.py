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
import cv2

import omni.isaac.lab.utils.math as math_utils
import omni.kit.commands

from omni.isaac.lab.managers import RewardTermCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.controllers import DifferentialIKControllerCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.assets.rigid_object import RigidObjectCfg
from multi_arm_assembly.utils import solve_ik, visualize_depth_image, visualize_rgb_image, wxyz_to_xyzw, xyzw_to_wxyz
from omni.isaac.lab.controllers.differential_ik import DifferentialIKController
from omni.isaac.core.utils.string import find_unique_string_name
import omni.isaac.core.utils.prims as prim_utils
from pxr import UsdPhysics
from pxr import Gf, Sdf

"""Rest everything follows."""

import torch
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject, AssetBase
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.envs.mdp as mdp
from typing import Sequence
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from omni.isaac.lab.sim import SimulationCfg
from multi_arm_assembly.utils import log_list, task_from_name, FLANGE_T_TOOL, set_seed
set_seed(123)
from omni.isaac.lab.sensors import TiledCameraCfg, Camera, TiledCamera, CameraCfg
from PIL import Image
import time
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene

import copy
import pickle
from datetime import datetime

NUM_ENVS = 1000
# NUM_ENVS = 2
home_dir = os.getcwd() + '/../'
# INITIAL_CFG = task_from_name("gear_in_taskboard")
# INITIAL_CFG = task_from_name("rod_in_gear")
# INITIAL_CFG = task_from_name("strut_plus_elbow_in_platform")
# INITIAL_CFG = task_from_name("strut_in_elbow")
# INITIAL_CFG = task_from_name("bolt_in_platform")
INITIAL_CFG = task_from_name("fmb_example")
INSERT_INITIAL_CFG = task_from_name("AssemblyInsert")
SCRIPTED_INSERT_INITIAL_CFG = task_from_name("AssemblyScriptedInsert")
GRASP_INITIAL_CFG = task_from_name("AssemblyGrasp")
SCRIPTED_GRASP_INITIAL_CFG = task_from_name("AssemblyScriptedGrasp")
SCRIPTED_GRASP_HORIZONTAL_INITIAL_CFG = task_from_name("AssemblyScriptedGraspHorizontal")
SCRIPTED_REGRASP_INITIAL_CFG = task_from_name("AssemblyScriptedRegrasp")
REGRASP_INITIAL_CFG = task_from_name("AssemblyRegrasp")
PLACE_INITIAL_CFG = task_from_name("AssemblyPlace")
SCRIPTED_PLACE_INITIAL_CFG = task_from_name("AssemblyScriptedPlace")
SCRIPTED_PLACE_HORIZONTAL_INITIAL_CFG = task_from_name("AssemblyScriptedPlaceHorizontal")
MOVE_INITIAL_CFG = task_from_name("AssemblyMove")
MOVE_TO_FIXTURE_INITIAL_CFG = task_from_name("AssemblyMoveToFixture")
MOVE_TO_FIXTURE_HORIZONTAL_INITIAL_CFG = task_from_name("AssemblyMoveToFixtureHorizontal")
MOVE_TO_BOARD_FROM_FIXTURE_INITIAL_CFG = task_from_name("AssemblyMoveToBoardFromFixture")
MOVE_TO_BOARD_FROM_GRASP_INITIAL_CFG = task_from_name("AssemblyMoveToBoardFromGrasp")

BEAM_INSERT_INITIAL_CFG = task_from_name("BeamInsert")
BEAM_SCRIPTED_INSERT_INITIAL_CFG = task_from_name("BeamScriptedInsert")
BEAM_SCRIPTED_GRASP_INITIAL_CFG = task_from_name("BeamScriptedGrasp")
BEAM_SCRIPTED_PLACE_INITIAL_CFG = task_from_name("BeamScriptedPlace")
BEAM_MOVE_TO_BOARD_INITIAL_CFG = task_from_name("BeamMoveToBoard")
STOOL_INSERT_INITIAL_CFG = task_from_name("StoolInsert")
STOOL_SCRIPTED_INSERT_INITIAL_CFG = task_from_name("StoolScriptedInsert")
STOOL_SCRIPTED_GRASP_INITIAL_CFG = task_from_name("StoolScriptedGrasp")
STOOL_SCRIPTED_GRASP_HORIZONTAL_INITIAL_CFG = task_from_name("StoolScriptedGraspHorizontal")
STOOL_SCRIPTED_PLACE_INITIAL_CFG = task_from_name("StoolScriptedPlace")
STOOL_MOVE_TO_BOARD_INITIAL_CFG = task_from_name("StoolMoveToBoard")

# USE_CAMERA = True  # False
# IMAGE_WIDTH = 128  # 60
# IMAGE_HEIGHT = 128  # 60

r1_offset_pos = (0.8367000122070312, 0.6095999755859375, 0.02250)
r1_offset_rot = (0.7071068, 0, 0, 0.7071068)

r2_offset_pos = (-0.8367000122070312, 0.6095999755859375, 0.02250)
r2_offset_rot = (0.7071068, 0, 0, -0.7071068)

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
SIM_DT = 1 / 120
SUBSTEPS = 1
DECIMATION = 5
UNIT_QUAT = torch.tensor([[1, 0, 0, 0]]).type(torch.FloatTensor).cuda()
UNIT_POINT = torch.zeros(1, 3).type(torch.FloatTensor).cuda()

MARKER_ENABLE = True # False #True

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return math_utils.quat_mul(
        math_utils.quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), math_utils.quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )

@torch.jit.script
def randomize_rotation_rpy_quat(rand0, rand1, rand2, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    quat_x = math_utils.quat_from_angle_axis(rand0 * np.pi, x_unit_tensor)
    quat_y = math_utils.quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    quat_z = math_utils.quat_from_angle_axis(rand2 * np.pi, z_unit_tensor)

    return math_utils.quat_mul(math_utils.quat_mul(quat_x, quat_y), quat_z)

@dataclass
class RobotState:
    init_world_force: torch.tensor = torch.zeros([NUM_ENVS, 3])
    init_world_torque: torch.tensor = torch.zeros([NUM_ENVS, 3])
    tool_pos: torch.tensor = torch.zeros([NUM_ENVS, 3])
    tool_quat: torch.tensor = torch.zeros([NUM_ENVS, 4])
    tool_vel: torch.tensor = torch.zeros([NUM_ENVS, 6])
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

@dataclass
class IPose():
    pos: torch.tensor
    quat: torch.tensor

    def multiply(self, p: IPose):
        return IPose(*math_utils.combine_frame_transforms(self.pos, self.quat, p.pos, p.quat))

    def invert(self):
        t12, q12 = math_utils.subtract_frame_transforms(self.pos, self.quat)
        return IPose(t12, q12)

    def to_vec(self):
        return torch.cat([self.pos, self.quat], dim=1)

    def to(self, device):
        return IPose(self.pos.to(device), self.quat.to(device))

    @staticmethod
    def from_pose(pose):
        pos, quat = pose
        return IPose(torch.Tensor([pos]).type(torch.FloatTensor).cuda(),
                     torch.Tensor([xyzw_to_wxyz(quat)]).type(torch.FloatTensor).cuda())

    def repeat(self, n):
        return IPose(self.pos.repeat((n, 1)), self.quat.repeat((n, 1)))


# ROBOT1_EFFORT_CFG = SceneEntityCfg("robot1", body_names=["flange"], joint_names=EFFORT_JOINTS)
# ROBOT2_EFFORT_CFG = SceneEntityCfg("robot2", body_names=["flange"], joint_names=EFFORT_JOINTS)
ROBOT1_EFFORT_CFG = SceneEntityCfg("robot1", body_names=["tool0"], joint_names=EFFORT_JOINTS)
ROBOT2_EFFORT_CFG = SceneEntityCfg("robot2", body_names=["tool0"], joint_names=EFFORT_JOINTS)
ROBOT1_CAMERA_CFG = SceneEntityCfg("tiled_camera")


def pose_from_state(state, body_id, origins):
    ee_pos_w = state[:, body_id, :3] - origins
    ee_quat_w = state[:, body_id, 3:7]
    ipose = IPose(ee_pos_w, ee_quat_w)
    return ipose


def save_depth(depth_data):
    depth_min = np.min(depth_data)
    depth_max = np.max(depth_data)
    normalized_depth = ((depth_data - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

    # Create a PIL Image from the numpy array
    image = Image.fromarray(normalized_depth, mode='L')

    # Save the image as PNG
    image.save('runs/depth_image_{}.png'.format(str(time.time())))


def update_rigid_attachments(fixed_joints, obj_T_flange: IPose):
    # Domain randomization
    translation_offset = (torch.rand(NUM_ENVS, 3).type(torch.FloatTensor).cuda() - 0.5) * 0.01
    ero = torch.rand(NUM_ENVS, 3).type(torch.FloatTensor).cuda() * 0
    rotation_offset = math_utils.quat_from_euler_xyz(ero[:, 0], ero[:, 1], ero[:, 2])

    for env_id, fixed_joint in enumerate(fixed_joints):
        trans_pose = IPose(pos=translation_offset[env_id:env_id + 1, :], quat=UNIT_QUAT)
        rot_pose = IPose(pos=UNIT_POINT, quat=rotation_offset[env_id:env_id + 1, :])
        joint_pose = trans_pose.multiply(obj_T_flange.invert()).multiply(rot_pose)

        peg_pos = joint_pose.pos[0].cpu().numpy().tolist()
        peg_quat = joint_pose.quat[0].cpu().numpy().tolist()

        fixed_joint.GetLocalPos0Attr().Set(Gf.Vec3f(*peg_pos))
        fixed_joint.GetLocalRot0Attr().Set(Gf.Quatf(*peg_quat))
        fixed_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed_joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    return fixed_joint


# def _refresh_asset(prim_path):
#     # Refreshing payloads manually is a way to get the Articulation to update immediately while the timeline is
#     # still playing.  Usd Physics should be doing this automatically, but there is currently a bug.  This function
#     # will eventually become unnecessary.
#     stage = stage_utils.get_current_stage()
#     prim = prim_utils.get_prim_at_path(prim_path)

#     composed_refs = omni.usd.get_composed_references_from_prim(prim)
#     if len(composed_refs) != 0:
#         reference = Sdf.Reference(prim_path)
#         omni.kit.commands.execute(
#             "RemoveReference", stage=stage, prim_path=Sdf.Path(prim_path), reference=reference
#         )
#         omni.kit.commands.execute("AddReference", stage=stage, prim_path=Sdf.Path(prim_path), reference=reference)

def motion_planning(sim: sim_utils.SimulationContext, scene, des_world_T_peg):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    if MARKER_ENABLE:
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        # frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        frame_marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals = des_world_T_peg
    # [
    #     [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
    #     [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
    #     [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
    # ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    # current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals  # [current_goal_idx]

    # Specify robot-specific parameters
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while True: # simulation_app.is_running():
        # reset
        if count % 150 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        if MARKER_ENABLE:
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


def close_gripper(env, env_ids: torch.Tensor):
    print(f"Closing gripper...")
    for i in range(10):
        for robot_name in ["robot1"]:
            # Simulate the scene a little to velocity control the robot to grasp the object
            _finger_joint_ids, _ = env.scene[robot_name].find_joints(FINGER_JOINTS)
            _joint_ids, _ = env.scene[robot_name].find_joints(EFFORT_JOINTS)

            current_joint_pos = env.scene[robot_name].data.joint_pos[:, _joint_ids]

            env.scene[robot_name].set_joint_position_target(current_joint_pos, joint_ids=_joint_ids)
            env.scene[robot_name].set_joint_velocity_target(24, joint_ids=_finger_joint_ids)
            # set actions into simulator
            env.scene.write_data_to_sim()
            # simulate
            env.sim.step(render=True)
            # update buffers at sim dt
            env.scene.update(dt=env.physics_dt)


def create_rigid_attachments(attachment_path: str, robot_path: str, extra_attachments=[]):
    # Domain randomization
    fixed_joints = []
    for env_id in range(NUM_ENVS):

        env0_path = f"/World/envs/env_{env_id}/{robot_path}"
        fixed_joint_path = env0_path + "/AssemblerFixedJoint"
        fixed_joint_path = find_unique_string_name(fixed_joint_path, lambda x: not prim_utils.is_prim_path_valid(x))
        stage = stage_utils.get_current_stage()
        fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)

        target0 = env0_path + "/flange"
        target1 = f"/World/envs/env_{env_id}/{attachment_path}"

        if target0 is not None:
            fixed_joint.GetBody0Rel().SetTargets([target0])
        if target1 is not None:
            fixed_joint.GetBody1Rel().SetTargets([target1])

        fixed_joints.append(fixed_joint)

        for ex_id, (attachment_a, attachment_b) in enumerate(extra_attachments):
            ex_path = f"/World/envs/env_{env_id}/ExFixedJoint_{ex_id}"
            stage = stage_utils.get_current_stage()
            fixed_joint = UsdPhysics.FixedJoint.Define(stage, ex_path)
            if attachment_a is not None:
                fixed_joint.GetBody0Rel().SetTargets([f"/World/envs/env_{env_id}/{attachment_a}"])
            if attachment_b is not None:
                fixed_joint.GetBody1Rel().SetTargets([f"/World/envs/env_{env_id}/{attachment_b}"])

    return fixed_joints


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

def next_pose(current_pose, target_pose, time_steps=1, max_angle_step=0.1, max_translation_step=1):
    t_start = current_pose[:, :3]  # [num_env, 3], torch.tensor
    r_start = current_pose[:, 3:]  # [num_env, 4], quat wxyz, torch.tensor
    t_end = target_pose[:, :3]     # [num_env, 3], torch.tensor
    r_end = target_pose[:, 3:]     # [num_env, 4], quat xyzw, torch.tensor

    # Linear interpolation for translation components
    delta_t = (t_end - t_start) / time_steps
    print('delta_t: ', delta_t, t_end, t_start)
    delta_t = torch.clamp(delta_t, -max_translation_step, max_translation_step)
    t_next = t_start + delta_t

    # Slerp for rotation components with max angle step
    r_start = r_start / torch.norm(r_start, dim=-1, keepdim=True)
    r_end = r_end / torch.norm(r_end, dim=-1, keepdim=True)

    # Ensure the shortest path is taken
    dot_product = torch.sum(r_start * r_end, dim=-1, keepdim=True)
    r_end = torch.where(dot_product < 0, -r_end, r_end)

    # Convert to scipy Rotation objects
    r_start_np = r_start.cpu().numpy()
    r_end_np = r_end.cpu().numpy()

    # Compute the angular distance between r_start and r_end
    angles = R.from_quat(r_start_np, scalar_first=True).inv() * R.from_quat(r_end_np, scalar_first=False)
    angle_rad_vec = np.asarray(angles.magnitude())
    angle_rad = angle_rad_vec.reshape(angle_rad_vec.shape[0], 1)
    rotation_axis = angles.as_rotvec()
    # Handle potential zero division issues
    non_zero_angle_mask = angle_rad_vec > 1e-6
    rotation_axis[~non_zero_angle_mask] = 0.0

    # Clamp the angle per step to max_angle_step
    # print('angle_rad.shape: ', angle_rad, angle_rad.shape)
    angle_rad_clamped = np.clip(angle_rad, -max_angle_step, max_angle_step)
    # fraction = angle_rad_clamped / angle_rad

    # Convert the clamped angle to a quaternion
    delta_quat_np = angle_axis_to_quat(angle_rad_clamped, rotation_axis)

    # Create Rotation object from the delta quaternion
    delta_rotation = R.from_quat(delta_quat_np, scalar_first=False)

    # Compute the next orientation by multiplying the quaternions
    r_next_np = (R.from_quat(r_start_np, scalar_first=False) * delta_rotation).as_quat(scalar_first=True)

    r_next = r_start # torch.tensor(r_next_np, dtype=torch.float32, device=current_pose.device)
    next_pose_value =  torch.cat((t_next, r_next), dim=-1)
    # print('current_pose: {}, \ntarget_pose: {}, \nnext_pose_value: {}'.format(current_pose, target_pose, next_pose_value))
    # print('delta_t, delta_r: ', delta_t, angle_rad_clamped)
    return next_pose_value

# Define the function to convert an angle-axis to a quaternion
def angle_axis_to_quat(angle_rad, axis):
    # Normalize the axis to ensure it is a unit vector
    axis = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-8)
    # Compute the quaternion components
    w = np.cos(angle_rad / 2.0)
    # print('axis, angle_rad: ', axis, angle_rad, axis.shape, angle_rad.shape)
    xyz = axis * np.sin(angle_rad / 2.0)
    return np.concatenate([xyz, w], axis=-1)   # Return in xyzw format

def interpolate_joint_angles(joint_angles1, joint_angles2, num_steps):
    # Convert input lists to numpy arrays for easier computation
    joint_angles1 = np.array(joint_angles1)
    joint_angles2 = np.array(joint_angles2)
    
    # Initialize an array to store the interpolated joint angles
    interpolated_angles = np.zeros((num_steps, len(joint_angles1)))
    
    # Calculate interpolation step size for each joint angle
    step_size = 1.0 / (num_steps - 1)
    
    # Interpolate between joint_angles1 and joint_angles2
    for i in range(num_steps):
        # Calculate the interpolated joint angles at step i
        interpolated_angles[i, :] = (1 - step_size * i) * joint_angles1 + (step_size * i) * joint_angles2
    
    return interpolated_angles

@configclass
class MyEnvCfg(DirectRLEnvCfg):
    """Configuration for the cartpole environment."""
    initial_cfg = INITIAL_CFG
    # Scene settings
    episode_length_s = 10 # 60 # 4.0 * 8 * 2
    # # max_episode_length = math.ceil(self.max_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation)) = 4 / (0.008 * 5) = 768
    # 4s = 100 frames
    decimation = DECIMATION * SUBSTEPS
    sim = SimulationCfg(dt=SIM_DT / SUBSTEPS)
    scene = InteractiveSceneCfg(num_envs=NUM_ENVS, env_spacing=4)

    # change viewer settings
    # viewer = ViewerCfg(eye=(-2.0, -2.5, 2.5), lookat=(-2.0, 0.5, 0.0))  # robot2

    # for beam
    # viewer = ViewerCfg(eye=(2.0, -2.5, 2.5), lookat=(2.0, 0.5, 0.0))  # robot1
    viewer = ViewerCfg(eye=(2.3, -0.6, 0.4), lookat=(2.2, 0.5, 0.0))  # robot1
    viewer = ViewerCfg(eye=(3., -1.0, 1.2), lookat=(2.2, 0.5, 0.0))  # robot1

    # for stool
    # viewer = ViewerCfg(eye=(2.2, -0.3, 0.5), lookat=(2.2, 0.5, 0.0))  # robot1
    # viewer = ViewerCfg(eye=(2.2, -1., 1.), lookat=(2.2, 0.5, 0.0))  # robot1

    def __post_init__(self):

        # num_actions = 3 * INITIAL_CFG.num_robots
        # num_actions = 4 * INITIAL_CFG.num_robots
        self.num_actions = 3 * self.initial_cfg.num_robots + (3 * int(self.initial_cfg.allow_hole_rotation)) + (
                3 * int(self.initial_cfg.allow_peg_rotation)) + 1 * self.initial_cfg.num_robots
        # [robot1_pos (3), robot1_rot (3), robot1_gripper (1), robot2_pos (3), robot2_rot (3), robot2_gripper (1)]
        self.num_observations = (9 + 6 * (
                    self.initial_cfg.USE_FT_SENSOR) + 1 * self.initial_cfg.GRIPPER_STATUS_OBS * self.initial_cfg.allow_gripper_status
                                 + 6 * self.initial_cfg.VELOCITY_OBS) * self.initial_cfg.num_robots
        self.camera_type = self.initial_cfg.CAMERA_TYPE  #
        # self.ft_sensor_type = self.initial_cfg.FT_SENSOR_TYPE  # 'ArticulationView'
        if self.initial_cfg.TASK_NAME in ['AssemblyMove-v0']:
            self.episode_length_s = 8  # 15
            if self.initial_cfg.SUBTASK in ['MoveToBoardFromGrasp']:
                self.episode_length_s = 10
        elif self.initial_cfg.TASK_NAME in ['AssemblyScriptedPlace-v0', 'AssemblyScriptedPlaceHorizontal-v2']:
            self.episode_length_s = 6
        elif self.initial_cfg.TASK_NAME in ['AssemblyScriptedGrasp-v0', 'AssemblyGrasp-v0', 'AssemblyScriptedGraspHorizontal-v0', 'BeamScriptedGrasp-v0', 'BeamScriptedPlace-v0', 'StoolScriptedPlace-v0']:
            self.episode_length_s = 4
        elif self.initial_cfg.TASK_NAME in ['AssemblyInsert-v0', 'AssemblyScriptedInsert-v0', 'BeamInsert-v0', 'StoolInsert-v0']: # 'BeamScriptedInsert-v0']:
            self.episode_length_s = 4  # * 5 * 4  # 4 / (0.008 * 5) = 100 timesteps
        # To keep the fingers stiff without bending
        # finger_stiffness = 100.0
        # finger_damping = 100.0
        finger_stiffness = 0.0 # 50.0
        finger_damping = 5000.0 # 100.0

        gripper_vel_limit = 1.0
        # gripper_effort = 5.0 # Close hard but requires 5 substeps
        gripper_effort = 1.5
        small_joint_damping = 0.0
        arm_effort = 100
        arm_stiffness = 400.0
        arm_damping = 800.0
        # arm_stiffness = 50.0
        # arm_damping = 100.0
        arm_vel_limit = 20.0
        self.parts = []
        self.peripherals = []
        self.articulated_parts = []
        self.robot1 = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                # usd_path=home_dir + "pls_updated_gripper.usd",
                usd_path=home_dir + "taskboard/fmb_example/Robots/pls_updated_gripper_v4.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=2.5,
                ),
                activate_contact_sensors=True,
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
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=EFFORT_JOINTS,
                    velocity_limit=arm_vel_limit,
                    effort_limit=arm_effort,
                    stiffness=arm_stiffness,
                    damping=arm_damping,
                ),
                "left_outer_knuckle_joint": ImplicitActuatorCfg(
                    joint_names_expr=["left_outer_knuckle_joint"],
                    velocity_limit=gripper_vel_limit,
                    effort_limit=gripper_effort,
                    stiffness=finger_stiffness, #0.0,
                    damping=finger_damping, # 5000.0,
                ),
                "right_outer_knuckle_joint": ImplicitActuatorCfg(
                    joint_names_expr=["right_outer_knuckle_joint"],
                    # velocity_limit=1.0,
                    # effort_limit=100,
                    velocity_limit=gripper_vel_limit,
                    effort_limit=gripper_effort,
                    stiffness=finger_stiffness, # 0,
                    damping=finger_damping,  # 0.0,
                ),
                "right_outer_finger_joint": ImplicitActuatorCfg(
                    joint_names_expr=["right_outer_finger_joint"],
                    velocity_limit=1.0,
                    effort_limit=100,
                    stiffness=0.0,
                    damping=0.05,
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
                    stiffness=0.0,
                    damping=0.05,
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
        ).replace(prim_path="/World/envs/env_.*/robot1")

        if self.initial_cfg.USE_CAMERA:
            # pos 2: back look at front
            camera_pos = self.initial_cfg.WORLD_T_ZIVID_HOME[0]  # (0.15, -0.6, 0.55)
            # camera_rot_wxyz =  (0.8660254, 0.5, 0, 0, )  # [60, 0, 0]
            camera_rot_wxyz = (0.9063078, 0.4226183, 0, 0, )  # [ 50, 0, 0 ]
            if self.camera_type == 'tiled':
                # camera_pos = list(self.initial_cfg.WORLD_T_PEG_START[0])
                # camera_pos[0] -= 0.2
                # camera_rot = (0.9945, 0.0, 0.1045, 0.0)
                # camera_pos = (-0.8, 0.70, 0.35)
                # camera_rot_wxyz = (0.50389, 0.42764, -0.49118, -0.56742)
                # pos 1: left look at right
                # camera_pos = (-0.73111, 0.60, 0.35)
                # camera_rot_wxyz = (0.56087, 0.41796, -0.44067, -0.56262)                
                self.tiled_camera = TiledCameraCfg(
                    prim_path="/World/envs/env_.*/camera",
                    # prim_path="/World/envs/env_.*/robot1/flange/camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=camera_pos, rot=camera_rot_wxyz, convention="opengl"),
                    data_types=["depth", 'rgb'],
                    # spawn=sim_utils.PinholeCameraCfg(
                    #     focal_length=24.0,  # Keep this as is, it's actually good for close-up
                    #     focus_distance=0.1,  # Set to 0.1 meters (about 4 inches)
                    #     horizontal_aperture=20.955,  # Keep this as is
                    #     clipping_range=(0.01, 1.0),  # Near plane at 1cm, far plane at 2m
                    # ),
                    # spawn=sim_utils.PinholeCameraCfg(
                    #     focal_length=111.4058349609375, # 24.0,  # in cm
                    #     focus_distance=0.1,  # in m
                    #     horizontal_aperture=1200,  # in mm
                    #     clipping_range=(0.3, 1.1),  # in m
                    # ),
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length= 24.0,  # in cm 62.10726318359375
                        focus_distance=0.1,  # in m
                        horizontal_aperture=20.955,  # in mm
                        clipping_range=(0.3, 1.1),  # in m
                    ),
                    width=self.initial_cfg.IMAGE_WIDTH,
                    height=self.initial_cfg.IMAGE_HEIGHT,
                    debug_vis=True,
                )
            else:
                self.tiled_camera = CameraCfg(
                    # prim_path="/World/envs/env_.*/camera",
                    prim_path="/World/envs/env_.*/robot1/flange/camera",
                    offset=CameraCfg.OffsetCfg(pos=camera_pos, rot=camera_rot_wxyz, convention="opengl"),
                    data_types=['rgb', 'distance_to_camera'],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,  # Keep this as is, it's actually good for close-up
                        focus_distance=0.1,  # Set to 0.1 meters (about 4 inches)
                        horizontal_aperture=20.955,  # Keep this as is
                        clipping_range=(0.01, 1.0),  # Near plane at 1cm, far plane at 2m
                    ),
                    width=self.initial_cfg.IMAGE_WIDTH,
                    height=self.initial_cfg.IMAGE_HEIGHT,
                    debug_vis=True,
                )

        if self.initial_cfg.num_robots == 2:
            self.robot2 = ArticulationCfg(
                spawn=sim_utils.UsdFileCfg(
                    usd_path=home_dir + "pls_updated_gripper.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True,
                        max_depenetration_velocity=2.5,
                    ),
                    activate_contact_sensors=False,
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=r2_offset_pos,
                    rot=r2_offset_rot,
                    joint_pos={
                                  "shoulder_pan_joint": -0.9920667074360754,
                                  "shoulder_lift_joint": -1.107338840559089,
                                  "elbow_joint": -2.473368842697666,
                                  "wrist_1_joint": -1.128834397685560,
                                  "wrist_2_joint": 1.563770470523882,
                                  "wrist_3_joint": 0.01950128190219402,
                              } | {k: 0 for k in FINGER_JOINTS},
                    joint_vel={
                                  "shoulder_pan_joint": 0.0,
                                  "shoulder_lift_joint": 0.0,
                                  "elbow_joint": 0.0,
                                  "wrist_1_joint": 0.0,
                                  "wrist_2_joint": 0.0,
                                  "wrist_3_joint": 0.0,
                              } | {k: 0 for k in FINGER_JOINTS}
                ),
                actuators={
                    "arm": ImplicitActuatorCfg(
                        joint_names_expr=EFFORT_JOINTS,
                        velocity_limit=arm_vel_limit,
                        effort_limit=arm_effort,
                        stiffness=arm_stiffness,
                        damping=arm_damping,
                    ),
                    "left_outer_knuckle_joint": ImplicitActuatorCfg(
                        joint_names_expr=["left_outer_knuckle_joint"],
                        velocity_limit=gripper_vel_limit,
                        effort_limit=gripper_effort,
                        stiffness=finger_stiffness,  # 0.0,
                        damping=finger_damping,  # 5000.0,
                    ),
                    "right_outer_knuckle_joint": ImplicitActuatorCfg(
                        joint_names_expr=["right_outer_knuckle_joint"],
                        velocity_limit=gripper_vel_limit,
                        effort_limit=gripper_effort,
                        stiffness=finger_stiffness,  # 0.0,
                        damping=finger_damping,  # 5000.0,
                    ),
                    "right_outer_finger_joint": ImplicitActuatorCfg(
                        joint_names_expr=["right_outer_finger_joint"],
                        velocity_limit=1.0,
                        effort_limit=100,
                        stiffness=0.0,
                        damping=0.05,
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
                        stiffness=0.0,
                        damping=0.05,
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
            ).replace(prim_path="/World/envs/env_.*/robot2")

        self.peripherals.append(
            AssetBaseCfg(
                prim_path="/World/envs/env_.*/table",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=home_dir + "table_top_collision.usd",
                )
            )
        )

        world_T_hole = IPose.from_pose(self.initial_cfg.WORLD_T_HOLE_START)

        if self.initial_cfg.HOLE_TYPE == "peripheral":
            self.peripherals.append(
                AssetBaseCfg(
                    prim_path="/World/envs/env_.*/hole",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=self.initial_cfg.HOLE_ASSET_NAME,
                    ),

                    init_state=AssetBaseCfg.InitialStateCfg(world_T_hole.pos[0], world_T_hole.quat[0])
                )
            )
        elif self.initial_cfg.HOLE_TYPE == "part":
            self.parts.append(
                RigidObjectCfg(
                    prim_path="/World/envs/env_.*/hole",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=self.initial_cfg.HOLE_ASSET_NAME,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True
                        )
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=world_T_hole.pos[0],
                                                              rot=world_T_hole.quat[0]),
                )
            )
        else:
            raise NotImplementedError

        world_T_peg_start = IPose.from_pose(self.initial_cfg.WORLD_T_PEG_START)

        if self.initial_cfg.PEG_TYPE == "part":
            if (self.initial_cfg.TASK_NAME == 'AssemblyRegrasp-v0' and not self.initial_cfg.SHOW_GOAL) or \
                (self.initial_cfg.TASK_NAME == 'AssemblyPlace-v0' and self.initial_cfg.SHOW_GOAL):
                disable_gravity = False
            elif (self.initial_cfg.TASK_NAME in ['BeamScriptedInsert-v0', 'StoolScriptedInsert-v0'] and not self.initial_cfg.SHOW_GOAL):
                disable_gravity = False
            else:
                disable_gravity = True
            self.parts.append(
                RigidObjectCfg(
                    prim_path="/World/envs/env_.*/peg",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=self.initial_cfg.PEG_ASSET_NAME,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=disable_gravity,
                        )
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=world_T_peg_start.pos[0],
                                                              rot=world_T_peg_start.quat[0]),
                )
            )
        else:
            raise NotImplementedError

        # Add extra parts
        for ei, (extra_path, extra_pose) in enumerate(
                zip(self.initial_cfg.EXTRA_PARTS, self.initial_cfg.EXTRA_PARTS_STARTING_POSE)):
            ipose = IPose.from_pose(extra_pose)
            self.peripherals.append(
                AssetBaseCfg(
                    prim_path="/World/envs/env_.*/extra" + str(ei),
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=extra_path,
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(ipose.pos[0], ipose.quat[0])
                )
            )

        # Initialize the gripper to be directly above the target object pose
        # This is where the object is initialized to be reset later inside the fingers
        base_q = wxyz_to_xyzw(self.robot1.init_state.rot)
        world_T_tool = IPose.from_pose(self.initial_cfg.WORLD_T_PEG_START).multiply(
            IPose.from_pose(self.initial_cfg.PEG_T_TOOL))
        if (self.initial_cfg.TASK_NAME == 'AssemblyRegrasp-v0' and not self.initial_cfg.SHOW_GOAL) or \
                (self.initial_cfg.TASK_NAME == 'AssemblyPlace-v0' and self.initial_cfg.SHOW_GOAL):
            print('0 world_T_tool: ', world_T_tool, world_T_tool.pos[0].tolist())
            # ([0, 0.54, 0.16], [-0.5, 0.5, 0.5, -0.5])
            # world_T_flange = world_T_flange.multiply(
            #     (IPose.from_pose(([0.3, -0.3, 0.0], [0, 0, 0.7071068, 0.7071068]))))
            # world_T_flange = IPose.from_pose((world_T_flange.pos[0].tolist(), [0.5, 0.5, -0.5, -0.5]))
            # print('1 world_T_flange: ', world_T_flange)
            world_T_tool = world_T_tool.multiply((IPose.from_pose(([0.3, -0.3, 0.0], [0, 0, 0.7071068, 0.7071068]))))
            # print('2 world_T_tool: ', world_T_tool)
        # print('\n\n\n\n\n\n\n\n\n\n\n\nworld_T_tool: {}, self.initial_cfg.WORLD_T_PEG_START: {}, self.initial_cfg.PEG_T_TOOL: {}'.format(world_T_tool, self.initial_cfg.WORLD_T_HOLE_START, self.initial_cfg.PEG_T_TOOL))
        # if self.initial_cfg.TASK_NAME in ['AssemblyScriptedPlace-v0']:
        #     world_T_tool = world_T_tool.multiply((IPose.from_pose(([0, 0, 0], [ 0, 0, 0.7071068, 0.7071068 ]))))
            # ([0, 0.84, 0.46], [0, 0.7071, 0., -0.7071])
        # print(1, self.robot1.init_state.pos,
        #                         base_q,
        #                         world_T_flange.pos[0].cpu(),
        #                         wxyz_to_xyzw(world_T_flange.quat[0].cpu()))
        # (0.8367000122070312, 0.6095999755859375, 0.0225) [0, 0, 0.7071068, 0.7071068] tensor([0.0200, 0.8000, 0.4800]) [tensor(0.7071), tensor(0.), tensor(-0.7071), tensor(0.)]
        joint_angles = solve_ik(self.robot1.init_state.pos,
                                base_q,
                                world_T_tool.pos[0].cpu(),
                                wxyz_to_xyzw(world_T_tool.quat[0].cpu()))
        assert len(joint_angles) == len(EFFORT_JOINTS)
        for joint_name, joint_angle in zip(EFFORT_JOINTS, joint_angles):
            self.robot1.init_state.joint_pos[joint_name] = joint_angle
        self.joint_angles_list = {'robot1': joint_angles}
        if self.initial_cfg.num_robots == 2:
            # Find joint angles for the desired gripper pose
            base_q = wxyz_to_xyzw(self.robot2.init_state.rot)
            world_T_tool = IPose.from_pose(self.initial_cfg.WORLD_T_HOLE_START).multiply(
                IPose.from_pose(self.initial_cfg.HOLE_T_TOOL))
            joint_angles = solve_ik(self.robot2.init_state.pos,
                                    base_q,
                                    world_T_tool.pos[0].cpu(),
                                    wxyz_to_xyzw(world_T_tool.quat[0].cpu()))
            assert len(joint_angles) == len(EFFORT_JOINTS)
            for joint_name, joint_angle in zip(EFFORT_JOINTS, joint_angles):
                self.robot2.init_state.joint_pos[joint_name] = joint_angle
            self.joint_angles_list['robot2'] = joint_angles

@configclass
class MoveEnvCfg(MyEnvCfg):
    initial_cfg = MOVE_INITIAL_CFG


@configclass
class MoveToFixtureEnvCfg(MyEnvCfg):
    initial_cfg = MOVE_TO_FIXTURE_INITIAL_CFG

@configclass
class MoveToFixtureHorizontalEnvCfg(MyEnvCfg):
    initial_cfg = MOVE_TO_FIXTURE_HORIZONTAL_INITIAL_CFG


@configclass
class MoveToBoardFromFixtureEnvCfg(MyEnvCfg):
    initial_cfg = MOVE_TO_BOARD_FROM_FIXTURE_INITIAL_CFG

@configclass
class MoveToBoardFromGraspEnvCfg(MyEnvCfg):
    initial_cfg = MOVE_TO_BOARD_FROM_GRASP_INITIAL_CFG

@configclass
class InsertEnvCfg(MyEnvCfg):
    initial_cfg = INSERT_INITIAL_CFG


@configclass
class GraspEnvCfg(MyEnvCfg):
    initial_cfg = GRASP_INITIAL_CFG

@configclass
class RegraspEnvCfg(MyEnvCfg):
    initial_cfg = REGRASP_INITIAL_CFG

@configclass
class PlaceEnvCfg(MyEnvCfg):
    initial_cfg = PLACE_INITIAL_CFG

@configclass
class ScriptedPlaceEnvCfg(MyEnvCfg):
    initial_cfg = SCRIPTED_PLACE_INITIAL_CFG

@configclass
class ScriptedPlaceHorizontalEnvCfg(MyEnvCfg):
    initial_cfg = SCRIPTED_PLACE_HORIZONTAL_INITIAL_CFG

@configclass
class ScriptedGraspEnvCfg(MyEnvCfg):
    initial_cfg = SCRIPTED_GRASP_INITIAL_CFG

@configclass
class ScriptedGraspHorizontalEnvCfg(MyEnvCfg):
    initial_cfg = SCRIPTED_GRASP_HORIZONTAL_INITIAL_CFG

@configclass
class ScriptedRegraspEnvCfg(MyEnvCfg):
    initial_cfg = SCRIPTED_REGRASP_INITIAL_CFG

@configclass
class ScriptedInsertEnvCfg(MyEnvCfg):
    initial_cfg = SCRIPTED_INSERT_INITIAL_CFG

@configclass
class BeamScriptedInsertEnvCfg(MyEnvCfg):
    initial_cfg = BEAM_SCRIPTED_INSERT_INITIAL_CFG

@configclass
class BeamScriptedGraspEnvCfg(MyEnvCfg):
    initial_cfg = BEAM_SCRIPTED_GRASP_INITIAL_CFG

@configclass
class BeamScriptedPlaceEnvCfg(MyEnvCfg):
    initial_cfg = BEAM_SCRIPTED_PLACE_INITIAL_CFG

@configclass
class BeamMoveToBoardEnvCfg(MyEnvCfg):
    initial_cfg = BEAM_MOVE_TO_BOARD_INITIAL_CFG

@configclass
class BeamInsertEnvCfg(MyEnvCfg):
    initial_cfg = BEAM_INSERT_INITIAL_CFG

@configclass
class StoolInsertEnvCfg(MyEnvCfg):
    initial_cfg = STOOL_INSERT_INITIAL_CFG

@configclass
class StoolScriptedInsertEnvCfg(MyEnvCfg):
    initial_cfg = STOOL_SCRIPTED_INSERT_INITIAL_CFG

@configclass
class StoolScriptedGraspEnvCfg(MyEnvCfg):
    initial_cfg = STOOL_SCRIPTED_GRASP_INITIAL_CFG

@configclass
class StoolScriptedGraspHorizontalEnvCfg(MyEnvCfg):
    initial_cfg = STOOL_SCRIPTED_GRASP_HORIZONTAL_INITIAL_CFG

@configclass
class StoolScriptedPlaceEnvCfg(MyEnvCfg):
    initial_cfg = STOOL_SCRIPTED_PLACE_INITIAL_CFG

@configclass
class StoolMoveToBoardEnvCfg(MyEnvCfg):
    initial_cfg = STOOL_MOVE_TO_BOARD_INITIAL_CFG

class MyEnv(DirectRLEnv):
    cfg: MyEnvCfg

    # cnn: bool = MyEnvCfg.USE_CAMERA
    # image_width: int = IMAGE_WIDTH
    # image_height: int = IMAGE_HEIGHT

    def __init__(self, cfg: MyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.cnn = self.cfg.initial_cfg.USE_CAMERA
        self.image_width = self.cfg.initial_cfg.IMAGE_WIDTH
        self.image_height = self.cfg.initial_cfg.IMAGE_HEIGHT
        self.translation_limit = 0.05 * 5  # in m
        self.rotation_limit = 3 / 180 * np.pi  # in rad

        self.force_limit = FORCE_LIMIT_TRANS
        self.velocity_limit = VELOCITY_LIMIT_TRANS
        self.torque_limit = TORQUE_LIMIT

        self.trans_action_scale = 0.05
        # self.rot_action_scale = 0.25
        self.rot_action_scale = 0.1
        self.fixed_quat = {}
        self._ik_controllers = {}
        self._joint_ids = {}
        self._jacobi_body_idx = {}
        self._finger_joint_ids = {}
        self._body_idx = {}
        self.wrist_3_joint_id = {}
        self.gripper_status = {}
        self.gripper_step = 10
        self.previous_ee_pos = {}
        self.previous_ee_quat = {}
        self.num_episode_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.image_dir = 'logs/images/'
        os.makedirs(self.image_dir, exist_ok=True)
        self.collect_demo_flag = False # True
        self.demo = []
        self.demo_idx = 0
        self.demo_dir = os.path.join('logs/demos_isaaclab/')
        current_datetime_str = datetime.now().strftime('%Y%m%d%H%M%S')
        self.demo_name = current_datetime_str + '_' + self.cfg.initial_cfg.TASK_NAME[:-3]  # self.cfg.initial_cfg.SUBTASK
        os.makedirs(self.demo_dir, exist_ok=True)
        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.debug_action = self.cfg.initial_cfg.DEBUG_ACTION  # True
        # viewport
        self.enable_viewport_animation = False # True
        if self.enable_viewport_animation:
            self.current_eye = list(self.viewport_camera_controller.cfg.eye)
            self.current_lookat = self.viewport_camera_controller.cfg.lookat
            self.zoom_transition_param = 1

        if self.cfg.initial_cfg.SAVE_IMG:
            self.image_dir_epoch = os.path.join(self.image_dir, '0')
            os.makedirs(self.image_dir_epoch, exist_ok=True)
        for robot_name, robot_asset in self.robots.items():
            self.robots[robot_name] = robot_asset
            # resolve the joints over which the action term is applied
            self._joint_ids[robot_name], _ = robot_asset.find_joints(EFFORT_JOINTS)
            self._finger_joint_ids[robot_name], _ = robot_asset.find_joints(FINGER_JOINTS)
            self.wrist_3_joint_id[robot_name], _ = robot_asset.find_joints(['wrist_3_joint'])  # 5
            self.gripper_status[robot_name] = [None] * self.scene.num_envs
            self.previous_ee_pos[robot_name] = [None] * self.scene.num_envs
            self.previous_ee_quat[robot_name] = [None] * self.scene.num_envs
            self._num_joints = len(self._joint_ids[robot_name])

            # parse the body index
            # body_ids, _ = robot_asset.find_bodies("flange")
            body_ids, _ = robot_asset.find_bodies("tool0")

            # check if articulation is fixed-base
            # if fixed-base then the jacobian for the base is not computed
            # this means that number of bodies is one less than the articulation's number of bodies
            self._body_idx[robot_name] = body_ids[0]
            if robot_asset.is_fixed_base:
                self._jacobi_body_idx[robot_name] = body_ids[0] - 1
            else:
                self._jacobi_body_idx[robot_name] = body_ids[0]

            # Avoid indexing across all joints for efficiency
            if self._num_joints == robot_asset.num_joints:
                self._joint_ids[robot_name] = slice(None)

            ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                                            ik_method="dls")
            self._ik_controllers[robot_name] = DifferentialIKController(
                cfg=ik_controller_cfg, num_envs=self.scene.num_envs, device=self.sim.device
            )
            self._ik_controllers[robot_name].reset()

            if not self.cfg.initial_cfg.USE_FIXED_RIGID_ATTACHMENT or self.cfg.initial_cfg.TASK_NAME in ['AssemblyInsert-v0', 'BeamInsert-v0', 'StoolInsert-v0',
                                                                                                         'AssemblyMove-v0', 'AssemblyScriptedPlace-v0',
                                                                                                         'AssemblyScriptedPlaceHorizontal-v0',
                                                                                                         'AssemblyScriptedGrasp-v0',
                                                                                                         'AssemblyScriptedGraspHorizontal-v0',
                                                                                                         'AssemblyScriptedRegrasp-v0',
                                                                                                         'AssemblyScriptedInsert-v0', 
                                                                                                         'BeamScriptedInsert-v0', 
                                                                                                         'BeamScriptedGrasp-v0',
                                                                                                         'BeamScriptedPlace-v0',
                                                                                                         'BeamMoveToBoard-v0',
                                                                                                         'StoolScriptedInsert-v0',
                                                                                                         'StoolScriptedGrasp-v0',
                                                                                                         'StoolScriptedPlace-v0',
                                                                                                         'StoolMoveToBoard-v0', 'StoolScriptedGraspHorizontal-v0',]:
                print('[init] Closing gripper ...')
                gripper_joint_pos_limits = robot_asset.root_physx_view.get_dof_limits().to(self.device)
                self.gripper_dof_lower_limits = gripper_joint_pos_limits[..., 0]
                self.gripper_dof_upper_limits = gripper_joint_pos_limits[..., 1]
                print('gripper_joint_pos_limits: ', gripper_joint_pos_limits, gripper_joint_pos_limits.shape) # (num_env, 14, 2)
                # robot_asset.write_joint_state_to_sim(self.gripper_dof_lower_limits, 0)
                for i in range(self.gripper_step):
                    robot_asset.set_joint_velocity_target(self.cfg.initial_cfg.JOINT_VELOCITY,
                                                          joint_ids=self._finger_joint_ids[robot_name])
                    self.scene.write_data_to_sim()
                    # simulate
                    self.sim.step(render=True)
                    # update buffers at sim dt
                    self.scene.update(dt=self.physics_dt)

        if self.cfg.initial_cfg.USE_FIXED_RIGID_ATTACHMENT:
            self.robot1_fixed_joints = self.create_rigid_attachments(robot_path="robot1",
                                                                attachment_path=self.cfg.initial_cfg.PEG_ATTACHMENT_PRIM,
                                                                extra_attachments=self.cfg.initial_cfg.EXTRA_ATTACHMENTS)
            if (self.cfg.initial_cfg.num_robots == 2):
                self.robot2_fixed_joints = self.create_rigid_attachments(robot_path="robot2",
                                                                    attachment_path=self.cfg.initial_cfg.HOLE_ATTACHMENT_PRIM,
                                                                    extra_attachments=self.cfg.initial_cfg.EXTRA_ATTACHMENTS)

        # print('\n\n\n\n\n\n\nself.sim.render_mode, self.cfg.viewer: ', self.sim.render_mode, self.cfg.viewer)
        # self.sim.render_mode, self.cfg.viewer:  RenderMode.FULL_RENDERING ViewerCfg(eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0), cam_prim_path='/OmniverseKit_Persp', resolution=(1280, 720), origin_type='world', env_index=0, asset_name=None)

        # self.prev_joint_position = [None] * NUM_ENVS
        if self.cfg.initial_cfg.TASK_NAME in ['AssemblyMove-v0', 'AssemblyScriptedPlace-v0', 'AssemblyScriptedPlaceHorizontal-v0', 'AssemblyScriptedGrasp-v0', 'AssemblyScriptedGraspHorizontal-v0',
                                              'AssemblyScriptedRegrasp-v0', 'AssemblyScriptedInsert-v0', 'BeamScriptedInsert-v0', 'BeamScriptedGrasp-v0', 'BeamScriptedPlace-v0', 'BeamMoveToBoard-v0', 'StoolScriptedInsert-v0', 'StoolScriptedGrasp-v0', 'StoolScriptedPlace-v0', 'StoolMoveToBoard-v0', 'BeamInsert-v0', 'StoolInsert-v0', 'StoolScriptedGraspHorizontal-v0',
                                              ]:
            # Markers
            if MARKER_ENABLE:
                frame_marker_cfg = FRAME_MARKER_CFG.copy()
                frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
                self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

            # self.cfg.initial_cfg.PEG_GOAL[0][2] += 0.3
            # self.cfg.initial_cfg.PEG_GOAL[0][1] -= 0.23
            # Define simulation stepping
            if self.cfg.initial_cfg.SUBTASK in ['MoveToFixture', 'MoveToFixtureHorizontal']:
                self.des_world_T_tool = IPose.from_pose(self.cfg.initial_cfg.WORLD_T_FIXTURE_START).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_T_TOOL)).repeat(self.scene.num_envs).to_vec().clone()
            elif self.cfg.initial_cfg.TASK_NAME in [ 'AssemblyScriptedPlace-v0', 'AssemblyScriptedPlaceHorizontal-v0', 'AssemblyScriptedGrasp-v0', 'AssemblyScriptedGraspHorizontal-v0', 'AssemblyScriptedRegrasp-v0']:
                self.des_world_T_tool = IPose.from_pose(self.cfg.initial_cfg.WORLD_T_FIXTURE_START).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_T_TOOL)).repeat(self.scene.num_envs).to_vec().clone()
                # self.des_world_T_tool = IPose.from_pose(self.cfg.initial_cfg.WORLD_T_FIXTURE_START).multiply(
                #     IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)).multiply(
                #     IPose.from_pose(self.cfg.initial_cfg.PEG_T_TOOL).multiply(
                #     IPose.from_pose(([0, 0, 0], [ 0, 0, 0.7071068, 0.7071068 ])))).repeat(self.scene.num_envs).to_vec().clone()
            else:  # 'MoveToBoard', 'BeamScriptedGrasp-v0'
                # print('3'*50)
                self.des_world_T_tool = IPose.from_pose(self.cfg.initial_cfg.WORLD_T_HOLE_START).multiply(
                        IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)).multiply(
                        IPose.from_pose(self.cfg.initial_cfg.PEG_T_TOOL)).repeat(self.scene.num_envs).to_vec().clone()

            print('**********************************\nself.des_world_T_tool: {}, self.cfg.initial_cfg.WORLD_T_HOLE_START: {}, self.cfg.initial_cfg.PEG_GOAL: {}, self.cfg.initial_cfg.PEG_T_TOOL: {}'.format(self.des_world_T_tool, self.cfg.initial_cfg.WORLD_T_HOLE_START, self.cfg.initial_cfg.PEG_GOAL, self.cfg.initial_cfg.PEG_T_TOOL))
            joint_angles = solve_ik(self.cfg.robot1.init_state.pos,
                                    wxyz_to_xyzw(self.cfg.robot1.init_state.rot),
                                    self.des_world_T_tool[0, 0:3].cpu(),
                                    wxyz_to_xyzw(self.des_world_T_tool[0, 3:].cpu()))
                                    # torch.tensor([0.7071068, 0, -0.7071068, 0]))
            # print(self.cfg.initial_cfg.SUBTASK, self.cfg.initial_cfg.TASK_NAME)
            # if self.cfg.initial_cfg.SUBTASK in ['MoveToBoardFromFixture']:
            # joint_angles = (1.9511570085939227, -0.9153944958917811, 1.1867143011111283, -1.8421138814680684, -1.5707965000743658, -1.19043554990969)
            print('*'*16, joint_angles, self.cfg.robot1.init_state.pos,
                                    wxyz_to_xyzw(self.cfg.robot1.init_state.rot),
                                    self.des_world_T_tool[0, 0:3].cpu(),
                                    wxyz_to_xyzw(self.des_world_T_tool[0, 3:].cpu()))
            self.joint_angles_des = torch.from_numpy(np.asarray(joint_angles)).float().to(self.des_world_T_tool.device)
            # interpolated_joint_angles_np = interpolate_joint_angles(self.cfg.joint_angles_list['robot1'], joint_angles, self.max_episode_length//20)
            # self.interpolated_joint_angles_np = torch.from_numpy(interpolated_joint_angles_np).float().to(self.des_world_T_peg.device)
            # print('joint_angles: ', joint_angles, self.interpolated_joint_angles_np.shape)
    def update_rigid_attachments(self, robot_name, fixed_joints, obj_T_flange: IPose, rigid_object, mode: str):
        if mode == 'easy':
            # Domain randomization
            translation_offset = torch.zeros(self.scene.num_envs, 3).type(torch.FloatTensor).cuda()
            # translation_offset = (torch.rand(NUM_ENVS,3).type(torch.FloatTensor).cuda()-0.5)*0.000
            ero = torch.zeros(self.scene.num_envs, 3).type(torch.FloatTensor).cuda()
        else:
            translation_offset = (torch.rand(self.scene.num_envs, 3).type(torch.FloatTensor).cuda() - 0.5) * 2 * self.cfg.initial_cfg.RESET_POSITION_NOISE
            ero = torch.rand(self.scene.num_envs, 3).type(torch.FloatTensor).cuda() * self.cfg.initial_cfg.RESET_ROTATION_NOISE
            translation_offset[:, 2] = 0
            ero[:, :2] = 0
        rotation_offset = math_utils.quat_from_euler_xyz(ero[:, 0], ero[:, 1], ero[:, 2])
        # default_root_state = rigid_object.data.default_root_state[:].clone()
        # ee_pos_curr, ee_quat_curr = self._compute_frame_pose(robot_name, self.robots[robot_name])
        # print(mode, translation_offset, translation_offset.max(), translation_offset.min())
        for env_id, fixed_joint in enumerate(fixed_joints):
            trans_pose = IPose(pos=translation_offset[env_id:env_id + 1, :], quat=UNIT_QUAT)
            rot_pose = IPose(pos=UNIT_POINT, quat=rotation_offset[env_id:env_id + 1, :])
            flange_T_obj = trans_pose.multiply(obj_T_flange.invert()).multiply(rot_pose)
            # print('obj_T_flange: ', obj_T_flange)
            # obj_T_flange:  IPose(pos=tensor([[0.0000, 0.0000, 0.2911]], device='cuda:0'), quat=tensor([[0., 1., 0., 0.]], device='cuda:0'))
            # world_T_obj = IPose(ee_pos_curr[env_id:env_id+1, :], ee_quat_curr[env_id:env_id+1, :]).multiply(flange_T_obj)
            # default_root_state[env_id:env_id+1, 0:3] = world_T_obj.pos
            # default_root_state[env_id:env_id+1, 3:7] = world_T_obj.quat
            # default_root_state[env_id:env_id+1, 0:3] += self.scene.env_origins[env_id:env_id+1, :3]

            # set into the physics simulation
            fixed_joint.GetLocalPos0Attr().Set(Gf.Vec3f(*(flange_T_obj.pos[0].cpu().numpy().tolist())))
            fixed_joint.GetLocalRot0Attr().Set(Gf.Quatf(*(flange_T_obj.quat[0].cpu().numpy().tolist())))
            fixed_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            fixed_joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        # env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        # rigid_object.write_root_state_to_sim(default_root_state, env_ids=env_ids)

        return fixed_joint

    def create_rigid_attachments(self, attachment_path: str, robot_path: str, extra_attachments=[]):

        # Domain randomization
        fixed_joints = []
        for env_id in range(self.scene.num_envs):

            env0_path = f"/World/envs/env_{env_id}/{robot_path}"
            fixed_joint_path = env0_path + "/AssemblerFixedJoint"
            fixed_joint_path = find_unique_string_name(fixed_joint_path, lambda x: not prim_utils.is_prim_path_valid(x))
            stage = stage_utils.get_current_stage()
            fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)

            target0 = env0_path + "/tool0"
            target1 = f"/World/envs/env_{env_id}/{attachment_path}"

            if target0 is not None:
                fixed_joint.GetBody0Rel().SetTargets([target0])
            if target1 is not None:
                fixed_joint.GetBody1Rel().SetTargets([target1])

            fixed_joints.append(fixed_joint)

            for ex_id, (attachment_a, attachment_b) in enumerate(extra_attachments):
                ex_path = f"/World/envs/env_{env_id}/ExFixedJoint_{ex_id}"
                stage = stage_utils.get_current_stage()
                fixed_joint = UsdPhysics.FixedJoint.Define(stage, ex_path)
                if attachment_a is not None:
                    fixed_joint.GetBody0Rel().SetTargets([f"/World/envs/env_{env_id}/{attachment_a}"])
                if attachment_b is not None:
                    fixed_joint.GetBody1Rel().SetTargets([f"/World/envs/env_{env_id}/{attachment_b}"])
        return fixed_joints

    @property
    def robots(self):
        return {k: v for k, v in self.scene.articulations.items() if "robot" in k}

    @property
    def articulated_parts(self):
        return {k: v for k, v in self.scene.articulations.items() if "robot" not in k}

    def _setup_scene(self):
        robot1 = Articulation(self.cfg.robot1)
        if self.cfg.initial_cfg.num_robots == 2:
            robot2 = Articulation(self.cfg.robot2)

        rigid_bodies = [RigidObject(part) for part in self.cfg.parts]
        articulated_parts = [Articulation(apart) for apart in self.cfg.articulated_parts]

        for asset_cfg in self.cfg.peripherals:
            print('asset_cfg: ', asset_cfg)
            asset_cfg.spawn.func(
                asset_cfg.prim_path,
                asset_cfg.spawn,
                translation=asset_cfg.init_state.pos,
                orientation=asset_cfg.init_state.rot,
            )

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.articulations["robot1"] = robot1
        if self.cfg.initial_cfg.num_robots == 2:
            self.scene.articulations["robot2"] = robot2

        for rigid_body in rigid_bodies:
            self.scene.rigid_objects[rigid_body.cfg.prim_path.split("/")[-1]] = rigid_body

        for articulated_part in articulated_parts:
            self.scene.articulations[articulated_part.cfg.prim_path.split("/")[-1]] = articulated_part

        if self.cfg.initial_cfg.USE_CAMERA:
            if self.cfg.camera_type == 'tiled':
                self.scene.sensors["camera"] = TiledCamera(self.cfg.tiled_camera)
            else:
                self.scene.sensors["camera"] = Camera(self.cfg.tiled_camera)

    def set_mode(self, stage_name, init_cfg):
        self.scene = InteractiveScene(init_cfg.scene)
        self._setup_scene()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        # For moving around the scene before things start
        # print("Sleeping ")
        # for _ in range(2000):
        #     self.step_sim()
        #     time.sleep(0.01)

        mdp.reset_scene_to_default(self, env_ids)

        if self.cfg.initial_cfg.USE_FIXED_RIGID_ATTACHMENT:
            self.update_rigid_attachments("robot1", self.robot1_fixed_joints,
                                          obj_T_flange=IPose.from_pose(self.cfg.initial_cfg.PEG_T_TOOL),
                                          rigid_object=self.scene.rigid_objects["peg"], mode=self.cfg.initial_cfg.DIFFICULTY_LEVEL)
            if (INITIAL_CFG.num_robots == 2):
                self.update_rigid_attachments("robot2", self.robot2_fixed_joints,
                                              obj_T_flange=IPose.from_pose(self.cfg.initial_cfg.HOLE_T_TOOL),
                                              rigid_object=self.scene.rigid_objects["hole"], mode=self.cfg.initial_cfg.DIFFICULTY_LEVEL)
        if self.cfg.initial_cfg.DIFFICULTY_LEVEL == 'hard':
            # https://github.com/isaac-sim/IsaacLab/blob/4def7a692ba7e0b5b59e9d37266e5701e6a2f680/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/shadow_hand_env.py#L207-L239
            peg = self.scene.rigid_objects['peg']
            # print('self.scene.rigid_objects: ', self.scene.rigid_objects)
            # reset object
            object_default_state = peg.data.default_root_state.clone()[env_ids]
            # pos_noise = (torch.rand(self.scene.num_envs, 3).type(torch.FloatTensor).cuda() - 0.5)
            pos_noise = math_utils.sample_uniform(-self.cfg.initial_cfg.RESET_POSITION_NOISE, self.cfg.initial_cfg.RESET_POSITION_NOISE, (self.scene.num_envs, 3), device=self.device)
            # pos_noise[:, 0] = pos_noise[:, 0] - 100
            # pos_noise[:, 1] = pos_noise[:, 1] - 100
            print('initially:', object_default_state, pos_noise)
            # global object positions
            object_default_state[:, 0:3] = (
                    object_default_state[:, 0:3] + pos_noise + self.scene.env_origins[
                env_ids]
            )
            # print(2, object_default_state)
            # peg.write_root_state_to_sim(object_default_state, env_ids)

            # rot_noise = math_utils.sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
            # object_default_state[:, 3:7] = randomize_rotation(
            #     rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
            # )

            rot_noise = math_utils.sample_uniform(-self.cfg.initial_cfg.RESET_ROTATION_NOISE, self.cfg.initial_cfg.RESET_ROTATION_NOISE, (self.scene.num_envs, 3),
                                                  device=self.device)  # noise for X and Y rotation
            delta_rotation = randomize_rotation_rpy_quat(
                rot_noise[:, 0], rot_noise[:, 1], rot_noise[:, 2], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids]
            )
            object_default_state[:, 3:7] = delta_rotation
            object_default_state[:, 7:] = torch.zeros_like(peg.data.default_root_state[:, 7:])
            print('target: ', object_default_state)
            peg.write_root_state_to_sim(object_default_state)

            for robot_name, robot_asset in self.robots.items():
                if self.cfg.initial_cfg.TASK_NAME in ['AssemblyInsert-v0', 'BeamInsert-v0', 'StoolInsert-v0']:
                    # orig_joint_angles = robot_asset.data.joint_pos[:, self._joint_ids[robot_name]].clone()
                    # print('orig_joint_angles:', orig_joint_angles, orig_joint_angles.shape)
                    # randomized_joint_angles = orig_joint_angles + math_utils.sample_uniform(-self.cfg.initial_cfg.RESET_POSITION_NOISE, self.cfg.initial_cfg.RESET_POSITION_NOISE, (orig_joint_angles.shape[0], orig_joint_angles.shape[1]), device=self.device)
                    # print('randomized_joint_angles', randomized_joint_angles, randomized_joint_angles.shape)
                    # for i in range(self.gripper_step):
                    #     robot_asset.set_joint_position_target(randomized_joint_angles, self._joint_ids[robot_name], env_ids=env_ids)
                    #     self.scene.write_data_to_sim()
                    #     # simulate
                    #     self.sim.step(render=True)
                    #     # update buffers at sim dt
                    #     self.scene.update(dt=self.physics_dt)
                    tool_trans, tool_quat = self._compute_frame_pose(robot_name, robot_asset)

                    log_list(f"{robot_name}_isaac_tool_quat", tool_quat)
                    # tool_trans, _ = self._compute_frame_pose(robot_name, robot_asset)
                    # delta_rotm = math_utils.matrix_from_euler(delta[:, 3:6], "XYZ")

                    # Target absolute position
                    # log_list(f"{robot_name}_delta", delta)
                    control_cmd = torch.zeros(self.scene.num_envs, 7).to(self.sim.device)
                    control_cmd[:, :3] = tool_trans + pos_noise

                    control_cmd[:, 3:7] = math_utils.quat_mul(tool_quat, delta_rotation)
                    ee_pos_curr, ee_quat_curr = self._compute_frame_pose(robot_name, robot_asset)

                    # print('control_cmd: ', control_cmd)
                    self._ik_controllers[robot_name].set_command(control_cmd, ee_pos_curr, ee_quat_curr)
                    joint_pos = robot_asset.data.joint_pos[:, self._joint_ids[robot_name]]

                    jacobian = self._compute_frame_jacobian(robot_name, robot_asset)
                    joint_pos_des = self._ik_controllers[robot_name].compute(ee_pos_curr, ee_quat_curr, jacobian,
                                                                             joint_pos)

                    # set the joint position command
                    for i in range(self.gripper_step):
                        robot_asset.set_joint_position_target(joint_pos_des, self._joint_ids[robot_name], env_ids=env_ids)
                        self.scene.write_data_to_sim()
                        # simulate
                        self.sim.step(render=False)
                        # update buffers at sim dt
                        self.scene.update(dt=self.physics_dt)
                        # print(i, 'randomization:', self.scene.rigid_objects['peg'].data.body_state_w.clone())

        self.step_sim()
        self.initialized = False
        for robot_name, robot_asset in self.robots.items():
            self._ik_controllers[robot_name].reset(env_ids)
            _, self.fixed_quat[robot_name] = self._compute_frame_pose(robot_name, robot_asset)
            for env_id in env_ids:
                self.previous_ee_pos[robot_name][env_id] = None
                self.previous_ee_quat[robot_name][env_id] = None

            if not self.cfg.initial_cfg.USE_FIXED_RIGID_ATTACHMENT or self.cfg.initial_cfg.TASK_NAME in ['AssemblyInsert-v0', 'BeamInsert-v0', 'StoolInsert-v0']:
                # print('[reset] Closing gripper ...')
                for i in range(self.gripper_step):
                    robot_asset.set_joint_velocity_target(self.cfg.initial_cfg.JOINT_VELOCITY,
                                                          joint_ids=self._finger_joint_ids[robot_name], env_ids=env_ids)
                    self.scene.write_data_to_sim()
                    # simulate
                    self.sim.step(render=True)
                    # update buffers at sim dt
                    self.scene.update(dt=self.physics_dt)

            # if self.cfg.initial_cfg.TASK_NAME == 'AssemblyInsert-v0' and self.cfg.initial_cfg.DIFFICULTY_LEVEL not in ['easy']:
            #     debug = False # True
            #     # print('rigid_object: ', self.scene.rigid_objects)
            #     # random_choice = random.choice(self.cfg.initial_cfg.WORLD_T_PEG_START_LIST)
            #     # random_choice = self.cfg.initial_cfg.WORLD_T_PEG_START_LIST[-1]
            #     # peg = self.scene.rigid_objects['peg']
            #     # default_root_state = peg.data.default_root_state[env_ids].clone()
            #     # default_root_state[:, 0:3] = torch.tensor([random_choice[0]])
            #     # default_root_state[:, 0:3] += self.scene.env_origins[env_ids]
            #     # print('default_root_state: ', default_root_state, random_choice)
            #     # peg.write_root_state_to_sim(default_root_state, env_ids=env_ids)
            #     # random_choice_index = random.randint(0, len(self.cfg.initial_cfg.WORLD_T_PEG_START_LIST) - 1)
            #     # if random_choice_index == 0:
            #     #     continue
            #     # else:
            #     #     random_choice_tuple = self.cfg.initial_cfg.WORLD_T_PEG_START_LIST[random_choice_index]
            #     if debug:
            #         random_choice_tuple = self.cfg.initial_cfg.WORLD_T_PEG_START_LIST[1]
            #         # print(1, random_choice_tuple)
            #         random_choice_tuple = random_choice_tuple[0] + random_choice_tuple[1]
            #         # print(2, random_choice_tuple)
            #         tool_trans, tool_quat = self._compute_frame_pose(robot_name, robot_asset)
            #         control_cmd = torch.cat((tool_trans, tool_quat), dim=-1)
            #         # Target absolute position
            #         # print(1, control_cmd[env_ids])
            #         control_cmd[env_ids] = torch.tensor(random_choice_tuple).to(self.sim.device)
            #         # print(2, control_cmd[env_ids])
            #         ee_pos_curr, self.fixed_quat[robot_name] = self._compute_frame_pose(robot_name, robot_asset)
            #         # set command into controller
            #         self._ik_controllers[robot_name].set_command(control_cmd, ee_pos_curr, self.fixed_quat[robot_name])
            #         # self._ik_controllers[robot_name].set_command(control_cmd)
            #         for pre_reset_idx in range(500):
            #             # obtain quantities from simulation
            #             ee_pos_curr, self.fixed_quat[robot_name] = self._compute_frame_pose(robot_name, robot_asset)
            #             joint_pos = robot_asset.data.joint_pos[:, self._joint_ids[robot_name]]
            #             # print('joint_pos: ', joint_pos)
            #             # print('ee_pos_curr, self.fixed_quat[robot_name]: ', ee_pos_curr, self.fixed_quat[robot_name])
            #             # compute the delta in joint-space
            #             jacobian = self._compute_frame_jacobian(robot_name, robot_asset)
            #             joint_pos_des = self._ik_controllers[robot_name].compute(ee_pos_curr, self.fixed_quat[robot_name], jacobian,
            #                                                                         joint_pos)
            #
            #             # set the joint position command
            #             print(pre_reset_idx, 'joint_pos_des: ', joint_pos_des)
            #             robot_asset.set_joint_position_target(joint_pos_des, self._joint_ids[robot_name], env_ids=env_ids)
            #
            #             # set actions into simulator
            #             self.scene.write_data_to_sim()
            #             # simulate
            #             self.sim.step(render=True)
            #             # update buffers at sim dt
            #             self.scene.update(dt=self.physics_dt)
            #     else:
            #         random_choice_index = random.randint(0, len(self.cfg.initial_cfg.WORLD_T_PEG_START_LIST) - 1)
            #         if random_choice_index == 0:
            #             continue
            #         joint_pos_des = torch.tensor([[ 5.0120,  4.6390, -2.3015, -3.9133, -1.5706,  0.2996]]).to(self.sim.device)
            #         for pre_reset_idx in range(400):
            #             robot_asset.set_joint_position_target(joint_pos_des, self._joint_ids[robot_name], env_ids=env_ids)
            #             # set actions into simulator
            #             self.scene.write_data_to_sim()
            #             # simulate
            #             self.sim.step(render=False)
            #             # update buffers at sim dt
            #             self.scene.update(dt=self.physics_dt)
        # print('[MyEnv] Resetting {}'.format(env_ids))
        # self.step_counter = 0
        self.num_episode_buf[env_ids] += 1
        if self.cfg.initial_cfg.SAVE_IMG:
            self.image_dir_epoch = os.path.join(self.image_dir, str(int(self.num_episode_buf[0].item())))
            os.makedirs(self.image_dir_epoch, exist_ok=True)
        if self.cfg.initial_cfg.TASK_NAME in ['AssemblyMove-v0', 'AssemblyScriptedPlace-v0', 'AssemblyScriptedPlaceHorizontal-v0', 'AssemblyScriptedGrasp-v0', 'AssemblyScriptedGraspHorizontal-v0',
                                              'AssemblyScriptedRegrasp-v0', 'AssemblyScriptedInsert-v0', 'BeamScriptedInsert-v0', 'BeamScriptedGrasp-v0', 'BeamScriptedPlace-v0', 'BeamMoveToBoard-v0', 'StoolScriptedInsert-v0', 'StoolScriptedGrasp-v0', 'StoolScriptedGraspHorizontal-v0','StoolScriptedPlace-v0', 'StoolMoveToBoard-v0']:
            if self.cfg.initial_cfg.SUBTASK in ['MoveToFixture', 'MoveToFixtureHorizontal']:
                self.des_world_T_tool = IPose.from_pose(self.cfg.initial_cfg.WORLD_T_FIXTURE_START).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_T_TOOL)).repeat(self.scene.num_envs).to_vec().clone()
            elif self.cfg.initial_cfg.TASK_NAME in ['AssemblyScriptedPlace-v0', 'AssemblyScriptedPlaceHorizontal-v0', 'AssemblyScriptedGrasp-v0', 'AssemblyScriptedGraspHorizontal-v0', 'AssemblyScriptedRegrasp-v0']:
                self.des_world_T_tool = IPose.from_pose(self.cfg.initial_cfg.WORLD_T_FIXTURE_START).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_T_TOOL)).repeat(self.scene.num_envs).to_vec().clone()
                # self.des_world_T_tool = IPose.from_pose(self.cfg.initial_cfg.WORLD_T_FIXTURE_START).multiply(
                #     IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)).multiply(
                #     IPose.from_pose(self.cfg.initial_cfg.PEG_T_TOOL).multiply(
                #     IPose.from_pose(([0, 0, 0], [0, 0, 0.7071068, 0.7071068])))).repeat(self.scene.num_envs).to_vec().clone()
            else:  # 'MoveToBoard', 'BeamScriptedGrasp-v0'
                self.des_world_T_tool = IPose.from_pose(self.cfg.initial_cfg.WORLD_T_HOLE_START).multiply(
                        IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)).multiply(
                        IPose.from_pose(self.cfg.initial_cfg.PEG_T_TOOL)).repeat(self.scene.num_envs).to_vec().clone()
            for robot_name, robot_asset in self.robots.items():
                # self._ik_controllers[robot_name].set_command(self.des_world_T_peg)
                ee_pose_w = robot_asset.data.body_state_w[:, self._body_idx[robot_name], 0:7]
                # update marker positions
                if MARKER_ENABLE:
                    self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                    self.goal_marker.visualize(self.des_world_T_tool[:, 0:3] + self.scene.env_origins, self.des_world_T_tool[:, 3:7])
        
        if self.collect_demo_flag:
            if len(self.demo) > 0: 
                demo_path = os.path.join(self.demo_dir, '{:04d}_{}.pkl'.format(self.demo_idx, self.demo_name))
                with open(demo_path, 'wb') as file:
                    pickle.dump(self.demo, file)
                    print('Save demo to {}'.format(demo_path))

            self.demo = []
            self.demo_idx += 1

            obs = self._get_observations()
            self.demo.append([obs, self.cfg.initial_cfg.TASK_NAME, self.cfg.initial_cfg.SUBTASK])

    def step_sim(self):
        for robot_name, robot_asset in self.robots.items():
            _joint_ids, _ = robot_asset.find_joints(EFFORT_JOINTS)
            current_joint_pos = robot_asset.data.joint_pos[:, _joint_ids]
            robot_asset.set_joint_position_target(current_joint_pos, joint_ids=_joint_ids)
            robot_asset.set_joint_velocity_target(0, joint_ids=_joint_ids)

            self.scene.write_data_to_sim()
            self.sim.step(render=True)
            self.scene.update(dt=self.physics_dt)

    def _pre_physics_step_sr(self, actions, robot_name, robot_asset):
        log_list(f"{robot_name}_isaac_action", actions)
        # self.step_counter = self.step_counter + 1
        # print(self.step_counter, self.physics_dt, self.max_episode_length)
        # (0.008, 768)
        # For debugging, go straight down
        #     actions = torch.zeros([self.scene.num_envs, self.cfg.num_actions]).cuda()
        #     actions[:, 0] = -1.0
        rotation_enabled = self.cfg.initial_cfg.allow_peg_rotation and robot_name == "robot1" or \
                           self.cfg.initial_cfg.allow_hole_rotation and robot_name == "robot2"
        
        # actions = torch.zeros((self.scene.num_envs, self.cfg.num_actions)).cuda()
        # actions[:, 2] = -1
        # print('actions.shape: ', actions)  # [2, 3]
        if self.cfg.initial_cfg.allow_gripper_status:
            self.gripper_status[robot_name] = (actions[:, 6:7] > 0) * 2.0 - 1.0  # [-1, 1]

        processed_actions = actions.clone()
        processed_actions[:, :3] = actions[:, :3] * self.trans_action_scale
        processed_actions[:, 3:6] = actions[:, 3:6] * self.rot_action_scale

        delta = torch.zeros(self.scene.num_envs, self.cfg.num_actions, device=self.sim.device)
        delta[:, :3] = processed_actions[:,
                       :3]  # / (torch.linalg.norm(self._processed_actions[:, :3], ord=2) + 1e-3) * self.translation_limit
        # delta[:, 3:6] = self._processed_actions[:, 3:6] / (torch.linalg.norm(self._processed_actions[:, 3:6], ord=2) + 1e-3) * self.rotation_limit
        # print('delta: ', delta)
        if (rotation_enabled):
            delta[:, 3:6] = processed_actions[:,
                            3:6]  # / (torch.linalg.norm(self._processed_actions[:, 3:6], ord=2) + 1e-3) * self.rotation_limit
            delta_rotm = math_utils.matrix_from_euler(delta[:, 3:6], "XYZ")
            delta_rotation = math_utils.quat_from_matrix(delta_rotm)

        tool_trans, tool_quat = self._compute_frame_pose(robot_name, robot_asset)

        log_list(f"{robot_name}_isaac_tool_quat", tool_quat)
        # tool_trans, _ = self._compute_frame_pose(robot_name, robot_asset)
        # delta_rotm = math_utils.matrix_from_euler(delta[:, 3:6], "XYZ")

        # Target absolute position
        # log_list(f"{robot_name}_delta", delta)
        control_cmd = torch.zeros(self.scene.num_envs, 7).to(self.sim.device)
        control_cmd[:, :3] = tool_trans + delta[:, :3]
        # delta_rotation = math_utils.quat_from_matrix(delta_rotm)
        if (rotation_enabled):
            control_cmd[:, 3:7] = math_utils.quat_mul(tool_quat, delta_rotation)
        else:
            control_cmd[:, 3:7] = self.fixed_quat[robot_name]  #math_utils.quat_mul(self.fixed_quat, delta_rotation)
        # self.control_cmd[:, 3:7] = math_utils.quat_mul(tool_quat, delta_rotation)

        if self.debug_action:
            print(
                4, 'step: {}, tool_trans: {}, delta[:3]: {}, delta[3:]: {}'.format(str(int(self.num_episode_buf[0].item())), tool_trans[0], delta[0, :3],
                                                                                      delta[0, 3:]))
            print(1, 'step: {}, control_cmd: {}'.format(str(int(self.num_episode_buf[0].item())), control_cmd[0]))
            print(2,'step: {}, action: {}'.format(str(int(self.num_episode_buf[0].item())), actions[0]))

        if False and self.cfg.initial_cfg.TASK_NAME in ['AssemblyMove-v0']:
            control_cmd = self.des_world_T_tool.clone()  # .detach()
            control_cmd[:, :3] = control_cmd[:, :3] + self.scene.env_origins
            # obtain quantities from simulation
            ee_pose_w = robot_asset.data.body_state_w[:, self._body_idx[robot_name], 0:7]
            root_pose_w = robot_asset.data.root_state_w[:, 0:7]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            recovered_ee_pos_b, recovered_ee_quat_b = math_utils.combine_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pos_b, ee_quat_b
            )
            current_ee_pose_from_root = torch.cat((ee_pos_b, ee_quat_b),dim=1)

            control_cmd[:, 3:] = ee_pose_w[:, 3:7].clone()
            control_cmd[:, 2:3] = ee_pose_w[:, 2:3].clone()
            control_cmd_pos_b, control_cmd_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], control_cmd[:, 0:3], control_cmd[:, 3:7]
            )
            recovered_control_cmd_pos_b, recovered_control_cmd_quat_b = math_utils.combine_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], control_cmd_pos_b, control_cmd_quat_b
            )
            control_cmd_from_root = torch.cat((control_cmd_pos_b, control_cmd_quat_b),dim=1)

            next_ee_pose_w = ee_pose_w.clone()
            # print(0, control_cmd)
            # print(1, next_ee_pose_w)
            # next_ee_pose_w[:, 1] = next_ee_pose_w[:, 1] + 0.05
            next_ee_pose_w[:, 0] = next_ee_pose_w[:, 0] - (control_cmd[:, 1]-next_ee_pose_w[:, 1])
            next_ee_pose_w[:, 1] = next_ee_pose_w[:, 1] + (control_cmd[:, 0]-next_ee_pose_w[:, 0])
            # print(2, next_ee_pose_w)
            # next_ee_pose_w[0, 0] = 2.0320
            # next_ee_pose_w[1, 0] = -1.9679
            # next_ee_pose_w[0, 0] = 0.3103
            # next_ee_pose_w[1, 0] = 0.3103
            # next_ee_pose_w[:, 2] = next_ee_pose_w[:, 2] + 0.5
            next_ee_pos_b, next_ee_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], next_ee_pose_w[:, 0:3], next_ee_pose_w[:, 3:7]
            )
            recovered_next_ee_pos_b, recovered_next_ee_quat_b = math_utils.combine_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], next_ee_pos_b, next_ee_quat_b
            )
            next_ee_pose_from_root = torch.cat((next_ee_pos_b, next_ee_quat_b),dim=1)

            # print('control_cmd: {}, \nee_pose_w: {}'.format(control_cmd, ee_pose_w))
            # print('control_cmd: {}, \nee_pose_w: {}, \nWORLD_T_HOLE_START: {}, \nPEG_GOAL: {}'.format(control_cmd, ee_pose_w,
            #                                                     self.cfg.initial_cfg.WORLD_T_HOLE_START, self.cfg.initial_cfg.PEG_GOAL))
            # interpolated_poses = next_pose(current_ee_pose_from_root, control_cmd, 100)
            control_cmd = next_ee_pose_from_root # control_cmd_from_root  # current_ee_pose_from_root  # relative to root frame
            # control_cmd[:, :3] = interpolated_poses[1:2, :3]
            # print('control_cmd: {}'.format(control_cmd))
            # update marker positions
            self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            # print(ee_pose_w[:, 3:7], current_ee_pose_from_root[:, 3:7])
            self.goal_marker.visualize(recovered_next_ee_pos_b, recovered_next_ee_quat_b)
            # self.goal_marker.visualize(recovered_control_cmd_pos_b, recovered_control_cmd_quat_b)
            # self.goal_marker.visualize(recovered_ee_pos_b, recovered_ee_quat_b)
            # self.goal_marker.visualize(root_pose_w[:, 0:3], root_pose_w[:, 3:7])
            # self.goal_marker.visualize(self.scene.env_origins, root_pose_w[:, 3:7])
            # self.goal_marker.visualize(self.des_world_T_peg.clone()[:, 0:3] + self.scene.env_origins, self.des_world_T_peg.clone()[:, 3:7])
            # self.goal_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose(robot_name, robot_asset)

        # print('control_cmd: ', control_cmd)
        # set command into controller
        self._ik_controllers[robot_name].set_command(control_cmd, ee_pos_curr, ee_quat_curr)

        log_list(f"{robot_name}_isaac_cmd", control_cmd)

        if self.enable_viewport_animation:
            print('self.current_eye: ', self.current_eye)
            # self.current_eye[1] += 0.0002 * self.zoom_transition_param
            # self.current_eye[2] += 0.0003 * self.zoom_transition_param
            self.current_eye[1] += 0.002 * self.zoom_transition_param
            self.current_eye[2] += 0.003 * self.zoom_transition_param
            # self.current_lookat = [0.85, 0.25, -0.15]
            self.viewport_camera_controller.update_view_location(eye=self.current_eye, lookat=self.current_lookat)
            self.zoom_transition_param += 1

    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        if (self.cfg.initial_cfg.num_robots == 1):
            self._pre_physics_step_sr(actions, "robot1", self.robots["robot1"])
        elif (self.cfg.initial_cfg.num_robots == 2):
            # robot_1 trans (3) + robot_1 rot (3) + robot_1 gripper (1) + robot_2 trans (3) + robot_2 rot (3) + robot_2 gripper (1)
            num_r1_actions = 3 + 3 * int(self.cfg.initial_cfg.allow_peg_rotation) + 1 * int(self.cfg.initial_cfg.allow_gripper_status)
            self._pre_physics_step_sr(actions[:, :num_r1_actions], "robot1", self.robots["robot1"])
            self._pre_physics_step_sr(actions[:, num_r1_actions:], "robot2", self.robots["robot2"])

    def _apply_action(self) -> None:
        for robot_name, robot_asset in self.robots.items():
            if self.cfg.initial_cfg.TASK_NAME in ['AssemblyMove-v0', 'AssemblyScriptedGrasp-v0', 'AssemblyScriptedGraspHorizontal-v0', 'AssemblyScriptedInsert-v0', 'BeamScriptedInsert-v0',
                                                  'BeamScriptedGrasp-v0', 'BeamScriptedPlace-v0', 'BeamMoveToBoard-v0', 'StoolScriptedInsert-v0', 'StoolScriptedGrasp-v0', 'StoolScriptedGraspHorizontal-v0','StoolScriptedPlace-v0', 'StoolMoveToBoard-v0', ]:
                # print(self.step_counter, self.physics_dt, self.max_episode_length, self.joint_angles_des)
                robot_asset.set_joint_position_target(self.joint_angles_des, self._joint_ids[robot_name])
            elif self.cfg.initial_cfg.TASK_NAME in ['AssemblyScriptedPlace-v0', 'AssemblyScriptedPlaceHorizontal-v0', 'AssemblyScriptedRegrasp-v0', 'StoolScriptedGraspHorizontal-v0',]:
                # print(self.step_counter, self.physics_dt, self.max_episode_length, self.joint_angles_des)
                if self.episode_length_buf[0] < 20:
                    print('_apply_action dummy')
                else:
                    robot_asset.set_joint_position_target(self.joint_angles_des, self._joint_ids[robot_name])
            else:
                # obtain quantities from simulation
                ee_pos_curr, ee_quat_curr = self._compute_frame_pose(robot_name, robot_asset)
                joint_pos = robot_asset.data.joint_pos[:, self._joint_ids[robot_name]]
                # compute the delta in joint-space
                if ee_quat_curr.norm() != 0:
                    jacobian = self._compute_frame_jacobian(robot_name, robot_asset)
                    joint_pos_des = self._ik_controllers[robot_name].compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
                else:
                    joint_pos_des = joint_pos.clone()

                error = torch.abs(joint_pos - joint_pos_des)
                log_list(f"{robot_name}_joint_error", error)

                # set the joint position command
                robot_asset.set_joint_position_target(joint_pos_des, self._joint_ids[robot_name])

            # fingers
            if (self.cfg.initial_cfg.num_robots == 1) and self.cfg.initial_cfg.allow_gripper_status:
                # print('self._finger_joint_ids: ', self._finger_joint_ids)
                # {'robot1': [7]}
                # current_finger_joint_pos = robot_asset.data.joint_pos[:, self._finger_joint_ids['robot1']]
                # print('1. current_finger_joint_pos: ', current_finger_joint_pos)
                current_finger_joint_pos = self.gripper_status[robot_name]  # current_finger_joint_pos + 0.5  # [0, 1]
                # print(current_finger_joint_pos)
                # print(24 * current_finger_joint_pos)
                # print('2. current_finger_joint_pos: ', current_finger_joint_pos)
                # robot_asset.set_joint_position_target(current_finger_joint_pos, joint_ids=self._finger_joint_ids[robot_name])
                # current_finger_joint_pos = robot_asset.data.joint_pos[:, self._finger_joint_ids['robot1']]
                # print('dir(robot_asset): ', dir(robot_asset), current_finger_joint_pos)
                # print('3. current_finger_joint_pos: ', current_finger_joint_pos)
                for i in range(self.gripper_step):
                    # > 0: close, < 0: open
                    robot_asset.set_joint_velocity_target(self.cfg.initial_cfg.JOINT_VELOCITY * current_finger_joint_pos,
                                                          joint_ids=self._finger_joint_ids[robot_name])
                    self.scene.write_data_to_sim()
                    # simulate
                    self.sim.step(render=True)
                    # update buffers at sim dt
                    self.scene.update(dt=self.physics_dt)

            if self.cfg.initial_cfg.TASK_NAME in ['AssemblyScriptedPlace-v0', 'AssemblyScriptedPlaceHorizontal-v0',] and self.episode_length_buf[0] > self.max_episode_length - 3:
                for i in range(self.gripper_step):
                    # > 0: close, < 0: open
                    robot_asset.set_joint_velocity_target(-self.cfg.initial_cfg.JOINT_VELOCITY,
                                                          joint_ids=self._finger_joint_ids[robot_name])
                    self.scene.write_data_to_sim()
                    # simulate
                    self.sim.step(render=True)
                    # update buffers at sim dt
                    self.scene.update(dt=self.physics_dt)
            # print('apply action:', self.scene.rigid_objects['peg'].data.body_state_w.clone())
            
        if self.collect_demo_flag:
            obs = self._get_observations()
            self.demo.append([obs, self.cfg.initial_cfg.TASK_NAME, self.cfg.initial_cfg.SUBTASK])
            # demo_path = os.path.join(self.demo_dir, '{:04d}_{}.pkl'.format(self.demo_idx, self.demo_name))
            # with open(demo_path, 'wb') as file:
            #     pickle.dump(self.demo, file)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def ee_pos(self, asset_cfg: SceneEntityCfg, obj_cfg: SceneEntityCfg, use_pose: bool = False, relative_pose_obs: bool = False):
        """The joint velocities of the asset w.r.t. the default joint velocities.

        Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
        """
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self.scene[asset_cfg.name]

        # obtain quantities from simulation
        # body_ids, _ = asset.find_bodies("flange")
        body_ids, _ = asset.find_bodies("tool0")
        # iflange = pose_from_state(asset.data.body_state_w, body_ids[0], self.scene.env_origins)
        # flange_pose = iflange.multiply(IPose.from_pose(FLANGE_T_TOOL).repeat(self.scene.num_envs))
        flange_pose = pose_from_state(asset.data.body_state_w, body_ids[0], self.scene.env_origins)

        # print("\n\n\n\n\n********************** relative_pose_obs: {} ********************".format(relative_pose_obs))
        if relative_pose_obs:
            world_T_tool = flange_pose

            if self.cfg.initial_cfg.TASK_NAME in ['AssemblyInsert-v0', 'AssemblyScriptedInsert-v0', 'BeamScriptedGrasp-v0', 'BeamScriptedPlace-v0', 'BeamMoveToBoard-v0', 'StoolScriptedGrasp-v0', 'StoolScriptedPlace-v0', 'StoolMoveToBoard-v0', 'BeamInsert-v0', 'StoolInsert-v0', 'StoolScriptedGraspHorizontal-v0',]:
                des_world_T_tool = IPose.from_pose(self.cfg.initial_cfg.WORLD_T_HOLE_START).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_T_TOOL))
            elif self.cfg.initial_cfg.TASK_NAME in ['MoveToFixture', 'MoveToFixtureHorizontal',
                                                'AssemblyScriptedPlace-v0', 'AssemblyScriptedPlaceHorizontal-v0',
                                                'AssemblyScriptedGrasp-v0', 'AssemblyScriptedGraspHorizontal-v0',
                                                'AssemblyScriptedRegrasp-v0']:
                des_world_T_tool = IPose.from_pose(self.cfg.initial_cfg.WORLD_T_FIXTURE_START).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_T_TOOL))
            elif self.cfg.initial_cfg.TASK_NAME in ['BeamScriptedInsert-v0', 'StoolScriptedInsert-v0']:
                des_world_T_tool = IPose.from_pose(self.cfg.initial_cfg.WORLD_T_HOLE_START).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_T_TOOL))
            else:
                raise NotImplementedError(
                    '{} _get_observations relative_pose_obs is not Implemented'.format(self.cfg.initial_cfg.TASK_NAME))
            flange_pose = des_world_T_tool.repeat(self.scene.num_envs).multiply(world_T_tool.invert())

        ee_rotm = math_utils.matrix_from_quat(flange_pose.quat)

        if (use_pose):
            obj_asset = self.scene[obj_cfg.name]
            body_ids, _ = obj_asset.find_bodies(".*")
            obj_pose = pose_from_state(obj_asset.data.body_state_w, body_ids[0], self.scene.env_origins)
            obj_rotm = math_utils.matrix_from_quat(obj_pose.quat)
            total_obs = torch.cat(
                [flange_pose.pos, ee_rotm[:, :3, 0], ee_rotm[:, :3, 1], obj_rotm[:, :3, 0], obj_rotm[:, :3, 1]], axis=1)
        else:
            # without object pose
            total_obs = torch.cat([flange_pose.pos, ee_rotm[:, :3, 0], ee_rotm[:, :3, 1]], axis=1)

        log_list(f"{asset_cfg.name}_isaac_obs", total_obs)

        return total_obs

    def depth_image(self, tiled_camera: SceneEntityCfg):
        camera_image = tiled_camera.data.output["depth"].clone()

        # Save the first camera image for debugging purposes
        # save_depth(torch.squeeze(camera_image[0, :, :]).cpu().numpy())    
        # import sys
        # sys.exit()

        return camera_image

    def _get_observations(self, relative_pose=True) -> dict:

        # read total overall joint forces and torques via
        # print(dir(self.robots['robot1']))
        # sensor_joint_forces = self.robots['robot1']._articulation_view.get_measured_joint_forces()
        # print('sensor_joint_forces', sensor_joint_forces)
        robot_name = 'robot1'

        obs = self.ee_pos(ROBOT1_EFFORT_CFG, obj_cfg=SceneEntityCfg("peg"), use_pose=False,
                          relative_pose_obs=self.cfg.initial_cfg.relative_pose_obs)

        if self.cfg.initial_cfg.GRIPPER_STATUS_OBS:
            obs = torch.cat([obs, self.gripper_status[robot_name]], dim=1)

        if self.cfg.initial_cfg.VELOCITY_OBS:
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose(robot_name, self.robots[robot_name])
            for i in range(ee_pos_curr.shape[0]):
                if(self.previous_ee_pos[robot_name][i] is None or self.previous_ee_quat[robot_name][i] is None):
                    self.previous_ee_pos[robot_name][i] = ee_pos_curr[i]
                    self.previous_ee_quat[robot_name][i] = ee_quat_curr[i]
                    # self.previous_ee_pos[robot_name] = torch.cat(self.previous_ee_pos[robot_name], dim=0)
                    # self.previous_ee_quat[robot_name] = torch.cat(self.previous_ee_quat[robot_name], dim=0)
            ee_linvel_curr = (ee_pos_curr-torch.stack(self.previous_ee_pos[robot_name], dim=0))/(SIM_DT*SUBSTEPS)

            quat_diff = self.get_quat_diff(torch.stack(self.previous_ee_quat[robot_name], dim=0), ee_quat_curr)
            ee_angvel_curr = math_utils.axis_angle_from_quat(quat_diff)/(SIM_DT*SUBSTEPS)
            self.previous_ee_pos[robot_name] = list(torch.unbind(ee_pos_curr, dim=0))
            self.previous_ee_quat[robot_name] = list(torch.unbind(ee_quat_curr, dim=0))

            obs = torch.cat([obs, ee_linvel_curr, ee_angvel_curr], dim=1)

        if self.cfg.initial_cfg.USE_FT_SENSOR:
            if self.cfg.initial_cfg.FT_SENSOR_TYPE == 'ArticulationView':
                robot1_ArticulationView = ArticulationView(prim_paths_expr="/World/envs/env_.*/robot1")
                robot1_ArticulationView.initialize()

                # reading the force/torque sensor values at the panda_hand_joint:
                force_torque = robot1_ArticulationView.get_measured_joint_forces()[:,
                               self.wrist_3_joint_id['robot1']]  # [0][9] [2, 20, 6]
                # print('force_torque: ', force_torque, force_torque.shape)
                # print('force_torque: ', force_torque.shape, self.wrist_3_joint_id[robot_name])  # [2, 1, 6], [5]
            elif self.cfg.initial_cfg.FT_SENSOR_TYPE == 'physx_view':
                # print('self._joint_ids: ', self._jacobi_body_idx[robot_name], self._joint_ids[robot_name]) # 8 [0, 1, 2, 3, 4, 5]
                jacobian = self.robots[robot_name].root_physx_view.get_jacobians()[:, self._jacobi_body_idx[robot_name],
                           :,
                           self._joint_ids[robot_name]]
                jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
                joint_torques = self.robots[robot_name].root_physx_view.get_dof_projected_joint_forces()[:,
                                self._joint_ids[robot_name]]
                end_effector_forces = jacobian_T @ joint_torques.unsqueeze(-1)
                end_effector_forces = end_effector_forces.squeeze(-1)
                # print('end_effector_forces: ', end_effector_forces, end_effector_forces.shape)
            else:
                pass  # No ft sensor observation
            obs = torch.cat([obs, end_effector_forces], dim=1)

        if self.cfg.initial_cfg.num_robots == 2:
            obs = torch.cat([obs, self.ee_pos(ROBOT2_EFFORT_CFG, obj_cfg=SceneEntityCfg("hole"), use_pose=False,
                                              relative_pose_obs=self.cfg.initial_cfg.relative_pose_obs),],
                            dim=1)
            robot2_name = 'robot2'
            if self.cfg.initial_cfg.GRIPPER_STATUS_OBS:
                obs = torch.cat([obs, self.gripper_status[robot2_name]], dim=1)

            if self.cfg.initial_cfg.VELOCITY_OBS:
                ee_pos_curr, ee_quat_curr = self._compute_frame_pose(robot2_name, self.robots[robot2_name])
                for i in range(ee_pos_curr.shape[0]):
                    if (self.previous_ee_pos[robot2_name][i] is None or self.previous_ee_quat[robot2_name][i] is None):
                        self.previous_ee_pos[robot2_name][i] = ee_pos_curr[i]
                        self.previous_ee_quat[robot2_name][i] = ee_quat_curr[i]
                        # self.previous_ee_pos[robot2_name] = torch.cat(self.previous_ee_pos[robot2_name], dim=0)
                        # self.previous_ee_quat[robot2_name] = torch.cat(self.previous_ee_quat[robot2_name], dim=0)
                ee_linvel_curr = (ee_pos_curr - torch.stack(self.previous_ee_pos[robot_name], dim=0)) / (
                            SIM_DT * SUBSTEPS)

                quat_diff = self.get_quat_diff(torch.stack(self.previous_ee_quat[robot2_name], dim=0), ee_quat_curr)
                ee_angvel_curr = math_utils.axis_angle_from_quat(quat_diff) / (SIM_DT * SUBSTEPS)
                self.previous_ee_pos[robot2_name] = list(torch.unbind(ee_pos_curr, dim=0))
                self.previous_ee_quat[robot2_name] = list(torch.unbind(ee_quat_curr, dim=0))

                obs = torch.cat([obs, ee_linvel_curr, ee_angvel_curr], dim=1)

            if self.cfg.initial_cfg.USE_FT_SENSOR:
                if self.cfg.initial_cfg.FT_SENSOR_TYPE == 'ArticulationView':
                    robot1_ArticulationView = ArticulationView(prim_paths_expr="/World/envs/env_.*/robot1")
                    robot1_ArticulationView.initialize()

                    # reading the force/torque sensor values at the panda_hand_joint:
                    force_torque = robot1_ArticulationView.get_measured_joint_forces()[:,
                                   self.wrist_3_joint_id['robot1']]  # [0][9] [2, 20, 6]
                    # print('force_torque: ', force_torque, force_torque.shape)
                    # print('force_torque: ', force_torque.shape, self.wrist_3_joint_id[robot2_name])  # [2, 1, 6], [5]
                elif self.cfg.initial_cfg.FT_SENSOR_TYPE == 'physx_view':
                    # print('self._joint_ids: ', self._jacobi_body_idx[robot2_name], self._joint_ids[robot2_name]) # 8 [0, 1, 2, 3, 4, 5]
                    jacobian = self.robots[robot2_name].root_physx_view.get_jacobians()[:,
                               self._jacobi_body_idx[robot2_name],
                               :,
                               self._joint_ids[robot2_name]]
                    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
                    joint_torques = self.robots[robot2_name].root_physx_view.get_dof_projected_joint_forces()[:,
                                    self._joint_ids[robot2_name]]
                    end_effector_forces = jacobian_T @ joint_torques.unsqueeze(-1)
                    end_effector_forces = end_effector_forces.squeeze(-1)
                    # print('end_effector_forces: ', end_effector_forces, end_effector_forces.shape)
                else:
                    pass  # No ft sensor observation
                obs = torch.cat([obs, end_effector_forces], dim=1)

        observations = {"policy": obs}
        if torch.isnan(obs).any():
            print(obs)
        if self.cfg.initial_cfg.USE_CAMERA:
            if self.cfg.camera_type == 'tiled':
                observations["image"] = self.depth_image(self.scene.sensors["camera"]).clone()
                if self.cfg.initial_cfg.SAVE_IMG:
                    colored_depth = visualize_depth_image(observations["image"].clone()[0, :, :, 0].cpu().numpy())
                    cv2.imwrite(os.path.join(self.image_dir_epoch, '{:04d}_vis_depth.png'.format(self.episode_length_buf[0])),
                                colored_depth)

                    rgb_image = self.scene.sensors["camera"].data.output["rgb"].clone()[0].cpu().numpy()
                    # visualize_rgb_image(rgb_image)
                    cv2.imwrite(os.path.join(self.image_dir_epoch, '{:04d}_vis_rgb.png'.format(self.episode_length_buf[0])),
                                rgb_image)
                    # print(observations["image"].shape)
                    '''
                    rgb_image = self.scene.sensors["camera"].data.output["rgb"].clone()[0].cpu().numpy()
                    # print(np.min(rgb_image), np.max(rgb_image))  # [0, 1]
                    # print('rgb_image.shape: ', rgb_image.shape)
                    visualize_rgb_image(rgb_image)
                    '''
            else:
                depth_image = self.scene.sensors["camera"].data.output["distance_to_camera"][:, :, :, None].clone()
                depth_image = torch.clip(depth_image, 0, 1)
                observations["image"] = depth_image
                # print(depth_image)
                # print(depth_image.shape, torch.min(depth_image), torch.max(depth_image))
                if self.cfg.initial_cfg.SAVE_IMG:
                    colored_depth = visualize_depth_image(depth_image[0, :, :, 0].cpu().numpy())
                    cv2.imwrite(os.path.join(self.image_dir_epoch, '{:04d}_vis_depth.png'.format(self.episode_length_buf[0])),
                                colored_depth)
                '''
                visualize_depth_image(depth_image[0, :, :, 0].cpu().numpy())
                rgb_image = self.scene.sensors["camera"].data.output["rgb"].clone()[0].cpu().numpy()
                # print(np.min(rgb_image), np.max(rgb_image))  # [0, 1]
                # print('rgb_image.shape: ', rgb_image.shape)
                visualize_rgb_image(rgb_image)
                '''
        # print('_get_observations step: {}, observations: {}'.format(str(int(self.num_episode_buf[0].item())), observations))
        if self.debug_action:
            print(3, 'step: {}, state: {}'.format(str(int(self.num_episode_buf[0].item())), observations['policy'][0]))
        return observations

    def get_quat_diff(self, q1, q2):
        # Compute quat error (i.e., difference quat)
        # Reference: https://personal.utdallas.edu/~sxb027100/dock/quat.html
        current_quat_norm = math_utils.quat_mul(q1, math_utils.quat_conjugate(q1))[:, 0]  # scalar component
        current_quat_inv = math_utils.quat_conjugate(q1) / current_quat_norm.unsqueeze(-1)
        quat_error = math_utils.quat_mul(q2, current_quat_inv)
        return quat_error

    def _compute_frame_pose_v0(self, asset_name, asset) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pose_w = asset.data.body_state_w[:, self._body_idx[asset_name], :7]
        root_pose_w = asset.data.root_state_w[:, :7]
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        return ee_pose_b, ee_quat_b

    def _compute_frame_pose(self, asset_name, asset) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = asset.data.body_state_w[:, self._body_idx[asset_name], :3]-self.scene.env_origins
        ee_quat_w = asset.data.body_state_w[:, self._body_idx[asset_name], 3:7]
        return ee_pos_w.clone(), ee_quat_w.clone()

    def _compute_frame_jacobian(self, asset_name, asset):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx[asset_name], :,
                   self._joint_ids[asset_name]]
        return jacobian

    def insert_relative_pose(self) -> torch.Tensor:
        """Reward the agent for reaching the object using tanh-kernel."""
        peg_distance = None
        hole_distance = None

        if (self.cfg.initial_cfg.relative_goal):
            if (self.cfg.initial_cfg.num_robots == 1):
                peg_obj = self.scene["peg"]
                des_world_T_peg = IPose.from_pose(self.cfg.initial_cfg.WORLD_T_HOLE_START).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL))
                world_T_peg = IPose(peg_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                    peg_obj.data.root_state_w[:, 3:7])
                peg_distance = des_world_T_peg.repeat(self.scene.num_envs).multiply(world_T_peg.invert())
                # des_pos_b = des_world_T_peg.repeat(NUM_ENVS).multiply(world_T_peg.invert()).pos
                # distance = torch.norm(des_pos_b, dim=1)

            elif (self.cfg.initial_cfg.num_robots == 2):
                peg_obj = self.scene["peg"]
                hole_obj = self.scene["hole"]
                world_T_peg = IPose(peg_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                    peg_obj.data.root_state_w[:, 3:7])
                world_T_hole = IPose(hole_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                     hole_obj.data.root_state_w[:, 3:7])
                hole_T_peg = world_T_hole.invert().multiply(world_T_peg)
                des_hole_T_peg = IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)
                peg_distance = des_hole_T_peg.repeat(self.scene.num_envs).multiply(hole_T_peg.invert())
                # des_pos_b = (des_hole_T_peg.repeat(NUM_ENVS).multiply(hole_T_peg.invert())).pos
                # # Reward shaping to prioritize x,y offsets
                # des_pos_b[:, :2] *= 5
                # distance = torch.norm(des_pos_b, dim=1)
        else:
            peg_obj = self.scene["peg"]
            world_T_peg = IPose(peg_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                peg_obj.data.root_state_w[:, 3:7])
            des_world_T_peg = IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)
            peg_distance = des_world_T_peg.repeat(self.scene.num_envs).multiply(world_T_peg.invert())
            # des_pos_b = (des_world_T_peg.repeat(NUM_ENVS).multiply(world_T_peg.invert())).pos
            # distance = torch.norm(des_pos_b, dim=1)

            if (self.cfg.initial_cfg.num_robots == 2):
                hole_obj = self.scene["hole"]
                world_T_hole = IPose(hole_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                     hole_obj.data.root_state_w[:, 3:7])
                des_world_T_hole = IPose.from_pose(self.cfg.initial_cfg.HOLE_GOAL)
                hole_distance = des_world_T_hole.repeat(self.scene.num_envs).multiply(world_T_hole.invert())
                # des_pos_b = (des_world_T_hole.repeat(NUM_ENVS).multiply(world_T_hole.invert())).pos
                # distance += torch.norm(des_pos_b, dim=1)
        distance = 0
        for (di, (pose_distance, weights)) in enumerate(
                [(peg_distance, self.cfg.initial_cfg.peg_goal_weights), (hole_distance, self.cfg.initial_cfg.hole_goal_weights)]):
            if (pose_distance is not None):
                diff_pos_b = pose_distance.pos
                diff_rot_b = math_utils.axis_angle_from_quat(pose_distance.quat)
                # if(di==0):
                #     print("----------------")
                #     print(diff_pos_b)
                #     print(diff_rot_b)
                #     print(self.cfg.initial_cfg.peg_goal_weights)
                weights = torch.tensor(weights).type(torch.FloatTensor).cuda()
                diff_pose = torch.cat([diff_pos_b, diff_rot_b], dim=1) * weights
                distance += torch.norm(diff_pose, dim=1)

        reward = torch.clip(-distance, -2.0, 2.0)
        # print("reward", reward)
        log_list("reward", reward)
        return reward

    def _get_rewards(self) -> torch.Tensor:
        return self.insert_relative_pose()


class MoveEnv(MyEnv):
    cfg: MoveEnvCfg

class InsertEnv(MyEnv):
    cfg: InsertEnvCfg

class ScriptedInsertEnv(MyEnv):
    cfg: ScriptedInsertEnvCfg

class GraspEnv(MyEnv):
    cfg: GraspEnvCfg

    def insert_relative_pose(self) -> torch.Tensor:
        """Reward the agent for reaching the object using tanh-kernel."""
        if (self.cfg.initial_cfg.relative_goal):
            if (self.cfg.initial_cfg.num_robots == 1):
                peg_obj = self.scene["peg"]
                des_world_T_peg = IPose.from_pose(self.cfg.initial_cfg.WORLD_T_FIXTURE_START).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL))
                world_T_peg = IPose(peg_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                    peg_obj.data.root_state_w[:, 3:7])
                des_pos_b = des_world_T_peg.repeat(self.scene.num_envs).multiply(world_T_peg.invert()).pos
                distance = torch.norm(des_pos_b, dim=1)

            elif (self.cfg.initial_cfg.num_robots == 2):
                peg_obj = self.scene["peg"]
                hole_obj = self.scene["hole"]
                world_T_peg = IPose(peg_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                    peg_obj.data.root_state_w[:, 3:7])
                world_T_hole = IPose(hole_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                     hole_obj.data.root_state_w[:, 3:7])
                hole_T_peg = world_T_hole.invert().multiply(world_T_peg)
                des_hole_T_peg = IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)
                des_pos_b = (des_hole_T_peg.repeat(self.scene.num_envs).multiply(hole_T_peg.invert())).pos
                # Reward shaping to prioritize x,y offsets
                des_pos_b[:, :2] *= 5
                distance = torch.norm(des_pos_b, dim=1)
        else:
            peg_obj = self.scene["peg"]
            world_T_peg = IPose(peg_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                peg_obj.data.root_state_w[:, 3:7])
            des_world_T_peg = IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)
            des_pos_b = (des_world_T_peg.repeat(self.scene.num_envs).multiply(world_T_peg.invert())).pos
            distance = torch.norm(des_pos_b, dim=1)

            if (self.cfg.initial_cfg.num_robots == 2):
                hole_obj = self.scene["hole"]
                world_T_hole = IPose(hole_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                     hole_obj.data.root_state_w[:, 3:7])
                des_world_T_hole = IPose.from_pose(self.cfg.initial_cfg.HOLE_GOAL)
                des_pos_b = (des_world_T_hole.repeat(self.scene.num_envs).multiply(world_T_hole.invert())).pos
                distance += torch.norm(des_pos_b, dim=1)

        reward = torch.clip(-distance, -0.5, 0.5)
        log_list("reward", reward)
        return reward

class ScriptedGraspEnv(GraspEnv):
    cfg: ScriptedGraspEnvCfg

class RegraspEnv(MyEnv):
    cfg: RegraspEnvCfg

    def insert_relative_pose(self) -> torch.Tensor:
        """Reward the agent for reaching the object using tanh-kernel."""
        if (self.cfg.initial_cfg.relative_goal):
            if (self.cfg.initial_cfg.num_robots == 1):
                peg_obj = self.scene["peg"]
                des_world_T_peg = IPose.from_pose(self.cfg.initial_cfg.WORLD_T_FIXTURE_START).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL))
                world_T_peg = IPose(peg_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                    peg_obj.data.root_state_w[:, 3:7])
                des_pos_b = des_world_T_peg.repeat(self.scene.num_envs).multiply(world_T_peg.invert()).pos
                # print('world_T_peg: ', world_T_peg)
                distance = torch.norm(des_pos_b, dim=1)

            elif (self.cfg.initial_cfg.num_robots == 2):
                peg_obj = self.scene["peg"]
                hole_obj = self.scene["hole"]
                world_T_peg = IPose(peg_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                    peg_obj.data.root_state_w[:, 3:7])
                world_T_hole = IPose(hole_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                     hole_obj.data.root_state_w[:, 3:7])
                hole_T_peg = world_T_hole.invert().multiply(world_T_peg)
                des_hole_T_peg = IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)
                des_pos_b = (des_hole_T_peg.repeat(self.scene.num_envs).multiply(hole_T_peg.invert())).pos
                # Reward shaping to prioritize x,y offsets
                des_pos_b[:, :2] *= 5
                distance = torch.norm(des_pos_b, dim=1)
        else:
            peg_obj = self.scene["peg"]
            world_T_peg = IPose(peg_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                peg_obj.data.root_state_w[:, 3:7])
            des_world_T_peg = IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)
            des_pos_b = (des_world_T_peg.repeat(self.scene.num_envs).multiply(world_T_peg.invert())).pos
            distance = torch.norm(des_pos_b, dim=1)

            if (self.cfg.initial_cfg.num_robots == 2):
                hole_obj = self.scene["hole"]
                world_T_hole = IPose(hole_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                     hole_obj.data.root_state_w[:, 3:7])
                des_world_T_hole = IPose.from_pose(self.cfg.initial_cfg.HOLE_GOAL)
                des_pos_b = (des_world_T_hole.repeat(self.scene.num_envs).multiply(world_T_hole.invert())).pos
                distance += torch.norm(des_pos_b, dim=1)

        reward = torch.clip(-distance, -0.5, 0.5)
        log_list("reward", reward)
        return reward

class ScriptedRegraspEnv(RegraspEnv):
    cfg: ScriptedRegraspEnvCfg


class PlaceEnv(MyEnv):
    cfg: PlaceEnvCfg

    def insert_relative_pose(self) -> torch.Tensor:
        """Reward the agent for reaching the object using tanh-kernel."""
        if (self.cfg.initial_cfg.relative_goal):
            if (self.cfg.initial_cfg.num_robots == 1):
                peg_obj = self.scene["peg"]
                des_world_T_peg = IPose.from_pose(self.cfg.initial_cfg.WORLD_T_FIXTURE_START).multiply(
                    IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL))
                world_T_peg = IPose(peg_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                    peg_obj.data.root_state_w[:, 3:7])
                des_pos_b = des_world_T_peg.repeat(self.scene.num_envs).multiply(world_T_peg.invert()).pos
                distance = torch.norm(des_pos_b, dim=1)

            elif (self.cfg.initial_cfg.num_robots == 2):
                peg_obj = self.scene["peg"]
                hole_obj = self.scene["hole"]
                world_T_peg = IPose(peg_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                    peg_obj.data.root_state_w[:, 3:7])
                world_T_hole = IPose(hole_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                     hole_obj.data.root_state_w[:, 3:7])
                hole_T_peg = world_T_hole.invert().multiply(world_T_peg)
                des_hole_T_peg = IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)
                des_pos_b = (des_hole_T_peg.repeat(self.scene.num_envs).multiply(hole_T_peg.invert())).pos
                # Reward shaping to prioritize x,y offsets
                des_pos_b[:, :2] *= 5
                distance = torch.norm(des_pos_b, dim=1)
        else:
            peg_obj = self.scene["peg"]
            world_T_peg = IPose(peg_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                peg_obj.data.root_state_w[:, 3:7])
            des_world_T_peg = IPose.from_pose(self.cfg.initial_cfg.PEG_GOAL)
            des_pos_b = (des_world_T_peg.repeat(self.scene.num_envs).multiply(world_T_peg.invert())).pos
            distance = torch.norm(des_pos_b, dim=1)

            if (self.cfg.initial_cfg.num_robots == 2):
                hole_obj = self.scene["hole"]
                world_T_hole = IPose(hole_obj.data.root_state_w[:, :3] - self.scene.env_origins,
                                     hole_obj.data.root_state_w[:, 3:7])
                des_world_T_hole = IPose.from_pose(self.cfg.initial_cfg.HOLE_GOAL)
                des_pos_b = (des_world_T_hole.repeat(self.scene.num_envs).multiply(world_T_hole.invert())).pos
                distance += torch.norm(des_pos_b, dim=1)

        reward = torch.clip(-distance, -0.5, 0.5)
        log_list("reward", reward)
        return reward

class ScriptedPlaceEnv(MyEnv):
    cfg: ScriptedPlaceEnvCfg

class ScriptedPlaceHorizontalEnv(MyEnv):
    cfg: ScriptedPlaceHorizontalEnvCfg

class BeamScriptedInsertEnv(MyEnv):
    cfg: BeamScriptedInsertEnvCfg

class BeamInsertEnv(MyEnv):
    cfg: BeamInsertEnvCfg

class BeamScriptedGraspEnv(MyEnv):
    cfg: BeamScriptedGraspEnvCfg

class BeamScriptedPlaceEnv(MyEnv):
    cfg: BeamScriptedPlaceEnvCfg

class BeamMoveToBoardEnv(MyEnv):
    cfg: BeamMoveToBoardEnvCfg

class StoolScriptedInsertEnv(MyEnv):
    cfg: StoolScriptedInsertEnvCfg

class StoolScriptedGraspEnv(MyEnv):
    cfg: StoolScriptedGraspEnvCfg

class StoolScriptedGraspHorizontalEnv(MyEnv):
    cfg: StoolScriptedGraspHorizontalEnvCfg

class StoolScriptedPlaceEnv(MyEnv):
    cfg: StoolScriptedPlaceEnvCfg

class StoolMoveToBoardEnv(MyEnv):
    cfg: StoolMoveToBoardEnvCfg

class StoolInsertEnv(MyEnv):
    cfg: StoolInsertEnvCfg

gym.register(
    id="DirectPositionCNN-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:MyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MyEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:myenv_cfg_cnn.yaml",
    },
)

gym.register(
    id="DirectPosition-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:MyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MyEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:myenv_cfg.yaml",
    },
)

gym.register(
    id="AssemblyMove-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:MoveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MoveEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_move_env_cfg.yaml",
    },
)

gym.register(
    id="AssemblyMoveToFixture-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:MoveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MoveToFixtureEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_move_env_cfg.yaml",
    },
)

gym.register(
    id="AssemblyMoveToFixtureHorizontal-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:MoveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MoveToFixtureHorizontalEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_move_env_cfg.yaml",
    },
)

gym.register(
    id="AssemblyMoveToBoardFromFixture-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:MoveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MoveToBoardFromFixtureEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_move_env_cfg.yaml",
    },
)

gym.register(
    id="AssemblyMoveToBoardFromGrasp-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:MoveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MoveToBoardFromGraspEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_move_env_cfg.yaml",
    },
)

gym.register(
    id="AssemblyInsert-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:InsertEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": InsertEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_insert_env_cfg.yaml",
    },
)

gym.register(
    id="AssemblyScriptedInsert-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:ScriptedInsertEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ScriptedInsertEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_scripted_insert_env_cfg.yaml",
    },
)

gym.register(
    id="AssemblyGrasp-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:GraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GraspEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_grasp_env_cfg.yaml",
    },
)

gym.register(
    id="AssemblyRegrasp-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:RegraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RegraspEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_regrasp_env_cfg.yaml",
    },
)

gym.register(
    id="AssemblyPlace-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:PlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PlaceEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_place_env_cfg.yaml",
    },
)

gym.register(
    id="AssemblyScriptedPlace-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:ScriptedPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ScriptedPlaceEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_scripted_place_env_cfg.yaml",
    },
)

gym.register(
    id="AssemblyScriptedPlaceHorizontal-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:ScriptedPlaceHorizontalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ScriptedPlaceHorizontalEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_scripted_place_env_cfg.yaml",
    },
)

gym.register(
    id="AssemblyScriptedGrasp-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:ScriptedGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ScriptedGraspEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_scripted_grasp_env_cfg.yaml",
    },
)

gym.register(
    id="AssemblyScriptedGraspHorizontal-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:ScriptedGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ScriptedGraspHorizontalEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_scripted_grasp_env_cfg.yaml",
    },
)

gym.register(
    id="AssemblyScriptedRegrasp-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:ScriptedRegraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ScriptedRegraspEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_scripted_regrasp_env_cfg.yaml",
    },
)

gym.register(
    id="BeamScriptedInsert-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:BeamScriptedInsertEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamScriptedInsertEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_insert_env_cfg.yaml",
    },
)

gym.register(
    id="BeamScriptedGrasp-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:BeamScriptedGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamScriptedGraspEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_grasp_env_cfg.yaml",
    },
)

gym.register(
    id="BeamScriptedPlace-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:BeamScriptedPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamScriptedPlaceEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_grasp_env_cfg.yaml",
    },
)

gym.register(
    id="BeamMoveToBoard-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:BeamMoveToBoardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamMoveToBoardEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_grasp_env_cfg.yaml",
    },
)

gym.register(
    id="BeamInsert-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:BeamInsertEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamInsertEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_insert_env_cfg.yaml",
    },
)


gym.register(
    id="StoolScriptedInsert-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:StoolScriptedInsertEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": StoolScriptedInsertEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_insert_env_cfg.yaml",
    },
)

gym.register(
    id="StoolScriptedGrasp-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:StoolScriptedGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": StoolScriptedGraspEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_grasp_env_cfg.yaml",
    },
)

gym.register(
    id="StoolScriptedGraspHorizontal-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:StoolScriptedGraspHorizontalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": StoolScriptedGraspHorizontalEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_grasp_env_cfg.yaml",
    },
)

gym.register(
    id="StoolScriptedPlace-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:StoolScriptedPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": StoolScriptedPlaceEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_grasp_env_cfg.yaml",
    },
)

gym.register(
    id="StoolMoveToBoard-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:StoolMoveToBoardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": StoolMoveToBoardEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_grasp_env_cfg.yaml",
    },
)

gym.register(
    id="StoolInsert-v0",
    entry_point="multi_arm_assembly.direct_isaac_lab_position:StoolInsertEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": StoolInsertEnvCfg,
        "rl_games_cfg_entry_point": "multi_arm_assembly:assembly_insert_env_cfg.yaml",
    },
)