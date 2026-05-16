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
import omni.kit.commands

from omni.isaac.lab.managers import RewardTermCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.controllers import DifferentialIKControllerCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.assets.rigid_object import RigidObjectCfg
from multi_arm_assembly.utils import solve_ik
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
from multi_arm_assembly.utils import log_list, task_from_name, FLANGE_T_TOOL
from omni.isaac.lab.sensors import TiledCameraCfg, Camera, TiledCamera
from PIL import Image
import time
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.scene import InteractiveSceneCfg

# NUM_ENVS = 1000
NUM_ENVS = 10
home_dir = os.getcwd() + '/../'
# INITIAL_CFG = task_from_name("gear_in_taskboard")
# INITIAL_CFG = task_from_name("rod_in_gear")
# INITIAL_CFG = task_from_name("strut_plus_elbow_in_platform")
# INITIAL_CFG = task_from_name("strut_in_elbow")
# INITIAL_CFG = task_from_name("bolt_in_platform")
INITIAL_CFG = task_from_name("fmb_example")

USE_CAMERA = False
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 60

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
UNIT_QUAT = torch.tensor([[1, 0, 0, 0]]).type(torch.FloatTensor).cuda()
UNIT_POINT = torch.zeros(1,3).type(torch.FloatTensor).cuda()


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
    
    @staticmethod
    def from_pose(pose):
        pos, quat = pose
        return IPose(torch.Tensor([pos]).type(torch.FloatTensor).cuda(),
                     torch.Tensor([xyzw_to_wxyz(quat)]).type(torch.FloatTensor).cuda())
    
    def repeat(self, n):
        return IPose(self.pos.repeat((n, 1)), self.quat.repeat((n, 1)))
    
ROBOT1_EFFORT_CFG = SceneEntityCfg("robot1", body_names=["flange"], joint_names=EFFORT_JOINTS)
ROBOT2_EFFORT_CFG = SceneEntityCfg("robot2", body_names=["flange"], joint_names=EFFORT_JOINTS)
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

def update_rigid_attachments(fixed_joints, obj_T_flange:IPose):
    # Domain randomization
    translation_offset = (torch.rand(NUM_ENVS,3).type(torch.FloatTensor).cuda()-0.5)*0.01
    ero = torch.rand(NUM_ENVS,3).type(torch.FloatTensor).cuda()*0
    rotation_offset = math_utils.quat_from_euler_xyz(ero[:, 0], ero[:, 1], ero[:, 2])

    for env_id, fixed_joint in enumerate(fixed_joints):

        trans_pose = IPose(pos = translation_offset[env_id:env_id+1, :], quat = UNIT_QUAT)
        rot_pose = IPose(pos = UNIT_POINT, quat=rotation_offset[env_id:env_id+1, :])
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
            
def create_rigid_attachments(attachment_path:str, robot_path:str, extra_attachments=[]):

    # Domain randomization
    fixed_joints = []
    for env_id in range(NUM_ENVS):

        env0_path = f"/World/envs/env_{env_id}/{robot_path}"
        fixed_joint_path = env0_path + "/AssemblerFixedJoint"
        fixed_joint_path = find_unique_string_name(fixed_joint_path, lambda x: not prim_utils.is_prim_path_valid(x))
        stage = stage_utils.get_current_stage()
        fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)

        target0 = env0_path+"/flange"        
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




@configclass
class MyEnvCfg(DirectRLEnvCfg):
    """Configuration for the cartpole environment."""
    
    # Scene settings
    episode_length_s = 4.0
    decimation = DECIMATION*SUBSTEPS
    sim = SimulationCfg(dt=SIM_DT/SUBSTEPS)
    scene = InteractiveSceneCfg(num_envs=NUM_ENVS, env_spacing=4)
    num_actions = 3*INITIAL_CFG.num_robots
    num_observations = 9*INITIAL_CFG.num_robots

    def __post_init__(self):

        # To keep the fingers stiff without bending
        finger_stiffness = 100.0
        finger_damping = 100.0

        gripper_vel_limit = 1.0
        # gripper_effort = 5.0 # Close hard but requires 5 substeps
        gripper_effort = 1.5
        small_joint_damping = 0.0 
        arm_effort = 100   
        arm_stiffness = 400.0
        arm_damping = 800.0
        arm_vel_limit = 20.0
        self.parts = []
        self.peripherals = []
        self.articulated_parts = []
        self.robot1 = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=home_dir + "pls_updated_gripper.usd",
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
        ).replace(prim_path="/World/envs/env_.*/robot1")

    
        if(USE_CAMERA):
            camera_pos = list(INITIAL_CFG.WORLD_T_PEG_START[0])
            camera_pos[0] -= 0.2

            self.tiled_camera = TiledCameraCfg(
                prim_path="/World/envs/env_.*/camera",
                offset=TiledCameraCfg.OffsetCfg(pos=camera_pos, rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
                data_types=["depth"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,  # Keep this as is, it's actually good for close-up
                    focus_distance=0.1,  # Set to 0.1 meters (about 4 inches)
                    horizontal_aperture=20.955,  # Keep this as is
                    clipping_range=(0.01, 1.0),  # Near plane at 1cm, far plane at 2m
                ),
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
            )

        if(INITIAL_CFG.num_robots == 2):
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
                    }| {k: 0 for k in FINGER_JOINTS}
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
            ).replace(prim_path="/World/envs/env_.*/robot2")

        self.peripherals.append(
            AssetBaseCfg(
                prim_path="/World/envs/env_.*/table",  
                spawn = sim_utils.UsdFileCfg(
                    usd_path=home_dir + "table_top_collision.usd", 
                )
            )
        )


        world_T_hole = IPose.from_pose(INITIAL_CFG.WORLD_T_HOLE_START)

        if(INITIAL_CFG.HOLE_TYPE == "peripheral"):
            self.peripherals.append(
                AssetBaseCfg(
                    prim_path="/World/envs/env_.*/hole",
                    spawn = sim_utils.UsdFileCfg(
                        usd_path=INITIAL_CFG.HOLE_ASSET_NAME,
                    ),
                    
                    init_state = AssetBaseCfg.InitialStateCfg(world_T_hole.pos[0], world_T_hole.quat[0])
                )
            )
        elif(INITIAL_CFG.HOLE_TYPE == "part"):
            self.parts.append(
                RigidObjectCfg( 
                    prim_path="/World/envs/env_.*/hole",  
                    spawn=sim_utils.UsdFileCfg(
                        usd_path = INITIAL_CFG.HOLE_ASSET_NAME,
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

        world_T_peg_start = IPose.from_pose(INITIAL_CFG.WORLD_T_PEG_START)

        if(INITIAL_CFG.PEG_TYPE == "part"):
            self.parts.append(
                RigidObjectCfg( 
                    prim_path="/World/envs/env_.*/peg",  
                    spawn=sim_utils.UsdFileCfg(
                        usd_path = INITIAL_CFG.PEG_ASSET_NAME,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True
                        )
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=world_T_peg_start.pos[0], 
                                                            rot=world_T_peg_start.quat[0]),
                )
            )
        else:
            raise NotImplementedError
        
        # Add extra parts
        for ei, (extra_path, extra_pose) in enumerate(zip(INITIAL_CFG.EXTRA_PARTS, INITIAL_CFG.EXTRA_PARTS_STARTING_POSE)):
            ipose = IPose.from_pose(extra_pose)
            self.peripherals.append(
                AssetBaseCfg(
                    prim_path="/World/envs/env_.*/extra"+str(ei),
                    spawn = sim_utils.UsdFileCfg(
                        usd_path=extra_path,
                    ),
                    init_state = AssetBaseCfg.InitialStateCfg(ipose.pos[0], ipose.quat[0])
                )
            )

        # Initialize the gripper to be directly above the target object pose
        # This is where the object is initialized to be reset later inside the fingers
        base_q = wxyz_to_xyzw(self.robot1.init_state.rot) 
        world_T_flange = IPose.from_pose(INITIAL_CFG.WORLD_T_PEG_START).multiply(IPose.from_pose(INITIAL_CFG.PEG_T_FLANGE))
        joint_angles = solve_ik(self.robot1.init_state.pos, 
                                base_q, 
                                world_T_flange.pos[0].cpu(),
                                wxyz_to_xyzw(world_T_flange.quat[0].cpu()))
        assert len(joint_angles) == len(EFFORT_JOINTS)
        for joint_name, joint_angle in zip(EFFORT_JOINTS, joint_angles):
            self.robot1.init_state.joint_pos[joint_name] = joint_angle

        if(INITIAL_CFG.num_robots==2):
            # Find joint angles for the desired gripper pose
            base_q = wxyz_to_xyzw(self.robot2.init_state.rot) 
            world_T_flange = IPose.from_pose(INITIAL_CFG.WORLD_T_HOLE_START).multiply(IPose.from_pose(INITIAL_CFG.HOLE_T_FLANGE))
            joint_angles = solve_ik(self.robot2.init_state.pos, 
                                    base_q, 
                                    world_T_flange.pos[0].cpu(),
                                    wxyz_to_xyzw(world_T_flange.quat[0].cpu()))
            assert len(joint_angles) == len(EFFORT_JOINTS)
            for joint_name, joint_angle in zip(EFFORT_JOINTS, joint_angles):
                self.robot2.init_state.joint_pos[joint_name] = joint_angle


class MyEnv(DirectRLEnv):
    cfg: MyEnvCfg

    cnn: bool = USE_CAMERA
    image_width: int = IMAGE_WIDTH
    image_height: int = IMAGE_HEIGHT

    def __init__(self, cfg: MyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.translation_limit = 0.05*5 # in m
        self.rotation_limit = 3 / 180 *np.pi # in rad

        self.force_limit = FORCE_LIMIT_TRANS
        self.velocity_limit = VELOCITY_LIMIT_TRANS
        self.torque_limit = TORQUE_LIMIT

        self.action_scale = 0.1
        self.fixed_quat = {}
        self._ik_controllers = {}
        self._joint_ids = {}
        self._jacobi_body_idx = {}
        self._finger_joint_ids = {}
        self._body_idx = {}

        for robot_name, robot_asset in self.robots.items():
            self.robots[robot_name] = robot_asset
             # resolve the joints over which the action term is applied
            self._joint_ids[robot_name], _ = robot_asset.find_joints(EFFORT_JOINTS)
            self._finger_joint_ids[robot_name], _ = robot_asset.find_joints(FINGER_JOINTS)
            self._num_joints = len(self._joint_ids[robot_name])

            # parse the body index
            body_ids, _ = robot_asset.find_bodies("flange")
            
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

            ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
            self._ik_controllers[robot_name] = DifferentialIKController(
                cfg=ik_controller_cfg, num_envs=NUM_ENVS, device=self.sim.device
            )

        self.robot1_fixed_joints = create_rigid_attachments(robot_path="robot1", attachment_path = INITIAL_CFG.PEG_ATTACHMENT_PRIM, extra_attachments=INITIAL_CFG.EXTRA_ATTACHMENTS)
        update_rigid_attachments(self.robot1_fixed_joints, obj_T_flange = IPose.from_pose(INITIAL_CFG.PEG_T_FLANGE))
        if(INITIAL_CFG.num_robots == 2):
            self.robot2_fixed_joints = create_rigid_attachments(robot_path="robot2", attachment_path = INITIAL_CFG.HOLE_ATTACHMENT_PRIM, extra_attachments=INITIAL_CFG.EXTRA_ATTACHMENTS)
            update_rigid_attachments(self.robot2_fixed_joints, obj_T_flange = IPose.from_pose(INITIAL_CFG.HOLE_T_FLANGE))
       
    @property
    def robots(self):
        return {k:v for k, v in self.scene.articulations.items() if "robot" in k}

    @property
    def articulated_parts(self):
        return {k:v for k, v in self.scene.articulations.items() if "robot" not in k}

    def _setup_scene(self):
        
        robot1 = Articulation(self.cfg.robot1)
        if(INITIAL_CFG.num_robots == 2):
            robot2 = Articulation(self.cfg.robot2)

        rigid_bodies = [RigidObject(part) for part in self.cfg.parts]
        articulated_parts = [Articulation(apart) for apart in self.cfg.articulated_parts]

        for asset_cfg in self.cfg.peripherals:
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
        if INITIAL_CFG.num_robots == 2:
            self.scene.articulations["robot2"] = robot2

        for rigid_body in rigid_bodies:
            self.scene.rigid_objects[rigid_body.cfg.prim_path.split("/")[-1]] = rigid_body

        for articulated_part in articulated_parts:
            self.scene.articulations[articulated_part.cfg.prim_path.split("/")[-1]] = articulated_part

        if(USE_CAMERA):
            self.scene.sensors["camera"] = TiledCamera(self.cfg.tiled_camera)


    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        # For moving around the scene before things start        
        # print("Sleeping ")
        # for _ in range(2000):
        #     self.step_sim()
        #     time.sleep(0.01)

        mdp.reset_scene_to_default(self, env_ids)
        
        update_rigid_attachments(self.robot1_fixed_joints, obj_T_flange = IPose.from_pose(INITIAL_CFG.PEG_T_FLANGE))
        if(INITIAL_CFG.num_robots == 2):
            update_rigid_attachments(self.robot2_fixed_joints, obj_T_flange = IPose.from_pose(INITIAL_CFG.HOLE_T_FLANGE))

        self.step_sim()
        self.initialized = False
        for robot_name, robot_asset in self.robots.items():
            _, self.fixed_quat[robot_name] = self._compute_frame_pose(robot_name, robot_asset)

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

        # For debugging, go straight down
        #     actions = torch.zeros([NUM_ENVS, self.cfg.num_actions]).cuda()
        #     actions[:, 0] = -1.0

        processed_actions = actions * self.action_scale
        processed_actions[:, 3:] = 0

        delta = torch.zeros(NUM_ENVS, self.cfg.num_actions, device=self.sim.device)
        delta[:, :3] = processed_actions[:, :3]# / (torch.linalg.norm(self._processed_actions[:, :3], ord=2) + 1e-3) * self.translation_limit
        # delta[:, 3:6] = self._processed_actions[:, 3:6] / (torch.linalg.norm(self._processed_actions[:, 3:6], ord=2) + 1e-3) * self.rotation_limit
        
        tool_trans, _ = self._compute_frame_pose(robot_name, robot_asset)
        # delta_rotm = math_utils.matrix_from_euler(delta[:, 3:6], "XYZ")

        # Target absolute position
        log_list(f"{robot_name}_delta", delta)
        control_cmd = torch.zeros(self.num_envs, 7).to(self.sim.device)
        control_cmd[:, :3] = tool_trans + delta[:, :3]

        # delta_rotation = math_utils.quat_from_matrix(delta_rotm)
        control_cmd[:, 3:7] = self.fixed_quat[robot_name] #math_utils.quat_mul(self.fixed_quat, delta_rotation)
        # self.control_cmd[:, 3:7] = math_utils.quat_mul(tool_quat, delta_rotation)

        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose(robot_name, robot_asset)

        # set command into controller
        self._ik_controllers[robot_name].set_command(control_cmd, ee_pos_curr, ee_quat_curr)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        if(INITIAL_CFG.num_robots == 1):
            self._pre_physics_step_sr(actions, "robot1", self.robots["robot1"])
        elif(INITIAL_CFG.num_robots == 2):
            self._pre_physics_step_sr(actions[:, :self.cfg.num_actions//2], "robot1", self.robots["robot1"])
            self._pre_physics_step_sr(actions[:, self.cfg.num_actions//2:], "robot2", self.robots["robot2"])

    def _apply_action(self) -> None:
        for robot_name, robot_asset in self.robots.items():
            # obtain quantities from simulation
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose(robot_name, robot_asset)
            joint_pos = robot_asset.data.joint_pos[:, self._joint_ids[robot_name]]
            # compute the delta in joint-space
            if ee_quat_curr.norm() != 0:
                jacobian = self._compute_frame_jacobian(robot_name, robot_asset)
                joint_pos_des = self._ik_controllers[robot_name].compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
            else:
                joint_pos_des = joint_pos.clone()


            error = torch.abs(joint_pos-joint_pos_des)
            log_list(f"{robot_name}_joint_error", error)

            # set the joint position command
            robot_asset.set_joint_position_target(joint_pos_des, self._joint_ids[robot_name])
            
            
            # Keep fingers closed
            # current_finger_joint_pos = robot_asset.data.joint_pos[:, self._finger_joint_ids]
            # robot_asset.set_joint_position_target(current_finger_joint_pos, joint_ids=self._finger_joint_ids)
            # robot_asset.set_joint_velocity_target(24, joint_ids=self._finger_joint_ids)


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out
    
    def ee_pos(self, asset_cfg: SceneEntityCfg, obj_cfg: SceneEntityCfg, use_pose:bool=False):
        """The joint velocities of the asset w.r.t. the default joint velocities.

        Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
        """
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self.scene[asset_cfg.name]
        
        # obtain quantities from simulation
        body_ids, _ = asset.find_bodies("flange")
        iflange = pose_from_state(asset.data.body_state_w, body_ids[0], self.scene.env_origins)
        flange_pose = iflange.multiply(IPose.from_pose(FLANGE_T_TOOL).repeat(NUM_ENVS))
        ee_rotm = math_utils.matrix_from_quat(flange_pose.quat)
        
        if(use_pose):
            obj_asset = self.scene[obj_cfg.name]
            body_ids, _ = obj_asset.find_bodies(".*")
            obj_pose = pose_from_state(obj_asset.data.body_state_w, body_ids[0], self.scene.env_origins)
            obj_rotm = math_utils.matrix_from_quat(obj_pose.quat)
            total_obs = torch.cat([flange_pose.pos, ee_rotm[:, :3, 0], ee_rotm[:, :3, 1], obj_rotm[:, :3, 0], obj_rotm[:, :3, 1]], axis=1)
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



    def _get_observations(self) -> dict:

        obs = self.ee_pos(ROBOT1_EFFORT_CFG, obj_cfg = SceneEntityCfg("peg"), use_pose=False)
        if(INITIAL_CFG.num_robots == 2):
            obs = torch.cat([obs, self.ee_pos(ROBOT2_EFFORT_CFG, obj_cfg = SceneEntityCfg("hole"), use_pose=False)], dim=1)
        
        observations = {"policy": obs}
        if(USE_CAMERA):
            observations["image"] = self.depth_image(self.scene.sensors["camera"])
        
        return observations
    
    def _compute_frame_pose(self, asset_name, asset) -> tuple[torch.Tensor, torch.Tensor]:
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


    def _compute_frame_jacobian(self, asset_name, asset):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx[asset_name], :, self._joint_ids[asset_name]]
        return jacobian
    
    def insert_relative_pose(self) -> torch.Tensor:    
        """Reward the agent for reaching the object using tanh-kernel."""
        if(INITIAL_CFG.relative_goal):
            if(INITIAL_CFG.num_robots == 1):
                peg_obj = self.scene["peg"]
                des_world_T_peg = IPose.from_pose(INITIAL_CFG.WORLD_T_HOLE_START).multiply(IPose.from_pose(INITIAL_CFG.PEG_GOAL))
                world_T_peg = IPose(peg_obj.data.root_state_w[:, :3]-self.scene.env_origins, peg_obj.data.root_state_w[:, 3:7])
                des_pos_b = des_world_T_peg.repeat(NUM_ENVS).multiply(world_T_peg.invert()).pos
                distance = torch.norm(des_pos_b, dim=1)

            elif(INITIAL_CFG.num_robots == 2):
                peg_obj = self.scene["peg"]
                hole_obj = self.scene["hole"]
                world_T_peg = IPose(peg_obj.data.root_state_w[:, :3]-self.scene.env_origins, peg_obj.data.root_state_w[:, 3:7])
                world_T_hole = IPose(hole_obj.data.root_state_w[:, :3]-self.scene.env_origins, hole_obj.data.root_state_w[:, 3:7])
                hole_T_peg = world_T_hole.invert().multiply(world_T_peg)
                des_hole_T_peg = IPose.from_pose(INITIAL_CFG.PEG_GOAL)
                des_pos_b =  (des_hole_T_peg.repeat(NUM_ENVS).multiply(hole_T_peg.invert())).pos
                # Reward shaping to prioritize x,y offsets
                des_pos_b[:, :2] *= 5
                distance = torch.norm(des_pos_b, dim=1)
        else:
            peg_obj = self.scene["peg"]
            world_T_peg = IPose(peg_obj.data.root_state_w[:, :3]-self.scene.env_origins, peg_obj.data.root_state_w[:, 3:7])
            des_world_T_peg =  IPose.from_pose(INITIAL_CFG.PEG_GOAL)
            des_pos_b = (des_world_T_peg.repeat(NUM_ENVS).multiply(world_T_peg.invert())).pos
            distance = torch.norm(des_pos_b, dim=1)

            if(INITIAL_CFG.num_robots == 2):
                hole_obj = self.scene["hole"]
                world_T_hole = IPose(hole_obj.data.root_state_w[:, :3]-self.scene.env_origins, hole_obj.data.root_state_w[:, 3:7])
                des_world_T_hole =  IPose.from_pose(INITIAL_CFG.HOLE_GOAL)
                des_pos_b = (des_world_T_hole.repeat(NUM_ENVS).multiply(world_T_hole.invert())).pos
                distance += torch.norm(des_pos_b, dim=1)
        
        reward = torch.clip(-distance, -0.5, 0.5)
        log_list("reward", reward)
        return reward


    def _get_rewards(self) -> torch.Tensor:
        return self.insert_relative_pose()


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

