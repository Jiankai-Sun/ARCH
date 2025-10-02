import torch
import json
import logging
from typing import List, Tuple, Dict
import os
import time
import pathlib
import numpy as np
from dataclasses import dataclass, field
import random
import pybullet as p
import pb_utils as pbu
import numpy as np 
import math
from pathlib import Path
import cv2
import sys
import copy
from icecream import ic

def set_seed(seed):
    """
    Set the random seed for reproducibility across numpy, random, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    import numpy as np
    import random
    import torch
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # If using GPU, you might also want to set the seed for CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you have multiple GPUs


BASE_LINK = -1

# Get the current directory
home_dir = os.getcwd() + '/../'
ROOT_DIR = Path(__file__).parent.parent

@dataclass
class JointInfo:
    jointIndex: int
    jointName: str
    jointType: int
    qIndex: int
    uIndex: int
    flags: List[str]
    jointDamping: float
    jointFriction: float
    jointLowerLimit: float
    jointUpperLimit: float
    jointMaxForce: float
    jointMaxVelocity: float
    linkName: float
    jointAxis: float
    parentFramePos: float
    parentFrameOrn: float
    parentIndex: float

@dataclass
class LinkState:
    linkWorldPosition: Tuple[float, float, float]
    linkWorldOrientation: Tuple[float, float, float, float]
    localInertialFramePosition: Tuple[float, float, float]
    localInertialFrameOrientation: Tuple[float, float, float, float]
    worldLinkFramePosition: Tuple[float, float, float]
    worldLinkFrameOrientation: Tuple[float, float, float, float]



@dataclass
class DynamicsInfo:
    mass: float
    lateral_friction: float
    local_inertia_diagonal: Tuple[float, float, float]
    local_inertial_pos: Tuple[float, float, float]
    local_inertial_orn: Tuple[float, float, float, float]
    restitution: float
    rolling_friction: float
    spinning_friction: float
    contact_damping: float
    contact_stiffness: float

def get_num_joints(body):
    return p.getNumJoints(int(body))

def get_joints(body):
    return list(range(get_num_joints(body)))

def get_joint_name(body, joint):
    return JointInfo(*p.getJointInfo(int(body), joint)).jointName.decode("UTF-8")
    
def get_joint_names(body, joints, **kwargs):
    return [get_joint_name(body, joint, **kwargs) for joint in joints]  # .encode('ascii')

def clip(value, min_value=-np.inf, max_value=+np.inf):
    return min(max(min_value, value), max_value)

def get_link_name(body, link, **kwargs):
    return JointInfo(*p.getJointInfo(int(body), link)).linkName.decode("UTF-8")

def link_from_name(body, name, **kwargs):

    for link in get_joints(body, **kwargs):
        if get_link_name(body, link, **kwargs) == name:
            return link
    raise ValueError(body, name)

def set_joint_positions(body, joints, values, **kwargs):
    for joint, value in zip(joints, values):
        set_joint_position(body, joint, value, **kwargs)


def set_joint_position(body, joint, value):
    p.resetJointState(int(body), joint, targetValue=value, targetVelocity=0)


def quat_angle_between(quat0, quat1):
    delta = p.getDifferenceQuaternion(quat0, quat1)
    d = clip(delta[-1], min_value=-1.0, max_value=1.0)
    angle = math.acos(d)
    return angle

def invert(pose):
    point, quat = pose
    return p.invertTransform(point, quat)


def get_link_state(body, link):
    return LinkState(*p.getLinkState(int(body), link))

def get_link_pose(body, link, **kwargs):
    link_state = get_link_state(body, link, **kwargs)  # , kinematics=True, velocity=False)
    return link_state.worldLinkFramePosition, link_state.worldLinkFrameOrientation

def get_dynamics_info(body, link=BASE_LINK, **kwargs):
    return DynamicsInfo(*p.getDynamicsInfo(int(body), link)[:10])

def get_mass(body, link=BASE_LINK, **kwargs):  # mass in kg
    # TODO: get full mass
    return get_dynamics_info(body, link, **kwargs).mass

def get_joint_type(body, joint, **kwargs):
    return get_joint_info(body, joint, **kwargs).jointType


def get_joint_info(body, joint, **kwargs):
    return JointInfo(*p.getJointInfo(int(body), joint))

def is_fixed(body, joint, **kwargs):
    return get_joint_type(body, joint, **kwargs) == p.JOINT_FIXED

def is_movable(body, joint, **kwargs):
    return not is_fixed(body, joint, **kwargs)

def prune_fixed_joints(body, joints, **kwargs):
    return [joint for joint in joints if is_movable(body, joint, **kwargs)]

def get_movable_joints(body, **kwargs):
    return prune_fixed_joints(body, get_joints(body, **kwargs), **kwargs)

def set_pose(body, pose):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(int(body), point, quat)

def get_joint_limits(body, joint, **kwargs):
    joint_info = get_joint_info(body, joint, **kwargs)
    return joint_info.jointLowerLimit, joint_info.jointUpperLimit

def get_length(vec, norm=2):
    return np.linalg.norm(vec, ord=norm)


def get_difference(p1, p2):
    assert len(p1) == len(p2)
    return np.array(p2) - np.array(p1)


def get_distance(p1, p2, **kwargs):
    return get_length(get_difference(p1, p2), **kwargs)



def get_pose_distance(pose1, pose2):
    pos1, quat1 = pose1
    pos2, quat2 = pose2
    pos_distance = get_distance(pos1, pos2)
    ori_distance = quat_angle_between(quat1, quat2)
    return pos_distance, ori_distance



def all_between(lower_limits, values, upper_limits):
    assert len(lower_limits) == len(values)
    assert len(values) == len(upper_limits)
    return np.less_equal(lower_limits, values).all() and np.less_equal(values, upper_limits).all()

def multiply(*poses):
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = p.multiplyTransforms(pose[0], pose[1], *next_pose)
    return pose


def visualize_depth_image(depth_image, show=False):
    # Normalize the depth image for better visualization
    depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_image_normalized = np.uint8(depth_image_normalized)

    # Apply a colormap to the normalized depth image for better visualization
    depth_image_colored = cv2.applyColorMap(depth_image_normalized, cv2.COLORMAP_JET)

    # Display the depth image
    if show:
        cv2.imshow('Depth Image', depth_image_colored)
        cv2.waitKey(1)
    return depth_image_colored


def visualize_rgb_image(rgb_image, show=False):
    # Display the depth image
    if show:
        cv2.imshow('RGB Image', rgb_image[:, :, ::-1])
        cv2.waitKey(1)

def wxyz_to_xyzw(q):
    return [q[1], q[2], q[3], q[0]]

def xyzw_to_wxyz(q):
    return [q[3], q[0], q[1], q[2]]

# stage_sequence = {
#     "AssemblyReset": 0,
#     "AssemblyScriptedGrasp": 1,
#     "AssemblyMoveToFixture": 2,
#     "AssemblyScriptedPlace": 3,
#     "AssemblyScriptedRegrasp": 4,
#     "AssemblyMoveToBoardFromFixture": 5,
#     "AssemblyScriptedInsert": 6,
#     "AssemblyInsert": os.path.join(ROOT_DIR, "multi_arm_assembly/good_logs/rl_games/apa_impedance/2024-08-20_12-44-08/nn/apa_impedance.pth"),
#     "AssemblyMoveToBoardFromGrasp": 8,
#     "AssemblyScriptedGraspHorizontal": 9,
#     "AssemblyMoveToFixtureHorizontal": 10,
#     "AssemblyScriptedPlaceHorizontal": 11,
# }

# stage_sequence = {
#     "AssemblyReset": 0,
#     "BeamScriptedGrasp": 1,
#     "BeamMoveToBoard": 2,
#     "BeamScriptedPlace": 3,
#     "BeamScriptedInsert": 4,
#     "BeamInsert": os.path.join(ROOT_DIR, "multi_arm_assembly/good_logs/rl_games/apa_impedance/2025-04-23_21-40-05/nn/last_apa_impedance_ep_1475_rew_-2.533575.pth"),
# }

stage_sequence = {
    "AssemblyReset": 0,
    "StoolScriptedGrasp": 1,
    "StoolMoveToBoard": 2,
    "StoolScriptedPlace": 3,
    "StoolScriptedInsert": 4,
    "StoolInsert": os.path.join(ROOT_DIR, "multi_arm_assembly/good_logs/rl_games/apa_impedance/2025-04-23_21-40-05/nn/last_apa_impedance_ep_1475_rew_-2.533575.pth"),
}


def display_menu(input_dict):
    """Display the menu of stage sequences."""
    print('----------------- Menu ---------------')
    for i, (name, value) in enumerate(input_dict.items()):
        print(f"{i}: {name}")

R0_POSE = ((0.8367000122070312, 0.6095999755859375, 0.02250), (0, 0, 0.7071068, 0.7071068))
R1_POSE = ((-0.8367000122070312, 0.6095999755859375, 0.02250), (0, 0, -0.7071068, 0.7071068))
R2_POSE = ((-0.037869995,-0.652639954,0.282559998), (-0.000176,0.000523,-0.003491,0.999994))

# self.r2 pos: -37.869995,-652.639954,282.559998 rot (x, y, z, w): -0.000176,0.000523,-0.003491,0.999994
# self.r1 pos: -834.599976,607.710083,19.250000 rot (x, y, z, w): 0.003764,0.000679,0.707106,0.707097                                                                                                           
# self.r0 pos: 836.479980,609.390015,27.080000, rot (x, y, z, w): 0.001358,0.001358,-0.707105,0.707105
IGNORE_COLLISIONS = {(6, 9), (2, 10), (3, 10), (4, 10), (6, 10), (3, 9), (4, 9), (2, 11)}

# https://github.com/bulletphysics/bullet3/issues/2170
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different

def setup_environment(ee_collisions=True, include_stewart=True, include_zivid=False):
    with suppress_stdout():
        if (ee_collisions):
            r0 = p.loadURDF("ur_description/urdf/ur10e.urdf", R0_POSE[0], R0_POSE[1], useFixedBase=True)
            r1 = p.loadURDF("ur_description/urdf/ur10e.urdf", R1_POSE[0], R1_POSE[1], useFixedBase=True)
            if include_zivid:
                # r2 = p.loadURDF("ur_description/urdf/ur10e.urdf", R2_POSE[0], R2_POSE[1], useFixedBase=True)
                pass
        else:
            r0 = p.loadURDF("ur_description/urdf/ur10e_no_hand.urdf", R0_POSE[0], R0_POSE[1], useFixedBase=True)
            r1 = p.loadURDF("ur_description/urdf/ur10e_no_hand.urdf", R1_POSE[0], R1_POSE[1], useFixedBase=True)
            if include_zivid:
                # r2 = p.loadURDF("ur_description/urdf/ur10e_no_hand.urdf", R2_POSE[0], R2_POSE[1], useFixedBase=True)
                pass

    robot_list = [r0, r1]
    if include_zivid:
        # robot_list.append(r2)
        pass
    
    # Create the table
    plane_id = p.createCollisionShape(shapeType=p.GEOM_PLANE)
    ground_id = p.createMultiBody(baseCollisionShapeIndex=plane_id)
    pbu.set_pose(ground_id, ((0, 0, 0.01), (0, 0, 0, 1)))

    taskboard_path = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "taskboard")
    if include_stewart:
        # Create the stewart platform
        platform_path = os.path.join(taskboard_path, "stand_for_stewart_platform_assembly_v69_vhacd.obj")
        platform = load_obj(platform_path)
        pbu.set_pose(platform, PLATFORM_POSE)

    # # Create the skateboard
    skateboard = load_obj(os.path.join(taskboard_path, "just_a_medium_cube.stl"), mesh_scale=[0.001, 0.001, 0.001])
    SKATEBOARD_POSE = ([-0.5, 0.09868, 0.09770], [0, 0, 0, 1])
    pbu.set_pose(skateboard, SKATEBOARD_POSE)

    obstacles = [ground_id, skateboard]
    if include_stewart:
        obstacles += [platform]
    print("Robots: " + str([r0, r1]))
    print("Obstacles: " + str(obstacles))
    return robot_list, obstacles


def load_obj(stl_file_path, mass=1.0, base_position=[0, 0, 0], base_orientation=[0, 0, 0, 1], mesh_scale=[1, 1, 1]):
    """
    Loads an STL file into PyBullet as a multibody object.

    Args:
        stl_file_path (str): Path to the STL file.
        mass (float): Mass of the object.
        base_position (list): Initial position [x, y, z] of the object.
        base_orientation (list): Initial orientation [x, y, z, w] (quaternion) of the object.
        mesh_scale (list): Scale factors [x, y, z] for the mesh.

    Returns:
        int: The body ID of the created object.
    """
    # Create visual shape from the STL
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=stl_file_path,
        meshScale=mesh_scale
    )

    # Create collision shape from the STL (optional, for physics interactions)
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=stl_file_path,
        meshScale=mesh_scale
    )

    # Create the multibody object using the visual and collision shapes
    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=base_position,
        baseOrientation=base_orientation
    )

    return body_id


def check_collision(q1, q2, ee_collisions=True, include_zivid=False):
    p.connect(p.DIRECT)
    robots, _ = setup_environment(ee_collisions=ee_collisions, include_zivid=include_zivid)
    pbu.set_joint_positions(robots[0], pbu.get_movable_joints(robots[0]), q1)
    pbu.set_joint_positions(robots[1], pbu.get_movable_joints(robots[1]), q2)
    contact_points = p.getClosestPoints(bodyA=robots[0], bodyB=robots[1], distance=0.0)
    p.disconnect()
    return len(contact_points) > 0


def solve_motion_plan(start_qs, target_q, target_robot=0, step_through=False, obstacle_paths=[], obstacle_poses=[],
                      ee_collisions=True, include_zivid=False):
    random.seed(0)
    np.random.seed(0)

    assert len(obstacle_paths) == len(obstacle_poses)

    if (step_through):
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    robots, obstacles = setup_environment(ee_collisions=ee_collisions, include_zivid=include_zivid)

    for robot, start_q in zip(robots, start_qs):
        pbu.set_joint_positions(robot, pbu.get_movable_joints(robot), start_q)

    robot = robots[target_robot]
    obstacle_robots = [r for r in robots if r != robot]

    motion_plan = pbu.plan_joint_motion(
        robot,
        pbu.get_movable_joints(robot),
        target_q,
        self_collisions=True,
        obstacles=obstacles + obstacle_robots,
        disabled_collisions=IGNORE_COLLISIONS
    )

    if (step_through):
        for q in motion_plan:
            pbu.set_joint_positions(robots[target_robot], pbu.get_movable_joints(robot), q)
            input("Next?")

    print("Motion plan: ")
    print(motion_plan)

    p.disconnect()
    return motion_plan


def solve_ik_araas(start_qs, target_poses, tool_name="tool0", ee_collisions=True, step_through=False, include_zivid=False):
    random.seed(0)
    np.random.seed(0)

    if (step_through):
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    robots, obstacles = setup_environment(ee_collisions=ee_collisions, include_zivid=include_zivid)

    for robot, start_q in zip(robots, start_qs):
        pbu.set_joint_positions(robot, pbu.get_movable_joints(robot), start_q)

    ik_solutions = []
    randomize_seed = [False, False]
    if include_zivid: 
        randomize_seed.append(False)
    max_attempts = 5000

    for i in range(max_attempts):

        if (len(ik_solutions) == len(robots)):
            p.disconnect()
            return ik_solutions

        robot_idx = len(ik_solutions)

        if (i % int(math.sqrt(max_attempts)) == 0):
            ik_solutions = []
            randomize_seed = [True, False, False]
            continue

        robot = robots[robot_idx]
        target_pose = target_poses[robot_idx]

        if (target_pose is None):
            ik_solutions.append(start_qs[robot_idx])
            continue

        link = pbu.link_from_name(robot, tool_name)
        joints = pbu.get_movable_joints(robot)
        ranges = [pbu.get_joint_limits(robot, joint) for joint in joints]

        # Start with the current joint positions and then randomize within limits after
        if (not randomize_seed[robot_idx]):
            initialization_sample = start_qs[robot_idx]
            randomize_seed[robot_idx] = True
        else:
            initialization_sample = [random.uniform(r[0], r[1]) for r in ranges]

        pbu.set_joint_positions(robot, joints, initialization_sample)

        conf = p.calculateInverseKinematics(
            int(robot), link, target_pose[0], target_pose[1],
            residualThreshold=0.00001, maxNumIterations=5000
        )

        lower, upper = list(zip(*ranges))
        if (not pbu.all_between(lower, conf, upper)):
            print("IK solution outside limits")
            continue

        assert len(joints) == len(conf)
        pbu.set_joint_positions(robot, joints, conf)

        contact_points = []
        for obstacle in obstacles:
            contact_points += p.getClosestPoints(bodyA=obstacle, bodyB=robot, distance=pbu.MAX_DISTANCE)

        for r_idx in range(len(ik_solutions)):
            contact_points += p.getClosestPoints(bodyA=robots[r_idx], bodyB=robot, distance=pbu.MAX_DISTANCE)

        all_joints = pbu.get_joints(robot)
        check_link_pairs = (
            pbu.get_self_link_pairs(robot, all_joints, IGNORE_COLLISIONS)
        )

        self_collision = False
        for link1, link2 in check_link_pairs:
            if pbu.pairwise_link_collision(robot, link1, robot, link2):
                print(link1, link2)
                self_collision = True

        if (self_collision):
            print("Self collision")
            continue

        # Print contact points if there are any
        if contact_points:
            print("Collision!")
            # time.sleep(0.5)
            continue

        pose = pbu.get_link_pose(robot, link)

        trans_diff, rot_diff = pbu.get_pose_distance(target_pose, pose)

        if (trans_diff < 0.001 and rot_diff < 0.01):
            ik_solutions.append(conf)
            continue
        else:
            print("IK Error: {}, {}".format(trans_diff, rot_diff))

    p.disconnect()
    return None

def solve_ik(base_pos, base_q, target_pos, target_q, tool_name="tool0"):
    random.seed(0)
    np.random.seed(0)
    
    print("Base pos"+str(base_pos))
    print("Base q"+str(base_q))
    p.connect(p.DIRECT)
    with suppress_stdout():
        robot = p.loadURDF("ur_description/urdf/ur10e.urdf", base_pos, base_q, useFixedBase=True)

    plane_id = p.createCollisionShape(shapeType=p.GEOM_PLANE)
    ground_id = p.createMultiBody(baseCollisionShapeIndex=plane_id)

    # link = link_from_name(robot, "flange")
    link = link_from_name(robot, tool_name)
    joints = get_movable_joints(robot)    
    
    for _ in range(1000):
        print(target_pos)
        ranges = [get_joint_limits(robot, joint) for joint in joints]
        initialization_sample = [random.uniform(r[0], r[1]) for r in ranges]
        set_joint_positions(robot, joints, initialization_sample)
        conf = p.calculateInverseKinematics(
            int(robot), link, target_pos, target_q,residualThreshold=0.001, maxNumIterations=200
        )

        lower, upper = list(zip(*ranges))
        if(not all_between(lower, conf, upper)):
            print("IK solution outside limits")
            continue
        
        assert len(joints) == len(conf)
        set_joint_positions(robot, joints, conf)
        p.stepSimulation()

        contact_points = p.getContactPoints(bodyA=ground_id, bodyB=robot)
        
        # Print contact points if there are any
        if contact_points:
            print("Collision with ground!")
            continue            

        pose = get_link_pose(robot, link)

        trans_diff, rot_diff = get_pose_distance((target_pos, target_q), pose)
        
        print("IK Error: ")
        print(trans_diff, rot_diff)
        if(trans_diff<0.001 and rot_diff < 0.01):
            return conf

    print("No ik solution found")

os.makedirs("logs", exist_ok=True)

class HighPrecisionUnixEpochFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # Get the current time in seconds
        current_time = time.time()
        # Convert to milliseconds
        millis = int(round(current_time * 1000))
        return str(millis)

logFormatter = HighPrecisionUnixEpochFormatter('%(asctime)s:%(message)s')
# logFormatter = logging.Formatter("%(asctime)s:%(message)s")
log = logging.getLogger()
log.setLevel(logging.INFO)  # Set the logger level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

logPath = os.path.join(pathlib.Path(__file__).parent.resolve(), "logs")
fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, str(time.time())))
fileHandler.setFormatter(logFormatter)
log.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)

def obj_to_str(d):
    return json.dumps(d)

def listify(v):
    if isinstance(v, torch.Tensor):
        return v.cpu().detach().numpy().tolist()
    elif isinstance(v, np.ndarray):
        return listify(v.tolist())
    elif isinstance(v, list):
        return [listify(ve) for ve in v]
    elif (isinstance(v, float) or isinstance(v, int)):
        return v
    else:
        raise NotImplementedError


CAPTURE_DEBUG_LOG = False

def log_list(name, vector: List[float]):
    if (CAPTURE_DEBUG_LOG):
        vector = listify(vector)
        assert isinstance(vector, list)
        log.info("ADKPlot:"+str(obj_to_str({name: vector})))
        


def get_scripted_actions(num_envs, num_robots=1, repeat=10):
    scripted_actions = [torch.tensor([[0, 0, 1]*num_robots]).repeat([num_envs, 1])]*repeat+\
                       [torch.tensor([[0, 0, -1]*num_robots]).repeat([num_envs, 1])]*repeat+\
                       [torch.tensor([[0, 1, 0]*num_robots]).repeat([num_envs, 1])]*repeat+\
                       [torch.tensor([[0, -1, 0]*num_robots]).repeat([num_envs, 1])]*repeat+\
                       [torch.tensor([[1, 0, 0]*num_robots]).repeat([num_envs, 1])]*repeat+\
                       [torch.tensor([[-1, 0, 0]*num_robots]).repeat([num_envs, 1])]*repeat
    return scripted_actions

FLANGE_T_TOOL = ([0,0,0], [0.5, 0.5, 0.5, 0.5]) # quat: [xyzw]
# PLATFORM_POSE = ([[0.0, 0.8286200561523438, 0.0],  [ 0, 0, -0.9659258, 0.258819 ]])
# TOOL_T_TIP = ([0, 0, 0.27000001072883606], [0, 0, 0, 1])
PLATFORM_POSE = ([[0.0, 0.809, 0.006], [0, 0, -0.9659258, 0.258819]])
TOOL_T_TIP = ([0, 0, 0.28], [0, 0, 0, 1])
ELBOW_PICK_POSE = ([-0.39973, 0.25, 0.270], [-1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.3267949e-06])
BOLT_PICK_POSE = ([-0.35973, 0.25, 0.290], [-1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.3267949e-06])
PLATFORM_T_MEDGEAR = ([0.2172, 0.0617, 0.2647], [-4.3298e-17, -7.0711e-01,  7.0711e-01, 4.3298e-17])
PLATFORM_T_SMALLGEAR =  ([0.1872, 0.0617, 0.2647], [-4.3298e-17, -7.0711e-01,  7.0711e-01, 4.3298e-17])
PLATFORM_T_LARGEGEAR = ([0.2672, 0.0617, 0.2647], [-4.3298e-17, -7.0711e-01,  7.0711e-01, 4.3298e-17])

GEAR_PICK_POSE = ([0.640, 0.11735, 0.270], [1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.3267949e-06])
TASKBOARD_POSE = ([0.150, 0.315, -0.027], [0.7071068, 0, 0, 0.7071068])

# Hardcoded kit pose and grasp info from isaac sim
WORLD_T_KIT = ([-0.073, 0.295, 0.01], [0, 0, -0.70711, 0.70711])
KIT_T_ELBOW_0 = ([0.09273, -0.07578, -0.01149], [0, -0.70711, 0.70711, 0])
KIT_T_STRUT_5 = ([-0.11382, 0.04123, 0.01], [-0.5, 0.5, 0.5, 0.5])
KIT_T_BOLT_0 = ([0.10745, 0, 0.0267], [0.70711, 0.70711, 0, 0])
STRUT_GRASP = ([0.022, 0.0, 0.032], [0.5, 0.5, 0.5, 0.5])

BOLT_GRASP = ([0, 0, 0.016], [0, 0, 0.70711, 0.70711])
ELBOW_GRASP = ([-0.01888, 0.00769, -0.01395], [-0.70711, 0, 0, 0.70711])
PLATFORM_T_STRUT = ([[0.0002, 0.15019, 0.23621], [0, 0, -0.88691, 0.46195]])
PLATFORM_T_BOLT = ([[-0.00792, 0.0856, 0.41553], [0.62253, 0.33534, -0.62253, 0.33534]])
STRUT_T_BOLT = ([[-0.01209, 0, 0.19243], [0, 0.70711, 0, 0.70711]])

# [x, y, z]
# y
# |
# table--x
# WORLD_T_FIXTURE_POSE = ([0.0, 0.8, 0.06], [0, 0, 0.7071068, -0.7071068])  # quat: [xyzw]
# WORLD_T_HOLE_START = ([0.1, 0.4, 0.025], [0, 0, 0, 1])
# MEDIUM_SHORT_OVAL_JEANSBLUE_POSE = ([0.3, 0.8, 0.1], [0, 0, 0, 1])
WORLD_T_FIXTURE_POSE = ([0.3, 0.2, 0.06], [0, 0, 0.7071068, -0.7071068])  # quat: [xyzw] # (0, 0, 270)
HOLE_ROTATE = ([0., 0., 0.], [0, 0, 1, 0])
WORLD_T_HOLE_START = multiply(([0., 0.18, 0.025], [0, 0, 0, 1]), HOLE_ROTATE)
# WORLD_T_HOLE_START = ([0., 0.22, 0.025], [0, 0, 0, 1])
# MEDIUM_SHORT_OVAL_JEANSBLUE_POSE = ([0.55, 0.25, 0.05], [0, 0, 0, 1])  # Good
MEDIUM_SHORT_OVAL_JEANSBLUE_POSE = ([0.85, 0.4, 0.05], [0, 0, 0, 1])
PEG_T_TIP_DEFAULT = ([0, 0, -0.01], [-1, 0, 0, 0])
# WORLD_T_ZIVID_HOME = ([0.251870987,-0.595301941,0.760453064], [-0.9110028, 0, 0, 0.4124002])
WORLD_T_ZIVID_HOME = ([0.2,-0.5,0.760453064], [-0.9110028, 0, 0, 0.4124002])

MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2 = ([0.85, 0.4, 0.05], [0, 0, 0, 1])

OBJECT_HORIZONTAL_ROTATION = ([0., 0., 0.], [ 0, 0, 0, 1 ])
# OBJECT_DICT_v0 = {
#     'Medium_Short_Hexagon_Green': {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.45, 0.3, 0.0], [0, 0, 1, 0])),
#                                    'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Hexagon_Green_Updated.usd",
#                                    'goal_pose': ([-0.070, -0.089, 0.03], [0,0,1,0]),
#                                    'peg_attachment_prim': "peg/Medium_Short_Hexagon_Green/Medium_Short_Hexagon_Green/obj1_012_obj1_005"},
#     'Medium_Short_Star_DarkBlue': {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([0., -0.15, 0.0], [0, 0, 1, 0])),
#                                    'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Star_DarkBlue_Updated.usd",
#                                    'goal_pose': ([-0.070, -0.019, 0.03], [0,0,1,0]),
#                                    'peg_attachment_prim': "peg/Medium_Short_Star_DarkBlue/Medium_Short_Star_DarkBlue/obj1_018_obj1_005"},
#     'Medium_Short_SquareCircle_Red':  {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([0., -0.3, 0.0], [0, 1, 0, 0])),
#                                        'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_SquareCircle_Red_Updated.usd",
#                                        'goal_pose': ([-0.069, 0.049, 0.03], [0,0,0,1]),
#                                        'peg_attachment_prim': "peg/Medium_Short_SquareCircle_Red/Medium_Short_SquareCircle_Red/obj1_002_obj1_006"},
#     'Medium_Short_3Prong_JeansRed':  {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.6, 0.15, 0.0], [0, 1, 0, 0])),
#                                       'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_3Prong_JeansRed_Updated.usd",
#                                       'goal_pose': ([-0.002, -0.089, 0.03], [0,0,0,1]),
#                                       'peg_attachment_prim': "peg/Medium_Short_3Prong_JeansRed/Medium_Short_3Prong_JeansRed/obj1_014_obj1_005"},
#     'Medium_Short_Circle_Brown': {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.15, -0.15, 0.0], [0, 0, 0, 1])),
#                                   'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Circle_Brown_Updated.usd",
#                                   'goal_pose': ([-0.0045, -0.0184, 0.03], [0,0,1,0]),
#                                   'peg_attachment_prim': "peg/Medium_Short_Circle_Brown/Medium_Short_Circle_Brown/obj1_011_obj1_005"},
#     'Medium_Short_Oval_JeansBlue': {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.15, -0.3, 0.0], [0, 0, 0, 1])),
#                                     'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Oval_JeansBlue_Updated.usd",
#                                     'goal_pose': ([0.00, 0.049, 0.03], [0,0,1,0]),
#                                     'peg_attachment_prim': "peg/Medium_Short_Oval_JeansBlue/Medium_Short_Oval_JeansBlue/obj1_004_obj1_005"},  # Oval goal_pose need to be adjusted
#     'Medium_Short_Arch_Yellow':  {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.3, -0., 0.0], [0, 0, 0, 1])),
#                                   'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Arch_Yellow_Updated.usd",
#                                   'goal_pose': ([0.070, -0.089, 0.03], [0,0,1,0]),
#                                   'peg_attachment_prim': "peg/Medium_Short_Arch_Yellow/Medium_Short_Arch_Yellow/obj1_020_obj1_005"},
#     'Medium_Short_DoubleSquare_Purple':  {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.45, -0., 0.0], [0, 1, 0, 0])),
#                                           'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_DoubleSquare_Purple_Updated.usd",
#                                           'goal_pose': ([0.070, -0.019, 0.03], [0,0,0,1]),
#                                           'peg_attachment_prim': "peg/Medium_Short_DoubleSquare_Purple/Medium_Short_DoubleSquare_Purple/obj1_015_obj1_005"},
#     'Medium_Short_Rectangle_JeansBlue':  {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.45, 0.15, 0.0], [0, 0, 1, 0])),
#                                           'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Rectangle_JeansBlue_Updated.usd",
#                                           'goal_pose': ([0.070, 0.049, 0.03], [0,0,0,1]),
#                                           'peg_attachment_prim': "peg/Medium_Short_Rectangle_JeansBlue/Medium_Short_Rectangle_JeansBlue/obj1_009"},
# }

OBJECT_DICT_v1 = {
    'Medium_Short_Hexagon_Green': {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.45, 0.3, 0.0], [0, 0, 1, 0])),
                                   'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Hexagon_Green_Updated.usd",
                                   'goal_pose': ([-0.0683, -0.0884, 0.03], [0,0,1,0]),
                                   'peg_attachment_prim': "peg/Medium_Short_Hexagon_Green/Medium_Short_Hexagon_Green/obj1_012_obj1_005"},
    'Medium_Short_Star_DarkBlue': {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([0., -0.15, 0.0], [0, 0, 1, 0])),
                                   'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Star_DarkBlue_Updated.usd",
                                   'goal_pose': ([-0.0688, -0.0189, 0.03], [0,0,1,0]),
                                   'peg_attachment_prim': "peg/Medium_Short_Star_DarkBlue/Medium_Short_Star_DarkBlue/obj1_018_obj1_005"},
    'Medium_Short_SquareCircle_Red':  {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([0., -0.3, 0.0], [0, 1, 0, 0])),
                                       'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_SquareCircle_Red_Updated.usd",
                                       'goal_pose': ([-0.0672, 0.051, 0.03], [0,0,0,1]),
                                       'peg_attachment_prim': "peg/Medium_Short_SquareCircle_Red/Medium_Short_SquareCircle_Red/obj1_002_obj1_006"},
    'Medium_Short_3Prong_JeansRed':  {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.6, 0.15, 0.0], [0, 1, 0, 0])),
                                      'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_3Prong_JeansRed_Updated.usd",
                                      'goal_pose': ([-0.0011, -0.0887, 0.03], [0,0,0,1]),
                                      'peg_attachment_prim': "peg/Medium_Short_3Prong_JeansRed/Medium_Short_3Prong_JeansRed/obj1_014_obj1_005"},
    'Medium_Short_Circle_Brown': {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.15, -0.15, 0.0], [0, 0, 0, 1])),
                                  'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Circle_Brown_Updated.usd",
                                  'goal_pose': ([-0.0028, -0.0163, 0.03], [0,0,1,0]),
                                  'peg_attachment_prim': "peg/Medium_Short_Circle_Brown/Medium_Short_Circle_Brown/obj1_011_obj1_005"},
    'Medium_Short_Oval_JeansBlue': {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.15, -0.3, 0.0], [0, 0, 0, 1])),
                                    'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Oval_JeansBlue_Updated.usd",
                                    'goal_pose': ([0.006, 0.0495, 0.03], [0,0,1,0]),
                                    'peg_attachment_prim': "peg/Medium_Short_Oval_JeansBlue/Medium_Short_Oval_JeansBlue/obj1_004_obj1_005"},  # Oval goal_pose need to be adjusted
    'Medium_Short_Arch_Yellow':  {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.3, -0., 0.0], [0, 0, 0, 1])),
                                  'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Arch_Yellow_Updated.usd",
                                  'goal_pose': ([0.0724, -0.0976, 0.03], [0,0,1,0]),
                                  'peg_attachment_prim': "peg/Medium_Short_Arch_Yellow/Medium_Short_Arch_Yellow/obj1_020_obj1_005"},
    'Medium_Short_DoubleSquare_Purple':  {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.45, -0., 0.0], [0, 1, 0, 0])),
                                          'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_DoubleSquare_Purple_Updated.usd",
                                          'goal_pose': ([0.0697, -0.017, 0.03], [0,0,0,1]),
                                          'peg_attachment_prim': "peg/Medium_Short_DoubleSquare_Purple/Medium_Short_DoubleSquare_Purple/obj1_015_obj1_005"},
    'Medium_Short_Rectangle_JeansBlue':  {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.45, 0.15, 0.0], [0, 0, 1, 0])),
                                          'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Rectangle_JeansBlue_Updated.usd",
                                          'goal_pose': ([0.0714, 0.0502, 0.03], [0,0,0,1]),
                                          'peg_attachment_prim': "peg/Medium_Short_Rectangle_JeansBlue/Medium_Short_Rectangle_JeansBlue/obj1_009"},
}


OBJECT_DICT = {
    'Medium_Short_Hexagon_Green': {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.45, 0.3, 0.0], [0, 0, 1, 0])),
                                   'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Hexagon_Green_Updated.usd",
                                   'goal_pose': ([-0.0695, -0.0895, 0.03], [0,0,1,0]),
                                   'peg_attachment_prim': "peg/Medium_Short_Hexagon_Green/Medium_Short_Hexagon_Green/obj1_012_obj1_005"},
    'Medium_Short_Star_DarkBlue': {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([0., -0.15, 0.0], [0, 0, 1, 0])),
                                   'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Star_DarkBlue_Updated.usd",
                                   'goal_pose': ([-0.0688, -0.0189, 0.03], [0,0,1,0]),
                                   'peg_attachment_prim': "peg/Medium_Short_Star_DarkBlue/Medium_Short_Star_DarkBlue/obj1_018_obj1_005"},
    'Medium_Short_SquareCircle_Red':  {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([0., -0.3, 0.0], [0, 1, 0, 0])),
                                       'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_SquareCircle_Red_Updated.usd",
                                       'goal_pose': ([-0.0674, 0.051, 0.03], [0,0,0,1]),
                                       'peg_attachment_prim': "peg/Medium_Short_SquareCircle_Red/Medium_Short_SquareCircle_Red/obj1_002_obj1_006"},
    'Medium_Short_3Prong_JeansRed':  {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.6, 0.15, 0.0], [0, 1, 0, 0])),
                                      'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_3Prong_JeansRed_Updated.usd",
                                      'goal_pose': ([-0.0011, -0.0887, 0.02], [0,0,0,1]),
                                      'peg_attachment_prim': "peg/Medium_Short_3Prong_JeansRed/Medium_Short_3Prong_JeansRed/obj1_014_obj1_005"},
    'Medium_Short_Circle_Brown': {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.15, -0.15, 0.0], [0, 0, 0, 1])),
                                  'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Circle_Brown_Updated.usd",
                                  'goal_pose': ([-0.0030, -0.0163, 0.03], [0,0,1,0]),
                                  'peg_attachment_prim': "peg/Medium_Short_Circle_Brown/Medium_Short_Circle_Brown/obj1_011_obj1_005"},
    'Medium_Short_Oval_JeansBlue': {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.15, -0.3, 0.0], [0, 0, 0, 1])),
                                    'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Oval_JeansBlue_Updated.usd",
                                    'goal_pose': ([0.00, 0.0495, 0.03], [0,0,1,0]),
                                    'peg_attachment_prim': "peg/Medium_Short_Oval_JeansBlue/Medium_Short_Oval_JeansBlue/obj1_004_obj1_005"},  # Oval goal_pose need to be adjusted
    'Medium_Short_Arch_Yellow':  {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.3, -0., 0.0], [0, 0, 0, 1])),
                                  'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Arch_Yellow_Updated.usd",
                                  'goal_pose': ([0.0724, -0.0976, 0.03], [0,0,1,0]),
                                  'peg_attachment_prim': "peg/Medium_Short_Arch_Yellow/Medium_Short_Arch_Yellow/obj1_020_obj1_005"},
    'Medium_Short_DoubleSquare_Purple':  {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.45, -0., 0.0], [0, 1, 0, 0])),
                                          'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_DoubleSquare_Purple_Updated.usd",
                                          'goal_pose': ([0.0697, -0.018, 0.03], [0,0,0,1]),
                                          'peg_attachment_prim': "peg/Medium_Short_DoubleSquare_Purple/Medium_Short_DoubleSquare_Purple/obj1_015_obj1_005"},
    'Medium_Short_Rectangle_JeansBlue':  {'start_pose': multiply(MEDIUM_SHORT_HEXAGON_GREEN_POSE_v2, ([-0.45, 0.15, 0.0], [0, 0, 1, 0])),
                                          'usd_path': home_dir + "taskboard/fmb_example/Medium_Short_Rectangle_JeansBlue_Updated.usd",
                                          'goal_pose': ([0.0712, 0.047, 0.03], [0,0,0,1]),
                                          'peg_attachment_prim': "peg/Medium_Short_Rectangle_JeansBlue/Medium_Short_Rectangle_JeansBlue/obj1_009"},
}

# goal_pose_0 = BEAM_OBJECT_DICT['1']['goal_pose'] #= multiply(BEAM_CENTER, CENTER_T_GOAL)
# CENTER_T_GOAL = multiply(invert(BEAM_CENTER), goal_pose_0)
# new_goal_pose_0 = multiply(BEAM_CENTER, CENTER_T_GOAL)

# print('A'*50, goal_pose_0, new_goal_pose_0, CENTER_T_GOAL)
# QIANZHONG tuned
# STOOL_WORLD_T_FIXTURE_POSE  STOOL_WORLD_T_HOLE_START 

# # Priviously used
# STOOL_CENTER =  ([0.6, 0.6, 0.0], [0, 0, 0, 1])
# STOOL_OBJECT_DICT = {
#     '0': {'start_pose': multiply(STOOL_CENTER, ([-0.463, -0.338, 0.02], [0, 0, 0, 1])), # red hat
#                                    'usd_path': home_dir + "taskboard/stool_circular/0_Updated.usd",
#                                    'goal_pose': multiply(STOOL_CENTER, ([-0.463, -0.338, 0.02], [0, 0, 0, 1])), # ([-0.463, -0.338, 0.02], [0, 0, 1, 0])),
#                                    'peg_attachment_prim': "peg/_/node_/mesh"},
#     '1': {'start_pose': multiply(STOOL_CENTER, ([-0.405, -0.395, 0.015], [0, 0, 0, 1])), # yellow pole
#                                    'usd_path': home_dir + "taskboard/stool_circular/1_Updated.usd", 
#                                    'goal_pose': multiply(STOOL_CENTER, ([-0.405, -0.395, 0.015], [0, 0, 0, 1])),
#                                    'peg_attachment_prim': "peg/_/node_/mesh"},
#     '2':  {'start_pose': multiply(STOOL_CENTER, ([-0.4, -0.4, 0.005], [0, 0, 0, 1])), # base
#                                     'usd_path': home_dir + "taskboard/stool_circular/2_Updated.usd", 
#                                     'goal_pose': multiply(STOOL_CENTER, ([-0.4, -0.4, 0.005], [0, 0, 0, 1])), # ([-0.0674, 0.051, 0.03], [0,0,0,1]),
#                                     'peg_attachment_prim': "peg/_/node_/mesh"},
#     '3':  {'start_pose': multiply(STOOL_CENTER, ([-0.456, -0.4050, 0.020], [0, 0, 0, 1])), # red pole
#                                     'usd_path': home_dir + "taskboard/stool_circular/3_Updated.usd", 
#                                     'goal_pose': multiply(STOOL_CENTER, ([-0.456, -0.4050, 0.020], [0, 0, 0, 1])), # ([-0.0011, -0.0887, 0.02], [0,0,0,1]),
#                                     'peg_attachment_prim': "peg/_/node_/mesh"},
#     '4': {'start_pose': multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), # green (yellow) hat
#                                   'usd_path': home_dir + "taskboard/stool_circular/4_Updated.usd",
#                                   'goal_pose': multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), # ([-0.0030, -0.0163, 0.03], [0,0,1,0]),
#                                   'peg_attachment_prim': "peg/_/node_/mesh"},
#     '5': {'start_pose': multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), # blue pole
#                                   'usd_path': home_dir + "taskboard/stool_circular/5_Updated.usd", 
#                                   'goal_pose': multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), # ([-0.0030, -0.0163, 0.03], [0,0,1,0]),
#                                   'peg_attachment_prim': "peg/_/node_/mesh"},
#     '6': {'start_pose': multiply(STOOL_CENTER, ([-0.404, -0.34, 0.015], [0, 0, 0, 1])), # purple hat
#                                   'usd_path': home_dir + "taskboard/stool_circular/6_Updated.usd",
#                                   'goal_pose': multiply(STOOL_CENTER, ([-0.404, -0.34, 0.015], [0, 0, 0, 1])), # ([-0.0030, -0.0163, 0.03], [0,0,1,0]),
#                                   'peg_attachment_prim': "peg/_/node_/mesh"},
#     '7': {'start_pose': multiply(STOOL_CENTER, ([-0.396, -0.3950, 0.018], [0, 0, 0, 1])), # green pole
#                                   'usd_path': home_dir + "taskboard/stool_circular/7_Updated.usd", 
#                                   'goal_pose': multiply(STOOL_CENTER, ([-0.396, -0.3950, 0.018], [0, 0, 0, 1])), # ([-0.0030, -0.0163, 0.03], [0,0,1,0]),
#                                   'peg_attachment_prim': "peg/_/node_/mesh"},
#     '8': {'start_pose': multiply(STOOL_CENTER, ([-0.336, -0.398, 0.03], [0, 0, 0, 1])), # yellow hat
#                                   'usd_path': home_dir + "taskboard/stool_circular/8_Updated.usd",
#                                   'goal_pose': multiply(STOOL_CENTER, ([-0.336, -0.398, 0.03], [0, 0, 0, 1])), # ([-0.0030, -0.0163, 0.03], [0,0,1,0]),
#                                   'peg_attachment_prim': "peg/_/node_/mesh"},
# }


STOOL_CENTER =  ([0.6, 0.6, 0.0], [0, 0, 0, 1])
STOOL_OBJECT_DICT = {
    '0': {'start_pose': multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), # red hat
        'usd_path': home_dir + "taskboard/stool_circular/0_Updated.usd",
        'goal_pose': multiply(STOOL_CENTER, ([-0.40, -0.40, 0.0], [0, 0, 0, 1])), # ([-0.463, -0.338, 0.02], [0, 0, 1, 0])),
        'peg_attachment_prim': "peg/_/node_/mesh"},
    '1': {'start_pose': multiply(STOOL_CENTER, ([-0.405, -0.395, 0.018], [0, 0, 0, 1])), # yellow pole
        'usd_path': home_dir + "taskboard/stool_circular/1_Updated.usd", 
        'goal_pose': multiply(STOOL_CENTER, ([-0.405, -0.395, 0.0], [0, 0, 0, 1])),
        'peg_attachment_prim': "peg/_/node_/mesh"},
    '2':  {'start_pose': multiply(STOOL_CENTER, ([-0.4, -0.4, 0.005], [0, 0, 0, 1])), # base
        'usd_path': home_dir + "taskboard/stool_circular/2_Updated.usd", 
        'goal_pose': multiply(STOOL_CENTER, ([-0.4, -0.4, 0.005], [0, 0, 0, 1])), # ([-0.0674, 0.051, 0.03], [0,0,0,1]),
        'peg_attachment_prim': "peg/_/node_/mesh"},
    # '3':  {'start_pose': multiply(STOOL_CENTER, ([-0.457, -0.4070, 0.016], [0, 0, 0, 1])), # red pole
    #     'usd_path': home_dir + "taskboard/stool_circular/3_Updated.usd", 
    #     'goal_pose': multiply(STOOL_CENTER, ([-0.457, -0.4070, 0.0], [0, 0, 0, 1])), # ([-0.0011, -0.0887, 0.02], [0,0,0,1]),
    #     'peg_attachment_prim': "peg/_/node_/mesh"},
    
    '3':  {'start_pose':multiply(STOOL_CENTER, ([-0.455, -0.403, 0.02], [0, 0, 0, 1])), # red pole
        'usd_path': home_dir + "taskboard/stool_circular/3_Updated.usd", 
        'goal_pose': multiply(STOOL_CENTER, ([-0.455, -0.403, 0.0], [0, 0, 0, 1])), # ([-0.0011, -0.0887, 0.02], [0,0,0,1]),
        'peg_attachment_prim': "peg/_/node_/mesh"},
    # '3':  {'start_pose': multiply(STOOL_CENTER, ([-0.454, -0.4070, 0.02], [0, 0, 0, 1])), # red pole
    #     'usd_path': home_dir + "taskboard/stool_circular/3_Updated.usd", 
    #     'goal_pose': multiply(STOOL_CENTER, ([-0.454, -0.4070, 0.0], [0, 0, 0, 1])), # ([-0.0011, -0.0887, 0.02], [0,0,0,1]),
    #     'peg_attachment_prim': "peg/_/node_/mesh"},
    '4': {'start_pose': multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), # blue hat
        'usd_path': home_dir + "taskboard/stool_circular/4_Updated.usd",
        'goal_pose': multiply(STOOL_CENTER, ([-0.465, -0.465, 0.00], [0, 0, 0, 1])), # ([-0.0030, -0.0163, 0.03], [0,0,1,0]),
        'peg_attachment_prim': "peg/_/node_/mesh"},
    '5': {'start_pose': multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), # blue pole
        'usd_path': home_dir + "taskboard/stool_circular/5_Updated.usd", 
        'goal_pose': multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.0], [0, 0, 0, 1])), # ([-0.0030, -0.0163, 0.03], [0,0,1,0]),
        'peg_attachment_prim': "peg/_/node_/mesh"},
    '6': {'start_pose': multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), # purple hat
        'usd_path': home_dir + "taskboard/stool_circular/6_Updated.usd",
        'goal_pose': multiply(STOOL_CENTER, ([-0.404, -0.338, 0.0], [0, 0, 0, 1])), # ([-0.0030, -0.0163, 0.03], [0,0,1,0]),
        'peg_attachment_prim': "peg/_/node_/mesh"},
    '7': {'start_pose': multiply(STOOL_CENTER, ([-0.396, -0.3950, 0.018], [0, 0, 0, 1])), # green pole
        'usd_path': home_dir + "taskboard/stool_circular/7_Updated.usd", 
        'goal_pose': multiply(STOOL_CENTER, ([-0.396, -0.3950, 0.0], [0, 0, 0, 1])), # ([-0.0030, -0.0163, 0.03], [0,0,1,0]),
        'peg_attachment_prim': "peg/_/node_/mesh"},
    '8': {'start_pose': multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), # yellow hat
        'usd_path': home_dir + "taskboard/stool_circular/8_Updated.usd",
        'goal_pose': multiply(STOOL_CENTER, ([-0.338, -0.397, 0.0], [0, 0, 0, 1])), # ([-0.0030, -0.0163, 0.03], [0,0,1,0]),
        'peg_attachment_prim': "peg/_/node_/mesh"},
}

EXTRA_PARTS_list = []
EXTRA_PARTS_STARTING_POSE_list = []
for k, v in OBJECT_DICT.items():
    # if k in ['Medium_Short_Oval_JeansBlue']:
    #     continue
    EXTRA_PARTS_list.append(v['usd_path'])
    EXTRA_PARTS_STARTING_POSE_list.append(v['start_pose'])
# part type is a single rigid body with no joints
# articulated_part type is a set of rigid bodies connected by joints
# peripheral is a single rigid body that doesn't move in the scene
OBJECT_HORIZONTAL_ROTATION = ([0., 0., -0.03], [0.7071068, 0, 0, 0.7071068])

def split_obj_dict(obj_dict=None, selected_obj_id=0, obj_style='vertical'):
    extra_obj_names = []
    extra_obj_usd_paths = []
    extra_obj_start_poses = []
    extra_obj_goal_poses = []

    for i, (k, v) in enumerate(obj_dict.items()):
        # print('i, selected_obj_id: ', i, selected_obj_id)
        if obj_style == 'horizontal':
            if k in ['Medium_Short_SquareCircle_Red', 'Medium_Short_3Prong_JeansRed', 'Medium_Short_DoubleSquare_Purple']:
                v['start_pose'] = multiply(v['start_pose'], ([OBJECT_HORIZONTAL_ROTATION[0][0], OBJECT_HORIZONTAL_ROTATION[0][1], -OBJECT_HORIZONTAL_ROTATION[0][2]], OBJECT_HORIZONTAL_ROTATION[1]))
            else:
                v['start_pose'] = multiply(v['start_pose'], OBJECT_HORIZONTAL_ROTATION)
        if int(i) == int(selected_obj_id):
            target_obj_name = k
            target_obj_usd_path = v['usd_path']
            target_obj_start_pose = v['start_pose']
            target_obj_goal_pose = v['goal_pose']
            target_peg_attachment_prim = v['peg_attachment_prim']
        else:
            extra_obj_names.append(k)
            extra_obj_usd_paths.append(v['usd_path'])
            extra_obj_start_poses.append(v['start_pose'])
            extra_obj_goal_poses.append(v['goal_pose'])
    return (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
            extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses)

def beam_split_obj_dict(obj_dict=None, selected_obj_id=0, obj_style='vertical'):
    extra_obj_names = []
    extra_obj_usd_paths = []
    extra_obj_start_poses = []
    extra_obj_goal_poses = []
    
    if selected_obj_id == 0:
        hole_index = 3
    elif selected_obj_id == 1:
        hole_index = 2
    elif selected_obj_id == 2:
        hole_index = 4
    elif selected_obj_id == 3:
        hole_index = 4
    else:
        hole_index = None
    
    for i, (k, v) in enumerate(obj_dict.items()):
        # print('i, selected_obj_id: ', i, selected_obj_id)
        # if obj_style == 'horizontal':
        #     if k in ['Medium_Short_SquareCircle_Red', 'Medium_Short_3Prong_JeansRed', 'Medium_Short_DoubleSquare_Purple']:
        #         v['start_pose'] = multiply(v['start_pose'], ([OBJECT_HORIZONTAL_ROTATION[0][0], OBJECT_HORIZONTAL_ROTATION[0][1], -OBJECT_HORIZONTAL_ROTATION[0][2]], OBJECT_HORIZONTAL_ROTATION[1]))
        #     else:
        #         v['start_pose'] = multiply(v['start_pose'], OBJECT_HORIZONTAL_ROTATION)
        if int(i) == int(selected_obj_id):
            target_obj_name = k
            target_obj_usd_path = v['usd_path']
            target_obj_start_pose = v['start_pose']
            target_obj_goal_pose = v['goal_pose']
            target_peg_attachment_prim = v['peg_attachment_prim']
        elif int(i) == int(hole_index):
            hole_obj_name = k
            hole_obj_usd_path = v['usd_path']
            hole_obj_start_pose = v['start_pose']
            hole_obj_goal_pose = v['goal_pose']
            hole_peg_attachment_prim = v['peg_attachment_prim']
        else:
            extra_obj_names.append(k)
            extra_obj_usd_paths.append(v['usd_path'])
            extra_obj_start_poses.append(v['start_pose'])
            extra_obj_goal_poses.append(v['goal_pose'])
    return (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
            extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses,
            hole_obj_name, hole_obj_usd_path, hole_obj_start_pose, hole_obj_goal_pose, hole_peg_attachment_prim,)

def stool_split_obj_dict(obj_dict=None, selected_obj_id=0, obj_style='vertical'):
    extra_obj_names = []
    extra_obj_usd_paths = []
    extra_obj_start_poses = []
    extra_obj_goal_poses = []
    
    if selected_obj_id == 0:
        hole_index = 1
    elif selected_obj_id == 1:
        hole_index = 2
    elif selected_obj_id == 3:
        hole_index = 2
    elif selected_obj_id == 4:
        hole_index = 3
    elif selected_obj_id == 5:
        hole_index = 2
    elif selected_obj_id == 6:
        hole_index = 7
    elif selected_obj_id == 7:
        hole_index = 2
    elif selected_obj_id == 8:
        hole_index = 5
    else:
        hole_index = None
    
    for i, (k, v) in enumerate(obj_dict.items()):
        # print('i, selected_obj_id: ', i, selected_obj_id)
        # if obj_style == 'horizontal':
        #     if k in ['Medium_Short_SquareCircle_Red', 'Medium_Short_3Prong_JeansRed', 'Medium_Short_DoubleSquare_Purple']:
        #         v['start_pose'] = multiply(v['start_pose'], ([OBJECT_HORIZONTAL_ROTATION[0][0], OBJECT_HORIZONTAL_ROTATION[0][1], -OBJECT_HORIZONTAL_ROTATION[0][2]], OBJECT_HORIZONTAL_ROTATION[1]))
        #     else:
        #         v['start_pose'] = multiply(v['start_pose'], OBJECT_HORIZONTAL_ROTATION)
        if int(i) == int(selected_obj_id):
            target_obj_name = k
            target_obj_usd_path = v['usd_path']
            target_obj_start_pose = v['start_pose']
            target_obj_goal_pose = v['goal_pose']
            target_peg_attachment_prim = v['peg_attachment_prim']
        elif int(i) == int(hole_index):
            hole_obj_name = k
            hole_obj_usd_path = v['usd_path']
            hole_obj_start_pose = v['start_pose']
            hole_obj_goal_pose = v['goal_pose']
            hole_peg_attachment_prim = v['peg_attachment_prim']
        else:
            extra_obj_names.append(k)
            extra_obj_usd_paths.append(v['usd_path'])
            extra_obj_start_poses.append(v['start_pose'])
            extra_obj_goal_poses.append(v['goal_pose'])
    return (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
            extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses,
            hole_obj_name, hole_obj_usd_path, hole_obj_start_pose, hole_obj_goal_pose, hole_peg_attachment_prim,)

SAFE_POSE_AFTER = {
    # 'AssemblyScriptedInsert': ([0.06829987, 0.26839996, 0.42500003], [-7.54978864e-08, -1.00000000e+00, -1.94707178e-07, 7.63183015e-08]),
    'AssemblyScriptedInsert': ([0.06829987, 0.26839996, 0.5], [-7.54978864e-08, -1.00000000e+00, -1.94707178e-07, 7.63183015e-08]),
    'AssemblyInsert': ([0.06829987, 0.26839996, 0.5], [-7.54978864e-08, -1.00000000e+00, -1.94707178e-07, 7.63183015e-08]),
    'AssemblyScriptedGraspHorizontal': None,
    'AssemblyScriptedGrasp': None,
    'AssemblyReset': ([0.06829987, 0.26839996, 0.5], [-7.54978864e-08, -1.00000000e+00, -1.94707178e-07, 7.63183015e-08]),
    'AssemblyMoveToBoard': ([0.00, 0.18, 0.6], [0, -1.0, 0, 0]),
    'AssemblyMoveToBoardFromFixture': ([0.00, 0.18, 0.6], [0, -1.0, 0, 0]),
    'AssemblyMoveToBoardFromGrasp': ([0.00, 0.18, 0.6], [0, -1.0, 0, 0]),
}

BEAM_SAFE_POSE_AFTER = {
    'BeamScriptedInsert': ([0.06829987, 0.26839996, 0.5], [-7.54978864e-08, -1.00000000e+00, -1.94707178e-07, 7.63183015e-08]),
    'BeamInsert': ([0.06829987, 0.26839996, 0.5], [-7.54978864e-08, -1.00000000e+00, -1.94707178e-07, 7.63183015e-08]),
    'BeamScriptedGraspHorizontal': None,
    'BeamScriptedGrasp': None,
    'BeamScriptedPlace': None,
    'BeamReset': ([0.06829987, 0.26839996, 0.5], [-7.54978864e-08, -1.00000000e+00, -1.94707178e-07, 7.63183015e-08]),
    'BeamMoveToBoard': ([0.00, 0.18, 0.6], [0, -1.0, 0, 0]),
    'BeamMoveToBoardFromFixture': ([0.00, 0.18, 0.6], [0, -1.0, 0, 0]),
    'BeamMoveToBoardFromGrasp': ([0.00, 0.18, 0.6], [0, -1.0, 0, 0]),
}

STOOL_SAFE_POSE_AFTER = {
    'StoolScriptedInsert': ([0.06829987, 0.26839996, 0.5], [-7.54978864e-08, -1.00000000e+00, -1.94707178e-07, 7.63183015e-08]),
    'StoolInsert': ([0.06829987, 0.26839996, 0.5], [-7.54978864e-08, -1.00000000e+00, -1.94707178e-07, 7.63183015e-08]),
    'StoolScriptedGraspHorizontal': None,
    'StoolScriptedGrasp': None,
    'StoolScriptedPlace': None,
    'StoolReset': ([0.06829987, 0.26839996, 0.5], [-7.54978864e-08, -1.00000000e+00, -1.94707178e-07, 7.63183015e-08]),
    'StoolMoveToBoard': ([0.00, 0.18, 0.6], [0, -1.0, 0, 0]),
    'StoolMoveToBoardFromFixture': ([0.00, 0.18, 0.6], [0, -1.0, 0, 0]),
    'StoolMoveToBoardFromGrasp': ([0.00, 0.18, 0.6], [0, -1.0, 0, 0]),
}

@dataclass
class TaskConfig():
    PEG_ASSET_NAME: str = None
    PEG_TYPE: str = "part"
    PEG_ATTACHMENT_PRIM: str = None

    HOLE_ASSET_NAME: str = None
    HOLE_TYPE: str = "part"
    HOLE_ATTACHMENT_PRIM: str = None

    # Add parts to the scene that aren't being manipulated and aren't part of the goal calculation
    EXTRA_PARTS: List = field(default_factory=list)
    EXTRA_PARTS_STARTING_POSE: List = field(default_factory=list)
    EXTRA_ATTACHMENTS: List = field(default_factory=list)

    HOLE_T_PEG_GOAL: Tuple = None
    PEG_GOAL: Tuple = ([0,0,0], [0, 0, 0, 1])
    HOLE_GOAL: Tuple = ([0,0,0], [0, 0, 0, 1])

    # These are used if the relevant part of the peg/hole is separate from the. 
    # The target could be defined by pose, but this can be used if the pose doesn't matter much
    # In most cases, these can be kept at identity
    PEG_T_ORIGIN_GOAL: Tuple = ([0,0,0], [0,0,0,1])
    HOLE_T_ORIGIN_GOAL: Tuple = ([0,0,0], [0,0,0,1])

    WORLD_T_PEG_START: Tuple = None
    WORLD_T_HOLE_START: Tuple = None
    WORLD_T_FIXTURE_START: Tuple = None
    WORLD_T_ZIVID_HOME: Tuple = WORLD_T_ZIVID_HOME

    PEG_T_TIP: Tuple = None
    HOLE_T_TIP: Tuple = None

    WORLD_T_PEG_PICK: Tuple = None
    WORLD_T_HOLE_PICK: Tuple = None

    relative_goal: bool = True
    num_robots: int = 2
    relative_pose_obs: bool = False
    desired_tool_T_world: Tuple = None

    allow_peg_rotation: bool = False
    allow_hole_rotation: bool = False

    # Weights on the x, y, z, rx, ry, rz components of the error
    # peg_goal_weights: List[float] = field(default_factory=lambda: [1, 1, 1, 0, 0, 0])
    peg_goal_weights: List[float] = field(default_factory=lambda: [1, 1, 1, 0.1, 0.1, 0.1])
    # Only used if absolute pose and hole part
    # hole_goal_weights: List[float] = field(default_factory=lambda: [1, 1, 1, 0, 0, 0])
    hole_goal_weights: List[float] = field(default_factory=lambda: [1, 1, 1, 0.1, 0.1, 0.1])

    PEG_IK_WORLD_T_TOOL: Tuple = None
    HOLE_IK_WORLD_T_TOOL: Tuple = None
    origin_regularization: float = 0
    # add gripper status into action space
    allow_gripper_status: bool = False  # True

    relative_observations: bool = False
    force_sensing: bool = False

    USE_FIXED_RIGID_ATTACHMENT: bool = False
    FT_SENSOR_TYPE: str = 'physx_view'
    USE_FT_SENSOR: bool = True
    JOINT_VELOCITY: int = 48
    CAMERA_TYPE: str = 'tiled'  # 'regular'
    TASK_NAME: str = None
    SHOW_GOAL: bool = False
    WORLD_T_PEG_START_LIST: List = None
    VELOCITY_OBS: bool = False # True # False
    # add gripper status into observation space
    GRIPPER_STATUS_OBS: bool = False
    DIFFICULTY_LEVEL: str = 'easy'  # 'easy', 'medium'
    SUBTASK: str = None
    # image as observation
    USE_CAMERA: bool = True
    IMAGE_WIDTH: int = 128  # 60
    IMAGE_HEIGHT: int = 128
    IMAGE_WIDTH_RAW: int = 512
    IMAGE_HEIGHT_RAW: int = 512
    # save images for debugging
    SAVE_IMG: bool = False # True
    better_tool_frame: bool = False
    RESET_POSITION_NOISE: float = 0.005 # 0.0025
    RESET_ROTATION_NOISE: float = 0.02

    DEBUG_ACTION: bool = False
    selected_obj_id: int = 0
    POSE_AFTER: Tuple = None

    @property
    def PEG_T_TOOL(self):
        if self.better_tool_frame:
            return multiply(multiply(self.PEG_T_TIP, invert(TOOL_T_TIP)),
                ([0, 0, 0], [0, 0, 0.7071068, 0.7071068]))
        else:
            return multiply(self.PEG_T_TIP, invert(TOOL_T_TIP))
    
    @property
    def HOLE_T_TOOL(self):
        if self.better_tool_frame:
            return multiply(multiply(self.HOLE_T_TIP, invert(TOOL_T_TIP)),
                ([0, 0, 0], [0, 0, 0.7071068, 0.7071068]))
        else:
            return multiply(self.HOLE_T_TIP, invert(TOOL_T_TIP))

    @property
    def PEG_T_FLANGE(self):
        if self.better_tool_frame:
            return multiply(multiply(self.PEG_T_TOOL, invert(FLANGE_T_TOOL)),
                ([0, 0, 0], [0, 0, 0.7071068, 0.7071068]))
        else:
            return multiply(self.PEG_T_TOOL, invert(FLANGE_T_TOOL))
    
    @property
    def HOLE_T_FLANGE(self):
        if self.better_tool_frame:
            return multiply(multiply(self.HOLE_T_TOOL, invert(FLANGE_T_TOOL)),
                ([0, 0, 0], [0, 0, 0.7071068, 0.7071068]))
        else:
            return multiply(self.HOLE_T_TOOL, invert(FLANGE_T_TOOL))
        
    @property
    def WORLD_T_TOOL_PICK_PEG(self):
        if self.WORLD_T_PEG_PICK is None:
            return None
        return pbu.multiply(pbu.multiply(self.WORLD_T_PEG_PICK, self.PEG_T_TIP), pbu.invert(TOOL_T_TIP))

    @property
    def WORLD_T_TOOL_PICK_HOLE(self):
        if (self.WORLD_T_HOLE_PICK is None):
            return None
        return pbu.multiply(pbu.multiply(self.WORLD_T_HOLE_PICK, self.HOLE_T_TIP), pbu.invert(TOOL_T_TIP))

    @property
    def WORLD_T_PEG_TOOL_START(self):
        return pbu.multiply(self.WORLD_T_PEG_START, self.PEG_T_TOOL)

    @property
    def WORLD_T_HOLE_TOOL_START(self):
        return pbu.multiply(self.WORLD_T_HOLE_START, self.HOLE_T_TOOL)

    @property
    def holding_peg(self):
        return self.PEG_T_TIP is not None

    @property
    def holding_hole(self):
        return self.HOLE_T_TIP is not None

    @property
    def robot_count(self):
        return int(self.holding_peg) + int(self.holding_hole)


def fmb_example(selected_obj_id=0, desired_tool_T_world=None, ):
    PEG_T_TIP = ([0,0,0.02113], [-1, 0, 0, 0])
    HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0,0,0,1])

    # WORLD_T_HOLE_START = ([0, 0, 0.025], [0, 0, 0, 1])
    WORLD_T_PEG_START = multiply(([0, 0, 0.1], [0,0,0,1]), multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))

    return TaskConfig(PEG_ASSET_NAME = home_dir + "taskboard/fmb_example/Medium_Short_Hexagon_Green_Updated.usd",
                      PEG_ATTACHMENT_PRIM = "peg/Medium_Short_Hexagon_Green/Medium_Short_Hexagon_Green/obj1_012_obj1_005",
                      HOLE_ASSET_NAME = home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd",],
                      EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE],
                      allow_peg_rotation=False, # True,
                      allow_gripper_status=False,
                      USE_FT_SENSOR=False,
                      USE_CAMERA=False,
                      )

def AssemblyMove(SUBTASK='MoveToBoardFromGrasp', SHOW_GOAL=False, better_tool_frame=True, selected_obj_id=0, obj_style='vertical', desired_tool_T_world=None, ):
    # 'MoveToBoardFromFixture', 'MoveToBoardFromGrasp'  # 'MoveToFixture'
    # PEG_T_TIP = ([0,0,0.02113], [-1, 0, 0, 0])
    # PEG_T_TIP = ([0, 0, -0.015], [-1, 0, 0, 0])
    # PEG_T_TIP = ([0, 0, 0.0], [-1, 0, 0, 0])
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses) = split_obj_dict(
        OBJECT_DICT, selected_obj_id, obj_style=obj_style)

    if target_obj_name in ['Medium_Short_SquareCircle_Red', 'Medium_Short_3Prong_JeansRed', 'Medium_Short_DoubleSquare_Purple']:
        target_obj_goal_pose = multiply(target_obj_goal_pose, ([0., 0., 0.0], [0, 1, 0, 0]))
        PEG_T_TIP = multiply(multiply(PEG_T_TIP_DEFAULT, ([0., 0., 0.0], [0, 1, 0, 0])), ([0., 0., 0.0], [0, 0, 1, 0]))
    elif target_obj_name in ['Medium_Short_Rectangle_JeansBlue']:
        PEG_T_TIP = multiply(PEG_T_TIP_DEFAULT, ([0., 0., 0.0], [0, 0, 1, 0]))
        
    else:
        PEG_T_TIP = PEG_T_TIP_DEFAULT

    if SUBTASK == 'MoveToFixture':
        # HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0,0,0,1])

        # # WORLD_T_HOLE_START = ([0, 0.4, 0.025], [0, 0, 0, 1])
        # WORLD_T_PEG_START = multiply(([0, 0, 0.1], [0,0,0,1]), multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))
        # if SHOW_GOAL:
        #     WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)
        # FIXTURE_T_PEG_START = ([0.2, 0.28, 0.16], [0, 0, 0, 1])
        # WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_START)
        WORLD_T_PEG_START = ([0.6, 0.2, 0.15], [0.0, 0.0, -0.7071067690849304, 0.7071067690849304])  # (0, 0, -90)
        # ((0.2800000011920929, 0.6000000238418579, 0.2199999988079071), (0.0, 0.0, -0.7071067690849304, 0.7071067690849304))
        # WORLD_T_PEG_START = ([0.3, 0.6, 0.2], [0, 0, 0, 1])
        FIXTURE_T_PEG_GOAL = ([0.0, 0.0, 0.16], [0, 0, 0, 1])
        if SHOW_GOAL:
            WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_GOAL)
        return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                          PEG_ATTACHMENT_PRIM=target_peg_attachment_prim,
                          HOLE_ASSET_NAME=home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                          HOLE_TYPE = "peripheral",
                          PEG_GOAL = FIXTURE_T_PEG_GOAL,
                          WORLD_T_PEG_START=WORLD_T_PEG_START,
                          WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                          PEG_T_TIP = PEG_T_TIP,
                          num_robots = 1,
                          EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd",] + extra_obj_usd_paths,
                          EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE, ] + extra_obj_start_poses,
                          allow_peg_rotation=True,
                          USE_FIXED_RIGID_ATTACHMENT=True,
                          allow_gripper_status=False,
                          TASK_NAME='AssemblyMove-v0',
                          SHOW_GOAL=SHOW_GOAL,
                          WORLD_T_PEG_START_LIST=[WORLD_T_PEG_START, ([0.3, 0.6, 0.1+0.18], [0.5000, -0.5000,  0.5000,  0.5000])],
                          WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                          SUBTASK=SUBTASK,
                          better_tool_frame=better_tool_frame,
                          DIFFICULTY_LEVEL='easy',
                          selected_obj_id=selected_obj_id,
                          )
    elif SUBTASK == 'MoveToFixtureHorizontal':
        # HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0,0,0,1])

        # # WORLD_T_HOLE_START = ([0, 0.4, 0.025], [0, 0, 0, 1])
        # WORLD_T_PEG_START = multiply(([0, 0, 0.1], [0,0,0,1]), multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))
        # if SHOW_GOAL:
        #     WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)
        # FIXTURE_T_PEG_START = ([0.2, 0.28, 0.16], [0, 0, 0, 1])
        # WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_START)
        PEG_T_TIP_HORIZONTAL = multiply(multiply(([0, 0.05, 0.04], [-0.7071068, 0, 0, 0.7071068]),
                                        ([0, 0, 0], [0, 0, -0.7071068, 0.7071068 ])), ([0, 0, 0], [0, 0, 1, 0]))

        # WORLD_T_PEG_START = ([0.6, 0.2, 0.15], [0.0, 0.0, -0.7071067690849304, 0.7071067690849304])
        # ((0.2800000011920929, 0.6000000238418579, 0.2199999988079071), (0.0, 0.0, -0.7071067690849304, 0.7071067690849304))
        # WORLD_T_PEG_START = ([0.3, 0.6, 0.2], [0, 0, 0, 1])
        # FIXTURE_T_PEG_GOAL = ([0.0, 0.0, 0.16], [0, 0, 0, 1])
        WORLD_T_PEG_START = ([0.6, 0.05, 0.15], [0.7071068, 0, 0, -0.7071068])  # (270, 0, 0)
        FIXTURE_T_PEG_GOAL = multiply(([0.0, 0.0, 0.16], [0, 0.7071068, 0, -0.7071068]), ([0, 0, 0], [0, 0, 0.7071068, 0.7071068]))  # (0, 270, 0)
        if SHOW_GOAL:
            WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_GOAL)
        return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                          PEG_ATTACHMENT_PRIM=target_peg_attachment_prim,
                          HOLE_ASSET_NAME=home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                          HOLE_TYPE = "peripheral",
                          PEG_GOAL = FIXTURE_T_PEG_GOAL,
                          WORLD_T_PEG_START=WORLD_T_PEG_START,
                          WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                          PEG_T_TIP=PEG_T_TIP_HORIZONTAL,
                          num_robots = 1,
                          EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd",] + extra_obj_usd_paths,
                          EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE, ] + extra_obj_start_poses,
                          allow_peg_rotation=True,
                          USE_FIXED_RIGID_ATTACHMENT=True,
                          allow_gripper_status=False,
                          TASK_NAME='AssemblyMove-v0',
                          SHOW_GOAL=SHOW_GOAL,
                          WORLD_T_PEG_START_LIST=[WORLD_T_PEG_START, ([0.3, 0.6, 0.1+0.18], [0.5000, -0.5000,  0.5000,  0.5000])],
                          WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                          SUBTASK=SUBTASK,
                          better_tool_frame=better_tool_frame,
                          DIFFICULTY_LEVEL='easy',
                          selected_obj_id=selected_obj_id,
                          )
    elif SUBTASK == 'MoveToBoardFromFixture':
        # HOLE_T_PEG_GOAL =  ([-0.0645, -0.087, 0.03], [0, 0, 1, 0]) # Good for board before rotate 180 deg
        # HOLE_T_PEG_GOAL = ([-0.070, -0.089, 0.03], [0,0,1,0])  # Good for board after rotate 180 deg
        HOLE_T_PEG_GOAL = target_obj_goal_pose
        HOLE_T_PEG_GOAL = multiply(([0, 0, 0.06], [0, 0, 0, 1]), HOLE_T_PEG_GOAL)
        HOLE_T_PEG_GOAL = multiply(HOLE_T_PEG_GOAL, HOLE_ROTATE)
        
        FIXTURE_T_PEG_GOAL = ([0.0, 0.0, 0.16], [0, 0, 1, 0])
        WORLD_T_PEG_START = multiply(multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_GOAL), ([0, 0, 0], [0, 0, 0.7071068, 0.7071068]))

        if SHOW_GOAL:
            WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)

        return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                          PEG_ATTACHMENT_PRIM=target_peg_attachment_prim,
                          HOLE_ASSET_NAME=home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                          HOLE_TYPE = "peripheral",
                          PEG_GOAL = HOLE_T_PEG_GOAL,
                          WORLD_T_PEG_START=WORLD_T_PEG_START,
                          WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                          PEG_T_TIP = PEG_T_TIP,  # multiply(PEG_T_TIP, ([0, 0, 0], [0, 0, 0.7071068, 0.7071068]),
                          num_robots = 1,
                          EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd",] + extra_obj_usd_paths,
                          EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE, ] + extra_obj_start_poses,
                          allow_peg_rotation=True,
                          USE_FIXED_RIGID_ATTACHMENT=True,
                          allow_gripper_status=False,
                          TASK_NAME='AssemblyMove-v0',
                          SHOW_GOAL=SHOW_GOAL,
                          WORLD_T_PEG_START_LIST=[WORLD_T_PEG_START, ([0.3, 0.6, 0.1+0.18], [0.5000, -0.5000,  0.5000,  0.5000])],
                          SUBTASK=SUBTASK,
                          DIFFICULTY_LEVEL='easy',
                          selected_obj_id=selected_obj_id,
                          POSE_AFTER=SAFE_POSE_AFTER['AssemblyMoveToBoardFromFixture'],
                          )
    elif SUBTASK == 'MoveToBoardFromGrasp':
        # HOLE_T_PEG_GOAL =  ([-0.0645, -0.087, 0.03], [0, 0, 1, 0])   # Good for board before rotate 180 deg
        # HOLE_T_PEG_GOAL = ([-0.070, -0.089, 0.03], [0,0,1,0])  # Good for board after rotate 180 deg
        HOLE_T_PEG_GOAL = target_obj_goal_pose
        HOLE_T_PEG_GOAL = multiply(([0, 0, 0.06], [0, 0, 0, 1]), HOLE_T_PEG_GOAL)
        HOLE_T_PEG_GOAL = multiply(HOLE_T_PEG_GOAL, HOLE_ROTATE)

        WORLD_T_PEG_START = ([0.6, 0.2, 0.15], [0, 0, 1, 0])
        if SHOW_GOAL:
            WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)

        return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                          PEG_ATTACHMENT_PRIM=target_peg_attachment_prim,
                          HOLE_ASSET_NAME=home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                          HOLE_TYPE = "peripheral",
                          PEG_GOAL = HOLE_T_PEG_GOAL,
                          WORLD_T_PEG_START=WORLD_T_PEG_START,
                          WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                          PEG_T_TIP = PEG_T_TIP,  # multiply(PEG_T_TIP, ([0, 0, 0], [0, 0, 0.7071068, 0.7071068]),
                          num_robots = 1,
                          EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd",] + extra_obj_usd_paths,
                          EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE, ] + extra_obj_start_poses,
                          allow_peg_rotation=True,
                          USE_FIXED_RIGID_ATTACHMENT=True,
                          allow_gripper_status=False,
                          TASK_NAME='AssemblyMove-v0',
                          SHOW_GOAL=SHOW_GOAL,
                          WORLD_T_PEG_START_LIST=[WORLD_T_PEG_START, ([0.3, 0.6, 0.1+0.18], [0.5000, -0.5000,  0.5000,  0.5000])],
                          SUBTASK=SUBTASK,
                          DIFFICULTY_LEVEL='easy',
                          selected_obj_id=selected_obj_id,
                          POSE_AFTER=SAFE_POSE_AFTER['AssemblyMoveToBoardFromGrasp'],
                          )
    elif SUBTASK == 'Reset':
        # HOLE_T_PEG_GOAL =  ([-0.0645, -0.087, 0.03], [0, 0, 1, 0])   # Good for board before rotate 180 deg
        # HOLE_T_PEG_GOAL = ([-0.070, -0.089, 0.03], [0,0,1,0])  # Good for board after rotate 180 deg
        HOLE_T_PEG_GOAL = target_obj_goal_pose
        HOLE_T_PEG_GOAL = multiply(([0, 0, 0.06], [0, 0, 0, 1]), HOLE_T_PEG_GOAL)
        HOLE_T_PEG_GOAL = multiply(HOLE_T_PEG_GOAL, HOLE_ROTATE)

        WORLD_T_PEG_START = ([0.6, 0.2, 0.15], [0, 0, 1, 0])
        if SHOW_GOAL:
            WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)

        return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                          PEG_ATTACHMENT_PRIM=target_peg_attachment_prim,
                          HOLE_ASSET_NAME=home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                          HOLE_TYPE = "peripheral",
                          PEG_GOAL = HOLE_T_PEG_GOAL,
                          WORLD_T_PEG_START=WORLD_T_PEG_START,
                          WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                          PEG_T_TIP = PEG_T_TIP,  # multiply(PEG_T_TIP, ([0, 0, 0], [0, 0, 0.7071068, 0.7071068]),
                          num_robots = 1,
                          EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd",] + extra_obj_usd_paths,
                          EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE, ] + extra_obj_start_poses,
                          allow_peg_rotation=True,
                          USE_FIXED_RIGID_ATTACHMENT=True,
                          allow_gripper_status=False,
                          TASK_NAME='AssemblyReset-v0',
                          SHOW_GOAL=SHOW_GOAL,
                          WORLD_T_PEG_START_LIST=[WORLD_T_PEG_START, ([0.3, 0.6, 0.1+0.18], [0.5000, -0.5000,  0.5000,  0.5000])],
                          SUBTASK=SUBTASK,
                          DIFFICULTY_LEVEL='easy',
                          selected_obj_id=selected_obj_id,
                          POSE_AFTER=SAFE_POSE_AFTER['AssemblyReset'],
                          )

def AssemblyMoveToFixture(SUBTASK='MoveToFixture', SHOW_GOAL=False, better_tool_frame=True, selected_obj_id=0, obj_style='vertical', desired_tool_T_world=None, ):
    return AssemblyMove(SUBTASK, SHOW_GOAL=SHOW_GOAL, better_tool_frame=better_tool_frame, selected_obj_id=selected_obj_id, obj_style=obj_style, desired_tool_T_world=desired_tool_T_world)

def AssemblyMoveToFixtureHorizontal(SUBTASK='MoveToFixtureHorizontal', SHOW_GOAL=False, better_tool_frame=True, selected_obj_id=0, obj_style='horizontal', desired_tool_T_world=None, ):
    return AssemblyMove(SUBTASK, SHOW_GOAL=SHOW_GOAL, better_tool_frame=better_tool_frame, selected_obj_id=selected_obj_id, obj_style=obj_style, desired_tool_T_world=desired_tool_T_world)

def AssemblyMoveToBoardFromFixture(SUBTASK='MoveToBoardFromFixture', SHOW_GOAL=False, better_tool_frame=False, selected_obj_id=0, obj_style='horizontal', desired_tool_T_world=None, ):
    return AssemblyMove(SUBTASK, SHOW_GOAL=SHOW_GOAL, better_tool_frame=better_tool_frame, selected_obj_id=selected_obj_id, obj_style=obj_style, desired_tool_T_world=desired_tool_T_world)

def AssemblyMoveToBoardFromGrasp(SUBTASK='MoveToBoardFromGrasp', SHOW_GOAL=False, better_tool_frame=False, selected_obj_id=0, obj_style='vertical', desired_tool_T_world=None, ):
    return AssemblyMove(SUBTASK, SHOW_GOAL=SHOW_GOAL, better_tool_frame=better_tool_frame, selected_obj_id=selected_obj_id, obj_style=obj_style, desired_tool_T_world=desired_tool_T_world)

def AssemblyReset(SUBTASK='Reset', SHOW_GOAL=False, better_tool_frame=False, selected_obj_id=0, obj_style='vertical', desired_tool_T_world=None, ):
    return AssemblyMove(SUBTASK, SHOW_GOAL=SHOW_GOAL, better_tool_frame=better_tool_frame, selected_obj_id=selected_obj_id, obj_style=obj_style, desired_tool_T_world=desired_tool_T_world)

def AssemblyInsert(selected_obj_id=0, desired_tool_T_world=None):
    SHOW_GOAL = False  # True
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses) = split_obj_dict(
        OBJECT_DICT, selected_obj_id, )

    version = 'v1'
    if target_obj_name in ['Medium_Short_SquareCircle_Red', 'Medium_Short_3Prong_JeansRed', 'Medium_Short_DoubleSquare_Purple']:
        target_obj_goal_pose = multiply(target_obj_goal_pose, ([0., 0., 0.0], [0, 1, 0, 0]))
        PEG_T_TIP = multiply(multiply(PEG_T_TIP_DEFAULT, ([0., 0., 0.0], [0, 1, 0, 0])), ([0., 0., 0.0], [0, 0, 1, 0]))
    elif target_obj_name in ['Medium_Short_Rectangle_JeansBlue']:
        PEG_T_TIP = multiply(PEG_T_TIP_DEFAULT, ([0., 0., 0.0], [0, 0, 1, 0]))
        
    else:
        PEG_T_TIP = PEG_T_TIP_DEFAULT
    # PEG_T_TIP = ([0,0,0.02113], [-1, 0, 0, 0])
    # PEG_T_TIP = ([0, 0, 0.0], [-1, 0, 0, 0])
    # HOLE_T_PEG_GOAL =  ([-0.0645, -0.087, 0.03], [0, 0, 1, 0]) # Good for board before rotate 180 deg
    HOLE_T_PEG_GOAL = target_obj_goal_pose  # ([-0.070, -0.089, 0.03], [0,0,1,0]) # # Good for board after rotate 180 deg
    HOLE_T_PEG_GOAL = multiply(HOLE_T_PEG_GOAL, HOLE_ROTATE)
    # HOLE_T_PEG_GOAL = multiply(([0, 0, 0], [0, 0, 1, 0]), HOLE_T_PEG_GOAL)
    if version == 'v1': # peg start on the board
        WORLD_T_PEG_START = multiply(([0, 0, 0.06], [0,0,0,1]), multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))
        # WORLD_T_PEG_START = multiply(([0, 0, 0.05], [0,0,0,1]), multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))

    elif version == 'v2':  # peg start on the fixture
        WORLD_T_PEG_START = multiply(([0.0, 0.0, 0.16], [0, 0, 0, 1]), WORLD_T_FIXTURE_POSE)

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)

    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim, # "peg/Medium_Short_Hexagon_Green/Medium_Short_Hexagon_Green/obj1_012_obj1_005",
                      HOLE_ASSET_NAME=home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd", ] + extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE, ] + extra_obj_start_poses,
                      allow_peg_rotation=True,
                      USE_FIXED_RIGID_ATTACHMENT=True,
                      allow_gripper_status=False,
                      TASK_NAME='AssemblyInsert-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      WORLD_T_PEG_START_LIST=[WORLD_T_PEG_START, ([0.3, 0.6, 0.1+0.18], [0.5000, -0.5000,  0.5000,  0.5000])],
                      USE_FT_SENSOR=True,
                      USE_CAMERA=False, # True,
                      # peg_goal_weights=[5, 5, 5, 0.1, 0.1, 0.1],
                      peg_goal_weights=[1, 1, 1, 1, 1, 1],
                      DIFFICULTY_LEVEL='easy',  # 'medium',
                      SAVE_IMG=False,
                      relative_pose_obs=True,
                      selected_obj_id=selected_obj_id,
                      POSE_AFTER=SAFE_POSE_AFTER['AssemblyInsert'],
                      )

def AssemblyScriptedInsert(selected_obj_id=8, desired_tool_T_world=None):
    SHOW_GOAL = False # True
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses) = split_obj_dict(
        OBJECT_DICT, selected_obj_id, )
    version = 'v1'
    # PEG_T_TIP = ([0,0,0.02113], [-1, 0, 0, 0])
    # PEG_T_TIP = ([0, 0, 0.0], [-1, 0, 0, 0])
    # HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0, 0, 0, 1])
    # HOLE_T_PEG_GOAL = multiply(([0, 0, 0], [0,0,1,0]), ([-0.06795, -0.086, 0.02422], [0, 0, 0, 1]))
    # HOLE_T_PEG_GOAL = ([-0.06795, -0.086, 0.02422], [0, 0, 0, 1])
    # HOLE_T_PEG_GOAL =  ([-0.06795, -0.08966, 0.02422], [0, 0, 1, 0])
    # HOLE_T_PEG_GOAL =  ([-0.0645, -0.087, 0.03], [0, 0, 1, 0])  # Good for board before rotate 180 deg
    if target_obj_name in ['Medium_Short_SquareCircle_Red', 'Medium_Short_3Prong_JeansRed', 'Medium_Short_DoubleSquare_Purple']:
        target_obj_goal_pose = multiply(target_obj_goal_pose, ([0., 0., 0.0], [0, 1, 0, 0]))
        PEG_T_TIP = multiply(multiply(PEG_T_TIP_DEFAULT, ([0., 0., 0.0], [0, 1, 0, 0])), ([0., 0., 0.0], [0, 0, 1, 0]))
    elif target_obj_name in ['Medium_Short_Rectangle_JeansBlue']:
        PEG_T_TIP = multiply(PEG_T_TIP_DEFAULT, ([0., 0., 0.0], [0, 0, 1, 0]))
        
    else:
        PEG_T_TIP = PEG_T_TIP_DEFAULT
    HOLE_T_PEG_GOAL = target_obj_goal_pose  #  ([-0.070, -0.089, 0.03], [0,0,1,0]) # # Good for board after rotate 180 deg
    HOLE_T_PEG_GOAL = multiply(HOLE_T_PEG_GOAL, HOLE_ROTATE)
    # HOLE_T_PEG_GOAL = multiply(([0, 0, 0], [0, 0, 1, 0]), HOLE_T_PEG_GOAL)
    if version == 'v1': # peg start on the board
        WORLD_T_PEG_START = multiply(([0, 0, 0.06], [0,0,0,1]), multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))
        # WORLD_T_PEG_START = multiply(([0, 0, 0.05], [0,0,0,1]), multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))

    elif version == 'v2':  # peg start on the fixture
        WORLD_T_PEG_START = multiply(([0.0, 0.0, 0.16], [0, 0, 0, 1]), WORLD_T_FIXTURE_POSE)

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)

    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME=home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd",] + extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE,] + extra_obj_start_poses,
                      allow_peg_rotation=True,
                      USE_FIXED_RIGID_ATTACHMENT=True,
                      allow_gripper_status=False,
                      TASK_NAME='AssemblyScriptedInsert-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      WORLD_T_PEG_START_LIST=[WORLD_T_PEG_START, ([0.3, 0.6, 0.1+0.18], [0.5000, -0.5000,  0.5000,  0.5000])],
                      USE_FT_SENSOR=True,
                      USE_CAMERA=False,
                      # peg_goal_weights=[5, 5, 5, 0.1, 0.1, 0.1],
                      peg_goal_weights=[1, 1, 1, 1, 1, 1],
                      DIFFICULTY_LEVEL='easy',
                      relative_pose_obs=True,
                      selected_obj_id=selected_obj_id,
                      POSE_AFTER=SAFE_POSE_AFTER['AssemblyScriptedInsert'],
                      )

# BEAM_CENTER = ([0.85, 0.4, 0.0], [0, 0, 0, 1])
BEAM_CENTER =  ([0.3, 0.2, 0.0], [0, 0, 0, 1])

# QIANZHONG tuned
# BEAM_WORLD_T_FIXTURE_POSE BEAM_WORLD_T_HOLE_START
BEAM_OBJECT_DICT = {
    # red
    '0': {'start_pose': multiply(BEAM_CENTER, [[0.0885, 0.0, 0.032999999225139618], [0.0, 0.0, 0.0, 1.0]]), #([0.3895, 0.20, 0.018], [0, 0, 0, 1]), # peg
          'usd_path': home_dir + "taskboard/beam/0_Updated.usd",
          'goal_pose': multiply(BEAM_CENTER, [[0.0885, 0.0, 0.005999999225139618], [0.0, 0.0, 0.0, 1.0]]), # ([0.3895, 0.20, 0.018], [0, 0, 0, 1]),
          'peg_attachment_prim': "peg/_/node_/mesh"}, 
    # yellow
    '1': {'start_pose': multiply(BEAM_CENTER, ([-0.0885, 0.0, 0.030999999225139618], [0.0, 0.0, 0, 1.0,])), # ([0.30149999260902405, 0.20000000298023224, 0.017999999225139618], [0.0, 0.0, 1.0, 0.0]), # hole
          'usd_path': home_dir + "taskboard/beam/1_Updated.usd",
          'goal_pose': multiply(BEAM_CENTER, ([-0.0885, 0.0, 0.005999999225139618], [0.0, 0.0, 0, 1.0, ])), # ([0.30149999260902405, 0.20000000298023224, 0.017999999225139618], [0.0, 0.0, 1.0, 0.0]),
          'peg_attachment_prim': "peg/_/node_/mesh"},
    # extras green
    '2':  {'start_pose': multiply(BEAM_CENTER, ([0.0, 0., 0.03], [0, 0, 0, 1])), # green, extra_1 
        'usd_path': home_dir + "taskboard/beam/2_Updated.usd",
        'goal_pose': multiply(BEAM_CENTER, ([0.0, 0., 0.005], [0, 0, 0, 1])),
        'peg_attachment_prim': "peg/_/node_/mesh"},
    '3':  {'start_pose': multiply(BEAM_CENTER, ([0.0, 0., 0.03], [0, 0, 0, 1])), # blue extra_2
           'usd_path': home_dir + "taskboard/beam/3_Updated.usd",
           'goal_pose': multiply(BEAM_CENTER, ([0.0, 0., 0.005], [0, 0, 0, 1])),
           'peg_attachment_prim': "peg/_/node_/mesh"},
    '4': {'start_pose': multiply(BEAM_CENTER, ([0.0, 0.0, 0.01], [0, 0, 0, 1])), # purple extra
          'usd_path': home_dir + "taskboard/beam/4_Updated.usd",
          'goal_pose': multiply(BEAM_CENTER, ([0.0, 0.0, 0.01], [0, 0, 0, 1])),
          'peg_attachment_prim': "peg/_/node_/mesh"},
}

# QIANZHONG tuned
def BeamScriptedInsert(selected_obj_id=0, desired_tool_T_world=None):
    selected_obj_id = 2 # 1 # 3
    SHOW_GOAL = False # True
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses,
     hole_obj_name, hole_obj_usd_path, hole_obj_start_pose, hole_obj_goal_pose, hole_peg_attachment_prim,) = beam_split_obj_dict(
        BEAM_OBJECT_DICT, selected_obj_id=selected_obj_id, )
    if selected_obj_id == 1:
        BEAM_PEG_T_TIP_DEFAULT = ([0.046, 0, 0.03], [-1, 0, 0, 0])

        extra_obj_start_poses = [multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]), 
                                multiply(BEAM_CENTER, [[-0.0885, -0.1, 0.010999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
                                extra_obj_start_poses[2]]
    elif selected_obj_id == 2:
        BEAM_PEG_T_TIP_DEFAULT = ([-0.043, 0, 0.01], [-1, 0, 0, 0])

        extra_obj_start_poses = [multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]), 
                        multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]),
                        multiply(BEAM_CENTER, [[-0.0885, -0.1, 0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])]
        # Modify the first element of the first tuple in the first sublist
        # ic(1, extra_obj_start_poses[1])
        # extra_obj_start_poses[1] = list(extra_obj_start_poses[1])  # Convert tuple to list for mutability
        # extra_obj_start_poses[1][0] = list(extra_obj_start_poses[1][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[1][0][0] -= 0.1  
        # extra_obj_start_poses[1][0][2] -= 0.028
        # extra_obj_start_poses[1][0] = tuple(extra_obj_start_poses[1][0])  # Convert back to tuple
        # extra_obj_start_poses[1] = tuple(extra_obj_start_poses[1])  # Convert back to tuple
        # ic(2, extra_obj_start_poses[1])
    elif selected_obj_id == 3:
        BEAM_PEG_T_TIP_DEFAULT = ([0.046, 0, 0.01], [-1, 0, 0, 0])
        # Modify the first element of the first tuple in the first sublist
        ic(1, extra_obj_start_poses[0])
        extra_obj_start_poses = [multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]), 
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2]]
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] -= 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        ic(2, extra_obj_start_poses[0])
    else: # selected_obj_id == 0:
        BEAM_PEG_T_TIP_DEFAULT = ([-0.043, 0, 0.03], [-1, 0, 0, 0])
    print(selected_obj_id)
    # version = 'v1'
    # if target_obj_name in ['Medium_Short_SquareCircle_Red', 'Medium_Short_3Prong_JeansRed', 'Medium_Short_DoubleSquare_Purple']:
    #     target_obj_goal_pose = multiply(target_obj_goal_pose, ([0., 0., 0.0], [0, 1, 0, 0]))
    #     PEG_T_TIP = multiply(multiply(PEG_T_TIP_DEFAULT, ([0., 0., 0.0], [0, 1, 0, 0])), ([0., 0., 0.0], [0, 0, 1, 0]))
    # elif target_obj_name in ['Medium_Short_Rectangle_JeansBlue']:
    #     PEG_T_TIP = multiply(PEG_T_TIP_DEFAULT, ([0., 0., 0.0], [0, 0, 1, 0]))
    # else:
    # print(BEAM_OBJECT_DICT)
    PEG_T_TIP = BEAM_PEG_T_TIP_DEFAULT
    # hole_obj_goal_pose = ([-0.4488, -0.0189+0.1, 0.018], [0,0,1,0])
    # HOLE_T_PEG_GOAL = multiply(target_obj_goal_pose, invert(hole_obj_goal_pose))  #  ([-0.070, -0.089, 0.03], [0,0,1,0]) # # Good for board after rotate 180 deg
    HOLE_T_PEG_GOAL = multiply(invert(hole_obj_goal_pose), target_obj_goal_pose)
    # new_target_obj_goal_pose = multiply(hole_obj_goal_pose, HOLE_T_PEG_GOAL)
    # print(target_obj_goal_pose, new_target_obj_goal_pose)
    # HOLE_T_PEG_GOAL = multiply(HOLE_T_PEG_GOAL, HOLE_ROTATE)
    # HOLE_T_PEG_GOAL = multiply(([0, 0, 0], [0, 0, 1, 0]), HOLE_T_PEG_GOAL)
    # if version == 'v1': # peg start on the board
    #     WORLD_T_PEG_START = multiply(([0, 0, 0.06], [0,0,0,1]), multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))
        # WORLD_T_PEG_START = multiply(([0, 0, 0.05], [0,0,0,1]), multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))

    # elif version == 'v2':  # peg start on the fixture
    WORLD_T_PEG_START = target_obj_start_pose  # multiply(([0.0, 0.0, 0.10], [0, 0, 0, 1]), BEAM_WORLD_T_FIXTURE_POSE)
    ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, target_obj_usd_path, hole_obj_usd_path)

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)
    # print('\n'*50, '************************', 'hole_obj_goal_pose: ', hole_obj_goal_pose, '************************', '\n'*50)
    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME=hole_obj_usd_path,
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=hole_obj_goal_pose,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=extra_obj_start_poses,
                      allow_peg_rotation=True,
                      USE_FIXED_RIGID_ATTACHMENT=True,
                      allow_gripper_status=False,
                      TASK_NAME='BeamScriptedInsert-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      WORLD_T_PEG_START_LIST=[WORLD_T_PEG_START, ([0.3, 0.6, 0.1+0.18], [0.5000, -0.5000,  0.5000,  0.5000])],
                      USE_FT_SENSOR=True,
                      USE_CAMERA=False,
                      # peg_goal_weights=[5, 5, 5, 0.1, 0.1, 0.1],
                      peg_goal_weights=[1, 1, 1, 1, 1, 1],
                      DIFFICULTY_LEVEL='easy',
                      relative_pose_obs=True,
                      selected_obj_id=selected_obj_id,
                      POSE_AFTER=BEAM_SAFE_POSE_AFTER['BeamScriptedInsert'],
                      )

def BeamInsert(selected_obj_id=0, desired_tool_T_world=None):
    selected_obj_id = 2 # 1 # 3
    SHOW_GOAL = False # True
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses,
     hole_obj_name, hole_obj_usd_path, hole_obj_start_pose, hole_obj_goal_pose, hole_peg_attachment_prim,) = beam_split_obj_dict(
        BEAM_OBJECT_DICT, selected_obj_id=selected_obj_id, )
    if selected_obj_id == 1:
        BEAM_PEG_T_TIP_DEFAULT = ([0.046, 0, 0.03], [-1, 0, 0, 0])

        extra_obj_start_poses = [multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]), 
                                multiply(BEAM_CENTER, [[-0.0885, -0.1, 0.010999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
                                extra_obj_start_poses[2]]
        
    elif selected_obj_id == 2:
        BEAM_PEG_T_TIP_DEFAULT = ([-0.043, 0, 0.01], [-1, 0, 0, 0])

        extra_obj_start_poses = [multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]), 
                        multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]),
                        multiply(BEAM_CENTER, [[-0.0885, -0.1, 0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])]
        # Modify the first element of the first tuple in the first sublist
        # ic(1, extra_obj_start_poses[1])
        # extra_obj_start_poses[1] = list(extra_obj_start_poses[1])  # Convert tuple to list for mutability
        # extra_obj_start_poses[1][0] = list(extra_obj_start_poses[1][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[1][0][0] -= 0.1  
        # extra_obj_start_poses[1][0][2] -= 0.028
        # extra_obj_start_poses[1][0] = tuple(extra_obj_start_poses[1][0])  # Convert back to tuple
        # extra_obj_start_poses[1] = tuple(extra_obj_start_poses[1])  # Convert back to tuple
        # ic(2, extra_obj_start_poses[1])
    elif selected_obj_id == 3:
        BEAM_PEG_T_TIP_DEFAULT = ([0.046, 0, 0.01], [-1, 0, 0, 0])
        # Modify the first element of the first tuple in the first sublist
        ic(1, extra_obj_start_poses[0])
        extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[0][0][0] += 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[0][0][2] -= 0.04
        extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        ic(2, extra_obj_start_poses[0])
    else: # selected_obj_id == 0:
        BEAM_PEG_T_TIP_DEFAULT = ([-0.043, 0, 0.03], [-1, 0, 0, 0])

        # for i, each_extra_obj_start_poses in enumerate(extra_obj_start_poses):
        #     if i == 2:
        #         continue
        #     each_extra_obj_start_poses = list(each_extra_obj_start_poses)
        #     each_extra_obj_start_poses[0] = list(each_extra_obj_start_poses[0])  # Convert inner tuple to list for mutability
        #     each_extra_obj_start_poses[0][0] += 0.4  # Add 0.1 to the first element
        #     # each_extra_obj_start_poses[0][2] += 0.04
        #     each_extra_obj_start_poses[0] = tuple(each_extra_obj_start_poses[0])  # Convert back to 
        #     each_extra_obj_start_poses = tuple(each_extra_obj_start_poses)
        #     extra_obj_start_poses[i] = each_extra_obj_start_poses
        # target_obj_start_pose = multiply(target_obj_start_pose, [[0.4, 0.1, -0.01], [0.0, 0.0, 0.0, 1.0]])
        # hole_obj_goal_pose = multiply(hole_obj_goal_pose, [[0.4, 0.1, -0.01], [0.0, 0.0, 0.0, 1.0]])

    print(selected_obj_id)
    # version = 'v1'
    # if target_obj_name in ['Medium_Short_SquareCircle_Red', 'Medium_Short_3Prong_JeansRed', 'Medium_Short_DoubleSquare_Purple']:
    #     target_obj_goal_pose = multiply(target_obj_goal_pose, ([0., 0., 0.0], [0, 1, 0, 0]))
    #     PEG_T_TIP = multiply(multiply(PEG_T_TIP_DEFAULT, ([0., 0., 0.0], [0, 1, 0, 0])), ([0., 0., 0.0], [0, 0, 1, 0]))
    # elif target_obj_name in ['Medium_Short_Rectangle_JeansBlue']:
    #     PEG_T_TIP = multiply(PEG_T_TIP_DEFAULT, ([0., 0., 0.0], [0, 0, 1, 0]))
    # else:
    # print(BEAM_OBJECT_DICT)
    PEG_T_TIP = BEAM_PEG_T_TIP_DEFAULT
    # hole_obj_goal_pose = ([-0.4488, -0.0189+0.1, 0.018], [0,0,1,0])
    # HOLE_T_PEG_GOAL = multiply(target_obj_goal_pose, invert(hole_obj_goal_pose))  #  ([-0.070, -0.089, 0.03], [0,0,1,0]) # # Good for board after rotate 180 deg
    HOLE_T_PEG_GOAL = multiply(invert(hole_obj_goal_pose), target_obj_goal_pose)
    # new_target_obj_goal_pose = multiply(hole_obj_goal_pose, HOLE_T_PEG_GOAL)
    # print(target_obj_goal_pose, new_target_obj_goal_pose)
    # HOLE_T_PEG_GOAL = multiply(HOLE_T_PEG_GOAL, HOLE_ROTATE)
    # HOLE_T_PEG_GOAL = multiply(([0, 0, 0], [0, 0, 1, 0]), HOLE_T_PEG_GOAL)
    # if version == 'v1': # peg start on the board
    #     WORLD_T_PEG_START = multiply(([0, 0, 0.06], [0,0,0,1]), multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))
        # WORLD_T_PEG_START = multiply(([0, 0, 0.05], [0,0,0,1]), multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))

    # elif version == 'v2':  # peg start on the fixture
    WORLD_T_PEG_START = target_obj_start_pose  # multiply(([0.0, 0.0, 0.10], [0, 0, 0, 1]), BEAM_WORLD_T_FIXTURE_POSE)
    ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, target_obj_usd_path, hole_obj_usd_path)

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)
    # print('\n'*50, '************************', 'hole_obj_goal_pose: ', hole_obj_goal_pose, '************************', '\n'*50)
    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME=hole_obj_usd_path,
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=hole_obj_goal_pose,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=extra_obj_start_poses,
                      allow_peg_rotation=True,
                      USE_FIXED_RIGID_ATTACHMENT=True,
                      allow_gripper_status=False,
                      TASK_NAME='BeamInsert-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      WORLD_T_PEG_START_LIST=[WORLD_T_PEG_START, ([0.3, 0.6, 0.1+0.18], [0.5000, -0.5000,  0.5000,  0.5000])],
                      USE_FT_SENSOR=True,
                      USE_CAMERA=False,
                      # peg_goal_weights=[5, 5, 5, 0.1, 0.1, 0.1],
                      peg_goal_weights=[1, 1, 1, 1, 1, 1],
                      DIFFICULTY_LEVEL='easy',
                      relative_pose_obs=True,
                      selected_obj_id=selected_obj_id,
                      POSE_AFTER=BEAM_SAFE_POSE_AFTER['BeamInsert'],
                      )


def BeamScriptedGrasp(selected_obj_id=1, desired_tool_T_world=None):
    selected_obj_id = 2
    SHOW_GOAL = False # True
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses,
     hole_obj_name, hole_obj_usd_path, hole_obj_start_pose, hole_obj_goal_pose, hole_peg_attachment_prim,) = beam_split_obj_dict(
        BEAM_OBJECT_DICT, selected_obj_id=selected_obj_id, )
    
    if selected_obj_id == 1:
        BEAM_PEG_T_TIP_DEFAULT = ([0.046, 0, 0.033], [-1, 0, 0, 0])
        target_obj_start_pose = multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])

        extra_obj_start_poses = [multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]), 
                                multiply(BEAM_CENTER, [[-0.0885, -0.1, 0.010999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
                                extra_obj_start_poses[2]]
        # hole_obj_goal_pose = multiply(BEAM_CENTER, [[0.0885, 0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])
        # extra_obj_start_poses = [multiply(BEAM_CENTER, [[0.0885, -0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                      multiply(BEAM_CENTER, [[0.1885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                      multiply(BEAM_CENTER, [[-0.0885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])]
    elif selected_obj_id == 2:
        BEAM_PEG_T_TIP_DEFAULT = ([-0.043, 0, 0.01], [-1, 0, 0, 0])

        target_obj_start_pose = multiply(BEAM_CENTER, [[-0.2885, -0., 0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(BEAM_CENTER, [[-0.2885, -0., 0.04000000077], [0.0, 0.0, 0.0, 1.0]])

        extra_obj_start_poses = [multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]), 
                                multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(BEAM_CENTER, [[-0.0885, -0.1, 0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])]
        
        # hole_obj_goal_pose = multiply(BEAM_CENTER, [[0.0885, 0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])
        # extra_obj_start_poses = [multiply(BEAM_CENTER, [[0.0885, -0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                         multiply(BEAM_CENTER, [[0.1885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                         multiply(BEAM_CENTER, [[-0.0885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])]
        # Modify the first element of the first tuple in the first sublist
        # ic(1, extra_obj_start_poses[1])
        # extra_obj_start_poses[1] = list(extra_obj_start_poses[1])  # Convert tuple to list for mutability
        # extra_obj_start_poses[1][0] = list(extra_obj_start_poses[1][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[1][0][0] -= 0.1  
        # extra_obj_start_poses[1][0][2] -= 0.028
        # extra_obj_start_poses[1][0] = tuple(extra_obj_start_poses[1][0])  # Convert back to tuple
        # extra_obj_start_poses[1] = tuple(extra_obj_start_poses[1])  # Convert back to tuple
        # ic(2, extra_obj_start_poses[1])
    elif selected_obj_id == 3:
        BEAM_PEG_T_TIP_DEFAULT = ([0.046, 0, 0.02], [-1, 0, 0, 0])

        target_obj_start_pose = multiply(BEAM_CENTER, [[-0.0885, -0.1, 0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(BEAM_CENTER, [[-0.0885, -0.1,  0.04000000077], [0.0, 0.0, 0.0, 1.0]])
        # hole_obj_goal_pose = multiply(BEAM_CENTER, [[0.0885, 0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])
        extra_obj_start_poses = [multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]), 
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2]]
        # Modify the first element of the first tuple in the first sublist
        # ic(1, extra_obj_start_poses[0])
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] -= 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        # ic(2, extra_obj_start_poses[0])
    elif selected_obj_id == 0:
        target_obj_start_pose = multiply(BEAM_CENTER, [[-0.0885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(BEAM_CENTER, [[-0.0885, -0.1, -0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])
        # hole_obj_goal_pose = multiply(BEAM_CENTER, [[0.0885, 0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])
        # extra_obj_start_poses = [multiply(BEAM_CENTER, [[0.0885, -0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                         multiply(BEAM_CENTER, [[0.1885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                         multiply(BEAM_CENTER, [[-0.0885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])]
        BEAM_PEG_T_TIP_DEFAULT = ([-0.043, 0, 0.033], [-1, 0, 0, 0])
    else:
        raise NotImplementedError(f"selected_obj_id {selected_obj_id} not supported")
    
    ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, hole_obj_goal_pose, extra_obj_start_poses)
    PEG_T_TIP = BEAM_PEG_T_TIP_DEFAULT
    HOLE_T_PEG_GOAL = multiply(invert(hole_obj_goal_pose), target_obj_goal_pose)

    # elif version == 'v2':  # peg start on the fixture
    WORLD_T_PEG_START = target_obj_start_pose  # multiply(([0.0, 0.0, 0.10], [0, 0, 0, 1]), BEAM_WORLD_T_FIXTURE_POSE)
    # ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, target_obj_usd_path, hole_obj_usd_path)

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)
    # print('\n'*50, '************************', 'hole_obj_goal_pose: ', hole_obj_goal_pose, '************************', '\n'*50)
    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME=hole_obj_usd_path,
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=hole_obj_goal_pose,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=extra_obj_start_poses,
                      allow_peg_rotation=True,
                      USE_FIXED_RIGID_ATTACHMENT=False, # True,
                      allow_gripper_status=False,
                      TASK_NAME='BeamScriptedGrasp-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      USE_FT_SENSOR=True,
                      USE_CAMERA=False,
                      WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                      peg_goal_weights=[1, 1, 1, 1, 1, 1],
                      DIFFICULTY_LEVEL='easy',
                      relative_pose_obs=True,
                      selected_obj_id=selected_obj_id,
                      POSE_AFTER=BEAM_SAFE_POSE_AFTER['BeamScriptedGrasp'],
                      )

def BeamScriptedPlace(selected_obj_id=1, desired_tool_T_world=None):
    selected_obj_id = 2 # 3
    SHOW_GOAL = False # True
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses,
     hole_obj_name, hole_obj_usd_path, hole_obj_start_pose, hole_obj_goal_pose, hole_peg_attachment_prim,) = beam_split_obj_dict(
        BEAM_OBJECT_DICT, selected_obj_id=selected_obj_id, )
    
    if selected_obj_id == 1:
        BEAM_PEG_T_TIP_DEFAULT = ([0.046, 0, 0.033], [-1, 0, 0, 0])
        target_obj_goal_pose = multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])
        hole_obj_goal_pose = multiply(BEAM_CENTER, [[0.0885, 0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])
        extra_obj_start_poses = [multiply(BEAM_CENTER, [[0.0885, -0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
                             multiply(BEAM_CENTER, [[0.1885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
                             multiply(BEAM_CENTER, [[-0.0885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])]
    elif selected_obj_id == 2:
        BEAM_PEG_T_TIP_DEFAULT = ([-0.043, 0, 0.01], [-1, 0, 0, 0])

        target_obj_goal_pose = multiply(BEAM_CENTER, [[-0.2885, -0., 0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(BEAM_CENTER, [[-0.2885, -0., 0.04000000077], [0.0, 0.0, 0.0, 1.0]])
        hole_obj_goal_pose = multiply(BEAM_CENTER, [[0.0885, 0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])

        extra_obj_start_poses = [multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]), 
                                multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(BEAM_CENTER, [[-0.0885, -0.1, 0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])]
        # Modify the first element of the first tuple in the first sublist
        # ic(1, extra_obj_start_poses[1])
        # extra_obj_start_poses[1] = list(extra_obj_start_poses[1])  # Convert tuple to list for mutability
        # extra_obj_start_poses[1][0] = list(extra_obj_start_poses[1][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[1][0][0] -= 0.1  
        # extra_obj_start_poses[1][0][2] -= 0.028
        # extra_obj_start_poses[1][0] = tuple(extra_obj_start_poses[1][0])  # Convert back to tuple
        # extra_obj_start_poses[1] = tuple(extra_obj_start_poses[1])  # Convert back to tuple
        # ic(2, extra_obj_start_poses[1])
    elif selected_obj_id == 3:
        BEAM_PEG_T_TIP_DEFAULT = ([0.046, 0, 0.01], [-1, 0, 0, 0])

        target_obj_goal_pose = multiply(BEAM_CENTER, [[-0.0885, -0.1, 0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(BEAM_CENTER, [[-0.0885, -0.1,  0.04000000077], [0.0, 0.0, 0.0, 1.0]])
        hole_obj_goal_pose = multiply(BEAM_CENTER, [[0.0885, 0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])
        extra_obj_start_poses = [multiply(BEAM_CENTER, [[0.0885, -0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
                                multiply(BEAM_CENTER, [[0.1885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
                             multiply(BEAM_CENTER, [[-0.0885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])]
        # Modify the first element of the first tuple in the first sublist
        # ic(1, extra_obj_start_poses[0])
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] -= 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        # ic(2, extra_obj_start_poses[0])
    else: # selected_obj_id == 0:
        target_obj_goal_pose = multiply(BEAM_CENTER, [[-0.0885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(BEAM_CENTER, [[-0.0885, -0.1, -0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])
        hole_obj_goal_pose = multiply(BEAM_CENTER, [[0.0885, 0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])
        extra_obj_start_poses = [multiply(BEAM_CENTER, [[0.0885, -0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
                                multiply(BEAM_CENTER, [[0.1885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
                                multiply(BEAM_CENTER, [[-0.0885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])]
        BEAM_PEG_T_TIP_DEFAULT = ([-0.043, 0, 0.033], [-1, 0, 0, 0])
    ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, hole_obj_goal_pose, extra_obj_start_poses)
    PEG_T_TIP = BEAM_PEG_T_TIP_DEFAULT
    HOLE_T_PEG_GOAL = multiply(invert(hole_obj_goal_pose), target_obj_goal_pose)

    # elif version == 'v2':  # peg start on the fixture
    WORLD_T_PEG_START = target_obj_start_pose  # multiply(([0.0, 0.0, 0.10], [0, 0, 0, 1]), BEAM_WORLD_T_FIXTURE_POSE)
    # ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, target_obj_usd_path, hole_obj_usd_path)

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)
    # print('\n'*50, '************************', 'hole_obj_goal_pose: ', hole_obj_goal_pose, '************************', '\n'*50)
    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME=hole_obj_usd_path,
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=hole_obj_goal_pose,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=extra_obj_start_poses,
                      allow_peg_rotation=True,
                      USE_FIXED_RIGID_ATTACHMENT=False, # True,
                      allow_gripper_status=False,
                      TASK_NAME='BeamScriptedPlace-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      USE_FT_SENSOR=True,
                      USE_CAMERA=False,
                      WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                      peg_goal_weights=[1, 1, 1, 1, 1, 1],
                      DIFFICULTY_LEVEL='easy',
                      relative_pose_obs=True,
                      selected_obj_id=selected_obj_id,
                      POSE_AFTER=BEAM_SAFE_POSE_AFTER['BeamScriptedPlace'],
                      )

def BeamMoveToBoard(selected_obj_id=1, desired_tool_T_world=None):
    selected_obj_id = 2 # 3
    SHOW_GOAL = False # True
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses,
     hole_obj_name, hole_obj_usd_path, hole_obj_start_pose, hole_obj_goal_pose, hole_peg_attachment_prim,) = beam_split_obj_dict(
        BEAM_OBJECT_DICT, selected_obj_id=selected_obj_id, )
    
    if selected_obj_id == 1:
        BEAM_PEG_T_TIP_DEFAULT = ([0.046, 0, 0.033], [-1, 0, 0, 0])
        target_obj_goal_pose = copy.copy(target_obj_start_pose)
        target_obj_start_pose = multiply(BEAM_CENTER, [[-0.1885, -0.1, 0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])

        extra_obj_start_poses = [multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]), 
                                multiply(BEAM_CENTER, [[-0.0885, -0.1, 0.010999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
                                extra_obj_start_poses[2]]
        # target_obj_start_pose = multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]])
        # target_obj_goal_pose = multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])
        # hole_obj_goal_pose = multiply(BEAM_CENTER, [[0.0885, 0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])
        # extra_obj_start_poses = [multiply(BEAM_CENTER, [[0.0885, -0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                      multiply(BEAM_CENTER, [[0.1885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                      multiply(BEAM_CENTER, [[-0.0885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])]
    elif selected_obj_id == 2:
        BEAM_PEG_T_TIP_DEFAULT = ([-0.043, 0, 0.01], [-1, 0, 0, 0])

        target_obj_goal_pose = multiply(copy.copy(target_obj_start_pose), [[0.0, 0., 0.05], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(BEAM_CENTER, [[-0.2885, -0., 0.05000000077], [0.0, 0.0, 0.0, 1.0]])

        extra_obj_start_poses = [multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]), 
                        multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]),
                        multiply(BEAM_CENTER, [[-0.0885, -0.1, 0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])]
        # Modify the first element of the first tuple in the first sublist
        # ic(1, extra_obj_start_poses[1])
        # extra_obj_start_poses[1] = list(extra_obj_start_poses[1])  # Convert tuple to list for mutability
        # extra_obj_start_poses[1][0] = list(extra_obj_start_poses[1][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[1][0][0] -= 0.1  
        # extra_obj_start_poses[1][0][2] -= 0.028
        # extra_obj_start_poses[1][0] = tuple(extra_obj_start_poses[1][0])  # Convert back to tuple
        # extra_obj_start_poses[1] = tuple(extra_obj_start_poses[1])  # Convert back to tuple
        # ic(2, extra_obj_start_poses[1])
    elif selected_obj_id == 3:
        BEAM_PEG_T_TIP_DEFAULT = ([0.046, 0, 0.01], [-1, 0, 0, 0])


        target_obj_goal_pose = multiply(copy.copy(target_obj_start_pose), [[0.0, 0., 0.05], [0.0, 0.0, 0.0, 1.0]]) # multiply(BEAM_CENTER, [[-0.0885, -0.1,  0.04000000077], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(BEAM_CENTER, [[-0.0885, -0.1,  0.04000000077], [0.0, 0.0, 0.0, 1.0]])
        # hole_obj_goal_pose = multiply(BEAM_CENTER, [[0.0885, 0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])
        # extra_obj_start_poses = [multiply(BEAM_CENTER, [[0.0885, -0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                         multiply(BEAM_CENTER, [[0.1885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                      multiply(BEAM_CENTER, [[-0.0885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])]
        # Modify the first element of the first tuple in the first sublist
        # ic(1, extra_obj_start_poses[0])
        extra_obj_start_poses = [multiply(BEAM_CENTER, [[-0.1885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]]), 
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2]]
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] -= 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        # ic(2, extra_obj_start_poses[0])
    else: # selected_obj_id == 0:
        # new_target_obj_start_pose = copy.copy(target_obj_goal_pose) # multiply(BEAM_CENTER, [[-0.0885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = copy.copy(target_obj_start_pose)
        target_obj_start_pose = multiply(BEAM_CENTER, [[-0.0885, -0.1, 0.010999999225139618], [0.0, 0.0, 0.0, 1.0]])
        # hole_obj_goal_pose = multiply(BEAM_CENTER, [[0.0885, 0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])
        # extra_obj_start_poses = [multiply(BEAM_CENTER, [[0.0885, -0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                         multiply(BEAM_CENTER, [[0.1885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                         multiply(BEAM_CENTER, [[-0.0885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])]
        BEAM_PEG_T_TIP_DEFAULT = ([-0.043, 0, 0.025], [-1, 0, 0, 0])
    ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, hole_obj_goal_pose, extra_obj_start_poses)
    PEG_T_TIP = BEAM_PEG_T_TIP_DEFAULT
    HOLE_T_PEG_GOAL = multiply(invert(hole_obj_goal_pose), target_obj_goal_pose)

    # elif version == 'v2':  # peg start on the fixture
    WORLD_T_PEG_START = target_obj_start_pose  # multiply(([0.0, 0.0, 0.10], [0, 0, 0, 1]), BEAM_WORLD_T_FIXTURE_POSE)
    # ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, target_obj_usd_path, hole_obj_usd_path)

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)
    # print('\n'*50, '************************', 'hole_obj_goal_pose: ', hole_obj_goal_pose, '************************', '\n'*50)
    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME=hole_obj_usd_path,
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=hole_obj_goal_pose,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=extra_obj_start_poses,
                      allow_peg_rotation=True,
                      USE_FIXED_RIGID_ATTACHMENT=False, # True,
                      allow_gripper_status=False,
                      TASK_NAME='BeamMoveToBoard-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      USE_FT_SENSOR=True,
                      USE_CAMERA=False,
                      WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                      peg_goal_weights=[1, 1, 1, 1, 1, 1],
                      DIFFICULTY_LEVEL='easy',
                      relative_pose_obs=True,
                      selected_obj_id=selected_obj_id,
                      POSE_AFTER=BEAM_SAFE_POSE_AFTER['BeamMoveToBoard'],
                      )


# QIANZHONG tuned
# STOOL_WORLD_T_FIXTURE_POSE = ([0.3895, 0.20, 0.048], [0, 0, 0, 1]) 
# STOOL_WORLD_T_FIXTURE_POSE = multiply(STOOL_CENTER, ([-0.463, -0.338, 0.02], [0, 0, 1, 0]))
STOOL_WORLD_T_HOLE_START = multiply(STOOL_CENTER, ([-0.405, -0.395, 0.015], [0, 0, 0, 1]))
STOOL_SELECTED_OBJ_ID = 3 # 3 # 7 # 5 # 4 # 0 # 6 # 8
# STOOL_PEG_T_TIP_DEFAULT = ([-0.4, 0.2, 0.8], [-1, 0, 0, 0])
def StoolScriptedInsert(selected_obj_id=0, desired_tool_T_world=None):
    SHOW_GOAL = False # True
    selected_obj_id = STOOL_SELECTED_OBJ_ID # 4 # 8 # 6 # 4  # should not be 2
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses,
     hole_obj_name, hole_obj_usd_path, hole_obj_start_pose, hole_obj_goal_pose, hole_peg_attachment_prim,) = stool_split_obj_dict(
        STOOL_OBJECT_DICT, selected_obj_id, )
    if selected_obj_id == 1:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.025, 0.029, 0.03], [-1, 0, 0, 0])
        # ic(1, extra_obj_start_poses[0])
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] += 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        # extra_obj_start_poses[4] = list(extra_obj_start_poses[4])  # Convert tuple to list for mutability
        # extra_obj_start_poses[4][0] = list(extra_obj_start_poses[4][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[4][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[4][0][2] -= 0.04
        # extra_obj_start_poses[4][0] = tuple(extra_obj_start_poses[4][0])  # Convert back to tuple
        # extra_obj_start_poses[4] = tuple(extra_obj_start_poses[4])  # Convert back to tuple
        # ic(2, extra_obj_start_poses[0])
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.455, -0.403, 0.02], [0, 0, 0, 1])), [[0.35, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]), 
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.396, -0.3950, 0.018], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
    elif selected_obj_id == 3:
        STOOL_PEG_T_TIP_DEFAULT = ([0.03, -0.032, 0.03], [-1, 0, 0, 0])
        # Modify the first element of the first tuple in the first sublist
        # extra_obj_start_poses[2] = list(extra_obj_start_poses[2])  # Convert tuple to list for mutability
        # extra_obj_start_poses[2][0] = list(extra_obj_start_poses[2][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[2][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[2][0][2] -= 0.04
        # extra_obj_start_poses[2][0] = tuple(extra_obj_start_poses[2][0])  # Convert back to tuple
        # extra_obj_start_poses[2] = tuple(extra_obj_start_poses[2])  # Convert back to tuple
        # extra_obj_start_poses[6] = list(extra_obj_start_poses[6])  # Convert tuple to list for mutability
        # extra_obj_start_poses[6][0] = list(extra_obj_start_poses[6][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[6][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[6][0][2] += 0.04
        # extra_obj_start_poses[6][0] = tuple(extra_obj_start_poses[6][0])  # Convert back to tuple
        # extra_obj_start_poses[6] = tuple(extra_obj_start_poses[6])  # Convert back to tuple
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                # extra_obj_start_poses[2],
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.396, -0.3950, 0.018], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
    elif selected_obj_id == 4:
        STOOL_PEG_T_TIP_DEFAULT = ([0.035, 0.03, 0.06], [-1, 0, 0, 0])
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                extra_obj_start_poses[3],
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
    elif selected_obj_id == 5:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.025, -0.028, 0.04], [-1, 0, 0, 0])
        # extra_obj_start_poses[3] = list(extra_obj_start_poses[3])  # Convert tuple to list for mutability
        # extra_obj_start_poses[3][0] = list(extra_obj_start_poses[3][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[3][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[3][0][2] += 0.04
        # extra_obj_start_poses[3][0] = tuple(extra_obj_start_poses[3][0])  # Convert back to tuple
        # extra_obj_start_poses[3] = tuple(extra_obj_start_poses[3])  # Convert back to tuple
        # extra_obj_start_poses[6] = list(extra_obj_start_poses[6])  # Convert tuple to list for mutability
        # extra_obj_start_poses[6][0] = list(extra_obj_start_poses[6][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[6][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[6][0][2] += 0.04
        # extra_obj_start_poses[6][0] = tuple(extra_obj_start_poses[6][0])  # Convert back to tuple
        # extra_obj_start_poses[6] = tuple(extra_obj_start_poses[6])  # Convert back to tuple
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
    elif selected_obj_id == 6:
        STOOL_PEG_T_TIP_DEFAULT = ([0.035, -0.03, 0.06], [-1, 0, 0, 0])
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] += 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        extra_obj_start_poses = [extra_obj_start_poses[0], 
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                extra_obj_start_poses[3],
                                extra_obj_start_poses[4],
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
    elif selected_obj_id == 7:
        STOOL_PEG_T_TIP_DEFAULT = ([0.027, 0.03, 0.04], [-1, 0, 0, 0])
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] += 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        # extra_obj_start_poses[5] = list(extra_obj_start_poses[5])  # Convert tuple to list for mutability
        # extra_obj_start_poses[5][0] = list(extra_obj_start_poses[5][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[5][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[5][0][2] += 0.04
        # extra_obj_start_poses[5][0] = tuple(extra_obj_start_poses[5][0])  # Convert back to tuple
        # extra_obj_start_poses[5] = tuple(extra_obj_start_poses[5])  # Convert back to tuple
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
    elif selected_obj_id == 8:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.03, -0.03, 0.07], [-1, 0, 0, 0])
    else: # selected_obj_id == 0:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.03, 0.032, 0.06], [-1, 0, 0, 0])
        extra_obj_start_poses = [extra_obj_start_poses[0], 
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                extra_obj_start_poses[3],
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.20, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.20, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
        
    print(selected_obj_id)

    # else:
    PEG_T_TIP = STOOL_PEG_T_TIP_DEFAULT
    HOLE_T_PEG_GOAL = target_obj_goal_pose  #  ([-0.070, -0.089, 0.03], [0,0,1,0]) # # Good for board after rotate 180 deg
    HOLE_T_PEG_GOAL = multiply(invert(hole_obj_start_pose), target_obj_goal_pose) # multiply(HOLE_T_PEG_GOAL, HOLE_ROTATE)
    # HOLE_T_PEG_GOAL = multiply(([0, 0, 0], [0, 0, 1, 0]), HOLE_T_PEG_GOAL)
    # if version == 'v1': # peg start on the board
    #     WORLD_T_PEG_START = multiply(([0, 0, 0.06], [0,0,0,1]), multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))
    #     # WORLD_T_PEG_START = multiply(([0, 0, 0.05], [0,0,0,1]), multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))

    # elif version == 'v2':  # peg start on the fixture
    WORLD_T_PEG_START = target_obj_start_pose

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)

    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME=hole_obj_usd_path,
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=hole_obj_start_pose,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=extra_obj_start_poses,
                      allow_peg_rotation=True,
                      USE_FIXED_RIGID_ATTACHMENT=True,
                      allow_gripper_status=False,
                      TASK_NAME='StoolScriptedInsert-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      WORLD_T_PEG_START_LIST=[WORLD_T_PEG_START, ([0.3, 0.6, 0.1+0.18], [0.5000, -0.5000,  0.5000,  0.5000])],
                      USE_FT_SENSOR=True,
                      USE_CAMERA=False,
                      # peg_goal_weights=[5, 5, 5, 0.1, 0.1, 0.1],
                      peg_goal_weights=[1, 1, 1, 1, 1, 1],
                      DIFFICULTY_LEVEL='easy',
                      relative_pose_obs=True,
                      selected_obj_id=selected_obj_id,
                      POSE_AFTER=STOOL_SAFE_POSE_AFTER['StoolScriptedInsert'],
                      )

def StoolInsert(selected_obj_id=0, desired_tool_T_world=None):
    SHOW_GOAL = False # True
    selected_obj_id = STOOL_SELECTED_OBJ_ID # 4 # 8 # 6 # 4 # 8  # should not be 2
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses,
     hole_obj_name, hole_obj_usd_path, hole_obj_start_pose, hole_obj_goal_pose, hole_peg_attachment_prim,) = stool_split_obj_dict(
        STOOL_OBJECT_DICT, selected_obj_id, )
    if selected_obj_id == 1:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.025, 0.029, 0.03], [-1, 0, 0, 0])
        # ic(1, extra_obj_start_poses[0])
        extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[0][0][2] += 0.04
        extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        extra_obj_start_poses[4] = list(extra_obj_start_poses[4])  # Convert tuple to list for mutability
        extra_obj_start_poses[4][0] = list(extra_obj_start_poses[4][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[4][0][0] += 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[4][0][2] -= 0.04
        extra_obj_start_poses[4][0] = tuple(extra_obj_start_poses[4][0])  # Convert back to tuple
        extra_obj_start_poses[4] = tuple(extra_obj_start_poses[4])  # Convert back to tuple
        # ic(2, extra_obj_start_poses[0])
    elif selected_obj_id == 3:
        STOOL_PEG_T_TIP_DEFAULT = ([0.03, -0.032, 0.03], [-1, 0, 0, 0])
        # Modify the first element of the first tuple in the first sublist
        extra_obj_start_poses[2] = list(extra_obj_start_poses[2])  # Convert tuple to list for mutability
        extra_obj_start_poses[2][0] = list(extra_obj_start_poses[2][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[2][0][0] -= 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[2][0][2] -= 0.04
        extra_obj_start_poses[2][0] = tuple(extra_obj_start_poses[2][0])  # Convert back to tuple
        extra_obj_start_poses[2] = tuple(extra_obj_start_poses[2])  # Convert back to tuple
        extra_obj_start_poses[6] = list(extra_obj_start_poses[6])  # Convert tuple to list for mutability
        extra_obj_start_poses[6][0] = list(extra_obj_start_poses[6][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[6][0][0] += 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[6][0][2] += 0.04
        extra_obj_start_poses[6][0] = tuple(extra_obj_start_poses[6][0])  # Convert back to tuple
        extra_obj_start_poses[6] = tuple(extra_obj_start_poses[6])  # Convert back to tuple
    elif selected_obj_id == 4:
        STOOL_PEG_T_TIP_DEFAULT = ([0.035, 0.03, 0.06], [-1, 0, 0, 0])
    elif selected_obj_id == 5:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.025, -0.028, 0.04], [-1, 0, 0, 0])
        extra_obj_start_poses[3] = list(extra_obj_start_poses[3])  # Convert tuple to list for mutability
        extra_obj_start_poses[3][0] = list(extra_obj_start_poses[3][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[3][0][0] -= 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[3][0][2] += 0.04
        extra_obj_start_poses[3][0] = tuple(extra_obj_start_poses[3][0])  # Convert back to tuple
        extra_obj_start_poses[3] = tuple(extra_obj_start_poses[3])  # Convert back to tuple
        extra_obj_start_poses[6] = list(extra_obj_start_poses[6])  # Convert tuple to list for mutability
        extra_obj_start_poses[6][0] = list(extra_obj_start_poses[6][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[6][0][0] += 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[6][0][2] += 0.04
        extra_obj_start_poses[6][0] = tuple(extra_obj_start_poses[6][0])  # Convert back to tuple
        extra_obj_start_poses[6] = tuple(extra_obj_start_poses[6])  # Convert back to tuple
    elif selected_obj_id == 6:
        STOOL_PEG_T_TIP_DEFAULT = ([0.035, -0.03, 0.06], [-1, 0, 0, 0])
        extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[0][0][2] += 0.04
        extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
    elif selected_obj_id == 7:
        STOOL_PEG_T_TIP_DEFAULT = ([0.027, 0.03, 0.04], [-1, 0, 0, 0])
        extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[0][0][2] += 0.04
        extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        extra_obj_start_poses[5] = list(extra_obj_start_poses[5])  # Convert tuple to list for mutability
        extra_obj_start_poses[5][0] = list(extra_obj_start_poses[5][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[5][0][0] += 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[5][0][2] += 0.04
        extra_obj_start_poses[5][0] = tuple(extra_obj_start_poses[5][0])  # Convert back to tuple
        extra_obj_start_poses[5] = tuple(extra_obj_start_poses[5])  # Convert back to tuple
    elif selected_obj_id == 8:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.03, -0.03, 0.07], [-1, 0, 0, 0])
    else: # selected_obj_id == 0:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.03, 0.032, 0.06], [-1, 0, 0, 0])
    print(selected_obj_id)

    # else:
    PEG_T_TIP = STOOL_PEG_T_TIP_DEFAULT
    HOLE_T_PEG_GOAL = target_obj_goal_pose  #  ([-0.070, -0.089, 0.03], [0,0,1,0]) # # Good for board after rotate 180 deg
    HOLE_T_PEG_GOAL = multiply(invert(hole_obj_start_pose), target_obj_goal_pose) # multiply(HOLE_T_PEG_GOAL, HOLE_ROTATE)
    # HOLE_T_PEG_GOAL = multiply(([0, 0, 0], [0, 0, 1, 0]), HOLE_T_PEG_GOAL)
    # if version == 'v1': # peg start on the board
    #     WORLD_T_PEG_START = multiply(([0, 0, 0.06], [0,0,0,1]), multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))
    #     # WORLD_T_PEG_START = multiply(([0, 0, 0.05], [0,0,0,1]), multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))

    # elif version == 'v2':  # peg start on the fixture
    WORLD_T_PEG_START = target_obj_start_pose

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)

    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME=hole_obj_usd_path,
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=hole_obj_goal_pose,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=extra_obj_start_poses,
                      allow_peg_rotation=True,
                      USE_FIXED_RIGID_ATTACHMENT=True,
                      allow_gripper_status=False,
                      TASK_NAME='StoolInsert-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      WORLD_T_PEG_START_LIST=[WORLD_T_PEG_START, ([0.3, 0.6, 0.1+0.18], [0.5000, -0.5000,  0.5000,  0.5000])],
                      USE_FT_SENSOR=True,
                      USE_CAMERA=False,
                      # peg_goal_weights=[5, 5, 5, 0.1, 0.1, 0.1],
                      peg_goal_weights=[1, 1, 1, 1, 1, 1],
                      DIFFICULTY_LEVEL='easy',
                      relative_pose_obs=True,
                      selected_obj_id=selected_obj_id,
                      POSE_AFTER=STOOL_SAFE_POSE_AFTER['StoolInsert'],
                      )

def StoolScriptedGrasp(selected_obj_id=1, desired_tool_T_world=None):
    selected_obj_id = STOOL_SELECTED_OBJ_ID # 5 # 8 # 6 # 4
    SHOW_GOAL = False # True
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses,
     hole_obj_name, hole_obj_usd_path, hole_obj_start_pose, hole_obj_goal_pose, hole_peg_attachment_prim,) = stool_split_obj_dict(
        STOOL_OBJECT_DICT, selected_obj_id=selected_obj_id, )
    
    if selected_obj_id == 1:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.025, 0.029, 0.03], [-1, 0, 0, 0])
        # target_obj_start_pose = multiply(target_obj_start_pose, [[-0.1, 0.1, -0.01], [0.0, 0.0, 0.0, 1.0]])
        # target_obj_goal_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.405, -0.395, 0.018], [0, 0, 0, 1])), [[0.35, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(multiply(STOOL_CENTER, ([-0.405, -0.395, 0.018], [0, 0, 0, 1])), [[0.35, 0.0, 0.03], [0.0, 0.0, 0.0, 1.0]])
        # hole_obj_goal_pose = multiply(BEAM_CENTER, [[0.0885, 0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])
        # extra_obj_start_poses = [multiply(BEAM_CENTER, [[0.0885, -0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                      multiply(BEAM_CENTER, [[0.1885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                      multiply(BEAM_CENTER, [[-0.0885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])]
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] += 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        # extra_obj_start_poses[4] = list(extra_obj_start_poses[4])  # Convert tuple to list for mutability
        # extra_obj_start_poses[4][0] = list(extra_obj_start_poses[4][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[4][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[4][0][2] -= 0.04
        # extra_obj_start_poses[4][0] = tuple(extra_obj_start_poses[4][0])  # Convert back to tuple
        # extra_obj_start_poses[4] = tuple(extra_obj_start_poses[4])  # Convert back to tuple
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.455, -0.403, 0.02], [0, 0, 0, 1])), [[0.35, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]), 
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.396, -0.3950, 0.018], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]

    elif selected_obj_id == 3:
        STOOL_PEG_T_TIP_DEFAULT = ([0.03, -0.032, 0.03], [-1, 0, 0, 0])
        
        # target_obj_start_pose = multiply(target_obj_start_pose, [[-0.1, 0.1, -0.01], [0.0, 0.0, 0.0, 1.0]])
        # target_obj_goal_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.455, -0.403, 0.02], [0, 0, 0, 1])), [[0.35, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(multiply(STOOL_CENTER, ([-0.455, -0.403, 0.02], [0, 0, 0, 1])), [[0.35, 0.0, 0.03], [0.0, 0.0, 0.0, 1.0]])
        # Modify the first element of the first tuple in the first sublist
        # extra_obj_start_poses[2] = list(extra_obj_start_poses[2])  # Convert tuple to list for mutability
        # extra_obj_start_poses[2][0] = list(extra_obj_start_poses[2][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[2][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[2][0][2] -= 0.04
        # extra_obj_start_poses[2][0] = tuple(extra_obj_start_poses[2][0])  # Convert back to tuple
        # extra_obj_start_poses[2] = tuple(extra_obj_start_poses[2])  # Convert back to tuple
        # extra_obj_start_poses[6] = list(extra_obj_start_poses[6])  # Convert tuple to list for mutability
        # extra_obj_start_poses[6][0] = list(extra_obj_start_poses[6][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[6][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[6][0][2] += 0.04
        # extra_obj_start_poses[6][0] = tuple(extra_obj_start_poses[6][0])  # Convert back to tuple
        # extra_obj_start_poses[6] = tuple(extra_obj_start_poses[6])  # Convert back to tuple
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.396, -0.3950, 0.018], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
    elif selected_obj_id == 4:
        STOOL_PEG_T_TIP_DEFAULT = ([0.035, 0.03, 0.06], [-1, 0, 0, 0])
        # target_obj_start_pose = multiply(target_obj_start_pose, [[-0.1, -0.1, -0.08], [0.0, 0.0, 0.0, 1.0]])
        # target_obj_goal_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        
        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        # hole_obj_goal_pose = multiply(hole_obj_goal_pose, [[-0., -0.2, -0.0], [0.0, 0.0, 0.0, 1.0]])
        # extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
        #                         extra_obj_start_poses[1], 
        #                         extra_obj_start_poses[2],
        #                         extra_obj_start_poses[3],
        #                         multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
        #                         extra_obj_start_poses[5],
        #                         multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
        #                         ]
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                extra_obj_start_poses[3],
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
    elif selected_obj_id == 5:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.025, -0.028, 0.04], [-1, 0, 0, 0])
        # extra_obj_start_poses[3] = list(extra_obj_start_poses[3])  # Convert tuple to list for mutability
        # extra_obj_start_poses[3][0] = list(extra_obj_start_poses[3][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[3][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[3][0][2] += 0.04
        # extra_obj_start_poses[3][0] = tuple(extra_obj_start_poses[3][0])  # Convert back to tuple
        # extra_obj_start_poses[3] = tuple(extra_obj_start_poses[3])  # Convert back to tuple
        # extra_obj_start_poses[6] = list(extra_obj_start_poses[6])  # Convert tuple to list for mutability
        # extra_obj_start_poses[6][0] = list(extra_obj_start_poses[6][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[6][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[6][0][2] += 0.04
        # extra_obj_start_poses[6][0] = tuple(extra_obj_start_poses[6][0])  # Convert back to tuple
        # extra_obj_start_poses[6] = tuple(extra_obj_start_poses[6])  # Convert back to tuple
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]

        # target_obj_start_pose = multiply(target_obj_start_pose, [[-0.2, 0.1, -0.01], [0.0, 0.0, 0.0, 1.0]])
        # target_obj_goal_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), [[0.40, 0.0, 0.03], [0.0, 0.0, 0.0, 1.0]])

        
    elif selected_obj_id == 6:
        STOOL_PEG_T_TIP_DEFAULT = ([0.035, -0.03, 0.06], [-1, 0, 0, 0])
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] += 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        extra_obj_start_poses = [extra_obj_start_poses[0], 
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                extra_obj_start_poses[3],
                                extra_obj_start_poses[4],
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
        # target_obj_start_pose = multiply(target_obj_start_pose, [[-0.2, 0.1, -0.08], [0.0, 0.0, 0.0, 1.0]])
        # target_obj_goal_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])

        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.20, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.20, 0.0, 0.04], [0.0, 0.0, 0.0, 1.0]])
    elif selected_obj_id == 7:
        STOOL_PEG_T_TIP_DEFAULT = ([0.027, 0.03, 0.04], [-1, 0, 0, 0])
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] += 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        # extra_obj_start_poses[5] = list(extra_obj_start_poses[5])  # Convert tuple to list for mutability
        # extra_obj_start_poses[5][0] = list(extra_obj_start_poses[5][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[5][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[5][0][2] += 0.04
        # extra_obj_start_poses[5][0] = tuple(extra_obj_start_poses[5][0])  # Convert back to tuple
        # extra_obj_start_poses[5] = tuple(extra_obj_start_poses[5])  # Convert back to tuple

        multiply(STOOL_CENTER, ([-0.396, -0.3950, 0.018], [0, 0, 0, 1]))
        # target_obj_start_pose = multiply(target_obj_start_pose, [[-0.2, 0.1, -0.01], [0.0, 0.0, 0.0, 1.0]])
        # target_obj_goal_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.396, -0.3950, 0.018], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.40, 0.0, 0.03], [0.0, 0.0, 0.0, 1.0]])
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
        
        
    elif selected_obj_id == 8:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.03, -0.03, 0.07], [-1, 0, 0, 0])
        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, 0.04], [0.0, 0.0, 0.0, 1.0]])
    else: # selected_obj_id == 0:
        # target_obj_start_pose = multiply(STOOL_CENTER, [[-0.50, -0.40, -0.04000000077-0.02], [0.0, 0.0, 0.0, 1.0]])
        # target_obj_goal_pose = multiply(STOOL_CENTER, [[-0.50, -0.40, -0.010999999225139618-0.02], [0.0, 0.0, 0.0, 1.0]])
        
        STOOL_PEG_T_TIP_DEFAULT = ([-0.03, 0.032, 0.06], [-1, 0, 0, 0])
        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        extra_obj_start_poses = [extra_obj_start_poses[0], 
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                extra_obj_start_poses[3],
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.20, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.20, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
        

    ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, hole_obj_goal_pose, extra_obj_start_poses)
    PEG_T_TIP = STOOL_PEG_T_TIP_DEFAULT
    HOLE_T_PEG_GOAL = multiply(invert(hole_obj_goal_pose), target_obj_goal_pose)

    # elif version == 'v2':  # peg start on the fixture
    WORLD_T_PEG_START = target_obj_start_pose  # multiply(([0.0, 0.0, 0.10], [0, 0, 0, 1]), BEAM_WORLD_T_FIXTURE_POSE)
    # ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, target_obj_usd_path, hole_obj_usd_path)

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)
    # print('\n'*50, '************************', 'hole_obj_goal_pose: ', hole_obj_goal_pose, '************************', '\n'*50)
    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME=hole_obj_usd_path,
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=hole_obj_goal_pose,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=extra_obj_start_poses,
                      allow_peg_rotation=True,
                      USE_FIXED_RIGID_ATTACHMENT=False, # True,
                      allow_gripper_status=False,
                      TASK_NAME='StoolScriptedGrasp-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      USE_FT_SENSOR=True,
                      USE_CAMERA=False,
                      WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                      peg_goal_weights=[1, 1, 1, 1, 1, 1],
                      DIFFICULTY_LEVEL='easy',
                      relative_pose_obs=True,
                      selected_obj_id=selected_obj_id,
                      POSE_AFTER=STOOL_SAFE_POSE_AFTER['StoolScriptedGrasp'],
                      )

def StoolScriptedGraspHorizontal(selected_obj_id=1, desired_tool_T_world=None):
    selected_obj_id = STOOL_SELECTED_OBJ_ID # 8 # 6 # 4
    SHOW_GOAL = False # True
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses,
     hole_obj_name, hole_obj_usd_path, hole_obj_start_pose, hole_obj_goal_pose, hole_peg_attachment_prim,) = stool_split_obj_dict(
        STOOL_OBJECT_DICT, selected_obj_id=selected_obj_id, )
    
    if selected_obj_id == 1:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.025, 0.029, 0.03], [-1, 0, 0, 0])
        target_obj_start_pose = multiply(target_obj_start_pose, [[-0.1, 0.1, -0.01], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        # hole_obj_goal_pose = multiply(BEAM_CENTER, [[0.0885, 0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])
        # extra_obj_start_poses = [multiply(BEAM_CENTER, [[0.0885, -0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                      multiply(BEAM_CENTER, [[0.1885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                      multiply(BEAM_CENTER, [[-0.0885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])]
        extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[0][0][2] += 0.04
        extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        extra_obj_start_poses[4] = list(extra_obj_start_poses[4])  # Convert tuple to list for mutability
        extra_obj_start_poses[4][0] = list(extra_obj_start_poses[4][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[4][0][0] += 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[4][0][2] -= 0.04
        extra_obj_start_poses[4][0] = tuple(extra_obj_start_poses[4][0])  # Convert back to tuple
        extra_obj_start_poses[4] = tuple(extra_obj_start_poses[4])  # Convert back to tuple

    elif selected_obj_id == 3:
        STOOL_PEG_T_TIP_DEFAULT = ([0.03, -0.032, 0.03], [-1, 0, 0, 0])
        target_obj_start_pose = multiply(target_obj_start_pose, [[-0.1, 0.1, -0.01], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        # Modify the first element of the first tuple in the first sublist
        extra_obj_start_poses[2] = list(extra_obj_start_poses[2])  # Convert tuple to list for mutability
        extra_obj_start_poses[2][0] = list(extra_obj_start_poses[2][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[2][0][0] -= 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[2][0][2] -= 0.04
        extra_obj_start_poses[2][0] = tuple(extra_obj_start_poses[2][0])  # Convert back to tuple
        extra_obj_start_poses[2] = tuple(extra_obj_start_poses[2])  # Convert back to tuple
        extra_obj_start_poses[6] = list(extra_obj_start_poses[6])  # Convert tuple to list for mutability
        extra_obj_start_poses[6][0] = list(extra_obj_start_poses[6][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[6][0][0] += 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[6][0][2] += 0.04
        extra_obj_start_poses[6][0] = tuple(extra_obj_start_poses[6][0])  # Convert back to tuple
        extra_obj_start_poses[6] = tuple(extra_obj_start_poses[6])  # Convert back to tuple
    elif selected_obj_id == 4:
        STOOL_PEG_T_TIP_DEFAULT = ([0.035, 0.03, 0.06], [-1, 0, 0, 0])
        target_obj_start_pose = multiply(target_obj_start_pose, [[-0.1, -0.1, -0.08], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        hole_obj_goal_pose = multiply(hole_obj_goal_pose, [[-0., -0.2, -0.0], [0.0, 0.0, 0.0, 1.0]])
    elif selected_obj_id == 5:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.025, -0.028, 0.04], [-1, 0, 0, 0])
        extra_obj_start_poses[3] = list(extra_obj_start_poses[3])  # Convert tuple to list for mutability
        extra_obj_start_poses[3][0] = list(extra_obj_start_poses[3][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[3][0][0] -= 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[3][0][2] += 0.04
        extra_obj_start_poses[3][0] = tuple(extra_obj_start_poses[3][0])  # Convert back to tuple
        extra_obj_start_poses[3] = tuple(extra_obj_start_poses[3])  # Convert back to tuple
        extra_obj_start_poses[6] = list(extra_obj_start_poses[6])  # Convert tuple to list for mutability
        extra_obj_start_poses[6][0] = list(extra_obj_start_poses[6][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[6][0][0] += 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[6][0][2] += 0.04
        extra_obj_start_poses[6][0] = tuple(extra_obj_start_poses[6][0])  # Convert back to tuple
        extra_obj_start_poses[6] = tuple(extra_obj_start_poses[6])  # Convert back to tuple

        target_obj_start_pose = multiply(target_obj_start_pose, [[-0.2, 0.1, -0.01], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
    elif selected_obj_id == 6:
        STOOL_PEG_T_TIP_DEFAULT = ([0.035, -0.03, 0.06], [-1, 0, 0, 0])
        extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[0][0][2] += 0.04
        extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        target_obj_start_pose = multiply(target_obj_start_pose, [[-0.2, 0.1, -0.08], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
    elif selected_obj_id == 7:
        STOOL_PEG_T_TIP_DEFAULT = ([0.027, 0.03, 0.04], [-1, 0, 0, 0])
        # STOOL_PEG_T_TIP_DEFAULT = ([0.0, 0.0, 0.14], [-1, 0, 0, 0])
        STOOL_PEG_T_TIP_DEFAULT = multiply(STOOL_PEG_T_TIP_DEFAULT, ([0, 0, 0], [0, 0, 1, 0]))
        extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[0][0][2] += 0.04
        extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        extra_obj_start_poses[5] = list(extra_obj_start_poses[5])  # Convert tuple to list for mutability
        extra_obj_start_poses[5][0] = list(extra_obj_start_poses[5][0])  # Convert inner tuple to list for mutability
        extra_obj_start_poses[5][0][0] += 0.1  # Add 0.1 to the first element
        extra_obj_start_poses[5][0][2] += 0.04
        extra_obj_start_poses[5][0] = tuple(extra_obj_start_poses[5][0])  # Convert back to tuple
        extra_obj_start_poses[5] = tuple(extra_obj_start_poses[5])  # Convert back to tuple

        # for i, each_extra_obj_start_poses in enumerate(extra_obj_start_poses):
        #     each_extra_obj_start_poses = list(each_extra_obj_start_poses)
        #     each_extra_obj_start_poses[0] = list(each_extra_obj_start_poses[0])  # Convert inner tuple to list for mutability
        #     each_extra_obj_start_poses[0][0] += 0.4  # Add 0.1 to the first element
        #     # each_extra_obj_start_poses[0][2] += 0.04
        #     each_extra_obj_start_poses[0] = tuple(each_extra_obj_start_poses[0])  # Convert back to 
        #     each_extra_obj_start_poses = tuple(each_extra_obj_start_poses)
        #     extra_obj_start_poses[i] = each_extra_obj_start_poses
        # target_obj_start_pose = multiply(target_obj_start_pose, [[0.4, 0.1, -0.01], [0.0, 0.0, 0.0, 1.0]])


        # target_obj_start_pose = multiply(target_obj_start_pose, [[-0.2, 0.1, -0.01], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(target_obj_start_pose, [[0, 0, 0.06], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(target_obj_start_pose, [[0, 0.1, 0.05],  [-0.6427876, 0, 0, 0.7660444]])  #[-80, 0, 0]
    elif selected_obj_id == 8:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.03, -0.03, 0.07], [-1, 0, 0, 0])
        target_obj_start_pose = multiply(target_obj_start_pose, [[-0.2, 0.1, -0.08], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
    else: # selected_obj_id == 0:
        target_obj_start_pose = multiply(STOOL_CENTER, [[-0.50, -0.40, -0.04000000077-0.02], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(STOOL_CENTER, [[-0.50, -0.40, -0.010999999225139618-0.02], [0.0, 0.0, 0.0, 1.0]])
        STOOL_PEG_T_TIP_DEFAULT = ([-0.03, 0.032, 0.06], [-1, 0, 0, 0])
    ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, hole_obj_goal_pose, extra_obj_start_poses)
    PEG_T_TIP = STOOL_PEG_T_TIP_DEFAULT
    HOLE_T_PEG_GOAL = multiply(invert(hole_obj_goal_pose), target_obj_goal_pose)

    # elif version == 'v2':  # peg start on the fixture
    WORLD_T_PEG_START = target_obj_start_pose  # multiply(([0.0, 0.0, 0.10], [0, 0, 0, 1]), BEAM_WORLD_T_FIXTURE_POSE)
    # ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, target_obj_usd_path, hole_obj_usd_path)

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)
    # print('\n'*50, '************************', 'hole_obj_goal_pose: ', hole_obj_goal_pose, '************************', '\n'*50)
    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME=hole_obj_usd_path,
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=hole_obj_goal_pose,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=extra_obj_start_poses,
                      allow_peg_rotation=True,
                      USE_FIXED_RIGID_ATTACHMENT=False, # True,
                      allow_gripper_status=False,
                      TASK_NAME='StoolScriptedGraspHorizontal-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      USE_FT_SENSOR=True,
                      USE_CAMERA=False,
                      WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                      peg_goal_weights=[1, 1, 1, 1, 1, 1],
                      DIFFICULTY_LEVEL='easy',
                      relative_pose_obs=True,
                      selected_obj_id=selected_obj_id,
                      POSE_AFTER=STOOL_SAFE_POSE_AFTER['StoolScriptedGraspHorizontal'],
                      )

def StoolScriptedPlace(selected_obj_id=1, desired_tool_T_world=None):
    selected_obj_id = STOOL_SELECTED_OBJ_ID # 8 # 6 # 4
    SHOW_GOAL = False # True
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses,
     hole_obj_name, hole_obj_usd_path, hole_obj_start_pose, hole_obj_goal_pose, hole_peg_attachment_prim,) = stool_split_obj_dict(
        STOOL_OBJECT_DICT, selected_obj_id=selected_obj_id, )
    
    if selected_obj_id == 1:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.025, 0.029, 0.03], [-1, 0, 0, 0])
        target_obj_goal_pose = multiply(target_obj_start_pose, [[-0.1, 0.1, -0.01], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(target_obj_goal_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] += 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        # extra_obj_start_poses[4] = list(extra_obj_start_poses[4])  # Convert tuple to list for mutability
        # extra_obj_start_poses[4][0] = list(extra_obj_start_poses[4][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[4][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[4][0][2] -= 0.04
        # extra_obj_start_poses[4][0] = tuple(extra_obj_start_poses[4][0])  # Convert back to tuple
        # extra_obj_start_poses[4] = tuple(extra_obj_start_poses[4])  # Convert back to tuple
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.455, -0.403, 0.02], [0, 0, 0, 1])), [[0.35, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]), 
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.396, -0.3950, 0.018], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]

    elif selected_obj_id == 3:
        STOOL_PEG_T_TIP_DEFAULT = ([0.03, -0.032, 0.03], [-1, 0, 0, 0])
        target_obj_goal_pose = multiply(target_obj_start_pose, [[-0.1, 0.1, -0.01], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(target_obj_goal_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        # Modify the first element of the first tuple in the first sublist
        # extra_obj_start_poses[2] = list(extra_obj_start_poses[2])  # Convert tuple to list for mutability
        # extra_obj_start_poses[2][0] = list(extra_obj_start_poses[2][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[2][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[2][0][2] -= 0.04
        # extra_obj_start_poses[2][0] = tuple(extra_obj_start_poses[2][0])  # Convert back to tuple
        # extra_obj_start_poses[2] = tuple(extra_obj_start_poses[2])  # Convert back to tuple
        # extra_obj_start_poses[6] = list(extra_obj_start_poses[6])  # Convert tuple to list for mutability
        # extra_obj_start_poses[6][0] = list(extra_obj_start_poses[6][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[6][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[6][0][2] += 0.04
        # extra_obj_start_poses[6][0] = tuple(extra_obj_start_poses[6][0])  # Convert back to tuple
        # extra_obj_start_poses[6] = tuple(extra_obj_start_poses[6])  # Convert back to tuple
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.396, -0.3950, 0.018], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
    elif selected_obj_id == 4:
        STOOL_PEG_T_TIP_DEFAULT = ([0.035, 0.03, 0.06], [-1, 0, 0, 0])
        target_obj_goal_pose = multiply(target_obj_start_pose, [[-0.1, -0.1, -0.08], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(target_obj_goal_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        hole_obj_goal_pose = multiply(hole_obj_goal_pose, [[-0., -0.2, -0.0], [0.0, 0.0, 0.0, 1.0]])
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                extra_obj_start_poses[3],
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
    elif selected_obj_id == 5:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.025, -0.028, 0.04], [-1, 0, 0, 0])
        # extra_obj_start_poses[3] = list(extra_obj_start_poses[3])  # Convert tuple to list for mutability
        # extra_obj_start_poses[3][0] = list(extra_obj_start_poses[3][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[3][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[3][0][2] += 0.04
        # extra_obj_start_poses[3][0] = tuple(extra_obj_start_poses[3][0])  # Convert back to tuple
        # extra_obj_start_poses[3] = tuple(extra_obj_start_poses[3])  # Convert back to tuple
        # extra_obj_start_poses[6] = list(extra_obj_start_poses[6])  # Convert tuple to list for mutability
        # extra_obj_start_poses[6][0] = list(extra_obj_start_poses[6][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[6][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[6][0][2] += 0.04
        # extra_obj_start_poses[6][0] = tuple(extra_obj_start_poses[6][0])  # Convert back to tuple
        # extra_obj_start_poses[6] = tuple(extra_obj_start_poses[6])  # Convert back to tuple
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]

        target_obj_goal_pose = multiply(target_obj_start_pose, [[-0.2, 0.1, -0.01], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(target_obj_goal_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
    elif selected_obj_id == 6:
        STOOL_PEG_T_TIP_DEFAULT = ([0.035, -0.03, 0.06], [-1, 0, 0, 0])
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] += 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        extra_obj_start_poses = [extra_obj_start_poses[0], 
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                extra_obj_start_poses[3],
                                extra_obj_start_poses[4],
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
        target_obj_goal_pose = multiply(target_obj_start_pose, [[-0.2, 0.1, -0.08], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(target_obj_goal_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
    elif selected_obj_id == 7:
        STOOL_PEG_T_TIP_DEFAULT = ([0.027, 0.03, 0.04], [-1, 0, 0, 0])
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] += 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        # extra_obj_start_poses[5] = list(extra_obj_start_poses[5])  # Convert tuple to list for mutability
        # extra_obj_start_poses[5][0] = list(extra_obj_start_poses[5][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[5][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[5][0][2] += 0.04
        # extra_obj_start_poses[5][0] = tuple(extra_obj_start_poses[5][0])  # Convert back to tuple
        # extra_obj_start_poses[5] = tuple(extra_obj_start_poses[5])  # Convert back to tuple
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
        target_obj_goal_pose = multiply(target_obj_start_pose, [[-0.2, 0.1, -0.01], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(target_obj_goal_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
    elif selected_obj_id == 8:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.03, -0.03, 0.07], [-1, 0, 0, 0])
        target_obj_goal_pose = multiply(target_obj_start_pose, [[-0.2, 0.1, -0.08], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(target_obj_goal_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
    else: # selected_obj_id == 0:
        target_obj_goal_pose = multiply(STOOL_CENTER, [[-0.50, -0.40, -0.04000000077-0.02], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(STOOL_CENTER, [[-0.50, -0.40, -0.010999999225139618-0.02], [0.0, 0.0, 0.0, 1.0]])
        STOOL_PEG_T_TIP_DEFAULT = ([-0.03, 0.032, 0.06], [-1, 0, 0, 0])
        extra_obj_start_poses = [extra_obj_start_poses[0], 
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                extra_obj_start_poses[3],
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.20, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.20, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
    ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, hole_obj_goal_pose, extra_obj_start_poses)
    PEG_T_TIP = STOOL_PEG_T_TIP_DEFAULT
    HOLE_T_PEG_GOAL = multiply(invert(hole_obj_goal_pose), target_obj_goal_pose)

    # elif version == 'v2':  # peg start on the fixture
    WORLD_T_PEG_START = target_obj_start_pose  # multiply(([0.0, 0.0, 0.10], [0, 0, 0, 1]), BEAM_WORLD_T_FIXTURE_POSE)
    # ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, target_obj_usd_path, hole_obj_usd_path)

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)
    # print('\n'*50, '************************', 'hole_obj_goal_pose: ', hole_obj_goal_pose, '************************', '\n'*50)
    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME=hole_obj_usd_path,
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=hole_obj_goal_pose,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=extra_obj_start_poses,
                      allow_peg_rotation=True,
                      USE_FIXED_RIGID_ATTACHMENT=False, # True,
                      allow_gripper_status=False,
                      TASK_NAME='StoolScriptedPlace-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      USE_FT_SENSOR=True,
                      USE_CAMERA=False,
                      WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                      peg_goal_weights=[1, 1, 1, 1, 1, 1],
                      DIFFICULTY_LEVEL='easy',
                      relative_pose_obs=True,
                      selected_obj_id=selected_obj_id,
                      POSE_AFTER=STOOL_SAFE_POSE_AFTER['StoolScriptedPlace'],
                      )

def StoolMoveToBoard(selected_obj_id=1, desired_tool_T_world=None):
    selected_obj_id = STOOL_SELECTED_OBJ_ID # 8 # 6 # 4
    SHOW_GOAL = False # True
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses,
     hole_obj_name, hole_obj_usd_path, hole_obj_start_pose, hole_obj_goal_pose, hole_peg_attachment_prim,) = stool_split_obj_dict(
        STOOL_OBJECT_DICT, selected_obj_id=selected_obj_id, )
    
    if selected_obj_id == 1:
        target_obj_goal_pose = copy.copy(target_obj_start_pose)
        STOOL_PEG_T_TIP_DEFAULT = ([-0.025, 0.029, 0.03], [-1, 0, 0, 0])
        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.405, -0.395, 0.018], [0, 0, 0, 1])), [[0.35, 0.0, 0.03], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        # hole_obj_goal_pose = multiply(BEAM_CENTER, [[0.0885, 0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])
        # extra_obj_start_poses = [multiply(BEAM_CENTER, [[0.0885, -0.1, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                      multiply(BEAM_CENTER, [[0.1885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]]), 
        #                      multiply(BEAM_CENTER, [[-0.0885, 0.0, 0.002999999225139618], [0.0, 0.0, 0.0, 1.0]])]
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] += 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        # extra_obj_start_poses[4] = list(extra_obj_start_poses[4])  # Convert tuple to list for mutability
        # extra_obj_start_poses[4][0] = list(extra_obj_start_poses[4][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[4][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[4][0][2] -= 0.04
        # extra_obj_start_poses[4][0] = tuple(extra_obj_start_poses[4][0])  # Convert back to tuple
        # extra_obj_start_poses[4] = tuple(extra_obj_start_poses[4])  # Convert back to tuple
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.455, -0.403, 0.02], [0, 0, 0, 1])), [[0.35, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]), 
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.396, -0.3950, 0.018], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]

    elif selected_obj_id == 3:
        STOOL_PEG_T_TIP_DEFAULT = ([0.03, -0.028, 0.03], [-1, 0, 0, 0])
        target_obj_goal_pose = multiply(copy.copy(target_obj_start_pose), [[0.0, 0., 0.02], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.455, -0.403, 0.02], [0, 0, 0, 1])), [[0.35, 0.0, 0.03], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        # Modify the first element of the first tuple in the first sublist
        # extra_obj_start_poses[2] = list(extra_obj_start_poses[2])  # Convert tuple to list for mutability
        # extra_obj_start_poses[2][0] = list(extra_obj_start_poses[2][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[2][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[2][0][2] -= 0.04
        # extra_obj_start_poses[2][0] = tuple(extra_obj_start_poses[2][0])  # Convert back to tuple
        # extra_obj_start_poses[2] = tuple(extra_obj_start_poses[2])  # Convert back to tuple
        # extra_obj_start_poses[6] = list(extra_obj_start_poses[6])  # Convert tuple to list for mutability
        # extra_obj_start_poses[6][0] = list(extra_obj_start_poses[6][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[6][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[6][0][2] += 0.04
        # extra_obj_start_poses[6][0] = tuple(extra_obj_start_poses[6][0])  # Convert back to tuple
        # extra_obj_start_poses[6] = tuple(extra_obj_start_poses[6])  # Convert back to tuple
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                # extra_obj_start_poses[2],
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.396, -0.3950, 0.018], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
    elif selected_obj_id == 4:
        STOOL_PEG_T_TIP_DEFAULT = ([0.035, 0.03, 0.06], [-1, 0, 0, 0])
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                extra_obj_start_poses[3],
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
        # hole_obj_goal_pose = multiply(hole_obj_goal_pose, [[-0., -0.2, -0.0], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(copy.copy(target_obj_start_pose), [[0.0, 0., 0.02], [0.0, 0.0, 0.0, 1.0]])
        # target_obj_start_pose = multiply(target_obj_start_pose, [[-0.2, 0.1, -0.08], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])

        # target_obj_goal_pose = multiply(copy.copy(target_obj_start_pose), [[0.0, 0., 0.02], [0.0, 0.0, 0.0, 1.0]])
        # target_obj_start_pose = multiply(target_obj_start_pose, [[-0.1, -0.1, -0.08], [0.0, 0.0, 0.0, 1.0]])
        # target_obj_start_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
    elif selected_obj_id == 5:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.025, -0.028, 0.03], [-1, 0, 0, 0])
        # extra_obj_start_poses[3] = list(extra_obj_start_poses[3])  # Convert tuple to list for mutability
        # extra_obj_start_poses[3][0] = list(extra_obj_start_poses[3][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[3][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[3][0][2] += 0.04
        # extra_obj_start_poses[3][0] = tuple(extra_obj_start_poses[3][0])  # Convert back to tuple
        # extra_obj_start_poses[3] = tuple(extra_obj_start_poses[3])  # Convert back to tuple
        # extra_obj_start_poses[6] = list(extra_obj_start_poses[6])  # Convert tuple to list for mutability
        # extra_obj_start_poses[6][0] = list(extra_obj_start_poses[6][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[6][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[6][0][2] += 0.04
        # extra_obj_start_poses[6][0] = tuple(extra_obj_start_poses[6][0])  # Convert back to tuple
        # extra_obj_start_poses[6] = tuple(extra_obj_start_poses[6])  # Convert back to tuple
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]

        target_obj_goal_pose = multiply(copy.copy(target_obj_start_pose), [[0.0, 0., 0.02], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), [[0.40, 0.0, 0.03], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
    elif selected_obj_id == 6:
        STOOL_PEG_T_TIP_DEFAULT = ([0.035, -0.03, 0.06], [-1, 0, 0, 0])
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] += 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        extra_obj_start_poses = [extra_obj_start_poses[0], 
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                extra_obj_start_poses[3],
                                extra_obj_start_poses[4],
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
        target_obj_goal_pose = multiply(copy.copy(target_obj_start_pose), [[0.0, 0., 0.02], [0.0, 0.0, 0.0, 1.0]])
        # target_obj_start_pose = multiply(target_obj_start_pose, [[-0.2, 0.1, -0.08], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.20, 0.0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
    elif selected_obj_id == 7:
        STOOL_PEG_T_TIP_DEFAULT = ([0.027, 0.03, 0.03], [-1, 0, 0, 0])
        # extra_obj_start_poses[0] = list(extra_obj_start_poses[0])  # Convert tuple to list for mutability
        # extra_obj_start_poses[0][0] = list(extra_obj_start_poses[0][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[0][0][0] -= 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[0][0][2] += 0.04
        # extra_obj_start_poses[0][0] = tuple(extra_obj_start_poses[0][0])  # Convert back to tuple
        # extra_obj_start_poses[0] = tuple(extra_obj_start_poses[0])  # Convert back to tuple
        # extra_obj_start_poses[5] = list(extra_obj_start_poses[5])  # Convert tuple to list for mutability
        # extra_obj_start_poses[5][0] = list(extra_obj_start_poses[5][0])  # Convert inner tuple to list for mutability
        # extra_obj_start_poses[5][0][0] += 0.1  # Add 0.1 to the first element
        # extra_obj_start_poses[5][0][2] += 0.04
        # extra_obj_start_poses[5][0] = tuple(extra_obj_start_poses[5][0])  # Convert back to tuple
        # extra_obj_start_poses[5] = tuple(extra_obj_start_poses[5])  # Convert back to tuple
        extra_obj_start_poses = [multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                multiply(multiply(STOOL_CENTER, ([-0.465, -0.465, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.344, -0.4050, 0.02], [0, 0, 0, 1])), [[0.40, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
        target_obj_goal_pose = multiply(copy.copy(target_obj_start_pose), [[0.0, 0., 0.02], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.396, -0.3950, 0.018], [0, 0, 0, 1])), [[0.40, 0.0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(target_obj_start_pose, [[0, 0, 0.04], [0.0, 0.0, 0.0, 1.0]])
    elif selected_obj_id == 8:
        STOOL_PEG_T_TIP_DEFAULT = ([-0.03, -0.03, 0.05], [-1, 0, 0, 0])
        target_obj_goal_pose = multiply(copy.copy(target_obj_start_pose), [[0.0, 0., 0.04], [0.0, 0.0, 0.0, 1.0]])
        # target_obj_start_pose = multiply(target_obj_start_pose, [[-0.2, 0.1, -0.08], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        target_obj_start_pose = multiply(target_obj_start_pose, [[0, 0, 0.05], [0.0, 0.0, 0.0, 1.0]])
    else: # selected_obj_id == 0:
        # new_target_obj_start_pose = copy.copy(target_obj_goal_pose) # multiply(BEAM_CENTER, [[-0.0885, -0.1, -0.04000000077], [0.0, 0.0, 0.0, 1.0]])
        target_obj_goal_pose = multiply(copy.copy(target_obj_start_pose), [[0.0, 0., 0.02], [0.0, 0.0, 0.0, 1.0]])
        # target_obj_start_pose = multiply(STOOL_CENTER, [[-0.50, -0.40, -0.010999999225139618-0.02], [0.0, 0.0, 0.0, 1.0]])
        STOOL_PEG_T_TIP_DEFAULT = ([-0.03, 0.032, 0.06], [-1, 0, 0, 0])
        target_obj_start_pose = multiply(multiply(STOOL_CENTER, ([-0.40, -0.40, 0.02], [0, 0, 0, 1])), [[0.15, 0.0, 0.04], [0.0, 0.0, 0.0, 1.0]])
        extra_obj_start_poses = [extra_obj_start_poses[0], 
                                extra_obj_start_poses[1], 
                                extra_obj_start_poses[2],
                                extra_obj_start_poses[3],
                                multiply(multiply(STOOL_CENTER, ([-0.404, -0.338, 0.015], [0, 0, 0, 1])), [[0.20, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                extra_obj_start_poses[5],
                                multiply(multiply(STOOL_CENTER, ([-0.338, -0.397, 0.02], [0, 0, 0, 1])), [[0.2, 0.0, -0.08], [0.0, 0.0, 0.0, 1.0]]),
                                ]
        
    # ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, hole_obj_goal_pose, extra_obj_start_poses)
    PEG_T_TIP = STOOL_PEG_T_TIP_DEFAULT
    HOLE_T_PEG_GOAL = multiply(invert(hole_obj_goal_pose), target_obj_goal_pose)

    # elif version == 'v2':  # peg start on the fixture
    WORLD_T_PEG_START = target_obj_start_pose  # multiply(([0.0, 0.0, 0.10], [0, 0, 0, 1]), BEAM_WORLD_T_FIXTURE_POSE)
    # ic(target_obj_start_pose, target_obj_goal_pose, selected_obj_id, target_obj_usd_path, hole_obj_usd_path)

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)
    # print('\n'*50, '************************', 'hole_obj_goal_pose: ', hole_obj_goal_pose, '************************', '\n'*50)
    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME=hole_obj_usd_path,
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=hole_obj_goal_pose,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=extra_obj_start_poses,
                      allow_peg_rotation=True,
                      USE_FIXED_RIGID_ATTACHMENT=False, # True,
                      allow_gripper_status=False,
                      TASK_NAME='StoolMoveToBoard-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      USE_FT_SENSOR=True,
                      USE_CAMERA=False,
                      WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                      peg_goal_weights=[1, 1, 1, 1, 1, 1],
                      DIFFICULTY_LEVEL='easy',
                      relative_pose_obs=True,
                      selected_obj_id=selected_obj_id,
                      POSE_AFTER=STOOL_SAFE_POSE_AFTER['StoolMoveToBoard'],
                      )

def AssemblyRegrasp(selected_obj_id=0, desired_tool_T_world=None):
    SHOW_GOAL = False  # True  # False
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses) = split_obj_dict(
        OBJECT_DICT, selected_obj_id, )
    # PEG_T_TIP = ([0,0,-0.02], [-1, 0, 0, 0])
    # HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0,0,0,1])

    # WORLD_T_HOLE_START = ([0, 0, 0.025], [0, 0, 0, 1])
    # WORLD_T_PEG_START = multiply(([0.03, 0, 0.1], [0,0,0,1]), WORLD_T_FIXTURE_POSE)
    # FIXTURE_T_PEG_START_ROT = [0, 0.7071068, 0, 0.7071068]
    # FIXTURE_T_PEG_ROT = [0, 0.7071068, 0, 0.7071068]
    # WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, ([0.01, 0, 0.03], FIXTURE_T_PEG_ROT))
    # FIXTURE_T_PEG_START = ([0.01, 0, 0.008], [0, -0.4226183, 0, 0.9063078])
    FIXTURE_T_PEG_START = ([0.01, 0, 0.001], [0, -0.4383711, 0, 0.898794])  # (0, -52, 0)
    WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_START)
    # WORLD_T_PEG_START = ([0.0, 0.8, 0.07], [0, -0.707, -0.707, 0.])
    # world_T_peg:  IPose(pos=tensor([[0.0027, 0.7552, 0.0628],
    #         [0.0011, 0.7566, 0.0628]], device='cuda:0'), quat=tensor([[-1.4252e-04, -7.0689e-01, -7.0732e-01, -7.9733e-06],
    #         [ 1.0723e-06, -6.8372e-01, -7.2975e-01,  6.6055e-06]], device='cuda:0'))
    FIXTURE_T_PEG_GOAL = ([0.0, 0.0, 0.16], [0, 0, 0, 1])
    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_GOAL)


    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME=home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = FIXTURE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                      PEG_T_TIP = PEG_T_TIP_DEFAULT,
                      num_robots = 1,
                      EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd",] + extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE,] + extra_obj_start_poses,
                      allow_peg_rotation=True,
                      TASK_NAME='AssemblyRegrasp-v0',
                      WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                      SHOW_GOAL=SHOW_GOAL,
                      better_tool_frame=True,
                      DIFFICULTY_LEVEL='easy',
                      selected_obj_id=selected_obj_id,
                      )

def AssemblyScriptedRegrasp(selected_obj_id=0, desired_tool_T_world=None):
    SHOW_GOAL = False  # True  # False
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses) = split_obj_dict(
        OBJECT_DICT, selected_obj_id, )
    if target_obj_name in ['Medium_Short_Oval_JeansBlue']:
        PEG_T_TIP = ([-0.0025, 0.002, -0.013], [-1, 0, 0, 0])
    elif target_obj_name in ['Medium_Short_SquareCircle_Red',]:
        PEG_T_TIP = ([-0.0026, 0.002, -0.009], [-1, 0, 0, 0])
    elif target_obj_name in ['Medium_Short_DoubleSquare_Purple']:
        PEG_T_TIP = ([-0.0025, 0.002, -0.007], [-1, 0, 0, 0])
    elif target_obj_name in ['Medium_Short_Arch_Yellow',]:
        PEG_T_TIP = ([-0.0026, 0.002, -0.020], [-1, 0, 0, 0])
    else:
        PEG_T_TIP = ([0,0.002,-0.013], [-1, 0, 0, 0])
    # HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0,0,0,1])

    # WORLD_T_HOLE_START = ([0, 0, 0.025], [0, 0, 0, 1])
    # WORLD_T_PEG_START = multiply(([0.03, 0, 0.1], [0,0,0,1]), WORLD_T_FIXTURE_POSE)
    # FIXTURE_T_PEG_START_ROT = [0, 0.7071068, 0, 0.7071068]
    # FIXTURE_T_PEG_START = ([0.01, 0, 0.00], [0, -0.4226183, 0, 0.9063078])  # (0, -50, 0)
    FIXTURE_T_PEG_START = ([0.01, 0, 0.001], [0, -0.4539905, 0, 0.8910065])  # (0, -54, 0)
    WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_START)
    # WORLD_T_PEG_START = ([0.0, 0.8, 0.07], [0, -0.707, -0.707, 0.])
    # world_T_peg:  IPose(pos=tensor([[0.0027, 0.7552, 0.0628],
    #         [0.0011, 0.7566, 0.0628]], device='cuda:0'), quat=tensor([[-1.4252e-04, -7.0689e-01, -7.0732e-01, -7.9733e-06],
    #         [ 1.0723e-06, -6.8372e-01, -7.2975e-01,  6.6055e-06]], device='cuda:0'))
    FIXTURE_T_PEG_GOAL = ([0.0, 0.0, 0.16], [0, 0, 0, 1])
    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_GOAL)


    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME=home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = FIXTURE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd",] + extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE,] + extra_obj_start_poses,
                      allow_peg_rotation=True,
                      TASK_NAME='AssemblyScriptedRegrasp-v0',
                      WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                      SHOW_GOAL=SHOW_GOAL,
                      better_tool_frame=True,
                      DIFFICULTY_LEVEL='easy',
                      selected_obj_id=selected_obj_id,
                      )

def AssemblyGrasp(selected_obj_id=0, desired_tool_T_world=None):
    SHOW_GOAL = False  # True
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses) = split_obj_dict(
        OBJECT_DICT, selected_obj_id, )
    # PEG_T_TIP = ([0,0,-0.02], [-1, 0, 0, 0])
    # HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0,0,0,1])

    # WORLD_T_HOLE_START = ([0, 0, 0.025], [0, 0, 0, 1])
    # WORLD_T_PEG_START = multiply(([0.03, 0, 0.1], [0,0,0,1]), WORLD_T_FIXTURE_POSE)
    # WORLD_T_PEG_START = ([0.3, 0.6, 0.1], [0, 0, 0, 1])
    WORLD_T_PEG_START = ([0.6, 0.2, 0.05], [0, 0, 1, 0])
    FIXTURE_T_PEG_START = multiply(invert(WORLD_T_FIXTURE_POSE), WORLD_T_PEG_START)
    WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_START)
    WORLD_T_PEG_GOAL = ([0.6, 0.2, 0.15], [0, 0, 1, 0])
    FIXTURE_T_PEG_GOAL = multiply(invert(WORLD_T_FIXTURE_POSE), WORLD_T_PEG_GOAL)
    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_GOAL)


    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME= home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = FIXTURE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                      PEG_T_TIP = PEG_T_TIP_DEFAULT,
                      num_robots = 1,
                      EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd",] + extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE,] + extra_obj_start_poses,
                      allow_peg_rotation=True,
                      TASK_NAME='AssemblyGrasp-v0',
                      WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                      SHOW_GOAL=SHOW_GOAL,
                      DIFFICULTY_LEVEL='easy',
                      selected_obj_id=selected_obj_id,
                      )

def AssemblyScriptedGrasp(selected_obj_id=0, desired_tool_T_world=None):
    SHOW_GOAL = False  # True
    # PEG_T_TIP = ([0,0,-0.02], [-1, 0, 0, 0])
    # HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0,0,0,1])

    # WORLD_T_HOLE_START = ([0, 0, 0.025], [0, 0, 0, 1])
    # WORLD_T_PEG_START = multiply(([0.03, 0, 0.1], [0,0,0,1]), WORLD_T_FIXTURE_POSE)
    # WORLD_T_PEG_START = ([0.3, 0.6, 0.1], [0, 0, 0, 1])
    # if selected_obj_id is not None:
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses) = split_obj_dict(OBJECT_DICT, selected_obj_id)

    WORLD_T_PEG_START = ([0.6, 0.2, 0.05], [0, 0, 1, 0])

    if target_obj_name in ['Medium_Short_SquareCircle_Red', 'Medium_Short_3Prong_JeansRed', 'Medium_Short_DoubleSquare_Purple']:
        WORLD_T_PEG_START = multiply(WORLD_T_PEG_START, ([0., 0., 0.0], [0, 1, 0, 0]))
        PEG_T_TIP = multiply(PEG_T_TIP_DEFAULT, ([0., 0., 0.0], [0, 1, 0, 0]))
    else:
        PEG_T_TIP = PEG_T_TIP_DEFAULT
    target_obj_start_pose = WORLD_T_PEG_START
    approach = ([0., 0., 0.1], [0, 0, 0, 1])
    WORLD_T_PEG_GOAL = multiply(approach, target_obj_start_pose)  # WORLD_T_PEG_GOAL = ([0.6, 0.2, 0.15], [0, 0, 1, 0])
    FIXTURE_T_PEG_GOAL = multiply(invert(WORLD_T_FIXTURE_POSE), WORLD_T_PEG_GOAL)
        # print('\n\n\n\n\n\n\n\n\n ----------------- extra_obj_usd_paths: ', extra_obj_usd_paths, extra_obj_start_poses, target_obj_usd_path, target_obj_start_pose)
        # print('\n\n\n\n\n\n\n\n\n ----------------- target_obj_start_pose: ', target_obj_usd_path, target_obj_start_pose)
    # else:
    #     WORLD_T_PEG_START = ([0.6, 0.2, 0.05], [0, 0, 1, 0])
    #     FIXTURE_T_PEG_START = multiply(invert(WORLD_T_FIXTURE_POSE), WORLD_T_PEG_START)
    #     WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_START)
    #     # WORLD_T_PEG_GOAL = ([0.6, 0.2, 0.15], [0, 0, 1, 0])
    #     # FIXTURE_T_PEG_GOAL = multiply(invert(WORLD_T_FIXTURE_POSE), WORLD_T_PEG_GOAL)
    #
    #     target_obj_usd_path = home_dir + "taskboard/fmb_example/Medium_Short_Hexagon_Green_Updated.usd"
    #     target_obj_start_pose = WORLD_T_PEG_START
    #     approach = ([0., 0., 0.1], [0, 0, 0, 1])
    #     WORLD_T_PEG_GOAL = multiply(approach, target_obj_start_pose)
    #     FIXTURE_T_PEG_GOAL = multiply(invert(WORLD_T_FIXTURE_POSE), WORLD_T_PEG_GOAL)
    #     extra_obj_usd_paths = [home_dir + "taskboard/fmb_example/Medium_Short_Oval_JeansBlue_Updated.usd",]
    #     extra_obj_start_poses = [MEDIUM_SHORT_OVAL_JEANSBLUE_POSE]

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_GOAL)
        target_obj_start_pose = WORLD_T_PEG_START

    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM=target_peg_attachment_prim,
                      HOLE_ASSET_NAME=home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = FIXTURE_T_PEG_GOAL,
                      WORLD_T_PEG_START=target_obj_start_pose,
                      WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd",] + extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE] + extra_obj_start_poses,
                      allow_peg_rotation=True,
                      allow_gripper_status=False,
                      TASK_NAME='AssemblyScriptedGrasp-v0',
                      WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                      SHOW_GOAL=SHOW_GOAL,
                      DIFFICULTY_LEVEL='easy',
                      WORLD_T_PEG_PICK=target_obj_start_pose,
                      selected_obj_id=selected_obj_id,
                      desired_tool_T_world=desired_tool_T_world,
                      POSE_AFTER=SAFE_POSE_AFTER['AssemblyScriptedGrasp'],
                      )

def AssemblyScriptedGraspHorizontal(selected_obj_id=0, desired_tool_T_world=None):
    SHOW_GOAL = False  # True
    if selected_obj_id is not None:
        PEG_T_TIP_HORIZONTAL = multiply(([0, 0.05, 0.04], [-0.7071068, 0, 0, 0.7071068]),
                                ([0, 0, 0], [0, 0, 1, 0]))  # Good
        (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
         extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses) = split_obj_dict(
            OBJECT_DICT, selected_obj_id, obj_style='horizontal')

        WORLD_T_PEG_START = ([0.6, 0.05, 0.03], [0.7071068, 0, 0, -0.7071068])

        if target_obj_name in ['Medium_Short_SquareCircle_Red', 'Medium_Short_3Prong_JeansRed',
                               'Medium_Short_DoubleSquare_Purple']:
            WORLD_T_PEG_START = multiply(WORLD_T_PEG_START, ([0., 0., 0.0], [0, 1, 0, 0]))
            PEG_T_TIP_HORIZONTAL = ([0, 0.05, -0.04], [-0.7071068, 0, 0, 0.7071068])
                                # ([0, 0, 0], [0, 0, 1, 0])) 
            # PEG_T_TIP_HORIZONTAL = multiply(PEG_T_TIP_HORIZONTAL, ([0., 0., 0.0], [0, 0, 1, 0]))
        target_obj_start_pose = WORLD_T_PEG_START
        approach = ([0., 0., 0.1], [0, 0, 0, 1])
        WORLD_T_PEG_GOAL = multiply(approach, target_obj_start_pose)
        FIXTURE_T_PEG_GOAL = multiply(invert(WORLD_T_FIXTURE_POSE), WORLD_T_PEG_GOAL)
        # print('\n\n\n\n\n\n\n\n\n ----------------- extra_obj_usd_paths: ', extra_obj_usd_paths, extra_obj_start_poses, target_obj_usd_path, target_obj_start_pose)
        # print('\n\n\n\n\n\n\n\n\n ----------------- target_obj_start_pose: ', target_obj_usd_path, target_obj_start_pose)
    else:
        PEG_T_TIP_HORIZONTAL = multiply(([0, 0.05, 0.04], [-0.7071068, 0, 0, 0.7071068]),
                                        ([0, 0, 0], [0, 0, 1, 0]))  # Good
        # HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0,0,0,1])

        # WORLD_T_HOLE_START = ([0, 0, 0.025], [0, 0, 0, 1])
        # WORLD_T_PEG_START = multiply(([0.03, 0, 0.1], [0,0,0,1]), WORLD_T_FIXTURE_POSE)
        # WORLD_T_PEG_START = ([0.3, 0.6, 0.1], [0, 0, 0, 1])
        # WORLD_T_PEG_START = ([0.6, 0.2, 0.05], [0, -0.4395489, 0.8790978, -0.1843469])
        # WORLD_T_PEG_START = ([0.6, 0.05, 0.03], [0.7071068, 0, 0, -0.7071068])  # Good for ARAAS --enable-hardware
        WORLD_T_PEG_START = ([0.6, 0.05, 0.05], [0.7071068, 0, 0, -0.7071068]) # Good for IsaacLab
        FIXTURE_T_PEG_START = multiply(invert(WORLD_T_FIXTURE_POSE), WORLD_T_PEG_START)
        WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_START)
        # WORLD_T_PEG_GOAL = ([0.6, 0.2, 0.15], [0, -0.4395489, 0.8790978, -0.1843469])
        WORLD_T_PEG_GOAL = ([0.6, 0.05, 0.15], [0.7071068, 0, 0, -0.7071068])
        FIXTURE_T_PEG_GOAL = multiply(invert(WORLD_T_FIXTURE_POSE), WORLD_T_PEG_GOAL)
        target_obj_usd_path = home_dir + "taskboard/fmb_example/Medium_Short_Hexagon_Green_Updated.usd"
        target_obj_start_pose = WORLD_T_PEG_START
        extra_obj_usd_paths = [home_dir + "taskboard/fmb_example/Medium_Short_Oval_JeansBlue_Updated.usd",]
        extra_obj_start_poses = [MEDIUM_SHORT_OVAL_JEANSBLUE_POSE]

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_GOAL)
        target_obj_start_pose = WORLD_T_PEG_START

    return TaskConfig(PEG_ASSET_NAME = target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM = target_peg_attachment_prim,
                      HOLE_ASSET_NAME = home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = FIXTURE_T_PEG_GOAL,
                      WORLD_T_PEG_START=target_obj_start_pose,
                      WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                      PEG_T_TIP = PEG_T_TIP_HORIZONTAL,
                      num_robots = 1,
                      EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd",] + extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE] + extra_obj_start_poses,
                      allow_peg_rotation=True,
                      allow_gripper_status=False,
                      TASK_NAME='AssemblyScriptedGraspHorizontal-v0',
                      WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                      SHOW_GOAL=SHOW_GOAL,
                      DIFFICULTY_LEVEL='easy',
                      WORLD_T_PEG_PICK=target_obj_start_pose,
                      selected_obj_id=selected_obj_id,
                      desired_tool_T_world=desired_tool_T_world,
                      POSE_AFTER=SAFE_POSE_AFTER['AssemblyScriptedGraspHorizontal'],
                      )

def AssemblyPlace(selected_obj_id=0, desired_tool_T_world=None):
    '''
    place on the fixture
    Returns:

    '''
    SHOW_GOAL = False # True
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses) = split_obj_dict(
        OBJECT_DICT, selected_obj_id, )
    # PEG_T_TIP = ([0,0,-0.02], [-1, 0, 0, 0])
    # Good 1
    FIXTURE_T_PEG_GOAL = ([0.01, 0, 0.03], [0, -0.7071068, 0, 0.7071068])
    # Good 2
    # FIXTURE_T_PEG_GOAL = ([0.01, 0, 0.008], [ 0, -0.4226183, 0, 0.9063078 ])  # angle: [0, -50, 0]
    FIXTURE_T_PEG_GOAL = ([0.01, 0, 0.001], [0, -0.4539905, 0, 0.8910065])  # (0, -52, 0)
    # print('WORLD_T_PEG_GOAL: ', WORLD_T_PEG_GOAL)  # [(0, 0.8, 0.16) (0.5, 0.5, -0.5, 0.5)]
    # FIXTURE_POSE = ([0.0, 0.8, 0.06], [0, 0, 0, 1])

    # WORLD_T_HOLE_START = ([0, 0, 0.025], [0, 0, 0, 1])
    # WORLD_T_PEG_START = multiply(([0, 0, 0.1], [0,0,0,1]), multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_GOAL))
    WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, ([0.0, 0.0, 0.16], [0, 0, 0, 1]))
    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_GOAL)

    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM=target_peg_attachment_prim,
                      HOLE_ASSET_NAME=home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                      HOLE_TYPE = "peripheral",
                      PEG_GOAL = FIXTURE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                      PEG_T_TIP = PEG_T_TIP_DEFAULT,
                      num_robots = 1,
                      EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd",] + extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE,] + extra_obj_start_poses,
                      WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                      allow_peg_rotation=True,
                      TASK_NAME='AssemblyPlace-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      DIFFICULTY_LEVEL='easy',
                      selected_obj_id=selected_obj_id,
                      )

def AssemblyScriptedPlace(selected_obj_id=0, desired_tool_T_world=None):
    '''
    place on the fixture
    Returns:

    '''
    SHOW_GOAL = False # True
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses) = split_obj_dict(
        OBJECT_DICT, selected_obj_id, )
    PEG_T_TIP = ([0, 0.002, -0.013], [-1, 0, 0, 0])
    # HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0,0,0,1])
    # FIXTURE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0, 0, 0, 1])
    # Good 1
    # FIXTURE_T_PEG_GOAL = ([0.01, 0, 0.03], [0, -0.7071068, 0, 0.7071068])
    # Good 2
    # FIXTURE_T_PEG_GOAL = ([0.01, 0, 0.00], [ 0, -0.4226183, 0, 0.9063078 ])  # angle: [0, -50, 0]
    FIXTURE_T_PEG_GOAL = ([0.01, 0, 0.001], [0, -0.4539905, 0, 0.8910065])  # (0, -52, 0)
    # print('WORLD_T_PEG_GOAL: ', WORLD_T_PEG_GOAL)  # [(0, 0.8, 0.16) (0.5, 0.5, -0.5, 0.5)]

    # WORLD_T_HOLE_START = ([0, 0, 0.025], [0, 0, 0, 1])
    # WORLD_T_PEG_START = multiply(([0, 0, 0.1], [0,0,0,1]), multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_GOAL))
    WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, ([0.0, 0.0, 0.16], [0, 0, 0, 1]))
    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_GOAL)

    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM=target_peg_attachment_prim,
                      HOLE_ASSET_NAME=home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                      HOLE_TYPE="peripheral",
                      PEG_GOAL=FIXTURE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                      PEG_T_TIP = PEG_T_TIP,
                      num_robots = 1,
                      EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd",] + extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE,] + extra_obj_start_poses,
                      WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                      allow_peg_rotation=True,
                      allow_gripper_status=False,
                      TASK_NAME='AssemblyScriptedPlace-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      SUBTASK='ScriptedPlace',
                      better_tool_frame=True,
                      DIFFICULTY_LEVEL='easy',
                      selected_obj_id=selected_obj_id,
                      )

def AssemblyScriptedPlaceHorizontal(selected_obj_id=0, desired_tool_T_world=None):
    '''
    place on the fixture
    Returns:

    '''
    PEG_T_TIP_HORIZONTAL = multiply(multiply(([0, 0.05, 0.04], [-0.7071068, 0, 0, 0.7071068]),
                                    ([0, 0, 0], [0, 0, -0.7071068, 0.7071068])), ([0, 0, 0], [0, 0, 1, 0]))
    SHOW_GOAL = False # True
    (target_obj_name, target_obj_usd_path, target_obj_start_pose, target_obj_goal_pose, target_peg_attachment_prim,
     extra_obj_names, extra_obj_usd_paths, extra_obj_start_poses, extra_obj_goal_poses) = split_obj_dict(
        OBJECT_DICT, selected_obj_id, )
    # HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0,0,0,1])
    # FIXTURE_T_PEG_GOAL = multiply(multiply(([0.01, 0, 0.008], [ 0, -0.4226183, 0, 0.9063078 ]), ([0, 0, 0], [0, 0, 0.7071068, 0.7071068])),
    #                               ([0, 0, 0], [-0.7071068, 0, 0, 0.7071068]))  # # angle: [0, -50, 0]
    if int(selected_obj_id) in [1, 7]:
        # FIXTURE_T_PEG_GOAL = multiply(([0.038, 0.0, 0.004], [0, -0.5, 0, 0.8660254]), ([0, 0, 0], [0, 0, 0.7071068, 0.7071068]))  # # angle: [0, -60, 0]
        FIXTURE_T_PEG_GOAL = multiply(([0.042, 0.0, 0.008], [0, -0.5, 0, 0.8660254]), ([0, 0, 0], [0, 0, 0.7071068, 0.7071068]))  # # angle: [0, -60, 0]
    elif int(selected_obj_id) in [2, ]:
        FIXTURE_T_PEG_GOAL = multiply(([0.04, 0.0, 0.004], [0, -0.5, 0, 0.8660254]), ([0, 0, 0], [0, 0, 0.7071068, 0.7071068]))  # # angle: [0, -60, 0]
    else:
        FIXTURE_T_PEG_GOAL = multiply(([0.04, 0.0, 0.008], [0, -0.5, 0, 0.8660254]), ([0, 0, 0], [0, 0, 0.7071068, 0.7071068]))  # # angle: [0, -60, 0]
    # FIXTURE_T_PEG_GOAL = ([0.01, 0, 0.008], [0, -0.4226183, 0, 0.9063078])
    # print('WORLD_T_PEG_GOAL: ', WORLD_T_PEG_GOAL)  # [(0, 0.8, 0.16) (0.5, 0.5, -0.5, 0.5)]

    # WORLD_T_HOLE_START = ([0, 0, 0.025], [0, 0, 0, 1])
    # WORLD_T_PEG_START = multiply(([0, 0, 0.1], [0,0,0,1]), multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_GOAL))
    WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, multiply(([0.0, 0.0, 0.16], [0, 0.7071068, 0, -0.7071068]),
                                  ([0, 0, 0], [0, 0, 0.7071068, 0.7071068])))

    if SHOW_GOAL:
        WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_GOAL)

    return TaskConfig(PEG_ASSET_NAME=target_obj_usd_path,
                      PEG_ATTACHMENT_PRIM=target_peg_attachment_prim,
                      HOLE_ASSET_NAME=home_dir + "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd",
                      HOLE_TYPE="peripheral",
                      PEG_GOAL=FIXTURE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                      PEG_T_TIP = PEG_T_TIP_HORIZONTAL,
                      num_robots = 1,
                      EXTRA_PARTS=[home_dir + "taskboard/fmb_example/fixture_Updated.usd",] + extra_obj_usd_paths,
                      EXTRA_PARTS_STARTING_POSE=[WORLD_T_FIXTURE_POSE,] + extra_obj_start_poses,
                      WORLD_T_FIXTURE_START=WORLD_T_FIXTURE_POSE,
                      allow_peg_rotation=True,
                      allow_gripper_status=False,
                      TASK_NAME='AssemblyScriptedPlaceHorizontal-v0',
                      SHOW_GOAL=SHOW_GOAL,
                      SUBTASK='ScriptedPlace',
                      better_tool_frame=True,
                      DIFFICULTY_LEVEL='easy',
                      selected_obj_id=selected_obj_id,
                      )

def screw_in_bolt(selected_obj_id=0, desired_tool_T_world=None):
    WORLD_T_HOLE_START = PLATFORM_POSE
    HOLE_T_PEG_GOAL = ([[-0.00792, 0.0856, 0.41553], 
                        [0.62253, 0.33534, -0.62253, 0.33534]])
    WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)
    WORLD_T_PEG_GOAL = WORLD_T_PEG_START

    PEG_T_TIP = ([0, 0, 0], [0, 0, 0, 1])

    return TaskConfig(PEG_ASSET_NAME=home_dir + "taskboard/bolt.usd",
                      PEG_ATTACHMENT_PRIM="peg/precision_shoulder_screw/node_/mesh_",
                      HOLE_ASSET_NAME=home_dir + "taskboard/platform.usd",
                      HOLE_TYPE="peripheral",
                      HOLE_T_PEG_GOAL=WORLD_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      PEG_T_TIP=PEG_T_TIP)

def bolt_in_platform(selected_obj_id=0, desired_tool_T_world=None):

    # TODO
    PEG_GOAL = multiply(PLATFORM_POSE, ([[-0.00232, 0.12511, 0.22997], [0.09517, 0.1081, -0.89127, 0.42999]]))
    HOLE_GOAL = multiply(PLATFORM_POSE, ([[-0.00792, 0.0856, 0.41553], [0.62253, 0.33534, -0.62253, 0.33534]]))

    # Where the insertion starts. Not where the object starts out
    PLATFORM_T_STRUT = ([[0.0002, 0.15019, 0.23621], [0, 0, 0, 1]])
    WORLD_T_PEG_START = multiply(PLATFORM_POSE, PLATFORM_T_STRUT) # Strut starts standing up
    WORLD_T_HOLE_START = multiply(HOLE_GOAL, ([0, 0, -0.15], [0, 0, 0, 1]))

    PEG_T_TIP = ([0.01733, 0.0, 0.02706], [0.5, 0.5, 0.5, 0.5]) # Strut grasp
    HOLE_T_TIP = ([0, 0, 0], [0, 0, 0, 1]) # Bolt grasp
    
    WORLD_T_TOOL_PICK_HOLE = ([-0.35973, 0.25, 0.290], [-1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.3267949e-06])

    return TaskConfig(PEG_ASSET_NAME = home_dir + "taskboard/strut_real.usd",
                      HOLE_ASSET_NAME = home_dir + "taskboard/bolt.usd",
                      PEG_ATTACHMENT_PRIM = "peg/strut_real/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM = "hole/precision_shoulder_screw/node_/mesh_",
                      PEG_GOAL = PEG_GOAL,
                      HOLE_GOAL = HOLE_GOAL, 
                      WORLD_T_PEG_START = WORLD_T_PEG_START, 
                      WORLD_T_HOLE_START = WORLD_T_HOLE_START, 
                      PEG_T_TIP = PEG_T_TIP, 
                      HOLE_T_TIP = HOLE_T_TIP, 
                      WORLD_T_TOOL_PICK_PEG = None, 
                      WORLD_T_TOOL_PICK_HOLE = WORLD_T_TOOL_PICK_HOLE,
                      EXTRA_PARTS = [home_dir + "taskboard/elbow_on_stand.usd"],
                      EXTRA_PARTS_STARTING_POSE=[PLATFORM_POSE],
                      EXTRA_ATTACHMENTS=[("extra0/on_drive_north", "peg/strut_real/node_/mesh_")],
                      num_robots = 2,
                      relative_goal=False,
                      allow_peg_rotation=True)


def strut_in_elbow(selected_obj_id=0, desired_tool_T_world=None):
    PEG_T_HOLE_GOAL = ([[0.01906, -0.01631, -0.0378], [1, 0, 0., 0.]])
    HOLE_T_PEG_GOAL = pbu.invert(PEG_T_HOLE_GOAL)

    # Where the insertion starts. Not where the object starts out
    WORLD_T_PEG_START = ([0.05, 0.125, 0.3217], [0.0000, 0.7071, 0.0000, 0.7071])
    WORLD_T_HOLE_START = pbu.multiply(WORLD_T_PEG_START,
                                      pbu.multiply(PEG_T_HOLE_GOAL, ([[0, 0, 0.045], [0., 0., 0., 1.]])))

    WORLD_T_STRUT = pbu.multiply(WORLD_T_KIT, KIT_T_STRUT_5)
    WORLD_T_ELBOW = pbu.multiply(WORLD_T_KIT, KIT_T_ELBOW_0)

    return TaskConfig(PEG_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/strut_real.usd"),
                      HOLE_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/elbow_no_articulation_root.usd"),
                      PEG_ATTACHMENT_PRIM="peg/strut_real/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM="hole/on_drive_south/node_/mesh_",
                      HOLE_T_PEG_GOAL=HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                      PEG_T_TIP=STRUT_GRASP,
                      HOLE_T_TIP=ELBOW_GRASP,
                      WORLD_T_PEG_PICK=WORLD_T_STRUT,
                      WORLD_T_HOLE_PICK=WORLD_T_ELBOW,
                      peg_goal_weights=[5, 5, 1, 0, 0, 0],
                      relative_observations=True)


def strut_plus_elbow_in_platform(selected_obj_id=0, tool_T_world=None):
    WORLD_T_PEG_START = pbu.multiply(([0, 0.00, 0.030], [0, 0, 0, 1]), pbu.multiply(PLATFORM_POSE, PLATFORM_T_STRUT))
    bolt_start = pbu.multiply(pbu.multiply(pbu.multiply(PLATFORM_POSE, PLATFORM_T_STRUT), STRUT_T_BOLT),
                              ([0, 0, -0.03], [0, 0, 0, 1]))
    constraint = pbu.multiply(bolt_start, pbu.multiply(BOLT_GRASP, pbu.invert(TOOL_T_TIP)))

    return TaskConfig(
        PEG_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/strut_plus_elbow_large.usd"),
        HOLE_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/platform.usd"),
        HOLE_TYPE="peripheral",
        PEG_ATTACHMENT_PRIM="peg/strut_real/node_/mesh_",
        HOLE_IK_WORLD_T_TOOL=constraint,
        HOLE_ATTACHMENT_PRIM=None,
        HOLE_T_PEG_GOAL=PLATFORM_T_STRUT,
        WORLD_T_PEG_START=WORLD_T_PEG_START,
        WORLD_T_HOLE_START=PLATFORM_POSE,
        PEG_T_TIP=STRUT_GRASP)

def med_gear_in_taskboard(selected_obj_id=0, desired_tool_T_world=None):
    WORLD_T_PEG_START = pbu.multiply(([0, 0.00, 0.030], [0, 0, 0, 1]), pbu.multiply(TASKBOARD_POSE, PLATFORM_T_MEDGEAR))

    PEG_T_TIP = ([0, 0, 0], [0, 0, 0, 1])
    TASKBOARD_T_PICK_PEG = ([0.21818, 0.05686, 0.03195], [-0.0, -0.70711, 0.70711, 0.0])
    WORLD_T_PEG_PICK = pbu.multiply(TASKBOARD_POSE, TASKBOARD_T_PICK_PEG)

    return TaskConfig(PEG_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/gear_medium.usd"),
                      HOLE_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/taskboard.usd"),
                      HOLE_TYPE="peripheral",
                      PEG_ATTACHMENT_PRIM="peg/GEABP1_0_40_10_B_10_Gear_40teeth/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM=None,
                      HOLE_T_PEG_GOAL=PLATFORM_T_MEDGEAR,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=TASKBOARD_POSE,
                      EXTRA_PARTS=[os.path.join(ROOT_DIR, "taskboard/gear_small.usd"), os.path.join(ROOT_DIR, "taskboard/gear_large.usd")],
                      EXTRA_PARTS_STARTING_POSE=[pbu.multiply(TASKBOARD_POSE, PLATFORM_T_SMALLGEAR), pbu.multiply(TASKBOARD_POSE, PLATFORM_T_LARGEGEAR)],
                      PEG_T_TIP=PEG_T_TIP,
                      WORLD_T_PEG_PICK=WORLD_T_PEG_PICK
                      )

def small_gear_in_taskboard(selected_obj_id=0, desired_tool_T_world=None):
    WORLD_T_PEG_START = pbu.multiply(([0, 0.00, 0.030], [0, 0, 0, 1]), pbu.multiply(TASKBOARD_POSE, PLATFORM_T_SMALLGEAR))

    PEG_T_TIP = ([0, 0, 0], [0, 0, 0, 1])

    TASKBOARD_T_PICK_PEG = ([0.14292, 0.05686, -0.03561], [-0.0, -0.70711, 0.70711, 0.0])
    WORLD_T_PEG_PICK = pbu.multiply(TASKBOARD_POSE, TASKBOARD_T_PICK_PEG)

    return TaskConfig(PEG_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/gear_small.usd"),
                      HOLE_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/taskboard.usd"),
                      HOLE_TYPE="peripheral",
                      PEG_ATTACHMENT_PRIM="peg/GEABP1_0_20_10_B_10_Gear_20teeth/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM=None,
                      HOLE_T_PEG_GOAL=PLATFORM_T_SMALLGEAR,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=TASKBOARD_POSE,
                      PEG_T_TIP=PEG_T_TIP,
                      WORLD_T_PEG_PICK=WORLD_T_PEG_PICK)


def large_gear_in_taskboard(selected_obj_id=0, desired_tool_T_world=None):
    WORLD_T_PEG_START = pbu.multiply(([0, 0.00, 0.030], [0, 0, 0, 1]), pbu.multiply(TASKBOARD_POSE, PLATFORM_T_LARGEGEAR))

    PEG_T_TIP = ([0, 0, 0], [0, 0, 0, 1])
    TASKBOARD_T_PICK_PEG = ([0.2926, 0.05686, -0.03619], [-0.0, -0.70711, 0.70711, 0.0])
    WORLD_T_PEG_PICK = pbu.multiply(TASKBOARD_POSE, TASKBOARD_T_PICK_PEG)

    return TaskConfig(PEG_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/gear_large.usd"),
                      HOLE_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/taskboard.usd"),
                      HOLE_TYPE="peripheral",
                      PEG_ATTACHMENT_PRIM="peg/GEABP1_0_60_10_B_10_Gear_60teeth/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM=None,
                      HOLE_T_PEG_GOAL=PLATFORM_T_LARGEGEAR,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=TASKBOARD_POSE,
                      PEG_T_TIP=PEG_T_TIP,
                      WORLD_T_PEG_PICK=WORLD_T_PEG_PICK)


def rod_in_gear(selected_obj_id=0, desired_tool_T_world=None):
    # TODO
    HOLE_T_PEG_GOAL = ([[0., 0., 0.], [0., 0., 0., 1.]])

    # Where the insertion starts. Not where the object starts out
    WORLD_T_PEG_START = ([0.035, 0.2247, 0.4217], [0.0000, -0.7071, 0.0000, 0.7071])
    WORLD_T_HOLE_START = ([-0.035, 0.2247, 0.4217], [0.0000, -0.7071, 0.0000, 0.7071])

    PEG_T_TIP = ([0, 0, 0], [0, 0, 0, 1])
    HOLE_T_TIP = pbu.invert(([0, 0, 0], [-1, 0, 0, 0]))

    WORLD_T_TOOL_PICK_PEG = GEAR_PICK_POSE
    WORLD_T_TOOL_PICK_HOLE = BOLT_PICK_POSE

    return TaskConfig(PEG_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/gear_medium.usd"),
                      HOLE_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/shaft.usd"),
                      PEG_ATTACHMENT_PRIM="peg/GEABP1_0_40_10_B_10_Gear_40teeth/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM="hole/Xform/Cylinder",
                      HOLE_T_PEG_GOAL=HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                      PEG_T_TIP=PEG_T_TIP,
                      HOLE_T_TIP=HOLE_T_TIP,
                      WORLD_T_PEG_PICK=WORLD_T_TOOL_PICK_PEG,
                      WORLD_T_HOLE_PICK=WORLD_T_TOOL_PICK_HOLE,
                      peg_goal_weights=[5, 5, 1, 0, 0, 0])


def task_from_name(name, selected_obj_id=0, desired_tool_T_world=None) -> TaskConfig:
    return globals()[name](selected_obj_id=selected_obj_id, desired_tool_T_world=desired_tool_T_world)



if __name__ == "__main__":
    '''
    base_pos = (0.8367000122070312, 0.6095999755859375, 0.0225)
    base_q = [0, 0, 0.7071068, 0.7071068]

    task = task_from_name("screw_in_bolt")
    target_pos, target_q = multiply(task.WORLD_T_PEG_START, task.PEG_T_FLANGE)
    # print(target_pos)
    solve_ik(base_pos, base_q, target_pos, target_q)

    task = task_from_name("AssemblyMove")
    print(task.WORLD_T_PEG_START)
    '''
    # WORLD_T_PEG_START = ([0.45, 0.1, 0.05], [0, 0, 0, 1])
    # FIXTURE_T_PEG_START = multiply(invert(WORLD_T_FIXTURE_POSE), WORLD_T_PEG_START)
    # print('FIXTURE_T_PEG_START: ', FIXTURE_T_PEG_START)
    # WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_START)
    # print('WORLD_T_PEG_START: ', WORLD_T_PEG_START)
    # FIXTURE_T_PEG_GOAL =  multiply(([0.2, 0.28, 0.16], [0, 0, 1, 0]), ([0.0, 0.0, 0.0], [0, 0, 0.7071068, 0.7071068]))
    # print('FIXTURE_T_PEG_GOAL: ', FIXTURE_T_PEG_GOAL)
    # WORLD_T_PEG_GOAL = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_GOAL)
    # print('WORLD_T_PEG_GOAL: ', WORLD_T_PEG_GOAL)
    # HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0, 0, 0, 1])
    # HOLE_T_PEG_GOAL_v1 = multiply(([0, 0, 0], [0, 0, 1, 0]), HOLE_T_PEG_GOAL)
    # print('HOLE_T_PEG_GOAL v1: ', HOLE_T_PEG_GOAL_v1)
    # HOLE_T_PEG_GOAL_v2 = multiply(invert(WORLD_T_FIXTURE_POSE), HOLE_T_PEG_GOAL_v1)
    # print('HOLE_T_PEG_GOAL v2: ', HOLE_T_PEG_GOAL_v2)
    # HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0, 0, 0, 1])
    # WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)
    # print('prev WORLD_T_PEG_START: ', WORLD_T_PEG_START)
    # HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0, 0, 1, 0])
    # WORLD_T_PEG_START = multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)
    # print('now WORLD_T_PEG_START: ', WORLD_T_PEG_START)
    # HOLE_T_PEG_GOAL = multiply(invert(WORLD_T_HOLE_START), WORLD_T_PEG_START)
    # print('now WORLD_T_PEG_START: ', HOLE_T_PEG_GOAL)
    # FIXTURE_T_PEG_START = ([0.2, 0.28, 0.16], [0, 0, 0, 1])
    # WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_START)
    # print('v1 WORLD_T_PEG_START: ', WORLD_T_PEG_START)
    # WORLD_T_PEG_START = ([0.6, 0.2, 0.15], [0, 0, 1, 0])
    # a = multiply(WORLD_T_PEG_START, invert(([0, 0, 0], [0, 0, 0.7071068, 0.7071068])))
    # print('v2 a: ', a)
    # b = multiply(a, ([0, 0, 0], [0, 0, 0.7071068, 0.7071068]))
    # print('v3 b: ', b)
    # WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, ([0.0, 0.0, 0.16], [0, 0, 0, 1]))
    # FIXTURE_T_PEG_START = ([0.01, 0, 0.02], [0, -0.4226183, 0, 0.9063078])
    # WORLD_T_PEG_START = multiply(WORLD_T_FIXTURE_POSE, FIXTURE_T_PEG_START)
    WORLD_T_PEG_START = ([0, 0, 0], [0, 0, 1, 0])
    WORLD_T_PEG_START = multiply(WORLD_T_PEG_START, ([0., 0., 0.0], [0, 0, 1, 0]))
    print('v4 WORLD_T_PEG_START: ', WORLD_T_PEG_START)
