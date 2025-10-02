import numpy as np
import pybullet as p

def multiply(*poses):
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = p.multiplyTransforms(pose[0], pose[1], *next_pose)
    return pose

def invert(pose):
    point, quat = pose
    return p.invertTransform(point, quat)

# ([tx, ty, tz], [rx, ry, rz, rw]), tx, ty, tz are in meters
WORLD_T_CAMERA = ([0.2,-0.5,0.760453064], [-0.9110028, 0, 0, 0.4124002])  # accurate
WORLD_T_PEG_START = ([0.6, 0.05, 0.03], [0.7071068, 0, 0, -0.7071068]) # this is the desired pose estimation outcome, WORLD_T_PEG_START = multiply(WORLD_T_CAMERA_HOME, CAM_T_PEG) 
PEG_T_TIP_HORIZONTAL = multiply(([0, 0.05, 0.04], [-0.7071068, 0, 0, 0.7071068]),
                                    ([0, 0, 0], [0, 0, 1, 0]))  # accurate
TOOL_T_TIP = ([0, 0, 0.28], [0, 0, 0, 1])
WORLD_T_TOOL_PICK_PEG = multiply(multiply(WORLD_T_PEG_START, PEG_T_TIP_HORIZONTAL), invert(TOOL_T_TIP))
print('WORLD_T_TOOL_PICK_PEG: ', WORLD_T_TOOL_PICK_PEG)
# WORLD_T_TOOL_PICK_PEG:  ((0.6000000238418579, 0.09000000357627869, 0.25999999046325684), (0.0, 1.0, 0.0, 0.0))
