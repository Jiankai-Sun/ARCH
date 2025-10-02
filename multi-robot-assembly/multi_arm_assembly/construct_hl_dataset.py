import numpy as np
import os
import matplotlib.pyplot as plt
import json
import cv2
import torch
from scipy.spatial.transform import Rotation as R
from multi_arm_assembly.utils import stage_sequence
import ast

input_type = ['primitive', 'obs/side_1', 'obs/side_2', 'obs/wrist_1', 'obs/wrist_2', 'obs/side_1_depth', 
              'obs/side_2_depth', 'obs/wrist_1_depth', 'obs/wrist_2_depth',
               'obs/tcp_pose', 'obs/tcp_vel', 'obs/tcp_force', 'obs/tcp_torque', 'obs/q', 'obs/dq', 'obs/jacobian',
                 'obs/gripper_pose', 'action', 'object_id', 'object_info']

def json_load(file_path):
    """
    Load JSON data from a file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict or list: The JSON data parsed into a Python dictionary or list.
    """
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file_path}: {e}")
            return None
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except IOError as e:
            print(f"IOError reading file {file_path}: {e}")
            return None

def quaternion_to_euler(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quat (list or tuple): Quaternion in the format (rx, ry, rz, rw).
        
    Returns:
        tuple: Euler angles (roll, pitch, yaw) in radians.
    """
    r = R.from_quat(quat)
    return r.as_euler('xyz', degrees=False)  # Convert to Euler angles (roll, pitch, yaw) in degrees


def calculate_relative_actions(data):
    """
    Calculate relative changes in translation and rotation between consecutive steps.
    
    Args:
        data (list of lists): List where each item is [tx, ty, tz, rx, ry, rz, rw].
        
    Returns:
        list of tuples: Relative changes in translation and rotation.
    """
    relative_actions = [[0, 0, 0, 0, 0, 0, 0]]
    
    for i in range(1, len(data)):
        prev = np.array(data[i-1])
        curr = np.array(data[i])
        
        # Translation differences
        translation_diff = curr[:3] - prev[:3]
        
        # Quaternion differences
        prev_quat = prev[3:]
        curr_quat = curr[3:]
        
        # Convert quaternions to Euler angles
        prev_euler = quaternion_to_euler(prev_quat)
        curr_euler = quaternion_to_euler(curr_quat)
        
        # Rotation differences (Euler angles)
        rotation_diff = np.array(curr_euler) - np.array(prev_euler)
        
        if i < len(data) - 10:
            gripper = 0  # open
        else:
            gripper = 1  # close
        
        relative_actions.append(translation_diff.tolist() + rotation_diff.tolist() + [gripper])
    
    return relative_actions

def action2state(action):
    trans = action[:3]
    rotm = R.from_quat(action[3:7]).as_matrix()
    state = np.concatenate([trans, rotm[:3, 0], rotm[:3, 1]], axis=0)
    return state


def construct_hl_dataset(demo_dir='logs/demos_araas', save_dir='logs/assembly_dataset_large/', img_size=256, n_obs_steps=1, min_seq_len = 14, use_image=False, task='assembly'):
    demo_idx = sorted([i for i in os.listdir(demo_dir) if os.path.isdir(os.path.join(demo_dir, i))])
    print('demo_idx: ', demo_idx)
    os.makedirs(save_dir, exist_ok=True)
    for each_demo in demo_idx:
        obs_side_1 = []
        obs_side_2 = []
        obs_wrist_1 = []
        obs_wrist_2 = []
        obs_side_1_depth = []
        obs_side_2_depth = []
        obs_wrist_1_depth = []
        obs_wrist_2_depth = []
        obs_tcp_pose = []
        obs_tcp_vel = []
        obs_tcp_force = []
        obs_tcp_torque = []
        obs_q = []
        obs_dq = []
        obs_jacobian = []
        obs_gripper_pose = []
        action = []

        primitive_list = []
        object_id = []
        object_info = []

        each_demo_dir = os.path.join(demo_dir, each_demo)
        print('each_demo_dir: ', each_demo_dir)
        all_primitives_seq = sorted(os.listdir(each_demo_dir), key=lambda x: int(x.split('_')[1]))
        print('primitives: ', all_primitives_seq)

        for prefill in range(n_obs_steps):
            if task == 'assembly':
                primitive_list.append('AssemblyReset')
                policy_obs = np.ones((9+6,))
                image_obs = np.ones((256, 256, 3))  # rgb
                depth = np.ones((256, 256, 1))   # depth


                obs_side_1.append(cv2.resize(image_obs, (img_size, img_size)))
                obs_side_2.append(cv2.resize(image_obs, (img_size, img_size)))
                obs_wrist_1.append(cv2.resize(image_obs, (img_size, img_size)))
                obs_wrist_2.append(cv2.resize(image_obs, (img_size, img_size)))
                obs_side_1_depth.append(cv2.resize(depth, (img_size, img_size)))
                obs_side_2_depth.append(cv2.resize(depth, (img_size, img_size)))
                obs_wrist_1_depth.append(cv2.resize(depth, (img_size, img_size)))
                obs_wrist_2_depth.append(cv2.resize(depth, (img_size, img_size)))

                # obs_tcp_pose.append(policy_obs[:9])  # (N, 9)
                obs_tcp_pose.append(policy_obs[:7])  # (N, 7)
                obs_tcp_vel.append(np.ones((6,)))
                obs_tcp_force.append(policy_obs[9:12])

                obs_tcp_torque.append(policy_obs[12:15])

                obs_q.append(np.ones((7,)))
                obs_dq.append(np.ones((7,)))
                obs_jacobian.append(np.ones((6, 7)))
                obs_gripper_pose.append(np.ones(()))

                action.append(np.ones((7,)))
                object_id.append(np.ones(()))
                object_info.append({})
        
        for each_primitive_seq_name in all_primitives_seq:
            primitives = np.load(os.path.join(each_demo_dir, each_primitive_seq_name), allow_pickle=True)
            # print('primitive data: ', primitives, len(primitives))
            for step_i, [obs, action_primitive] in enumerate(primitives):
                print('Appending {}'.format(action_primitive))
                primitive_list.append(action_primitive)
                policy_obs = np.round(obs['policy'], decimals=1)  # obs['policy']
                # print(obs['image'].shape)
                image_obs = np.ones((256, 256, 3)) # depth
                if use_image:
                    depth = obs['image'] if 'image' in obs else np.ones((256, 256, 1))   # depth
                else:
                    depth = np.ones((256, 256, 1))

                obs_side_1.append(cv2.resize(image_obs, (img_size, img_size)))
                obs_side_2.append(cv2.resize(image_obs, (img_size, img_size)))
                obs_wrist_1.append(cv2.resize(image_obs, (img_size, img_size)))
                obs_wrist_2.append(cv2.resize(image_obs, (img_size, img_size)))
                obs_side_1_depth.append(cv2.resize(depth, (img_size, img_size)))
                obs_side_2_depth.append(cv2.resize(depth, (img_size, img_size)))
                obs_wrist_1_depth.append(cv2.resize(depth, (img_size, img_size)))
                obs_wrist_2_depth.append(cv2.resize(depth, (img_size, img_size)))

                # obs_tcp_pose.append(policy_obs[:9])  # (N, 9)
                obs_tcp_pose.append(policy_obs[:7])  # (N, 7)
                obs_tcp_vel.append(np.ones((6,)))
                obs_tcp_force.append(policy_obs[9:12])

                obs_tcp_torque.append(policy_obs[12:15])

                obs_q.append(np.ones((7,)))
                obs_dq.append(np.ones((7,)))
                obs_jacobian.append(np.ones((6, 7)))
                obs_gripper_pose.append(np.ones(()))

                action.append(np.ones((7,)))
                object_id.append(np.ones(()))
                object_info.append({})

                if step_i >= 0:
                    break
        
        # print(min_seq_len, len(primitive_list))
        # if min_seq_len > len(primitive_list):
        #     for i in range(min_seq_len - len(primitive_list)):
        #         primitive_list.append(action_primitive)
        #
        #         obs_side_1.append(cv2.resize(image_obs, (img_size, img_size)))
        #         obs_side_2.append(cv2.resize(image_obs, (img_size, img_size)))
        #         obs_wrist_1.append(cv2.resize(image_obs, (img_size, img_size)))
        #         obs_wrist_2.append(cv2.resize(image_obs, (img_size, img_size)))
        #         obs_side_1_depth.append(cv2.resize(depth, (img_size, img_size)))
        #         obs_side_2_depth.append(cv2.resize(depth, (img_size, img_size)))
        #         obs_wrist_1_depth.append(cv2.resize(depth, (img_size, img_size)))
        #         obs_wrist_2_depth.append(cv2.resize(depth, (img_size, img_size)))
        #
        #         # obs_tcp_pose.append(policy_obs[:9])  # (N, 9)
        #         obs_tcp_pose.append(policy_obs[:7])  # (N, 7)
        #         obs_tcp_vel.append(np.ones((6,)))
        #         obs_tcp_force.append(policy_obs[9:12])
        #
        #         obs_tcp_torque.append(policy_obs[12:15])
        #
        #         obs_q.append(np.ones((7,)))
        #         obs_dq.append(np.ones((7,)))
        #         obs_jacobian.append(np.ones((6, 7)))
        #         obs_gripper_pose.append(np.ones(()))
        #
        #         action.append(np.ones((7,)))
        #         object_id.append(np.ones(()))
        #         object_info.append({})

        demo_dict = {'primitive': primitive_list, 
                     'obs/side_1': np.asarray(obs_side_1), 
                     'obs/side_2': np.asarray(obs_side_2), 
                     'obs/wrist_1': np.asarray(obs_wrist_1),
                     'obs/wrist_2': np.asarray(obs_wrist_2),
                     'obs/side_1_depth': np.asarray(obs_side_1_depth),
                     'obs/side_2_depth': np.asarray(obs_side_2_depth),
                     'obs/wrist_1_depth': np.asarray(obs_wrist_1_depth),
                     'obs/wrist_2_depth': np.asarray(obs_wrist_2_depth),
                     'obs/tcp_pose': np.asarray(obs_tcp_pose),
                     'obs/tcp_vel': np.asarray(obs_tcp_vel),
                     'obs/tcp_force': np.asarray(obs_tcp_force),
                     'obs/tcp_torque': np.asarray(obs_tcp_torque),
                     'obs/q': np.asarray(obs_q),
                     'obs/dq': np.asarray(obs_dq),
                     'obs/jacobian': np.asarray(obs_jacobian),
                     'obs/gripper_pose': np.asarray(obs_gripper_pose),
                     'actions': np.asarray(action),
                     'object_id': np.asarray(object_id),
                     'object_info': object_info
                     }

        save_path = os.path.join(save_dir, '2_M_S_{}.npy'.format(each_demo))
        np.save(save_path, demo_dict)
        for k, v in demo_dict.items():
            if k in ['primitive', 'object_info']:
                print(k, len(v))
            else:
                print(k, v.shape)
        print('Saved to {}'.format(save_path))

def construct_ll_dataset(demo_dir='logs/demos_araas', save_dir='logs/assembly_dataset_ll/', 
                         img_size=256, n_obs_steps=1, min_seq_len = 14, use_image=True, num_interpolate_step=1000):
    demo_idx = sorted([i for i in os.listdir(demo_dir) if os.path.isdir(os.path.join(demo_dir, i))])
    print('demo_idx: ', demo_idx)
    os.makedirs(save_dir, exist_ok=True)
    for each_demo in demo_idx:
        obs_side_1 = []
        obs_side_2 = []
        obs_wrist_1 = []
        obs_wrist_2 = []
        obs_side_1_depth = []
        obs_side_2_depth = []
        obs_wrist_1_depth = []
        obs_wrist_2_depth = []
        obs_tcp_pose = []
        obs_tcp_vel = []
        obs_tcp_force = []
        obs_tcp_torque = []
        obs_q = []
        obs_dq = []
        obs_jacobian = []
        obs_gripper_pose = []
        action = []

        primitive_list = []
        object_id = []
        object_info = []

        each_demo_dir = os.path.join(demo_dir, each_demo)
        print('each_demo_dir: ', each_demo_dir)
        all_primitives_seq = sorted(os.listdir(each_demo_dir), key=lambda x: int(x.split('_')[1]))
        print('primitives: ', all_primitives_seq)

        # for prefill in range(n_obs_steps):
        #     primitive_list.append('AssemblyReset')
        #     policy_obs = np.ones((9+6,))
        #     image_obs = np.ones((256, 256, 3))  # rgb
        #     depth = np.ones((256, 256, 1))   # depth


        #     obs_side_1.append(cv2.resize(image_obs, (img_size, img_size)))
        #     obs_side_2.append(cv2.resize(image_obs, (img_size, img_size)))
        #     obs_wrist_1.append(cv2.resize(image_obs, (img_size, img_size)))
        #     obs_wrist_2.append(cv2.resize(image_obs, (img_size, img_size)))
        #     obs_side_1_depth.append(cv2.resize(depth, (img_size, img_size)))
        #     obs_side_2_depth.append(cv2.resize(depth, (img_size, img_size)))
        #     obs_wrist_1_depth.append(cv2.resize(depth, (img_size, img_size)))
        #     obs_wrist_2_depth.append(cv2.resize(depth, (img_size, img_size)))

        #     # obs_tcp_pose.append(policy_obs[:9])  # (N, 9)
        #     obs_tcp_pose.append(policy_obs[:7])  # (N, 7)
        #     obs_tcp_vel.append(np.ones((6,)))
        #     obs_tcp_force.append(policy_obs[9:12])

        #     obs_tcp_torque.append(policy_obs[12:15])

        #     obs_q.append(np.ones((7,)))
        #     obs_dq.append(np.ones((7,)))
        #     obs_jacobian.append(np.ones((6, 7)))
        #     obs_gripper_pose.append(np.ones(()))

        #     action.append(np.ones((7,)))
        #     object_id.append(np.ones(()))
        #     object_info.append({})
        
        for each_primitive_seq_name in all_primitives_seq:
            primitives = np.load(os.path.join(each_demo_dir, each_primitive_seq_name), allow_pickle=True)
            # print('primitive data: ', primitives, len(primitives))
            for step_i, [obs, action_primitive] in enumerate(primitives):
                print('Appending {}'.format(action_primitive))
                policy_obs = np.round(obs['policy'], decimals=1)  # obs['policy']
                # print(obs['image'].shape)
                image_obs = np.ones((256, 256, 3)) # depth
                if use_image:
                    depth = obs['image'] if 'image' in obs else np.ones((256, 256, 1))   # depth
                else:
                    depth = np.ones((256, 256, 1))
                # if action_primitive in ['AssemblyScriptedGrasp']:
                if action_primitive in ['AssemblyReset']:
                    continue

                # Open and read the file
                with open('logs/demos_araas/0000_0_0_{}.json'.format(action_primitive), 'r') as file:
                    # Read all lines
                    lines = file.readlines()

                # Convert each line to a list using ast.literal_eval
                trajectory = [ast.literal_eval(line.strip()) for line in lines]
                trajectory = trajectory[::100]
                # print('trajectory: ', trajectory)
                relative_actions = calculate_relative_actions(trajectory)
                for trajectory_i in range(len(trajectory)):
                    primitive_list.append(action_primitive)
                    obs_side_1.append(cv2.resize(image_obs, (img_size, img_size)))
                    obs_side_2.append(cv2.resize(image_obs, (img_size, img_size)))
                    obs_wrist_1.append(cv2.resize(image_obs, (img_size, img_size)))
                    obs_wrist_2.append(cv2.resize(image_obs, (img_size, img_size)))
                    obs_side_1_depth.append(cv2.resize(depth, (img_size, img_size)))
                    obs_side_2_depth.append(cv2.resize(depth, (img_size, img_size)))
                    obs_wrist_1_depth.append(cv2.resize(depth, (img_size, img_size)))
                    obs_wrist_2_depth.append(cv2.resize(depth, (img_size, img_size)))

                    # obs_tcp_pose.append(policy_obs[:9])  # (N, 9)
                    obs_tcp_pose.append(policy_obs[:7])  # (N, 7)
                    obs_tcp_vel.append(np.ones((6,)))
                    obs_tcp_force.append(policy_obs[9:12])

                    obs_tcp_torque.append(policy_obs[12:15])

                    obs_q.append(np.ones((7,)))
                    obs_dq.append(np.ones((7,)))
                    obs_jacobian.append(np.ones((6, 7)))
                    obs_gripper_pose.append(np.ones(()))

                    action.append(relative_actions[trajectory_i])
                    object_id.append(np.ones(()))
                    object_info.append({})
                    # if trajectory[trajectory_i] is absolute pose
                    policy_obs[0:9] = action2state(trajectory[trajectory_i])

                if step_i >= 0:
                    break
                # else:
                #     for trajectory_i in range(num_interpolate_step):
                #         obs_side_1.append(cv2.resize(image_obs, (img_size, img_size)))
                #         obs_side_2.append(cv2.resize(image_obs, (img_size, img_size)))
                #         obs_wrist_1.append(cv2.resize(image_obs, (img_size, img_size)))
                #         obs_wrist_2.append(cv2.resize(image_obs, (img_size, img_size)))
                #         obs_side_1_depth.append(cv2.resize(depth, (img_size, img_size)))
                #         obs_side_2_depth.append(cv2.resize(depth, (img_size, img_size)))
                #         obs_wrist_1_depth.append(cv2.resize(depth, (img_size, img_size)))
                #         obs_wrist_2_depth.append(cv2.resize(depth, (img_size, img_size)))

                #         # obs_tcp_pose.append(policy_obs[:9])  # (N, 9)
                #         obs_tcp_pose.append(policy_obs[:7])  # (N, 7)
                #         obs_tcp_vel.append(np.ones((6,)))
                #         obs_tcp_force.append(policy_obs[9:12])

                #         obs_tcp_torque.append(policy_obs[12:15])

                #         obs_q.append(np.ones((7,)))
                #         obs_dq.append(np.ones((7,)))
                #         obs_jacobian.append(np.ones((6, 7)))
                #         obs_gripper_pose.append(np.ones(()))

                #         action.append(np.ones((7,)))
                #         object_id.append(np.ones(()))
                #         object_info.append({})

                #     if step_i >= 0:
                #         break
        
        # print(min_seq_len, len(primitive_list))
        # if min_seq_len > len(primitive_list):
        #     for i in range(min_seq_len - len(primitive_list)):
        #         primitive_list.append(action_primitive)
        #
        #         obs_side_1.append(cv2.resize(image_obs, (img_size, img_size)))
        #         obs_side_2.append(cv2.resize(image_obs, (img_size, img_size)))
        #         obs_wrist_1.append(cv2.resize(image_obs, (img_size, img_size)))
        #         obs_wrist_2.append(cv2.resize(image_obs, (img_size, img_size)))
        #         obs_side_1_depth.append(cv2.resize(depth, (img_size, img_size)))
        #         obs_side_2_depth.append(cv2.resize(depth, (img_size, img_size)))
        #         obs_wrist_1_depth.append(cv2.resize(depth, (img_size, img_size)))
        #         obs_wrist_2_depth.append(cv2.resize(depth, (img_size, img_size)))
        #
        #         # obs_tcp_pose.append(policy_obs[:9])  # (N, 9)
        #         obs_tcp_pose.append(policy_obs[:7])  # (N, 7)
        #         obs_tcp_vel.append(np.ones((6,)))
        #         obs_tcp_force.append(policy_obs[9:12])
        #
        #         obs_tcp_torque.append(policy_obs[12:15])
        #
        #         obs_q.append(np.ones((7,)))
        #         obs_dq.append(np.ones((7,)))
        #         obs_jacobian.append(np.ones((6, 7)))
        #         obs_gripper_pose.append(np.ones(()))
        #
        #         action.append(np.ones((7,)))
        #         object_id.append(np.ones(()))
        #         object_info.append({})

        demo_dict = {'primitive': primitive_list, 
                     'obs/side_1': np.asarray(obs_side_1), 
                     'obs/side_2': np.asarray(obs_side_2), 
                     'obs/wrist_1': np.asarray(obs_wrist_1),
                     'obs/wrist_2': np.asarray(obs_wrist_2),
                     'obs/side_1_depth': np.asarray(obs_side_1_depth),
                     'obs/side_2_depth': np.asarray(obs_side_2_depth),
                     'obs/wrist_1_depth': np.asarray(obs_wrist_1_depth),
                     'obs/wrist_2_depth': np.asarray(obs_wrist_2_depth),
                     'obs/tcp_pose': np.asarray(obs_tcp_pose),
                     'obs/tcp_vel': np.asarray(obs_tcp_vel),
                     'obs/tcp_force': np.asarray(obs_tcp_force),
                     'obs/tcp_torque': np.asarray(obs_tcp_torque),
                     'obs/q': np.asarray(obs_q),
                     'obs/dq': np.asarray(obs_dq),
                     'obs/jacobian': np.asarray(obs_jacobian),
                     'obs/gripper_pose': np.asarray(obs_gripper_pose),
                     'actions': np.asarray(action),
                     'object_id': np.asarray(object_id),
                     'object_info': object_info
                     }

        save_path = os.path.join(save_dir, '2_M_S_{}.npy'.format(each_demo))
        np.save(save_path, demo_dict)
        for k, v in demo_dict.items():
            if k in ['primitive', 'object_info']:
                print(k, len(v))
            else:
                print(k, v.shape)
        print('Saved to {}'.format(save_path))


def print_info(key_list=['primitive', 'obs/tcp_pose'], npy_path='logs/assembly_dataset/2_M_S_0.npy'):
    traj = np.load(npy_path, allow_pickle=True).item()
    print(traj.keys())
    dict_json = dict()
    for k in key_list:
        if isinstance(traj[k], np.ndarray):
            dict_json[k] = traj[k].tolist()
        else:
            dict_json[k] = traj[k]
    json_path = npy_path[:-4] + '.json'
    with open(json_path, 'w') as f:
        json.dump(dict_json, f, indent=4)
        print('Saved to {}'.format(json_path))


if __name__ == '__main__':
    # construct_hl_dataset(demo_dir='logs/demos_araas')
    # construct_ll_dataset(demo_dir='logs/demos_araas')
    construct_hl_dataset(demo_dir='logs/demos_isaaclab',task='beam')
    # print_info(npy_path='logs/assembly_dataset/2_M_S_0.npy')