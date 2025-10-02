from distutils.util import strtobool
import argparse, os, yaml
import gym
import time
import multi_arm_assembly.araas_environment as araas_environment
from rl_games.common import env_configurations
from rl_games.torch_runner import Runner
from multi_arm_assembly.utils import (log, get_scripted_actions, task_from_name, stage_sequence, set_seed, display_menu,
                                      split_obj_dict, OBJECT_DICT)
from multi_arm_assembly.rl_components.my_network import MyNetworkBuilder 
from multi_arm_assembly.rl_components.my_a2c_model import MyA2CContinuousLogStd
from multi_arm_assembly.rl_components.my_network import MyNetworkBuilder 
from multi_arm_assembly.rl_components.my_agent import MyA2CAgent
from multi_arm_assembly.rl_components.my_player import MyPpoPlayerContinuous
from rl_games.algos_torch import model_builder
import torch
import numpy as np
set_seed(123)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
home_dir = os.getcwd() + '/../'

from collections import deque
import hydra
import dill
import sys
import pickle
sys.path.insert(0, '../../diffusion_policy_v1/')
from diffusion_policy.common.pytorch_util import dict_apply
from matplotlib import pyplot as plt
import cv2
'''
pip install hydra-core==1.3.2 dill==0.3.8 diffusers==0.27.2 zarr==2.16.1 robomimic==0.2.0
'''
sys.path.append('../../CPPF2_assembly/')
sys.path.append('../../CPPF2_assembly/SAM2')
# sys.path.append('../../CPPF2_assembly/SAM2/sam2')
from interactive import CPPF2_wrapper

torch.set_grad_enabled(False)

def compare_batch(batch_1, batch_2, step_counter=0):
    # Compare agent_pos values
    agent_pos_1 = batch_1['agent_pos'][0, :2].cpu().numpy()
    agent_pos_2 = batch_2['agent_pos'][0, :2].cpu().numpy()
    
    # Calculate and print the value difference for agent_pos
    if agent_pos_1.shape != agent_pos_2.shape:
        print("Shapes of 'agent_pos' do not match:", agent_pos_1.shape, agent_pos_2.shape)
    else:
        value_diff = agent_pos_1 - agent_pos_2
        print(f"'agent_pos 1': {agent_pos_1}")
        print(f"'agent_pos 2': {agent_pos_2}")
        print(f"Total difference in 'agent_pos': {value_diff}")

    # Compare images
    depth_image_1 = batch_1['image'].permute(0, 1, 3, 4, 2).cpu().numpy()[0, :2]
    depth_image_2 = batch_2['image'].permute(0, 1, 3, 4, 2).cpu().numpy()[0, :2]

    if depth_image_1.shape != depth_image_2.shape:
        print("Shapes of images do not match:", depth_image_1.shape, depth_image_2.shape)
    else:
        print('np.max(depth_image_2)', np.max(depth_image_2))  # 4.26
        # # Split depth images into individual channels
        # depth_image_1_chan1, depth_image_1_chan2 = depth_image_1[..., 0], depth_image_1[..., 1]
        # depth_image_2_chan1, depth_image_2_chan2 = depth_image_2[..., 0], depth_image_2[..., 1]

        # # Compute absolute differences
        # diff_chan1 = np.abs(depth_image_1_chan1 - depth_image_2_chan1)
        # diff_chan2 = np.abs(depth_image_1_chan2 - depth_image_2_chan2)

        # # Normalize images for visualization
        # def normalize_image(image):
        #     return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # depth_image_1_chan1_norm = normalize_image(depth_image_1_chan1)
        # depth_image_1_chan2_norm = normalize_image(depth_image_1_chan2)
        # depth_image_2_chan1_norm = normalize_image(depth_image_2_chan1)
        # depth_image_2_chan2_norm = normalize_image(depth_image_2_chan2)
        # diff_chan1_norm = normalize_image(diff_chan1)
        # diff_chan2_norm = normalize_image(diff_chan2)

        # # Display depth images and differences
        # plt.figure(figsize=(12, 8))

        # plt.subplot(2, 3, 1)
        # plt.title('Depth Image 1 - Channel 1')
        # plt.imshow(depth_image_1_chan1_norm, cmap='plasma')
        # plt.axis('off')

        # plt.subplot(2, 3, 2)
        # plt.title('Depth Image 2 - Channel 1')
        # plt.imshow(depth_image_2_chan1_norm, cmap='plasma')
        # plt.axis('off')

        # plt.subplot(2, 3, 3)
        # plt.title('Difference - Channel 1')
        # plt.imshow(diff_chan1_norm, cmap='plasma')
        # plt.axis('off')

        # plt.subplot(2, 3, 4)
        # plt.title('Depth Image 1 - Channel 2')
        # plt.imshow(depth_image_1_chan2_norm, cmap='plasma')
        # plt.axis('off')

        # plt.subplot(2, 3, 5)
        # plt.title('Depth Image 2 - Channel 2')
        # plt.imshow(depth_image_2_chan2_norm, cmap='plasma')
        # plt.axis('off')

        # plt.subplot(2, 3, 6)
        # plt.title('Difference - Channel 2')
        # plt.imshow(diff_chan2_norm, cmap='plasma')
        # plt.axis('off')

        # plt.tight_layout()
        # plt.show()

        # # Optionally, save the images for further analysis
        # cv2.imwrite('depth_image_1_chan1.png', depth_image_1_chan1_norm)
        # cv2.imwrite('depth_image_1_chan2.png', depth_image_1_chan2_norm)
        # cv2.imwrite('depth_image_2_chan1.png', depth_image_2_chan1_norm)
        # cv2.imwrite('depth_image_2_chan2.png', depth_image_2_chan2_norm)
        # cv2.imwrite('diff_chan1.png', diff_chan1_norm)
        # cv2.imwrite('diff_chan2.png', diff_chan2_norm)
        # Split the depth images into individual channels
        def split_channels(depth_images):
            channels = []
            for i in range(depth_images.shape[0]):  # Loop over the 2 depth images
                img = depth_images[i]
                channels.append(img[..., 0])  # Channel 1
                channels.append(img[..., 1])  # Channel 2
            return channels
        
        # Split both depth images
        depth_image_1_channels = split_channels(depth_image_1)
        depth_image_2_channels = split_channels(depth_image_2)

        # Compute absolute differences
        def compute_differences(channels_1, channels_2):
            differences = []
            diff_sum = 0 
            for c1, c2 in zip(channels_1, channels_2):
                diff = np.abs(c1 - c2)
                diff_sum = np.sum(diff)
                differences.append(diff)
                print(f"Difference stats:\nSum: {diff_sum}\nMax: {np.max(diff)}\nMin: {np.min(diff)}")
                print(f"Channel 1 stats:\nMax: {np.max(c1)}\nMin: {np.min(c1)}")
                print(f"Channel 2 stats:\nMax: {np.max(c2)}\nMin: {np.min(c2)}")
                print(f"Channel 1 sample:\n{c1}\nChannel 2 sample:\n{c2}")
                plt.imshow(diff)
                plt.show()
            return differences, diff_sum
        
        differences, diff_sum = compute_differences(depth_image_1_channels, depth_image_2_channels)

        print('Image Diff Sum: ', diff_sum)
        # Normalize images for visualization
        def normalize_image(image):
            return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        depth_image_1_channels_norm = [normalize_image(chan) for chan in depth_image_1_channels]
        depth_image_2_channels_norm = [normalize_image(chan) for chan in depth_image_2_channels]
        differences_norm = [normalize_image(diff) for diff in differences]

        # Display depth images and differences
        plt.figure(figsize=(16, 8))

        # Plot images from depth_image_1
        for i in range(4):
            plt.subplot(3, 4, i + 1)
            plt.title(f'Depth Image 1 - Channel {i + 1}')
            plt.imshow(depth_image_1_channels_norm[i], cmap='plasma')
            plt.axis('off')

        # Plot images from depth_image_2
        for i in range(4):
            plt.subplot(3, 4, i + 5)
            plt.title(f'Depth Image 2 - Channel {i + 1}')
            plt.imshow(depth_image_2_channels_norm[i], cmap='plasma')
            plt.axis('off')

        # Plot differences
        for i in range(4):
            plt.subplot(3, 4, i + 9)
            plt.title(f'Difference - Channel {i + 1}')
            plt.imshow(differences_norm[i], cmap='plasma')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        # # Optionally, save the images for further analysis
        # for i in range(4):
        #     cv2.imwrite(f'depth_image_1_chan_{i+1}.png', depth_image_1_channels_norm[i])
        #     cv2.imwrite(f'depth_image_2_chan_{i+1}.png', depth_image_2_channels_norm[i])
        #     cv2.imwrite(f'diff_chan_{i+1}.png', differences_norm[i])

def run_high_level_pipeline():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, required=False,
                    help="random seed, if larger than 0 will overwrite the value in yaml config")
    ap.add_argument("-tf", "--tf", required=False, help="run tensorflow runner", action='store_true')
    ap.add_argument("-t", "--train", required=False, help="train network", action='store_true')
    ap.add_argument("-rr", "--enable-hardware", action="store_true", help="path to config")
    ap.add_argument("-na", "--num_actors", type=int, default=1, required=False,
                    help="number of envs running in parallel, if larger than 0 will overwrite the value in yaml config")
    ap.add_argument("-s", "--sigma", type=float, required=False,
                    help="sets new sigma value in case if 'fixed_sigma: True' in yaml config")
    ap.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                    help="if toggled, this experiment will be tracked with Weights and Biases")
    ap.add_argument("--task", type=str, default="med_gear_in_taskboard",
                    help="task name")
    ap.add_argument("--wandb-project-name", type=str, default="rl_games",
                    help="the wandb's project name")
    ap.add_argument("--wandb-entity", type=str, default=None,
                    help="the entity (team) of wandb's project")
    ap.add_argument("-f", "--file", default="./assembly_insert_env_cfg.yaml", help="path to config")
    ap.add_argument("--prefix", type=str, default="",
                    help="name to tag to the end of the project name")
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    args = vars(ap.parse_args())
    model_builder.register_network('my_network', MyNetworkBuilder)
    model_builder.register_model('my_actor_model', lambda network, **kwargs: MyA2CContinuousLogStd(network))
    araas_environment.register_envs(enable_hardware=args["enable_hardware"], is_remote=True, task_name=args["task"],
                                    render=True)

    config_name = args['file']
    args["play"] = True

    print('Loading config: ', config_name)
    env = gym.make("AssemblyMultiArm-v0")

    env_configurations.register("rlgpu", {"vecenv_type": "RAY", "env_creator": lambda **kwargs: env})
    algorithm_name = 'ours'  # 'diffusion_policy', 'ours', 'luo'
    failure_recovery = True
    # without regrasping
    if algorithm_name == 'ours':
        # checkpoint = '../../diffusion_policy_v1/data/outputs/2024.08.25/13.40.46_train_dit_hybrid_fmb/checkpoints/latest.ckpt'
        # with regrasping
        # checkpoint = '../../diffusion_policy_v1/data/outputs/2024.08.25/16.46.24_train_dit_hybrid_fmb/checkpoints/latest.ckpt'
        # checkpoint = '../../diffusion_policy_v1/data/outputs/2024.09.01/18.38.28_train_dit_hybrid_fmb/checkpoints/latest.ckpt'  # good
        checkpoint = '../../diffusion_policy_v1/data/outputs/2024.09.12/14.59.58_train_dit_hybrid_fmb/checkpoints/latest.ckpt'  # good with parameter
    elif algorithm_name == 'diffusion_policy':
        checkpoint = '../../diffusion_policy_v1/data/outputs/2024.09.11/23.13.55_train_diffusion_unet_hybrid_fmb/checkpoints/latest.ckpt'
    elif algorithm_name == 'luo':
        checkpoint = '../../diffusion_policy_v1/data/outputs/2024.09.12/16.13.50_train_dit_hybrid_fmb/checkpoints/latest.ckpt'

    output_dir = 'data/hl_eval_output'
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    if algorithm_name == 'ours':
        cfg.task.dataset.zarr_path = 'logs/assembly_dataset'
    elif algorithm_name == 'diffusion_policy':
        cfg.task.dataset.zarr_path = 'logs/assembly_dataset_ll'
    # print('cfg: ', cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device('cuda:0')
    policy.to(device)
    policy.eval()
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer().to(device)
    # print('normalizer: ', normalizer.params_dict)
    policy.set_normalizer(normalizer)

    downsampling_index = np.linspace(0, 256 - 1, 96, dtype=int)
    x_indices, y_indices = np.meshgrid(downsampling_index, downsampling_index, indexing='ij')

    stop = False
    step_counter = 0
    bs = 1
    n_obs_steps = 2
    img_size = 256
    use_image = True  # False
    scripted_step_type = '' # 'finegrained'

    image_queue = deque(maxlen=2)
    state_queue = deque(maxlen=2)

    cppf2 = CPPF2_wrapper(cppf2_relative_path='../../CPPF2_assembly/', )
    stage_sequence_keys_list = list(stage_sequence.keys())
    OBJECT_DICT_keys_list = list(OBJECT_DICT.keys())
    # Reset
    last_selected_primitive = 'AssemblyReset'
    stage_name = last_selected_primitive
    selected_obj_id = 0
    desired_tool_T_world = None
    init_cfg = task_from_name(stage_name, selected_obj_id=int(selected_obj_id),
                                desired_tool_T_world=desired_tool_T_world)
    print("Running stage: " + str(stage_name), init_cfg.selected_obj_id)
    env.set_mode(stage_name, init_cfg)
    env.reset()

    while not stop:
        if step_counter == 0 or last_selected_primitive == 'AssemblyReset':
            obs = {'image': np.ones((128, 128, 1)), 'policy': np.ones((9 + 6))}
            state = obs['policy']
            if use_image:
                read_image = obs['image'] if 'image' in obs else np.ones((128, 128, 1))
                read_image = cv2.resize(read_image, (img_size, img_size))[
                    ..., None]  # to be consistent with diffusion policy buffer
            else:
                read_image = np.ones((img_size, img_size, 1))
            image = read_image[x_indices, y_indices]
            image_cat = np.concatenate([image, image], axis=-1)  # ['obs/side_1_depth', 'obs/wrist_1_depth']
            state_cat = np.concatenate((np.ones((7,)), np.ones((6,)), np.ones((6,)), np.ones((1,))), axis=0)
            image_queue.append(image_cat)
            # image_queue.append(image_cat)
            state_queue.append(state_cat)
            # state_queue.append(state_cat)
        # else:
        if algorithm_name in ['diffusion_policy']:
            obs = env.get_high_level_policy_state(last_selected_primitive=last_selected_primitive, use_image=True)
        else:
            obs = env.get_high_level_policy_state(last_selected_primitive=last_selected_primitive)
        print('-------------------------- step: {}, hl_obs: {}'.format(step_counter, obs['policy']))
        state = np.round(obs['policy'], decimals=1)
        if use_image:
            read_image = obs['image'] if 'image' in obs else np.ones((128, 128, 1))
            read_image = cv2.resize(read_image, (img_size, img_size))[
                ..., None]  # to be consistent with diffusion policy buffer
        else:
            read_image = np.ones((img_size, img_size, 1))
        image = read_image[x_indices, y_indices]
        image_cat = np.concatenate([image, image], axis=-1)  # ['obs/side_1_depth', 'obs/wrist_1_depth']
        state_cat = np.concatenate((state[:7], np.ones((6,)), state[9:15], np.ones((1,))), axis=0)
        image_queue.append(image_cat)
        state_queue.append(state_cat)

        batch = {'image': torch.Tensor(np.asarray(list(image_queue)))[None].permute(0, 1, 4, 2, 3),
                 'agent_pos': torch.Tensor(np.asarray(list(state_queue)))[None]}
        # if step_counter == 0:
        #     loaded_batch = torch.load('/home/rlab/Programs/long-horizon-assembly/packages/diffusion_policy_v1/4_[[0, 0, 1]].pt'.format(''), weights_only=False)
        # elif step_counter == 1:
        #     loaded_batch = torch.load('/home/rlab/Programs/long-horizon-assembly/packages/diffusion_policy_v1/3_[[0, 1, 8]].pt'.format(''), weights_only=False)
        # elif step_counter == 2:
        #     loaded_batch = torch.load('/home/rlab/Programs/long-horizon-assembly/packages/diffusion_policy_v1/1_[[1, 8, 7]].pt'.format(''), weights_only=False)
        # elif step_counter == 3:
        #     loaded_batch = torch.load('/home/rlab/Programs/long-horizon-assembly/packages/diffusion_policy_v1/0_[[8, 7, 1]].pt'.format(''), weights_only=False)
        # elif step_counter == 4:
        #     loaded_batch = torch.load('/home/rlab/Programs/long-horizon-assembly/packages/diffusion_policy_v1/5_[[7, 1, 8]].pt'.format(''), weights_only=False)
        # # elif step_counter == 5:
        # else:
        #     loaded_batch = torch.load('/home/rlab/Programs/long-horizon-assembly/packages/diffusion_policy_v1/1_[[1, 8, 7]].pt'.format(''), weights_only=False)
        # compare_batch(batch_1=batch, batch_2=loaded_batch, step_counter=step_counter)
        # batch = loaded_batch
        # for k, v in batch.items():
        #     print(k, v.shape)
        desired_tool_T_world = None
        grasp_type = None
        stop_grasp_pose = False
        if failure_recovery and last_selected_primitive not in ['AssemblyScriptedPlaceHorizontal']:
            display_menu(input_dict=OBJECT_DICT)
            selected_obj_id = input('Select object manually ([0-8], q for stop): ').lower()
            if selected_obj_id in ['q', 'Q']:
                stop = True
                return
            int_selected_obj_id = int(selected_obj_id)
            true_ee_xyz = obs['policy'][:3]
            obs_pose = env.get_image_for_pose_estimation()
            desired_tool_T_world, grasp_type = cppf2.inference(image_rgba=obs_pose['image_pose_estimation'], selected_obj_name=OBJECT_DICT_keys_list[int_selected_obj_id],)
            desired_ee_xyz = list(desired_tool_T_world[0])
            print('desired_tool_T_world[:3]: {}, true_ee_xyz: {}'.format(desired_ee_xyz, true_ee_xyz))
            diff_ee = desired_ee_xyz - true_ee_xyz
            distance_ee = np.linalg.norm(diff_ee)
            print('distance_ee: ', distance_ee, diff_ee)
            grasp_pose_correct = input(
                'tool_T_world: {}, grasp type: {}, correct? (y/n/b/o/c/s/q)'.format(desired_tool_T_world, grasp_type)).lower()

            if distance_ee > 0.1:
                if grasp_type in ['grasp']:
                    selected_primitive_id = 1
                else:
                    selected_primitive_id = 9
                int_selected_primitive_id = int(selected_primitive_id)
                stop_grasp_pose = True

        if not stop_grasp_pose:
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            action_dict = policy.predict_action(batch)
            # pred_only_action = action_dict['action'].float()  # unnormalized [:, 0, :]  # (64, 8, 7) -> (1, 7)
            pred_action = action_dict['action_pred'].float()  # unnormalized  # (64, 16, 7)
            if 'continous_action_pred' in action_dict:
                continous_action_pred = action_dict['continous_action_pred'].float()
                print('continous_action_pred: ', continous_action_pred)
            if algorithm_name in ['diffusion_policy']:
                # print('pred_action.shape: ', pred_action.shape, pred_action)
                diffusion_pred_action = pred_action[0, n_obs_steps].cpu()
                print('diffusion_pred_action: ', diffusion_pred_action)
                env.execute_low_level_action(diffusion_pred_action.cpu())
                continue
            # print('pred_action: ', pred_action, pred_action.shape)
            # gt_action = batch['primitive_id'][:, policy.n_obs_steps].float()
            # print('gt_action: ', gt_action, gt_action.shape)
            # print('diff: ', gt_action - pred_action)
            display_menu(input_dict=stage_sequence)
            selected_primitive_id = input('Predicted action {}, is the correct? (y/n/b/o/c/s/q): '.format(pred_action)).lower()
            selected_obj_id = 0
            if selected_primitive_id in ['q', 'Q']:
                stop = True
                return
            elif selected_primitive_id in ['s', 'S']:
                print('Reselect primitive')
                continue
            elif selected_primitive_id in ['n', 'N']:
                selected_primitive_id = input('Select Primitive manually ([0-9], q for stop): ').lower()
                if selected_primitive_id in ['q', 'Q']:
                    stop = True
                    return
            elif selected_primitive_id in ['b', 'B']:
                env.rollback()
                continue
            elif selected_primitive_id in ['o', 'O']:
                env.open_gripper()
                continue
            elif selected_primitive_id in ['c', 'C']:
                env.close_gripper()
                continue
            else:
                selected_primitive_id = pred_action
            int_selected_primitive_id = int(selected_primitive_id)

            if stage_sequence_keys_list[int_selected_primitive_id] in [#'AssemblyMoveToBoardFromFixture', 'AssemblyMoveToBoardFromGrasp',
                                    'AssemblyScriptedGrasp',
                                    'AssemblyScriptedRegrasp',
                                    'AssemblyScriptedGraspHorizontal',
                                    'AssemblyScriptedInsert', 'AssemblyInsert',
            'AssemblyScriptedPlace', 'AssemblyScriptedPlaceHorizontal']:
                display_menu(input_dict=OBJECT_DICT)
                selected_obj_id = input('Select object manually ([0-8], q for stop): ').lower()
                if selected_obj_id in ['q', 'Q']:
                    stop = True
                    return
            int_selected_obj_id = int(selected_obj_id)

            while not stop_grasp_pose and (int_selected_primitive_id in [1, 9]):
                # replace with obejct pose estimation method
                # img4pose_est = obs['image_pose_estimation']
                # cppf2.load_model(selected_obj_name=OBJECT_DICT_keys_list[int_selected_obj_id],)
                obs_pose = env.get_image_for_pose_estimation()
                desired_tool_T_world, grasp_type = cppf2.inference(image_rgba=obs_pose['image_pose_estimation'], selected_obj_name=OBJECT_DICT_keys_list[int_selected_obj_id],)
                grasp_pose_correct = input(
                    'tool_T_world: {}, grasp type: {}, correct? (y/n/b/o/c/s/q)'.format(desired_tool_T_world, grasp_type)).lower()
                if grasp_pose_correct in ['q', 'Q']:
                    stop_grasp_pose = True
                    stop = True
                    return
                elif grasp_pose_correct in ['s', 'S']:
                    print('Reselect grasp pose')
                    desired_tool_T_world = None
                    continue
                elif grasp_pose_correct in ['n', 'N']:
                    desired_tool_T_world = None
                    grasp_type = input('Select object pose ([vertical: 0, horizontal: 1], q for stop): ').lower()
                    if grasp_type in ['q', 'Q']:
                        stop_grasp_pose = True
                        stop = True
                        return
                    elif int(grasp_type) == 0:
                        selected_primitive_id = 1  # AssemblyScriptedGrasp
                    elif int(grasp_type) == 1:
                        selected_primitive_id = 9  # AssemblyScriptedGraspHorizontal
                    else:  #
                        grasp_type = None
                        print('Invalid object pose, retry.')
                    break
                elif grasp_pose_correct in ['b', 'B']:
                    env.rollback()
                    continue
                elif selected_primitive_id in ['o', 'O']:
                    env.open_gripper()
                    continue
                elif selected_primitive_id in ['c', 'C']:
                    env.close_gripper()
                    continue
                else:
                    if grasp_type in ['grasp']:
                        selected_primitive_id = 1
                    else:  # grasp_type in ['grasp']:
                        selected_primitive_id = 9
                    stop_grasp_pose = True
        # desired_tool_T_world = None
        int_selected_primitive_id = int(selected_primitive_id)

        try:
            selected_primitive = stage_sequence_keys_list[int_selected_primitive_id]
        except Exception as e:
            print(e)
            continue

        stage_name = selected_primitive
        init_cfg = task_from_name(stage_name, selected_obj_id=int_selected_obj_id,
                                  desired_tool_T_world=desired_tool_T_world)
        print("Running stage: " + str(stage_name), init_cfg.selected_obj_id)
        env.set_mode(stage_name, init_cfg)
        if selected_primitive in ['AssemblyReset']:
            env.reset()
        elif selected_primitive not in ['AssemblyInsert']:
            if scripted_step_type == 'finegrained':
                env.finegrained_scripted_step(stage_name)
            else:
                env.scripted_step(stage_name)
        else:
            if scripted_step_type == 'finegrained':
                env.finegrained_scripted_step(stage_name)
            else:
                env.scripted_step(stage_name)
            checkpoint_path = stage_sequence[selected_primitive]
            with open(config_name, 'r') as stream:
                config = yaml.safe_load(stream)
                if args['num_actors'] > 0:
                    config['params']['config']['num_actors'] = args['num_actors']

                if args['seed'] > 0:
                    config['params']['seed'] = args['seed']
                    config['params']['config']['env_config']['seed'] = args['seed']

                if checkpoint_path is not None:
                    args["checkpoint"] = checkpoint_path
                runner = Runner()
                runner.algo_factory.register_builder('my_agent', lambda **kwargs: MyA2CAgent(**kwargs))
                runner.player_factory.register_builder('my_agent', lambda **kwargs: MyPpoPlayerContinuous(**kwargs))
                runner.load(config)
                runner.run(args)

        step_counter += 1
        demo_path = os.path.join(env.demo_dir,
                                 '{:04d}_{}_{}.pkl'.format(env.demo_idx, env.current_datetime_str, selected_primitive))
        env.demo = [[obs, last_selected_primitive]]
        with open(demo_path, 'wb') as file:
            pickle.dump(env.demo, file)
            print('Save demo to {}'.format(demo_path))
        last_selected_primitive = selected_primitive

if __name__ == '__main__':
    run_high_level_pipeline()


