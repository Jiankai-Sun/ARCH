# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import wandb
from omni.isaac.lab.app import AppLauncher
import os
os.environ["WANDB_API_KEY"] = "b00b2711e75723b6df804b383842ad17c46a84b0"
# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("-c", "--checkpoint", required=False, help="path to checkpoint")
parser.add_argument("-p", "--play", required=False, help="play(test) network", action='store_true')
parser.add_argument("-rs", "--run-scripted", required=False, help="play(test) network", action='store_true')
parser.add_argument("-rsm", "--run-scripted-multiarm", required=False, help="play(test) network", action='store_true')
parser.add_argument("--prefix", type=str, default="",
    help="name to tag to the end of the project name")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb-project-name", type=str, default="rl_games",
    help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None,
    help="the entity (team) of wandb's project")
parser.add_argument("--use-wandb", type=int, default=0,
    help="whether to use wandb or not")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv
from multi_arm_assembly.rl_components.isaac_rlgames_wrapper import MyRlGamesVecEnvWrapper
import multi_arm_assembly.isaac_lab_impedance as isaac_lab_impedance
import multi_arm_assembly.direct_isaac_lab_position as direct_isaac_lab_position
"""Rest everything follows."""

import gymnasium as gym
import math
import os
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

import pathlib
import logging
import time
from multi_arm_assembly.utils import log, get_scripted_actions
from multi_arm_assembly.rl_components.my_a2c_model import MyA2CContinuousLogStd
from multi_arm_assembly.rl_components.my_network import MyNetworkBuilder 
from multi_arm_assembly.rl_components.my_agent import MyA2CAgent
from multi_arm_assembly.rl_components.my_player import MyPpoPlayerContinuous
# import multi_arm_assembly.cartpole_env as cartpole_env
from multi_arm_assembly.utils import log_list, task_from_name, display_menu, ROOT_DIR, stage_sequence, set_seed
import torch
import numpy as np
set_seed(123)
from rl_games.algos_torch import model_builder
from collections import deque
import hydra
import dill
import sys
import pickle
sys.path.insert(0, '../../diffusion_policy_v1/')
from diffusion_policy.common.pytorch_util import dict_apply
from matplotlib import pyplot as plt
import cv2
import multiprocessing
# Set the start method for multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def launch_simulation_app(selected_primitive, selected_obj_id):
    stage_name = list(stage_sequence.keys())[int(selected_primitive)]
    checkpoint_path = stage_sequence[stage_name]
    if stage_name in ['AssemblyInsert', 'BeamInsert', 'StoolInsert']:
        args_cli.checkpoint == checkpoint_path
    elif stage_name == "AssemblyReset":
        print('Resetting ...')
        return None
    stage_name = stage_name + '-v0'
    # parse seed from command line
    args_cli_seed = args_cli.seed
    model_builder.register_network('my_network', MyNetworkBuilder)
    model_builder.register_model('my_actor_model', lambda network, **kwargs: MyA2CContinuousLogStd(network))
    # parse configuration
    env_cfg = parse_env_cfg(
        task_name=stage_name, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(stage_name, "rl_games_cfg_entry_point")
    print('agent_cfg: ', agent_cfg)
    # override from command line
    if args_cli_seed is not None:
        agent_cfg["params"]["seed"] = args_cli_seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name",
                                                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # max iterations
    if args_cli.max_iterations:
        agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_iterations

    # dump the configuration into log-directory
    # dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    # dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(stage_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": env.max_episode_length,  # args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rl-games
    env = MyRlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    if(args_cli.run_scripted or args_cli.run_scripted_multiarm):
        env.reset()
        scripted_actions = get_scripted_actions(env.num_envs, num_robots=1+int(args_cli.run_scripted_multiarm))
        for scripted_action in scripted_actions:
            env.step(scripted_action)
        env.close()
    else:
        # register the environment to rl-games registry
        # note: in agents configuration: environment name must be "rlgpu"
        vecenv.register(
            "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        # set number of actors into agent config
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        agent_cfg["params"]["config"]["minibatch_size"] = env.unwrapped.num_envs*2

        # create runner from rl-games
        runner = Runner(IsaacAlgoObserver())

        runner.algo_factory.register_builder('my_agent', lambda **kwargs : MyA2CAgent(**kwargs))
        runner.player_factory.register_builder('my_agent', lambda **kwargs : MyPpoPlayerContinuous(**kwargs))
        runner.load(agent_cfg)

        # set seed of the env
        env.seed(agent_cfg["params"]["seed"])
        # reset the agent and env
        runner.reset()

        # train the agent
        runner.run({"train": False, "play": args_cli.play, "sigma": None, "checkpoint": args_cli.checkpoint})
        print('finished running')
        # close the simulator
        env.close()
        print('env closed')
        # close sim app
        simulation_app.close()
        
def main():
    """Train with RL-Games agent."""

    demo_dir = os.path.join(ROOT_DIR, 'multi_arm_assembly/logs/demos_isaaclab/0/')
    os.makedirs(demo_dir, exist_ok=True)
    demo_idx = 0
    algorithm_name = 'ours'  # 'diffusion_policy', 'ours', 'luo', 'collect_demo'
    failure_recovery = True
    # without regrasping
    if algorithm_name == 'ours':
        checkpoint = '../../diffusion_policy_v1/data/outputs/2025.04.28/19.00.14_train_dit_hybrid_fmb/checkpoints/latest.ckpt'  # good with parameter
    elif algorithm_name == 'diffusion_policy':
        checkpoint = '../../diffusion_policy_v1/data/outputs/2024.09.11/23.13.55_train_diffusion_unet_hybrid_fmb/checkpoints/latest.ckpt'
    elif algorithm_name == 'luo':
        checkpoint = '../../diffusion_policy_v1/data/outputs/2024.09.12/16.13.50_train_dit_hybrid_fmb/checkpoints/latest.ckpt'
    else:
        checkpoint = None
    output_dir = 'data/hl_eval_output'
    # load checkpoint
    if checkpoint is not None:
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
    # with KeystrokeCounter() as key_counter:
    step_counter = 0
    bs = 1
    n_obs_steps = 2
    img_size = 256
    use_image = True  # False
    scripted_step_type = '' # 'finegrained'

    image_queue = deque(maxlen=2)
    state_queue = deque(maxlen=2)
    
    # desired_tool_T_world = None
    # init_cfg = task_from_name(stage_name, selected_obj_id=int(selected_obj_id),
    #                         desired_tool_T_world=desired_tool_T_world)
    # print("Running stage: " + str(stage_name), init_cfg.selected_obj_id)
    selected_primitive = 0
    while not stop:
        if checkpoint is not None:
            if step_counter == 0:
                selected_primitive = 0
                
                obs = {'image': np.ones((128, 128, 1)) * int(selected_primitive) / len(stage_sequence), 'policy': np.ones((9 + 6)) * int(selected_primitive) / len(stage_sequence)}
                state = obs['policy']
                read_image = np.ones((img_size, img_size, 1)) * int(selected_primitive) / len(stage_sequence)
                image = read_image[x_indices, y_indices]
                image_cat = np.concatenate([image, image], axis=-1)  # ['obs/side_1_depth', 'obs/wrist_1_depth']
                state_cat = np.concatenate((np.ones((7,)), np.ones((6,)), np.ones((6,)), np.ones((1,))), axis=0)
                image_queue.append(image_cat)
                state_queue.append(state_cat)
            else:
                read_image = np.ones((img_size, img_size, 1)) * int(selected_primitive) / len(stage_sequence)
                image = read_image[x_indices, y_indices]
                image_cat = np.concatenate([image, image], axis=-1)  # ['obs/side_1_depth', 'obs/wrist_1_depth']
                state = np.ones((9 + 6)) * int(selected_primitive) / len(stage_sequence)
                state_cat = np.concatenate((state[:7], np.ones((6,)), state[9:15], np.ones((1,))), axis=0)
                image_queue.append(image_cat)
                state_queue.append(state_cat)
                batch = {'image': torch.Tensor(np.asarray(list(image_queue)))[None].permute(0, 1, 4, 2, 3),
                         'agent_pos': torch.Tensor(np.asarray(list(state_queue)))[None]}
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                action_dict = policy.predict_action(batch)
                pred_action = action_dict['action_pred'].float()
                if 'continous_action_pred' in action_dict:
                    continous_action_pred = action_dict['continous_action_pred'].float()
                    print('continous_action_pred: ', continous_action_pred)
                display_menu(input_dict=stage_sequence)
                selected_primitive_id = input('Predicted action {}, is the correct? (y/n/b/o/c/s/q): '.format(pred_action)).lower()

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
                    # env.rollback()
                    continue
                elif selected_primitive_id in ['o', 'O']:
                    # env.open_gripper()
                    continue
                elif selected_primitive_id in ['c', 'C']:
                    # env.close_gripper()
                    continue
                else:
                    selected_primitive_id = pred_action
                selected_primitive = int(selected_primitive_id)
        else:
            display_menu(input_dict=stage_sequence)
            selected_primitive = input('Select Primitive ([0-9], q for stop): ').lower()

            obs = {'image': np.ones((128, 128, 1)) * int(selected_primitive) / len(stage_sequence), 'policy': np.ones((9 + 6)) * int(selected_primitive) / len(stage_sequence)}
            # state = np.round(obs['policy'], decimals=1)
            # read_image = np.ones((img_size, img_size, 1))
            # image = read_image[x_indices, y_indices]
            # image_cat = np.concatenate([image, image], axis=-1)  # ['obs/side_1_depth', 'obs/wrist_1_depth']
            # state_cat = np.concatenate((np.ones((7,)), np.ones((6,)), np.ones((6,)), np.ones((1,))), axis=0)
            # image_queue.append(image_cat)
            # state_queue.append(state_cat)
        if selected_primitive in ['q', 'Q']:
            stop = True
            break

        # selected_obj_id = 0
    #     if selected_primitive in ['AssemblyScriptedInsert', 'AssemblyInsert', "BeamScriptedGrasp",
    # "BeamMoveToBoard", "BeamScriptedPlace", "BeamScriptedInsert", "BeamInsert", "StoolScriptedGrasp",
    # "StoolMoveToBoard", "StoolScriptedPlace", "StoolScriptedInsert", "StoolInsert"]:
        selected_obj_id = input('Select Object manually ([0-8], q for stop): ').lower()
        if selected_obj_id in ['q', 'Q']:
            stop = True
            break
        print('selected_primitive, len(stage_sequence): ', selected_primitive, len(stage_sequence))

        process = multiprocessing.Process(target=launch_simulation_app, args=(selected_primitive, selected_obj_id))
        process.start()
        # thread = threading.Thread(target=launch_simulation_app, args=(selected_primitive,))
        # thread.start()
        process.join()
        print("Process execution finished.")
    
        step_counter += 1
        current_datetime_str = datetime.now().strftime('%Y%m%d%H%M%S')
        demo_path = os.path.join(demo_dir, '{:04d}_{}_{}.pkl'.format(step_counter, current_datetime_str, selected_primitive))
        demo = [[obs, selected_primitive]]
        with open(demo_path, 'wb') as file:
            pickle.dump(demo, file)
            print('Save demo to {}'.format(demo_path))
        


if __name__ == "__main__":
    # run the main function
    main()

