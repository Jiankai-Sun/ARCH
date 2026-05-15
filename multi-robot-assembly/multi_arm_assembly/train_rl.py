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
parser.add_argument("--use-wandb", type=int, default=1,
    help="whether to use wandb or not")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv
from multi_arm_assembly.rl_components.isaac_rlgames_wrapper import MyRlGamesVecEnvWrapper

import multi_arm_assembly.isaac_lab_impedance as isaac_lab_impedance
import multi_arm_assembly.direct_isaac_lab_position as direct_isaac_lab_position

import pathlib
import logging
import time
from multi_arm_assembly.utils import log, get_scripted_actions
from multi_arm_assembly.rl_components.my_a2c_model import MyA2CContinuousLogStd
from multi_arm_assembly.rl_components.my_network import MyNetworkBuilder 
from multi_arm_assembly.rl_components.my_agent import MyA2CAgent
from multi_arm_assembly.rl_components.my_player import MyPpoPlayerContinuous
import multi_arm_assembly.cartpole_env as cartpole_env

from rl_games.algos_torch import model_builder

def main():
    """Train with RL-Games agent."""
    # parse seed from command line
    args_cli_seed = args_cli.seed
    model_builder.register_network('my_network', MyNetworkBuilder)
    model_builder.register_model('my_actor_model', lambda network, **kwargs: MyA2CContinuousLogStd(network))
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    print(agent_cfg)
    # override from command line
    if args_cli_seed is not None:
        agent_cfg["params"]["seed"] = args_cli_seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
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
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
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

        if not args_cli.play and args_cli.use_wandb:
            wandb.init(
                project=args_cli.wandb_project_name,
                name="run_{}".format(datetime.now().strftime("%Y%m%d%H%M%S")),
                entity=args_cli.wandb_entity,
                sync_tensorboard=True,
                config=args_cli,
                monitor_gym=True,
                save_code=True,
            )
        
        # train the agent
        runner.run({"train": not args_cli.play, "play": args_cli.play, "sigma": None, "checkpoint": args_cli.checkpoint})

        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
