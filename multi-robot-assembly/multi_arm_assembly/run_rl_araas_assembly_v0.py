from distutils.util import strtobool
import argparse, os, yaml
import gym
import time
import multi_arm_assembly.araas_environment as araas_environment
from rl_games.common import env_configurations
from rl_games.torch_runner import Runner
from multi_arm_assembly.utils import log, get_scripted_actions, task_from_name
from multi_arm_assembly.rl_components.my_network import MyNetworkBuilder 
from multi_arm_assembly.rl_components.my_a2c_model import MyA2CContinuousLogStd
from multi_arm_assembly.rl_components.my_network import MyNetworkBuilder 
from multi_arm_assembly.rl_components.my_agent import MyA2CAgent
from multi_arm_assembly.rl_components.my_player import MyPpoPlayerContinuous
from rl_games.algos_torch import model_builder

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
home_dir = os.getcwd() + '/../'

import ray
ray.init(logging_level="DEBUG")

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, required=False, 
                    help="random seed, if larger than 0 will overwrite the value in yaml config")
    ap.add_argument("-tf", "--tf", required=False, help="run tensorflow runner", action='store_true')
    ap.add_argument("-t", "--train", required=False, help="train network", action='store_true')
    ap.add_argument("-rr", "--enable-hardware", action="store_true", help="path to config")
    ap.add_argument("-na", "--num_actors", type=int, default=1, required=False,
                    help="number of envs running in parallel, if larger than 0 will overwrite the value in yaml config")
    ap.add_argument("-s", "--sigma", type=float, required=False, help="sets new sigma value in case if 'fixed_sigma: True' in yaml config")
    ap.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    ap.add_argument("--task", type=str, default="gear_in_taskboard",
        help="task name")
    ap.add_argument("--wandb-project-name", type=str, default="rl_games",
        help="the wandb's project name")
    ap.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    ap.add_argument("-f", "--file", default="./myenv_cfg.yaml", help="path to config")
    ap.add_argument("--prefix", type=str, default="",
        help="name to tag to the end of the project name")
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    
    args = vars(ap.parse_args())
    model_builder.register_network('my_network', MyNetworkBuilder)
    model_builder.register_model('my_actor_model', lambda network, **kwargs: MyA2CContinuousLogStd(network))
    araas_environment.register_envs(enable_hardware=args["enable_hardware"], is_remote=True, task_name=args["task"], render=True)
    
    config_name = args['file']
    args["play"] = True
    
    print('Loading config: ', config_name)
    env = gym.make("AssemblyMultiArm-v0")

    env_configurations.register("rlgpu", {"vecenv_type": "RAY", "env_creator": lambda **kwargs: env})

    stage_sequence = {
        "strut_in_elbow": home_dir + "multi_arm_assembly/logs/rl_games/apa_impedance/2024-07-10_12-20-46/nn/apa_impedance.pth",
        "strut_plus_elbow_in_platform": home_dir + "multi_arm_assembly/logs/rl_games/apa_impedance/2024-07-10_13-09-14/nn/apa_impedance.pth",
        # "bolt_in_platform": "",
        # "screw_in_bolt": (None, task_from_name("screw_in_bolt"))
    }

    for stage_name, checkpoint_path in stage_sequence.items():
        init_cfg = task_from_name(stage_name)
        print("Running stage: "+str(stage_name))
        env.set_mode(stage_name, init_cfg)
        if(checkpoint_path is None):
            env.reset()
        else:
            with open(config_name, 'r') as stream:
                config = yaml.safe_load(stream)
                if args['num_actors'] > 0:
                    config['params']['config']['num_actors'] = args['num_actors']

                if args['seed'] > 0:
                    config['params']['seed'] = args['seed']
                    config['params']['config']['env_config']['seed'] = args['seed']

                runner = Runner()
                runner.algo_factory.register_builder('my_agent', lambda **kwargs : MyA2CAgent(**kwargs))
                runner.player_factory.register_builder('my_agent', lambda **kwargs : MyPpoPlayerContinuous(**kwargs))
                runner.load(config)
                runner.run(args)
        print("finished running")

