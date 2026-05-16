from distutils.util import strtobool
import argparse, os, yaml
import gym
import time
import multi_arm_assembly.araas_environment as araas_environment
from rl_games.common import env_configurations
from rl_games.torch_runner import Runner
from multi_arm_assembly.utils import log, get_scripted_actions, task_from_name, display_menu, ROOT_DIR, stage_sequence, set_seed, split_obj_dict, OBJECT_DICT
from multi_arm_assembly.rl_components.my_network import MyNetworkBuilder 
from multi_arm_assembly.rl_components.my_a2c_model import MyA2CContinuousLogStd
from multi_arm_assembly.rl_components.my_network import MyNetworkBuilder 
from multi_arm_assembly.rl_components.my_agent import MyA2CAgent
from multi_arm_assembly.rl_components.my_player import MyPpoPlayerContinuous
from rl_games.algos_torch import model_builder

set_seed(123)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
home_dir = os.getcwd() + '/../'

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
    araas_environment.register_envs(enable_hardware=args["enable_hardware"], is_remote=True, task_name=args["task"], render=True)
    
    config_name = args['file']
    args["play"] = True
    
    print('Loading config: ', config_name)
    env = gym.make("AssemblyMultiArm-v0")

    env_configurations.register("rlgpu", {"vecenv_type": "RAY", "env_creator": lambda **kwargs: env})

    stop = False
    while not stop:
        display_menu(stage_sequence=stage_sequence)
        selected_primitive_id = input('Select Primitive ([0-9], q for stop): ')
        if selected_primitive_id in ['q', 'Q']:
            stop = True
            break
        try:
            selected_primitive = list(stage_sequence.keys())[int(selected_primitive_id)]
        except Exception as e:
            print(e)
            continue
        
        selected_obj_id = 0
        if selected_primitive in ['AssemblyScriptedInsert', 'AssemblyInsert']:
            selected_obj_id = input('Select Object manually ([0-8], q for stop): ').lower()
            if selected_obj_id in ['q', 'Q']:
                stop = True
                break

        stage_name = selected_primitive
        print("Running stage: " + str(stage_name))
        if selected_primitive in ['AssemblyReset']:
            env.reset()
        elif selected_primitive not in ['AssemblyInsert']:
            init_cfg = task_from_name(stage_name, selected_obj_id=selected_obj_id)
            env.set_mode(stage_name, init_cfg)
            env.scripted_step(stage_name)
        else:
            init_cfg = task_from_name(stage_name, selected_obj_id=selected_obj_id)
            env.set_mode(stage_name, init_cfg)
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


