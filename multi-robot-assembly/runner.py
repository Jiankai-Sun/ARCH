from distutils.util import strtobool
import argparse, os, yaml
import gym

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, required=False, 
                    help="random seed, if larger than 0 will overwrite the value in yaml config")
    ap.add_argument("-tf", "--tf", required=False, help="run tensorflow runner", action='store_true')
    ap.add_argument("-t", "--train", required=False, help="train network", action='store_true')
    ap.add_argument("-p", "--play", required=False, help="play(test) network", action='store_true')
    ap.add_argument("-c", "--checkpoint", required=False, help="path to checkpoint")
    ap.add_argument("-rr", "--real-robot", action="store_true", help="path to config")
    ap.add_argument("-na", "--num_actors", type=int, default=1, required=False,
                    help="number of envs running in parallel, if larger than 0 will overwrite the value in yaml config")
    ap.add_argument("-s", "--sigma", type=float, required=False, help="sets new sigma value in case if 'fixed_sigma: True' in yaml config")
    ap.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    ap.add_argument("--wandb-project-name", type=str, default="rl_games",
        help="the wandb's project name")
    ap.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    ap.add_argument("-f", "--file", required=True, help="path to config")

    ap.add_argument("--prefix", type=str, default="",
        help="name to tag to the end of the project name")
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    args = vars(ap.parse_args())
    config_name = args['file']

    print('Loading config: ', config_name)
    with open(config_name, 'r') as stream:
        config = yaml.safe_load(stream)
        config["params"]["config"]["env_config"]["real_robot"] = args["real_robot"]
        if args['num_actors'] > 0:
            config['params']['config']['num_actors'] = args['num_actors']

        if args['seed'] > 0:
            config['params']['seed'] = args['seed']
            config['params']['config']['env_config']['seed'] = args['seed']

        from rl_games.torch_runner import Runner

        try:
            import ray
        except ImportError:
            pass
        else:
            ray.init(
                _memory=1024 * 1024 * 1024 * 100,
                object_store_memory=1024*1024*1000,
                _redis_max_memory=1024 * 1024 * 1024 * 100,
                logging_level="DEBUG")

        runner = Runner()
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)

    global_rank = int(os.getenv("RANK", "0"))
    if args["track"] and global_rank == 0:
        import wandb
        wandb.init(
            project=args["wandb_project_name"],
            name="{}:{}".format(args["prefix"], args["file"].split("/")[-1]),
            entity=args["wandb_entity"],
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )

    runner.run(args)

    try:
        import ray
    except ImportError:
        pass
    else:
        ray.shutdown()

    if args["track"] and global_rank == 0:
        wandb.finish()
