import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
from datetime import datetime
import gym
import hydra
from omegaconf import OmegaConf
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class IsaacGymImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            render_size=96,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            env_seed=42,
            task_name='Assembly',
            sim_device='cuda:0',
            rl_device='cuda:0',
            graphics_device_id=0,
            # headless=False,
            multi_gpu=False,
            capture_video=False,
            force_render=True,
            capture_video_freq=1464,
            capture_video_len=100,
        ):
        super().__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test
        self.n_envs = n_envs

        # steps_per_render = max(10 // fps, 1)
        def env_fn():
            import isaacgymenvs
            cfg = OmegaConf.create({
                "task": {
                    "include_bg": False,
                    'mode':
                        {
                            'export_scene': False,
                            'export_states': False
                        },
                    'sim': {
                        'dt': 0.016667, 'substeps': 2, 'up_axis': 'z', 'use_gpu_pipeline': True,
                        'gravity': [0.0, 0.0, -9.81], 'add_damping': True,
                        'physx': {'solver_type': 1, 'num_threads': 4, 'num_subscenes': 4, 'use_gpu': True,
                                  'num_position_iterations': 16, 'num_velocity_iterations': 0,
                                  'contact_offset': 0.005, 'rest_offset': 0.0,
                                  'bounce_threshold_velocity': 0.2, 'max_depenetration_velocity': 5.0,
                                  'friction_offset_threshold': 0.01, 'friction_correlation_distance': 0.00625,
                                  'max_gpu_contact_pairs': 1048576, 'default_buffer_size_multiplier': 8.0,
                                  'contact_collection': 1}},
                    'env': {'env_spacing': 0.5, 'franka_depth': 0.5, 'table_height': 0.4, 'franka_friction': 1.0,
                            'table_friction': 0.3, 'table_vertical_offset': -0.17, 'numEnvs': 1,
                            'numObservations': [96, 96, 16], 'numStates': 13, 'numActions': 7,
                            'camera_img_width': 96, 'camera_img_height': 96,
                            'close_and_lift': True,  # close gripper and lift after last step of episode
                            'num_gripper_move_sim_steps': 20,
                            # number of timesteps to reserve for moving gripper before first step of episode
                            'num_gripper_close_sim_steps': 25,
                            # number of timesteps to reserve for closing gripper after last step of episode
                            'num_gripper_lift_sim_steps': 25,
                            # number of timesteps to reserve for lift after last step of episode
                            },
                    'name': 'Assembly', 'physics_engine': 'physx',
                    'randomize': {'joint_noise': 0.0, 'initial_state': 'random', 'plug_bias_y': 0,
                                  'plug_bias_z': 0.08, 'plug_noise_xy': 0.04},
                    'rl': {'max_episode_length': max_steps, 'pos_action_scale': [0.01, 0.01, 0.01],
                           'rot_action_scale': [0.01, 0.01, 0.01], 'force_action_scale': [1.0, 1.0, 1.0],
                           'torque_action_scale': [1.0, 1.0, 1.0], 'unidirectional_rot': True,
                           'unidirectional_force': False, 'clamp_rot': True, 'clamp_rot_thresh': 1e-06,
                           'num_keypoints': 4,
                           'keypoint_scale': 0.5, 'interpen_thresh': 0.001, 'sdf_reward_scale': 10.0,
                           'sdf_reward_num_samples': 5000, 'initial_max_disp': 0.01,
                           'curriculum_success_thresh': 0.6,
                           'curriculum_failure_thresh': 0.3, 'curriculum_height_step': [-0.005, 0.002],
                           'curriculum_height_bound': [-0.005, 0.015], 'close_error_thresh': 0.1,
                           'success_height_thresh': 0.01, 'engagement_bonus': 10.0},
                    'ctrl': {'ctrl_type': 'task_space_impedance',
                             'all': {'jacobian_type': 'geometric', 'gripper_prop_gains': [500, 500],
                                     'gripper_deriv_gains': [2, 2]},
                             'gym_default': {'ik_method': 'dls', 'joint_prop_gains': [40, 40, 40, 40, 40, 40, 40],
                                             'joint_deriv_gains': [8, 8, 8, 8, 8, 8, 8],
                                             'gripper_prop_gains': [500, 500],
                                             'gripper_deriv_gains': [20, 20]},
                             'joint_space_ik': {'ik_method': 'dls', 'joint_prop_gains': [1, 1, 1, 1, 1, 1, 1],
                                                'joint_deriv_gains': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
                             'joint_space_id': {'ik_method': 'dls',
                                                'joint_prop_gains': [40, 40, 40, 40, 40, 40, 40],
                                                'joint_deriv_gains': [8, 8, 8, 8, 8, 8, 8]},
                             'task_space_impedance': {'motion_ctrl_axes': [1, 1, 1, 1, 1, 1],
                                                      'task_prop_gains': [300, 300, 600, 50, 50, 50],
                                                      'task_deriv_gains': [34, 34, 34, 1.4, 1.4, 1.4]},
                             'operational_space_motion': {'motion_ctrl_axes': [1, 1, 1, 1, 1, 1],
                                                          'task_prop_gains': [20, 20, 100, 0, 0, 100],
                                                          'task_deriv_gains': [1, 1, 1, 1, 1, 1]},
                             'open_loop_force': {'force_ctrl_axes': [0, 0, 1, 0, 0, 0]},
                             'closed_loop_force': {'force_ctrl_axes': [0, 0, 1, 0, 0, 0],
                                                   'wrench_prop_gains': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
                             'hybrid_force_motion': {'motion_ctrl_axes': [1, 1, 0, 1, 1, 1],
                                                     'task_prop_gains': [40, 40, 40, 40, 40, 40],
                                                     'task_deriv_gains': [8, 8, 8, 8, 8, 8],
                                                     'force_ctrl_axes': [0, 0, 1, 0, 0, 0],
                                                     'wrench_prop_gains': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}
                             },

                },
            })
            # print('cfg: ', cfg)
            headless=False
            issacgym_envs = isaacgymenvs.make(
                env_seed,
                task_name,
                n_envs,
                sim_device,
                rl_device,
                graphics_device_id,
                headless,
                multi_gpu,
                capture_video,
                force_render,
                cfg=cfg,
            )
            time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_name = f"isaacgym_{time_str}"
            if capture_video:
                issacgym_envs.is_vector_env = True
                issacgym_envs = gym.wrappers.RecordVideo(
                    issacgym_envs,
                    f"videos/{run_name}",
                    step_trigger=lambda step: step % capture_video_freq == 0,
                    video_length=capture_video_len,
                )

            # return MultiStepWrapper(
            #     VideoRecordingWrapper(
            #         issacgym_envs,
            #         video_recoder=VideoRecorder.create_h264(
            #             fps=fps,
            #             codec='h264',
            #             input_pix_fmt='rgb24',
            #             crf=crf,
            #             thread_type='FRAME',
            #             thread_count=1
            #         ),
            #         file_path=None,
            #         steps_per_render=steps_per_render
            #     ),
            #     n_obs_steps=n_obs_steps,
            #     n_action_steps=n_action_steps,
            #     max_episode_steps=max_steps
            # )
            return issacgym_envs

        # env_fns = [env_fn] * n_envs
        env_fns = env_fn
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = env_fn()
        # env = AsyncVectorEnv(env_fns, shared_memory=False,)

        # test env
        # env.reset(seed=env_seeds)
        # import pdb; pdb.set_trace()
        # # x = env.step(torch.Tensor(env.action_space.sample()))
        # x = env.step(torch.Tensor(env.action_space.sample()))
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = self.n_envs  # len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            # start = chunk_idx * n_envs
            # end = min(n_inits, start + n_envs)
            # this_global_slice = slice(start, end)
            # this_n_active_envs = end - start
            # this_local_slice = slice(0,this_n_active_envs)
            
            # this_init_fns = self.env_init_fn_dills[this_global_slice]
            # n_diff = n_envs - len(this_init_fns)
            # if n_diff > 0:
            #     this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            # assert len(this_init_fns) == n_envs

            # init envs
            # env.call_each('run_dill_function',
            #     args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval IsaacGymImageRunner {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            image_buffer = [torch.ones(1, 1, 3, 96, 96)] * 16
            agent_pos_buffer = [torch.ones(1, 1, 15)] * 16
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                # obs_dict = dict_apply(np_obs_dict,
                #     lambda x: torch.from_numpy(x).to(
                #         device=device))
                # print('np_obs_dict: ', np_obs_dict.keys())
                # np_obs_dict:  dict_keys(['obs', 'states'])
                # for k, v in np_obs_dict.items():
                #     print(k, v.shape)
                    # np_obs_dict:  dict_keys(['obs', 'states'])
                    # obs torch.Size([1, 256, 256, 16])
                    # states torch.Size([1, 13])
                # input_image = torch.zeros((1, 16, 3, 96, 96))
                # input_state = torch.zeros((1, 16, 15))
                image_buffer[1:].append(torch.moveaxis(np_obs_dict['obs'][..., :3],-1,1)[None, ...]/255.)  # [1, 1, 3, 96, 96]
                agent_pos_buffer[1:].append(torch.cat((np_obs_dict['states'], torch.zeros(1, 15-np_obs_dict['states'].shape[-1]).to(np_obs_dict['states'].device)), dim=-1)[None, ...])  # [1, 1, 15]
                # print('max, shape, ', np_obs_dict['obs'][..., :3].max(), np_obs_dict['obs'].shape) # 255, [1, 96， 96， 16】
                obs_dict = {'image': torch.cat(image_buffer, dim=1).to(np_obs_dict['states'].device),
                            'agent_pos': torch.cat(agent_pos_buffer, dim=1).to(np_obs_dict['states'].device)
                            }
                # for k, v in obs_dict.items():
                #     print(k, v.shape)
                # obs_dict = {'image': input_image,
                #             'agent_pos': input_state}

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                # print('action_dict: ', action_dict['action'].shape)
                # action_dict:  torch.Size([1, 8, 7])
                # np_action_dict = dict_apply(action_dict,
                #     lambda x: x.detach().to('cpu').numpy())
                #
                # action = np_action_dict['action']
                action = action_dict['action'][:, 0, :] # (1, 8, 7) -> (1, 7)
                print('action: ', action)
                # step env
                obs, reward, done, info = env.step(action)
                # print('1 done: ', done)
                done = torch.all(done)
                # print('2 done: ', done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # all_video_paths[this_global_slice] = env.render()[this_local_slice]
            # all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        print('n_envs: {}, n_inits: {}, n_chunks: {}'.format(n_envs, n_inits, n_chunks))
        # n_envs: 56, n_inits: 56, n_chunks: 1
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            # print('value: ', value)
            try:
                value = np.mean(value)
            except:
                value = None
            log_data[name] = value

        return log_data
