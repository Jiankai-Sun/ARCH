if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
os.environ["WANDB_API_KEY"] = "b00b2711e75723b6df804b383842ad17c46a84b0"
import tqdm
import numpy as np
from datetime import datetime
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

def fmb_collate_fn(batch):
    # data = {'obs': {'image': [], 'agent_pos': []}, 'action': []} # current element and next element in the list
    image_list = []
    agent_pos_list = []
    action_list = []
    trajectory_id = []
    # print(0, len(batch))
    for i in range(len(batch)):
        # print(batch[i][0]['obs']['image'].shape)
        image_list.append(batch[i][0]['obs']['image'][None])
        agent_pos_list.append(batch[i][0]['obs']['agent_pos'][None])
        action_list.append(batch[i][0]['action'][None])
        trajectory_id.append(batch[i][1])
    image_tensor = torch.cat(image_list, dim=0)
    agent_pos_tensor = torch.cat(agent_pos_list, dim=0)
    action_tensor = torch.cat(action_list, dim=0)
    data = {'obs': {'image': image_tensor, 'agent_pos': agent_pos_tensor},
            'action': action_tensor}
    # print(data['obs']['image'].shape, data['obs']['agent_pos'].shape, data['action'].shape, len(trajectory_id), len(trajectory_id[0]))
    # torch.Size([64, 16, 3, 96, 96]) torch.Size([64, 16, 10]) torch.Size([64, 16, 7]) 64 16
    # print(1, trajectory_id)
    # print(2, len(data))
    # print(1, len(trajectory_id), len(data)) # data['action'].shape)
    return data, trajectory_id

class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        # multi object: board 3
        # self.selected_train_trajectory_id = ['trajectory_1_102.npy_000056', 'trajectory_1_102.npy_000057', 'trajectory_1_102.npy_000058', 'trajectory_1_102.npy_000059', 'trajectory_1_102.npy_000060', 'trajectory_1_102.npy_000061', 'trajectory_1_102.npy_000062', 'trajectory_1_102.npy_000063', 'trajectory_1_102.npy_000064', 'trajectory_1_102.npy_000065', 'trajectory_1_102.npy_000066', 'trajectory_1_102.npy_000067', 'trajectory_1_102.npy_000068', 'trajectory_1_102.npy_000069', 'trajectory_1_102.npy_000070', 'trajectory_1_102.npy_000071']
        # self.selected_val_trajectory_id = ['trajectory_1_0.npy_000174', 'trajectory_1_0.npy_000175', 'trajectory_1_0.npy_000176', 'trajectory_1_0.npy_000177', 'trajectory_1_0.npy_000178', 'trajectory_1_0.npy_000179', 'trajectory_1_0.npy_000180', 'trajectory_1_0.npy_000181', 'trajectory_1_0.npy_000182', 'trajectory_1_0.npy_000183', 'trajectory_1_0.npy_000184', 'trajectory_1_0.npy_000185', 'trajectory_1_0.npy_000186', 'trajectory_1_0.npy_000187', 'trajectory_1_0.npy_000188', 'trajectory_1_0.npy_000189']
        # single object
        self.selected_train_trajectory_id = ['1_L_S_6_horizontal_y_4.npy_000105', '1_L_S_6_horizontal_y_4.npy_000106', '1_L_S_6_horizontal_y_4.npy_000107', '1_L_S_6_horizontal_y_4.npy_000108', '1_L_S_6_horizontal_y_4.npy_000109', '1_L_S_6_horizontal_y_4.npy_000110', '1_L_S_6_horizontal_y_4.npy_000111', '1_L_S_6_horizontal_y_4.npy_000112', '1_L_S_6_horizontal_y_4.npy_000113', '1_L_S_6_horizontal_y_4.npy_000114', '1_L_S_6_horizontal_y_4.npy_000115', '1_L_S_6_horizontal_y_4.npy_000116', '1_L_S_6_horizontal_y_4.npy_000117', '1_L_S_6_horizontal_y_4.npy_000118', '1_L_S_6_horizontal_y_4.npy_000119', '1_L_S_6_horizontal_y_4.npy_000120']
        self.selected_val_trajectory_id = ['1_L_L_4_horizontal_n_12.npy_000000', '1_L_L_4_horizontal_n_12.npy_000001', '1_L_L_4_horizontal_n_12.npy_000002', '1_L_L_4_horizontal_n_12.npy_000003', '1_L_L_4_horizontal_n_12.npy_000004', '1_L_L_4_horizontal_n_12.npy_000005', '1_L_L_4_horizontal_n_12.npy_000006', '1_L_L_4_horizontal_n_12.npy_000007', '1_L_L_4_horizontal_n_12.npy_000008', '1_L_L_4_horizontal_n_12.npy_000009', '1_L_L_4_horizontal_n_12.npy_000010', '1_L_L_4_horizontal_n_12.npy_000011', '1_L_L_4_horizontal_n_12.npy_000012', '1_L_L_4_horizontal_n_12.npy_000013', '1_L_L_4_horizontal_n_12.npy_000014', '1_L_L_4_horizontal_n_12.npy_000015']

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, collate_fn=fmb_collate_fn, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, collate_fn=fmb_collate_fn, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        # print('cfg.task.env_runner: ', cfg.task.env_runner)
        if (cfg.training.rollout_every > 0):
            env_runner: BaseImageRunner
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        if cfg.training.use_wandb:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )
            wandb.run.name = '{}_{}'.format(datetime.now().strftime("%Y%m%d%H%M%S"),
                                               wandb.run.name.split('-')[-1])

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # step_log['test_mean_score'] = 0
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, (batch, train_trajectory_id) in enumerate(tepoch):
                        # device transfer
                        # print(len(trajectory_id), batch['action'].shape)
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        # for k, v in batch['obs'].items():
                        #     print(k, v.shape)
                            # image torch.Size([64, 16, channel, 96, 96])
                            # agent_pos torch.Size([64, 16, state_dim])
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if cfg.training.use_wandb:
                                wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1
                        shown_selected_train_trajectory_id = False
                        for t_i in range(len(train_trajectory_id)):
                            # print('train', train_trajectory_id[t_i])
                            if shown_selected_train_trajectory_id == False and self.selected_train_trajectory_id == train_trajectory_id[
                                t_i]:  # and self.selected_train_trajectory_id[-1] == train_trajectory_id[t_i][-1]:
                                print('train', train_trajectory_id[t_i])
                                selected_traj_index = t_i
                                # break
                                # if selected_traj_index:
                                # train_selected_batch = {
                                #     'obs': {'image': batch['obs']['image'][selected_traj_index][None],
                                #             'agent_pos': batch['obs']['agent_pos'][selected_traj_index][
                                #                 None]},
                                #     'action': batch['action'][selected_traj_index][None], }
                                with torch.no_grad():
                                    # sample trajectory from training set, and evaluate difference
                                    # batch = dict_apply(train_selected_batch, lambda x: x.to(device, non_blocking=True))
                                    obs_dict = batch['obs']

                                    result = self.model.predict_action(obs_dict)
                                    gt_action = batch['action'][selected_traj_index]
                                    pred_action = result['action_pred'][selected_traj_index]
                                    # print(pred_action.shape, gt_action.shape)
                                    # torch.Size([1, 16, 7]) torch.Size([1, 16, 7])
                                    print('train gt_action: ', gt_action)
                                    print('train pred_action: ', pred_action)
                                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                                    step_log['train_action_o_mse_error'] = mse.item()
                                    # (XYZ, RPY, gripper)
                                    r_mse = torch.nn.functional.mse_loss(pred_action[:, 3:6], gt_action[:, 3:6])
                                    step_log['train_action_r_mse_error'] = r_mse.item()
                                    t_mse = torch.nn.functional.mse_loss(pred_action[:, :3], gt_action[:, :3])
                                    step_log['train_action_t_mse_error'] = t_mse.item()
                                    g_mse = torch.nn.functional.mse_loss(pred_action[:, 6:],
                                                                         gt_action[:, 6:])
                                    step_log['train_action_g_mse_error'] = g_mse.item()
                                shown_selected_train_trajectory_id = True

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (cfg.training.rollout_every > 0) and (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    shown_selected_val_trajectory_id = False
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, (batch, val_trajectory_id) in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                                # print('batch: ', batch['obs'].keys()) # dict_keys(['obs' {'image', 'agent_pos'}, 'action'])
                                # print('batch[obs][image]', batch['obs']['image'].shape) # batch[obs][image] torch.Size([64, 16, 3, 96, 96])
                                # print('val', val_trajectory_id[0][0])
                                for t_i in range(len(val_trajectory_id)):
                                    # print('val: ', val_trajectory_id[t_i])
                                    if shown_selected_val_trajectory_id == False and self.selected_val_trajectory_id == val_trajectory_id[t_i]:
                                        selected_traj_index = t_i
                                        # print('val selected_traj_index', val_trajectory_id[t_i])
                                        # with torch.no_grad():
                                        policy.reset()

                                        # obs_dict = {'image': batch['obs']['image'][selected_traj_index][None],
                                        #             'agent_pos': batch['obs']['agent_pos'][selected_traj_index][None]}
                                        obs_dict = batch['obs']
                                        action_dict = policy.predict_action(obs_dict)
                                        gt_action = batch['action'][selected_traj_index] #[None]
                                        pred_action = action_dict['action_pred'][selected_traj_index]
                                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                                        step_log['val_action_o_mse_error'] = mse.item()
                                        # (XYZ, RPY, gripper)
                                        r_mse = torch.nn.functional.mse_loss(pred_action[:, 3:6],
                                                                             gt_action[:, 3:6])
                                        step_log['val_action_r_mse_error'] = r_mse.item()
                                        t_mse = torch.nn.functional.mse_loss(pred_action[:, :3],
                                                                             gt_action[:, :3])
                                        step_log['val_action_t_mse_error'] = t_mse.item()
                                        g_mse = torch.nn.functional.mse_loss(pred_action[:, 6:],
                                                                             gt_action[:, 6:])
                                        step_log['val_action_g_mse_error'] = g_mse.item()
                                        print('val gt_action: ', gt_action)
                                        print('val action_pred: ', pred_action)  # action_dict['action'][0],
                                        shown_selected_val_trajectory_id = True
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                # if (self.epoch % cfg.training.sample_every) == 0:
                # if 'trajectory_1_102.npy_000052' in train_sampling_batch:
                # # index = train_sampling_batch['trajectory_id'].index(train_sampling_batch['trajectory_id'][0])
                # for k, v in train_sampling_batch.items():
                #     if k == 'trajectory_id':
                #         print(k, v)
                #     else:
                #         if isinstance(v, dict):
                #             for k2, v2 in v.items():
                #                 print(k2, v2.shape)
                #         else:
                #             print(k, v.shape)
                    # image torch.Size([64, 16, 3, 96, 96])
                    # agent_pos torch.Size([64, 16, 2])
                    # trajectory_id
                # selected_traj_index = None

                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                # print('self.epoch, cfg.training.checkpoint_every: ', self.epoch, cfg.training.checkpoint_every)
                # print('self._output_dir', self._output_dir)
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    if not 'test_mean_score' in step_log:
                        step_log['test_mean_score'] = 0
                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    # print('step_log: ', step_log)
                    # print('metric_dict: ', metric_dict)
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                if cfg.training.use_wandb:
                    wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
