"""
Usage:
python eval_high_level_policy.py --checkpoint data/outputs/2024.08.23/20.23.26_train_dit_hybrid_fmb/checkpoints/latest.ckpt -o data/hl_eval_output
"""
try:
    import isaacgym
    import isaacgymenvs
except ImportError as e:
    print('IsaacGym: ', e)

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import pickle
import tqdm
from torch.utils.data import DataLoader
from diffusion_policy.dataset.base_dataset import BaseImageDataset
# from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import fmb_collate_fn
from diffusion_policy.workspace.train_dit_hybrid_workspace import fmb_collate_fn
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device, split='train'):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.val_dataloader.batch_size = 1
    cfg.val_dataloader.num_workers = 1
    cfg.dataloader.batch_size = 1
    cfg.dataloader.num_workers = 1

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    if (cfg.training.rollout_every > 0):
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=output_dir)
        runner_log = env_runner.run(policy)
    else:
        runner_log = {}

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        normalizer = dataset.get_normalizer().to(device)

        if split == 'train':
            train_dataloader = DataLoader(dataset, collate_fn=fmb_collate_fn, **cfg.dataloader)
            vis_dataloader = train_dataloader
        else:
            # configure validation dataset
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(val_dataset, collate_fn=fmb_collate_fn, **cfg.val_dataloader)
            vis_dataloader = val_dataloader
        policy.set_normalizer(normalizer)

        shown_selected_val_trajectory_id = False
        selected_val_trajectory_id = ['1_L_L_4_horizontal_n_12.npy_000000', '1_L_L_4_horizontal_n_12.npy_000001',
                                       '1_L_L_4_horizontal_n_12.npy_000002', '1_L_L_4_horizontal_n_12.npy_000003',
                                       '1_L_L_4_horizontal_n_12.npy_000004', '1_L_L_4_horizontal_n_12.npy_000005',
                                       '1_L_L_4_horizontal_n_12.npy_000006', '1_L_L_4_horizontal_n_12.npy_000007',
                                       '1_L_L_4_horizontal_n_12.npy_000008', '1_L_L_4_horizontal_n_12.npy_000009',
                                       '1_L_L_4_horizontal_n_12.npy_000010', '1_L_L_4_horizontal_n_12.npy_000011',
                                       '1_L_L_4_horizontal_n_12.npy_000012', '1_L_L_4_horizontal_n_12.npy_000013',
                                       '1_L_L_4_horizontal_n_12.npy_000014', '1_L_L_4_horizontal_n_12.npy_000015']

        val_losses = list()
        seq_name_list = []
        pred_traj_list = []
        pred_only_traj_list = []
        gt_traj_list = []
        gt_img_list = []
        with torch.no_grad():
            with tqdm.tqdm(vis_dataloader, desc=f"Validation",
                           leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, (batch, trajectory_id) in enumerate(tepoch):
                    if batch_idx >= 10:
                        break
                    # print(batch_idx)
                    # print('device: ', device)
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    # for k, v in batch.items():
                    #     print(k, v)
                    # image torch.Size([7, 16, 2, 96, 96])
                    # agent_pos torch.Size([7, 16, 20])
                    # action torch.Size([7, 16, 7])
                    # primitive_id torch.Size([7, 16])
                    loss = policy.compute_loss(batch)
                    val_losses.append(loss)
                    action_dict = policy.predict_action(batch['obs'])
                    pred_only_action = action_dict['action'].float() # unnormalized [:, 0, :]  # (64, 8, 7) -> (1, 7)
                    pred_action = action_dict['action_pred'].float()  # unnormalized  # (64, 16, 7)
                    print('pred_action: ', pred_action, pred_action.shape)
                    gt_action = batch['primitive_id'][:, policy.n_obs_steps].float()
                    print('gt_action: ', gt_action, gt_action.shape)
                    print('diff: ', gt_action - pred_action)
                    print('primitive_id: ', batch['primitive_id'])
                    if abs(gt_action - pred_action) < 0.5:
                        torch.save(batch['obs'], '{}_{}.pt'.format(batch_idx, str(batch['primitive_id'].cpu().tolist()), ))
                    # print('pred_action.shape: ', pred_action.shape)
                    # if (cfg.training.max_val_steps is not None) \
                    #         and batch_idx >= (cfg.training.max_val_steps - 1):
                    #     break
                    # print('batch: ', batch['obs'].keys()) # dict_keys(['obs' {'image', 'agent_pos'}, 'action'])
                    # print('batch[obs][image]', batch['obs']['image'].shape) # batch[obs][image] torch.Size([64, 16, 3, 96, 96])
                    # print('val', val_trajectory_id[0][0])
                    for t_i in range(len(trajectory_id)):
                        # print('val: ', val_trajectory_id[t_i])
                        if shown_selected_val_trajectory_id == False and selected_val_trajectory_id == \
                                trajectory_id[t_i]:
                            selected_traj_index = t_i
                            # print('val selected_traj_index', val_trajectory_id[t_i])
                            # with torch.no_grad():
                            policy.reset()

                            # obs_dict = {'image': batch['obs']['image'][selected_traj_index][None],
                            #             'agent_pos': batch['obs']['agent_pos'][selected_traj_index][None]}
                            obs_dict = batch['obs']
                            action_dict = policy.predict_action(obs_dict)
                            gt_action = batch['action'][selected_traj_index, policy.n_obs_steps].float()  # [None]
                            pred_action = action_dict['action_pred'][selected_traj_index].flaot()
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            runner_log['val_action_mse_error'] = mse.item()
                            print('val gt_action: ', gt_action)
                            print('val action_pred: ', pred_action)  # action_dict['action'][0],
                            print('diff: ', gt_action - pred_action)
                            shown_selected_val_trajectory_id = True
                    seq_name_list.append(trajectory_id)
                    pred_traj_list.append(pred_action.cpu().numpy())
                    pred_only_traj_list.append(pred_only_action.cpu().numpy())
                    gt_traj_list.append(batch['action'].cpu().numpy())
                    gt_img_list.append(batch['obs']['image'].permute(0, 1, 3, 4, 2).cpu().numpy())  # [64, 16, 3, 96, 96]
        if len(val_losses) > 0:
            val_loss = torch.mean(torch.tensor(val_losses)).item()
            # log epoch average validation loss
            runner_log['val_loss'] = val_loss
        vis_data = {'seq_name_list': seq_name_list, 'pred_traj_list': pred_traj_list, 'gt_traj_list': gt_traj_list,
                'gt_img_list': gt_img_list,}

        vis_path = os.path.join(output_dir, '{}_vis.pkl'.format(split))
        with open(vis_path, 'wb') as handle:
            pickle.dump(vis_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Save to {}'.format(vis_path))

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    print('Saved to {}'.format(out_path))



if __name__ == '__main__':
    main()
