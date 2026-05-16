import numpy as np
from pathlib import Path
import os
import cv2
import torch
import trimesh
from scipy.stats import special_ortho_group
from omegaconf import OmegaConf
import sys
sys.path.append('SAM2')

import pickle
from src_shot.build import shot
from dino import DINOV2, resize_crop
import torch_scatter
from cppf2_utils.util import downsample, backproject, fibonacci_sphere, calculate_2d_projections, draw, get_3d_bbox, transform_coordinates_3d
from cppf2_utils.voting import generate_target_pairs, vote_center, vote_rotation, get_topk_dir
import torch.optim as optim
from tqdm.notebook import tqdm
from PIL import Image
import matplotlib
# matplotlib.use('Agg') # for avoiding memory leak
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pybullet as p
from scipy.spatial.transform import Rotation
from SAM2.sam2.sam2_image_predictor import SAM2ImagePredictor
from train_shot import BeyondCPPF as BeyondCPPFSHOT
from train_dino import BeyondCPPF as BeyondCPPFDINO
from pytorch_lightning.callbacks import ModelCheckpoint
from time import time
from multiprocessing import cpu_count
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch.multiprocessing as mp
import multiprocessing
# Set the start method for multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
            
torch.set_grad_enabled(False)

# INPUT_POINT = np.array([[0, 0]])
def get_mouse_click(event, x, y, flags, param):
    # Function to capture the mouse click
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        # global INPUT_POINT
        param['point'] = np.array([[x, y]])
        print(f"Selected point: {param}")
        # cv2.destroyAllWindows()  # Close the window after getting the point
        # print('Window destroyed.')
    # return INPUT_POINT

def rottrans2pybullet(rot, trans):
    quat = Rotation.from_matrix(rot).as_quat()
    return (list(trans), list(quat))

def multiply(*poses):
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = p.multiplyTransforms(pose[0], pose[1], *next_pose)
    return pose

def invert(pose):
    point, quat = pose
    return p.invertTransform(point, quat)


def run_sam2(rgb, input_point, input_label):
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        predictor.set_image(rgb)
        masks, _, _ = predictor.predict(point_coords=input_point,
                                            point_labels=input_label,
                                            multimask_output=False)
    mask = masks[0]
    cv2.imwrite('mask.png', mask.astype(np.uint8) * 255)
    print('Mask detected! Save to {}'.format('mask.png'))
    # Optionally, you can display the mask or save it
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return mask


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_points_cv2(image, coords, labels, marker_size=15):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    for point in pos_points:
        cv2.drawMarker(image, (int(point[0]), int(point[1])), color=(0, 255, 0), markerType=cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=2, line_type=cv2.LINE_AA)
    for point in neg_points:
        cv2.drawMarker(image, (int(point[0]), int(point[1])), color=(0, 0, 255), markerType=cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=2, line_type=cv2.LINE_AA)

def on_key(event, image):
    if event == 32:  # 32 is the ASCII code for the SPACE key
        cv2.destroyAllWindows()

def on_key(event):
    if event.key == ' ':
        plt.close(event.canvas.figure)
        
def draw_pose_zoom_in(img, intrinsics, rot, center, scale, color=(255, 0, 0), out_size=256, bbox=None):
    mat = np.eye(4)
    scale_norm = np.linalg.norm(scale)
    mat[:3, :3] = rot * scale_norm
    mat[:3, -1] = center
    
    xyz_axis = 0.3 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
    transformed_axes = transform_coordinates_3d(xyz_axis, mat)
    projected_axes = calculate_2d_projections(transformed_axes, intrinsics)

    bbox_3d = get_3d_bbox(scale / scale_norm, 0)
    transformed_bbox_3d = transform_coordinates_3d(bbox_3d, mat)
    projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
    if bbox is None:
        tl_corner = projected_bbox.min(0)
        br_corner = projected_bbox.max(0)
    else:
        tl_corner = np.array([bbox[0], bbox[1]])
        br_corner = np.array([bbox[2], bbox[3]])
    
    draw_image_bbox = draw(img, projected_bbox, projected_axes, color, size=(br_corner - tl_corner).max() // 30)
    if bbox is None:
        bbox = (tl_corner[0], tl_corner[1], br_corner[0], br_corner[1])
    draw_image_bbox = resize_crop(draw_image_bbox, padding=1.0, out_size=out_size, bbox=bbox)[0]
    return draw_image_bbox, bbox

def draw_pose(img, intrinsics, rot, center, scale, color):
    mat = np.eye(4)
    scale_norm = np.linalg.norm(scale)
    mat[:3, :3] = rot * scale_norm
    mat[:3, -1] = center
    
    xyz_axis = 0.3 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
    transformed_axes = transform_coordinates_3d(xyz_axis, mat)
    projected_axes = calculate_2d_projections(transformed_axes, intrinsics)

    bbox_3d = get_3d_bbox(scale / scale_norm, 0)
    transformed_bbox_3d = transform_coordinates_3d(bbox_3d, mat)
    projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
    draw_image_bbox = draw(img, projected_bbox, projected_axes, color)
    return draw_image_bbox

def get_obj_pose(rgb, depth, mask, intrinsics, cfg, desc_model, num_pairs, dino_model, shot_model, 
                 num_rots, backproj_ratio, imp_wt_margin, sphere_pts, bmm_size, angle_tol, mesh_pts,
                 WORLD_T_CAM,
                 use_opt, num_iter):
    with torch.no_grad():
        mask = mask.astype(bool)
        rgb_masked = np.ones_like(rgb)
        rgb_masked[mask] = rgb[mask]
        rgb_local, transform = resize_crop(rgb_masked, bbox=Image.fromarray(rgb_masked).getbbox(), padding=0, out_size=256)

        pc, idxs = backproject(depth, intrinsics, mask)
        idxs = np.stack(idxs, -1)  # K x 2
        pc[:, 0] = -pc[:, 0]
        pc[:, 1] = -pc[:, 1]
        pc = pc.astype(np.float32)
        indices = downsample(pc, cfg.res)
        try:
            pc = pc[indices]
        except Exception as e:
            print(e)
            print(indices)  # []
        idxs = idxs[indices]
        if pc.shape[0] > 50000:
            sub_idx = np.random.randint(pc.shape[0], size=(50000,))
            pc = pc[sub_idx]
            idxs = idxs[sub_idx]
        # import pdb; pdb.set_trace()
        kp = np.flip(idxs, -1)
        kp_local = (np.linalg.inv(transform) @ np.concatenate([kp, np.ones((kp.shape[0], 1))], -1).T).T[:, :2]

        desc = desc_model(torch.from_numpy(rgb_local).cuda().float().permute(2, 0, 1) / 255., torch.from_numpy(kp_local).float().cuda()).cpu().numpy()

        point_idxs_all = np.random.randint(0, pc.shape[0], (num_pairs, 2 + cfg.num_more))
        input_pairs = pc[point_idxs_all[:, :2]]

        shot_feat, normal = shot.compute(pc, cfg.res * 10, cfg.res * 10)
        shot_feat = shot_feat.reshape(-1, 352).astype(np.float32)
        # shot_feat = np.zeros((pc.shape[0], 352), dtype=np.float32)

        normal = normal.reshape(-1, 3).astype(np.float32)
        shot_feat[np.isnan(shot_feat)] = 0
        normal[np.isnan(normal)] = 0

        best_loss, best_idx = np.inf, 0
        best_RT = np.eye(4)
        best_scale = np.zeros((3,))
        for model_idx, model in enumerate([dino_model, shot_model]):
            # if model_idx == 1:
            #     continue
            if model_idx == 0:
                pred_cls, pred_scales = model(torch.from_numpy(pc).float().cuda(), torch.from_numpy(desc).float().cuda(), torch.from_numpy(point_idxs_all).long().cuda())
            else:
                pred_cls, pred_scales = model(torch.from_numpy(pc).float().cuda(), torch.from_numpy(point_idxs_all).long().cuda(),
                                            torch.from_numpy(shot_feat).float().cuda(), torch.from_numpy(normal).float().cuda())
            # pred_pairs = pred_cls.reshape(pred_cls.shape[0], 2, 3, -1)
            num_bins = pred_cls.shape[-1]
            prob = torch.softmax(pred_cls, -1)

            pred_pairs = torch.multinomial(prob.reshape(np.prod(pred_cls.shape[:-1]), -1), 1).float().reshape(-1, 2, 3)
            pred_pairs = (pred_pairs / (num_bins - 1) - 0.5)
            # pred_pairs = (prob2real(prob, 1., prob.shape[-1]) - 0.5).reshape(-1, 2, 3)  # this line gives worse results, just do multinomial by treating it as a sampling rather than expectation

            scale = torch.from_numpy(np.linalg.norm(input_pairs[:, 1] - input_pairs[:, 0], axis=-1)).float().cuda() \
                / torch.clamp_min(torch.norm(pred_pairs[:, 1] - pred_pairs[:, 0], dim=-1), 1e-7)
            pred_pairs_scaled = pred_pairs * scale[:, None, None]

            targets_tr, targets_rot = generate_target_pairs(pred_pairs_scaled.cpu().numpy(),
                                                            np.array(cfg.up),
                                                            np.array(cfg.front),
                                                            np.array(cfg.right))
            grid_obj, pred_trans = vote_center(torch.from_numpy(pc).float().cuda(),
                torch.from_numpy(targets_tr).float().cuda(),
                cfg.res,
                torch.from_numpy(point_idxs_all[:, :2]).long().cuda(),
                num_rots=num_rots,
                vis=None)

            # if model_idx == 0:
            T_est = pred_trans

            # backvoting
            targets_tr_back, _ = generate_target_pairs(input_pairs,
                                                            np.array(cfg.up),
                                                            np.array(cfg.front),
                                                            np.array(cfg.right),
                                                            T_est)
            back_errs = np.linalg.norm(targets_tr - targets_tr_back, axis=-1)
            pairs_mask = back_errs < np.percentile(back_errs, backproj_ratio * 100)
            # pairs_mask = back_errs < backproj_thres
            # print(pairs_mask.sum() / point_idxs_all.shape[0])
            point_idxs_pair_flattened = point_idxs_all[pairs_mask, :2].reshape(-1)
            unique_idx = np.unique(point_idxs_pair_flattened)
            pc_masked = pc[unique_idx]
            point_idxs_pair_flattened = torch.from_numpy(point_idxs_pair_flattened).cuda().long()
            imp_wt = torch_scatter.scatter_add(torch.ones_like(point_idxs_pair_flattened),
                                            point_idxs_pair_flattened, dim=-1, dim_size=pc.shape[0]).cpu().numpy()


            point_idxs_all_filtered = point_idxs_all[pairs_mask]
            targets_rot = targets_rot[pairs_mask]
            scale = scale[pairs_mask]
            pred_scales = pred_scales[pairs_mask]

            imp_wt = imp_wt / imp_wt.max()
            imp_pair_wt = torch.from_numpy(imp_wt[point_idxs_all_filtered[:, :2]]).cuda().sum(-1) + imp_wt_margin  # N
            # imp_pair_wt.fill_(1.)
            preds_up, valid_mask = vote_rotation(torch.from_numpy(pc).float().cuda(),
                torch.from_numpy(targets_rot[..., 0]).float().cuda(),
                torch.from_numpy(point_idxs_all_filtered[:, :2]).long().cuda(),
                num_rots)
            preds_up = preds_up.reshape(-1, 3)
            preds_ups, cnts = get_topk_dir(preds_up, sphere_pts, bmm_size, angle_tol,
                                    imp_pair_wt[valid_mask, None].expand(-1, num_rots).reshape(-1, 1), topk=1)
            preds_up = preds_ups[0]

            preds_right, valid_mask = vote_rotation(torch.from_numpy(pc).float().cuda(),
                torch.from_numpy(targets_rot[..., 2]).float().cuda(),
                torch.from_numpy(point_idxs_all_filtered[:, :2]).long().cuda(),
                num_rots)
            preds_right = preds_right.reshape(-1, 3)
            preds_rights, cnts = get_topk_dir(preds_right, sphere_pts, bmm_size, angle_tol,
                                    imp_pair_wt[valid_mask, None].expand(-1, num_rots).reshape(-1, 1), topk=1)
            preds_right = preds_rights[0]

            preds_right -= np.dot(preds_up, preds_right) * preds_up
            preds_right /= (np.linalg.norm(preds_right) + 1e-9)

            up_loc = np.where(cfg.up)[0][0]
            right_loc = np.where(cfg.right)[0][0]
            R_est = np.eye(3)
            R_est[:3, up_loc] = preds_up
            R_est[:3, right_loc] = preds_right

            if model_idx == 0:
                pred_scale = torch.median(pred_scales, 0)[0].cpu().numpy()
                pred_scale_norm = np.linalg.norm(pred_scale)

            other_loc = list(set([0, 1, 2]) - set([up_loc, right_loc]))[0]
            R_est[:3, other_loc] = np.cross(R_est[:3, (other_loc + 1) % 3], R_est[:3, (other_loc + 2) % 3])
            if use_opt:
                from lietorch import SO3
                with torch.enable_grad():
                    opt_trans = torch.nn.Parameter(torch.from_numpy(T_est).cuda().float(), requires_grad=True)
                    delta_rot = torch.tensor([0, 0, 0, 1.], requires_grad=True, device='cuda')
                    pc_cuda = torch.from_numpy(pc).float().cuda()
                    opt = optim.Adam([opt_trans, delta_rot], lr=1e-2)
                    # tq = tqdm(range(100))
                    for _ in range(num_iter):
                        opt.zero_grad()
                        rot = SO3.InitFromVec(delta_rot).matrix()[:3, :3] @ torch.from_numpy(R_est).float().cuda()
                        pc_canon = (pc_cuda - opt_trans) @ rot
                        loss = torch.abs(pc_canon[point_idxs_all_filtered[:, :2]] - pred_pairs_scaled[pairs_mask])
                        loss = loss.mean()
                        loss.backward()
                        # opt_trans.grad = opt_trans.grad * 1e-2
                        delta_rot.grad = delta_rot.grad / 180 * np.pi
                        opt.step()
                        # tq.set_description(f'loss: {loss.item():.4f}')

                T_est = opt_trans.detach().cpu().numpy()
                R_est = (SO3.InitFromVec(delta_rot).matrix()[:3, :3] @ torch.from_numpy(R_est).float().cuda()).detach().cpu().numpy()

                with torch.enable_grad():
                    opt_trans = torch.nn.Parameter(torch.from_numpy(T_est).cuda().float(), requires_grad=True)
                    delta_rot = torch.tensor([0, 0, 0, 1.], requires_grad=True, device='cuda')
                    opt1 = optim.Adam([opt_trans], lr=3e-4)
                    opt2 = optim.Adam([delta_rot], lr=1e-2)
                    mesh_pts_cuda = torch.from_numpy(mesh_pts).float().cuda()
                    # tq = tqdm(range(100))
                    tq = range(num_iter)
                    for _ in tq:
                        opt1.zero_grad()
                        opt2.zero_grad()
                        rot = SO3.InitFromVec(delta_rot).matrix()[:3, :3] @ torch.from_numpy(R_est).float().cuda()
                        # compute chamfer distance between mesh and point cloud
                        mesh_pts_cuda_transformed = (mesh_pts_cuda @ rot.T + opt_trans)
                        cdist = torch.cdist(mesh_pts_cuda_transformed, pc_cuda)
                        loss = torch.mean(torch.clamp_max(torch.min(cdist, dim=1)[0], 0.025))
                        loss.backward()
                        delta_rot.grad = delta_rot.grad / 180 * np.pi
                        opt1.step()
                        opt2.step()

                T_est = opt_trans.detach().cpu().numpy()
                R_est = (SO3.InitFromVec(delta_rot).matrix()[:3, :3] @ torch.from_numpy(R_est).float().cuda()).detach().cpu().numpy()
                
                with torch.enable_grad():
                    opt_trans = torch.nn.Parameter(torch.from_numpy(T_est).cuda().float(), requires_grad=True)
                    delta_rot = torch.tensor([0, 0, 0, 1.], requires_grad=True, device='cuda')
                    opt1 = optim.Adam([opt_trans], lr=3e-4)
                    opt2 = optim.Adam([delta_rot], lr=1e-2)
                    mesh_pts_cuda = torch.from_numpy(mesh_pts).float().cuda()
                    tq = range(num_iter)
                    for _ in tq:
                        opt1.zero_grad()
                        opt2.zero_grad()
                        rot = SO3.InitFromVec(delta_rot).matrix()[:3, :3] @ torch.from_numpy(R_est).float().cuda()
                        # compute chamfer distance between mesh and point cloud
                        mesh_pts_cuda_transformed = (mesh_pts_cuda @ rot.T + opt_trans)
                        cdist = torch.cdist(mesh_pts_cuda_transformed, pc_cuda)
                        loss = torch.mean(torch.clamp_max(torch.min(cdist, dim=0)[0], 0.025))
                        loss.backward()
                        delta_rot.grad = delta_rot.grad / 180 * np.pi
                        opt1.step()
                        opt2.step()
                        
                T_est = opt_trans.detach().cpu().numpy()
                R_est = (SO3.InitFromVec(delta_rot).matrix()[:3, :3] @ torch.from_numpy(R_est).float().cuda()).detach().cpu().numpy()


            pc_canon = (pc - T_est) @ R_est / pred_scale_norm
            loss = np.abs(pc_canon[point_idxs_all_filtered[:, :2]] - pred_pairs[pairs_mask].cpu().numpy())
            loss = np.clip(loss, 0, 0.1)
            loss = loss.mean()

            if loss < best_loss:
                best_loss = loss
                best_idx = model_idx
                best_RT[:3, :3] = R_est * pred_scale_norm
                best_RT[:3, -1] = T_est
                best_scale = pred_scale / pred_scale_norm
                # print('pred_scale_norm, best_scale: ', pred_scale_norm, best_scale)
        pred_scale_norm = np.cbrt(np.linalg.det(best_RT[:3, :3]))
        # mesh_pts_transformed = (self.mesh_pts @ best_RT[:3, :3].T / pred_scale_norm) + best_RT[:3, -1]

    rot = best_RT[:3, :3] / pred_scale_norm  # T_CAM
    trans = best_RT[:3, -1]

    PEG_T_CAM = (trans, Rotation.from_matrix(rot).as_quat())
    PEG_T_WORLD = multiply(invert(WORLD_T_CAM), PEG_T_CAM)
    if np.isnan(PEG_T_WORLD[1]).any() | np.isinf(PEG_T_WORLD[1]).any() or np.linalg.norm(PEG_T_WORLD[1]) == 0:
        print("Found zero norm in {}. Retry".format(PEG_T_WORLD))
        find_valid_pose = False
    else:
        find_valid_pose = True
    return PEG_T_WORLD, find_valid_pose, trans, best_RT, pred_scale_norm, best_scale, rot

class CPPF2_wrapper():
    def __init__(self, cppf2_relative_path=''):
        self.grasp_type = 'grasp'  # 'grasp_horizontal'

        self.save_dir = 'cppf2_output'
        os.makedirs(self.save_dir, exist_ok=True)
        self.angle_tol=1.
        self.imp_wt_margin=0.01
        self.backproj_ratio=.3
        self.num_pairs=100000
        self.num_rots=180
        self.use_opt=True
        self.num_samples = int(4 * np.pi / (self.angle_tol / 180 * np.pi))
        self.sphere_pts = np.array(fibonacci_sphere(self.num_samples), dtype=np.float32)
        self.default_bmm_size = 100000
        self.bmm_size = 100000
        self.default_num_iter = 100
        self.num_iter = 100

        self.cppf2_relative_path = cppf2_relative_path
        self.cfg = OmegaConf.load(os.path.join(cppf2_relative_path, 'config/custom.yaml'))
        
        self.cfg.res = 2e-3
        self.desc_model = DINOV2().eval().cuda()
        
        self.intrinsics = np.array([[621.0726318359375, 0, 253.0572814941406], [0, 620.7985229492188, 258.5174255371094], [0, 0, 1]])

        self.WORLD_T_CAM = ((-0.24853374063968658, 0.1551879644393921, 0.881916880607605), (0.9233834147453308, 0.0076868268661201, 0.01738572120666504, 0.3834081292152405))
        self.CAM_T_WORLD = invert(self.WORLD_T_CAM)
        print('self.CAM_T_WORLD: ', self.CAM_T_WORLD)
        self.sam_version = 'sam_2'
        if self.sam_version == 'sam_1':
            # sam ckpt is place under packages/CPPF2_assembly/sam_vit_h_4b8939.pth
            sam = sam_model_registry["vit_h"](checkpoint=os.path.join(cppf2_relative_path, "sam_vit_h_4b8939.pth"))
            self.predictor = SamPredictor(sam)
        else:
            # Run the SAM2 predictor with the selected point
            self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

        self.zoom_in_sam = True
        self.crop_size = 35
        self.plot_lib = 'cv2'
        self.current_obj_name = 'Medium_Short_Hexagon_Green'
        self.obj_name2vert_candidates = {
            'Medium_Short_Hexagon_Green': range(0, 360, 60),
            'Medium_Short_Star_DarkBlue': range(0, 360, 72),
            'Medium_Short_Oval_JeansBlue': range(0, 360, 180),
            
            # other objects
            'Medium_Short_3Prong_JeansRed': range(0, 360, 180),
            'Medium_Short_Arch_Yellow': range(0, 360, 180),
            'Medium_Short_Circle_Brown': range(0, 360, 2),
            'Medium_Short_DoubleSquare_Purple': range(0, 360, 180),
            'Medium_Short_Rectangle_JeansBlue': range(0, 360, 180),
            'Medium_Short_SquareCircle_Red': range(0, 360, 360),
        }
        self.obj_name2horiz_candidates = {
            'Medium_Short_Hexagon_Green': range(0, 360, 30),
            'Medium_Short_Star_DarkBlue': range(0, 360, 36),
            'Medium_Short_Oval_JeansBlue': range(0, 360, 30),
            
            # other objects
            'Medium_Short_3Prong_JeansRed': range(0, 360, 180),
            'Medium_Short_Arch_Yellow': range(0, 360, 180),
            'Medium_Short_Circle_Brown': range(0, 360, 2),
            'Medium_Short_DoubleSquare_Purple': range(0, 360, 180),
            'Medium_Short_Rectangle_JeansBlue': range(0, 360, 180),
            'Medium_Short_SquareCircle_Red': range(0, 360, 360),
        }
        self.load_model(selected_obj_name=self.current_obj_name)

    def load_model(self, selected_obj_name='Medium_Short_Hexagon_Green'):
        self.dino_model = BeyondCPPFDINO.load_from_checkpoint(
            os.path.join(self.cppf2_relative_path, 'logs_all_models/dino-{}/version_0/checkpoints/last.ckpt'.format(selected_obj_name)),
            cfg=self.cfg).cuda().eval()
        self.shot_model = BeyondCPPFSHOT.load_from_checkpoint(
            os.path.join(self.cppf2_relative_path, 'logs_all_models/shot-{}/version_0/checkpoints/last.ckpt'.format(selected_obj_name)),
            cfg=self.cfg).cuda().eval()
        
        self.mesh = trimesh.load(os.path.join(self.cppf2_relative_path, f'data/{selected_obj_name}.obj'))
        self.mesh.apply_scale(0.001)  # convert mm to m
        self.mesh_pts = trimesh.sample.sample_surface(self.mesh, 2048)[0]
        print('CPPF2 model loaded for object {}'.format(selected_obj_name))


    def inference(self, image_rgba=None, selected_obj_name='Medium_Short_Hexagon_Green'):
        if selected_obj_name != self.current_obj_name:
            self.load_model(selected_obj_name=selected_obj_name)
            self.current_obj_name = selected_obj_name
            
        if self.current_obj_name == 'Medium_Short_Star_DarkBlue':
            self.backproj_ratio = 0.1
        else:
            self.backproj_ratio = 0.3

        INPUT_POINT = {'point': np.array([[0, 0]])}
        if image_rgba is None:
            # image = cv2.imread(f"data/{self.grasp_type}/0000_rgb_vis.png")[..., ::-1].copy()
            rgb = cv2.imread(f'data/{self.grasp_type}/0000_rgb_vis.png')[..., ::-1].copy()
            depth = cv2.imread(f'data/{self.grasp_type}/0000_00.png', cv2.IMREAD_UNCHANGED)[..., -1] / 65535. * 5
        else:
            # image = image_rgba[..., :3].copy()
            rgb = image_rgba[..., :3][..., ::-1].copy()
            rgb = (rgb / 255).astype(np.uint8)
            # print(rgb.max(), rgb.min())  # (255, 0)
            depth = image_rgba[..., -1] / 65535. * 5  # in meters
        img_w = rgb.shape[1]
        img_h = rgb.shape[0]
        # Assuming input_label is fixed
        input_label = np.array([1])

        # Display the image and set the mouse callback
        cv2.imshow('Select Input Point', rgb[..., ::-1])
        cv2.setMouseCallback('Select Input Point', get_mouse_click, param=INPUT_POINT)
        cv2.waitKey(0)  # Wait for a key press to proceed # press SPACE key
        cv2.destroyAllWindows()

        INPUT_POINT_value = INPUT_POINT['point']
        # if self.plot_lib == 'matplotlib':
        #     # Display the RGB image using matplotlib and set up the key event handler
        #     fig, ax = plt.subplots()
        #     plt.imshow(rgb.copy())
        #     show_points(np.array(INPUT_POINT_value), input_label, plt.gca(), marker_size=375)
        #     plt.axis('on')
        #     # Connect the key event handler to the matplotlib figure
        #     fig.canvas.mpl_connect('key_press_event', on_key)  # press SPACE key
        #     plt.show()
        # else:
        #     display_image = rgb.copy()[:, :, ::-1].astype(np.uint8)
        #     show_points_cv2(display_image, np.array(INPUT_POINT_value), input_label)
        #     # Display the image using OpenCV
        #     cv2.imshow('Image with Points', display_image)
        #     cv2.waitKey(0)  # Wait for a key press to proceed # press SPACE key
        #     cv2.destroyAllWindows()


        if self.zoom_in_sam:
            offset_x, offset_y = INPUT_POINT_value[0][0], INPUT_POINT_value[0][1]
            rgb_sam = rgb[max(offset_y - self.crop_size, 0):offset_y + self.crop_size,
                  max(offset_x - self.crop_size, 0):offset_x + self.crop_size].copy()
            print('rgb_sam.shape: ', rgb_sam.shape, rgb.shape, max(offset_y - self.crop_size, 0), offset_y + self.crop_size,
                  max(offset_x - self.crop_size, 0), offset_x + self.crop_size)
            INPUT_POINT_value = np.array([[offset_x - max(offset_x - self.crop_size, 0), offset_y - max(offset_y - self.crop_size, 0)]])
            if self.plot_lib == 'matplotlib':
                fig, ax = plt.subplots()
                plt.imshow(rgb_sam)
                plt.axis('off')
                fig.canvas.mpl_connect('key_press_event', on_key)  # press SPACE key
                plt.show()
            else:
                display_image = rgb_sam.copy()[:, :, ::-1]
                # show_points_cv2(display_image, np.array(INPUT_POINT_value), input_label)
                # Display the image using OpenCV
                cv2.imshow('Cropped Image', display_image)
                cv2.waitKey(0)  # Wait for a key press to proceed # press SPACE key
                cv2.destroyAllWindows()
            rgb_sam = rgb_sam.astype(np.uint8)
        else:
            rgb_sam = rgb.astype(np.uint8)

        if self.sam_version == 'sam_2':
            # SAM2_success = False
            # while not SAM2_success:
            #     output_queue = mp.Queue()
            #     process = mp.Process(target=run_sam2, args=(rgb, INPUT_POINT, input_label, output_queue))
            #     process.start()
            #     process.join()

            #     # Check if the process has exited normally
            #     if process.exitcode == 0:
            #         # Get the result from the queue
            #         result = output_queue.get()
            #         if isinstance(result, Exception):
            #             # If the result is an exception, handle it
            #             print(f"Attempt failed with exception: {result}")
            #         else:
            #             # If result is valid, return it
            #             print("Process completed successfully.")
            #             mask = result
            #             SAM2_success = True

            #     else:
            #         print(f"Attempt failed. Exit code: {process.exitcode}. Retrying SAM2")

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                self.predictor.set_image(rgb_sam)  # .astype(np.float32)
                masks, _, _ = self.predictor.predict(point_coords=INPUT_POINT_value,
                                                    point_labels=input_label,
                                                    multimask_output=False)
        else:
            self.predictor.set_image(rgb_sam)
            masks, scores, logits = self.predictor.predict(
                point_coords=INPUT_POINT_value,
                point_labels=input_label,
                multimask_output=True,
            )

        mask = masks[0]
        if self.zoom_in_sam:
            binary_mask = mask  # Convert to binary image
            mask = np.zeros((img_w, img_h), dtype=np.uint8)
            mask[max(offset_y - self.crop_size, 0):offset_y + self.crop_size, max(offset_x - self.crop_size, 0):offset_x + self.crop_size] = binary_mask

        mask = mask.astype(np.uint8) * 255  # should be in [0, 255]
        # cv2.imwrite(os.path.join(self.save_dir, 'mask.png'), mask.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(self.save_dir, 'mask_{}.png'.format(self.current_obj_name)), mask)  # imwrite(): should be in [0, 255]
        print('Mask detected! mask.max(): {}, mask.min(): {}, Save to {}'.format(mask.max(), mask.min(), 'mask.png'))
        # Optionally, you can display the mask or save it
        cv2.imshow('Mask', mask)  # imshow(): should be in [0, 255]
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        draw_image_bbox = rgb.copy()
        find_valid_pose_flag = False
        while not find_valid_pose_flag:
            PEG_T_WORLD, find_valid_pose_flag, trans, best_RT, pred_scale_norm, best_scale, rot = \
            get_obj_pose(rgb=rgb, depth=depth, mask=mask, intrinsics=self.intrinsics, 
                        cfg=self.cfg, desc_model=self.desc_model, num_pairs=self.num_pairs, 
                        dino_model=self.dino_model, shot_model=self.shot_model, 
                        num_rots=self.num_rots, backproj_ratio=self.backproj_ratio,
                        imp_wt_margin=self.imp_wt_margin, sphere_pts=self.sphere_pts, 
                        bmm_size=self.bmm_size, angle_tol=self.angle_tol,
                        mesh_pts=self.mesh_pts, WORLD_T_CAM=self.WORLD_T_CAM, use_opt=self.use_opt,
                         num_iter=self.num_iter)

        rot_world = Rotation.from_quat(PEG_T_WORLD[1]).as_matrix()
        print('rot_world: ', rot_world)
        # print(np.abs(np.arccos(rot_world[:3, 2][2]) * 180 / np.pi - 90))
        z_angle = np.abs(np.arccos(rot_world[:3, 2][2]) * 180 / np.pi - 90)
        # 0: horizontal, 90: vertical
        grasp_type = 'grasp_horizontal' if z_angle < 45 else 'grasp'
        print('grasp_type: ', grasp_type)

        if grasp_type == 'grasp':
            PEG_T_CAM = rottrans2pybullet(rot, trans)
            PEG_T_WORLD = multiply(self.CAM_T_WORLD, PEG_T_CAM)
            if Rotation.from_quat(PEG_T_WORLD[1]).as_matrix()[:3, 2][2] < 0:
                rot = rot @ Rotation.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
            else:
                if self.current_obj_name == 'Medium_Short_Star_DarkBlue':
                    rot = rot @ Rotation.from_euler('xyz', [0, 0, 36], degrees=True).as_matrix()
            best_rot, best_x = None, -1.
            for deg in self.obj_name2vert_candidates[self.current_obj_name]:
                cand_rot = rot @ Rotation.from_euler('xyz', [0, 0, deg], degrees=True).as_matrix()
                PEG_T_CAM = rottrans2pybullet(cand_rot, trans)
                PEG_T_WORLD = multiply(self.CAM_T_WORLD, PEG_T_CAM)
                if Rotation.from_quat(PEG_T_WORLD[1]).as_matrix()[:3, 0][0] > best_x:
                    best_x = cand_rot[:3, 0][0]
                    best_rot = cand_rot
            rot = best_rot
            rot = rot @ Rotation.from_euler('xyz', [0, 0, 180], degrees=True).as_matrix()
        elif grasp_type == 'grasp_horizontal':
            PEG_T_CAM = rottrans2pybullet(rot, trans)
            PEG_T_WORLD = multiply(self.CAM_T_WORLD, PEG_T_CAM)
            if Rotation.from_quat(PEG_T_WORLD[1]).as_matrix()[:3, 2][1] < 0:
                rot = rot @ Rotation.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
            best_rot, best_x = None, -1.
            for deg in self.obj_name2horiz_candidates[self.current_obj_name]:
                cand_rot = rot @ Rotation.from_euler('xyz', [0, 0, deg], degrees=True).as_matrix()
                PEG_T_CAM = rottrans2pybullet(cand_rot, trans)
                PEG_T_WORLD = multiply(self.CAM_T_WORLD, PEG_T_CAM)
                if Rotation.from_quat(PEG_T_WORLD[1]).as_matrix()[:3, 0][0] > best_x:
                    best_x = cand_rot[:3, 0][0]
                    best_rot = cand_rot
            rot = best_rot
            
        if self.current_obj_name == 'Medium_Short_SquareCircle_Red':
            rot = rot @ Rotation.from_euler('xyz', [0, 0, 180], degrees=True).as_matrix()

        PEG_T_CAM = rottrans2pybullet(rot, trans)
        print('PEG_T_CAM: ', PEG_T_CAM)

        PEG_T_WORLD = multiply(self.CAM_T_WORLD, PEG_T_CAM)
        print('PEG_T_WORLD: ', PEG_T_WORLD)
        
        # vertical
        if grasp_type == 'grasp':
            if self.current_obj_name in ['Medium_Short_Star_DarkBlue']:
                # PEG_T_TIP = invert(([0, 0.007, -0.01], [-1, 0, 0, 0]))  # accurate
                PEG_T_TIP = invert(([0, 0.003, -0.01], [-1, 0, 0, 0]))  # accurate
            # elif self.current_obj_name in ['Medium_Short_Oval_JeansBlue']:
            #     PEG_T_TIP = invert(([0, 0.017, -0.0], [-1, 0, 0, 0]))  # accurate
            elif self.current_obj_name in ['Medium_Short_SquareCircle_Red', 'Medium_Short_DoubleSquare_Purple']:
                PEG_T_TIP = invert(([0, 0, 0.015], [-1, 0, 0, 0]))
            else:
                PEG_T_TIP = invert(([0, 0, -0.01], [-1, 0, 0, 0]))  # accurate
        # horizontal
        elif grasp_type == 'grasp_horizontal':
            if self.current_obj_name in ['Medium_Short_Star_DarkBlue']:
                # PEG_T_TIP = invert(multiply(([0, 0.03, 0.01], [-0.7071068, 0, 0, 0.7071068]), ([0, 0, 0], [0, 0, 1, 0])))  # accurate
                PEG_T_TIP = invert(multiply(([0, 0.03, 0.01], [-0.7071068, 0, 0, 0.7071068]), ([0, 0, 0], [0, 0, 1, 0])))  # accurate
            else:
                # PEG_T_TIP = invert(multiply(([0, 0.04, 0.03], [-0.7071068, 0, 0, 0.7071068]), ([0, 0, 0], [0, 0, 1, 0])))  # accurate
                PEG_T_TIP = invert(multiply(([0, 0.04, 0.01], [-0.7071068, 0, 0, 0.7071068]), ([0, 0, 0], [0, 0, 1, 0])))  # accurate

        TOOL_T_TIP = invert(([0, 0, 0.28], [0, 0, 0, 1]))
        TOOL_T_PEG = multiply(invert(PEG_T_TIP), TOOL_T_TIP)
        print('TOOL_T_PEG: ', TOOL_T_PEG)

        TOOL_T_WORLD = multiply(PEG_T_WORLD, TOOL_T_PEG)
        print('TOOL_T_WORLD: ', TOOL_T_WORLD)

        # visualize pose
        draw_image_bbox = draw_pose(rgb.copy(), self.intrinsics, rot, trans, best_scale * pred_scale_norm, color=(255, 255, 0))
        cv2.imwrite(os.path.join(self.save_dir, 'output_bbox_{}.png'.format(self.current_obj_name)), draw_image_bbox[..., ::-1])

        draw_image_bbox_zoom_in = draw_pose_zoom_in(rgb.copy(), self.intrinsics, rot, trans, best_scale * pred_scale_norm, color=(255, 255, 0), out_size=512)[0]
        cv2.imwrite(os.path.join(self.save_dir, 'output_bbox_zoom_in_{}.png'.format(self.current_obj_name)), draw_image_bbox_zoom_in[..., ::-1])
        
        return TOOL_T_WORLD, grasp_type

        


if __name__ == "__main__":
    CPPF2 = CPPF2_wrapper()
    CPPF2.inference()


