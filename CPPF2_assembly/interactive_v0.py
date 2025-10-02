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
from utils.util import downsample, backproject, fibonacci_sphere, calculate_2d_projections, draw, get_3d_bbox, transform_coordinates_3d
from utils.voting import generate_target_pairs, vote_center, vote_rotation, get_topk_dir
import torch.optim as optim
from tqdm.notebook import tqdm
from PIL import Image
import matplotlib
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

torch.set_grad_enabled(False)

grasp_type = 'grasp_horizontal'

angle_tol=1.
imp_wt_margin=0.01
backproj_ratio=.3
num_pairs=100000
num_rots=180
opt=True
num_samples = int(4 * np.pi / (angle_tol / 180 * np.pi))
sphere_pts = np.array(fibonacci_sphere(num_samples), dtype=np.float32)
bmm_size = 100000

cfg = OmegaConf.load('config/custom.yaml')
mesh = trimesh.load('data/Medium_Short_Hexagon_Green.obj')
mesh.apply_scale(0.001)  # convert mm to m
cfg.res = 2e-3
dino_model = BeyondCPPFDINO.load_from_checkpoint('logs/dino_2e-3/version_0/checkpoints/last.ckpt', cfg=cfg).cuda().eval()
shot_model = BeyondCPPFSHOT.load_from_checkpoint('logs/shot_2e-3/version_0/checkpoints/last.ckpt', cfg=cfg).cuda().eval()
desc_model = DINOV2().eval().cuda()

mesh_pts = trimesh.sample.sample_surface(mesh, 2048)[0]
intrinsics = np.array([[621.0726318359375, 0, 253.0572814941406], [0, 620.7985229492188, 258.5174255371094], [0, 0, 1]])

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
image = cv2.imread(f"data/{grasp_type}/0000_rgb_vis.png")[..., ::-1].copy()

# Function to capture the mouse click
def get_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        global input_point
        input_point = np.array([[x, y]])
        print(f"Selected point: {input_point}")
        cv2.destroyAllWindows()  # Close the window after getting the point
        
# Display the image and set the mouse callback
cv2.imshow('Select Input Point', image)
cv2.setMouseCallback('Select Input Point', get_mouse_click)
cv2.waitKey(0)  # Wait for a key press to proceed

# Assuming input_label is fixed
input_label = np.array([1])

# Run the SAM2 predictor with the selected point
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    predictor.set_image(image)
    masks, _, _ = predictor.predict(point_coords=input_point,
                                    point_labels=input_label,
                                    multimask_output=False)
mask = masks[0]

cv2.imwrite('mask.png', mask.astype(np.uint8) * 255)

# Optionally, you can display the mask or save it
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

with torch.no_grad():
    rgb = cv2.imread(f'data/{grasp_type}/0000_rgb_vis.png')[..., ::-1]
    depth = cv2.imread(f'data/{grasp_type}/0000_00.png', cv2.IMREAD_UNCHANGED)[..., -1] / 65535. * 5
    draw_image_bbox = rgb.copy()

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
    pc = pc[indices]
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
        pred_pairs = pred_cls.reshape(pred_cls.shape[0], 2, 3, -1)
        num_bins = pred_cls.shape[-1]
        prob = torch.softmax(pred_cls, -1)

        pred_pairs = torch.multinomial(prob.reshape(np.product(pred_cls.shape[:-1]), -1), 1).float().reshape(-1, 2, 3)
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
        if opt:
            from lietorch import SO3
            with torch.enable_grad():
                opt_trans = torch.nn.Parameter(torch.from_numpy(T_est).cuda().float(), requires_grad=True)
                delta_rot = torch.tensor([0, 0, 0, 1.], requires_grad=True, device='cuda')
                pc_cuda = torch.from_numpy(pc).float().cuda()
                opt = optim.Adam([opt_trans, delta_rot], lr=1e-2)
                # tq = tqdm(range(100))
                for _ in range(100):
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
                tq = tqdm(range(100))
                for _ in tq:
                    opt1.zero_grad()
                    opt2.zero_grad()
                    rot = SO3.InitFromVec(delta_rot).matrix()[:3, :3] @ torch.from_numpy(R_est).float().cuda()
                    # compute chamfer distance between mesh and point cloud
                    mesh_pts_cuda_transformed = (mesh_pts_cuda @ rot.T + opt_trans)
                    cdist = torch.cdist(mesh_pts_cuda_transformed, pc_cuda)
                    loss = torch.mean(torch.min(cdist, dim=1)[0])
                    loss.backward()
                    # opt_trans.grad = opt_trans.grad * 1e-2
                    delta_rot.grad = delta_rot.grad / 180 * np.pi
                    opt1.step()
                    opt2.step()
                    tq.set_description(f'loss: {loss.item():.4f}')
                    
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
            print(pred_scale_norm, best_scale)
    pred_scale_norm = np.cbrt(np.linalg.det(best_RT[:3, :3]))
    mesh_pts_transformed = (mesh_pts @ best_RT[:3, :3].T / pred_scale_norm) + best_RT[:3, -1]



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

rot = best_RT[:3, :3] / pred_scale_norm
trans = best_RT[:3, -1]

# rot = rot @ Rotation.from_euler('xyz', [0, 0, 120], degrees=True).as_matrix()

if grasp_type == 'grasp':
    WORLD_T_CAMERA = ((0.24755612015724182, 0.20443850755691528, 1.052464246749878), (0.0315367691218853, 0.9263785481452942, 0.3737049400806427, -0.03424801304936409))
elif grasp_type == 'grasp_horizontal':
    WORLD_T_CAMERA =  ((0.9520325064659119, 0.986301600933075, -0.07431097328662872), (-0.35780030488967896, -0.0066874707117676735, 0.03702797740697861, 0.9330397248268127))

if grasp_type == 'grasp':
    PEG_T_CAM = rottrans2pybullet(rot, trans)
    WORLD_T_PEG_START = multiply(WORLD_T_CAMERA, invert(PEG_T_CAM))
    if Rotation.from_quat(WORLD_T_PEG_START[1]).as_matrix()[:3, 2][2] < 0:
        rot = rot @ Rotation.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
    best_rot, best_x = None, -1.
    for deg in range(0, 360, 60):
        cand_rot = rot @ Rotation.from_euler('xyz', [0, 0, deg], degrees=True).as_matrix()
        PEG_T_CAM = rottrans2pybullet(cand_rot, trans)
        WORLD_T_PEG_START = multiply(WORLD_T_CAMERA, invert(PEG_T_CAM))
        if Rotation.from_quat(WORLD_T_PEG_START[1]).as_matrix()[:3, 0][0] > best_x:
            best_x = cand_rot[:3, 0][0]
            best_rot = cand_rot
    rot = best_rot
elif grasp_type == 'grasp_horizontal':
    PEG_T_CAM = rottrans2pybullet(rot, trans)
    WORLD_T_PEG_START = multiply(WORLD_T_CAMERA, invert(PEG_T_CAM))
    # print(Rotation.from_quat(WORLD_T_PEG_START[1]).as_matrix()[:3, 2])
    if Rotation.from_quat(WORLD_T_PEG_START[1]).as_matrix()[:3, 2][2] < 0:
        rot = rot @ Rotation.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
    best_rot, best_x = None, -1.
    for deg in range(0, 360, 30):
        cand_rot = rot @ Rotation.from_euler('xyz', [0, 0, deg], degrees=True).as_matrix()
        PEG_T_CAM = rottrans2pybullet(cand_rot, trans)
        WORLD_T_PEG_START = multiply(WORLD_T_CAMERA, invert(PEG_T_CAM))
        if Rotation.from_quat(WORLD_T_PEG_START[1]).as_matrix()[:3, 0][0] > best_x:
            best_x = cand_rot[:3, 0][0]
            best_rot = cand_rot
    rot = best_rot

# visualize pose
best_RT[:3, :3] = rot * pred_scale_norm
xyz_axis = 0.3 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
transformed_axes = transform_coordinates_3d(xyz_axis, best_RT)
projected_axes = calculate_2d_projections(transformed_axes, intrinsics)

bbox_3d = get_3d_bbox(best_scale, 0)
transformed_bbox_3d = transform_coordinates_3d(bbox_3d, best_RT)
projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
draw_image_bbox = draw(draw_image_bbox, projected_bbox, projected_axes, (255, 0, 0))
cv2.imwrite('output.png', draw_image_bbox[..., ::-1])


PEG_T_CAM = rottrans2pybullet(rot, trans)
print('PEG_T_CAM: ', PEG_T_CAM)
print('invert(PEG_T_CAM): ', invert(PEG_T_CAM))
WORLD_T_PEG_START = multiply(WORLD_T_CAMERA, invert(PEG_T_CAM))
print('WORLD_T_PEG_START: ', WORLD_T_PEG_START)
# vertical
if grasp_type == 'grasp':
    PEG_T_TIP = ([0, 0, -0.01], [-1, 0, 0, 0])  # accurate
    TOOL_T_TIP = ([0, 0, 0.28], [0, 0, 0, 1])
# horizontal
elif grasp_type == 'grasp_horizontal':
    PEG_T_TIP = multiply(([0, 0.05, 0.04], [-0.7071068, 0, 0, 0.7071068]),
                                        ([0, 0, 0], [0, 0, 1, 0]))  # accurate
    TOOL_T_TIP = ([0, 0, 0.28], [0, 0, 0, 1])
WORLD_T_TOOL_PICK_PEG = multiply(multiply(WORLD_T_PEG_START, PEG_T_TIP), invert(TOOL_T_TIP))
print('WORLD_T_TOOL_PICK_PEG: ', WORLD_T_TOOL_PICK_PEG)  



