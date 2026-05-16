import torch
import cv2
import torch_scatter
import numpy as np


def get_topk_dir(pred, sphere_pts, bmm_size, angle_tol, wt=None, topk=1):
    sph_cp = torch.tensor(sphere_pts.T, dtype=torch.float32).cuda()
    counts = torch.zeros((sphere_pts.shape[0],), dtype=torch.float32, device='cuda')
    if wt is None:
        wt = torch.ones((pred.shape[0], 1)).to(pred)

    for i in range((pred.shape[0] - 1) // bmm_size + 1):
        cos = pred[i * bmm_size:(i + 1) * bmm_size].mm(sph_cp)
        counts += torch.sum((cos > np.cos(2 * angle_tol / 180 * np.pi)).float() / wt[i * bmm_size:(i + 1) * bmm_size], 0)

    # best_dir = np.array(sphere_pts[np.argmax(counts.cpu().numpy())])
    # return best_dir
    topk_idx = torch.topk(counts, topk)[1].cpu().numpy()
    topk_dir = np.array(sphere_pts[topk_idx])
    return topk_dir, counts.cpu().numpy()[topk_idx]


def generate_target_pairs(point_pairs, up, right, front, center=np.zeros((3,))):
    a = point_pairs[:, 0]
    b = point_pairs[:, 1]
    pdist = a - b
    pdist_unit = pdist / (np.linalg.norm(pdist, axis=-1, keepdims=True) + 1e-7)
    proj_len = np.sum((a - center) * pdist_unit, -1)
    oc = (a - center) - proj_len[..., None] * pdist_unit
    dist2o = np.linalg.norm(oc, axis=-1)
    # print(proj_len.shape, dist2o.shape)
    # print(proj_len.min(), proj_len.max())
    target_tr = np.stack([proj_len, dist2o], -1)
    
    up_cos = np.arccos(np.sum(pdist_unit * up, -1))
    right_cos = np.arccos(np.sum(pdist_unit * right, -1))
    front_cos = np.arccos(np.sum(pdist_unit * front, -1))
    target_rot = np.stack([up_cos, right_cos, front_cos], -1)
    
    return target_tr.astype(np.float32).reshape(-1, 2), target_rot.astype(np.float32).reshape(-1, 3)


def vote_center(pc, preds_tr, res, point_idxs, num_rots=36, vis=None):
    corners = torch.stack([pc.min(0)[0], pc.max(0)[0]])
    grid_res = ((corners[1] - corners[0]) / res).long() + 1
    grid_obj = torch.zeros([*grid_res]).to(pc)
    
    proj_len = preds_tr[:, 0]
    odist = preds_tr[:, 1]
    pairs = pc[point_idxs[:, :2]]
    a = pairs[:, 0]
    b = pairs[:, 1]
    ab = a - b
    mask = (torch.norm(ab, dim=-1) > 1e-7) & (odist > res)
    pairs = None
    proj_len, odist, a, b, ab = proj_len[mask], odist[mask], a[mask], b[mask], ab[mask]
    ab = ab / torch.clamp_min(torch.norm(ab, dim=-1, keepdim=True), 1e-7)
    c = a - ab * proj_len[..., None]
    co = torch.stack([torch.zeros((ab.shape[0],)).to(pc), -ab[..., 2], ab[..., 1]], -1)
    invalid = torch.norm(co, dim=-1) < 1e-7
    co[invalid] = torch.stack([-ab[invalid][..., 1], ab[invalid][..., 0], torch.zeros((ab[invalid].shape[0],)).to(pc)], -1)
    
    x = co / torch.norm(co, dim=-1, keepdim=True) * odist[..., None]
    y = torch.cross(x, ab, dim=-1)
    
    # TODO: adaptive num rots
    angles = torch.arange(num_rots).to(pc) / num_rots * 2 * np.pi
    offset = torch.cos(angles[None, :, None]) * x[:, None] + torch.sin(angles[None, :, None]) * y[:, None]  # n x numrot x 3
    center_grid = (c[:, None] + offset - corners[0]) / res
    center_grid = (center_grid + 0.5).long().reshape(-1, 3)
    
    valid = torch.all(center_grid > 0, -1) & torch.all(center_grid < grid_res, -1)
    center_grid = center_grid[valid]
    
    center_grid_1d = center_grid[:, 0] * grid_res[1] * grid_res[2] + center_grid[:, 1] * grid_res[2] + center_grid[:, 2]
    grid_obj = torch_scatter.scatter_add(torch.ones_like(center_grid_1d), center_grid_1d, dim_size=grid_res[0] * grid_res[1] * grid_res[2]).reshape(*grid_res)
    
    grid_obj = grid_obj.cpu().numpy()
    if vis is not None:
        vis.heatmap(cv2.rotate(grid_obj.max(0), cv2.ROTATE_90_COUNTERCLOCKWISE), win='33', opts=dict(title='front'))
        vis.heatmap(cv2.rotate(grid_obj.max(1), cv2.ROTATE_90_COUNTERCLOCKWISE), win='34', opts=dict(title='bird'))
        vis.heatmap(cv2.rotate(grid_obj.max(2), cv2.ROTATE_90_COUNTERCLOCKWISE), win='35', opts=dict(title='side'))

    cand = np.array(np.unravel_index([np.argmax(grid_obj, axis=None)], grid_obj.shape)).T[::-1][0]
    cand_world = corners[0].cpu().numpy() + cand * res
    # print(cand, cand_world)
    return grid_obj, cand_world


def vote_rotation(pc, preds_rot, point_idxs, num_rots=36):
    pairs = pc[point_idxs[:, :2]]
    a = pairs[:, 0]
    b = pairs[:, 1]
    ab = a - b
    mask = torch.norm(ab, dim=-1) > 1e-7
    pairs = None
    a, b, ab, preds_rot = a[mask], b[mask], ab[mask], preds_rot[mask]
    ab = ab / torch.clamp_min(torch.norm(ab, dim=-1, keepdim=True), 1e-7)
    co = torch.stack([torch.zeros((ab.shape[0],)).to(pc), -ab[..., 2], ab[..., 1]], -1)
    invalid = torch.norm(co, dim=-1) < 1e-7
    co[invalid] = torch.stack([-ab[invalid][..., 1], ab[invalid][..., 0], torch.zeros((ab[invalid].shape[0],)).to(pc)], -1)
    
    x = co / torch.clamp_min(torch.norm(co, dim=-1, keepdim=True), 1e-7)
    y = torch.cross(x, ab, dim=-1)
    angles = torch.arange(num_rots).to(pc) / num_rots * 2 * np.pi
    offset = torch.cos(angles[None, :, None]) * x[:, None] + torch.sin(angles[None, :, None]) * y[:, None]  # n x numrot x 3
    tan = torch.tan(preds_rot)
    up = tan[:, None, None] * offset + torch.where(tan > 0, 1., -1.)[:, None, None] * ab[:, None]
    up = up / torch.clamp_min(torch.norm(up, dim=-1, keepdim=True), 1e-7)
    
    return up, mask