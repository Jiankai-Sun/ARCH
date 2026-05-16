import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms import functional
import numpy as np
from PIL import Image
import torchvision.transforms as T

def resize_crop(img, padding=0.2, out_size=224, bbox=None):
    # return np.array(img), np.eye(3)
    img = Image.fromarray(img)
    if bbox is None:
        bbox = img.getbbox()
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    size = max(height, width) * (1 + padding)
    center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
    bbox_enlarged = center[0] - size / 2, center[1] - size / 2, \
        center[0] + size / 2, center[1] + size / 2
    img = functional.resize(functional.crop(img, bbox_enlarged[1], bbox_enlarged[0], size, size), (out_size, out_size))
    transform = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1.]])  \
        @ np.array([[size / out_size, 0, 0], [0, size / out_size, 0], [0, 0, 1]]) \
        @ np.array([[1, 0, -out_size / 2], [0, 1, -out_size / 2], [0, 0, 1.]])
    return np.array(img), transform

def interpolate_features(descriptors, pts, strides=8, normalize=True):
    # Normalize keypoints to [-1, 1]
    h, w = descriptors.shape[-2], descriptors.shape[-1]
    
    keypoints = pts.clone()
    # convert keypoint location to pixel center
    keypoints[..., 0] = ((keypoints[..., 0] + 0.5) / w / strides) * 2 - 1  # x coordinates
    keypoints[..., 1] = ((keypoints[..., 1] + 0.5) / h / strides) * 2 - 1  # y coordinates
    
    # Expand dimensions for grid sampling
    keypoints = keypoints.unsqueeze(-3)  # Shape becomes [batch_size, 1, num_keypoints, 2]
    
    # Interpolate using bilinear sampling
    interpolated_features = F.grid_sample(descriptors, keypoints, align_corners=False)
    
    # interpolated_features will have shape [batch_size, channels, 1, num_keypoints]
    # You might want to squeeze or reshape as necessary.
    interpolated_features = interpolated_features.squeeze(-2)
    
    return F.normalize(interpolated_features, dim=1) if normalize else interpolated_features
    
    
class DINOV2(nn.Module):
    def __init__(self, stride=4):
        super().__init__()
        self.dinov2_vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').eval()
        self.transform = None
        self.stride = stride
    
    # rgb: 3, h, w
    def forward(self, rgb, pts):
        if self.transform is None:
            self.patch_h, self.patch_w = rgb.shape[-2] // self.stride, rgb.shape[-1] // self.stride
            self.transform = T.Compose([
                T.Resize((self.patch_h * 14, self.patch_w * 14)),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        result = self.dinov2_vit.forward_features(self.transform(rgb.unsqueeze(0)))
        raw_descs = result['x_norm_patchtokens'].reshape(1, self.patch_h, self.patch_w, -1).permute(0, 3, 1, 2)
        features = interpolate_features(raw_descs, pts[None], strides=self.stride, normalize=True)[0].T
        return features