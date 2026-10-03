# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2023 Tal Daniel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Extracted from utils/util_func.py at LPWM 4cf53c403433e64c01652ac2adbec66231a46dea.
# ruff: noqa
# fmt: off

import math
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F


def reparameterize(mu, logvar, eps=None, return_eps=False):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    if eps is None:
        eps = torch.randn_like(mu, device=device)
    z = mu + eps * std
    if return_eps:
        return z, eps
    else:
        return z


def create_masks_fast(center, anchor_s, feature_dim=16, patch_size=None):
    # center: [batch_size, n_kp, 2] in kp_range
    # anchor_h, anchor_w: size of anchor in [0, 1]
    batch_size, n_kp = center.shape[0], center.shape[1]
    if patch_size is None:
        patch_size = np.round(anchor_s * (feature_dim - 1)).astype(int)
    # create white rectangles
    masks = torch.ones(batch_size * n_kp, 1, patch_size, patch_size, device=center.device).float()
    # pad the masks to image size
    pad_size = (feature_dim - patch_size) // 2
    padded_patches_batch = F.pad(masks, pad=[pad_size] * 4)
    # move the masks to be centered around the kp
    delta_t_batch = 0.0 - center
    delta_t_batch = delta_t_batch.reshape(-1, delta_t_batch.shape[-1])  # [bs * n_kp, 2]
    zeros = torch.zeros([delta_t_batch.shape[0], 1], device=delta_t_batch.device).float()
    ones = torch.ones([delta_t_batch.shape[0], 1], device=delta_t_batch.device).float()
    theta = torch.cat([ones, zeros, delta_t_batch[:, 1].unsqueeze(-1),
                       zeros, ones, delta_t_batch[:, 0].unsqueeze(-1)], dim=-1)
    theta = theta.view(-1, 2, 3)  # [batch_size * n_kp, 2, 3]
    mode = "nearest"
    # mode = 'bilinear'

    trans_padded_patches_batch = affine_grid_sample(padded_patches_batch, theta, padded_patches_batch.shape, mode=mode)

    trans_padded_patches_batch = trans_padded_patches_batch.view(batch_size, n_kp, *padded_patches_batch.shape[1:])
    # [bs, n_kp, 1, feature_dim, feature_dim]
    return trans_padded_patches_batch


def create_masks_with_scale(kp_batch, anchor_s, image_size, scale=None, scale_normalized=False):
    """
    translate patches to be centered around given keypoints
    kp_batch: [bs, n_kp, 2] in [-1, 1]
    patches: [bs, n_kp, ch_patches, patch_size, patch_size]
    scale: None or [bs, n_kp, 2] or [bs, n_kp, 1]
    scale_normalized: False if scale is not in [0, 1]
    :return: translated_padded_patches [bs, n_kp, ch, img_size, img_size]
    """
    patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
    patches_batch = torch.ones(kp_batch.shape[0], kp_batch.shape[1], 1, patch_size, patch_size,
                               device=kp_batch.device, dtype=torch.float)
    batch_size, n_kp, ch_patch, patch_size, _ = patches_batch.shape
    img_size = image_size
    if scale is None:
        z_scale = (patch_size / img_size) * torch.ones_like(kp_batch)
    else:
        # normalize to [0, 1]
        if scale_normalized:
            z_scale = scale
        else:
            z_scale = torch.sigmoid(scale)  # -> [0, 1]
    z_pos = kp_batch.reshape(-1, kp_batch.shape[-1])  # [bs * n_kp, 2]
    z_scale = z_scale.view(-1, z_scale.shape[-1])  # [bs * n_kp, 2]
    patches_batch = patches_batch.reshape(-1, *patches_batch.shape[2:])
    out_dims = (batch_size * n_kp, ch_patch, img_size, img_size)
    trans_patches_batch = spatial_transform(patches_batch, z_pos, z_scale, out_dims, inverse=True)
    trans_padded_patches_batch = trans_patches_batch.view(batch_size, n_kp, *trans_patches_batch.shape[1:])
    # [bs, n_kp, 1, img_size, img_size]
    return trans_padded_patches_batch


def spatial_transform(image, z_pos, z_scale, out_dims, inverse=False, eps=1e-9, padding_mode="zeros"):
    """
    https://github.com/zhixuan-lin/G-SWM
    spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    ------
    image: [batch_size * n_kp, ch, h, w]
    z_pos: [batch_size * n_kp, 2]
    z_scale: [batch_size * n_kp, 2]
    out_dims: tuple (batch_size * n_kp, ch, h*, w*)
    """
    # 0. validate values range
    # z_pos = z_pos.clamp(-1, 1)
    # z_scale = z_scale.clamp(0, 1)
    # 1. construct 2x3 affine matrix for each datapoint in the batch
    theta = torch.zeros(2, 3, device=image.device).repeat(image.shape[0], 1, 1)
    # set scaling
    theta[:, 0, 0] = z_scale[:, 1] if not inverse else 1 / (z_scale[:, 1] + eps)
    theta[:, 1, 1] = z_scale[:, 0] if not inverse else 1 / (z_scale[:, 0] + eps)

    # set translation
    theta[:, 0, -1] = z_pos[:, 1] if not inverse else - z_pos[:, 1] / (z_scale[:, 1] + eps)
    theta[:, 1, -1] = z_pos[:, 0] if not inverse else - z_pos[:, 0] / (z_scale[:, 0] + eps)
    # construct sampling grid and sample image from grid
    return affine_grid_sample(image, theta, out_dims, mode='bilinear', padding_mode=padding_mode)


def modulate(x, scale, shift, residual=False):
    if residual:
        return (scale + 1.0) * x + shift
    else:
        return scale * x + shift


@torch.jit.script
def affine_grid_sample(x, theta, out_dims: Tuple[int, int, int, int], mode: str, align_corners: bool = False,
                       padding_mode: str = 'zeros'):
    # construct sampling grid
    grid = F.affine_grid(theta, torch.Size(out_dims), align_corners=align_corners)
    # sample image from grid
    return F.grid_sample(x, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)
