# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def interpolate_camera_poses(
    src_indices: np.ndarray,
    src_rot_mat: np.ndarray,
    src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> torch.Tensor:
    interp_func_trans = interp1d(
        src_indices,
        src_trans_vec,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    quats = src_quat_vec.as_quat().copy()
    for index in range(1, len(quats)):
        if np.dot(quats[index], quats[index - 1]) < 0:
            quats[index] = -quats[index]
    src_quat_vec = Rotation.from_quat(quats)
    interpolated_rot_quat = Slerp(src_indices, src_quat_vec)(tgt_indices)
    interpolated_rot_mat = interpolated_rot_quat.as_matrix()

    poses = np.zeros((len(tgt_indices), 4, 4), dtype=np.float32)
    poses[:, :3, :3] = interpolated_rot_mat
    poses[:, :3, 3] = interpolated_trans_vec
    poses[:, 3, 3] = 1.0
    return torch.from_numpy(poses).float()


def se3_inverse(matrix: torch.Tensor) -> torch.Tensor:
    rotation = matrix[:, :3, :3]
    translation = matrix[:, :3, 3:]
    rotation_inv = rotation.transpose(-1, -2)
    translation_inv = -torch.bmm(rotation_inv, translation)
    matrix_inv = torch.eye(4, device=matrix.device, dtype=matrix.dtype)[None, :, :].repeat(matrix.shape[0], 1, 1)
    matrix_inv[:, :3, :3] = rotation_inv
    matrix_inv[:, :3, 3:] = translation_inv
    return matrix_inv


def compute_relative_poses(
    c2ws_mat: torch.Tensor,
    *,
    framewise: bool = False,
    normalize_trans: bool = True,
) -> torch.Tensor:
    ref_w2cs = se3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise and relative_poses.shape[0] > 1:
        relative_poses[1:] = torch.bmm(se3_inverse(relative_poses[:-1]), relative_poses[1:])
    if normalize_trans:
        translations = relative_poses[:, :3, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    return relative_poses


@torch.no_grad()
def create_meshgrid(
    n_frames: int,
    height: int,
    width: int,
    *,
    bias: float = 0.5,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias
    return grid_xy[None, ...].repeat(n_frames, 1, 1)


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    *,
    only_rays_d: bool = False,
) -> torch.Tensor:
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(n_frames, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    fx, fy, cx, cy = intrinsics.chunk(4, dim=-1)

    i = grid_xy[..., 0]
    j = grid_xy[..., 1]
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack([xs, ys, zs], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    rays_d = directions @ c2ws_mat[:, :3, :3].transpose(-1, -2)

    if only_rays_d:
        return rays_d.view([n_frames, height, width, 3])

    rays_o = c2ws_mat[:, :3, 3][:, None, :].expand_as(rays_d)
    return torch.cat([rays_o, rays_d], dim=-1).view([n_frames, height, width, 6])


def get_intrinsics_transformed(
    intrinsics: torch.Tensor,
    *,
    height_org: int,
    width_org: int,
    height_resize: int,
    width_resize: int,
    height_final: int,
    width_final: int,
) -> torch.Tensor:
    fx, fy, cx, cy = intrinsics.chunk(4, dim=-1)

    scale_x = width_resize / width_org
    scale_y = height_resize / height_org

    fx_resize = fx * scale_x
    fy_resize = fy * scale_y
    cx_resize = cx * scale_x
    cy_resize = cy * scale_y

    crop_offset_x = (width_resize - width_final) / 2
    crop_offset_y = (height_resize - height_final) / 2

    intrinsics_transformed = torch.zeros_like(intrinsics)
    intrinsics_transformed[:, 0:1] = fx_resize
    intrinsics_transformed[:, 1:2] = fy_resize
    intrinsics_transformed[:, 2:3] = cx_resize - crop_offset_x
    intrinsics_transformed[:, 3:4] = cy_resize - crop_offset_y
    return intrinsics_transformed
