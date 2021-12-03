import numpy as np
import scipy
import torch
from scipy.spatial import Delaunay
import torch
from ..ops.roiaware_pool3d import roiaware_pool3d_utils
from . import common_utils


def points_in_box_3d_label(points, boxes, slack=1.0, shift=np.array([[0,0,0,0,0,0]])):
    '''
    :param points: N, 3
    :param boxes: M, 8
    :param slack: scalar
    :return:
        foreground_pnt_mask: N
    '''
    center = boxes[:, :3]
    dim = boxes[:, 3:6]
    heading = boxes[:, 6]
    box_label = boxes[..., 7]
    N = points.shape[0]
    M = boxes.shape[0]
    assert N > 0, "point number == 0"
    if M == 0:
        return np.zeros_like(points[...,0])
    # M, 3, 3
    rotation = get_yaw_rotation(heading)
    # M, 4, 4
    transform = get_transform(rotation, center)
    # M, 4, 4
    transform = np.linalg.inv(transform)
    # M, 3, 3
    rotation = transform[:, :3, :3]
    # M, 3
    translation = transform[:, :3, 3]
    # N, M, 3
    point_in_box_frame = np.einsum("nj,mij->nmi", points, rotation) + translation
    # N, M, 3
    point_in_box_mask = np.logical_and(point_in_box_frame <= dim * 0.5 * slack + shift[:, 3:], point_in_box_frame >= -dim * 0.5 * slack + shift[:, :3])
    # N, M
    point_in_box_mask = np.prod(point_in_box_mask.astype(np.int8), axis=-1, dtype=np.int8)
    # N, M
    point_in_box_label = point_in_box_mask * box_label[np.newaxis, ...]
    # N
    point_label = np.max(point_in_box_label, axis=1)
    # print("point_label", point_label.shape)
    return point_label

def get_yaw_rotation(yaw):
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    ones = np.ones_like(yaw)
    zeros = np.zeros_like(yaw)
    return np.stack([
        np.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
        np.stack([sin_yaw, cos_yaw, zeros], axis=-1),
        np.stack([zeros, zeros, ones], axis=-1),
    ], axis=-2)

def get_transform(rotation, translation):
    translation_n_1=translation[..., np.newaxis]
    transform=np.concatenate([rotation, translation_n_1], axis=-1)
    last_row=np.zeros_like(translation)
    last_row = np.concatenate([last_row, np.ones_like(last_row[..., 0:1])], axis=-1)
    transform = np.concatenate([transform, last_row[..., np.newaxis, :]], axis=-2)
    return transform


def torch_points_and_sym_in_box_3d_batch(valid_points, valid_coords, boxes, boxes_num, batch_size, mirr_flag, slack=1.0):
    b_ind = valid_coords[:, 0]
    label_array = torch.zeros_like(valid_points[..., 0], dtype=torch.int8, device="cuda")
    mirr_inbox_point_lst = []
    mirr_bind_lst = []
    for i in range(batch_size):
        valid_b_num_box = boxes_num[i]
        valid_b_box = boxes[i, :valid_b_num_box, :]
        valid_b_mirr_flag = mirr_flag[i, :valid_b_num_box]
        vox_ind = torch.nonzero(torch.eq(b_ind, i))[:, 0]
        # print("list(vox_ind.shape)[0]",list(vox_ind.shape)[0])
        if list(vox_ind.shape)[0] > 0:
            valid_b_points = valid_points[vox_ind, :]
            label_array[vox_ind], mirr_points = torch_points_in_box_3d_label_mirr_points(valid_b_points, valid_b_box, valid_b_mirr_flag)
            if mirr_points is not None:
                mirr_bind_lst.append(torch.ones(mirr_points.shape[0], device=mirr_points.device, dtype=torch.int64) * i)
                mirr_inbox_point_lst.append(mirr_points)
        else:
            print("skip batch_{}, valid_points ind:".format(i), list(vox_ind.shape))
    if len(mirr_bind_lst) > 1:
        mirr_inbox_point = torch.cat(mirr_inbox_point_lst, dim=0)
        mirr_binds = torch.cat(mirr_bind_lst, dim=0)
    elif len(mirr_bind_lst) == 1:
        mirr_inbox_point = mirr_inbox_point_lst[0]
        mirr_binds = mirr_bind_lst[0]
    else:
        mirr_inbox_point, mirr_binds = None, None
    return label_array > 0, mirr_inbox_point, mirr_binds



def torch_points_in_box_3d_label_batch(valid_points, valid_coords, boxes, boxes_num, batch_size, slack=1.0):
    '''
    :param voxel_points: V, N, 3
    :param boxes: B M, 8
    :param slack: scalar
    :return:
        foreground_pnt_mask: N
    '''
    b_ind = valid_coords[:, 0]
    label_array = torch.zeros_like(valid_points[..., 0], dtype=torch.int8, device="cuda")
    for i in range(batch_size):
        valid_b_num_box = boxes_num[i]
        valid_b_box = boxes[i, :valid_b_num_box, :]
        vox_ind = torch.eq(b_ind, i).nonzero()[:,0]
        # print("list(vox_ind.shape)[0]",list(vox_ind.shape)[0])
        if list(vox_ind.shape)[0] > 0:
            valid_b_points = valid_points[vox_ind, :]
            label_array[vox_ind] = torch_points_in_box_3d_label(valid_b_points, valid_b_box, valid_b_num_box)[0]
        # else:
        #     print("119 skip batch_{}, valid_points ind:".format(i), list(vox_ind.shape))
    return label_array


def torch_points_in_box_3d_box_label_batch(valid_points, valid_coords, boxes, boxes_num, batch_size, slack=1.0):
    '''
    :param voxel_points: V, N, 3
    :param boxes: B M, 8
    :param slack: scalar
    :return:
        foreground_pnt_mask: N
    '''
    b_ind = valid_coords[:, 0]
    # label_array = torch.zeros_like(valid_points[..., 0], dtype=torch.int8, device="cuda")
    box_point_mask_lst = []
    valid_b_point_ind_lst = []
    for i in range(batch_size):
        valid_b_num_box = boxes_num[i]
        valid_b_box = boxes[i, :valid_b_num_box, :]
        vox_ind = torch.eq(b_ind, i).nonzero()[:,0]
        # print("list(vox_ind.shape)[0]",list(vox_ind.shape)[0])
        if list(vox_ind.shape)[0] > 0:
            valid_b_points = valid_points[vox_ind, :]
            point_in_box_mask = torch_points_in_box_3d_label_box_label(valid_b_points, valid_b_box, valid_b_num_box)
            box_point_mask_lst.append(point_in_box_mask)
            valid_b_point_ind_lst.append(vox_ind)
    return box_point_mask_lst, valid_b_point_ind_lst


def torch_points_in_box_3d_label_box_label(points, boxes, boxes_num, shift=0):
    '''
    :param points: N, 3
    :param boxes: M, 8
    :param slack: scalar
    :return:
        foreground_pnt_mask: N
    '''
    N = points.shape[0]
    M = boxes.shape[0]
    assert N > 0, "point number {} == 0".format(N)
    if M == 0:
        point_in_box_mask = torch.zeros((N,M), dtype=torch.int8, device="cuda")
    else:
        center = boxes[:, :3]
        dim = boxes[:, 3:6]
        heading = boxes[:, 6]
        box_label = boxes[..., 7].to(torch.int8)

        # M, 3, 3
        rotation = torch_get_yaw_rotation(heading)
        # M, 4, 4
        transform = torch_get_transform(rotation, center)
        # M, 4, 4
        transform = torch.inverse(transform)
        # M, 3, 3
        rotation = transform[:, :3, :3]
        # M, 3
        translation = transform[:, :3, 3]
        # N, M, 3
        # print("points, rotation, translation", points.shape, rotation.shape, translation.shape)
        point_in_box_frame = torch.einsum("nj,mij->nmi", points, rotation) + translation
        # N, M, 3
        point_in_box_mask = (point_in_box_frame <= dim * 0.5 + shift) & (point_in_box_frame >= -dim * 0.5 - shift)
        # N, M
        point_in_box_mask = torch.prod(point_in_box_mask, axis=-1, dtype=torch.int8)
        # point_labels = torch.max(point_in_box_mask, axis=1)[0]
        # box_labels = torch.max(point_in_box_mask, axis=0)[0]
    return point_in_box_mask


def torch_points_in_box_3d_label(points, boxes, boxes_num, shift=0):
    '''
    :param points: N, 3
    :param boxes: M, 8
    :param slack: scalar
    :return:
        foreground_pnt_mask: N
    '''
    N = points.shape[0]
    M = boxes.shape[0]
    assert N > 0, "point number {} == 0".format(N)
    if M == 0:
        if not isinstance(boxes_num, int):
            # N
            point_label_lst = [torch.zeros((N), dtype=torch.int8, device="cuda") for num in boxes_num]
        else:
            point_label_lst = [torch.zeros((N), dtype=torch.int8, device="cuda")]
    else:
        center = boxes[:, :3]
        dim = boxes[:, 3:6]
        heading = boxes[:, 6]
        box_label = boxes[..., 7].to(torch.int8)

        # M, 3, 3
        rotation = torch_get_yaw_rotation(heading)
        # M, 4, 4
        transform = torch_get_transform(rotation, center)
        # M, 4, 4
        transform = torch.inverse(transform)
        # M, 3, 3
        rotation = transform[:, :3, :3]
        # M, 3
        translation = transform[:, :3, 3]
        # N, M, 3
        # print("points, rotation, translation", points.shape, rotation.shape, translation.shape)
        point_in_box_frame = torch.einsum("nj,mij->nmi", points, rotation) + translation
        # N, M, 3
        point_in_box_mask = (point_in_box_frame <= dim * 0.5 + shift) & (point_in_box_frame >= -dim * 0.5 - shift)
        # N, M
        point_in_box_mask = torch.prod(point_in_box_mask, axis=-1, dtype=torch.int8)
        # N, M
        point_in_box_label = point_in_box_mask * torch.unsqueeze(box_label, 0)
        if not isinstance(boxes_num, int):
            # N
            point_in_box_label_lst = torch.split(point_in_box_label, boxes_num, dim=1)
            point_label_lst = [torch.max(point_in_box_label_lst[i], axis=1)[0] if boxes_num[i] > 0 else torch.zeros((N), dtype=torch.int8, device="cuda") for i in range(len(boxes_num))]
        else:
            point_label_lst=[torch.max(point_in_box_label, axis=1)[0]]
    return point_label_lst


def rotatez(points, zyaw, dim=-1):
    if list(points.shape)[dim] == 3:
        rotation = torch.transpose(torch_get_yaw_rotation(zyaw * np.pi / 180.), 0, 1)
    else:
        rotation = torch.transpose(torch_get_yaw_rotation_2d(zyaw * np.pi / 180.), 0, 1)

    points_rot = torch.matmul(points, rotation)
    # print("points", points.shape, rotation.shape, points_rot.shape)
    return points_rot


def torch_points_in_box_3d_label_mirr_points(points, boxes, mirr_flag, shift=0):
    '''
    :param points: N, 3
    :param boxes: M, 8
    :param slack: scalar
    :return:
        foreground_pnt_mask: N
    '''
    N = points.shape[0]
    M = boxes.shape[0]
    assert N > 0, "point number {} == 0".format(N)
    if M == 0:
        point_label = torch.zeros((N), dtype=torch.int8, device="cuda")
        mirror_fore_point = None
    else:
        center = boxes[:, :3]
        dim = boxes[:, 3:6]
        heading = boxes[:, 6]
        box_label = boxes[..., 7].to(torch.int8)
        # M, 3, 3
        rotation = torch_get_yaw_rotation(heading)
        # M, 4, 4
        transform = torch_get_transform(rotation, center)

        # M, 4, 4
        transform = torch.inverse(transform)
        # M, 3, 3
        reversed_rotation = transform[:, :3, :3]
        # M, 3
        reversed_translation = transform[:, :3, 3]
        # N, M, 3
        # print("points, rotation, translation", points.shape, rotation.shape, translation.shape)
        point_in_box_frame = torch.einsum("nj,mij->nmi", points, reversed_rotation) + reversed_translation
        # N, M, 3
        point_in_box_mask = (point_in_box_frame <= dim * 0.5 + shift) & (point_in_box_frame >= -dim * 0.5 - shift)
        # N, M
        point_in_box_mask = torch.prod(point_in_box_mask, axis=-1, dtype=torch.int8)
        # N, M
        mirr_point_in_box_mask = point_in_box_mask * (mirr_flag > 0.5).to(torch.int8).unsqueeze(0)
        # V, 2
        # point_in_box_inds = torch.nonzero(point_in_box_mask)
        mirr_point_in_box_inds = torch.nonzero(mirr_point_in_box_mask)

        mirror_point_in_box_frame = point_in_box_frame.clone()
        # N, M, 3
        mirror_point_in_box_frame[:, :, 1] = -mirror_point_in_box_frame[:, :, 1]
        # N, M, 3
        mirror_point_in_box_frame = torch.einsum("nmj,mij->nmi", mirror_point_in_box_frame, rotation) + center
        # P, 3
        mirror_fore_point = mirror_point_in_box_frame[mirr_point_in_box_inds[:, 0], mirr_point_in_box_inds[:, 1], :]
        # N, M
        point_in_box_label = point_in_box_mask * torch.unsqueeze(box_label, 0)
        # print("point_in_box_label", point_in_box_label.shape)
        point_label = torch.max(point_in_box_label, axis=1)[0]
    return point_label, mirror_fore_point



def torch_get_yaw_rotation(yaw):
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    ones = torch.ones_like(yaw)
    zeros = torch.zeros_like(yaw)
    return torch.stack([
        torch.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
        torch.stack([sin_yaw, cos_yaw, zeros], axis=-1),
        torch.stack([zeros, zeros, ones], axis=-1),
    ], axis=-2)



def torch_get_transform(rotation, translation):
    translation_n_1 = torch.unsqueeze(translation, -1)
    transform=torch.cat([rotation, translation_n_1], axis=-1)
    last_row=torch.zeros_like(translation)
    last_row = torch.cat([last_row, torch.ones_like(last_row[..., 0:1])], axis=-1)
    transform = torch.cat([transform, torch.unsqueeze(last_row, -2)], axis=-2)
    return transform


def torch_points_in_box_2d_mask(points, boxes, shift=0):
    '''
    :param points: N, 3
    :param boxes: M, 8
    :param slack: scalar
    :return:
        foreground_pnt_mask: N
    '''
    N = points.shape[0]
    assert N > 0, "point number == 0"
    M = boxes.shape[0]
    if M == 0:
        return torch.zeros((N), dtype=torch.int8, device="cuda")
    else:
        center = boxes[:, :2]
        dim = boxes[:, 3:5]
        heading = boxes[:, 6]
        # M, 3, 3
        rotation = torch_get_yaw_rotation_2d(heading)
        # M, 4, 4
        transform = torch_get_transform_2d(rotation, center)
        # M, 4, 4
        transform = torch.inverse(transform)
        # M, 3, 3
        rotation = transform[:, :2, :2]
        # M, 3
        translation = transform[:, :2, 2]
        # N, M, 3
        point_in_box_frame = torch.einsum("nj,mij->nmi", points, rotation) + translation
        # N, M, 3
        point_in_box_mask = (point_in_box_frame <= dim * 0.5 + shift) & (point_in_box_frame >= -dim * 0.5 - shift)
        # N, M
        point_in_box_mask = torch.prod(point_in_box_mask, axis=-1, dtype=torch.int8) > 0
        # N, M
        point_mask = torch.any(point_in_box_mask, axis=-1)
        # print("point_label", point_label.shape)
    return point_mask


def torch_get_yaw_rotation_2d(yaw):
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    return torch.stack([
        torch.stack([cos_yaw, -1.0 * sin_yaw], axis=-1),
        torch.stack([sin_yaw, cos_yaw], axis=-1),
    ], axis=-2)


def torch_get_transform_2d(rotation, translation):
    translation_n_1 = torch.unsqueeze(translation, -1)
    transform=torch.cat([rotation, translation_n_1], axis=-1)
    last_row=torch.zeros_like(translation)
    last_row = torch.cat([last_row, torch.ones_like(last_row[..., 0:1])], axis=-1)
    transform = torch.cat([transform, torch.unsqueeze(last_row, -2)], axis=-2)
    return transform