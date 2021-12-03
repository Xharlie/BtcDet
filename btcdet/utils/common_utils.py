import logging
import os
import pickle
import random
import shutil
import subprocess

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(angle.shape[0])
    ones = angle.new_ones(angle.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:%d' % tcp_port,
        rank=local_rank,
        world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def trilinear_interpolate_torch(im, x, y, z):
    """
    Args:
        im: (Z, H, W, C) [z, y, x]
        x: (N)
        y: (N)
        z: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    z0 = torch.floor(z).long()
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, im.shape[2] - 1)
    x1 = torch.clamp(x1, 0, im.shape[2] - 1)
    y0 = torch.clamp(y0, 0, im.shape[1] - 1)
    y1 = torch.clamp(y1, 0, im.shape[1] - 1)
    z0 = torch.clamp(z0, 0, im.shape[0] - 1)
    z1 = torch.clamp(z1, 0, im.shape[0] - 1)

    I000 = im[z0, y0, x0]
    I010 = im[z0, y1, x0]
    I001 = im[z0, y0, x1]
    I011 = im[z0, y1, x1]
    I100 = im[z1, y0, x0]
    I110 = im[z1, y1, x0]
    I101 = im[z1, y0, x1]
    I111 = im[z1, y1, x1]

    w000 =  (z1.type_as(z) - z) * (y1.type_as(y) - y) * (x1.type_as(x) - x)
    w010 = -(z1.type_as(z) - z) * (y0.type_as(y) - y) * (x1.type_as(x) - x)
    w001 = -(z1.type_as(z) - z) * (y1.type_as(y) - y) * (x0.type_as(x) - x)
    w011 =  (z1.type_as(z) - z) * (y0.type_as(y) - y) * (x0.type_as(x) - x)
    w100 = -(z0.type_as(z) - z) * (y1.type_as(y) - y) * (x1.type_as(x) - x)
    w110 =  (z0.type_as(z) - z) * (y0.type_as(y) - y) * (x1.type_as(x) - x)
    w101 =  (z0.type_as(z) - z) * (y1.type_as(y) - y) * (x0.type_as(x) - x)
    w111 = -(z0.type_as(z) - z) * (y0.type_as(y) - y) * (x0.type_as(x) - x)

    ans = torch.t((torch.t(I000) * w000)) + torch.t((torch.t(I010) * w010)) + torch.t((torch.t(I001) * w001)) + torch.t((torch.t(I011) * w011)) + torch.t((torch.t(I100) * w100)) + torch.t((torch.t(I110) * w110)) + torch.t((torch.t(I101) * w101)) + torch.t((torch.t(I111) * w111))
    return ans


def reverse_sparse_trilinear_interpolate_torch(feat, b, zyx, normalize=False):
    """
    Args:
        im: (Z, H, W, C) [z, y, x]
        x: (N)
        y: (N)
        z: (N)

    Returns:

    """
    im, spatial_shape = feat.dense(), feat.spatial_shape
    x, y, z = zyx[..., 2], zyx[..., 1], zyx[..., 0]
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    z0 = torch.floor(z).long()
    z1 = z0 + 1

    if normalize:
        z0_mask = 1
        z1_mask = 1
        y0_mask = 1
        y1_mask = 1
        x0_mask = 1
        x1_mask = 1
    else:
        z0_mask = ((z0 >= 0) & (z0 < spatial_shape[0])).unsqueeze(-1)
        z1_mask = ((z1 >= 0) & (z1 < spatial_shape[0])).unsqueeze(-1)
        y0_mask = ((y0 >= 0) & (y0 < spatial_shape[1])).unsqueeze(-1)
        y1_mask = ((y1 >= 0) & (y1 < spatial_shape[1])).unsqueeze(-1)
        x0_mask = ((x0 >= 0) & (x0 < spatial_shape[2])).unsqueeze(-1)
        x1_mask = ((x1 >= 0) & (x1 < spatial_shape[2])).unsqueeze(-1)

    w000 = torch.abs((z1.type_as(z) - z) * (y1.type_as(y) - y) * (x1.type_as(x) - x))
    w010 = torch.abs(-(z1.type_as(z) - z) * (y0.type_as(y) - y) * (x1.type_as(x) - x))
    w001 = torch.abs(-(z1.type_as(z) - z) * (y1.type_as(y) - y) * (x0.type_as(x) - x))
    w011 = torch.abs((z1.type_as(z) - z) * (y0.type_as(y) - y) * (x0.type_as(x) - x))
    w100 = torch.abs(-(z0.type_as(z) - z) * (y1.type_as(y) - y) * (x1.type_as(x) - x))
    w110 = torch.abs((z0.type_as(z) - z) * (y0.type_as(y) - y) * (x1.type_as(x) - x))
    w101 = torch.abs((z0.type_as(z) - z) * (y1.type_as(y) - y) * (x0.type_as(x) - x))
    w111 = torch.abs(-(z0.type_as(z) - z) * (y0.type_as(y) - y) * (x0.type_as(x) - x))

    x0 = torch.clamp(x0, 0, spatial_shape[2] - 1)
    x1 = torch.clamp(x1, 0, spatial_shape[2] - 1)
    y0 = torch.clamp(y0, 0, spatial_shape[1] - 1)
    y1 = torch.clamp(y1, 0, spatial_shape[1] - 1)
    z0 = torch.clamp(z0, 0, spatial_shape[0] - 1)
    z1 = torch.clamp(z1, 0, spatial_shape[0] - 1)

    I000 = im[b, :, z0, y0, x0] * z0_mask * y0_mask * x0_mask # [1, 65536, 352]
    I010 = im[b, :, z0, y1, x0] * z0_mask * y1_mask * x0_mask
    I001 = im[b, :, z0, y0, x1] * z0_mask * y0_mask * x1_mask
    I011 = im[b, :, z0, y1, x1] * z0_mask * y1_mask * x1_mask
    I100 = im[b, :, z1, y0, x0] * z1_mask * y0_mask * x0_mask
    I110 = im[b, :, z1, y1, x0] * z1_mask * y1_mask * x0_mask
    I101 = im[b, :, z1, y0, x1] * z1_mask * y0_mask * x1_mask
    I111 = im[b, :, z1, y1, x1] * z1_mask * y1_mask * x1_mask


    ans = I000 * w000.unsqueeze(-1) + I010 * w010.unsqueeze(-1) + I001 * w001.unsqueeze(-1) + I011 * w011.unsqueeze(-1) + I100 * w100.unsqueeze(-1) + I110 * w110.unsqueeze(-1) + I101 * w101.unsqueeze(-1) + I111 * w111.unsqueeze(-1)
    return ans



def sparse_trilinear_interpolate_torch(features, binds, idxs, grid_size):
    N, F = list(features.shape)
    z, y, x = idxs[...,2], idxs[...,1], idxs[...,0],
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1
    z0 = torch.floor(z)
    z1 = z0 + 1

    w000 =  torch.abs((z1 - z) * (y1 - y) * (x1 - x))
    w010 =  torch.abs(-(z1 - z) * (y0 - y) * (x1 - x))
    w001 =  torch.abs(-(z1 - z) * (y1 - y) * (x0 - x))
    w011 =  torch.abs((z1 - z) * (y0 - y) * (x0 - x))
    w100 =  torch.abs(-(z0 - z) * (y1 - y) * (x1 - x))
    w110 =  torch.abs((z0 - z) * (y0 - y) * (x1 - x))
    w101 =  torch.abs((z0 - z) * (y1 - y) * (x0 - x))
    w111 =  torch.abs(-(z0 - z) * (y0 - y) * (x0 - x))

    x0 = torch.clamp(x0, -1, grid_size[2])
    x1 = torch.clamp(x1, -1, grid_size[2])
    y0 = torch.clamp(y0, -1, grid_size[1])
    y1 = torch.clamp(y1, -1, grid_size[1])
    z0 = torch.clamp(z0, -1, grid_size[0])
    z1 = torch.clamp(z1, -1, grid_size[0])


    weights = torch.stack([w000, w010, w001, w011, w100, w110, w101, w111], dim=-1).unsqueeze(-1) # N, 8, 1
    x0, y0, z0, x1, y1, z1 = x0.type_as(binds), y0.type_as(binds), z0.type_as(binds), x1.type_as(binds), y1.type_as(binds), z1.type_as(binds)
    idxs = torch.stack([torch.stack([z0, y0, x0], dim=-1), torch.stack([z0, y1, x0], dim=-1), torch.stack([z0, y0, x1], dim=-1), torch.stack([z0, y1, x1], dim=-1), torch.stack([z1, y0, x0], dim=-1), torch.stack([z1,y1,x0], dim=-1), torch.stack([z1, y0, x1], dim=-1), torch.stack([z1, y1, x1], dim=-1)], dim=1).view(N*8, 3) # N*8, 3
    bidxs = torch.cat([binds.repeat(1,8).view(-1,1), idxs], dim=-1)
    features = (features.view(N, 1, F) * weights).view(-1, F) # N*8, F
    idx_mask = torch.all(idxs >= 0, dim=-1) & torch.all(idxs < torch.as_tensor(grid_size, device=idxs.device).unsqueeze(0), dim=-1)
    features, bidxs = features[idx_mask], bidxs[idx_mask]
    unique_idxs, reverse_idxs, labels_count = bidxs.unique(dim=0, return_inverse=True, return_counts=True)
    # print(binds.dtype, bidxs.dtype, unique_idxs.dtype, reverse_idxs.dtype)
    features = torch.zeros([unique_idxs.shape[0], F], dtype=torch.float32, device=features.device).index_add_(0, reverse_idxs.long(), features)
    # print("features", unique_idxs.shape, reverse_idxs.shape, features.shape) # torch.Size([147940, 4]) torch.Size([313096]) torch.Size([147940, 64])
    # print("unique_idxs", unique_idxs[0], unique_idxs[-1])
    return unique_idxs.long(), features



def bilinear_interpolate_torch(im, x, y, normalize=False):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1
    if normalize:
        y0_mask = 1
        y1_mask = 1
        x0_mask = 1
        x1_mask = 1
    else:
        y0_mask = ((y0 >= 0) & (y0 < im.shape[0])).unsqueeze(-1)
        y1_mask = ((y1 >= 0) & (y1 < im.shape[0])).unsqueeze(-1)
        x0_mask = ((x0 >= 0) & (x0 < im.shape[1])).unsqueeze(-1)
        x1_mask = ((x1 >= 0) & (x1 < im.shape[1])).unsqueeze(-1)

    wa = torch.abs((x1.type_as(x) - x) * (y1.type_as(y) - y))
    wb = torch.abs((x1.type_as(x) - x) * (y - y0.type_as(y)))
    wc = torch.abs((x - x0.type_as(x)) * (y1.type_as(y) - y))
    wd = torch.abs((x - x0.type_as(x)) * (y - y0.type_as(y)))

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0] * y0_mask * x0_mask
    Ib = im[y1, x0] * y1_mask * x0_mask
    Ic = im[y0, x1] * y0_mask * x1_mask
    Id = im[y1, x1] * y1_mask * x1_mask


    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans