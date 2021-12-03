import copy
import pickle
import sys
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from skimage import io
import mayavi.mlab as mlab
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from ..ops.roiaware_pool3d import roiaware_pool3d_utils
from ..utils import box_utils, calibration_kitti, common_utils, object3d_kitti, point_box_utils
from .dataset import DatasetTemplate
import torch
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
sys.path.append('/home/xharlie/dev/occlusion_pcd/tools/visual_utils')
import visualize_utils as vu
from PIL import ImageColor
from ..ops.chamfer_distance import ChamferDistance
from ..ops.iou3d_nms import iou3d_nms_utils
chamfer_dist = ChamferDistance()


NUM_POINT_FEATURES = 4
def extract_allpnts(root_path=None, splits=['train','val'], type='Car', apply_mirror=True):
    all_db_infos_lst = []
    box_dims_lst = []
    pnts_lst = []
    mirrored_pnts_lst = []
    for split in splits:
        db_info_save_path = Path(root_path) / ('kitti_dbinfos_%s.pkl' % split)
        with open(db_info_save_path, 'rb') as f:
            all_db_infos = pickle.load(f)[type]
        for k in range(len(all_db_infos)):
            info = all_db_infos[k]
            obj_type = info['name']
            if obj_type != type:
                continue
            gt_box = info['box3d_lidar']
            box_dims_lst.append(np.concatenate([np.zeros_like(gt_box[0:3]), np.array(gt_box[3:6]), np.zeros_like(gt_box[6:7])], axis=-1))
            all_db_infos_lst.append(info)
            obj_pnt_fpath = "/home/xharlie/dev/occlusion_pcd/data/kitti/detection3d/" + info['path']
            car_pnts = get_normalized_cloud(str(obj_pnt_fpath), gt_box, bottom=0.15)[:,:3]
            mirrored_car_pnts = mirror(car_pnts)
            pnts_lst.append(car_pnts)
            if apply_mirror:
                mirrored_pnts_lst.append(mirrored_car_pnts)
            else:
                mirrored_pnts_lst.append(car_pnts)
    return all_db_infos_lst, box_dims_lst, pnts_lst, mirrored_pnts_lst

def clustering(m_nm, num_cluster, box_dims_lst):
    train_box_dims, val_box_dims = box_dims_lst[0], box_dims_lst[1]
    if m_nm=="kmeans":
        clusterer = KMeans(n_clusters=num_cluster, random_state=1).fit(train_box_dims)
    elif m_nm == "DBSCAN":
        clusterer = DBSCAN(eps=0.3, min_samples=10).fit(train_box_dims)
        core_samples_mask = np.zeros_like(clusterer.labels_, dtype=bool)
        core_samples_mask[clusterer.core_sample_indices_] = True
        labels = clusterer.labels_
    # print(train_box_dims.shape, clusterer.labels_.shape)
    # print(clusterer.cluster_centers_)
    # print(np.min(train_box_dims, axis=0))
    indices = [np.asarray((clusterer.labels_ == i).nonzero())[0,:] for i in range(cluster_num)]
    return clusterer, indices

def get_normalized_cloud(obj_pnt_fpath, gt_box, bottom=0.0):
    pnts = np.fromfile(str(obj_pnt_fpath), dtype=np.float32).reshape([-1,4])
    pnts = np.concatenate([single_rotate_points_along_z(pnts[:,:3], -gt_box[6]), pnts[:,3:]], axis=1)
    return remove_bottom(pnts, gt_box, bottom)

def remove_bottom(pnts, gt_box, bottom):
    if bottom == 0.0:
        return pnts
    zthresh =  - gt_box[5] / 2 + bottom
    keep_bool = pnts[:, 2] > zthresh
    return pnts[keep_bool]

def vis_cluster(clusterer, box_dims, cluster_num):
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # For each set of style and range settings, plot n random points in the box
    # # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    # for i in range(box_dims.shape[0]):
    #     xs = box_dims[i,0]
    #     ys = box_dims[i,1]
    #     zs = box_dims[i,2]
    #     ax.scatter(xs, ys, zs, c=colors[clusterer.labels_[i]])
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # ax.set_aspect('equal')
    #
    # plt.show()
    binary = [clusterer.labels_ == i for i in range(cluster_num)]
    box_pnt_lst = [box_dims[binary[i]] for i in range(cluster_num)]
    colors_lst = [tuple(np.array(ImageColor.getcolor(colors[i], "RGB"))/255.0) for i in range(cluster_num)]
    size_lst = [0.02 for i in range(cluster_num)]
    mode_lst = ["sphere" for i in range(cluster_num)]
    # vu.draw_scenes_multi(box_pnt_lst, colors_lst, size_lst, mode_lst, bgcolor=(1,1,1))
    vu.draw_scenes_multi(box_pnt_lst, colors_lst, size_lst, mode_lst, bgcolor=(1,1,1))
    axes = mlab.axes()
    mlab.show()

def save_pnts_box(pnts, box, name, path):
    template = {
        "name": name,
        "points": pnts,
        "box": box
    }
    with open(path, 'wb') as f:
        pickle.dump(template, f)


def find_overlaps(base, aug):
    x, y = base, aug
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y
    return mask

def coords3inds(coords, ny, nx):
    gperm1 = nx * ny
    gperm2 = nx
    zdim = coords[:, 2] * gperm1
    ydim = coords[:, 1] * gperm2
    xdim = coords[:, 0]
    inds = zdim + ydim + xdim
    return inds.astype(np.int32)



def mirror(pnts, lastchannel=3):
    mirror_pnts = np.concatenate([pnts[...,0:1], -pnts[...,1:2], pnts[...,2:lastchannel]], axis=-1)
    mirror_pnts = remove_voxelpnts(pnts, mirror_pnts, nearest_dist=0.05)
    return np.concatenate([pnts, mirror_pnts], axis=0)


def batch_vis_pair(template, temp_box, pnts_lst, gt_box_arry, ranks):
    moved_temp_lst = []
    moved_pnts_lst = []
    temp_box = np.tile(temp_box, [len(pnts_lst), 1])
    temp_box[:, -1] = np.zeros_like(temp_box[:, -1])
    gt_box_arry[:, -1] = np.zeros_like(gt_box_arry[:, -1])
    width = int(np.ceil(np.sqrt(len(pnts_lst))) * 1.2)
    height = int(np.ceil(np.sqrt(len(pnts_lst))) / 1.2)
    x = (np.arange(height) - height // 2) * 6
    y = (np.arange(width) - width // 2) * 4
    xv, yv = np.meshgrid(x, y, indexing='ij')
    xv, yv = xv.reshape(-1), yv.reshape(-1)

    for ind in range(len(pnts_lst)):
        i = ranks[ind]
        shift = np.array([[xv[ind], yv[ind], 0]], dtype=np.float)
        temp_box[i,:3], gt_box_arry[i,:3] = shift[0], shift[0]
        # print(template.shape, pnts.shape)
        colors = ['#e6194b', '#4363d8', '#3cb44b', '#ffe119', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
        moved_pnts_lst.append(pnts_lst[i] + shift)
        moved_temp_lst.append(template + shift)
        # print(shift, template.shape, shift.shape, np.mean((template + shift), axis=0))
    # print("xv",xv.shape, len(moved_pnts_lst), len(moved_temp_lst))
    moved_temp_pnt = np.concatenate(moved_temp_lst, axis=0)
    moved_pnts = np.concatenate(moved_pnts_lst, axis=0)
    tmp_section = len(moved_temp_pnt) // 200000 + 1
    pnt_section = len(moved_pnts) // 200000 + 1
    render_pnts_lst = [moved_temp_pnt[i*200000:(i+1)*200000] for i in range(tmp_section)] + [moved_pnts[i*200000:(i+1)*200000] for i in range(pnt_section)]
    colors_lst = [tuple(np.array(ImageColor.getcolor(colors[0], "RGB")) / 255.0) for i in range(tmp_section)] + [tuple(np.array(ImageColor.getcolor(colors[1], "RGB")) / 255.0) for i in range(pnt_section)]
    size_lst = [0.02 for i in range(tmp_section)] + [0.04 for i in range(pnt_section)]
    mode_lst = ["sphere" for i in range(tmp_section)] + ["sphere" for i in range(pnt_section)]

    vu.draw_scenes_multi(render_pnts_lst, colors_lst, size_lst, mode_lst, bgcolor=(1, 1, 1)) #, gt_boxes=temp_box, gt_boxes_color=colors_lst[0], ref_boxes=gt_box_arry, ref_boxes_color=colors_lst[1])

    # vu.draw_scenes_multi([np.concatenate(moved_temp_lst[:3], axis=0), np.concatenate(moved_pnts_lst[:3], axis=0)], colors_lst, size_lst, mode_lst, bgcolor=(1, 1, 1), gt_boxes=temp_box, gt_boxes_color=colors_lst[0], ref_boxes=gt_box_arry, ref_boxes_color=colors_lst[1])
    mlab.show()


def vis_pair(template, temp_box, pnts, pnts_box):
    if temp_box is not None and pnts_box is not None:
        temp_box[:,:3] = np.zeros_like(temp_box[:,:3])
        temp_box[:,-1] = np.zeros_like(pnts_box[:,-1])
        pnts_box[:,:3] = np.zeros_like(pnts_box[:,:3])
        pnts_box[:,-1] = np.zeros_like(pnts_box[:,-1])
    # print(template.shape, pnts.shape)
    colors = ['#e6194b', '#4363d8', '#3cb44b', '#ffe119', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
              '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
              '#000075', '#808080', '#ffffff', '#000000']
    pnts_lst = [template, pnts]
    colors_lst = [tuple(np.array(ImageColor.getcolor(colors[i], "RGB")) / 255.0) for i in range(2)]
    size_lst = [0.02 for i in range(2)]
    mode_lst = ["sphere" for i in range(2)]
    # vu.draw_scenes_multi(pnts_lst, colors_lst, size_lst, mode_lst, bgcolor=(1, 1, 1), gt_boxes=temp_box, ref_boxes=pnts_box)

    vu.draw_scenes_multi([pnts_lst[0]], [colors_lst[0]], [size_lst[0]], [mode_lst[0]], bgcolor=(1, 1, 1), gt_boxes=None, ref_boxes=None)

    vu.draw_scenes_multi([pnts_lst[1]], [colors_lst[1]], [size_lst[1]], [mode_lst[1]], bgcolor=(1, 1, 1), gt_boxes=None, ref_boxes=None)
    mlab.show()


def cd_4pose(scene, template):
    # points and points_reconstructed are n_points x 3 matrices
    dist1, _ = chamfer_dist(toTensor(scene), toTensor(template))
    # print("dist1.shape", dist1.shape)
    dist_l1 = torch.sqrt(dist1)
    # mean_l1, min_l1, max_l1 = torch.mean(dist_l1, dim=1), torch.min(dist_l1, dim=1)[0], torch.max(dist_l1, dim=1)[0]
    # print("mean_l1 {}, min_l1 {}, max_l1 {}".format(mean_l1, min_l1, max_l1))
    return dist_l1

def single_rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """

    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros_like(angle)
    ones = np.ones_like(angle)
    rot_matrix = np.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=0).reshape(3, 3)
    points_rot = np.matmul(points[:, 0:3], rot_matrix)
    return points_rot


def get_iou(box_tensor):
    limit = len(box_dims_lst)
    start = 0
    iou3d_lst = []
    for i in range(11):
        end = min(start + limit // 10, limit)
        iou3d = iou3d_nms_utils.boxes_iou3d_gpu(box_tensor[start:end, :], box_tensor)
        iou3d_lst.append(iou3d)
        start = end
    iou3d = torch.cat(iou3d_lst, dim=0)
    print("iou3d", iou3d.shape)
    return iou3d


def padding_pnt_tensors(pnts_lst, max_num_pnts = None, num_pnts_arry = None):
    pnts_padding_lst = []
    mask_lst = []
    reversemask_lst = []
    for i in range(len(pnts_lst)):
        if isinstance(pnts_lst[i], torch.Tensor):
            padding_pnts = torch.cat([pnts_lst[i], torch.zeros([max_num_pnts - num_pnts_arry[i], 3], dtype=torch.float, device="cuda")], dim=0)
        else:
            padding_pnts = np.concatenate([pnts_lst[i], np.zeros([max_num_pnts - num_pnts_arry[i], 3])], axis=0)
        mask = np.concatenate([np.ones([num_pnts_arry[i]], dtype=np.float),
            np.zeros([max_num_pnts - num_pnts_arry[i]], dtype=np.float)])
        reversemask = np.concatenate([np.zeros([num_pnts_arry[i]], dtype=np.float),
            10.0 * np.ones([max_num_pnts - num_pnts_arry[i]], dtype=np.float)])
        pnts_padding_lst.append(padding_pnts)
        mask_lst.append(mask)
        reversemask_lst.append(reversemask)
    if isinstance(pnts_padding_lst[0], torch.Tensor):
        pnts_padding_tensor = torch.stack(pnts_padding_lst, dim=0)
    else:
        pnts_padding_tensor = toTensor(np.array(pnts_padding_lst))
    mask_tensor = toTensor(np.array(mask_lst))
    reversemask_tensor = toTensor(np.array(reversemask_lst))
    num_pnts_tensor = toTensor(num_pnts_arry)
    return pnts_padding_tensor, mask_tensor, reversemask_tensor, num_pnts_tensor


def toTensor(sample):
    return torch.from_numpy(sample).float().to("cuda")


def get_padding_boxpnts_tensors(point_in_box_lst):
    max_num_pnts = 0
    num_pnts_lst = []
    for point_in_box in point_in_box_lst:
        max_num_pnts = max(max_num_pnts, len(point_in_box))
        num_pnts_lst.append(len(point_in_box))
    num_pnts_array = np.array(num_pnts_lst)
    box_pnts_padding_tensor, box_mask_tensor, box_reversemask_tensor, box_num_pnts_tensor = padding_pnt_tensors(point_in_box_lst, max_num_pnts, num_pnts_array)
    return box_pnts_padding_tensor, box_mask_tensor, box_reversemask_tensor, box_num_pnts_tensor, num_pnts_array


def repeat_boxpoints_tensor(boxpoint_tensor, candidate_num):
    if boxpoint_tensor.dim() == 3:
        gt_boxnum, max_point_num, point_dims = list(boxpoint_tensor.shape)
        box_pnts_padding_tensor = torch.unsqueeze(boxpoint_tensor, dim=1).repeat(1, candidate_num, 1, 1).view(gt_boxnum * candidate_num, max_point_num, point_dims)
    elif boxpoint_tensor.dim() == 2:
        gt_boxnum, max_point_num = list(boxpoint_tensor.shape)
        box_pnts_padding_tensor = torch.unsqueeze(boxpoint_tensor, dim=1).repeat(1, candidate_num, 1).view(gt_boxnum * candidate_num, max_point_num)
    else:
        gt_boxnum = list(boxpoint_tensor.shape)[0]
        box_pnts_padding_tensor = torch.unsqueeze(boxpoint_tensor, dim=1).repeat(1, candidate_num).view(gt_boxnum * candidate_num)
    return box_pnts_padding_tensor



def find_best_match_boxpnts(all_db_infos_lst, box_dims_lst, sorted_iou, pnt_thresh_best_iou_indices, mirrored_pnts_lst, pnts_lst, coords_num, occ_map, bm_dir, allrange, nx, ny, voxel_size, max_num_bm=5, num_extra_coords=2000, iou_thresh=0.84, ex_coords_ratio=10., nearest_dist=0.16, vis=False, save=False):
    '''
    :param all_db_infos_lst: list of info
    :param box_dims_lst: M * 7
    :param sorted_iou: sorted top 800 iou: M * 800
    :param pnt_thresh_best_iou_indices: mirror car indices with coords num > 400 and top 800 iou: M * 800
    :param mirrored_pnts_lst: M lst
    :param coords_num: M
    :param occ_map: M * dim
    :param max_num_bm: 5
    :return:
    '''
    for car_id in range(0, len(mirrored_pnts_lst)):
        cur_mirrored_pnts_lst = [mirrored_pnts_lst[car_id]]
        cur_pnts_lst = [pnts_lst[car_id]]
        print("pnt_thresh_best_iou_indices", pnt_thresh_best_iou_indices.shape)
        picked_indices = tuple(pnt_thresh_best_iou_indices[car_id].cpu())
        selected_mirrored_pnts_lst = [mirrored_pnts_lst[i] for i in picked_indices]
        selected_pnts_lst = [pnts_lst[i] for i in picked_indices]
        # print("pnt_thresh_best_iou_indices[car_id]", pnt_thresh_best_iou_indices[car_id].shape, coords_num.shape)
        cur_occ_map = occ_map[car_id]
        selected_occ_map = torch.stack([torch.as_tensor(space_occ_voxelpnts(remove_outofbox(selected_mirrored_pnts_lst[i], box_dims_lst[car_id]), allrange, nx, ny, voxel_size=voxel_size), device="cuda", dtype=torch.int32) for i in range(len(selected_mirrored_pnts_lst))], dim=0)  # M nx ny
        selected_sorted_iou, cur_box, selected_pnt_thresh_best_iou_indices = sorted_iou[car_id], box_dims_lst[car_id], pnt_thresh_best_iou_indices[car_id]
        bm_pnts, bm_coords_num = find_multi_best_match_boxpnts(selected_sorted_iou, cur_box, cur_mirrored_pnts_lst, cur_pnts_lst, selected_mirrored_pnts_lst, selected_pnts_lst, selected_pnt_thresh_best_iou_indices, cur_occ_map, selected_occ_map, max_num_bm=max_num_bm, num_extra_coords=num_extra_coords, iou_thresh=iou_thresh, ex_coords_ratio=ex_coords_ratio, nearest_dist=nearest_dist, vis=vis)

        info = all_db_infos_lst[car_id]
        image_idx, gt_idx = str(int(info['image_idx'])), str(int(info['gt_idx']))
        if save:
            with open(os.path.join(bm_dir, image_idx+"_"+gt_idx+".pkl"), 'wb') as f:
                pickle.dump(bm_pnts.astype(np.float32), f)
            print("{}/{}: bm_vox_num {}, bm_pnt_num {} ".format(car_id, len(mirrored_pnts_lst), bm_coords_num, bm_pnts.shape[0]))


def remove_outofbox(pnts, box):
    dim = box[3:6]
    point_in_box_mask = np.logical_and(pnts <= dim * 0.5, pnts >= -dim * 0.5)
    # N, M
    point_in_box_mask = np.prod(point_in_box_mask.astype(np.int8), axis=-1, dtype=bool)
    return pnts[point_in_box_mask, :]


def get_batch_stats(dist, num_pnts_tensor, mask_arry, reversemask_arry):
    masked_dist = dist * mask_arry
    addmin_dist = masked_dist + reversemask_arry
    addmax_dist = masked_dist - reversemask_arry
    mean_instance = torch.sum(masked_dist, dim=1) / num_pnts_tensor  # N CARS to the template
    min_instance = torch.min(addmin_dist, dim=1)[0]
    max_instance = torch.max(addmax_dist, dim=1)[0]
    mean_instance[mean_instance != mean_instance] = 100.0
    return mean_instance, min_instance, max_instance


def find_multi_best_match_boxpnts(sorted_iou, gt_box, cur_mirrored_pnts_lst, cur_pnts_lst, picked_mirrored_pnts_lst, picked_pnts_lst, selected_indices, cur_occ_map, selected_occ_map, max_num_bm=5, num_extra_coords=2000, iou_thresh = 0.84, ex_coords_ratio=10., nearest_dist=0.16, vis=False):

    gt_boxnum = len(cur_mirrored_pnts_lst)
    box_pnts_padding_tensor, box_mask_tensor, box_reversemask_tensor, box_num_pnts_tensor, box_num_pnts_array = get_padding_boxpnts_tensors(cur_mirrored_pnts_lst)
    mirr_box_pnts_padding_tensor, mirr_box_mask_tensor, mirr_box_reversemask_tensor, mirr_box_num_pnts_tensor, mirr_box_num_pnts_array = get_padding_boxpnts_tensors(picked_mirrored_pnts_lst)

    candidate_num, num_max_template_points, point_dims = list(mirr_box_pnts_padding_tensor.shape)
    mirr_box_reversemask_tensor_remote = mirr_box_pnts_padding_tensor + torch.unsqueeze(mirr_box_reversemask_tensor, dim=-1)
    box_pnts_padding_tensor = repeat_boxpoints_tensor(box_pnts_padding_tensor, candidate_num)
    box_num_pnts_tensor = repeat_boxpoints_tensor(box_num_pnts_tensor, candidate_num)
    box_mask_tensor = repeat_boxpoints_tensor(box_mask_tensor, candidate_num)
    box_reversemask_tensor = repeat_boxpoints_tensor(box_reversemask_tensor, candidate_num)
    if box_pnts_padding_tensor.shape[-2] > 0:
        dist1, _ = chamfer_dist(box_pnts_padding_tensor, mirr_box_reversemask_tensor_remote) # candidate_num X max num pnt X 3
        dist_l1 = torch.sqrt(dist1)
        # print("dist_l1", dist_l1.shape, mirr_box_pnts_padding_tensor.shape)
        mean_instance, min_instance, max_instance = get_batch_stats(dist_l1, box_num_pnts_tensor, box_mask_tensor, box_reversemask_tensor)
        mean_instance = mean_instance.view(gt_boxnum, candidate_num)
        # min_instance = min_instance.view(gt_boxnum, candidate_num)
        max_instance = max_instance.view(gt_boxnum, candidate_num)
    else:
        mean_instance = torch.zeros([gt_boxnum, candidate_num], device="cuda", dtype=torch.float32)
        max_instance = mean_instance.clone()
    aug_map = cur_occ_map
    bm_pnts = cur_mirrored_pnts_lst[0]
    oneside_bm_pnts = cur_pnts_lst[0]
    aug_coords_num = 0
    for round in range(max_num_bm):
        extra_coord_nums = extra_occ(aug_map, selected_occ_map)
        heuristic = max_instance + ex_coords_ratio / extra_coord_nums.unsqueeze(0) + (sorted_iou.unsqueeze(0) < iou_thresh) * 2.0  + (extra_coord_nums.unsqueeze(0) < 30) * 1.0 # mean_instance + 10. / extra_coord_nums + (sorted_iou < 0.84) * 1.0 #
        min_heur_sorted, min_heur_indices = torch.min(heuristic, dim=1)
        bm_iou, bm_match_car_ind, bm_extra_vox_num, bm_match_occ_map = sorted_iou[min_heur_indices], selected_indices[min_heur_indices], extra_coord_nums[min_heur_indices], selected_occ_map[min_heur_indices, ...]
        if (bm_iou.cpu() < iou_thresh and bm_pnts.shape[0] > 0) or bm_extra_vox_num.cpu() == 0:
            break
        ind = min_heur_indices.cpu().item()
        added_pnts = remove_voxelpnts(bm_pnts, picked_mirrored_pnts_lst[ind], nearest_dist=nearest_dist)
        if vis:
            # vis_pair(added_pnts, None, bm_pnts, np.expand_dims(gt_box, axis=0))
            vis_pair(picked_mirrored_pnts_lst[ind], None, bm_pnts, np.expand_dims(gt_box, axis=0))
            vis_pair(picked_pnts_lst[ind], None, oneside_bm_pnts, np.expand_dims(gt_box, axis=0))
        if added_pnts.shape[0] > 4:
            bm_pnts = np.concatenate([bm_pnts, added_pnts], axis=0)
            aug_map = aug_map | bm_match_occ_map
            aug_coords_num = torch.sum(aug_map).cpu()
            print("added_pnts", bm_pnts.shape, added_pnts.shape, ind, picked_mirrored_pnts_lst[ind].shape, aug_coords_num, "bm_extra_vox_num", bm_extra_vox_num)
        if len(sorted_iou) == 1 or aug_coords_num >= num_extra_coords:
            break
        elif ind == len(sorted_iou) - 1:
            sorted_iou, selected_indices, selected_occ_map, max_instance, mean_instance = sorted_iou[:ind], selected_indices[:ind], selected_occ_map[:ind], max_instance[:,:ind], mean_instance[:,:ind]
        elif ind == 0:
            sorted_iou, selected_indices, selected_occ_map, max_instance, mean_instance = sorted_iou[ind + 1:], selected_indices[ind + 1:], selected_occ_map[ind + 1:], max_instance[:,ind + 1:], mean_instance[:,ind + 1:]
        else:
            sorted_iou = torch.cat([sorted_iou[:ind], sorted_iou[ind+1:]], dim=0)
            selected_indices = torch.cat([selected_indices[:ind], selected_indices[ind+1:]], dim=0)
            selected_occ_map = torch.cat([selected_occ_map[:ind], selected_occ_map[ind+1:]], dim=0)
            max_instance = torch.cat([max_instance[:,:ind], max_instance[:,ind+1:]], dim=1)
            mean_instance = torch.cat([mean_instance[:,:ind], mean_instance[:,ind+1:]], dim=1)

    print("finish one ")
    return bm_pnts, aug_coords_num


def remove_voxelpnts(sourcepnts, target_pnts, voxel_size=np.array([0.08, 0.08, 0.08]), nearest_dist=None):
    augpnts = target_pnts[:,:3]
    gtpnts = sourcepnts[:,:3]
    if nearest_dist is None:
        min_gtpnts, max_gtpnts, min_augpnts, max_augpnts = np.min(gtpnts, axis=0), np.max(gtpnts, axis=0), np.min(augpnts, axis=0), np.max(augpnts, axis=0)
        range = np.concatenate([np.minimum(min_gtpnts, min_augpnts), np.maximum(max_gtpnts, max_augpnts)], axis=0)
        gtpnts_ind = np.floor((gtpnts - np.expand_dims(range[:3], axis=0)) / np.expand_dims(voxel_size, axis=0))
        augpnts_ind = np.floor((augpnts - np.expand_dims(range[:3], axis=0)) / np.expand_dims(voxel_size, axis=0))
        nx, ny = np.ceil((range[3]-range[0]) / voxel_size[0]).astype(np.int), np.ceil((range[4]-range[1]) / voxel_size[1]).astype(np.int)
        mask = find_overlaps(coords3inds(gtpnts_ind, nx, ny), coords3inds(augpnts_ind, nx, ny))
        # print("augpnts_ind", mask.shape, augpnts_ind.shape, augpnts_ind[mask].shape)
    else:
        dist_l1 = cd_4pose(np.expand_dims(augpnts, axis=0), np.expand_dims(gtpnts, axis=0))
        mask = dist_l1.cpu().numpy()[0] > nearest_dist
    return target_pnts[mask]


def extra_occ(cur_occ_map, selected_occ_map):
    # print("cur_occ_map, selected_occ_map", cur_occ_map.shape, selected_occ_map.shape)
    candi_num, nx, ny = list(selected_occ_map.shape)
    excluded_map = (1-cur_occ_map).view(1, nx, ny).repeat(candi_num, 1, 1)
    return torch.sum((selected_occ_map * excluded_map).view(-1, nx*ny), dim=1)


def space_occ_voxelpnts(sourcepnts, allrange, nx, ny, voxel_size=[0.08, 0.08, 0.08]):
    occmap = np.zeros([nx, ny], dtype=np.int32)
    if sourcepnts.shape[0] > 0:
        voxel_size = np.array(voxel_size)
        gtpnts = sourcepnts[:, :3]
        gtpnts_ind = np.floor((gtpnts - np.expand_dims(allrange[:3], axis=0)) / np.expand_dims(voxel_size, axis=0)).astype(int)
        occmap[gtpnts_ind[...,0], gtpnts_ind[...,1]] = np.ones_like(gtpnts_ind[...,0], dtype = np.int32)
        # unique_coords_num = np.sum(occmap)
    return occmap


if __name__ == '__main__':
    vis = False
    save = True # False
    voxel_size = [0.16, 0.16, 0.16]

    obj_types = ['Car', 'Cyclist', 'Pedestrian']
    apply_mirror_lst = [True, True, False]
    PNT_THRESH_lst = [80, 5, 5]
    ex_coords_ratio_lst = [50, 5, 5]
    max_num_bm_lst = [2, 1, 1]
    nearest_dist_lst = [0.10, 0.05, 0.05]
    iou_thresh_lst = [0.90, 0.90, 0.90]
    num_extra_coords_lst = [2000, 2000, 2000]
    for i, obj_type in enumerate(obj_types):
        ROOT_DIR = (Path(__file__).resolve().parent / '../../').resolve()
        print("ROOT_DIR", ROOT_DIR)
        path = ROOT_DIR / 'data' / 'kitti' / 'detection3d'
        bm_dir_save_path = path / "bm_{}maxdist_{}num_{}/".format(ex_coords_ratio_lst[i], max_num_bm_lst[i], obj_type)
        os.makedirs(bm_dir_save_path, exist_ok=True)
        all_db_infos_lst, box_dims_lst, pnts_lst, mirrored_pnts_lst = extract_allpnts(
            root_path=path, splits=['train','val'], type=obj_type, apply_mirror=apply_mirror_lst[i]
        )
        box_tensor = torch.as_tensor(box_dims_lst, device="cuda", dtype=torch.float32)
        iou3d = get_iou(box_tensor)
        range_mirrored = np.array([np.concatenate([np.min(mirrored_pnts_lst[i], axis=0), np.max(mirrored_pnts_lst[i], axis=0)], axis=-1) for i in range(len(mirrored_pnts_lst)) if mirrored_pnts_lst[i].shape[0] > 0])
        allrange = np.concatenate([np.min(range_mirrored[...,:3], axis=0), np.max(range_mirrored[...,3:], axis=0)], axis=-1)
        nx, ny = np.ceil((allrange[3] - allrange[0]) / voxel_size[0]).astype(np.int), np.ceil((allrange[4] - allrange[1]) / voxel_size[1]).astype(np.int32)
        occ_map = torch.stack([torch.as_tensor(space_occ_voxelpnts(mirrored_pnts_lst[i], allrange, nx, ny, voxel_size=voxel_size), device="cuda", dtype=torch.int32) for i in range(len(mirrored_pnts_lst))], dim=0) # M nx ny
        coords_num = torch.sum(occ_map.view(-1, nx*ny), dim=1)
        print("coords_num", coords_num.shape, torch.min(coords_num), torch.max(coords_num))
        coord_inds = torch.nonzero(coords_num > PNT_THRESH_lst[i])[...,0]
        print("coord_inds", coord_inds.shape)
        iou3d = iou3d[:, coord_inds]
        sorted_iou, best_iou_indices = torch.topk(iou3d, min(800, len(iou3d)), dim=-1, sorted=True, largest=True)
        pnt_thresh_best_iou_indices = coord_inds[best_iou_indices]
        print("best_iou_indices", best_iou_indices.shape, "pnt_thresh_best_iou_indices", pnt_thresh_best_iou_indices.shape, len(mirrored_pnts_lst), )
        # exit()
        # print("sorted_iou", torch.min(sorted_iou), best_iou_indices.shape, best_iou_indices[0,:5])
        find_best_match_boxpnts(all_db_infos_lst, box_dims_lst, sorted_iou, pnt_thresh_best_iou_indices, mirrored_pnts_lst, pnts_lst, coords_num, occ_map, bm_dir_save_path, allrange, nx, ny, voxel_size, max_num_bm=max_num_bm_lst[i], num_extra_coords=num_extra_coords_lst[i], iou_thresh=iou_thresh_lst[i], ex_coords_ratio=ex_coords_ratio_lst[i], nearest_dist=nearest_dist_lst[i], vis=vis, save=save)







