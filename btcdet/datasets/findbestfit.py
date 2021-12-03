import copy
import pickle
import sys
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from skimage import io
import mayavi.mlab as mlab
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from ..ops.roiaware_pool3d import roiaware_pool3d_utils
from ..utils import box_utils, calibration_kitti, common_utils, object3d_kitti, point_box_utils
from .dataset import DatasetTemplate
import torch
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
sys.path.append('/home/xharlie/dev/match2det/tools/visual_utils')
import visualize_utils as vu
from PIL import ImageColor
from ..ops.chamfer_distance import ChamferDistance
from ..ops.iou3d_nms import iou3d_nms_utils
chamfer_dist = ChamferDistance()


NUM_POINT_FEATURES = 4
def extract_allpnts(root_path=None, splits=['train','val']):
    all_db_infos_lst = []
    box_dims_lst = []
    pnts_lst = []
    mirrored_pnts_lst = []
    for split in splits:
        db_info_save_path = Path(root_path) / ('kitti_dbinfos_%s.pkl' % split)
        with open(db_info_save_path, 'rb') as f:
            all_db_infos = pickle.load(f)['Car']
        for k in range(len(all_db_infos)):
            info = all_db_infos[k]
            obj_type = info['name']
            if obj_type != "Car":
                continue
            gt_box = info['box3d_lidar']
            box_dims_lst.append(np.concatenate([np.zeros_like(gt_box[0:3]), np.array(gt_box[3:6]), np.zeros_like(gt_box[6:7])], axis=-1))
            all_db_infos_lst.append(info)
            obj_pnt_fpath = "/home/xharlie/dev/occlusion_pcd/data/kitti/detection3d/" + info['path']
            car_pnts = get_normalized_cloud(str(obj_pnt_fpath), gt_box, bottom=0.15)[:,:3]
            mirrored_car_pnts = mirror(car_pnts)
            pnts_lst.append(car_pnts)
            mirrored_pnts_lst.append(mirrored_car_pnts)
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

def sample_template(all_db_infos, indices, cluster_num, clu_id=None):
    if clu_id is None:
        clu_id = [i for i in range(len(indices))]
    for i in clu_id:
        inds = indices[i]
        # print("inds", inds.shape)
        pnts_lst = []
        gt_box_lst = []
        max_num_pnts = 0
        num_pnts_lst = []
        print("-----------cluster{}------------".format(i))
        for k in inds:
            info = all_db_infos[k]
            obj_pnt_fpath = "/home/xharlie/dev/occlusion_pcd/data/kitti/detection3d/" + info['path']
            gt_box = info['box3d_lidar']
            car_pnts = get_normalized_cloud(str(obj_pnt_fpath), gt_box, bottom=0.15)
            if car_pnts.shape[0] == 0:
                continue
            gt_box_lst.append(gt_box)
            pnts_lst.append(car_pnts)
            max_num_pnts = max(max_num_pnts, car_pnts.shape[0])
            num_pnts_lst.append(car_pnts.shape[0])
        num_pnts_arry = np.array(num_pnts_lst)
        ranks = np.argsort(num_pnts_arry)[::-1]
        gt_box_arry = np.array(gt_box_lst)
        num_template = 2
        tmplate_id_lst = cal_in_cluster_template(pnts_lst, max_num_pnts, num_pnts_arry, num_template=num_template)
        path = "/home/xharlie/dev/occlusion_pcd/kitti/detection3d/template/{}_{}/".format(cluster_num,num_template)
        os.makedirs(path, exist_ok=True)
        # ######## save ########
        # template_pnts = None
        # for ind in range(len(tmplate_id_lst)):
        #     tmp_id = tmplate_id_lst[ind]
        #     secnd_tmp = mirror(pnts_lst[tmp_id])
        #     save_pnts_box(secnd_tmp, gt_box_arry[tmp_id, :], "clstr_{}_tmplt_{}.bin".format(i,ind), path + "clstr_{}_tmplt_{}.bin".format(i,ind))
        #     if template_pnts is None:
        #         template_pnts = secnd_tmp
        #     else:
        #         secnd_tmp = remove_voxelpnts(template_pnts, secnd_tmp, voxel_size=[0.02, 0.02, 0.02], nearest_dist=0.05)
        #         template_pnts = np.concatenate([template_pnts, secnd_tmp], axis=0)
        # save_pnts_box(template_pnts, gt_box_arry[tmplate_id_lst[0], :], "clstr_{}_combine.bin".format(i), path + "clstr_{}_combine.bin".format(i))

        ####### iter view
        for tmp_id in tmplate_id_lst:
            for i in range(len(pnts_lst)):
                tind, sind = tmp_id, ranks[10]
                template_pnts = mirror(pnts_lst[tmp_id])
                vis_pair(template_pnts, gt_box_arry[tind:tind+1,:], pnts_lst[sind], gt_box_arry[sind:sind+1,:])
                break

        # ####### combine view
        # template_pnts = mirror(pnts_lst[tmplate_id_lst[0]])
        # for tmp_id in tmplate_id_lst[1:]:
        #     secnd_tmp = mirror(pnts_lst[tmp_id])
        #     secnd_tmp = remove_voxelpnts(template_pnts, secnd_tmp, voxel_size=[0.02, 0.02, 0.02], nearest_dist=0.05)
        #     template_pnts = np.concatenate([template_pnts, secnd_tmp], axis=0)
        # #
        # cal_in_cluster_template(pnts_lst, ranks, max_num_pnts, num_pnts_arry, num_template=1, template_pnts=template_pnts)
        # for i in range(len(pnts_lst)):
        #     tind, sind = tmplate_id_lst[0], ranks[10]
        #     # template_pnts = mirror(pnts_lst[tmp_id])
        #     vis_pair(template_pnts, gt_box_arry[tind:tind+1,:], pnts_lst[sind], gt_box_arry[sind:sind+1,:])
        #     break

        ######## batch vis
        # tind = tmplate_id_lst[0]
        # batch_vis_pair(template_pnts, gt_box_arry[tind:tind+1,:], pnts_lst, gt_box_arry, ranks)

def save_pnts_box(pnts, box, name, path):
    template = {
        "name": name,
        "points": pnts,
        "box": box
    }
    with open(path, 'wb') as f:
        pickle.dump(template, f)

def toTensor(sample):
    return torch.from_numpy(sample).float().to("cuda")

def get_batch_stats(dist, num_pnts_tensor, mask_arry, reversemask_arry):
    masked_dist = dist * mask_arry
    addmin_dist = masked_dist + reversemask_arry
    addmax_dist = masked_dist - reversemask_arry
    mean_instance = torch.sum(masked_dist, dim=1) / num_pnts_tensor # N CARS to the template
    min_instance = torch.min(addmin_dist, dim=1)[0]
    max_instance = torch.max(addmax_dist, dim=1)[0]
    return mean_instance, min_instance, max_instance


def cal_in_cluster_template(pnts_lst, max_num_pnts, num_pnts_arry, num_template=1, template_pnts=None):
    ########## single for loop ###########
    # tind, sind = ranks[rtind], ranks[1]
    # template_pnts = mirror(pnts_lst[tind][:, :3])
    # cd_4pose(pnts_lst[sind][:, :3], template_pnts)
    # vis_pair(template_pnts, gt_box_arry[tind:tind+1,:], pnts_lst[sind], gt_box_arry[sind:sind+1,:])
    ########## match for loop ###########
    pnts_padding_lst = []
    mask_lst = []
    reversemask_lst = []
    for i in range(len(pnts_lst)):
        padding_pnts = np.concatenate([pnts_lst[i], np.zeros([max_num_pnts-num_pnts_arry[i],4])], axis=0)
        mask = np.concatenate([np.ones([num_pnts_arry[i]], dtype=np.float), np.zeros([max_num_pnts-num_pnts_arry[i]], dtype=np.float)])
        reversemask =np.concatenate([np.zeros([num_pnts_arry[i]], dtype=np.float), 10.0 * np.ones([max_num_pnts-num_pnts_arry[i]], dtype=np.float)])
        pnts_padding_lst.append(padding_pnts)
        mask_lst.append(mask)
        reversemask_lst.append(reversemask)
    pnts_padding_arry = toTensor(np.array(pnts_padding_lst))
    mask_arry = toTensor(np.array(mask_lst))
    reversemask_arry = toTensor(np.array(reversemask_lst))
    num_pnts_tensor = toTensor(num_pnts_arry)

    if template_pnts is None:
        mean_lst = []
        max_lst = []
        min_lst = []
        for rtind in range(len(pnts_lst)):
            tind = rtind # ranks[rtind]
            template_pnts = mirror(pnts_lst[tind][:, :3])
            mean_instance, min_instance, max_instance = cal_template_cd_pntlst(template_pnts, pnts_padding_arry, num_pnts_tensor, mask_arry, reversemask_arry)
            mean_lst.append(mean_instance)
            max_lst.append(max_instance)
            min_lst.append(min_instance)

        mean_array = np.array(mean_lst)
        max_array = np.array(max_lst)
        min_array = np.array(min_lst)
        tmplate_id_lst = []

        for num in range(num_template):
            mean_max_array = np.mean(max_array, axis=1)
            max_max_array = np.max(max_array, axis=1)
            mean_mean_array = np.mean(mean_array, axis=1)
            fit_ranks = np.argsort(mean_max_array)
            for i in range(len(fit_ranks)):
                tmp_id = fit_ranks[i]
                # print("best tmplt id", tmp_id, "mean_mean", mean_mean_array[tmp_id], "mean_max", mean_max_array[tmp_id], "max_max", max_max_array[tmp_id])
                leftover_ind = max_array[tmp_id] > 0.3
                preserve_ind = max_array[tmp_id] <= 0.3
                if np.sum(preserve_ind) > 0:
                    print(i, " max_array", max_array.shape, "leftover_ind", np.sum(leftover_ind), "preserve_ind", np.sum(preserve_ind))
                    break
            if num == num_template-1:
                print("last tmplt id", tmp_id, "mean_mean", mean_mean_array[tmp_id], "mean_max", mean_max_array[tmp_id], "max_max", max_max_array[tmp_id])
            else:
                print("preserved tmplt id ", tmp_id, "mean_mean", np.mean(mean_array[tmp_id][preserve_ind]), "mean_max", np.mean(max_array[tmp_id][preserve_ind]), "max_max", np.max(max_array[tmp_id][preserve_ind]))
                mean_array = mean_array[:, leftover_ind]
                max_array = max_array[:, leftover_ind]
            tmplate_id_lst.append(tmp_id)
    else:
        mean_instance, min_instance, max_instance = cal_template_cd_pntlst(template_pnts, pnts_padding_arry, num_pnts_tensor, mask_arry, reversemask_arry)
        print("target tmplt ", "mean_mean", np.mean(mean_instance), "mean_max", np.mean(max_instance), "max_max", np.max(max_instance))
        tmplate_id_lst=None
    return tmplate_id_lst

def cal_template_cd_pntlst(template_pnts, pnts_padding_arry, num_pnts_tensor, mask_arry, reversemask_arry):
    batch_template_pnts = np.stack([template_pnts[:,:3] for j in range(list(pnts_padding_arry.shape)[0])])
    dist1, _ = chamfer_dist(pnts_padding_arry[:,:,:3], toTensor(batch_template_pnts))
    # print("dist1.shape", dist1.shape, pnts_padding_arry.shape, batch_template_pnts.shape)
    dist_l1 = torch.sqrt(dist1)
    mean_instance, min_instance, max_instance = get_batch_stats(dist_l1, num_pnts_tensor, mask_arry, reversemask_arry)
    return mean_instance.cpu().numpy(), min_instance.cpu().numpy(), max_instance.cpu().numpy()


def remove_voxelpnts(sourcepnts, target_pnts, voxel_size=np.array([0.01, 0.01, 0.01]), nearest_dist=None):
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

    vu.draw_scenes_multi(render_pnts_lst, colors_lst, size_lst, mode_lst, bgcolor=(1, 1, 1))#, gt_boxes=temp_box, gt_boxes_color=colors_lst[0], ref_boxes=gt_box_arry, ref_boxes_color=colors_lst[1])

    # vu.draw_scenes_multi([np.concatenate(moved_temp_lst[:3], axis=0), np.concatenate(moved_pnts_lst[:3], axis=0)], colors_lst, size_lst, mode_lst, bgcolor=(1, 1, 1), gt_boxes=temp_box, gt_boxes_color=colors_lst[0], ref_boxes=gt_box_arry, ref_boxes_color=colors_lst[1])

    mlab.show()


def vis_pair(template, temp_box, pnts, pnts_box):
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
    # vu.draw_scenes_multi(box_pnt_lst, colors_lst, size_lst, mode_lst, bgcolor=(1,1,1))
    vu.draw_scenes_multi(pnts_lst, colors_lst, size_lst, mode_lst, bgcolor=(1, 1, 1), gt_boxes=temp_box, gt_boxes_color=colors_lst[0], ref_boxes=pnts_box, ref_boxes_color=colors_lst[1])
    # axes = mlab.axes()
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



def find_best_match_boxpnts(sorted_iou, pnt_thresh_best_iou_indices, mirrored_pnts_lst, coords_num, occ_map):
    best_match_ind_lst = []
    # best_match_pnts_lst = []
    best_match_iou_lst = []
    best_match_mean_dist_lst = []
    best_match_max_dist_lst = []
    for car_id in range(len(mirrored_pnts_lst)):
        cur_mirrored_pnts_lst = [mirrored_pnts_lst[car_id]]
        picked_indices = tuple(pnt_thresh_best_iou_indices[car_id].cpu())
        selected_mirrored_pnts_lst = [mirrored_pnts_lst[i] for i in picked_indices]
        # print("pnt_thresh_best_iou_indices[car_id]", pnt_thresh_best_iou_indices[car_id].shape, coords_num.shape)
        cur_occ_map = occ_map[car_id]
        selected_occ_map = occ_map[pnt_thresh_best_iou_indices[car_id]]
        selected_coord_nums = coords_num[pnt_thresh_best_iou_indices[car_id]]
        iou, match_car_ind, vox_num, extra_vox_num, mean_dist, max_dist = find_single_best_match_boxpnts(sorted_iou[car_id], cur_mirrored_pnts_lst, selected_mirrored_pnts_lst, selected_coord_nums, pnt_thresh_best_iou_indices[car_id], cur_occ_map, selected_occ_map)
        print("{}/{}: match_car_ind {}, iou {}, vox_num {}, extra_vox_num {}, mean_dist {}, max_dist {}".format(car_id, len(mirrored_pnts_lst), match_car_ind, iou, vox_num, extra_vox_num, mean_dist, max_dist))
        best_match_ind_lst.append(match_car_ind.cpu().numpy().astype(np.int32))
        # best_match_pnts_lst.append(mirrored_pnts_lst[match_car_ind])
        best_match_iou_lst.append(iou.cpu().numpy())
        best_match_mean_dist_lst.append(mean_dist.cpu().numpy()),
        best_match_max_dist_lst.append(max_dist.cpu().numpy())
    return best_match_ind_lst, best_match_iou_lst, best_match_mean_dist_lst, best_match_max_dist_lst


def get_batch_stats(dist, num_pnts_tensor, mask_arry, reversemask_arry):
    masked_dist = dist * mask_arry
    addmin_dist = masked_dist + reversemask_arry
    addmax_dist = masked_dist - reversemask_arry
    mean_instance = torch.sum(masked_dist, dim=1) / num_pnts_tensor  # N CARS to the template
    min_instance = torch.min(addmin_dist, dim=1)[0]
    max_instance = torch.max(addmax_dist, dim=1)[0]
    mean_instance[mean_instance != mean_instance] = 100.0
    return mean_instance, min_instance, max_instance


def find_single_best_match_boxpnts(sorted_iou, cur_mirrored_pnts_lst, picked_mirrored_pnts_lst, selected_coord_nums, selected_indices, cur_occ_map, selected_occ_map):
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
        print("dist_l1", dist_l1.shape, mirr_box_pnts_padding_tensor.shape)
        mean_instance, min_instance, max_instance = get_batch_stats(dist_l1, box_num_pnts_tensor, box_mask_tensor, box_reversemask_tensor)
        mean_instance = mean_instance.view(gt_boxnum, candidate_num)
        min_instance = min_instance.view(gt_boxnum, candidate_num)
        max_instance = max_instance.view(gt_boxnum, candidate_num)
    else:
        mean_instance = torch.zeros([gt_boxnum, candidate_num], device="cuda", dtype=torch.float32)
        max_instance = mean_instance.clone()

    extra_coord_nums = extra_occ(cur_occ_map, selected_occ_map, selected_coord_nums)
    heuristic = max_instance + 10. / extra_coord_nums + (sorted_iou < 0.84) * 1.0 # mean_instance + 10. / extra_coord_nums + (sorted_iou < 0.84) * 1.0 #
    heur_sorted, heur_indices = torch.sort(heuristic, dim=1)
    gt_index = torch.arange(gt_boxnum, device="cuda", dtype=torch.int64).unsqueeze(1).repeat(1, candidate_num)
    # print("heur_indices", heur_indices.shape, mean_instance.shape, torch.max(selected_coord_nums))
    # print("selected_coord_nums {}, mean_instance {}, min_instance {}, max_instance {}".format(selected_coord_nums[heur_indices][0,0], mean_instance[gt_index[0,0], heur_indices[0,0]], min_instance[gt_index[0,0], heur_indices[0,0]], max_instance[gt_index[0,0], heur_indices[0,0]]))
    # print("selected_indices", selected_indices.shape, heur_indices[0,0].shape, selected_indices[heur_indices[0,0]].shape, selected_coord_nums[heur_indices[0,0]].shape, mean_instance[gt_index[0,0]].shape, max_instance[gt_index[0,0]].shape)
    return sorted_iou[heur_indices[0,0]], selected_indices[heur_indices[0,0]], selected_coord_nums[heur_indices[0,0]], extra_coord_nums[heur_indices[0,0]], mean_instance[gt_index[0,0], heur_indices[0,0]], max_instance[gt_index[0,0], heur_indices[0,0]]


def extra_occ(cur_occ_map, selected_occ_map, selected_coord_nums):
    print("cur_occ_map, selected_occ_map", cur_occ_map.shape, selected_occ_map.shape)
    candi_num, nx, ny = list(selected_occ_map.shape)
    cur_occ_map = cur_occ_map.view(1, nx, ny).repeat(candi_num, 1, 1)
    return selected_coord_nums - torch.sum((selected_occ_map * cur_occ_map).view(-1, nx*ny), dim=1)


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
    PNT_THRESH = 400
    ROOT_DIR = (Path(__file__).resolve().parent / '../../').resolve()
    print("ROOT_DIR", ROOT_DIR)
    path = ROOT_DIR / 'data' / 'kitti' / 'detection3d'
    match_info_save_path = path / "match_maxdist_10extcrdsnum_info_car.pkl"
    cluster_num = 20
    voxel_size = [0.08, 0.08, 0.08]
    all_db_infos_lst, box_dims_lst, pnts_lst, mirrored_pnts_lst = extract_allpnts(
        root_path=path, splits=['train','val']
    )
    box_tensor = torch.as_tensor(box_dims_lst, device="cuda", dtype=torch.float32)
    iou3d = get_iou(box_tensor)
    range_mirrored = np.array([np.concatenate([np.min(mirrored_pnts_lst[i], axis=0), np.max(mirrored_pnts_lst[i], axis=0)], axis=-1) for i in range(len(mirrored_pnts_lst)) if mirrored_pnts_lst[i].shape[0] > 0])
    allrange = np.concatenate([np.min(range_mirrored[...,:3], axis=0), np.max(range_mirrored[...,3:], axis=0)], axis=-1)
    nx, ny = np.ceil((allrange[3] - allrange[0]) / voxel_size[0]).astype(np.int), np.ceil((allrange[4] - allrange[1]) / voxel_size[1]).astype(np.int32)
    occ_map = torch.stack([torch.as_tensor(space_occ_voxelpnts(mirrored_pnts_lst[i], allrange, nx, ny, voxel_size=voxel_size), device="cuda", dtype=torch.int32) for i in range(len(mirrored_pnts_lst))], dim=0) # M nx ny
    coords_num = torch.sum(occ_map.view(-1, nx*ny), dim=1)
    print("coords_num", coords_num.shape)
    coord_inds = torch.nonzero(coords_num > PNT_THRESH)[...,0]
    iou3d = iou3d[:, coord_inds]
    sorted_iou, best_iou_indices = torch.topk(iou3d, 800, dim=-1, sorted=True)
    pnt_thresh_best_iou_indices = coord_inds[best_iou_indices]
    # print("sorted_iou", torch.min(sorted_iou), best_iou_indices.shape, best_iou_indices[0,:5])
    best_match_ind_lst, best_match_iou_lst, best_match_mean_dist_lst, best_match_max_dist_lst = find_best_match_boxpnts(sorted_iou, pnt_thresh_best_iou_indices, mirrored_pnts_lst, coords_num, occ_map)
    # db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
    #            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
    #            'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
    match_info = {'Car':{}}
    for i in range(len(all_db_infos_lst)):
        info = all_db_infos_lst[i]
        # print("info", info)
        match_info['Car'][(int(info['image_idx']), int(info['gt_idx']))] = {"dbinfo_ind": best_match_ind_lst[i], "bbox_iou": best_match_iou_lst[i], "mean_dist": best_match_mean_dist_lst[i], "max_dist": best_match_max_dist_lst[i], "path": all_db_infos_lst[best_match_ind_lst[i]]['path'], "box3d_lidar": all_db_infos_lst[best_match_ind_lst[i]]['box3d_lidar']}
    print("length", len(match_info.keys()), len(all_db_infos_lst))
    with open(match_info_save_path, 'wb') as f:
        pickle.dump(match_info, f)






