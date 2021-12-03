import copy
import pickle
import sys
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from skimage import io
import mayavi.mlab as mlab
import os
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
chamfer_dist = ChamferDistance()


NUM_POINT_FEATURES = 4
def extract_template(root_path=None, splits=['train','val']):
    all_db_infos_lst = []
    box_dims_lst = []
    for split in splits:
        db_info_save_path = Path(root_path) / ('kitti_dbinfos_%s.pkl' % split)

        with open(db_info_save_path, 'rb') as f:
            all_db_infos = pickle.load(f)['Car']
        box_dims = []
        pnts = []
        for k in range(len(all_db_infos)):
            info = all_db_infos[k]
            obj_type = info['name']
            if obj_type != "Car":
                continue
            gt_box = info['box3d_lidar']
            box_dims.append(gt_box[3:6])
        all_db_infos_lst.append(all_db_infos[::1])
        box_dims_lst.append(np.array(box_dims)[::1,:])
    return all_db_infos_lst, box_dims_lst

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
        num_template = 5
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
        # for tmp_id in tmplate_id_lst:
        #     for i in range(len(pnts_lst)):
        #         tind, sind = tmp_id, ranks[10]
        #         template_pnts = mirror(pnts_lst[tmp_id])
        #         vis_pair(template_pnts, gt_box_arry[tind:tind+1,:], pnts_lst[sind], gt_box_arry[sind:sind+1,:])
        #         break

        ####### combine view
        # template_pnts = mirror(pnts_lst[tmplate_id_lst[0]])
        # for tmp_id in tmplate_id_lst[1:]:
        #     secnd_tmp = mirror(pnts_lst[tmp_id])
        #     secnd_tmp = remove_voxelpnts(template_pnts, secnd_tmp, voxel_size=[0.02, 0.02, 0.02], nearest_dist=0.05)
        #     template_pnts = np.concatenate([template_pnts, secnd_tmp], axis=0)
        #
        # cal_in_cluster_template(pnts_lst, ranks, max_num_pnts, num_pnts_arry, num_template=1, template_pnts=template_pnts)
        # for i in range(len(pnts_lst)):
        #     tind, sind = tmplate_id_lst[0], ranks[10]
        #     # template_pnts = mirror(pnts_lst[tmp_id])
        #     vis_pair(template_pnts, gt_box_arry[tind:tind+1,:], pnts_lst[sind], gt_box_arry[sind:sind+1,:])
        #     break

        ######## batch vis
        # template_pnts = mirror(pnts_lst[tmplate_id_lst[0]])
        # for tmp_id in tmplate_id_lst[1:]:
        #     secnd_tmp = mirror(pnts_lst[tmp_id])
        #     secnd_tmp = remove_voxelpnts(template_pnts, secnd_tmp, voxel_size=[0.02, 0.02, 0.02], nearest_dist=0.05)
        #     template_pnts = np.concatenate([template_pnts, secnd_tmp], axis=0)
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



def mirror(pnts):
    mirror_pnts = np.concatenate([pnts[...,0:1], -pnts[...,1:2], pnts[...,2:4]], axis=-1)
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

    colors = ['#e6194b', '#4363d8', '#3cb44b', '#ffe119', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
              '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
              '#000075', '#808080', '#ffffff', '#000000']

    for ind in range(len(pnts_lst)):
        i = ranks[ind]
        shift = np.array([[xv[ind], yv[ind], 0]], dtype=np.float)
        temp_box[i,:3], gt_box_arry[i,:3] = shift[0], shift[0]
        # print(template.shape, pnts.shape)
        moved_pnts_lst.append(pnts_lst[i][:,:3] + shift)
        moved_temp_lst.append(template[:,:3] + shift)
        # print(shift, template.shape, shift.shape, np.mean((template + shift), axis=0))
    # print("xv",xv.shape, len(moved_pnts_lst), len(moved_temp_lst))
    moved_temp_pnt = np.concatenate(moved_temp_lst, axis=0)
    moved_pnts = np.concatenate(moved_pnts_lst, axis=0)
    tmp_section = len(moved_temp_pnt) // 200000 + 1
    pnt_section = len(moved_pnts) // 200000 + 1
    render_pnts_lst = [moved_temp_pnt[i*200000:(i+1)*200000] for i in range(tmp_section)] + [moved_pnts[i*200000:(i+1)*200000] for i in range(pnt_section)]
    colors_lst = [tuple(np.array(ImageColor.getcolor(colors[0], "RGB"))) for i in range(tmp_section)] + [tuple(np.array(ImageColor.getcolor(colors[1], "RGB")) / 255.0) for i in range(pnt_section)]
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


if __name__ == '__main__':

    ROOT_DIR = (Path(__file__).resolve().parent / '../../').resolve()
    print("ROOT_DIR", ROOT_DIR)
    path = ROOT_DIR / 'data' / 'kitti' / 'detection3d'
    cluster_num = 20
    all_db_infos_lst, box_dims_lst = extract_template(
        root_path=path
    )
    clusterer, indices = clustering("kmeans", cluster_num, box_dims_lst)
    # vis_cluster(clusterer, box_dims_lst[0], cluster_num)
    # vis_cluster_nonum(clusterer, box_dims_lst[0])
    # sample_template(all_db_infos_lst[0], indices, clusterer, cluster_num, clu_id=[1])
    sample_template(all_db_infos_lst[0], indices, cluster_num)





