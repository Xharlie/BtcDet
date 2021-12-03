import pickle
import sys
import numpy as np

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, point_box_utils, coords_utils


class MltBestMatchQuerier(object):
    def __init__(self, root_path, querier_cfg, class_names, db_infos, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.querier_cfg = querier_cfg
        self.logger = logger
        self.bmatch_infos = {}

        self.mlt_bm_root = {
            "Car":self.root_path.resolve() / querier_cfg.CAR_MLT_BM_ROOT,
            "Cyclist":self.root_path.resolve() / querier_cfg.CYC_MLT_BM_ROOT,
            "Pedestrian":self.root_path.resolve() / querier_cfg.PED_MLT_BM_ROOT,
        }
        self.db_infos, self.vis = db_infos, False
        self.load_point_features = querier_cfg.get("LOAD_POINT_FEATURES", 3)
        self.add_bm_2_raw = querier_cfg.get("ADD_BM_2_RAW", False)
        if querier_cfg.get("ABLATION", None) is not None:
            self.rmv_self_occ = querier_cfg.ABLATION.get("RMV_SELF_OCC", False)
            self.rmv_miss = querier_cfg.ABLATION.get("RMV_MISS", False)
            self.num_point_features = querier_cfg.ABLATION.get("NUM_POINT_FEATURES", 4)
            self.vis = querier_cfg.ABLATION.get("VIS", False)
        # self.sphere_coords_res = np.asarray([[0.2, 0.0875*2, 0.4203125]])
        self.sphere_coords_res = np.asarray([[0.32, 0.5184, 0.4203125]])
        self.expand = False
        # self.expand = False

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d


    def __setstate__(self, d):
        self.__dict__.update(d)


    def mirror(self, pnts, lastchannel=3):
        mirror_pnts = np.concatenate([pnts[..., 0:1], -pnts[..., 1:2], pnts[..., 2:lastchannel]], axis=-1)
        return np.concatenate([pnts, mirror_pnts], axis=0)


    def add_gtbox_best_match_points_to_scene(self, data_dict):
        obj_points_list = []
        aug_boxes_num = data_dict['aug_boxes_image_idx'].shape[0] if 'aug_boxes_image_idx' in data_dict else 0
        gt_boxes_num = data_dict['gt_boxes'].shape[0] - aug_boxes_num
        image_idx = int(data_dict['frame_id'])
        assert  gt_boxes_num == data_dict["gt_boxes_inds"].shape[0]
        for idx in range(gt_boxes_num):
            gt_box = data_dict['gt_boxes'][idx]
            gt_name = data_dict['gt_names'][idx]
            # print("self.bmatch_infos[gt_names]", self.bmatch_infos[gt_names].keys())
            # print("gt_names", gt_names, gt_boxes_num, aug_boxes_num, data_dict["gt_boxes_inds"])
            if gt_name in self.class_names:
                gt_box_id = data_dict["gt_boxes_inds"][idx]
                file_path = self.mlt_bm_root[gt_name] / "{}_{}.pkl".format(image_idx, gt_box_id)
                with open(file_path, 'rb') as f:
                    obj_points = pickle.load(f)
                obj_points = obj_points.reshape(
                    [-1, self.load_point_features])[:,:3].astype(np.float32)
                gtrotation = point_box_utils.get_yaw_rotation(gt_box[6])
                obj_points = np.einsum("nj,ij->ni", obj_points, gtrotation) + gt_box[:3]
                obj_points_list.append(obj_points)
            # else:
            #     print("found ", gt_name," skip")
        if "bm_points" in data_dict:
            data_dict['bm_points'].extend(obj_points_list)
        else:
            data_dict['bm_points'] = obj_points_list
        return data_dict

    def add_sampled_boxes_best_match_points_to_scene(self, data_dict):
        aug_boxes_image_idx = data_dict['aug_boxes_image_idx']
        aug_length = aug_boxes_image_idx.shape[0]
        aug_boxes_gt_idx = data_dict['aug_boxes_gt_idx']
        aug_box = data_dict['gt_boxes'][-aug_length:]
        aug_box_names = data_dict['gt_names'][-aug_length:]
        obj_points_list = []
        for ind in range(aug_length):
            gt_box = aug_box[ind]
            gt_name = aug_box_names[ind]
            file_path = self.mlt_bm_root[gt_name] / "{}_{}.pkl".format(aug_boxes_image_idx[ind], aug_boxes_gt_idx[ind])
            with open(file_path, 'rb') as f:
                obj_points = pickle.load(f)
            obj_points = obj_points.reshape(
                [-1, self.load_point_features])[:, :3].astype(np.float32)
            gtrotation = point_box_utils.get_yaw_rotation(gt_box[6])
            obj_points = np.einsum("nj,ij->ni", obj_points, gtrotation) + gt_box[:3]
            obj_points_list.append(obj_points)
        data_dict['bm_points'].extend(obj_points_list)
        return data_dict


    def filter_bm(self, data_dict):
        gt_boxes_filtered = data_dict['gt_boxes'][data_dict['gt_boxes_mask']] if 'gt_boxes_mask' in data_dict else data_dict['gt_boxes']

        assert len(gt_boxes_filtered) == len(data_dict['bm_points']), "{}_{}".format(len(data_dict['gt_boxes']), len(data_dict['bm_points']))
        boxes_xy_dist = np.linalg.norm(gt_boxes_filtered[:,:2], axis=-1)
        box_inds = np.argsort(boxes_xy_dist)
        gt_boxes, bm_points_lst = gt_boxes_filtered[box_inds, :], [data_dict['bm_points'][box_inds[i]] for i in range(len(box_inds))]
        box_labels = np.arange(len(gt_boxes_filtered), dtype=np.float32).reshape(-1, 1) + 1
        boxes = np.concatenate([gt_boxes, box_labels], axis=-1)
        pointwise_box_label = np.round(point_box_utils.points_in_box_3d_label(data_dict["points"][:,:3], boxes, slack=1.0, shift=np.array([[0., 0., 0.15, 0., 0., 0.]]))).astype(np.int)

        _, raw_sphere_points, raw_sphere_coords, raw_expand_sphere_coords, voxelwise_mask, sphere_coords_min, nx, ny, nz, _ = self.get_coords(data_dict["points"], nx=None, ny=None, nz=None, sphere_coords_min=None)
        occ_mask = (np.cumsum(voxelwise_mask, axis=0) > 0.5).astype(np.uint8)
        raw_points_len = len(data_dict["points"])
        other_bm_points_lst, self_bm_points_lst, miss_bm_points_lst, coverage_rate_lst = [], [], [], []
        if self.vis:
            other_occluder_mask, miss_tot_mask, bm_tot_mask, other_tot_mask = np.zeros([nx, ny, nz], dtype=np.uint8),np.zeros([nx, ny, nz], dtype=np.uint8), np.zeros([nx, ny, nz], dtype=np.uint8), np.zeros([nx, ny, nz], dtype=np.uint8)

        for i in range(len(bm_points_lst)):
            obj_points = data_dict["points"][:raw_points_len,:][pointwise_box_label==(i+1), :]
            avg_feature = np.mean(obj_points[:, 3:self.num_point_features], axis=0, keepdims=True) if len(obj_points) > 0 else np.zeros([1, self.num_point_features-3])
            _, obj_sphere_points, obj_sphere_coords, obj_expand_sphere_coords, obj_voxelwise_mask, _, _, _, _, num_unique_obj = self.get_coords(obj_points, nx=nx, ny=ny, nz=nz, sphere_coords_min=sphere_coords_min, expand=False, x_expand=True)

            bm_points = np.concatenate([bm_points_lst[i], np.tile(avg_feature, (bm_points_lst[i].shape[0], 1))], axis=-1)

            bm_points, bm_sphere_points, bm_sphere_coords, bm_expand_sphere_coords, bm_voxelwise_mask, _, _, _, _, num_unique_bm = self.get_coords(bm_points, nx=nx, ny=ny, nz=nz, sphere_coords_min=sphere_coords_min, expand=False)
            coverage_rate_lst.append(num_unique_obj / max(1, num_unique_bm))

            keep_mask = 1 - obj_voxelwise_mask[bm_sphere_coords[..., 0], bm_sphere_coords[..., 1], bm_sphere_coords[..., 2]]
            rmv_miss_filter_mask = occ_mask[bm_sphere_coords[..., 0], bm_sphere_coords[..., 1], bm_sphere_coords[..., 2]]
            bm_occ_mask = (np.cumsum(bm_voxelwise_mask, axis=0) < 1.5).astype(np.uint8)
            rmv_self_filter_mask = bm_occ_mask[bm_sphere_coords[..., 0], bm_sphere_coords[..., 1], bm_sphere_coords[..., 2]]

            if self.vis:
                miss_mask = (keep_mask * rmv_self_filter_mask * (1 - rmv_miss_filter_mask)).astype(np.bool)
                self_mask = (keep_mask * (1-rmv_self_filter_mask)).astype(np.bool)
                other_mask = (keep_mask * rmv_miss_filter_mask * rmv_self_filter_mask).astype(np.bool)
                miss_sphere_coords, miss_points = self.get_nearest_points(miss_mask, bm_sphere_points, bm_sphere_coords, bm_points, axis=0)
                self_sphere_coords, self_points = self.get_nearest_points(self_mask, bm_sphere_points, bm_sphere_coords, bm_points, axis=0)
                other_sphere_coords, other_points = self.get_nearest_points(other_mask, bm_sphere_points, bm_sphere_coords, bm_points, axis=0)
                other_bm_points_lst.append(other_points), self_bm_points_lst.append(self_points), miss_bm_points_lst.append(miss_points)
                miss_tot_mask[miss_sphere_coords[..., 0], miss_sphere_coords[..., 1], miss_sphere_coords[..., 2]] = np.ones_like(miss_sphere_coords[..., 0], dtype=np.uint8)
                bm_tot_mask = np.maximum(bm_tot_mask, bm_voxelwise_mask)
                other_tot_mask[other_sphere_coords[..., 0], other_sphere_coords[..., 1], other_sphere_coords[..., 2]] = np.ones_like(other_sphere_coords[..., 0], dtype=np.uint8)

            if self.rmv_self_occ:
                keep_mask *= rmv_self_filter_mask
            if self.rmv_miss:
                keep_mask *= rmv_miss_filter_mask

            keep_mask = keep_mask.astype(np.bool)
            bm_sphere_coords, bm_points = self.get_nearest_points(keep_mask, bm_sphere_points, bm_sphere_coords, bm_points, axis=0)
            if self.expand:
                bm_expand_sphere_coords = self.expand_voxel(bm_sphere_coords, nx, ny, nz)
            else:
                bm_expand_sphere_coords = bm_sphere_coords

            ## update voxel mask and occ mask
            voxelwise_mask[bm_expand_sphere_coords[..., 0], bm_expand_sphere_coords[..., 1], bm_expand_sphere_coords[..., 2]] = np.ones_like(bm_expand_sphere_coords[..., 0], dtype=np.uint8)
            occ_mask = (np.cumsum(voxelwise_mask, axis=0) > 0.5).astype(np.uint8)

            if self.add_bm_2_raw:
                data_dict["points"] = np.concatenate([data_dict["points"], bm_points], axis=0)
        if self.vis:
            self_cum = np.cumsum(bm_tot_mask, axis=0)
            self_reverse_cum = np.flip(np.cumsum(np.flip(bm_tot_mask, axis=0), axis=0), axis=0)
            self_tot_occ_mask = (self_cum > 0.5).astype(np.uint8) # - bm_tot_mask.astype(np.uint8)
            self_limit_occ_mask = (self_reverse_cum > 0.5).astype(np.uint8) * self_tot_occ_mask

            other_occluder_mask[raw_sphere_coords[..., 0], raw_sphere_coords[..., 1], raw_sphere_coords[..., 2]] = np.ones_like(raw_sphere_coords[..., 0], dtype=np.uint8)

            other_full_tot_occ_mask = (np.cumsum(other_occluder_mask, axis=0) > 0.5).astype(np.uint8) * self.propagate_323(other_tot_mask)
            other_tot_occ_mask = other_full_tot_occ_mask * (1 - self_tot_occ_mask)
            miss_full_tot_occ_mask = self.propagate_323(miss_tot_mask)
            miss_tot_occ_mask = miss_full_tot_occ_mask * (1 - self_tot_occ_mask)

            data_dict.update({
                'miss_points': self.combine_lst(miss_bm_points_lst),
                'self_points': self.combine_lst(self_bm_points_lst),
                'other_points': self.combine_lst(other_bm_points_lst),
                'miss_occ_points': self.get_voxel_centers(miss_tot_occ_mask, sphere_coords_min, self.sphere_coords_res),
                'miss_full_occ_points': self.get_voxel_centers(miss_full_tot_occ_mask, sphere_coords_min, self.sphere_coords_res),
                'self_occ_points': self.get_voxel_centers(self_tot_occ_mask, sphere_coords_min, self.sphere_coords_res),
                'self_limit_occ_mask': self.get_voxel_centers(self_limit_occ_mask, sphere_coords_min, self.sphere_coords_res),
                'other_occ_points': self.get_voxel_centers(other_tot_occ_mask, sphere_coords_min, self.sphere_coords_res),
                'other_full_occ_points': self.get_voxel_centers(other_full_tot_occ_mask, sphere_coords_min, self.sphere_coords_res),
            })
        # if len(coverage_rate_lst) == 0:
        #     data_dict["coverage_rates"] = np.zeros([0,1], dtype=np.float32)
        # elif len(coverage_rate_lst) == 1:
        #     data_dict["coverage_rates"] = np.asarray(coverage_rate_lst[0]).reshape(-1, 1)
        # else:
        #     data_dict["coverage_rates"] = np.stack(coverage_rate_lst, axis=0)
        return data_dict


    def get_voxel_centers(self, mask, sphere_coords_min, sphere_coords_res):
        coords = np.stack(np.nonzero(mask), axis=-1)
        # print("coords", coords.shape, sphere_coords_min.shape, sphere_coords_res.shape)
        sphere_points = sphere_coords_min + (coords + 0.5) * sphere_coords_res
        return coords_utils.uvd2absxyz_np(sphere_points[..., 0], sphere_points[..., 1], sphere_points[..., 2], type="sphere")


    def propagate_323(self, mask_3d):
        mask_2d = np.max(mask_3d, axis=0, keepdims=True)
        mask_3d = np.tile(mask_2d, [mask_3d.shape[0], 1, 1])
        return mask_3d


    def exclude_map(self, occ_coords, x_epansion=2):
        x = np.arange(-x_epansion, 1)
        y, z = np.zeros_like(x), np.zeros_like(x)
        xyz_shift = np.stack([x, y, z], axis=-1).reshape(1, len(x), 3)

        expand_occ_coords = (np.expand_dims(occ_coords, axis=1) + xyz_shift).reshape(-1, 3)
        expand_occ_coords = np.maximum(np.array([[0, 0, 0]]), expand_occ_coords)
        return expand_occ_coords


    def expand_voxel(self, occ_coords, nx, ny, nz, yz_epansion=2):
        y_ind = np.arange(-1, yz_epansion)
        z_ind = np.arange(-1, yz_epansion)
        y, z = np.meshgrid(y_ind, z_ind)
        x = np.ones_like(y)
        xyz_shift = np.stack([x, y, z], axis=-1).reshape(1, 9, 3)
        xyz_shift[0, 4, 0] = 0
        # print("xyz_shift", y.shape, y, xyz_shift.shape, xyz_shift[0,0], xyz_shift[0,1], xyz_shift[0,2])
        expand_occ_coords = (np.expand_dims(occ_coords, axis=1) + xyz_shift).reshape(-1, 3)
        expand_occ_coords = np.minimum(np.maximum(np.array([[0,0,0]]), expand_occ_coords), np.array([[nx-1, ny-1, nz-1]]))

        return expand_occ_coords


    def get_nearest_points(self, keep_mask, bm_sphere_points, bm_sphere_coords, bm_points, axis=0):
        bm_sphere_coords = bm_sphere_coords[keep_mask, :]
        bm_sphere_points = bm_sphere_points[keep_mask, :]
        bm_points = bm_points[keep_mask, :]
        closeind = np.argsort(bm_sphere_points[..., 0])
        bm_points, bm_sphere_coords = bm_points[closeind, :], bm_sphere_coords[closeind, :]
        uniq_coords, indices = np.unique(bm_sphere_coords, axis=0, return_index=True)
        return bm_sphere_coords, bm_points[indices, :]


    def get_coords(self, points, nx=None, ny=None, nz=None, sphere_coords_min=None, expand=True, x_expand=False):
        raw_sphere_points = coords_utils.absxyz_2_spherexyz_np(points[..., :3])
        if sphere_coords_min is None:
            sphere_coords_min = np.min(raw_sphere_points, axis=0, keepdims=True) - np.asarray(
                [[0.2 * 10, 0.0875 * 2 * 10, 0.4203125 * 10]])
        raw_sphere_coords = np.floor_divide(raw_sphere_points - sphere_coords_min, self.sphere_coords_res).astype(np.int32)
        unique_occupied = 0
        if nx is None:
            nx, ny, nz = list(np.max(raw_sphere_coords, axis=0))
            nx, ny, nz = nx + 1 + 10, ny + 1 + 10, nz + 1 + 10
        else:
            keep_mask = np.all(raw_sphere_coords >= 0, axis=-1) & np.all(raw_sphere_coords < np.array([[nx, ny, nz]]), axis=-1)
            raw_sphere_points = raw_sphere_points[keep_mask, :]
            raw_sphere_coords = raw_sphere_coords[keep_mask, :]
            points = points[keep_mask, :]
            unique_occupied = len(np.unique(raw_sphere_coords, axis=0))
        # print("nx",nx,ny,nz)
        raw_expand_sphere_coords = self.expand_voxel(raw_sphere_coords, nx, ny, nz) if (expand and self.expand) else raw_sphere_coords
        raw_expand_sphere_coords = self.exclude_map(raw_sphere_coords) if x_expand else raw_expand_sphere_coords
        voxelwise_mask = np.zeros([nx, ny, nz], dtype=np.uint8)
        voxelwise_mask[raw_expand_sphere_coords[..., 0], raw_expand_sphere_coords[..., 1], raw_expand_sphere_coords[..., 2]] = np.ones_like(raw_expand_sphere_coords[..., 0], dtype=np.uint8)
        return points, raw_sphere_points, raw_sphere_coords, raw_expand_sphere_coords, voxelwise_mask, sphere_coords_min, nx, ny, nz, unique_occupied



    def combine_lst(self, bm_points_lst):
        if len(bm_points_lst) > 1:
            return np.concatenate(bm_points_lst, axis=0)[...,:3]
        elif len(bm_points_lst) == 1:
            return bm_points_lst[0][...,:3]
        else:
            return np.zeros([0,3], dtype=np.float32)


    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        # data_dict['bm_points'] = []
        data_dict = self.add_gtbox_best_match_points_to_scene(data_dict)
        if "aug_boxes_image_idx" in data_dict and 'pre_aug_bm' not in data_dict:
            data_dict = self.add_sampled_boxes_best_match_points_to_scene(data_dict)
        if self.querier_cfg.get("ABLATION", None) is not None and len(data_dict['bm_points']) > 0:
            data_dict = self.filter_bm(data_dict)
        data_dict['bm_points'] = self.combine_lst(data_dict['bm_points'])
        # print(data_dict['bm_points'].shape)

        return data_dict
