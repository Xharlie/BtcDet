import pickle
import sys
import numpy as np

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, point_box_utils


class BestMatchQuerier(object):
    def __init__(self, root_path, querier_cfg, class_names, db_infos, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.querier_cfg = querier_cfg
        self.logger = logger
        self.bmatch_infos = {}

        for bm_info_path in querier_cfg.BM_INFO_PATH:
            bm_info_path = self.root_path.resolve() / bm_info_path
            with open(str(bm_info_path), 'rb') as f:
                infos = pickle.load(f)
                self.bmatch_infos = infos

        self.db_infos = db_infos


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
        for idx in range(gt_boxes_num):
            gt_box = data_dict['gt_boxes'][idx]
            gt_name = data_dict['gt_names'][idx]
            # print("self.bmatch_infos[gt_names]", self.bmatch_infos[gt_names].keys())
            # print("gt_names", gt_names, gt_boxes_num, aug_boxes_num, data_dict["gt_boxes_inds"])
            if gt_name in self.class_names:
                info = self.bmatch_infos[gt_name][(image_idx, data_dict["gt_boxes_inds"][idx])]
                file_path = self.root_path / info['path']
                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                    [-1, self.querier_cfg.NUM_POINT_FEATURES])[:,:3]
                rverbm = point_box_utils.get_yaw_rotation(-info['box3d_lidar'][6])
                obj_points = np.einsum("nj,ij->ni", obj_points, rverbm)
                obj_points = self.mirror(obj_points)
                gtrotation = point_box_utils.get_yaw_rotation(gt_box[6])
                obj_points = np.einsum("nj,ij->ni", obj_points, gtrotation) + gt_box[:3]
                obj_points_list.append(obj_points)
            # else:
            #     print("found ", gt_name," skip")
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
            image_idx, idx = aug_boxes_image_idx[ind], aug_boxes_gt_idx[ind]
            info = self.bmatch_infos[aug_box_names[ind]][(image_idx, idx)]
            gt_box = aug_box[ind]
            file_path = self.root_path / info['path']
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.querier_cfg.NUM_POINT_FEATURES])[:,:3]
            rverbm = point_box_utils.get_yaw_rotation(-info['box3d_lidar'][6])
            obj_points = np.einsum("nj,ij->ni", obj_points, rverbm)
            obj_points = self.mirror(obj_points)
            gtrotation = point_box_utils.get_yaw_rotation(gt_box[6])
            obj_points = np.einsum("nj,ij->ni", obj_points, gtrotation) + gt_box[:3]
            obj_points_list.append(obj_points)
        data_dict['bm_points'].extend(obj_points_list)
        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        data_dict['bm_points'] = []
        data_dict = self.add_gtbox_best_match_points_to_scene(data_dict)
        if "aug_boxes_image_idx" in data_dict:
            data_dict = self.add_sampled_boxes_best_match_points_to_scene(data_dict)

        if len(data_dict['bm_points']) > 1:
            data_dict['bm_points'] = np.concatenate(data_dict['bm_points'], axis=0)
        elif len(data_dict['bm_points']) == 1:
            data_dict['bm_points'] = np.array(data_dict['bm_points']).reshape(-1, 3)
        else:
            data_dict['bm_points'] = np.zeros([0,3], dtype=np.float32)
        return data_dict
