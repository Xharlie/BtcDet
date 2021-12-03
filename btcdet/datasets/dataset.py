from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.occ_point_cloud_range = self.point_cloud_range if dataset_cfg.get('OCC', None) is None else np.array(self.dataset_cfg.OCC.POINT_CLOUD_RANGE, dtype=np.float32)

        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training or self.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST[1].NAME in ["add_best_match","add_multi_best_match"] else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.occ_point_cloud_range, training=self.training, occ_config=dataset_cfg.get('OCC', None), det_point_cloud_range=self.point_cloud_range)
        self.occ_dim = self.data_processor.occ_dim
        self.det_grid_size = getattr(self.data_processor, 'det_grid_size', None)
        self.det_voxel_size = getattr(self.data_processor, 'det_voxel_size', None)

        self.occ_grid_size = getattr(self.data_processor, 'occ_grid_size', None)
        self.occ_voxel_size = getattr(self.data_processor, 'occ_voxel_size', None)
        self.min_points_in_box = self.dataset_cfg.get("MIN_POINTS_IN_BOX", 0)
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if self.dataset_cfg.get("SKIP_NOBOX", None) is None or self.dataset_cfg.SKIP_NOBOX:
                if len(data_dict['gt_boxes']) == 0:
                    new_index = np.random.randint(self.__len__())
                    return self.__getitem__(new_index)
        elif self.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST[1].NAME == "add_best_match" or self.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST[1].NAME == "add_multi_best_match":
            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                },
                validation=True
            )

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            if data_dict['gt_boxes'].ndim == 1:
                print("!!!!!!!!!!!!!", data_dict['gt_boxes'].shape, data_dict["frame_id"])
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            

            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        data_dict['box_mirr_flag'] = data_dict['gt_names'] != "Pedestrian" if len(data_dict['gt_names']) > 0 else np.array([])
        data_dict.pop('gt_names', None)
        data_dict['is_train'] = self.training
        if "augment_box_num" not in data_dict:
            data_dict["augment_box_num"] = 0
        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
                if key == "voxel_num_points":
                    data_dict["batch_voxel_num"].append(val.shape[0])
                if key == "det_voxel_num_points":
                    data_dict["batch_det_voxel_num"].append(val.shape[0])
        batch_size = len(batch_list)
        data_dict.pop('aug_boxes_image_idx', None)
        data_dict.pop('aug_boxes_gt_idx', None)
        data_dict.pop('aug_boxes_obj_ids', None)
        data_dict.pop('obj_ids', None)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points', 'voxel_points_label', 'det_voxels', 'det_voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords', 'det_voxel_coords', 'bm_points']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    ret["gt_boxes_num"] = []
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                        ret["gt_boxes_num"].append(val[k].__len__())
                    ret[key] = batch_gt_boxes3d
                elif key in ['coverage_rates']:
                    max_gt = max([len(x) for x in val])
                    batch_coverage_rates = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_coverage_rates[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_coverage_rates
                elif key in ['box_mirr_flag']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_box_mirr_flag = np.zeros((batch_size, max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_box_mirr_flag[k, :val[k].__len__()] = val[k].astype(np.float32)
                    ret[key] = batch_gt_box_mirr_flag
                elif key in ["miss_points", "self_points", "other_points", "miss_occ_points", "self_occ_points", "other_occ_points"]:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        ret['is_train'] = ret['is_train'][0]
        return ret
