import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .occ_targets_template import OccTargetsTemplate
from ....utils import coords_utils, point_box_utils

class OccTargetsPillar(OccTargetsTemplate):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, data_cfg, grid_size,
                 num_class, voxel_centers, finer_indx_range):
        super().__init__(model_cfg, voxel_size, point_cloud_range, data_cfg, grid_size,
                 num_class, voxel_centers, finer_indx_range)

    def create_predict_area(self, voxel_bnysynxsxnzsz, voxel_num_points_float, batch_size, batch_dict):
        return self.create_predict_area2d(voxel_bnysynxsxnzsz, voxel_num_points_float, batch_size, batch_dict)

    def forward(self, batch_dict, **kwargs):

        # voxels: [M, max_points, ndim] float tensor. only contain points.
        # voxel_coords: [M, 3] int32 tensor. zyx format.
        # voxel_num_points: [M] int32 tensor.

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']
        # print("voxel_features", voxel_features.shape)


        voxel_count = voxel_features.shape[1]
        # print("voxel_count", voxel_features.shape[0])
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        batch_dict["voxel_point_mask"] = mask
        batch_dict = self.create_voxel_res_label(batch_dict, mask)
        # if test inference speed
        # if batch_dict["is_train"]:
        #     batch_dict = self.create_voxel_res_label(batch_dict, mask)
        # else:
        #     batch_dict["point_dist_mask"] = torch.zeros((batch_dict["gt_boxes"].shape[0], self.ny, self.nx, self.nz * self.sz * self.sy * self.sx), device="cuda")
        if "point_drop_inds" in batch_dict.keys():
            inds = batch_dict["point_drop_inds"]
            mask[inds[:, 0], inds[:, 1]] = torch.zeros_like(inds[:, 0], dtype=torch.bool)
        batch_dict["final_point_mask"] = mask

        return batch_dict
