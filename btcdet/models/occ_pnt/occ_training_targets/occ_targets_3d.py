import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .occ_targets_template import OccTargetsTemplate
from ....utils import coords_utils, point_box_utils

class OccTargets3D(OccTargetsTemplate):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, data_cfg, grid_size,
                 num_class, voxel_centers):
        super().__init__(model_cfg, voxel_size, point_cloud_range, data_cfg, grid_size,
                 num_class, voxel_centers)
        self.reg = model_cfg.PARAMS.get("REG", False)

    def create_predict_area(self, voxel_bnysynxsxnzsz, voxel_num_points_float, batch_size, batch_dict):
        return self.create_predict_area2d(voxel_bnysynxsxnzsz, voxel_num_points_float, batch_size, batch_dict)

    def forward(self, batch_dict, **kwargs):

        # voxels: [M, max_points, ndim] float tensor. only contain points.
        # voxel_coords: [M, 3] int32 tensor. zyx format.
        # voxel_num_points: [M] int32 tensor.

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # print("voxel_features", voxel_features.shape)

        voxel_count = voxel_features.shape[1]
        # print("voxel_count", voxel_features.shape[0])
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        batch_dict["voxel_point_mask"] = mask
        batch_dict = self.create_voxel_res_label(batch_dict, mask) if self.reg else self.create_voxel_label(batch_dict, mask)
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
    
    def create_voxel_res_label(self, batch_dict, valid_mask):
        occ_pnts = torch.cat([coords_utils.uvd2absxyz(batch_dict['voxels'][..., 0], batch_dict['voxels'][..., 1], batch_dict['voxels'][..., 2], self.data_cfg.OCC.COORD_TYPE), batch_dict['voxels'][..., 3:]], dim=-1)
        if self.point_coding == "absxyz" or self.point_coding == True:
            batch_dict['voxels'] = occ_pnts
        elif self.point_coding == "both":
            batch_dict['voxels'] = torch.cat([occ_pnts[..., :3], batch_dict["voxels"]], dim=-1)
        voxel_features, voxel_coords, gt_boxes_num, gt_boxes, bs = occ_pnts, batch_dict['voxel_coords'], batch_dict[
            "gt_boxes_num"], batch_dict["gt_boxes"], batch_dict["gt_boxes"].shape[0]
        if self.num_class == 1:
            gt_label = (gt_boxes[..., -1:] > 1e-2).to(torch.float32)
            gt_boxes = torch.cat([gt_boxes[..., :-1], gt_label], dim=-1)
        valid_coords_bnznynx, valid_voxel_features = self.get_valid(valid_mask, voxel_coords, voxel_features)
        voxelwise_mask = self.get_voxelwise_mask(valid_coords_bnznynx, bs)
        vcc_mask = self.create_predict_area3d(bs, valid_coords_bnznynx)
        occ_voxelwise_mask = self.filter_occ(self.occ_from_ocp(vcc_mask, batch_dict, bs, voxelwise_mask, valid_voxel_features[..., :3], valid_coords_bnznynx[..., 0], empty_sur_thresh=self.data_cfg.OCC.EMPT_SUR_THRESH, type=self.data_cfg.OCC.COORD_TYPE), occ_pnts, voxelwise_mask)
        fore_voxelwise_mask, fore_res_mtrx, mirr_fore_voxelwise_mask, mirr_res_mtrx = self.get_fore_mirr_voxelwise_mask_res(batch_dict, bs, valid_coords_bnznynx, valid_voxel_features, gt_boxes_num, gt_boxes)
        mirr_fore_voxelwise_mask = mirr_fore_voxelwise_mask * (1 - voxelwise_mask)  # exclude original occupied
        mirr_res_mtrx = mirr_res_mtrx * (1 - voxelwise_mask).unsqueeze(1)

        if self.model_cfg.TARGETS.TMPLT:
            bm_voxelwise_mask, bm_res_mtrx = self.get_bm_voxelwise_mask_res(batch_dict, bs, gt_boxes_num, gt_boxes)
            bm_voxelwise_mask = bm_voxelwise_mask * (1 - voxelwise_mask) * (1 - mirr_fore_voxelwise_mask)
            bm_res_mtrx = bm_res_mtrx * (1 - voxelwise_mask).unsqueeze(1) * (1 - mirr_fore_voxelwise_mask).unsqueeze(1)
        else:
            bm_voxelwise_mask = torch.zeros_like(voxelwise_mask, dtype=voxelwise_mask.dtype,
                                                 device=voxelwise_mask.device)
        ##### forebox_label #####
        forebox_label = None
        if self.data_cfg.OCC.BOX_WEIGHT != 1.0:
            bs, max_num_box, box_c = list(gt_boxes.shape)
            forebox_label = torch.zeros([bs, self.nz, self.ny, self.nx], dtype=torch.int8, device="cuda")
            shift = torch.tensor(np.asarray([[0.0, 0.0, 0.0]]), device="cuda", dtype=torch.float32)
            for i in range(bs):
                cur_gt_boxes = gt_boxes[i, :gt_boxes_num[i]]
                all_voxel_centers_2d = point_box_utils.rotatez(self.all_voxel_centers_2d, batch_dict["rot_z"][i]) if "rot_z" in batch_dict else self.all_voxel_centers_2d
                voxel_box_label2d = point_box_utils.torch_points_in_box_2d_mask(all_voxel_centers_2d, cur_gt_boxes, shift=shift[..., :2]).view(self.ny, self.nx).nonzero()
                if voxel_box_label2d.shape[0] > 0:
                    all_voxel_centers_filtered = self.all_voxel_centers[:, voxel_box_label2d[:, 0],
                                                 voxel_box_label2d[:, 1], ...].reshape(-1, 3)
                    if "rot_z" in batch_dict:
                        all_voxel_centers_filtered = point_box_utils.rotatez(all_voxel_centers_filtered, batch_dict["rot_z"][i])
                    voxel_box_label = point_box_utils.torch_points_in_box_3d_label(all_voxel_centers_filtered, cur_gt_boxes, gt_boxes_num[i], shift=shift)[0]
                    forebox_label[i, :, voxel_box_label2d[:, 0], voxel_box_label2d[:, 1]] = voxel_box_label.view(self.nz, -1)
        if self.data_cfg.OCC.DROPOUT_RATE > 1e-3 and batch_dict["is_train"]:
            batch_dict = self.dropout(batch_dict, fore_voxelwise_mask)
        batch_dict = self.prepare_cls_loss_map(batch_dict, vcc_mask, voxelwise_mask, occ_voxelwise_mask, fore_voxelwise_mask, mirr_fore_voxelwise_mask, bm_voxelwise_mask, forebox_label=forebox_label)

        batch_dict = self.prepare_reg_loss_map(batch_dict, fore_res_mtrx, mirr_res_mtrx, bm_res_mtrx)

        return batch_dict


    def get_bm_voxelwise_mask_res(self, batch_dict, bs, gt_boxes_num, gt_boxes):

        bm_voxelwise_mask = torch.zeros([bs, self.nz, self.ny, self.nx], dtype=torch.uint8, device="cuda")
        if "bm_points" in batch_dict and len(batch_dict["bm_points"]) > 0:
            bm_binds, bm_carte_points = batch_dict["bm_points"][..., 0:1].to(torch.int64), batch_dict["bm_points"][...,1:]
            label_array = torch.nonzero(point_box_utils.torch_points_in_box_3d_label_batch(bm_carte_points, bm_binds, gt_boxes, gt_boxes_num, bs))[..., 0]
            bm_binds = bm_binds[..., 0][label_array]
            bm_carte_points = bm_carte_points[label_array, :]
            occ_coords_bm_points = coords_utils.cartesian_occ_coords(bm_carte_points, type=self.data_cfg.OCC.COORD_TYPE)
            if "rot_z" in batch_dict:
                rot_z = batch_dict["rot_z"][bm_binds]
                if  self.data_cfg.OCC.COORD_TYPE == "cartesian":
                    noise_rotation = -rot_z * np.pi / 180
                    occ_coords_bm_points = common_utils.rotate_points_along_z(occ_coords_bm_points.unsqueeze(1), noise_rotation).squeeze(1)
                else:
                    occ_coords_bm_points[..., 1] += rot_z
            inrange_coords_bm, inrange_inds_bm = self.point2coords_inrange(occ_coords_bm_points, self.point_origin_tensor, self.point_max_tensor, self.max_grid_tensor, self.min_grid_tensor, self.voxel_size)
            bm_coords = torch.cat([bm_binds[inrange_inds_bm].unsqueeze(-1), self.xyz2zyx(inrange_coords_bm)], dim=-1)
            bm_res_mtrx = self.get_mean_res(bm_carte_points[inrange_inds_bm], bm_coords, bs, self.nz, self.ny, self.nx, batch_dict, rot=True)
            bm_voxelwise_mask[bm_coords[..., 0], bm_coords[..., 1], bm_coords[..., 2], bm_coords[..., 3]] = torch.ones_like(bm_coords[..., 0], dtype=torch.uint8, device=bm_voxelwise_mask.device)  ##
        else:
            bm_res_mtrx = torch.zeros([bs, 3, self.nz, self.ny, self.nx], dtype=torch.float32, device="cuda")

        return bm_voxelwise_mask, bm_res_mtrx


    def get_mean_res(self, feat, coords, bs, nz, ny, nx, batch_dict, rot=False):
        xyz_spatial = torch.zeros([bs, 3, nz, ny, nx], dtype=torch.float32, device="cuda")
        if len(coords) > 0:
            uni_coords, inverse_indices, labels_count = torch.unique(coords, return_inverse=True, return_counts=True, dim=0)
            mean_xyz = torch.zeros([uni_coords.shape[0], 3], dtype=feat.dtype, device=feat.device).scatter_add_(0, inverse_indices.view(inverse_indices.size(0), 1).expand(-1, 3), feat[..., :3]) / labels_count.float().unsqueeze(1)
            # mean_xyz = torch_scatter.scatter_mean(feat[..., :3], inverse_indices, dim=0)
            mean_xyz -= self.get_voxel_center_xyz(uni_coords, batch_dict, rot=rot)
            xyz_spatial[uni_coords[..., 0], :, uni_coords[..., 1], uni_coords[..., 2], uni_coords[..., 3]] = mean_xyz
        return xyz_spatial


    def get_voxel_center_xyz(self, coords, batch_dict, rot=True):
        voxel_centers = (coords[:, [3, 2, 1]].float() + 0.5) * self.voxel_size + self.point_origin_tensor
        if self.data_cfg.OCC.COORD_TYPE == "cartesian":
            if "rot_z" in batch_dict and rot:
                rot_z = batch_dict["rot_z"][coords[:, 0]]
                noise_rotation = rot_z * np.pi / 180
                voxel_centers = common_utils.rotate_points_along_z(voxel_centers.unsqueeze(1), noise_rotation).squeeze(1)
        else:
            if "rot_z" in batch_dict and rot:
                rot_z = batch_dict["rot_z"][coords[:, 0]]
                voxel_centers[..., 1] -= rot_z
            voxel_centers = coords_utils.uvd2absxyz(voxel_centers[..., 0], voxel_centers[..., 1], voxel_centers[..., 2], self.data_cfg.OCC.COORD_TYPE)
        return voxel_centers


    def get_fore_mirr_voxelwise_mask_res(self, batch_dict, bs, valid_coords_bnznynx, valid_voxel_features, gt_boxes_num, gt_boxes):
        fore_voxelwise_mask, mirr_fore_voxelwise_mask = [torch.zeros([bs, self.nz, self.ny, self.nx], dtype=torch.uint8, device="cuda") for i in range(2)]
        fore_inds, mirr_inbox_point, mirr_binds = point_box_utils.torch_points_and_sym_in_box_3d_batch( valid_voxel_features[..., :3], valid_coords_bnznynx, gt_boxes, gt_boxes_num, bs, batch_dict['box_mirr_flag'])
        fore_coords = valid_coords_bnznynx[fore_inds]  # b zyx
        fore_voxelwise_mask[fore_coords[..., 0], fore_coords[..., 1], fore_coords[..., 2], fore_coords[..., 3]] = torch.ones_like(fore_coords[..., 0], dtype=torch.uint8, device=fore_voxelwise_mask.device)
        fore_res_mtrx = self.get_mean_res(valid_voxel_features[fore_inds], fore_coords, bs, self.nz, self.ny, self.nx, batch_dict, rot=True)
        mirr_res_mtrx = torch.zeros([bs, 3, self.nz, self.ny, self.nx], device=fore_voxelwise_mask.device, dtype=torch.float32)
        if mirr_inbox_point is not None:
            occ_coords_mirr_points = coords_utils.cartesian_occ_coords(mirr_inbox_point, type=self.data_cfg.OCC.COORD_TYPE)  # sphere x y z
            if "rot_z" in batch_dict:
                rot_z = batch_dict["rot_z"][mirr_binds]
                if self.data_cfg.OCC.COORD_TYPE == "cartesian":
                    noise_rotation = -rot_z * np.pi / 180
                    occ_coords_mirr_points = common_utils.rotate_points_along_z(occ_coords_mirr_points.unsqueeze(1), noise_rotation).squeeze(1)
                else:
                    occ_coords_mirr_points[..., 1] += rot_z

            inrange_coords_mirr, inrange_inds_mirr = self.point2coords_inrange(occ_coords_mirr_points, self.point_origin_tensor, self.point_max_tensor, self.max_grid_tensor, self.min_grid_tensor, self.voxel_size)
            mirr_coords = torch.cat([mirr_binds[inrange_inds_mirr].unsqueeze(-1), self.xyz2zyx(inrange_coords_mirr)], dim=-1)  # mirror sphere b z y x
            mirr_res_mtrx = self.get_mean_res(mirr_inbox_point[inrange_inds_mirr], mirr_coords, bs, self.nz, self.ny, self.nx, batch_dict, rot=True)

            mirr_fore_voxelwise_mask[mirr_coords[..., 0], mirr_coords[..., 1], mirr_coords[..., 2], mirr_coords[..., 3]] = torch.ones_like(mirr_coords[..., 0], dtype=torch.uint8, device=mirr_fore_voxelwise_mask.device)

        return fore_voxelwise_mask, fore_res_mtrx, mirr_fore_voxelwise_mask, mirr_res_mtrx