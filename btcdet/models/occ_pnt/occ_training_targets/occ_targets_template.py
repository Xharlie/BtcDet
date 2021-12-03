import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from .vfe_template import VFETemplate
from ....utils import coords_utils, point_box_utils, common_utils
from functools import partial
import spconv
from ...backbones_3d.spconv_backbone import post_act_block as block

class OccTargetsTemplate(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, data_cfg, grid_size,
                 num_class, voxel_centers):
        super().__init__()
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.num_class = num_class

        self.nx, self.ny, self.nz = grid_size
        self.min_grid_tensor = torch.as_tensor([[0, 0, 0]], device="cuda", dtype=torch.int64)
        self.max_grid_tensor = torch.as_tensor([[self.nx-1, self.ny-1, self.nz-1]], device="cuda", dtype=torch.int64)
        self.nvx, self.nvy, self.nvz = voxel_size
        self.point_cloud_range = point_cloud_range
        self.det_point_cloud_range = data_cfg.POINT_CLOUD_RANGE
        # 1 X 3
        self.voxel_size = torch.as_tensor([voxel_size], dtype=torch.float32, device="cuda")
        self.point_origin_tensor = torch.as_tensor([point_cloud_range[:3]], dtype=torch.float32, device="cuda")
        self.point_max_tensor = torch.as_tensor([point_cloud_range[3:]], dtype=torch.float32, device="cuda")

        self.fix_conv_2dzy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.fix_conv_2dzy.weight.data.fill_(1.0)
        self.fix_conv_2dzy.requires_grad_(False)

        # self.fix_conv_3d = spconv.SparseSequential(
        #     block(1, 1, data_cfg.OCC.DIST_KERN, norm_fn=None, stride=1, padding=[dist // 2 for dist in data_cfg.OCC.DIST_KERN], indice_key='spconvfix', conv_type='fixspconv'))
        # self.fix_conv_3d.requires_grad_(False)
        self.all_voxel_centers = voxel_centers["all_voxel_centers"]
        self.all_voxel_centers_2d = voxel_centers["all_voxel_centers_2d"]

        if hasattr(self.data_cfg.OCC, 'SUPPORT_SPHERE_RANGE'):
            sphere_range = np.asarray(self.data_cfg.OCC.SUPPORT_SPHERE_RANGE)
            self.sphere_origin_tensor = torch.as_tensor([sphere_range[:3]], dtype=torch.float32, device="cuda")
            self.rever_sphere_origin_tensor = torch.as_tensor([sphere_range[2::-1]], dtype=torch.float32, device="cuda")
            self.sphere_max_tensor = torch.as_tensor([sphere_range[3:6]], dtype=torch.float32, device="cuda")
            if hasattr(self.data_cfg.OCC, 'SUPPORT_SPHERE_VOXEL_SIZE'):
                sphere_voxel_size = np.array([self.data_cfg.OCC.SUPPORT_SPHERE_VOXEL_SIZE[0], self.data_cfg.OCC.SUPPORT_SPHERE_VOXEL_SIZE[1], sphere_range[6]])
            else:
                sphere_voxel_size = np.array([voxel_size[0], voxel_size[1], sphere_range[6]])
            self.sphere_voxel_size = torch.as_tensor(sphere_voxel_size, dtype=torch.float32, device="cuda")
            self.reverse_sphere_voxel_size = torch.as_tensor([sphere_voxel_size[2], sphere_voxel_size[1], sphere_voxel_size[0]], dtype=torch.float32, device="cuda")
            sphere_grid_size = ((sphere_range[3:6] - sphere_range[:3]) / sphere_voxel_size).astype(np.int)
            self.sphere_nx, self.sphere_ny, self.sphere_nz = sphere_grid_size[0], sphere_grid_size[1], sphere_grid_size[2]
            self.sphere_min_grid_tensor = torch.as_tensor([[0, 0, 0]], device="cuda", dtype=torch.int64)
            self.sphere_max_grid_tensor = torch.as_tensor([[self.sphere_nx - 1, self.sphere_ny - 1, self.sphere_nz - 1]], device="cuda",  dtype=torch.int64)
        self.box_weight = data_cfg.OCC.BOX_WEIGHT
        self.occ_fore_cls_weight = self.model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS["occ_fore_cls_weight"]
        self.occ_mirr_cls_weight = self.model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS["occ_mirr_cls_weight"]
        self.occ_bm_cls_weight = self.model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS["occ_bm_cls_weight"]
        self.occ_neg_cls_weight = self.model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS["occ_neg_cls_weight"]
        self.fore_dropout_cls_weight = self.model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS["fore_dropout_cls_weight"]
        self.fore_dropout_reg_weight = self.model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS["fore_dropout_reg_weight"]

        self.occ_fore_res_weight = self.model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.get("occ_fore_res_weight", 0.1)
        self.occ_mirr_res_weight = self.model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.get("occ_mirr_res_weight", 0.1)
        self.occ_bm_res_weight = self.model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.get("occ_bm_res_weight", 0.1)

        self.reverse_vis = self.model_cfg.PARAMS.get("REVERSE_VIS", "NOTHING")
        self.concede_x = self.data_cfg.OCC.DIST_KERN[-1]//2 if self.data_cfg.OCC.get("HALF_X", False) else 0
        self.concede_x = self.data_cfg.OCC.get("CONCEDE_X", self.concede_x)
        self.point_coding = self.data_cfg.OCC.get("USE_ABSXYZ", "original")
        self.sphere_offset = torch.as_tensor([self.data_cfg.OCC.get("SPHERE_OFFSET", [0.0, 0.0, 0.0])], device="cuda", dtype=torch.float32)

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator


    def point2coords_inrange(self, points, point_origin_tensor, point_max_tensor, max_grid_tensor, min_grid_tensor, voxel_size):
        inrange_mask = torch.cat([points[:, :3] >= point_origin_tensor,
                                  points[:, :3] <= point_max_tensor], dim=-1).all(-1)
        inrange_inds = torch.nonzero(inrange_mask)[..., 0]
        points = points[inrange_inds, :]
        coords = ((points - point_origin_tensor) / voxel_size).to(torch.int64)
        coords = torch.minimum(coords, max_grid_tensor)
        coords = torch.maximum(coords, min_grid_tensor)
        return coords, inrange_inds

    def xyz2zyx(self, xyz):
        return torch.stack([xyz[..., 2], xyz[..., 1], xyz[..., 0]], dim=-1)

    def xyz2yxz(self, xyz):
        return torch.stack([xyz[..., 1], xyz[..., 0], xyz[..., 2]], dim=-1)

    def occ_from_ocp_2d(self, voxelwise_mask, empty_sur_thresh="None"):
        # if empty_sur_thresh != "None" and empty_sur_thresh < 9:
        #     occ_voxelwise_mask = voxelwise_mask
        #     occ_voxelwise_mask[:,:,:,0] = self.get_empty_mask(voxelwise_mask, empty_sur_thresh)
        # else:
        #     occ_voxelwise_mask = voxelwise_mask
        occ_voxelwise_mask = voxelwise_mask
        occ_voxelwise_mask_2d = torch.max(occ_voxelwise_mask, dim=1)[0]
        occ_voxelwise_mask_2d = torch.cumsum(occ_voxelwise_mask_2d, dim=2) > 0.9
        return occ_voxelwise_mask_2d.unsqueeze(1).repeat(1, self.nz, 1, 1)


    def occ_from_sphere_ocp(self, vcc_mask, voxelwise_mask, empty_sur_thresh="None"):
        if self.reverse_vis == "VCC":
            stride = self.data_cfg.OCC.DIST_KERN[2]+1
            inds = torch.nonzero(voxelwise_mask)
            # print("pre", inds[0:2])
            inds = inds.unsqueeze(1).repeat(1,stride // 2,1)
            inds[..., :, 3:4] -= torch.arange(1, stride // 2 + 1, device=inds.device).view(1, stride // 2, 1).repeat(inds.shape[0], 1, 1)
            inds[..., :, 3] = torch.clamp(inds[..., :, 3], min=0, max=None)
            inds = inds.view(-1, 4)
            # print("after", inds[0:10])
            occ_voxelwise_mask = torch.ones_like(voxelwise_mask, device=voxelwise_mask.device, dtype=voxelwise_mask.dtype)
            occ_voxelwise_mask[inds[..., 0], inds[..., 1], inds[..., 2], inds[..., 3]] = 0
            return occ_voxelwise_mask | voxelwise_mask
        elif self.reverse_vis == "BACK_TRACK":
            reverse_voxelwise_mask = torch.flip(voxelwise_mask, [3])
            occ_voxelwise_mask = torch.flip(torch.cumsum(reverse_voxelwise_mask, dim=3) < 0.9, [3])
            return occ_voxelwise_mask | (torch.cumsum(voxelwise_mask, dim=3) > 0.9)
        else:
            if empty_sur_thresh != "None" and empty_sur_thresh < 9:
                occ_voxelwise_mask = voxelwise_mask
                occ_voxelwise_mask[:,:,:,0] = self.get_empty_mask(voxelwise_mask, empty_sur_thresh)
            else:
                occ_voxelwise_mask = voxelwise_mask
            occ_voxelwise_mask = torch.cumsum(occ_voxelwise_mask, dim=3)
            return occ_voxelwise_mask > 0.9

    def occ_from_cylin_ocp(self, vcc_mask, batch_dict, bs, voxelwise_mask, occ_pnts, occ_b, empty_sur_thresh="None"):
        occ_sphere_pnts = coords_utils.cartesian_sphere_coords(occ_pnts + self.sphere_offset)  # M, spherexyz
        if "rot_z" in batch_dict:
            rot_z = batch_dict["rot_z"][occ_b]
            occ_sphere_pnts[..., 1] += rot_z
        sphere_voxelwise_map = torch.zeros([bs, self.sphere_nz, self.sphere_ny, self.sphere_nx], dtype=torch.uint8, device="cuda")
        inrange_coords, inrange_inds = self.point2coords_inrange(occ_sphere_pnts, self.sphere_origin_tensor, self.sphere_max_tensor, self.sphere_max_grid_tensor, self.sphere_min_grid_tensor, self.sphere_voxel_size)
        inrange_occ_b = occ_b[inrange_inds]
        sphere_voxelwise_map[inrange_occ_b, inrange_coords[..., 2], inrange_coords[..., 1], inrange_coords[..., 0]] = torch.ones_like(inrange_occ_b, dtype=torch.uint8, device=sphere_voxelwise_map.device)
        sphere_voxelwise_ind = torch.nonzero(self.occ_from_sphere_ocp(vcc_mask, sphere_voxelwise_map, empty_sur_thresh=empty_sur_thresh))  # M nz ny nx
        occ_sphere_b = sphere_voxelwise_ind[..., 0]
        occ_sphere_pnts = sphere_voxelwise_ind[..., 1:] * self.reverse_sphere_voxel_size + self.rever_sphere_origin_tensor
        occ_carte_pnts = coords_utils.sphere_uvd2absxyz(occ_sphere_pnts[..., 2], occ_sphere_pnts[..., 1], occ_sphere_pnts[..., 0])  # M 3(xyz)
        occ_cylin_pnts = coords_utils.cartesian_cylinder_coords(occ_carte_pnts - self.sphere_offset)  # M 3(xyz)
        inrange_coords_cylin, inrange_inds_cylin = self.point2coords_inrange(occ_cylin_pnts, self.point_origin_tensor, self.point_max_tensor, self.max_grid_tensor, self.min_grid_tensor, self.voxel_size)
        inrange_b_cylin = occ_sphere_b[inrange_inds_cylin]
        # print("inrange_inds_cylin", inrange_inds_cylin.shape, occ_cylin_pnts.shape)
        occ_voxelwise_mask = torch.zeros_like(voxelwise_mask, device=voxelwise_mask.device)
        occ_voxelwise_mask[inrange_b_cylin, inrange_coords_cylin[..., 2], inrange_coords_cylin[..., 1], inrange_coords_cylin[..., 0]] = torch.ones_like(inrange_b_cylin, dtype=torch.uint8, device=sphere_voxelwise_map.device)
        return occ_voxelwise_mask > 0.9


    def occ_from_carte_ocp(self, vcc_mask, batch_dict, bs, voxelwise_mask, occ_pnts, occ_b, empty_sur_thresh="None"):
        occ_sphere_pnts = coords_utils.cartesian_sphere_coords(occ_pnts + self.sphere_offset)  # M, spherexyz
        if "rot_z" in batch_dict:
            rot_z = batch_dict["rot_z"][occ_b]
            occ_sphere_pnts[..., 1] += rot_z
        sphere_voxelwise_map = torch.zeros([bs, self.sphere_nz, self.sphere_ny, self.sphere_nx], dtype=torch.uint8, device="cuda")
        inrange_coords, inrange_inds = self.point2coords_inrange(occ_sphere_pnts, self.sphere_origin_tensor, self.sphere_max_tensor, self.sphere_max_grid_tensor, self.sphere_min_grid_tensor, self.sphere_voxel_size)
        inrange_occ_b = occ_b[inrange_inds]
        sphere_voxelwise_map[inrange_occ_b, inrange_coords[..., 2], inrange_coords[..., 1], inrange_coords[..., 0]] = torch.ones_like(inrange_occ_b, dtype=torch.uint8, device=sphere_voxelwise_map.device)
        sphere_voxelwise_ind = torch.nonzero(self.occ_from_sphere_ocp(vcc_mask, sphere_voxelwise_map, empty_sur_thresh=empty_sur_thresh))  # M nz ny nx
        occ_sphere_b = sphere_voxelwise_ind[..., 0]
        occ_sphere_pnts = sphere_voxelwise_ind[..., 1:] * self.reverse_sphere_voxel_size + self.rever_sphere_origin_tensor
        occ_carte_pnts = coords_utils.sphere_uvd2absxyz(occ_sphere_pnts[..., 2], occ_sphere_pnts[..., 1], occ_sphere_pnts[..., 0]) - self.sphere_offset # M 3(xyz)
        inrange_coords_carte, inrange_inds_carte = self.point2coords_inrange(occ_carte_pnts, self.point_origin_tensor, self.point_max_tensor, self.max_grid_tensor, self.min_grid_tensor, self.voxel_size)
        inrange_b_carte = occ_sphere_b[inrange_inds_carte]
        occ_voxelwise_mask = torch.zeros_like(voxelwise_mask, device=voxelwise_mask.device)
        occ_voxelwise_mask[inrange_b_carte, inrange_coords_carte[..., 2], inrange_coords_carte[..., 1], inrange_coords_carte[..., 0]] = torch.ones_like(inrange_b_carte, dtype=torch.uint8, device=sphere_voxelwise_map.device)
        return occ_voxelwise_mask > 0.9


    def occ_from_ocp(self, vcc_mask, batch_dict, bs, voxelwise_mask, occ_pnts, occ_b, empty_sur_thresh="None", type="sphere"):
        if type == "sphere":
            return self.occ_from_sphere_ocp(vcc_mask, voxelwise_mask, empty_sur_thresh=empty_sur_thresh)
        elif type == "cylinder":
            return self.occ_from_cylin_ocp(vcc_mask, batch_dict, bs, voxelwise_mask, occ_pnts, occ_b, empty_sur_thresh=empty_sur_thresh)
        elif type == "cartesian":
            return self.occ_from_carte_ocp(vcc_mask, batch_dict, bs, voxelwise_mask, occ_pnts, occ_b, empty_sur_thresh=empty_sur_thresh)

    def get_empty_mask(self, voxelwise_mask, surround_thresh=4):
        occ_2d_mask = torch.sum(voxelwise_mask, dim=3)
        empty_2d_mask = occ_2d_mask == 0
        neighbor_2d_mask = self.create_predict_area2d(occ_2d_mask.unsqueeze(1)) > surround_thresh
        # print("empty2dmask", neighbor_2d_mask.shape, empty_2d_mask.shape)
        return empty_2d_mask & neighbor_2d_mask.squeeze(1)


    def get_valid(self, valid_mask, voxel_coords, voxel_features):
        valid_inds = torch.nonzero(valid_mask)
        return voxel_coords[valid_inds[:, 0]].to(torch.int64), voxel_features[valid_inds[:, 0], valid_inds[:, 1]]


    def get_voxelwise_mask(self, valid_coords_bnznynx, bs):
        voxelwise_mask = torch.zeros([bs, self.nz, self.ny, self.nx], dtype=torch.uint8, device="cuda")
        voxelwise_mask[valid_coords_bnznynx[..., 0], valid_coords_bnznynx[..., 1], valid_coords_bnznynx[..., 2], valid_coords_bnznynx[..., 3]] = torch.ones_like(valid_coords_bnznynx[..., 0], dtype=torch.uint8, device=valid_coords_bnznynx.device)
        return voxelwise_mask


    def get_fore_mirr_voxelwise_mask(self, batch_dict, bs, valid_coords_bnznynx, valid_voxel_features, gt_boxes_num, gt_boxes):
        fore_voxelwise_mask, mirr_fore_voxelwise_mask = [torch.zeros([bs, self.nz, self.ny, self.nx], dtype=torch.uint8, device="cuda") for i in range(2)]
        fore_inds, mirr_inbox_point, mirr_binds = point_box_utils.torch_points_and_sym_in_box_3d_batch( valid_voxel_features[..., :3], valid_coords_bnznynx, gt_boxes, gt_boxes_num, bs, batch_dict['box_mirr_flag'])
        fore_coords = valid_coords_bnznynx[fore_inds]  # b zyx
        fore_voxelwise_mask[fore_coords[..., 0], fore_coords[..., 1], fore_coords[..., 2], fore_coords[..., 3]] = torch.ones_like(fore_coords[..., 0], dtype=torch.uint8, device=fore_voxelwise_mask.device)

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

            mirr_fore_voxelwise_mask[mirr_coords[..., 0], mirr_coords[..., 1], mirr_coords[..., 2], mirr_coords[..., 3]] = torch.ones_like(mirr_coords[..., 0], dtype=torch.uint8, device=mirr_fore_voxelwise_mask.device)

        return fore_voxelwise_mask, mirr_fore_voxelwise_mask


    def get_bm_voxelwise_mask(self, batch_dict, bs, gt_boxes_num, gt_boxes):

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
            bm_voxelwise_mask[bm_coords[..., 0], bm_coords[..., 1], bm_coords[..., 2], bm_coords[..., 3]] = torch.ones_like(bm_coords[..., 0], dtype=torch.uint8, device=bm_voxelwise_mask.device)  ##
        return bm_voxelwise_mask

    def filter_occ(self, occ_voxelwise_mask, occ_pnts, voxelwise_mask):
        B, Z, Y, X = list(voxelwise_mask.shape)
        voxelwise_mask_z = (1-voxelwise_mask) * 100.0 + self.all_voxel_centers[..., 2].unsqueeze(0)
        voxelwise_mask_z = torch.min(voxelwise_mask_z.view(B, Z*Y, X), dim=1, keepdim=True)[0].unsqueeze(1)
        # print("voxelwise_mask_z", voxelwise_mask_z.shape, voxelwise_mask_z, torch.min(self.all_voxel_centers[..., 2]))
        voxelwise_mask_z -= (voxelwise_mask_z > 20.0) * 200
        return occ_voxelwise_mask & (self.all_voxel_centers[..., 2].unsqueeze(0) > torch.clamp(voxelwise_mask_z, min=self.det_point_cloud_range[2], max=None)) & (self.all_voxel_centers[..., 2].unsqueeze(0) < self.det_point_cloud_range[5])
        # return occ_voxelwise_mask & (self.all_voxel_centers[..., 2].unsqueeze(0) > max(-2.7, torch.topk(occ_pnts[..., 2].view(-1), 500, largest=False, sorted=True)[0][-1])) & (self.all_voxel_centers[..., 2].unsqueeze(0) < 1.0)

    def create_voxel_label(self, batch_dict, valid_mask):
        occ_pnts = torch.cat([coords_utils.uvd2absxyz(batch_dict['voxels'][..., 0], batch_dict['voxels'][..., 1], batch_dict['voxels'][..., 2], self.data_cfg.OCC.COORD_TYPE), batch_dict['voxels'][..., 3:]], dim=-1)
        if self.point_coding == "absxyz" or self.point_coding == True:
            batch_dict['voxels'] = occ_pnts
        elif self.point_coding == "both":
            batch_dict['voxels'] = torch.cat([occ_pnts[..., :3], batch_dict["voxels"]], dim=-1)
        voxel_features, voxel_coords, gt_boxes_num, gt_boxes, bs = occ_pnts, batch_dict['voxel_coords'], batch_dict["gt_boxes_num"], batch_dict["gt_boxes"], batch_dict["gt_boxes"].shape[0]
        if self.num_class == 1:
            gt_label = (gt_boxes[..., -1:] > 1e-2).to(torch.float32)
            gt_boxes = torch.cat([gt_boxes[..., :-1], gt_label], dim=-1)
        valid_coords_bnznynx, valid_voxel_features = self.get_valid(valid_mask, voxel_coords, voxel_features)
        voxelwise_mask = self.get_voxelwise_mask(valid_coords_bnznynx, bs)
        vcc_mask = self.create_predict_area3d(bs, valid_coords_bnznynx)
        occ_voxelwise_mask = self.filter_occ(self.occ_from_ocp(vcc_mask, batch_dict, bs, voxelwise_mask, valid_voxel_features[..., :3], valid_coords_bnznynx[..., 0], empty_sur_thresh=self.data_cfg.OCC.EMPT_SUR_THRESH, type=self.data_cfg.OCC.COORD_TYPE), occ_pnts, voxelwise_mask)
        fore_voxelwise_mask, mirr_fore_voxelwise_mask = self.get_fore_mirr_voxelwise_mask(batch_dict, bs, valid_coords_bnznynx, valid_voxel_features, gt_boxes_num, gt_boxes)
        mirr_fore_voxelwise_mask = mirr_fore_voxelwise_mask  * (1 - voxelwise_mask)  # exclude original occupied

        if self.model_cfg.TARGETS.TMPLT:
            bm_voxelwise_mask = self.get_bm_voxelwise_mask(batch_dict, bs, gt_boxes_num, gt_boxes)  * (1 - voxelwise_mask) * (1 - mirr_fore_voxelwise_mask)
        else:
            bm_voxelwise_mask = torch.zeros_like(voxelwise_mask, dtype=voxelwise_mask.dtype, device=voxelwise_mask.device)
        # print("grid_size", self.nz, self.ny, self.nx)
        # print("voxelwise_mask", voxelwise_mask.shape)
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
                    all_voxel_centers_filtered = self.all_voxel_centers[:, voxel_box_label2d[:, 0], voxel_box_label2d[:, 1], ...].reshape(-1, 3)
                    if "rot_z" in batch_dict:
                        all_voxel_centers_filtered = point_box_utils.rotatez(all_voxel_centers_filtered, batch_dict["rot_z"][i])
                    voxel_box_label = point_box_utils.torch_points_in_box_3d_label(all_voxel_centers_filtered, cur_gt_boxes, gt_boxes_num[i], shift=shift)[0]
                    forebox_label[i, :, voxel_box_label2d[:, 0], voxel_box_label2d[:, 1]] = voxel_box_label.view(self.nz, -1)
        ############## dropout #################
        if self.data_cfg.OCC.DROPOUT_RATE > 1e-3 and batch_dict["is_train"]:
            batch_dict = self.dropout(batch_dict, fore_voxelwise_mask)

        batch_dict = self.prepare_cls_loss_map(batch_dict, vcc_mask, voxelwise_mask, occ_voxelwise_mask, fore_voxelwise_mask, mirr_fore_voxelwise_mask, bm_voxelwise_mask, forebox_label=forebox_label)
        return batch_dict


    def dropout(self, batch_dict, fore_voxelwise_mask):
        bs = batch_dict["batch_size"]
        rand_ratios = np.random.uniform(high=self.data_cfg.OCC.DROPOUT_RATE, size=bs)
        drop_inds_lst = []
        for i in range(bs):
            bmask = batch_dict['voxel_coords'][..., 0] == i
            binds = torch.nonzero(bmask)[...,0]
            drop_vox_ind = torch.randint(low=0, high=len(binds), size=[int(len(binds) * rand_ratios[i])])
            drop_inds_lst.append(binds[drop_vox_ind])
        drop_voxel_inds = torch.cat(drop_inds_lst, dim=0) if len(drop_inds_lst) > 1 else drop_inds_lst[0]
        drop_voxel_coords = batch_dict['voxel_coords'][drop_voxel_inds].long()
        voxel_drop_mask = torch.zeros([bs, self.nz, self.ny, self.nx], dtype=torch.uint8, device="cuda")
        voxel_drop_mask[drop_voxel_coords[...,0], drop_voxel_coords[...,1], drop_voxel_coords[...,2], drop_voxel_coords[...,3]] = torch.ones(len(drop_voxel_coords), dtype=torch.uint8, device="cuda")
        batch_dict["voxel_drop_mask"] = voxel_drop_mask
        batch_dict["fore_voxel_drop_mask"] = fore_voxelwise_mask & voxel_drop_mask
        if self.data_cfg.OCC.get("DROPOUT_RMV", False):
            keep_mask = torch.ones([len(batch_dict['voxel_coords'])], dtype=torch.bool, device="cuda")
            keep_mask[drop_voxel_inds] = torch.zeros_like(keep_mask[drop_voxel_inds], dtype=torch.bool, device="cuda")
            batch_dict['voxels'] = batch_dict['voxels'][keep_mask, :, :]
            batch_dict['voxel_coords'] = batch_dict['voxel_coords'][keep_mask, ...]
            batch_dict['voxel_num_points'] = batch_dict['voxel_num_points'][keep_mask]
        else:
            batch_dict['voxels'][drop_voxel_inds] = torch.zeros_like(batch_dict['voxels'][drop_voxel_inds], device=batch_dict['voxels'].device, dtype=batch_dict['voxels'].dtype)
        return batch_dict

    def prepare_cls_loss_map(self, batch_dict, vcc_mask, voxelwise_mask, occ_voxelwise_mask, fore_voxelwise_mask, mirr_fore_voxelwise_mask, bm_voxelwise_mask, forebox_label=None):

        ##### create cls loss mask #####
        general_cls_loss_mask = vcc_mask & occ_voxelwise_mask
        occ_fore_cls_mask = fore_voxelwise_mask & general_cls_loss_mask
        occ_mirr_cls_mask = mirr_fore_voxelwise_mask & general_cls_loss_mask
        occ_bm_cls_mask = bm_voxelwise_mask & general_cls_loss_mask
        pos_mask = occ_fore_cls_mask | occ_mirr_cls_mask | occ_bm_cls_mask
        neg_mask = general_cls_loss_mask & (1 - pos_mask)

        general_cls_loss_mask_float = occ_fore_cls_mask.to(torch.float32) * self.occ_fore_cls_weight + occ_mirr_cls_mask.to(torch.float32) * self.occ_mirr_cls_weight + occ_bm_cls_mask.to(torch.float32) * self.occ_bm_cls_weight + neg_mask.to(torch.float32) * self.occ_neg_cls_weight

        if self.data_cfg.OCC.DROPOUT_RATE > 1e-3 and self.fore_dropout_cls_weight > 1e-4 and batch_dict["is_train"]:
            general_cls_loss_mask_float += (general_cls_loss_mask & batch_dict["fore_voxel_drop_mask"]).to(torch.float32) * self.fore_dropout_cls_weight
              
        if forebox_label is not None:
            box_neg_mask = neg_mask & (forebox_label > 1e-3)
            box_neg_mask_float = box_neg_mask.to(torch.float32) * (self.box_weight - self.occ_neg_cls_weight)
            general_cls_loss_mask_float += box_neg_mask_float
        ######### create cls label ########

        # print("neg_mask",neg_mask.shape)
        # print("pos_mask",pos_mask.shape)
        # print("general_cls_loss_mask",general_cls_loss_mask.shape)
        # print("general_cls_loss_mask_float",general_cls_loss_mask_float.shape)
        # print("occ_fore_cls_mask",occ_fore_cls_mask.shape)
        # print("occ_mirr_cls_mask",occ_mirr_cls_mask.shape)
        # print("occ_bm_cls_mask",occ_bm_cls_mask.shape)
        # print("forebox_label",forebox_label.shape)

        batch_dict["occ_fore_cls_mask"] = occ_fore_cls_mask
        batch_dict["occ_mirr_cls_mask"] = occ_mirr_cls_mask
        batch_dict["occ_bm_cls_mask"] = occ_bm_cls_mask
        batch_dict["forebox_label"] = forebox_label
        # batch_dict["positive_voxelwise_labels"] = pos_mask.to(torch.int64)
        # batch_dict["neg_mask"] = neg_mask
        if not batch_dict["is_train"]:
            batch_dict["neg_mask"] = neg_mask

        batch_dict.update({
            "vcc_mask": vcc_mask,
            "voxelwise_mask": voxelwise_mask,
            "bm_voxelwise_mask": bm_voxelwise_mask,
            "occ_voxelwise_mask": occ_voxelwise_mask,
            "fore_voxelwise_mask": fore_voxelwise_mask,
            "pos_mask": pos_mask,
            "pos_all_num": torch.sum(fore_voxelwise_mask | mirr_fore_voxelwise_mask | bm_voxelwise_mask),
            "general_cls_loss_mask_float": general_cls_loss_mask_float,
            "general_cls_loss_mask": general_cls_loss_mask,
        })
        return batch_dict


    def prepare_reg_loss_map(self, batch_dict, fore_res_mtrx, mirr_res_mtrx, bm_res_mtrx):
        ##### create reg loss mask #####
        general_reg_loss_mask_float = batch_dict["occ_fore_cls_mask"].to(
            torch.float32) * self.occ_fore_res_weight + batch_dict["occ_mirr_cls_mask"].to(
            torch.float32) * self.occ_mirr_res_weight + batch_dict["occ_bm_cls_mask"].to(
            torch.float32) * self.occ_bm_res_weight
        general_reg_loss_mask = (general_reg_loss_mask_float > 0).to(torch.uint8)

        if self.data_cfg.OCC.DROPOUT_RATE > 1e-3 and self.fore_dropout_reg_weight > 1e-4 and batch_dict["is_train"]:
            general_reg_loss_mask_float += (general_reg_loss_mask & batch_dict["fore_voxel_drop_mask"]).to(torch.float32) * self.fore_dropout_reg_weight


        res_mtrx = fore_res_mtrx * general_reg_loss_mask.unsqueeze(1) + mirr_res_mtrx * general_reg_loss_mask.unsqueeze(1) + bm_res_mtrx * general_reg_loss_mask.unsqueeze(1)
        batch_dict.update({
            "res_mtrx": res_mtrx,
            "general_reg_loss_mask": general_reg_loss_mask,
            "general_reg_loss_mask_float": general_reg_loss_mask_float,
        })
        return batch_dict


    def create_predict_area2d(self, bevcount_mask):
        bevcount_mask = bevcount_mask.to(torch.float32)
        point_dist_mask = self.fix_conv_2dzy(bevcount_mask)
        return point_dist_mask


    # def create_predict_area3d(self, batch_size, voxel_coords):
    #     # vfe_features: (num_voxels, C)
    #     # voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
    #     voxel_one = torch.ones([list(voxel_coords.shape)[0], 1], device=voxel_coords.device, dtype=torch.float32)
    #     sparse_shape = [self.nz, self.ny, self.nx]
    #     input_sp_tensor = spconv.SparseConvTensor(
    #         features=voxel_one,
    #         indices=voxel_coords.int(),
    #         spatial_shape=sparse_shape,
    #         batch_size=batch_size
    #     )
    #     indices_map = self.fix_conv_3d(input_sp_tensor).dense().squeeze(1) > 1e-3 # [N, C, Z, H, W]
    #     N, Z, H, W = list(indices_map.shape)
    #     # print("N, Z, H, W", N, Z, H, W) # 181 926 190
    #     indices_map = indices_map.view(N, self.nz, self.ny, self.nx)
    #     if self.concede_x != 0:
    #         real_indices_map = torch.zeros_like(indices_map, dtype=indices_map.dtype)
    #         real_indices_map[:,:,:,self.concede_x:] = indices_map[:,:,:,:-self.concede_x]
    #         return real_indices_map.to(torch.uint8)
    #     else:
    #         return indices_map.to(torch.uint8)

    def create_predict_area3d(self, bs, voxel_coords):
        # vfe_features: (num_voxels, C)
        # voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        voxel_coords = voxel_coords.view(-1, 1, 4)
        nz, ny, nx = self.data_cfg.OCC.DIST_KERN
        startz, starty, startx = -(nz//2), -(ny//2), -(nx//2)
        startx += self.concede_x
        x_ind = torch.arange(startx, startx+nx, device="cuda")
        y_ind = torch.arange(starty, starty+ny, device="cuda")
        z_ind = torch.arange(startz, startz+nz, device="cuda")
        z, y, x = torch.meshgrid(z_ind, y_ind, x_ind)
        bzyx = torch.stack([torch.zeros_like(z, device=z.device, dtype=z.dtype), z, y, x], axis=-1).view(1, -1, 4)
        voxel_coords = (voxel_coords + bzyx).view(-1, 4)
        vcc_mask = torch.zeros([bs, self.nz, self.ny, self.nx], dtype=torch.uint8, device="cuda")
        vcc_mask[torch.clamp(voxel_coords[..., 0], min=0, max=bs-1), torch.clamp(voxel_coords[..., 1], min=0, max=self.nz-1), torch.clamp(voxel_coords[..., 2], min=0, max=self.ny-1), torch.clamp(voxel_coords[..., 3], min=0, max=self.nx-1)] = 1
        return vcc_mask

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
        batch_dict = self.create_voxel_res_label(batch_dict, mask)
        # if test inference speed
        # if batch_dict["is_train"]:
        #     batch_dict = self.create_voxel_res_label(batch_dict, mask)
        # else:
        #     batch_dict["point_dist_mask"] = torch.zeros((batch_dict["gt_boxes"].shape[0], self.ny, self.nx, self.nz * self.sz * self.sy * self.sx), device="cuda")

        return batch_dict
