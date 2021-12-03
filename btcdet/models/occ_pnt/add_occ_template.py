import torch
import torch.nn as nn
from ...utils import coords_utils, vis_occ_utils, point_box_utils, common_utils
import numpy as np

class AddOccTemplate(nn.Module):
    def __init__(self, model_cfg, data_cfg, point_cloud_range, occ_voxel_size, occ_grid_size, det_voxel_size, det_grid_size, mode, voxel_centers, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.occ_thresh = model_cfg.PARAMS.OCC_THRESH
        self.eval_occ_thresh = model_cfg.PARAMS.EVAL_OCC_THRESH
        self.max_add_occpnts_num = model_cfg.PARAMS.MAX_NUM_OCC_PNTS
        self.eval_max_add_occpnts_num = model_cfg.PARAMS.EVAL_MAX_NUM_OCC_PNTS
        self.pass_gradient = model_cfg.OCC_PNT_UPDATE.PASS_GRAD
        self.res_num_dim = data_cfg.OCC.RES_NUM_DIM
        self.point_cloud_range = torch.tensor(point_cloud_range, device="cuda", dtype=torch.float32) # original pc range
        self.occ_voxel_size = occ_voxel_size
        self.nvx, self.nvy, self.nvz = occ_voxel_size
        self.occ_grid_size = occ_grid_size
        self.det_grid_size = det_grid_size
        self.det_voxel_size = torch.tensor(det_voxel_size, device="cuda", dtype=torch.float32)

        self.occ_point_cloud_range = data_cfg.OCC.POINT_CLOUD_RANGE
        self.occ_point_cloud_range_tensor = torch.tensor(self.occ_point_cloud_range, device="cuda").view(1, 6).contiguous()
        self.occ_x_origin = data_cfg.OCC.POINT_CLOUD_RANGE[0]
        self.occ_y_origin = data_cfg.OCC.POINT_CLOUD_RANGE[1]
        self.occ_z_origin = data_cfg.OCC.POINT_CLOUD_RANGE[2]

        self.all_voxel_centers = voxel_centers["all_voxel_centers"]
        print("self.all_voxel_centers", self.all_voxel_centers.shape)
        self.xrange = [point_cloud_range[0], point_cloud_range[3]]
        self.yrange = [point_cloud_range[1], point_cloud_range[4]]
        self.vis_r = 0.1
        self.bev_img_h = int(np.ceil((self.yrange[1] - self.yrange[0]) / self.vis_r)) + 1
        self.bev_img_w = int(np.ceil((self.xrange[1] - self.xrange[0]) / self.vis_r)) + 1
        self.config_realdrop = self.data_cfg.OCC.get('REAL_DROP', None) is None or self.data_cfg.OCC.REAL_DROP
        self.config_rawadd = self.data_cfg.OCC.get('RAW_ADD', False)
        self.code_num_dim = self.data_cfg.OCC.get('CODE_NUM_DIM', 2)
        self.reg = model_cfg.PARAMS.get("REG", False)
        self.db_proj = model_cfg.OCC_PNT_UPDATE.get('DB_PROJ', False)
        self.remain_percentage = model_cfg.PARAMS.get('REMAIN_PERCENTAGE', None)

    def db_proj_func(self, occ_pnts, occ_coords, occ_carte_coords, batch_dict, expand=[2,6,3], stride=[2,2,2]):
        # occ_carte_coords: M X 3 (z, y, x)
        occ_zyx_origin = torch.tensor([self.occ_z_origin, self.occ_y_origin, self.occ_x_origin], device="cuda")
        det_voxel_size_zyx = torch.tensor([self.det_voxel_size[2], self.det_voxel_size[1], self.det_voxel_size[0]], device="cuda", dtype=torch.float32)
        point_cloud_origin_zyx = torch.tensor([self.point_cloud_range[2], self.point_cloud_range[1], self.point_cloud_range[0]], device="cuda", dtype=torch.float32)  # original pc range
        occ_voxel_size_zyx = torch.tensor([self.nvz, self.nvy, self.nvx], device="cuda", dtype=torch.float32)
        min_det_grid_ind = torch.tensor([[0, 0, 0]], device="cuda", dtype=torch.int64)
        max_det_grid_ind = torch.tensor([[self.det_grid_size[2]-1, self.det_grid_size[1]-1, self.det_grid_size[0]-1]], device="cuda", dtype=torch.int64)
        expandz, expandy, expandx = expand
        z_ind = torch.arange(-expandz, expandz+1, stride[0], device="cuda")
        y_ind = torch.arange(-expandy, expandy+1, stride[1], device="cuda")
        x_ind = torch.arange(-expandx, expandx+1, stride[2], device="cuda")
        z, y, x = torch.meshgrid(z_ind, y_ind, x_ind)
        zyx = torch.stack([z, y, x], axis=-1).view(1, -1, 3)
        occ_carte_coords_aug = occ_carte_coords[..., 1:].unsqueeze(1) + zyx # N X 5X13X7 X 3
        occ_carte_loc = (0.5 + occ_carte_coords_aug.to(torch.float32)) * det_voxel_size_zyx.view(1, 1, 3) + point_cloud_origin_zyx.view(1, 1, 3)
        occ_coords_loc = coords_utils.cartesian_occ_coords(occ_carte_loc, self.data_cfg.OCC.COORD_TYPE, perm="zyx")
        if "rot_z" in batch_dict:
            rot_z = batch_dict["rot_z"][occ_coords[...,0]].unsqueeze(-1)
            if self.data_cfg.OCC.COORD_TYPE == "cartesian":
                noise_rotation = -rot_z * np.pi / 180
                occ_coords_loc = common_utils.rotate_points_along_z(occ_coords_loc.unsqueeze(1), noise_rotation).squeeze(1)
            else:
                occ_coords_loc[..., 1] += rot_z
        occ_coords_coords = (torch.floor((occ_coords_loc - occ_zyx_origin.view(1, 1, 3)) / occ_voxel_size_zyx)).to(torch.int64) # N X 5X13X7 X 3
        pick_masks = torch.all((occ_coords_coords - occ_coords[..., 1:].unsqueeze(1)) == 0, dim=-1, keepdims=True) # N X 5X13X7
        occ_exp_pnts = torch.cat([occ_carte_loc[..., 2:3], occ_carte_loc[..., 1:2], occ_carte_loc[..., 0:1], occ_pnts[..., 3:].unsqueeze(1).repeat(1, zyx.shape[1], 1)], dim=-1)
        occ_carte_coords_aug = torch.cat([occ_carte_coords[..., :1].unsqueeze(1).repeat(1, zyx.shape[1], 1), occ_carte_coords_aug], dim=-1)
        occ_exp_pnts, occ_carte_coords_aug = torch.masked_select(occ_exp_pnts, pick_masks).view(-1,occ_pnts.shape[-1]), torch.masked_select(occ_carte_coords_aug, pick_masks).view(-1, occ_carte_coords.shape[1])
        inrange_mask = torch.cat([occ_carte_coords_aug[:, 1:4] >= min_det_grid_ind, occ_carte_coords_aug[:, 1:4] <= max_det_grid_ind], dim=-1).all(dim=-1, keepdims=True)

        return torch.masked_select(occ_exp_pnts, inrange_mask).view(-1,occ_pnts.shape[-1]), torch.masked_select(occ_carte_coords_aug, inrange_mask).view(-1, occ_carte_coords.shape[1])

    def trans_voxel_grid(self, occ_xyz, b_inds, voxel_size, grid_size, point_cloud_range):
        nx, ny, nz = grid_size[0], grid_size[1], grid_size[2]

        f_corner = occ_xyz - point_cloud_range[0:3].unsqueeze(0)
        coords = torch.div(f_corner, voxel_size.unsqueeze(0))
        coords_x = torch.clamp(torch.floor(coords[..., 0]), min=0, max=nx - 1).to(torch.int64)
        coords_y = torch.clamp(torch.floor(coords[..., 1]), min=0, max=ny - 1).to(torch.int64)
        coords_z = torch.clamp(torch.floor(coords[..., 2]), min=0, max=nz - 1).to(torch.int64)
        # print("b_inds.shape, coords_x.shape", b_inds.shape, occ_xyz.shape, f_corner.shape, coords_x.shape)
        # print(nz, ny, nx, coords.shape)
        return torch.stack([b_inds, coords_z, coords_y, coords_x], axis=-1)

    def get_rand_range(self, rand_range):
        r1, r2 = rand_range
        return (r1 - r2) * torch.rand(1, device="cuda") + r2

    def filter_occ_points(self, batch_size, occ_probs, batch_dict):
        # occ_probs B X NZ X NY X NX
        probs_lst = []
        res_lst = []
        occ_coords_lst = []
        B, NZ, NY, NX = list(occ_probs.size())
        occ_thresh = self.occ_thresh if batch_dict["is_train"] else self.eval_occ_thresh
        max_add_occpnts_num = self.max_add_occpnts_num if batch_dict["is_train"] else self.eval_max_add_occpnts_num
        for i in range(batch_size):
            occ_mask = occ_probs[i] > self.occ_thresh
            occ_coords = torch.nonzero(occ_mask)
            numpass = torch.sum(occ_mask)
            if numpass > 0 and batch_dict["use_occ_prob"][i]:
                top_prob = occ_probs[i][occ_mask]
                if self.reg:
                    top_pnt = batch_dict["pred_sem_residuals"][i][:, occ_mask].permute(1,0)
                if (self.remain_percentage is not None) and batch_dict["is_train"]:
                    rand_use_mask = torch.cuda.FloatTensor(top_prob.shape).uniform_() <= self.get_rand_range(self.remain_percentage)
                    # print(top_prob.shape, torch.sum(rand_use_mask))
                    top_prob = top_prob[rand_use_mask]
                    if self.reg:
                        top_pnt = top_pnt[rand_use_mask, ...]
                    occ_coords = occ_coords[rand_use_mask]

                if top_prob.shape[0] > max_add_occpnts_num:
                    top_prob, top_ind = torch.topk(top_prob, max_add_occpnts_num, largest=True, sorted=False)
                    occ_coords = occ_coords[top_ind, ...]
                    if self.reg:
                        top_pnt = top_pnt[top_ind, ...]
                top_occ_coords = torch.cat((torch.unsqueeze(torch.ones_like(occ_coords[...,0], device=occ_coords.device, dtype=occ_coords.dtype), -1) * i, occ_coords), axis=-1)
                occ_coords_lst.append(top_occ_coords)
                probs_lst.append(top_prob)
                if self.reg:
                    res_lst.append(top_pnt)
        return res_lst, probs_lst, occ_coords_lst


    def occ_coords2absxyz(self, occ_coords, type, rot_z=None):
        # occ_coords  P X 5 (B,NZ,NY,NX)
        coord_center_x = self.occ_x_origin + (occ_coords[..., 3] + 0.5) * self.nvx
        coord_center_y = self.occ_y_origin + (occ_coords[..., 2] + 0.5) * self.nvy
        coord_center_z = self.occ_z_origin + (occ_coords[..., 1] + 0.5) * self.nvz
        if rot_z is not None:
            rot_z_batch = rot_z[occ_coords[..., 0]]
            if self.data_cfg.OCC.COORD_TYPE == "cartesian":
                noise_rotation = rot_z_batch * np.pi / 180
                points = torch.stack([coord_center_x, coord_center_y, coord_center_z], dim=-1)
                points = common_utils.rotate_points_along_z(points.unsqueeze(1), noise_rotation).squeeze(1)
                return points
            else:
                coord_center_y -= rot_z_batch
        occpnt_absxyz = coords_utils.uvd2absxyz(coord_center_x, coord_center_y, coord_center_z, type)
        return occpnt_absxyz


    def assemble_occ_points(self, occ_pnts, pnt_feat_dim, occ_probs):
        if self.res_num_dim < pnt_feat_dim:
            # feat_zeros = torch.zeros_like(occ_pnts[..., :(pnt_feat_dim - self.res_num_dim)], device="cuda")
            default_inten = self.data_cfg.OCC.INTEN if self.data_cfg.OCC.get("INTEN", None) is not None else 0.0
            feat_inten = torch.ones_like(occ_pnts[..., :1], device="cuda") * default_inten
            feat_padding = feat_inten
            if pnt_feat_dim > 4:
                feat_elong = torch.zeros_like(occ_pnts[..., :1], device="cuda")
                feat_padding = torch.cat([feat_inten, feat_elong], axis=-1)
            occ_pnts = torch.cat([occ_pnts, feat_padding], axis=-1)
        occ_probs = torch.unsqueeze(occ_probs, -1)
        # print("occ_probs.shape", occ_probs.shape)
        # print("occ_pnts.shape", occ_pnts.shape)
        occ_pnts = torch.cat([occ_pnts, occ_probs], axis=-1)
        if self.code_num_dim > 1:
            occ_pnts = torch.cat([occ_pnts, torch.ones_like(occ_probs, device="cuda")], axis=-1)
        return occ_pnts


    def assemble_gt_vox_points(self, batch_dict, dropmask=None, voxel_size = None, grid_size = None,  finer_point_cloud_range=None, batch_size=None, code_num_dim=None):
        gt_voxels, gt_voxel_num_points, gt_voxel_coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        M, P, C = list(gt_voxels.size())
        voxel_count = gt_voxels.shape[1]
        if self.config_realdrop and batch_dict["final_point_mask"] is not None and voxel_count == batch_dict["final_point_mask"].shape[1]:
            mask = batch_dict["final_point_mask"]
        else:
            mask = self.get_paddings_indicator(gt_voxel_num_points, voxel_count, axis=0)
        inds = mask.nonzero()
        det_gt_points = gt_voxels[inds[:,0], inds[:,1], :]
        det_gt_voxel_coords = gt_voxel_coords[inds[:,0], :].to(torch.int64)
        if self.config_realdrop and dropmask is not None:
            gt_points, gt_voxel_coords, drop_valid_inds = self.drop_points_with_drop_mask(det_gt_points, det_gt_voxel_coords, dropmask, voxel_size, grid_size, finer_point_cloud_range, batch_size)
            batch_dict["drop_det_voxel_point_xyz"] = det_gt_points[drop_valid_inds,:]
            batch_dict["drop_det_voxel_coords"] = det_gt_voxel_coords[drop_valid_inds,:]
        else:
            gt_points = det_gt_points
            gt_voxel_coords = det_gt_voxel_coords
        # ones = torch.unsqueeze(torch.ones_like(gt_points[...,0], device="cuda"),-1) #  ps, 1
        zeros = torch.unsqueeze(torch.zeros_like(gt_points[...,0], device="cuda"),-1)
        for i in range(code_num_dim if code_num_dim is not None else self.code_num_dim):
            gt_points = torch.cat((gt_points, zeros), axis=-1)
        return gt_points, gt_voxel_coords


    def drop_points_with_drop_mask(self, gt_points, gt_voxel_coords, voxel_drop_mask, voxel_size, grid_size, finer_point_cloud_range, bs):
        # batch_dict["voxel_drop_mask"] = voxel_drop_mask.view(bs, self.ny, self.nx, self.nz * self.sz * self.sy * self.sx)
        nvx, nvy, nvz = voxel_size
        nx, ny, nz = grid_size
        P, _ = list(gt_points.shape)
        finer_x_origin, finer_y_origin, finer_z_origin = finer_point_cloud_range[0], finer_point_cloud_range[1], finer_point_cloud_range[2]
        inrange_mask = torch.cat([gt_points[:, :3] >= self.finer_range[..., :3],
                                  gt_points[:, :3] <= self.finer_range[..., 3:]], axis=-1).all(-1)
        valid_inds = inrange_mask.nonzero()[...,0]
        # print(gt_points.shape, self.finer_range.shape, "inrange_mask:", inrange_mask.shape, "valid_inds:", valid_inds.shape, "gt_voxel_coords.shape", gt_voxel_coords.shape)
        points_inrange = gt_points[valid_inds,...][..., :3]
        voxel_coords_inrange = gt_voxel_coords[valid_inds,...]
        # print("points_inrange", points_inrange.shape)

        points_bnznynx = self.trans_voxel_grid(points_inrange, voxel_coords_inrange[...,0], torch.tensor(voxel_size, device="cuda", dtype=torch.float32), grid_size, torch.tensor(finer_point_cloud_range,device="cuda"))

        f_corner_x = points_inrange[:, 0] - (
                points_bnznynx[:, 3].to(points_inrange.dtype) * nvx + finer_x_origin)
        f_corner_y = points_inrange[:, 1] - (
                points_bnznynx[:, 2].to(points_inrange.dtype) * nvy + finer_y_origin)
        f_corner_z = points_inrange[:, 2] - (
                points_bnznynx[:, 1].to(points_inrange.dtype) * nvz + finer_z_origin)
        f_corner = torch.stack([f_corner_x, f_corner_y, f_corner_z], axis=-1)

        sub_coords = torch.floor(torch.div(f_corner, self.sub_voxel_size))
        sub_coords[..., 0] = torch.clamp(sub_coords[..., 0], min=0, max=self.sx - 1)
        sub_coords[..., 1] = torch.clamp(sub_coords[..., 1], min=0, max=self.sy - 1)
        sub_coords[..., 2] = torch.clamp(sub_coords[..., 2], min=0, max=self.sz - 1)
        point_coords_szsysx = torch.stack([sub_coords[..., 2], sub_coords[..., 1], sub_coords[..., 0]], axis=-1).to(torch.int64)
        drop_mask = voxel_drop_mask.view(bs, ny, nx, nz, self.sz, self.sy, self.sx)[points_bnznynx[...,0], points_bnznynx[...,2], points_bnznynx[...,3], points_bnznynx[...,1], point_coords_szsysx[...,0], point_coords_szsysx[...,1], point_coords_szsysx[...,2]]
        drop_ind = drop_mask.nonzero()[...,0]
        zeros = torch.zeros_like(drop_ind, dtype=torch.int8, device="cuda")
        keep_mask = torch.ones((P), dtype=torch.int8, device="cuda")
        drop_valid_inds = valid_inds[drop_ind]
        keep_mask[drop_valid_inds] = zeros
        keep_ind = keep_mask.nonzero()[...,0]
        return gt_points[keep_ind, ...], gt_voxel_coords[keep_ind, ...], drop_valid_inds

    def assemble_gt_points(self, batch_dict):
        gt_points = batch_dict["points"]
        # ones = torch.unsqueeze(torch.ones_like(gt_points[...,0], device="cuda"),-1) #  ps, 1
        zeros = torch.unsqueeze(torch.zeros_like(gt_points[...,0], device="cuda"),-1)
        for i in range(self.code_num_dim):
            gt_points = torch.cat((gt_points, zeros), axis=-1)
        return gt_points

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num > max_num
        return paddings_indicator


    def voxelize_pad(self, voxel_num_points, inds, inverse_indices, points):

        cluster_num = voxel_num_points.size()[0]
        max_points = torch.max(voxel_num_points).data.cpu().numpy()
        P, C = points.size()
        points = points[inds, :]
        inverse_indices = inverse_indices[inds] # 0, 0, 0, 1, 1, 1
        range_indices = torch.arange(0, P, device="cuda")
        voxel_num_points_addaxis = torch.cumsum(torch.cat([torch.zeros([1], dtype=torch.int64, device='cuda'), voxel_num_points[:-1]],dim=0), dim=0)
        indices_voxel = range_indices - voxel_num_points_addaxis[inverse_indices]
        voxel_points = torch.zeros((cluster_num, max_points, C), dtype = torch.float32, device="cuda")
        voxel_points[inverse_indices, indices_voxel, :] = points
        return voxel_points

    def combine_gt_occ_voxel_point(self, gt_points, gt_voxel_coords, occ_pnts, occ_voxel_coords, grid_size):
        points = torch.cat((gt_points, occ_pnts), axis=0)
        coords = torch.cat((gt_voxel_coords, occ_voxel_coords), axis=0)
        voxel_coords, inverse_indices, voxel_num_points = torch.unique(coords, dim=0, sorted=True, return_inverse=True, return_counts=True)
        _, inds = torch.sort(inverse_indices)
        voxel_points = self.voxelize_pad(voxel_num_points, inds, inverse_indices, points)
        return voxel_points, voxel_num_points, voxel_coords

    def pad_tensor(self, tensor, intent_length, dim=0):
        shape_lst = list(tensor.size())
        if intent_length - shape_lst[dim] > 0:
            shape_lst[dim] = intent_length - shape_lst[dim]
            zeros = torch.zeros(shape_lst, device='cuda')
            return torch.cat((tensor, zeros), axis=dim)
        else:
            return tensor


    def transform_points_to_voxels(self, points):
        voxel_output = self.voxel_generator.generate(points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output
        return voxels, coordinates, num_points


    def visualize(self, batch_dict, binds):
        tb = {}
        pc = {}
        occ_fore_center, occ_mirr_center, occ_bm_center, occ_pos_center, occ_neg_center = None, None, None, None, None
        if batch_dict["gt_b_ind"] is None:
            return tb, pc
        all_voxel_centers = point_box_utils.rotatez(self.all_voxel_centers, batch_dict["rot_z"][binds]) if "rot_z" in batch_dict else self.all_voxel_centers
        if self.reg:
            gt_voxel_centers = all_voxel_centers + batch_dict["res_mtrx"][binds].permute(1,2,3,0)
        else:
            gt_voxel_centers = all_voxel_centers
        points, bm_points, fore_gt_center, filter_center, boxvoxel_center, addpnt_view, drop_voxel_center, drop_det_voxel_point, drop_det_point_xyz = [np.zeros([0,3], dtype=np.float) for i in range(9)]
        gt_batch_inds = (batch_dict["gt_b_ind"] == binds).nonzero()
        points = batch_dict["gt_points_xyz"][gt_batch_inds[:, 0], ...].data.cpu().numpy()
        occ_batch_inds = (batch_dict["added_occ_b_ind"] == binds).nonzero()
        occ_points = batch_dict["added_occ_xyz"][occ_batch_inds[:, 0], ...].data.cpu().numpy()
        filtered_occpoints = occ_points[...,:3]
        predicted_occ_prob = batch_dict['batch_pred_occ_prob'][binds, ...]
        predicted_occ_abspred = all_voxel_centers #
        # print("predicted_occ_abspred", torch.min(predicted_occ_abspred.view(-1,3),dim=0)[0], torch.max(predicted_occ_abspred.view(-1,3),dim=0)[0])
        occ_center=None
        general_cls_loss_center=None
        point_dist_mask = batch_dict["vcc_mask"][binds, ...]
        box_3d = batch_dict["gt_boxes"][binds][:batch_dict["gt_boxes_num"][binds], ...].data.cpu().numpy()
        ################# fore_voxel_groundtruth view #################
        if self.model_cfg.OCC_PNT_UPDATE.VIS.FORE_VOX_GT_VIEW:
            inds = batch_dict["fore_voxelwise_mask"][binds,...].nonzero()
            fore_gt_center = gt_voxel_centers[inds[:, 0], inds[:, 1], inds[:, 2], :]
            forevox_gt_view = vis_occ_utils.draw_lidars_box3d_on_birdview(points, fore_gt_center.data.cpu().numpy(), [(200, 200, 200), (255, 211, 0)], box_3d, self.bev_img_h, self.bev_img_w, self.xrange[0], self.xrange[1], self.yrange[0], self.yrange[1])
            tb["forevox_gt_view_img"] = forevox_gt_view

        ################# occ complex view #################
        if self.model_cfg.OCC_PNT_UPDATE.VIS.OCC_FORE_VOX_GT_VIEW:
            inds = batch_dict["occ_fore_cls_mask"][binds,...].nonzero()
            occ_fore_center = gt_voxel_centers[inds[:, 0], inds[:, 1], inds[:, 2], :]
        if self.model_cfg.OCC_PNT_UPDATE.VIS.OCC_MIRR_VOX_GT_VIEW:
            inds = batch_dict["occ_mirr_cls_mask"][binds, ...].nonzero()
            occ_mirr_center = gt_voxel_centers[inds[:, 0], inds[:, 1], inds[:, 2], :]
        if self.model_cfg.OCC_PNT_UPDATE.VIS.OCC_BM_VOX_GT_VIEW:
            inds = batch_dict["occ_bm_cls_mask"][binds, ...].nonzero()
            occ_bm_center = gt_voxel_centers[inds[:, 0], inds[:, 1], inds[:, 2], :]
        if self.model_cfg.OCC_PNT_UPDATE.VIS.OCC_POS_VOX_GT_VIEW:
            inds = batch_dict["pos_mask"][binds, ...].nonzero()
            occ_pos_center = gt_voxel_centers[inds[:, 0], inds[:, 1], inds[:, 2], :]
        if self.model_cfg.OCC_PNT_UPDATE.VIS.OCC_NEG_VOX_GT_VIEW:
            inds = batch_dict["neg_mask"][binds, ...].nonzero()
            occ_neg_center = gt_voxel_centers[inds[:, 0], inds[:, 1], inds[:, 2], :]

        ################# occ view #########################
        if self.model_cfg.OCC_PNT_UPDATE.VIS.OCC_VOX:
            inds = batch_dict["occ_voxelwise_mask"][binds, ...].nonzero()
            occ_center = gt_voxel_centers[inds[:, 0], inds[:, 1], inds[:, 2], :]

        ################# best_match view #########################
        if self.model_cfg.OCC_PNT_UPDATE.VIS.BM_VOX_VIEW:
            inds = batch_dict["bm_voxelwise_mask"][binds, ...].nonzero()
            bmvoxel_center = gt_voxel_centers[inds[:, 0], inds[:, 1], inds[:, 2], :]

        ################# general_cls_loss view #########################
        if self.model_cfg.OCC_PNT_UPDATE.VIS.CLS_LOSS:
            inds = batch_dict["general_cls_loss_mask"][binds, ...].nonzero()
            general_cls_loss_center = gt_voxel_centers[inds[:, 0], inds[:, 1], inds[:, 2], :]

        ################# filter view #################
        if self.model_cfg.OCC_PNT_UPDATE.VIS.FILTER_VIEW and self.data_cfg.OCC.DIST_KERN[0] > 0:
            # (ny, nx, nz * sz * sy * sx) -> p x (nyi, nxj, permk)
            inds = (point_dist_mask > 1e-3).nonzero()
            # print("point_dist_mask",point_dist_mask.shape, self.all_voxel_centers.shape)
            filter_center = all_voxel_centers[inds[:, 0], inds[:, 1], inds[:, 2], :]
            filter_view = vis_occ_utils.draw_lidars_box3d_on_birdview(points, filter_center.data.cpu().numpy(), [(200, 200, 200), (255, 211, 0)], box_3d, self.bev_img_h, self.bev_img_w, self.xrange[0], self.xrange[1], self.yrange[0], self.yrange[1])
            tb["filter_view_img"] = filter_view
        ################# bbox view #################
        if self.model_cfg.OCC_PNT_UPDATE.VIS.BOX_VIEW and (self.data_cfg.OCC.BOX_WEIGHT != "None" or self.data_cfg.OCC.BOX_POSITIVE):
            inds = batch_dict["forebox_label"][binds, ...].nonzero()
            boxvoxel_center = all_voxel_centers[inds[:, 0], inds[:, 1], inds[:, 2], :]
            box_view = vis_occ_utils.draw_lidars_box3d_on_birdview(points, boxvoxel_center.data.cpu().numpy(), [(200, 200, 200), (255, 211, 0)], box_3d, self.bev_img_h, self.bev_img_w, self.xrange[0], self.xrange[1], self.yrange[0], self.yrange[1])
            tb["box_view_img"] = box_view
        ################# predicted foreground points view #################
        if self.model_cfg.OCC_PNT_UPDATE.VIS.PRED_FORE_VIEW:
            threshs = [0.9, .8, .7, .6, .5, .4, .3, .2, .1]
            for thresh in threshs:
                inds = (predicted_occ_prob >= thresh).nonzero()
                predicted_occ_abspred_fitlered = predicted_occ_abspred[inds[:, 0], inds[:, 1], inds[:, 2], :]
                predicted_occ_abspred_fitlered = predicted_occ_abspred_fitlered.detach().cpu().numpy()
                if thresh == 0.5:
                    filtered_occpoints = predicted_occ_abspred_fitlered
                forepnt_view = vis_occ_utils.draw_lidars_box3d_on_birdview(points, predicted_occ_abspred_fitlered, [(200, 200, 200), (255, 69, 0)], box_3d, self.bev_img_h, self.bev_img_w, self.xrange[0], self.xrange[1], self.yrange[0], self.yrange[1])
                tb["pred_fore_{}_img".format(thresh)] = forepnt_view
        ################# added point view #################
        if self.model_cfg.OCC_PNT_UPDATE.VIS.ADD_PNT_VIEW and (batch_dict["added_occ_xyz"] is not None):
            addpnt_view = vis_occ_utils.draw_lidars_box3d_on_birdview(points, occ_points, [(200, 200, 200), (255, 69, 0)], box_3d, self.bev_img_h, self.bev_img_w, self.xrange[0], self.xrange[1], self.yrange[0], self.yrange[1])
            tb["addpnt_img"] = addpnt_view
        ################# drop point view #################
        if self.data_cfg.OCC.DROPOUT_RATE > 1e-3 and self.model_cfg.OCC_PNT_UPDATE.VIS.DROP_VOX_VIEW and batch_dict["is_train"]:
            fordrop_inds = batch_dict["voxel_drop_mask"][binds, ...].nonzero()
            drop_voxel_center = all_voxel_centers[fordrop_inds[:, 0], fordrop_inds[:, 1], fordrop_inds[:, 2], :]
            # tb["drop_view_img"] = vis_occ_utils.draw_lidars_box3d_on_birdview(gt_points, drop_voxel_center.data.cpu().numpy(), [(200, 200, 200), (255, 211, 0)], box_3d, self.bev_img_h, self.bev_img_w, self.xrange[0], self.xrange[1], self.yrange[0], self.yrange[1])
        if self.model_cfg.OCC_PNT_UPDATE.VIS.get("BM_POINTS", False):
            gt_batch_inds = (batch_dict["bm_points"][:, 0] == binds).nonzero()
            bm_points = batch_dict["bm_points"][gt_batch_inds[:, 0], 1:4].data.cpu().numpy()


        if self.model_cfg.OCC_PNT_UPDATE.VIS.OUTPUT_CLOUD:
            pc.update({
                "gt_points": points,
                "bm_points": bm_points,
                "fore_gt_center": fore_gt_center,
                "occ_center": occ_center,
                "general_cls_loss_center": general_cls_loss_center,
                "filter_center": filter_center,
                "boxvoxel_center": boxvoxel_center,
                "addpnt_view": occ_points,
                "proboccpoints": filtered_occpoints,
                "drop_voxel_center": drop_voxel_center,
                "drop_det_voxel_point": drop_det_voxel_point,
                "drop_det_point_xyz": drop_det_point_xyz,
                "bmvoxel_center": bmvoxel_center,
                "gt_boxes": batch_dict["gt_boxes"][binds, :batch_dict["gt_boxes_num"][binds], ...]
            })

            pc.update({
                "occ_fore_center": occ_fore_center,
                "occ_mirr_center": occ_mirr_center,
                "occ_bm_center": occ_bm_center,
                "occ_pos_center": occ_pos_center,
                "occ_neg_center": occ_neg_center
            })

            # combined_cloud_ind = (batch_dict['voxel_coords'][:, 0] == binds).nonzero()
            # combined_cloud_pnts = batch_dict['voxels'][combined_cloud_ind[:, 0], ..., :3].view(-1, 3).data.cpu().numpy()
            # pc.update({"combined_cloud_pnts": combined_cloud_pnts})
            # pc.update(self.update_occ_vox(batch_dict))
        return tb, pc

    def filter_by_bind(self, trgt_binds, binds, points):
        trgt_binds = trgt_binds.to(torch.int64)
        inds = (binds == trgt_binds).nonzero()
        points = points[inds[:, 0], ...].data.cpu().numpy()
        return points

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C)
        """
        raise NotImplementedError


    def update_occ_vox(self, batch_dict):
        nx, ny, nz = [1408, 1600, 40]
        x_origin = self.data_cfg.POINT_CLOUD_RANGE[0]
        y_origin = self.data_cfg.POINT_CLOUD_RANGE[1]
        z_origin = self.data_cfg.POINT_CLOUD_RANGE[2]
        range_origin = torch.tensor([x_origin, y_origin, z_origin], dtype=torch.float32, device="cuda")
        grids_num = torch.tensor([nx, ny, nz], dtype=torch.int32, device="cuda")
        voxel_size = torch.tensor([0.05, 0.05, 0.1], dtype=torch.float32, device="cuda")
        all_voxel_centers = coords_utils.get_all_voxel_centers_xyz(1, grids_num, range_origin, voxel_size)[0, ...] #
        print("occ_voxel_features", range_origin, torch.max(all_voxel_centers.view(-1,3), dim=0)[0], torch.max(batch_dict["voxel_coords"], dim=0)[0], torch.min(batch_dict["voxel_coords"], dim=0)[0])
        inds = (batch_dict["occ_voxel_features"][..., 0] >= 0.1).nonzero()[...,0]
        vox_inds = batch_dict["voxel_coords"][inds, :]
        batch_vox_inds = (vox_inds[...,0] == 0).nonzero()[...,0]
        vox_inds = vox_inds[batch_vox_inds,:]
        print("inds, voxel_coords, vox_inds", inds.shape, batch_vox_inds.shape, vox_inds.shape, torch.sum(vox_inds[...,0]))

        predicted_occ_abspred_fitlered = all_voxel_centers[vox_inds[...,3], vox_inds[...,2], vox_inds[...,1], :].detach().cpu().numpy()

        return {"proboccpoints": predicted_occ_abspred_fitlered}