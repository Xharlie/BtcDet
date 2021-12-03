import torch
import torch.nn as nn
from .add_occ_template import AddOccTemplate


class PassOccVox(AddOccTemplate):
    def __init__(self, model_cfg, data_cfg, point_cloud_range, occ_voxel_size, occ_grid_size, det_voxel_size, det_grid_size, mode, voxel_centers, **kwargs):
        super().__init__(model_cfg, data_cfg, point_cloud_range, occ_voxel_size, occ_grid_size, det_voxel_size, det_grid_size, mode, voxel_centers)

    def forward(self, batch_dict, **kwargs):
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']
        _, _, pnt_feat_dim = list(voxel_features.size())

        batch_size, pre_occ_probs = batch_dict['batch_size'], batch_dict['batch_pred_occ_prob']
        res_lst, probs_lst, occ_coords_lst = self.filter_occ_points(batch_size, pre_occ_probs, batch_dict)
        batch_dict["added_occ_xyz"] = None
        batch_dict["occ_pnts"] = None
        batch_dict["added_occ_b_ind"] = None
        # debug
        # valid_inds=torch.nonzero(batch_dict["voxel_point_mask"])
        # batch_dict["gt_points_xyz"] = voxel_features[valid_inds[:, 0], valid_inds[:, 1]][...,:3]
        # batch_dict["gt_b_ind"] = coords[valid_inds[:, 0]][...,0]
        batch_dict["gt_points_xyz"] = batch_dict["points"][..., 1:4]
        batch_dict["gt_b_ind"] = batch_dict["points"][..., 0]

        if 'det_voxel_coords' in batch_dict:
            batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords'] = batch_dict['det_voxels'], batch_dict['det_voxel_num_points'], batch_dict['det_voxel_coords']

        if len(probs_lst) > 0:
            occ_probs = probs_lst[0] if len(probs_lst)==1 else torch.cat(probs_lst, axis=0)
            occ_coords = occ_coords_lst[0] if len(occ_coords_lst) == 1 else torch.cat(occ_coords_lst, axis=0)
            batch_dict["added_occ_xyz"] = self.occ_coords2absxyz(occ_coords, self.data_cfg.OCC.COORD_TYPE, rot_z=batch_dict["rot_z"] if "rot_z" in batch_dict else None)
            if self.reg:
                occ_res = res_lst[0] if len(res_lst) == 1 else torch.cat(res_lst, axis=0)
                batch_dict["added_occ_xyz"] += occ_res
            occ_pnts = torch.cat([batch_dict["added_occ_xyz"], occ_probs.unsqueeze(-1)], dim=-1)
            batch_dict["added_occ_b_ind"] = occ_coords[..., 0]
            batch_dict["occ_pnts"] = occ_pnts
            occ_carte_coords = self.trans_voxel_grid(batch_dict["added_occ_xyz"], occ_coords[..., 0], self.det_voxel_size, self.det_grid_size, self.point_cloud_range)
            occ_pnts = self.assemble_occ_points(batch_dict["added_occ_xyz"], pnt_feat_dim, occ_probs)
            if self.db_proj:
                occ_pnts, occ_carte_coords = self.db_proj_func(occ_pnts, occ_coords, occ_carte_coords, batch_dict, expand=[1.0, 5.0, 3.0], stride=[1.0, 2.5, 1.5])
            gt_points, gt_voxel_coords = self.assemble_gt_vox_points(batch_dict)
            voxels, voxel_num_points, voxel_coords = self.combine_gt_occ_voxel_point(gt_points, gt_voxel_coords, occ_pnts, occ_carte_coords, self.det_grid_size)
            batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords'] = voxels, voxel_num_points, voxel_coords
        else:
            zeros = torch.zeros_like(batch_dict['voxels'][..., 0:self.code_num_dim], device="cuda")
            batch_dict["added_occ_b_ind"] = torch.zeros([1],dtype=torch.int64, device="cuda")
            batch_dict["added_occ_xyz"] = torch.zeros([1, 3],dtype=torch.float32, device="cuda")
            batch_dict["occ_pnts"] = torch.zeros([1, 4],dtype=torch.float32, device="cuda")
            batch_dict['voxels'] = torch.cat((batch_dict['voxels'], zeros), axis=-1)

        if not self.pass_gradient:
            batch_dict['occ_pnts'] = batch_dict['occ_pnts'].detach()
            batch_dict['added_occ_xyz'] = batch_dict['added_occ_xyz'].detach()
            batch_dict['added_occ_b_ind'] = batch_dict['added_occ_b_ind'].detach()
            batch_dict['voxels'] = batch_dict['voxels'].detach()
        return batch_dict

