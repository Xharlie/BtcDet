import torch

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, data_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.maxprob = kwargs["maxprob"]
        self.OCC_CODE = model_cfg.get("OCC_CODE", None)
        self.xyz_dim = 6 if data_cfg.get("OCC", None) is not None and data_cfg.OCC.USE_ABSXYZ == "both" else 3
        self.num_point_features = num_point_features + (1 if self.OCC_CODE is not None and self.OCC_CODE else 0) + self.xyz_dim - 3
        self.num_raw_features = len(data_cfg.POINT_FEATURE_ENCODING.used_feature_list) + self.xyz_dim - 3


    def get_output_feature_dim(self):
        return self.num_point_features

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num_range = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = max_num_range < actual_num.int()
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']

        if not self.maxprob:
            points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
            points_mean = points_mean / normalizer
            batch_dict['voxel_features'] = points_mean.contiguous()

        # max prob
        else:
            voxel_count = voxel_features.shape[1]
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
            normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
            raw_mask = (voxel_features[:, :, -1] < 0.1) & mask
            raw_normalizer = torch.clamp_min(raw_mask[:,:].sum(dim=1, keepdim=False).view(-1, 1), min=1.0).type_as(voxel_features)

            xyz_mean = voxel_features[:, :, :self.xyz_dim].sum(dim=1, keepdim=False) / normalizer
            inten_elong_mean =  voxel_features[:, :, self.xyz_dim:self.num_raw_features].sum(dim=1, keepdim=False) / raw_normalizer
            occ_max = voxel_features[:,:,self.num_raw_features:].max(dim=1, keepdim=False)[0]
            batch_dict['voxel_features'] = torch.cat([xyz_mean, inten_elong_mean, occ_max], axis=-1).contiguous()
        #
        if self.OCC_CODE is not None:
            M, F = list(batch_dict['voxel_features'].shape)
            occ_bzyx = torch.nonzero((batch_dict["general_cls_loss_mask"] & (1-batch_dict["voxelwise_mask"])) > 0)
            N, _ = list(occ_bzyx.shape)
            batch_dict['voxel_coords'] = torch.cat([batch_dict['voxel_coords'], occ_bzyx], dim=0)
            if not self.OCC_CODE:
                batch_dict['voxel_features'] = torch.cat([batch_dict['voxel_features'], torch.zeros([N, F], device=batch_dict['voxel_features'].device, dtype=torch.float32)], dim=0)
            else:
                batch_dict['voxel_features'] = torch.cat([torch.cat([batch_dict['voxel_features'], torch.ones([M, 1], device=batch_dict['voxel_features'].device, dtype=torch.float32)], dim=-1), torch.cat([torch.zeros([N, F], device=batch_dict['voxel_features'].device, dtype=torch.float32), torch.zeros([N, 1], device=batch_dict['voxel_features'].device, dtype=torch.float32)], dim=-1)], dim=0)
        return batch_dict

