import torch

from .vfe_template import VFETemplate


class OccVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, data_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.maxprob = kwargs["maxprob"]
        self.num_raw_features = len(data_cfg.POINT_FEATURE_ENCODING.used_feature_list) # DATA_AUGMENTOR.AUG_CONFIG_LIST[0].get('NUM_POINT_FEATURES', None)

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
        voxel_features, voxel_num_points, voxel_coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']

        voxel_count = voxel_features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        raw_mask = (voxel_features[:, :, -1] < 0.05) & mask
        occ_mask = (voxel_features[:, :, -1] >= 0.05) & mask
        raw_normalizer = raw_mask[:, :].sum(dim=1, keepdim=False).view(-1, 1)
        occ_normalizer = occ_mask[:, :].sum(dim=1, keepdim=False).view(-1, 1)
        occ_voxel_mask = (occ_normalizer > 0.5) & (raw_normalizer < 0.5)
        raw_normalizer = torch.clamp_min(raw_normalizer, min=1.0).type_as(voxel_features)
        occ_normalizer = torch.clamp_min(occ_normalizer, min=1.0).type_as(voxel_features)

        voxel_features_raw = (raw_mask.unsqueeze(-1) * voxel_features[:, :, :self.num_raw_features]).sum(dim=1, keepdim=False) / raw_normalizer
        voxel_features_occ = (occ_mask.unsqueeze(-1) * voxel_features[:, :, :self.num_raw_features]).sum(dim=1, keepdim=False) / occ_normalizer

        batch_dict['voxel_features'] = voxel_features_raw + occ_voxel_mask * voxel_features_occ
        occ_max = voxel_features[:, :, self.num_raw_features:].max(dim=1, keepdim=False)[0]
        batch_dict['voxel_features'] = torch.cat([batch_dict['voxel_features'], occ_max], dim=-1)
        batch_dict['occ_voxel_features'] = occ_max
        # print("voxel_features", torch.min(batch_dict['voxel_features'],dim=0)[0], torch.max(batch_dict['voxel_features'],dim=0)[0])
        return batch_dict

