import torch
import torch.nn as nn
#
from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ....utils import common_utils


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class AbstractionTemplate(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, original_num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        self.num_rawpoint_features = num_rawpoint_features
        self.original_num_rawpoint_features = original_num_rawpoint_features
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['raw_points', 'occ_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
            mlps = SA_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [mlps[k][0]] + mlps[k]
            cur_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg[src_name].POOL_RADIUS,
                nsamples=SA_cfg[src_name].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool',
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += sum([x[-1] for x in mlps])

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            mlps = SA_cfg['raw_points'].MLPS
            for k in range(len(mlps)):
                mlps[k] = [num_rawpoint_features - 3] + mlps[k]

            self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg['raw_points'].POOL_RADIUS,
                nsamples=SA_cfg['raw_points'].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool'
            )
            c_in += sum([x[-1] for x in mlps])
            
        if 'occ_points' in self.model_cfg.FEATURES_SOURCE:
            mlps = SA_cfg['occ_points'].MLPS
            for k in range(len(mlps)):
                mlps[k] = [1] + mlps[k]

            self.SA_occpoints = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg['occ_points'].POOL_RADIUS,
                nsamples=SA_cfg['occ_points'].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool'
            )
            c_in += sum([x[-1] for x in mlps])
            
        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in
        print("c_in", c_in)

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride, correction=False):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride - (0.5 if correction else 0.0)
        y_idxs = y_idxs / bev_stride - (0.5 if correction else 0.0)
        point_bev_features_list = []
        bilinear_interpolate = common_utils.bilinear_interpolate_torch if correction else bilinear_interpolate_torch
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features


    def get_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'occ_points':
            src_points = batch_dict['added_occ_xyz']
            batch_indices = batch_dict['added_occ_b_ind'].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)
            if sampled_points.shape[1] == 0:
                sampled_points = torch.zeros([1,self.model_cfg.NUM_KEYPOINTS, 3], dtype=torch.float32, device=src_points.device)
            else:
                if self.model_cfg.SAMPLE_METHOD == 'FPS':
                    cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                        sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                    ).long()

                    if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                        empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                        if empty_num < self.model_cfg.NUM_KEYPOINTS / 2:
                            cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
                        else:
                            rand_cur_pt_idxs = cur_pt_idxs[:, torch.randint(sampled_points.shape[1], (empty_num,), dtype=torch.int64)]
                            cur_pt_idxs = torch.cat([cur_pt_idxs[:, :sampled_points.shape[1]], rand_cur_pt_idxs], dim=-1)
                    sampled_points = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

                elif self.model_cfg.SAMPLE_METHOD == 'FastFPS':
                    raise NotImplementedError
                else:
                    raise NotImplementedError
            keypoints_list.append(sampled_points)
        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints


    def multi_get_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']
        multi_keypoints_list=[]
        for i in range(len(self.model_cfg.POINT_SOURCE)):
            if self.model_cfg.POINT_SOURCE[i] == 'raw_points':
                src_points = batch_dict['points'][:, 1:4]
                batch_indices = batch_dict['points'][:, 0].long()
            elif self.model_cfg.POINT_SOURCE[i] == 'voxel_centers':
                src_points = common_utils.get_voxel_centers(
                    batch_dict['voxel_coords'][:, 1:4],
                    downsample_times=1,
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )
                batch_indices = batch_dict['voxel_coords'][:, 0].long()
            elif self.model_cfg.POINT_SOURCE[i] == 'occ_points':
                src_points = batch_dict['added_occ_xyz']
                batch_indices = batch_dict['added_occ_b_ind'].long()
            else:
                raise NotImplementedError
            keypoints_list = []
            for bs_idx in range(batch_size):
                bs_mask = (batch_indices == bs_idx)
                sampled_points = src_points[bs_mask].unsqueeze(dim=0)
                if sampled_points.shape[1] == 0:
                    sampled_points = torch.zeros([1,self.model_cfg.NUM_KEYPOINTS[i], 3], dtype=torch.float32, device=src_points.device)
                else:
                    if self.model_cfg.SAMPLE_METHOD[i] == 'FPS':
                        cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                            sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS[i]
                        ).long()
                        if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS[i]:
                            empty_num = self.model_cfg.NUM_KEYPOINTS[i] - sampled_points.shape[1]
                            if empty_num < self.model_cfg.NUM_KEYPOINTS[i] / 2:
                                cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
                            else:
                                rand_cur_pt_idxs = cur_pt_idxs[:, torch.randint(sampled_points.shape[1], (empty_num,), dtype=torch.int64)]
                                cur_pt_idxs = torch.cat([cur_pt_idxs[:, :sampled_points.shape[1]], rand_cur_pt_idxs], dim=-1)
                        sampled_points = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
                    elif self.model_cfg.SAMPLE_METHOD[i] == 'FastFPS':
                        raise NotImplementedError
                    else:
                        raise NotImplementedError
                keypoints_list.append(sampled_points)
            keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
            multi_keypoints_list.append(keypoints)
        keypoints = torch.cat(multi_keypoints_list, dim=1)
        return keypoints


    def forward(self, batch_dict):
        raise NotImplementedError
