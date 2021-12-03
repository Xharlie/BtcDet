import torch.nn as nn
import numpy as np
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils, point_box_utils
from .roi_head_template import RoIHeadTemplate
import torch
from ..backbones_3d.spconv_backbone import post_act_block as block
from functools import partial
import spconv
# from ..ops.pointnet2 import pointnet2_utils

class ConvHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg, **kwargs)
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        # for k in range(len(mlps)):
        #     mlps[k] = [input_channels] + mlps[k]
        #
        # self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
        #     radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
        #     nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
        #     mlps=mlps,
        #     use_xyz=True,
        #     pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
        # )

        self.fix_dims = self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', None) == "AbsResidualCoder"
        self.grid_size = self.model_cfg.CONV_GRID_POOL.GRID_SIZE
        if not isinstance(self.grid_size, list):
            self.grid_size = [self.grid_size, self.grid_size, self.grid_size]
        self.grid_num = np.prod(self.grid_size)
        self.pyramid_cfg = self.model_cfg.CONV_GRID_POOL.CONV_LAYER
        self.vis = getattr(self.model_cfg.CONV_GRID_POOL, "VIS" , False)
        self.dim_times = getattr(self.model_cfg.CONV_GRID_POOL, "DIM_TIMES" , 1.0)
        self.downsample_times_map = {}
        self.size_map = {}
        self.src_features = {}
        self.conv_layers = nn.ModuleList()
        self.conv_layer_names = []
        self.raw_points_agg, self.occ_points_agg = None, None
        self.point_rot = getattr(self.model_cfg.CONV_GRID_POOL, "POINT_ROT" , False)
        self.point_scale = getattr(self.model_cfg.CONV_GRID_POOL, "POINT_SCALE" , False)
        self.intrp_norm = getattr(self.model_cfg.CONV_GRID_POOL, "INTRP_NORM" , False)
        c_out = 0

        if 'raw_points' in self.model_cfg.CONV_GRID_POOL.FEATURES_SOURCE:
            mlps = self.pyramid_cfg['raw_points'].MLPS
            for k in range(len(mlps)):
                mlps[k] = [kwargs['num_rawpoint_features'] - 3] + mlps[k]
            self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSG(
                radii=self.pyramid_cfg['raw_points'].POOL_RADIUS,
                nsamples=self.pyramid_cfg['raw_points'].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool'
            )
            agg_mlps = getattr(self.pyramid_cfg['raw_points'], "AGG_MLPS", None)
            if agg_mlps is not None:
                self.raw_points_agg = []
                for k in range(len(agg_mlps) - 1):
                    self.raw_points_agg.extend([
                        nn.Conv2d(agg_mlps[k], agg_mlps[k + 1], kernel_size=1, bias=False),
                        nn.BatchNorm2d(agg_mlps[k + 1]),
                        nn.ReLU()
                    ])
                self.raw_points_agg = nn.Sequential(*self.raw_points_agg)
                c_out += agg_mlps[-1]
            else:
                c_out += sum([x[-1] for x in mlps])

        if 'occ_points' in self.model_cfg.CONV_GRID_POOL.FEATURES_SOURCE:
            mlps = self.pyramid_cfg['occ_points'].MLPS
            for k in range(len(mlps)):
                mlps[k] = [1] + mlps[k]

            self.SA_occpoints = pointnet2_stack_modules.StackSAModuleMSG(
                radii=self.pyramid_cfg['occ_points'].POOL_RADIUS,
                nsamples=self.pyramid_cfg['occ_points'].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool'
            )
            agg_mlps = getattr(self.pyramid_cfg['occ_points'], "AGG_MLPS", None)
            if agg_mlps is not None:
                self.occ_points_agg = []
                for k in range(len(agg_mlps) - 1):
                    self.occ_points_agg.extend([
                        nn.Conv2d(agg_mlps[k], agg_mlps[k + 1], kernel_size=1, bias=False),
                        nn.BatchNorm2d(agg_mlps[k + 1]),
                        nn.ReLU()
                    ])
                self.occ_points_agg = nn.Sequential(*self.occ_points_agg)
                c_out += agg_mlps[-1]
            else:
                c_out += sum([x[-1] for x in mlps])

        for src_name in self.model_cfg.CONV_GRID_POOL.FEATURES_SOURCE:
            if src_name in ['bev_conv', 'raw_points', 'occ_points']:
                continue
            self.downsample_times_map[src_name] = self.pyramid_cfg[src_name].DOWNSAMPLE_FACTOR
            part_scene_size = np.array(self.pyramid_cfg[src_name].PART_SCENE_SIZE) # zyx
            ker_size = self.pyramid_cfg[src_name].KER_SIZE # zyx
            strides = self.pyramid_cfg[src_name].STRIDE # zyx
            channels = self.pyramid_cfg[src_name].CHANNEL # zyx
            kernels = self.pyramid_cfg[src_name].KERNEL # zyx
            paddings = self.pyramid_cfg[src_name].PADDING # zyx
            dms = len(part_scene_size)
            local_grid_size = np.around(((part_scene_size[dms//2:dms] - part_scene_size[:dms//2]) / ker_size)).astype(int)
            self.size_map[src_name] = {
                "local_grid_size": local_grid_size,
                "scene_xyz_orgin_shifted": torch.as_tensor([part_scene_size[i] + 0.5 * ker_size[i] for i in range(dms//2-1, -1, -1)], device="cuda", dtype=torch.float32),
                "ker_xyz_size": torch.as_tensor([ker_size[i] for i in range(dms//2-1, -1, -1)], device="cuda", dtype=torch.float32),
                "dims": torch.as_tensor([(part_scene_size[i+dms//2] - part_scene_size[i]) for i in range(dms//2-1, -1, -1)], device='cuda', dtype=torch.float32),
                "scene_times": self.pyramid_cfg[src_name].get("SCENE_TIMES", 1)
            }
            print("!!!local_grid_size",  self.size_map[src_name]["local_grid_size"])
            print("!!!dims",  self.size_map[src_name]["dims"])
            if dms == 6:
                cur_layer = spconv.SparseSequential(*[block(channels[i], channels[i+1], kernels[i], norm_fn=norm_fn, stride=strides[i], padding=paddings[i], indice_key='{}_spconv{}'.format(src_name,i), conv_type='spconv') for i in range(len(strides))])
            else:
                bev_conv = []
                for i in range(len(strides)):
                    bev_conv.extend([nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernels[i], padding=paddings[i], stride=strides[i], bias=False), nn.BatchNorm2d(channels[i + 1], eps=1e-3, momentum=0.01), nn.ReLU()])
                cur_layer = nn.Sequential(*bev_conv)
            self.conv_layers.append(cur_layer)
            self.conv_layer_names.append(src_name)
            c_out += channels[-1]

        pre_channel = self.grid_num * c_out
        self.det_voxel_size = kwargs["det_voxel_size"]
        self.point_cloud_range = kwargs["point_cloud_range"]

        if getattr(self.model_cfg, "SHARED_3D_CONV", None) is not None and self.model_cfg.SHARED_3D_CONV.KERNEL.__len__() > 0:
            pre_channel = self.create_shared_3dconv(c_out)

        if getattr(self.model_cfg, "SHARED_FC", None) is not None and self.model_cfg.SHARED_FC.__len__() > 0:
            pre_channel = self.create_shared_fc(pre_channel)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')


    def create_shared_fc(self, pre_channel):
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        return pre_channel

    def create_shared_3dconv(self, pre_channel):
        shared_3dconv_list = []
        kernels =self.model_cfg.SHARED_3D_CONV.KERNEL
        paddings = self.model_cfg.SHARED_3D_CONV.PADDING
        strides = self.model_cfg.SHARED_3D_CONV.STRIDE
        channels = self.model_cfg.SHARED_3D_CONV.CHANNEL
        for k in range(0, kernels.__len__()):
            shared_3dconv_list.extend([
                nn.Conv3d(pre_channel, channels[k], kernel_size=kernels[k], bias=False, stride=strides[k], padding=paddings[k]),
                nn.BatchNorm3d(channels[k]),
                nn.ReLU()
            ])
            pre_channel = channels[k]

            if (getattr(self.model_cfg, "SHARED_FC", None) is None or self.model_cfg.SHARED_FC.__len__() == 0) and k != kernels.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_3dconv_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_3dconv_layer = nn.Sequential(*shared_3dconv_list)
        return pre_channel

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def create_local_conv_grid(self, global_roi_grid_points, rois, local_grid_size, dims, scene_times):
        # rois (B, num_rois, 7 + C)
        # global_roi_grid_points (BxNXx6x6, 3)
        B, N_box, F = list(rois.shape)
        local_rois = rois.view(-1, rois.shape[-1]).unsqueeze(1).repeat(1, self.grid_num, 1).view(-1, F)
        local_rois[...,:3] = global_roi_grid_points # (BxNX6x6x6, 7+C)
        if self.fix_dims:
            local_rois[..., 3:6] = dims.view(1,3).repeat(local_rois.shape[0],1)
        else:
            local_rois[..., 3:6] *= scene_times
        global_conv_grid_points, dense_idx = self.get_global_grid_points_of_roi(
            local_rois, grid_size=local_grid_size, dim_times=1.0
        ) # BXNX6X6X6, 8x32x32, 3
        return global_conv_grid_points.view(B, -1, 3), dense_idx  # B, NX6X6X6 X 8X32X32, 3


    def get_local_conv_regions(self, points, yaw, dims):
        # points (BxN, 6x6x6, 3)
        # yaw (B, N, 1)
        B, N, _ = list(yaw.shape)
        BNG, _ = list(points.shape)
        local_yaw = yaw.view(-1, yaw.shape[-1]).unsqueeze(1).repeat(1, self.grid_num, 1).view(-1, 1) # (BXN, 6X6X6, 1)
        dims = dims.view(1,3).repeat(BNG, 1).view(-1, 3)
        return torch.cat([points, dims, local_yaw], dim=-1).view(B, -1, 7)


    def splat_features_2_grids(self, batch_size, conv_grid_points, dense_idx, conv_regions, features, src_name, size_map, N_boxes):
        indices = features.indices
        # center_xyz = common_utils.get_voxel_centers(
        #     indices[:, 1:4],
        #     downsample_times=self.downsample_times_map[src_name],
        #     voxel_size=self.det_voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # ) # M, 3
        # unique_idxs, features = self.interpolate_from_3d_sparse_features(conv_regions, features, center_xyz, indices, batch_size, size_map, N_boxes)
        unique_idxs, features = self.interpolate_from_3d_features(conv_grid_points, dense_idx, features, self.downsample_times_map[src_name])
        return unique_idxs, features

    def roi_conv_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        vis_dict = {}
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'] # (B, num_rois, 7 + C)

        B, N_box, _ = list(rois.shape)
        global_roi_grid_points, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.grid_size, e2e=False, dim_times=self.dim_times
        )  # (BxN, 6x6x6, 3)
        if self.vis:
            vis_dict["rois"] = batch_dict['rois']
            vis_dict["rois_global_grid_points"] = global_roi_grid_points.view(B, -1, 3)[0]
        # print("global_roi_grid_points", global_roi_grid_points.shape)
        global_roi_grid_points = global_roi_grid_points.view(-1, 3)
        new_xyz = global_roi_grid_points
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(N_box*self.grid_num)
        out_features_lst = []
        rotateMatrix, xyscales, zscales = None, None, None
        if self.point_rot:
            rotateMatrix = point_box_utils.torch_get_yaw_rotation(-rois[..., 6].view(-1))
        if self.point_scale:
            xyscales, zscales = torch.sqrt(rois[..., 3] ** 2 + rois[..., 4] ** 2), rois[..., 5]
            xyscales = xyscales.view(-1, 1, 1, 1).repeat(1, self.grid_num, 1, 1).view(-1, 1, 1)
            zscales = zscales.view(-1, 1, 1, 1).repeat(1, self.grid_num, 1, 1).view(-1, 1, 1)
        if 'raw_points' in self.model_cfg.CONV_GRID_POOL.FEATURES_SOURCE:
            raw_points = batch_dict['points']
            # print("raw_points", raw_points.shape)
            xyz = raw_points[:, 1:4]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()
            point_features = raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None

            result_lst = self.SA_rawpoints(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features,
                rotateMatrix=rotateMatrix,
                xyscales=xyscales,
                zscales=zscales,
                vis=self.vis
            )
            if self.vis:
                pooled_points, pooled_features, point_lst_lst = result_lst
                points_lst, prerot_points_lst = point_lst_lst
                for i in range(len(points_lst)):
                    vis_dict["rois_raw_points_{}".format(i)] = (points_lst[i] + global_roi_grid_points.view(B, -1, 3)[0].reshape(-1, 1, 3)).reshape(-1,3) # [6912, 3]
                    vis_dict["rois_raw_rot_points_{}".format(i)] = (prerot_points_lst[i] + global_roi_grid_points.view(B,-1,3)[0].reshape(-1, 1, 3)).reshape(-1,3) # [6912, 3]
            else:
                pooled_points, pooled_features = result_lst
            pooled_features = self.raw_points_agg(pooled_features.view(B * N_box * self.grid_num, -1, 1, 1)) if self.raw_points_agg is not None else pooled_features
            out_features_lst.append(pooled_features.view(B * N_box * self.grid_num, -1))

        if 'occ_points' in self.model_cfg.CONV_GRID_POOL.FEATURES_SOURCE:
            occ_points = batch_dict['occ_pnts']
            # print("raw_points", raw_points.shape)
            xyz = occ_points[:, 0:3]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (batch_dict['added_occ_b_ind'] == bs_idx).sum()
            point_features = occ_points[:, 3:].contiguous() if occ_points.shape[1] > 3 else None

            result_lst = self.SA_occpoints(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features,
                rotateMatrix=rotateMatrix,
                xyscales=xyscales,
                zscales=zscales,
                vis=self.vis
            )
            if self.vis:
                pooled_points, pooled_features, point_lst_lst = result_lst
                points_lst, prerot_points_lst = point_lst_lst
                for i in range(len(points_lst)):
                    vis_dict["rois_occ_points_{}".format(i)] = (points_lst[i] + global_roi_grid_points.view(B, -1, 3)[0].reshape(-1, 1, 3)).reshape(-1,3) # [6912, 3]
                    vis_dict["rois_occ_rot_points_{}".format(i)] = (prerot_points_lst[i] + global_roi_grid_points.view(B,-1,3)[0].reshape(-1, 1, 3)).reshape(-1,3) # [6912, 3]
            else:
                pooled_points, pooled_features = result_lst
            if self.occ_points_agg is not None:
                pooled_features = self.occ_points_agg(pooled_features.view(B * N_box * self.grid_num, -1, 1, 1))
            out_features_lst.append(pooled_features.view(B * N_box * self.grid_num, -1))

        for src_name, conv_layer in zip(self.conv_layer_names, self.conv_layers):
            size_map = self.size_map[src_name]
            # conv_regions = self.get_local_conv_regions(global_roi_grid_points, rois[..., 6:7], size_map["dims"]) # B, N X 666, 7
            conv_grid_points, dense_idx = self.create_local_conv_grid(global_roi_grid_points, rois, size_map["local_grid_size"] if len(size_map["local_grid_size"])==3 else np.concatenate([np.ones_like(size_map["local_grid_size"][0:1]), size_map["local_grid_size"]], axis=0), self.size_map[src_name]["dims"], self.size_map[src_name]["scene_times"]) # B, NX6X6X6 * 8x32x32, 3
            # print("conv_grid_points", conv_grid_points.shape)

            in_features = batch_dict["spatial_features"] if src_name == "bev" else batch_dict['multi_scale_3d_features'][src_name]
            if len(size_map["local_grid_size"]) == 3:
                voxel_coords, voxel_features = self.splat_features_2_grids(batch_size, conv_grid_points, dense_idx, None, in_features, src_name, size_map, N_box)
                if self.vis:
                    vis_dict["rois_conv_grid_points"] = conv_grid_points[0]
                    vis_dict["rois_sparse_grid"] = self.vis_voxel(B, voxel_coords, global_roi_grid_points, size_map["ker_xyz_size"])

                input_sp_tensor = spconv.SparseConvTensor(
                    features=voxel_features,
                    indices=voxel_coords.int(),
                    spatial_shape=size_map["local_grid_size"],
                    batch_size=batch_size * N_box * self.grid_num
                )
                out_features = conv_layer(input_sp_tensor).dense()
                out_features = torch.squeeze(out_features)
                # print("out_features", out_features.shape, B, N_box)
            else:
                out_features = self.interpolate_from_bev_features(conv_grid_points.view(batch_size, -1, 3), in_features, batch_dict['batch_size'], bev_stride=self.downsample_times_map[src_name])  # B, NboxX666X
                out_features = out_features.view(batch_size * N_box * self.grid_num, size_map["local_grid_size"][0], size_map["local_grid_size"][1], out_features.shape[-1]).permute(0, 3, 1, 2)
                out_features = torch.squeeze(conv_layer(out_features))
            out_features_lst.append(out_features)
        out_features = torch.cat(out_features_lst, dim=-1) if len(out_features_lst) > 0 else out_features_lst[0]
        out_features = out_features.view(B * N_box, *self.grid_size, out_features.shape[-1]).permute(0,4,1,2,3).contiguous() # BN, C, 6, 6, 6

        if getattr(self, "shared_3dconv_layer", None) is not None:
            out_features = self.shared_3dconv_layer(out_features) # BN, C, 1, 1, 1
        # print("out_features", out_features.shape)
        return out_features.view(B * N_box, -1, 1), vis_dict


    def vis_voxel(self, B, voxel_coords, global_roi_grid_points, ker_xyz_size):
        center_xyz = common_utils.get_voxel_centers(
            voxel_coords[...,1:4],
            downsample_times=1,
            voxel_size=ker_xyz_size,
            point_cloud_range=self.point_cloud_range
        ) # M, 3
        center_xyz += global_roi_grid_points[voxel_coords[..., 0].long(),:]
        BN = list(global_roi_grid_points.shape)[0]
        mask = voxel_coords[..., 0].long() < (BN // B)
        return center_xyz[mask]

    def get_global_grid_points_of_roi(self, rois, grid_size, e2e=False, dim_times=1.0):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points, dense_idx = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size, e2e=e2e, dim_times=dim_times)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ) # .squeeze(dim=1)
        # print("local_roi_grid_points", local_roi_grid_points.shape, dense_idx.shape, global_roi_grid_points.shape)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, dense_idx


    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size, e2e=False, dim_times=1.0):
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        grid_size = np.array(grid_size)
        faked_features = rois.new_ones((grid_size[0], grid_size[1], grid_size[2]))
        dense_idx = faked_features.nonzero()  # (N, 666, 3) [z_idx, y_idx, x_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)
        # print("dense_idx", dense_idx.shape)
        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6] * dim_times
        if e2e and grid_size[2] > 1:
            roi_grid_points = (torch.flip(dense_idx, [-1])) * local_roi_size.unsqueeze(dim=1) / torch.as_tensor((grid_size-1).reshape(1, 1, 3), device=dense_idx.device) - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        else:
            roi_grid_points = (torch.flip(dense_idx, [-1]) + 0.5) * local_roi_size.unsqueeze(dim=1) / torch.flip(torch.as_tensor(grid_size, device=dense_idx.device), [0]).reshape(1, 1, 3) - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        # print("dense_idx",  torch.flip(dense_idx, [-1])[0,:,:], torch.flip(torch.as_tensor(grid_size, device=dense_idx.device), [0]), local_roi_size)

        return roi_grid_points, dense_idx

    
    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds


    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST'])

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        shared_features, vis_dict = self.roi_conv_pool(batch_dict)  # (BxN, 6x6x6, C)
        batch_dict["pooled_features"] = shared_features
        if getattr(self, "shared_fc_layer", None) is not None:
            shared_features = self.shared_fc_layer(shared_features)
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            self.forward_ret_dict = targets_dict
        if self.vis:
            batch_dict['conv_vis_dict'] = vis_dict
        return batch_dict


    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.det_voxel_size[0] / bev_stride -0.5
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.det_voxel_size[1] / bev_stride -0.5
        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = common_utils.bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs, normalize=self.intrp_norm)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features


    def interpolate_from_3d_features(self, conv_grid_points, dense_idx, features, stride):
        if isinstance(stride, int):
            stride = [stride, stride, stride]
        # B, N*666*, 3
        B, NP, _ = list(conv_grid_points.shape)
        BN, P, _ = list(dense_idx.shape)
        dense_idx = dense_idx.view(-1, 3)
        x_target_idxs = (conv_grid_points[:, :, 0] - self.point_cloud_range[0]) / self.det_voxel_size[0] / stride[2] - 0.5
        y_target_idxs = (conv_grid_points[:, :, 1] - self.point_cloud_range[1]) / self.det_voxel_size[1] / stride[1] - 0.5
        z_target_idxs = (conv_grid_points[:, :, 2] - self.point_cloud_range[2]) / self.det_voxel_size[2] / stride[0] - 0.5
        spatial_target_idxs = torch.stack([z_target_idxs.view(-1), y_target_idxs.view(-1), x_target_idxs.view(-1)], dim=-1)
        b_target_idxs = torch.arange(B, device=conv_grid_points.device, dtype=torch.long).view(B, 1).repeat(1, NP).view(-1)
        b_idxs = torch.arange(BN, device=conv_grid_points.device, dtype=torch.long).view(BN, 1).repeat(1, P).view(-1)
        bzyx = torch.stack([b_idxs, dense_idx[..., 0], dense_idx[..., 1], dense_idx[..., 2]], dim=-1)

        # print("conv_grid_points", conv_grid_points.shape, b_target_idxs.shape, spatial_target_idxs.shape)

        feat = common_utils.reverse_sparse_trilinear_interpolate_torch(features, b_target_idxs, spatial_target_idxs, normalize=self.intrp_norm)
        mask = torch.any(torch.abs(feat) > 0.0, dim=-1)
        inds = torch.nonzero(mask)[..., 0]
        return bzyx[inds,:], feat[inds,:]



    def interpolate_from_3d_sparse_features(self, conv_regions, in_features, center_xyz, indices, batch_size, size_map, N_boxes):
        # conv_regions  B, NX666, 7
        # center_xyz    M, 3
        # print("x_conv1", x_conv1.spatial_shape) # [41 1600 1408]
        # print("x_conv2", x_conv2.spatial_shape) # [21, 800, 704]
        # print("x_conv3", x_conv3.spatial_shape) # [11, 400, 352]
        # print("x_conv4", x_conv4.spatial_shape) # [5, 200, 176]
        idxs_list = []
        features_list = []
        for k in range(batch_size):
            mask = indices[..., 0] == k
            cur_center_xyz = center_xyz[mask, :]
            binds, center_idxs, center_in_box_mask = self.retrieve_feat_in_region_norot(conv_regions[k], cur_center_xyz, size_map)
            N, M = list(center_in_box_mask.shape)
            _, F = list(in_features.shape)
            cur_features = in_features[mask, :].unsqueeze(1).repeat(1, M, 1)  # (C, F
            valid_features = torch.masked_select(cur_features, center_in_box_mask.unsqueeze(-1)).view(-1, F)
            valid_binds = torch.masked_select(binds, center_in_box_mask.unsqueeze(-1)).view(-1, 1)
            valid_center_idxs = torch.masked_select(center_idxs, center_in_box_mask.unsqueeze(-1)).view(-1, 3)
            # print(conv_regions.shape,"valid_features", valid_features.shape, valid_binds.shape, valid_center_idxs.shape, valid_binds.dtype, valid_center_idxs.dtype)
            valid_features, valid_binds, valid_center_idxs = torch.cat([valid_features.new_zeros(N_boxes, F), valid_features], dim=0), torch.cat([torch.arange(N_boxes, device=valid_binds.device, dtype=valid_binds.dtype).unsqueeze(-1), valid_binds], dim=0), torch.cat([valid_center_idxs.new_zeros(N_boxes, 3), valid_center_idxs], dim=0)
            unique_idxs, features = common_utils.sparse_trilinear_interpolate_torch(valid_features, valid_binds, valid_center_idxs, size_map['local_grid_size'])
            unique_idxs[..., 0] += k*M
            features_list.append(features)
            idxs_list.append(unique_idxs)
        features = torch.cat(features_list, dim=0)  # (B, N, C0)
        unique_idxs = torch.cat(idxs_list, dim=0)  # (B, N, C0)
        return unique_idxs, features


    def retrieve_feat_in_region(self, regions, points, size_map, shift=0.0):
        center = regions[:, :3]
        dim = regions[:, 3:6]
        heading = regions[:, 6]
        # M, 3, 3
        rotation = point_box_utils.torch_get_yaw_rotation(heading)
        # M, 4, 4
        transform = point_box_utils.torch_get_transform(rotation, center)
        # M, 4, 4
        transform = torch.inverse(transform)
        # M, 3, 3
        reversed_rotation = transform[:, :3, :3]
        # M, 3
        reversed_translation = transform[:, :3, 3]
        # N, M, 3
        # print("points, rotation, translation", points.shape, rotation.shape, translation.shape)
        point_in_box_frame = torch.einsum("nj,mij->nmi", points, reversed_rotation) + reversed_translation
        # N, M, 3
        point_in_box_mask = (point_in_box_frame <= dim * 0.5 + shift) & (point_in_box_frame >= -dim * 0.5 - shift)
        point_in_box_mask = torch.prod(point_in_box_mask, axis=-1, dtype=torch.int8) # N, M
        idxs = (point_in_box_frame - size_map['scene_xyz_orgin_shifted'].view(1, 1, 3)) / size_map['ker_xyz_size'].view(1,1,3) # N, M, 3
        binds = torch.arange(idxs.shape[1], device=idxs.device, dtype=torch.int16).view(1,idxs.shape[1],1).repeat(idxs.shape[0],1,1)
        return binds, idxs, point_in_box_mask > 0

    def retrieve_feat_in_region_norot(self, regions, points, size_map, shift=0.0):
        center = regions[:, :3]
        dim = regions[:, 3:6]
        # heading = regions[:, 6]
        # M, 3, 3
        # rotation = point_box_utils.torch_get_yaw_rotation(heading)
        # # M, 4, 4
        # transform = point_box_utils.torch_get_transform(rotation, center)
        # # M, 4, 4
        # transform = torch.inverse(transform)
        # # M, 3, 3
        # reversed_rotation = transform[:, :3, :3]
        # # M, 3
        # reversed_translation = transform[:, :3, 3]
        # # N, M, 3
        # # print("points, rotation, translation", points.shape, rotation.shape, translation.shape)
        # point_in_box_frame = torch.einsum("nj,mij->nmi", points, reversed_rotation) + reversed_translation
        point_in_box_frame = points.unsqueeze(1) - center.unsqueeze(0)
        # N, M, 3
        point_in_box_mask = (point_in_box_frame <= dim * 0.5 + shift) & (point_in_box_frame >= -dim * 0.5 - shift)
        point_in_box_mask = torch.prod(point_in_box_mask, axis=-1, dtype=torch.int8) # N, M
        idxs = (point_in_box_frame - size_map['scene_xyz_orgin_shifted'].view(1, 1, 3)) / size_map['ker_xyz_size'].view(1,1,3) # N, M, 3
        binds = torch.arange(idxs.shape[1], device=idxs.device, dtype=torch.int16).view(1,idxs.shape[1],1).repeat(idxs.shape[0],1,1)
        return binds, idxs, point_in_box_mask > 0