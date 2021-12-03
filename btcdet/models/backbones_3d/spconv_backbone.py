from functools import partial

import spconv
import torch.nn as nn
import torch

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None, defaultvalue=1.0, activation=nn.ReLU):
    # spconv.SparseConv3d(64, 128, (3, 3, 3), stride=(1, 1, 1), padding=1,
    #                     bias=False, indice_key='spconv_down2'),
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'subm2d':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'fixspconv':
        conv = fixSparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key, defaultvalue=defaultvalue)
        conv.requires_grad_(False)
    elif conv_type == 'spdeconv':
        conv = spconv.SparseConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    elif conv_type == 'submbias':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=True, indice_key=indice_key)
    elif conv_type == 'maxpool':
        conv = spconv.SparseMaxPool3d(kernel_size, stride=stride, padding=padding)
    else:
        raise NotImplementedError

    if norm_fn is not None:
        m = spconv.SparseSequential(
            conv,
            norm_fn(out_channels),
            activation(),
        )
    else:
        m = spconv.SparseSequential(
            conv
        )
    return m

class fixSparseConv3d(spconv.SparseConv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, indice_key=None, defaultvalue=1.0):
        super(fixSparseConv3d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, indice_key=indice_key)
        self.weight.data.fill_(defaultvalue)

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out




class VoxelBackBoneDeconv(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.y_shift = model_cfg.get("SHIFT", 0)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] # + [1, 0, 0]
        self.sparse_shape[1] += self.y_shift * 2

        print("self.sparse_shape", self.sparse_shape) # [  10 400 352]
        print("input_channels", input_channels)
        block = post_act_block
        channels = [16, 32, 64] # [16, 64, 128] # [16, 32, 64] # [16, 64, 128]

        self.conv1 = spconv.SparseSequential(
            block(input_channels, channels[0], 3, norm_fn=norm_fn, padding=1, indice_key='spconv1',  conv_type='spconv'),
        ) # 1

        self.conv2 = spconv.SparseSequential(
            block(channels[0], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(channels[1], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        ) # 2

        self.conv3 = spconv.SparseSequential(
            block(channels[1], channels[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(channels[2], channels[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )  # 2

        self.deconv4 = spconv.SparseSequential(
            block(channels[2], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spdeconv'),
            block(channels[1], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        )

        self.deconv5 = spconv.SparseSequential(
            block(channels[1], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv5', conv_type='spdeconv'),
            block(channels[1], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm5')
        )

        #
        # self.convlast = spconv.SparseSequential(
        #     block(channels[1], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='sublast'),
        #     # block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv4'),
        # )

        self.num_point_features = channels[1]

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        # voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords'].int() #batch_dict["valid_coords_bnznynx"]
        batch_size = batch_dict['batch_size']
        # print("voxel_features",voxel_features.shape, voxel_coords.shape)
        if self.y_shift > 0:
            voxel_features, voxel_coords = self.add_shift(voxel_features, voxel_coords)
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords,
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        # print("voxel_features", voxel_features.shape)
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv2d = self.deconv4(x_conv3)
        x_conv1d = self.deconv5(x_conv2d)
        if self.y_shift > 0:
            x_conv1d = self.remove_shift(x_conv1d)
        # print(input_sp_tensor.spatial_shape)
        # print(x_conv1.spatial_shape)
        # print(x_conv2.spatial_shape)
        # print(x_conv3.spatial_shape)
        # print(x_conv2d.spatial_shape)
        # print(x_conv1d.spatial_shape)
        # out = self.convlast(x_conv1d)

        # for name, param in self.conv1.named_parameters():
        #     print("name grad:", param.data, name)
        #     break
        # print("input_sp_tensor", torch.nonzero(input_sp_tensor.dense()).shape)
        # print("x_conv1", x_conv1.features.shape, x_conv1.dense().shape)
        # print("x_conv2", x_conv2.features.shape, x_conv2.dense().shape)
        # print("x_conv3", x_conv3.features.shape, x_conv3.dense().shape)
        # print("x_conv2d", x_conv2d.features.shape, x_conv2d.dense().shape)
        # print("x_conv1d", x_conv1d.features.shape, x_conv1d.dense().shape)
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        # out = x_conv4

        batch_dict.update({
            'encoded_spconv_tensor': x_conv1d,
            'encoded_spconv_tensor_stride': 1
        })
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv2d': x_conv2d,
        #         'x_conv1d': x_conv1d,
        #     }
        # })
        return batch_dict

    def remove_shift(self, sparse_feat):
        y_max = self.sparse_shape[1] - 2 * self.y_shift
        sparse_feat.indices[...,2] -= self.y_shift
        keep_inds = (sparse_feat.indices[..., 2] >= 0) & (sparse_feat.indices[..., 2] < y_max)
        sparse_feat.features = sparse_feat.features[keep_inds,:]
        sparse_feat.indices = sparse_feat.indices[keep_inds,:]
        sparse_feat.spatial_shape[1] -= self.y_shift * 2
        return sparse_feat

    def add_shift(self, voxel_features, voxel_coords):
        y_max = self.sparse_shape[1] - 2 * self.y_shift
        left_ind = voxel_coords[..., 2] < self.y_shift
        right_ind = voxel_coords[..., 2] >= (y_max - self.y_shift)
        left_feat, left_coords, right_feat, right_coords = voxel_features[left_ind,:], voxel_coords[left_ind,:].clone(), voxel_features[right_ind,:], voxel_coords[right_ind,:].clone()
        right_coords[..., 2] -= y_max
        left_coords[..., 2] += y_max
        all_features = torch.cat([right_feat, voxel_features, left_feat], dim=0)
        all_coords = torch.cat([right_coords, voxel_coords, left_coords], dim=0)
        all_coords[..., 2] += self.y_shift
        return all_features, all_coords

class VoxelBackBoneDeconvRes(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1]  # + [1, 0, 0]
        print("self.sparse_shape", self.sparse_shape)  # [  10 400 352]
        print("input_channels", input_channels)
        block = post_act_block
        channels = [16, 32, 64]  # [16, 64, 128] # [16, 32, 64] # [16, 64, 128]

        self.conv1 = spconv.SparseSequential(
            block(input_channels, channels[0], 3, norm_fn=norm_fn, padding=1, indice_key='spconv1', conv_type='spconv'),
        )  # 1

        self.conv2 = spconv.SparseSequential(
            block(channels[0], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2',
                  conv_type='spconv'),
            block(channels[1], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )  # 2

        self.conv3 = spconv.SparseSequential(
            block(channels[1], channels[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3',
                  conv_type='spconv'),
            block(channels[2], channels[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )  # 2

        self.conv22 = spconv.SparseSequential(
            block(channels[1], channels[1], 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv22',
                  conv_type='spconv'),
        )

        self.deconv2 = spconv.SparseSequential(
            block(channels[2], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconvd2',
                  conv_type='spdeconv'),
        )

        self.comb_conv2 = spconv.SparseSequential(
            block(channels[1]+channels[1], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='submd2')
        )

        self.conv11 = spconv.SparseSequential(
            block(channels[0], channels[0], 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv11',
                  conv_type='spconv'),
        )

        self.deconv1 = spconv.SparseSequential(
            block(channels[1], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconvd1',
                  conv_type='spdeconv'),
        )

        self.comb_conv1 = spconv.SparseSequential(
            block(channels[1] + channels[0], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='submd1')
        )

        # self.deconv5 = spconv.SparseSequential(
        #     block(channels[1], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconvd1',
        #           conv_type='spdeconv'),
        #     block(channels[1], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm5')
        # )

        # self.convlast = spconv.SparseSequential(
        #     block(channels[1], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='sublast'),
        #     # block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv4'),
        # )
        self.num_point_features = channels[1]


    # def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
    #     x_trans = conv_t(x_lateral)
    #     x = x_trans
    #     x.features = torch.cat((x_bottom.features, x_trans.features), dim=1)
    #     x_m = conv_m(x)
    #     x = self.channel_reduction(x, x_m.features.shape[1])
    #     x.features = x_m.features + x.features
    #     x = conv_inv(x)
    #     return x


    def combine(self, x_lateral, x_bottom, conv_expnd, comb_conv):
        x_expnd = conv_expnd(x_lateral)
        N, Fe = list(x_expnd.features.shape)
        M, Fb = list(x_bottom.features.shape)
        ind_e = x_expnd.indices
        ind_b = x_bottom.indices
        ind_all = torch.cat([ind_b, ind_e], dim=0)
        _, rinds = torch.unique(ind_all, dim=0, return_inverse=True)
        pad_feat = torch.zeros([M, Fe], dtype=torch.float32, device="cuda")
        pad_feat[rinds[M:], :] = x_expnd.features
        x_bottom.features = torch.cat([x_bottom.features, pad_feat], dim=-1)
        # print("ind_e", ind_e.shape, ind_b.shape)
        return comb_conv(x_bottom)


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        # voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords'] #batch_dict["valid_coords_bnznynx"]
        batch_size = batch_dict['batch_size']
        # print("voxel_features",voxel_features.shape, voxel_coords.shape)
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv2d = self.deconv2(x_conv3)
        x_conv2d = self.combine(x_conv2, x_conv2d, self.conv22, self.comb_conv2)
        x_conv1d = self.deconv1(x_conv2d)
        x_conv1d = self.combine(x_conv1, x_conv1d, self.conv11, self.comb_conv1)
        # x_conv1d = self.deconv5(x_conv2d)
        # out = self.convlast(x_conv1d)

        # for name, param in self.conv1.named_parameters():
        #     print("name grad:", param.data, name)
        #     break
        # print("input_sp_tensor", torch.nonzero(input_sp_tensor.dense()).shape)
        # print("x_conv1", x_conv1.features.shape, x_conv1.dense().shape)
        # print("x_conv2", x_conv2.features.shape, x_conv2.dense().shape)
        # print("x_conv3", x_conv3.features.shape, x_conv3.dense().shape)
        # print("x_conv2d", x_conv2d.features.shape, x_conv2d.dense().shape)
        # print("x_conv1d", x_conv1d.features.shape, x_conv1d.dense().shape)
        # print("out", out.features.shape, out.dense().shape)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        # out = x_conv4

        batch_dict.update({
            'encoded_spconv_tensor': x_conv1d,
            'encoded_spconv_tensor_stride': 1
        })
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv2d': x_conv2d,
        #         'x_conv1d': x_conv1d,
        #     }
        # })
        return batch_dict



class VoxelBackBoneInverseRes(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1]  # + [1, 0, 0]
        print("self.sparse_shape", self.sparse_shape)  # [  10 400 352]
        print("input_channels", input_channels)
        block = post_act_block
        channels = [16, 32, 64]  # [16, 64, 128] # [16, 32, 64] # [16, 64, 128]

        self.conv1 = spconv.SparseSequential(
            block(input_channels, channels[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )  # 1

        self.conv2 = spconv.SparseSequential(
            block(channels[0], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2',
                  conv_type='spconv'),
            block(channels[1], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )  # 2

        self.conv3 = spconv.SparseSequential(
            block(channels[1], channels[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3',
                  conv_type='spconv'),
            block(channels[2], channels[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )  # 2

        self.conv22 = spconv.SparseSequential(
            block(channels[1], channels[1], 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='subm2'),
        )

        self.deconv2 = spconv.SparseSequential(
            block(channels[2], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3',
                  conv_type='inverseconv'),
        )

        self.comb_conv2 = spconv.SparseSequential(
            block(channels[1]+channels[1], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2')
        )

        self.conv11 = spconv.SparseSequential(
            block(channels[0], channels[0], 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='subm1'),
        )

        self.deconv1 = spconv.SparseSequential(
            block(channels[1], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='inverseconv'),
        )

        self.comb_conv1 = spconv.SparseSequential(
            block(channels[1] + channels[0], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.num_point_features = channels[1]


    # def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
    #     x_trans = conv_t(x_lateral)
    #     x = x_trans
    #     x.features = torch.cat((x_bottom.features, x_trans.features), dim=1)
    #     x_m = conv_m(x)
    #     x = self.channel_reduction(x, x_m.features.shape[1])
    #     x.features = x_m.features + x.features
    #     x = conv_inv(x)
    #     return x


    def combine(self, x_lateral, x_bottom, conv_expnd, comb_conv):
        x_expnd = conv_expnd(x_lateral)
        N, Fe = list(x_expnd.features.shape)
        M, Fb = list(x_bottom.features.shape)
        ind_e = x_expnd.indices
        ind_b = x_bottom.indices
        ind_all = torch.cat([ind_b, ind_e], dim=0)
        _, rinds = torch.unique(ind_all, dim=0, return_inverse=True)
        pad_feat = torch.zeros([M, Fe], dtype=torch.float32, device="cuda")
        pad_feat[rinds[M:], :] = x_expnd.features
        x_bottom.features = torch.cat([x_bottom.features, pad_feat], dim=-1)
        # print("ind_e", ind_e.shape, ind_b.shape)
        return comb_conv(x_bottom)


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        # voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords'] #batch_dict["valid_coords_bnznynx"]
        batch_size = batch_dict['batch_size']
        # print("voxel_features",voxel_features.shape, voxel_coords.shape)
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv2d = self.deconv2(x_conv3)
        x_conv2d = self.combine(x_conv2, x_conv2d, self.conv22, self.comb_conv2)
        x_conv1d = self.deconv1(x_conv2d)
        x_conv1d = self.combine(x_conv1, x_conv1d, self.conv11, self.comb_conv1)
        # x_conv1d = self.deconv5(x_conv2d)
        # out = self.convlast(x_conv1d)

        # for name, param in self.conv1.named_parameters():
        #     print("name grad:", param.data, name)
        #     break
        # print("input_sp_tensor", torch.nonzero(input_sp_tensor.dense()).shape)
        # print("x_conv1", x_conv1.features.shape, x_conv1.dense().shape)
        # print("x_conv2", x_conv2.features.shape, x_conv2.dense().shape)
        # print("x_conv3", x_conv3.features.shape, x_conv3.dense().shape)
        # print("x_conv2d", x_conv2d.features.shape, x_conv2d.dense().shape)
        # print("x_conv1d", x_conv1d.features.shape, x_conv1d.dense().shape)
        # print("out", out.features.shape, out.dense().shape)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        # out = x_conv4

        batch_dict.update({
            'encoded_spconv_tensor': x_conv1d,
            'encoded_spconv_tensor_stride': 1
        })
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv2d': x_conv2d,
        #         'x_conv1d': x_conv1d,
        #     }
        # })
        return batch_dict



class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)


        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict


class VoxelBackBone8xOcc(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        # self.sparse_shape = grid_size
        self.occ_conv_type = self.model_cfg.OCC_CONV_TYPE
        self.occ_conv_exec = self.model_cfg.OCC_CONV_EXECUTE
        self.out_feat_type = getattr(self.model_cfg, "OUT_FEAT_TYPE", ["None","None","None","None","combine"])
        self.out_att = getattr(self.model_cfg, "OCC_ATT", [False, False, False, False])

        print("self.sparse_shape", self.sparse_shape) # [  41 1600 1408]
        channels = [16, 32, 64, 64, 128]
        print("input_channels", input_channels)
        self.occ_code_num = input_channels - kwargs["original_num_rawpoint_features"]
        block = post_act_block
        add_channels = [self.occ_code_num if t else 0 for t in self.occ_conv_exec] + [0 for i in range(len(self.occ_conv_exec), 4)]

        for i in range(1, len(self.occ_conv_exec)):
            getattr(self, 'build_occ_%s_net' % self.occ_conv_type[i])(norm_fn, i)
        for i in range(0, len(self.occ_conv_exec)):
            if self.out_att[i]:
                self.build_occ_att_net(norm_fn, i, channels[i] + add_channels[i])


        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )

        self.conv1_combine = spconv.SparseSequential(
            block(channels[0] + add_channels[0], channels[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        )
        self.conv2_combine = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[1] + add_channels[1], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(channels[1], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )
        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
        )
        self.conv3_combine = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[2] + add_channels[2], channels[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(channels[2], channels[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(channels[2], channels[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        )
        self.conv4_combine = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(channels[3] + add_channels[3], channels[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(channels[3], channels[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(channels[3], channels[4], (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(channels[4]),
            nn.ReLU(),
        )
        self.num_point_features = 128

        for i in range(4):
            z_channel = [41,21,11,5][i]
            if self.out_feat_type[i] == "2D":
                self.build_3d22d_net(norm_fn, i, channels[i]*z_channel)
        self.build_combine_net(norm_fn, channels, comb_type=self.out_feat_type[4])

    def build_3d22d_net(self, norm_fn, ind, in_channel):
        # print("x_conv1", x_conv1.spatial_shape) # [41 1600 1408]
        # print("x_conv2", x_conv2.spatial_shape) # [21, 800, 704]
        # print("x_conv3", x_conv3.spatial_shape) # [11, 400, 352]
        # print("x_conv4", x_conv4.spatial_shape) # [5, 200, 176]
        block = post_act_block
        if ind == 0:
            self.squeeze_z_conv1 = spconv.SparseSequential(block(in_channel, in_channel // 2, 3, norm_fn=norm_fn, padding=1, indice_key='submsqueez1', conv_type='subm2d'))
        elif ind == 1:
            self.squeeze_z_conv2 = spconv.SparseSequential(block(in_channel, in_channel // 2, 3, norm_fn=norm_fn, padding=1, indice_key='submsqueez2', conv_type='subm2d'))
        elif ind == 2:
            # self.squeeze_z_conv3 = nn.Sequential(nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, padding=1, stride=1, bias=False), nn.BatchNorm2d(in_channel // 2, eps=1e-3, momentum=0.01), nn.ReLU())
            self.squeeze_z_conv3 = spconv.SparseSequential(block(in_channel, in_channel // 2, 3, norm_fn=norm_fn, padding=1, indice_key='submsqueez3', conv_type='subm2d'))
        elif ind == 3:
            self.squeeze_z_conv4 = spconv.SparseSequential(block(in_channel, in_channel // 2, 3, norm_fn=norm_fn, padding=1, indice_key='submsqueez4', conv_type='subm2d'))


    def build_combine_net(self, norm_fn, channels, comb_type="combine"):
        # print("x_conv1", x_conv1.spatial_shape) # [41 1600 1408]
        # print("x_conv2", x_conv2.spatial_shape) # [21, 800, 704]
        # print("x_conv3", x_conv3.spatial_shape) # [11, 400, 352]
        # print("x_conv4", x_conv4.spatial_shape) # [5, 200, 176]
        block = post_act_block
        self.down2 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(channels[1], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3',
                  conv_type='spconv'),
            block(channels[1], channels[2], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4',
                  conv_type='spconv'),
        )
        self.down3 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(channels[2], channels[2], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4',
                  conv_type='spconv'),
        )
        if comb_type=="big_combine":
            self.down_combine = spconv.SparseSequential(
                # [400, 352, 5]
                block(channels[2] * 2 + channels[3], channels[3] * 2, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
                block(channels[3] * 2, channels[3] * 2, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
            )
        elif comb_type=="combine":
            self.down_combine = spconv.SparseSequential(
                # [200, 176, 5]
                block(channels[2] * 2 + channels[3], channels[3] * 2, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
                block(channels[3] * 2, channels[3] * 2, 3, norm_fn=norm_fn, stride=[1, 2, 2], padding=(1, 1, 1),
                      indice_key='spconv5', conv_type='spconv'),
                block(channels[3] * 2, channels[3] * 2, 3, norm_fn=norm_fn, padding=1, indice_key='subm5')
            )
        elif comb_type=="big_bev_combine":
            self.squeezeBev = spconv.SparseSequential(
                block(channels[4], channels[3], (2, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=0, indice_key='subm_down2', conv_type='spconv'),
            )
            self.down_combine = spconv.SparseSequential(
                # [400, 352, 5]
                block(channels[2] * 2 + channels[3] * 2, channels[3] * 2, 3, norm_fn=norm_fn, padding=1,
                      indice_key='subm4'),
                block(channels[3] * 2, channels[3] * 2, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
            )

    def build_occ_att_net(self, norm_fn, ind, channels):
        block = post_act_block
        if ind == 0:
            self.att_conv1 = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(channels, channels, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='subm1', activation=nn.LeakyReLU)
            )
        elif ind == 1:
            self.att_conv2 = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(channels, channels, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='subm2', activation=nn.LeakyReLU)
            )
        elif ind == 2:
            self.att_conv3 = spconv.SparseSequential(
                # [800, 704, 21] <- [400, 352, 11]
                block(channels, channels, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='subm3', activation=nn.LeakyReLU)
            )
        elif ind == 3:
            self.att_conv4 = spconv.SparseSequential(
                # [400, 352, 11] <- [200, 176, 5]
                block(channels, channels, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='subm4', activation=nn.LeakyReLU)
            )


    def build_occ_weight_net(self, norm_fn, ind):
        block = post_act_block
        if ind == 1:
            self.occ_conv2 = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(self.occ_code_num, self.occ_code_num, 3, norm_fn=None, stride=2, padding=1, indice_key='spconv2', conv_type='spconv')
            )
        elif ind == 2:
            self.occ_conv3 = spconv.SparseSequential(
                # [800, 704, 21] <- [400, 352, 11]
                block(self.occ_code_num, self.occ_code_num, 3, norm_fn=None, stride=2, padding=1, indice_key='spconv3', conv_type='spconv')
            )
        elif ind == 3:
            self.occ_conv4 = spconv.SparseSequential(
                # [400, 352, 11] <- [200, 176, 5]
                block(self.occ_code_num, self.occ_code_num, 3, norm_fn=None, stride=2, padding=(1, 1, 1), indice_key='spconv4', conv_type='spconv')
            )


    def build_occ_fix_net(self, norm_fn, ind):
        block = post_act_block
        if ind == 1:
            self.occ_conv2 = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(self.occ_code_num, self.occ_code_num, 3, norm_fn=None, stride=2, padding=1, indice_key='spconv2', conv_type='fixspconv', defaultvalue=1.0/27)
        )
        elif ind == 2:
            self.occ_conv3 = spconv.SparseSequential(
                # [800, 704, 21] <- [400, 352, 11]
                block(self.occ_code_num, self.occ_code_num, 3, norm_fn=None, stride=2, padding=1, indice_key='spconv3', conv_type='fixspconv', defaultvalue=1.0/27)
        )
        elif ind == 3:
            self.occ_conv4 = spconv.SparseSequential(
                # [400, 352, 11] <- [200, 176, 5]
                block(self.occ_code_num, self.occ_code_num, 3, norm_fn=None, stride=2, padding=(1, 1, 1), indice_key='spconv4', conv_type='fixspconv', defaultvalue=1.0/27)
            )


    def build_occ_maxpool_net(self, norm_fn, ind):
        block = post_act_block
        if ind == 1:
            self.occ_conv2 = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(self.occ_code_num, self.occ_code_num, 3, norm_fn=None, stride=2, padding=1, indice_key='spconv2', conv_type='maxpool')
            )
        elif ind == 2:
            self.occ_conv3 = spconv.SparseSequential(
                # [800, 704, 21] <- [400, 352, 11]
                block(self.occ_code_num, self.occ_code_num, 3, norm_fn=None, stride=2, padding=1, indice_key='spconv3', conv_type='maxpool')
            )
        elif ind == 3:
            self.occ_conv4 = spconv.SparseSequential(
                # [400, 352, 11] <- [200, 176, 5]
                block(self.occ_code_num, self.occ_code_num, 3, norm_fn=None, stride=2, padding=(1, 1, 1), indice_key='spconv4', conv_type='maxpool')
            )


    def build_occ_avgpool_net(self, norm_fn, ind):
        block = post_act_block
        if ind == 1:
            self.occ_conv2 = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(self.occ_code_num, self.occ_code_num, 2, norm_fn=None, stride=2, padding=1, indice_key='spconv2', conv_type='fixspconv', defaultvalue=1)
        )
        elif ind == 2:
            self.occ_conv3 = spconv.SparseSequential(
                # [800, 704, 21] <- [400, 352, 11]
                block(self.occ_code_num, self.occ_code_num, 2, norm_fn=None, stride=2, padding=1, indice_key='spconv3', conv_type='fixspconv', defaultvalue=1)
        )
        elif ind == 3:
            self.occ_conv4 = spconv.SparseSequential(
                # [400, 352, 11] <- [200, 176, 5]
                block(self.occ_code_num, self.occ_code_num, 2, norm_fn=None, stride=2, padding=(1, 1, 1), indice_key='spconv4', conv_type='fixspconv', defaultvalue=1)
            )


    def sparse_cat(self, input_lst):
        xrep, xocc = input_lst
        # print("xrep.features, xocc.features", xrep.features.shape, xocc.features.shape)
        xrep.features = torch.cat((xrep.features, xocc.features), dim=1)
        return xrep


    def apply_att(self, x_conv, att_conv):
        x_conv_att = att_conv(x_conv)
        # print("xrep.features, xocc.features", xrep.features.shape, xocc.features.shape)
        x_conv.features = x_conv.features * x_conv_att.features + x_conv.features
        return x_conv


    def suqeeze(self, feat, i, type):
        if type == "None":
            return None
        elif type == "3D":
            return feat
        else:
            conv = getattr(self, "squeeze_z_conv{}".format(i), None)
            threeD_feat = feat.dense()
            B, C, Z, Y, X = list(threeD_feat.shape)
            inds = torch.unique(torch.cat([feat.indices[...,0:1], feat.indices[..., 2:]], dim=-1), dim=0).long()
            # print("inds",inds.shape, inds.dtype)
            twoD_feat = threeD_feat.reshape(B, C * Z, Y, X)
            pixel_features = twoD_feat[inds[..., 0], :, inds[..., 1], inds[..., 2]]
            input_sp_tensor = spconv.SparseConvTensor(
                features=pixel_features,
                indices=inds.int(),
                spatial_shape=[Y, X],
                batch_size=B
            )
            return conv(input_sp_tensor).dense()


    def res_combine(self, x_conv2, x_conv3, x_conv4, bev_conv, out_feat_type="combine"):
        if getattr(self, "down3", None) is not None:
            x_conv2 = self.down2(x_conv2)
            x_conv3 = self.down3(x_conv3)
            x_conv4.features = torch.cat((x_conv2.features, x_conv3.features, x_conv4.features), dim=1)
            if out_feat_type == "big_bev_combine":
                bev_conv = self.squeezeBev(bev_conv)
                bev_conv = self.compress_height(bev_conv)
                inds = x_conv4.indices.long()
                bev_conv = bev_conv[inds[...,0], :, inds[...,2], inds[...,3]]
                x_conv4.features = torch.cat((x_conv4.features, bev_conv), dim=1)
            x_conv4 = self.down_combine(x_conv4)
            # print(x_conv4.dense().shape, x_conv4.spatial_shape)
            return x_conv4 # [2, 64, 5, 200, 176] [2, 128, 5, 100, 88]

    def compress_height(self, bev_conv):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        spatial_features = bev_conv.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        return spatial_features


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        # print("voxel_features",voxel_features.shape)
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x_conv1 = self.conv1(input_sp_tensor)
        if len(self.occ_conv_exec) > 0:
            occ_input_sp_tensor = spconv.SparseConvTensor(
                features=batch_dict["occ_voxel_features"],
                indices=voxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
        if len(self.occ_conv_exec) > 0 and self.occ_conv_exec[0]:
            x_conv1 = self.sparse_cat([x_conv1, occ_input_sp_tensor])
            if self.out_att[0]:
                x_conv1 = self.apply_att(x_conv1, self.att_conv1)
        x_conv1 = self.conv1_combine(x_conv1)
        x_conv2 = self.conv2(x_conv1)
        if len(self.occ_conv_exec) > 1:
            x_occ_conv2 = self.occ_conv2(occ_input_sp_tensor)
            if self.occ_conv_exec[1]:
                x_conv2 = self.sparse_cat([x_conv2, x_occ_conv2])
                if self.out_att[1]:
                    x_conv2 = self.apply_att(x_conv2, self.att_conv2)
        x_conv2 = self.conv2_combine(x_conv2)
        x_conv3 = self.conv3(x_conv2)
        if len(self.occ_conv_exec) > 2:
            x_occ_conv3 = self.occ_conv3(x_occ_conv2)
            if self.occ_conv_exec[2]:
                x_conv3 = self.sparse_cat([x_conv3, x_occ_conv3])
                if self.out_att[2]:
                    x_conv3 = self.apply_att(x_conv3, self.att_conv3)
        x_conv3 = self.conv3_combine(x_conv3)

        x_conv4 = self.conv4(x_conv3)
        if len(self.occ_conv_exec) > 3:
            x_occ_conv4 = self.occ_conv4(x_occ_conv3)
            if self.occ_conv_exec[3]:
                x_conv4 = self.sparse_cat([x_conv4, x_occ_conv4])
                if self.out_att[3]:
                    x_conv4 = self.apply_att(x_conv4, self.att_conv4)
        x_conv4 = self.conv4_combine(x_conv4)

        # print("x", x.spatial_shape) # [41 1600 1408]
        # print("x_conv1", x_conv1.spatial_shape) # [41 1600 1408]
        # print("x_conv2", x_conv2.spatial_shape) # [21, 800, 704]
        # print("x_conv3", x_conv3.spatial_shape) # [11, 400, 352]
        # print("x_conv4", x_conv4.spatial_shape) # [5, 200, 176]
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })

        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': self.suqeeze(x_conv1, 1, self.out_feat_type[0]),
                'x_conv2': self.suqeeze(x_conv2, 2, self.out_feat_type[1]),
                'x_conv3': self.suqeeze(x_conv3, 3, self.out_feat_type[2]),
                'x_conv4': self.suqeeze(x_conv4, 4, self.out_feat_type[3]),
                'x_combine': self.res_combine(x_conv2, x_conv3, x_conv4, out, out_feat_type=self.out_feat_type[4])
            }
        })

        return batch_dict