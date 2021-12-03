import torch
import numpy as np
from ....utils import box_utils
from .occ_head_template import OccHeadTemplate
from ...backbones_3d.spconv_backbone import post_act_block as block
import spconv
from functools import partial
import torch.nn as nn

class OccHead3D(OccHeadTemplate):
    def __init__(self, model_cfg, data_cfg, input_channels, num_class, grid_size):
        self.is_softmax = model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG.get("CLS_LOSS_TYPE", None) is not None and model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG.CLS_LOSS_TYPE == "softmax"
        super().__init__(
            model_cfg=model_cfg, data_cfg=data_cfg, num_class=num_class, grid_size=grid_size)
        print("OccHead num_class", num_class)
        print("OccHead input_channels", input_channels)

        self.stride = int(model_cfg.BACKBONE_3D.STRIDE)

        print("self.stride !!!!!!!!", self.stride, self.num_class)
        print("is_softmax:", self.is_softmax)
        cls_channel = num_class + 1 if self.is_softmax else num_class
        # norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_cls = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, (self.stride **3) * cls_channel, 3, padding=1, bias=True, indice_key='cls_ind'),
        )
        self.logit2prob = torch.nn.Softmax(dim=1) if self.is_softmax else torch.nn.Sigmoid()

        if self.reg:
            self.conv_res = spconv.SparseSequential(spconv.SubMConv3d(input_channels, (self.stride ** 3) * self.num_class * self.res_num_dim, 3, padding=1, bias=False, indice_key='res_ind'))

    def dim_transfrom(self, tensor):
        N, C, Z, H, W = list(tensor.size())
        print("N, C, Z, H, W", N, C, Z, H, W)
        # [N, 2, 2, 2, PERM, C, Z, H, W] -> [N, H, 2, W, 2, Z, 2, PERM, C] : [0, 1, 2, 3, 4, 5, 6, 7, 8] -> [0, 7, 2, 8, 3, 6, 1, 4, 5]
        tensor = tensor.permute(0, 3, 4, 2, 1)
        tensor = tensor.reshape(N, H, W, Z, C) #.contiguous()
        return tensor

    def forward(self, data_dict):
        # if self.training or hasattr(self.model_cfg, "OCC_POST_PROCESSING"):
        data_dict = super(OccHead3D, self).prepare_loss_map(data_dict)
        spatial_features = data_dict['encoded_spconv_tensor']
        # print("spatial_features.shape", spatial_features.shape)
        logit_preds = self.conv_cls(spatial_features).dense()
        prob_preds = self.logit2prob(logit_preds)[:,-1:,...] # can fit both sigmoid and softmax
        data_dict['pred_occ_logit'] = logit_preds # * torch.unsqueeze(data_dict["point_dist_mask"], 3)
        data_dict['batch_pred_occ_prob'] = prob_preds[:,-1,...] * data_dict["general_cls_loss_mask"]
        if self.reg:
            data_dict['pred_sem_residuals'] = self.conv_res(spatial_features).dense()
        return data_dict
