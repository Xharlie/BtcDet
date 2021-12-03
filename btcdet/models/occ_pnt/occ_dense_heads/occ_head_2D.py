import torch
import numpy as np
from ....utils import box_utils
from .occ_head_template import OccHeadTemplate


class OccHead2D(OccHeadTemplate):
    def __init__(self, model_cfg, data_cfg, input_channels, num_class, grid_size, finer_indx_range):
        self.is_softmax = model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG.get("CLS_LOSS_TYPE", None) is not None and model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG.CLS_LOSS_TYPE == "softmax"
        super().__init__(
            model_cfg=model_cfg, data_cfg=data_cfg, num_class=num_class, grid_size=grid_size, finer_indx_range=finer_indx_range)
        print("OccHead num_class", num_class)
        print("OccHead input_channels", input_channels)

        print("self.perm", self.perm)
        if model_cfg.get("BACKBONE_2D", None) is None:
            self.stride = 2
        else:
            self.stride = int(model_cfg.BACKBONE_2D.LAYER_STRIDES[0]) // int(model_cfg.BACKBONE_2D.UPSAMPLE_STRIDES[0])
        print("self.stride !!!!!!!!", self.stride, self.num_class)
        print("is_softmax:", self.is_softmax)
        cls_channel = num_class + 1 if self.is_softmax else num_class
        self.conv_cls = torch.nn.Conv2d(
            input_channels, self.perm * self.stride * self.stride * cls_channel,
            kernel_size=3, padding=1
        )
        self.conv_res = torch.nn.Conv2d(
            input_channels, self.perm * self.stride * self.stride * self.num_class * self.res_num_dim,
            kernel_size=3, padding=1
        )
        self.logit2prob = torch.nn.Softmax(dim=-1) if self.is_softmax else torch.nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        torch.nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        torch.nn.init.normal_(self.conv_res.weight, mean=0, std=0.001)

    def dim_transfrom(self, tensor, hstride, wstride, perm):
        N, CC, H, W = list(tensor.size())
        C = CC // hstride // wstride // perm
        tensor = tensor.view(N, hstride, wstride, perm, C, H, W) # [N, CC, H, W] -> [N, C, PERM, 2, 2, H, W]
        # [N, 2, 2, PERM, C, H, W] -> [N, H, 2, W, 2, PERM, C] : [0, 1, 2, 3, 4, 5, 6] -> [0, 5, 1, 6, 2, 3, 4]
        tensor = tensor.permute(0, 5, 1, 6, 2, 3, 4)
        tensor = tensor.reshape(N, H * hstride, W * wstride, perm, C) #.contiguous()
        return tensor

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features_2d'] if 'spatial_features_2d' in data_dict else data_dict["spatial_features"]
        bs = list(spatial_features.shape)[0]
        logit_preds = self.conv_cls(spatial_features) # [N, C, H, W] -> [N, C, PERM, 2, 2, H, W]
        res_preds = self.conv_res(spatial_features)
        # if
        logit_preds = self.dim_transfrom(logit_preds, self.stride, self.stride, self.perm)
        res_preds = self.dim_transfrom(res_preds, self.stride, self.stride, self.perm)
        N, H, W, perm, C = list(res_preds.shape)
        res_preds = res_preds.view(N, H, W, perm, self.num_class, self.res_num_dim)

        prob_preds = self.logit2prob(logit_preds)[..., -1:] # can fit both sigmoid and softmax
        self.forward_ret_dict['pred_occ_logit'] = logit_preds

        if self.num_class > 1:
            if not hasattr(self, 'first_perm'):
                self.first_perm = torch.arange(bs * self.ny * self.nx * self.nz * self.sz * self.sy * self.sx, dtype=torch.int64, device="cuda")

            label_ind = torch.max(logit_preds, axis=-1)[1].view(-1).to(torch.int64)
            res_preds_picked = res_preds.view(-1, self.num_class, self.res_num_dim)[self.first_perm, label_ind, :].view(bs, self.ny, self.nx, self.nz * self.sz * self.sy * self.sx, self.res_num_dim)
            prob_preds_picked = prob_preds.view(-1, self.num_class)[self.first_perm, label_ind].view(bs, self.ny, self.nx, self.nz * self.sz * self.sy * self.sx)
            data_dict['batch_pred_occ_label'] = label_ind + 1
            data_dict['batch_pred_occ_pnts'] = res_preds_picked
        else:
            prob_preds_picked = prob_preds[..., 0]
            data_dict['batch_pred_occ_label'] = torch.ones_like(prob_preds[...,0], dtype=torch.float32)
            res_preds_picked = res_preds.squeeze(-2)
        data_dict['batch_pred_occ_pnts'] = res_preds_picked
        data_dict['batch_pred_occ_prob'] = prob_preds_picked * data_dict["point_dist_mask"]
        # data_dict['batch_pred_occ_prob_prefilter'] = prob_preds_picked
        self.forward_ret_dict['pred_occ_residuals'] = res_preds_picked

        if self.training:
            super(OccHead2D, self).prepare_loss_map(data_dict)

        return data_dict
