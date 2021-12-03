import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ....ops.roiaware_pool3d import roiaware_pool3d_utils
from ....utils import common_utils, loss_utils

class OccHeadTemplate(nn.Module):
    def __init__(self, model_cfg, data_cfg, num_class, grid_size):
        super().__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.noloss = self.model_cfg.OCC_DENSE_HEAD.get("NOLOSS", None) is not None and self.model_cfg.OCC_DENSE_HEAD.NOLOSS
        self.num_class = num_class
        self.forward_ret_dict = {}
        self.nx, self.ny, self.nz = grid_size
        self.occ_fore_res_weight = self.model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.get("occ_fore_res_weight", 0.1)
        self.occ_fore_cls_weight = self.model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.get("occ_fore_cls_weight", 1.0)
        self.res_num_dim = data_cfg.OCC.RES_NUM_DIM
        self.reg = model_cfg.PARAMS.get("REG", False)
        self.build_losses(self.model_cfg.OCC_DENSE_HEAD.LOSS_CONFIG)

    def prepare_loss_map(self, batch_dict):
        # self.forward_ret_dict.update({
        #     "pos_mask": batch_dict['pos_mask'],
        #     "general_cls_loss_mask_float": batch_dict['general_cls_loss_mask_float'],
        #     "general_cls_loss_mask": batch_dict['general_cls_loss_mask'],
        # })
        # batch_dict.pop("general_cls_loss_mask_float")
        # batch_dict.pop("general_cls_loss_mask")
        # batch_dict.pop("pos_mask")
        return batch_dict

    def build_losses(self, losses_cfg):
        if self.noloss:
            return
        if self.is_softmax:
            self.add_module(
                'cls_loss_func',
                loss_utils.SoftmaxFocalClassificationLoss(alpha=1.0, gamma=2.0)
            )
        else:
            self.add_module(
                'cls_loss_func',
                loss_utils.SigmoidFocalClassificationLoss(alpha=losses_cfg.LOSS_WEIGHTS['cls_alpha'], gamma=2.0)
            )
        if self.reg:
            reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None else losses_cfg.REG_LOSS_TYPE
            self.add_module( 'reg_loss_func', getattr(loss_utils, reg_loss_name)(beta=losses_cfg.LOSS_WEIGHTS['res_beta'], code_weights=[1.0 for i in range(self.res_num_dim)]))


    def get_res_layer_loss(self, batch_dict):
        general_reg_loss_mask = batch_dict["general_reg_loss_mask"]
        general_reg_loss_mask_float = batch_dict["general_reg_loss_mask_float"].unsqueeze(1)
        preds_sem_res = batch_dict['pred_sem_residuals']
        res_mtrx = batch_dict['res_mtrx']
        bs = int(preds_sem_res.shape[0])

        if self.num_class > 1:
            label_ind = torch.clamp((positive_voxelwise_labels - 1), min=0).view(-1)
            preds_sem_res = preds_sem_res.view(-1, self.num_class, self.res_num_dim)[self.first_perm, label_ind, :].view(bs, self.ny, self.nx, self.nz * self.sz * self.sy * self.sx, self.res_num_dim)
        reg_loss = self.mean_masked_loss(preds_sem_res, res_mtrx, self.reg_loss_func, general_reg_loss_mask_float, mask=general_reg_loss_mask) * self.occ_fore_res_weight

        tb_dict = {
            'occ_loss_res': reg_loss.item()
        }

        return reg_loss, tb_dict


    def get_cls_layer_loss(self, batch_dict):

        general_cls_loss_mask_float = batch_dict["general_cls_loss_mask_float"].unsqueeze(1)
        pred_occ_logit = batch_dict['pred_occ_logit']
        pos_mask = batch_dict["pos_mask"].to(pred_occ_logit.dtype)
        complimentary_mask = torch.ones_like(pos_mask, device=pos_mask.device, dtype=pos_mask.dtype) - pos_mask
        one_hot_targets = torch.stack([complimentary_mask, pos_mask], dim=-1)
        one_hot_targets = one_hot_targets if self.is_softmax else one_hot_targets[..., 1:]
        one_hot_targets = one_hot_targets.permute(0, 4, 1, 2, 3)

        cls_loss = self.mean_masked_loss(pred_occ_logit, one_hot_targets, self.cls_loss_func, general_cls_loss_mask_float, mask=batch_dict['general_cls_loss_mask']) * self.occ_fore_cls_weight
        tb_dict = {
            'occ_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict


    def mean_masked_loss(self, pred, target, loss_func, loss_weight_float, mask=None):
        # inds = (loss_weight_float > 1e-4).nonzero()
        # valid_weight = torch.clamp(torch.sum(loss_weight_float), min=1.0)
        # return torch.sum(loss_map[inds[:,0], inds[:,1], inds[:,2], inds[:,3], :]) / valid_weight
        inds = mask.nonzero() if mask is not None else (loss_weight_float > 1e-4).nonzero()
        loss_weight_float = loss_weight_float[inds[:, 0], :, inds[:, 1], inds[:, 2], inds[:, 3]]
        target = target[inds[:, 0], :, inds[:, 1], inds[:, 2], inds[:, 3]]
        pred = pred[inds[:, 0], :, inds[:, 1], inds[:, 2], inds[:, 3]]
        loss = loss_func(pred, target, weights=None)
        # print("pred", pred.shape, "target", target.shape, "loss",loss.shape, loss_weight_float.shape)
        loss *= loss_weight_float
        return torch.sum(loss) / torch.clamp(torch.sum(loss_weight_float), min=1.0)


    def get_loss(self, batch_dict):
        if self.noloss:
            return torch.tensor(0.0, device="cuda"), {}
        occ_loss, tb_dict = self.get_cls_layer_loss(batch_dict)
        if self.reg:
            reg_loss, tb_dict_res = self.get_res_layer_loss(batch_dict)
            occ_loss += reg_loss
            tb_dict.update(tb_dict_res)
        # tb_dict['occ_loss'] = occ_loss.item()
        return occ_loss, tb_dict


    def forward(self, **kwargs):
        raise NotImplementedError
