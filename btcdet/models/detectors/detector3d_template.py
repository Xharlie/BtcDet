import os

import torch
import torch.nn as nn

from ...ops.iou3d_nms import iou3d_nms_utils
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads, occ_pnt
from ..occ_pnt import occ_dense_heads, occ_training_targets
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils
from ...utils import coords_utils, point_box_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils

class Detector3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset, full_config=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        if model_cfg.get("OCC", None) is not None:
            self.occ_num_class = 1 if model_cfg.OCC.PARAMS.CLASS_AGNOSTIC else num_class
        else:
            self.occ_num_class = None
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'occ', 'vfe', 'backbone_3d', 'map_to_bev_module', 'backbone_2d', 'occ_pfe', 'occ_point_head', 'pfe', 'dense_head',  'point_head', 'roi_head'
        ]

        self.occ_module_topology = [
            'occ_targets', 'vfe', 'backbone_3d', 'map_to_bev_module', 'backbone_2d', 'occ_dense_head', 'occ_pnt_update'
        ]

        self.voxel_centers = None
        if self.dataset.dataset_cfg.get('OCC', None) is not None:
            self.voxel_centers = self.create_subvox_loc()

        self.occ_modules = nn.Module()
        self.det_modules = nn.Module()
        self.clamp_max = self.dataset.dataset_cfg.get("CLAMP", None)
        self.occ_dim = self.dataset.occ_dim
        self.print= False

    def filter_by_bind(self, trgt_binds, binds, points):
        trgt_binds = trgt_binds.to(torch.int64)
        inds = torch.nonzero(binds == trgt_binds)
        points = points[inds[:, 0], ...].data.cpu().numpy()
        return points
    
    def create_subvox_loc(self):
        nx, ny, nz = self.dataset.occ_grid_size
        x_origin = self.dataset.dataset_cfg.OCC.POINT_CLOUD_RANGE[0]
        y_origin = self.dataset.dataset_cfg.OCC.POINT_CLOUD_RANGE[1]
        z_origin = self.dataset.dataset_cfg.OCC.POINT_CLOUD_RANGE[2]
        range_origin = [x_origin, y_origin, z_origin]
        grids_num = torch.tensor([nx, ny, nz], dtype=torch.int32, device="cuda")
        voxel_size = torch.tensor(self.dataset.dataset_cfg.OCC.VOXEL_SIZE, dtype=torch.float32, device="cuda")
        all_voxel_centers = coords_utils.get_all_voxel_centers_zyx(1, grids_num, range_origin, voxel_size)[0, ...] # 3 sphere zyx nz ny nx
        all_voxel_centers = coords_utils.uvd2absxyz(all_voxel_centers[2, ...], all_voxel_centers[1, ...], all_voxel_centers[0, ...], self.dataset.dataset_cfg.OCC.COORD_TYPE, dim=-1) #  nz X ny X nx X 3
        all_voxel_centers_2d = torch.mean(all_voxel_centers[:, :, :, :2], dim=0).view(-1, 2) # N, 2
        return {"all_voxel_centers": all_voxel_centers, "all_voxel_centers_2d": all_voxel_centers_2d}


    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'occ_module_list': [],
            'det_module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_voxel_point_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'det_grid_size': self.dataset.det_grid_size,
            'occ_grid_size': self.dataset.occ_grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'occ_point_cloud_range': self.dataset.occ_point_cloud_range,
            'occ_voxel_size': self.dataset.occ_voxel_size,
            'det_voxel_size': self.dataset.det_voxel_size,
        }

        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            if module_name != "occ":
                # self.add_module(module_name, module)
                if module is not None:
                    self.det_modules.add_module(module_name, module)
                    if self.print: print("det module:", module_name)

        return model_info_dict['occ_module_list'], model_info_dict['det_module_list']


    def build_occ(self, model_info_dict):
        if self.model_cfg.get('OCC', None) is None:
            return None, model_info_dict

        for module_name in self.occ_module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict, occ=True
            )
            # self.add_module('occ_%s' % module_name, module)
            if module is not None:
                self.occ_modules.add_module(module_name, module)
                if self.print: print("occ module:", module_name)
        return None, model_info_dict


    def build_occ_targets(self, model_info_dict, occ=True):
        model_cfg = self.model_cfg.OCC
        if model_cfg.get('TARGETS', None) is None:
            return None, model_info_dict
        occ_target_module = occ_training_targets.__all__[model_cfg.TARGETS.NAME](
            model_cfg=model_cfg,
            point_cloud_range=model_info_dict['occ_point_cloud_range'],
            voxel_size=model_info_dict['occ_voxel_size'],
            data_cfg=self.dataset.dataset_cfg,
            grid_size=model_info_dict['occ_grid_size'],
            num_class=self.occ_num_class,
            voxel_centers=self.voxel_centers,
        )
        model_info_dict['occ_module_list'].append(occ_target_module)
        return occ_target_module, model_info_dict


    def build_occ_pnt_update(self, model_info_dict, occ=True):
        model_cfg = self.model_cfg.OCC if occ else self.model_cfg
        if model_cfg.get('OCC_PNT_UPDATE', None) is None:
            return None, model_info_dict

        occ_pnt_module = occ_pnt.__all__[model_cfg.OCC_PNT_UPDATE.NAME](
            model_cfg=model_cfg,
            data_cfg=self.dataset.dataset_cfg,
            point_cloud_range=model_info_dict['point_cloud_range'],
            occ_voxel_size=model_info_dict['occ_voxel_size'],
            occ_grid_size=model_info_dict['occ_grid_size'],
            det_voxel_size=model_info_dict['det_voxel_size'],
            det_grid_size=model_info_dict['det_grid_size'],
            mode = self.dataset.mode,
            voxel_centers=self.voxel_centers,
        )
        if self.print: print("build occ pnt update mode:", self.dataset.mode)
        model_info_dict['occ_module_list'].append(occ_pnt_module)
        if occ_pnt_module.config_rawadd:
            model_info_dict['num_rawpoint_features'] = model_info_dict['num_rawpoint_features'] + occ_pnt_module.code_num_dim
        model_info_dict['num_voxel_point_features']  = model_info_dict['num_voxel_point_features'] + occ_pnt_module.code_num_dim
        return occ_pnt_module, model_info_dict

    def build_vfe(self, model_info_dict, occ=False):
        model_cfg = self.model_cfg.OCC if occ else self.model_cfg
        if model_cfg.get('VFE', None) is None:
            return None, model_info_dict
        maxprob = False
        if not occ:
            maxprob = self.dataset.dataset_cfg.OCC.MAX_VFE if self.dataset.dataset_cfg.get('OCC', None) is not None and self.dataset.dataset_cfg.OCC.get("MAX_VFE", None) is not None else False
        vfe_module = vfe.__all__[model_cfg.VFE.NAME](
            model_cfg=model_cfg.VFE,
            num_point_features=model_info_dict['num_voxel_point_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['occ_voxel_size'] if occ else model_info_dict['det_voxel_size'],
            data_cfg=self.dataset.dataset_cfg,
            grid_size=model_info_dict['occ_grid_size'] if occ else model_info_dict['det_grid_size'],
            num_class=self.occ_num_class if occ else self.num_class,
            maxprob = maxprob
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        if occ:
            model_info_dict['occ_module_list'].append(vfe_module)
        else:
            model_info_dict['det_module_list'].append(vfe_module)
        return vfe_module, model_info_dict


    def build_backbone_3d(self, model_info_dict, occ=False):
        model_cfg = self.model_cfg.OCC if occ else self.model_cfg
        if model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict
        input_channels = model_info_dict['num_point_features']

        backbone_3d_module = backbones_3d.__all__[model_cfg.BACKBONE_3D.NAME](
            model_cfg=model_cfg.BACKBONE_3D,
            input_channels=input_channels,
            grid_size=model_info_dict['occ_grid_size'] if occ else model_info_dict['det_grid_size'],
            voxel_size=model_info_dict['occ_voxel_size'] if occ else model_info_dict['det_voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            original_num_rawpoint_features=self.dataset.point_feature_encoder.num_point_features,
        )
        if occ:
            model_info_dict['occ_module_list'].append(backbone_3d_module)
            model_info_dict['num_occ_3d_features'] = backbone_3d_module.num_point_features
        else:
            model_info_dict['det_module_list'].append(backbone_3d_module)
            model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        return backbone_3d_module, model_info_dict


    def build_map_to_bev_module(self, model_info_dict, occ=False):
        model_cfg = self.model_cfg.OCC if occ else self.model_cfg
        if model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict
        map_to_bev_module = map_to_bev.__all__[model_cfg.MAP_TO_BEV.NAME](
            model_cfg=model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['occ_grid_size'] if occ else model_info_dict['det_grid_size'],
            occ_dim = self.occ_dim
        )
        if occ:
            model_info_dict['occ_module_list'].append(map_to_bev_module)
        else:
            model_info_dict['det_module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        model_info_dict['pre_conv_num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict, occ=False):
        model_cfg = self.model_cfg.OCC if occ else self.model_cfg
        if model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[model_cfg.BACKBONE_2D.NAME](
            model_cfg=model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features'],
        )
        if occ:
            model_info_dict['occ_module_list'].append(backbone_2d_module)
        else:
            model_info_dict['det_module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['det_voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features'],
            original_num_rawpoint_features=self.dataset.point_feature_encoder.num_point_features,
            pre_conv_num_bev_features=model_info_dict['pre_conv_num_bev_features']
        )
        model_info_dict['det_module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict


    def build_occ_pfe(self, model_info_dict):
        if self.model_cfg.get('OCC_PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.OCC_PFE.NAME](
            model_cfg=self.model_cfg.OCC_PFE,
            voxel_size=model_info_dict['det_voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features'],
            original_num_rawpoint_features=self.dataset.point_feature_encoder.num_point_features,
            pre_conv_num_bev_features=model_info_dict['pre_conv_num_bev_features']
        )
        model_info_dict['det_module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict


    def build_occ_point_head(self, model_info_dict):
        if self.model_cfg.get('OCC_POINT_HEAD', None) is None:
            return None, model_info_dict

        if self.model_cfg.OCC_POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']

        point_head_module = dense_heads.__all__[self.model_cfg.OCC_POINT_HEAD.NAME](
            model_cfg=self.model_cfg.OCC_POINT_HEAD,
            input_channels=num_point_features,
            num_class=self.num_class if not self.model_cfg.OCC_POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )

        model_info_dict['det_module_list'].append(point_head_module)
        return point_head_module, model_info_dict



    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['det_grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
        )
        model_info_dict['det_module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_occ_dense_head(self, model_info_dict, occ=True):
        model_cfg = self.model_cfg.OCC
        if model_cfg.get('OCC_DENSE_HEAD', None) is None:
            return None, model_info_dict
        occ_dense_head_module = occ_pnt.occ_dense_heads.__all__[model_cfg.OCC_DENSE_HEAD.NAME](
            model_cfg=model_cfg,
            data_cfg=self.dataset.dataset_cfg,
            input_channels=model_info_dict['num_occ_3d_features'] if "num_occ_3d_features" in model_info_dict else model_info_dict['num_bev_features'],
            num_class=self.occ_num_class,
            grid_size=model_info_dict['occ_grid_size'],
        )
        model_info_dict['occ_module_list'].append(occ_dense_head_module)
        return occ_dense_head_module, model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict

        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']

        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg=self.model_cfg.POINT_HEAD,
            input_channels=num_point_features,
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )

        model_info_dict['det_module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
            det_voxel_size=model_info_dict['det_voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features'],
            pre_conv_num_bev_features=model_info_dict['pre_conv_num_bev_features'],
        )

        model_info_dict['det_module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            record_dict = {}
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1

                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict, iou3d_rcnn = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )
            if iou3d_rcnn is None or len(iou3d_rcnn) == 0:
                iou3d = None
            else:
                iou3d = torch.max(iou3d_rcnn, dim=1)[0]
            record_dict.update({
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
                'iou': iou3d.cpu().numpy() if iou3d is not None and len(iou3d) == len(final_boxes) else None
            })

            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict


    def occ_post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """

        general_cls_loss_mask = batch_dict['general_cls_loss_mask']
        total = torch.sum(general_cls_loss_mask)
        pos_mask = batch_dict["pos_mask"]
        pos_num = torch.sum(pos_mask)
        neg_mask = batch_dict["neg_mask"]
        neg_num = torch.sum(neg_mask)
        batch_pred_occ_prob = batch_dict['batch_pred_occ_prob']
        precision, recall, f1 = self.call_precision_recall_f1(pos_num, pos_mask, batch_pred_occ_prob, 0.5)
        match_dicts = {
            "pos_num": pos_num.cpu(),
            "neg_num": neg_num.cpu(),
            "pos_all_num": batch_dict["pos_all_num"].cpu(),
            "total": total.cpu(),
            "precision": precision.cpu(),
            "recall": recall.cpu(),
            "f1": f1.cpu(),
        }
        if "occ_pnts" in batch_dict:
            batch_size, added_occ_b_ind, occ_pnts, gt_boxes_num, gt_boxes = batch_dict["batch_size"], batch_dict["added_occ_b_ind"], batch_dict["occ_pnts"], batch_dict["gt_boxes_num"], batch_dict["gt_boxes"]

            box_point_mask_lst, valid_b_point_ind_lst = point_box_utils.torch_points_in_box_3d_box_label_batch(occ_pnts[..., :3], added_occ_b_ind.unsqueeze(-1), gt_boxes, gt_boxes_num, batch_size)
            occ_point_lst = [occ_pnts[vox_ind, :] for vox_ind in valid_b_point_ind_lst]
            occ_thresh_box_num = []
            box_num_sum = sum(gt_boxes_num)
            for i in range(1, 10):
                thresh = i * 0.1
                box_occ_num_sum = 0
                for occ, mask in zip(occ_point_lst, box_point_mask_lst):
                    thresh_ind = torch.nonzero(occ[..., 3] >= thresh)
                    # print("thresh_ind", thresh_ind.shape, occ.shape, mask.shape)
                    if thresh_ind.shape[0] > 0 and mask.shape[1] > 0:
                        box_labels = torch.max(mask[thresh_ind[..., 0], :], dim=0)[0]
                        box_occ_num_sum += torch.sum(box_labels).item()
                occ_thresh_box_num.append(box_occ_num_sum)
            match_dicts.update({
                "box_num_sum": box_num_sum,
                "occ_box_num": occ_thresh_box_num
            })
            # print(box_num_sum)
            # print(occ_thresh_box_num)
        return match_dicts, batch_dict

    def call_precision_recall_f1(self, pos_num, pos_mask, batch_pred_occ_prob, thresh):
        pos_predict = batch_pred_occ_prob >= thresh
        pos_predict_num = torch.sum(pos_predict)
        pos_correct = torch.sum(pos_mask & pos_predict)
        precision = pos_correct / torch.clamp(pos_predict_num, min=1.0)
        recall = pos_correct / torch.clamp(pos_num, min=1.0)
        f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-8)
        return precision, recall, f1

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        iou3d_rcnn = None
        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict, iou3d_rcnn


    def load_params_from_file(self, filename, logger, to_cpu=False, verbose=True, prefix=""):
        assert os.path.isfile(filename), "wrong file path: {}".format(filename)
            
        logger.info('==> Loading %s parameters from checkpoint %s to %s' % (prefix, filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']


        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape and key.startswith(prefix):
                update_model_state[key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                if verbose:
                    logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        if verbose:
            logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        logger.info('==> Done')

        return it, epoch


    def load_params_with_optimizer_lst(self, filename, to_cpu=False, optimizer_lst=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer_lst is not None:
            if 'optimizer_state_lst' in checkpoint and checkpoint['optimizer_state_lst'] is not None:
                logger.info('==> Loading optimizer_lst parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                for i in range(len(optimizer_lst)):
                    optimizer_lst[i].load_state_dict(checkpoint['optimizer_state_lst'][i])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        logger.info('==> Done')

        return it, epoch
