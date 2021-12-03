from .detector3d_template import Detector3DTemplate
import torch
import numpy as np

EVL_VIS=800
class BtcNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, full_config=None):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, full_config=full_config)
        self.occ_module_list, self.det_module_list = self.build_networks()
        self.eval_count = -1
        self.occ_nograd = True if full_config is not None and full_config.OCC_OPTIMIZATION.LR == 0.0 else False
        self.det_nograd = True if full_config is not None and full_config.OPTIMIZATION.LR == 0.0 else False
        self.percentage = full_config.DATA_CONFIG.OCC.USEOCC_PERCENTAGE if full_config is not None and full_config.DATA_CONFIG.get("OCC", None) is not None and full_config.DATA_CONFIG.OCC.get("USEOCC_PERCENTAGE", None) is not None else 1.0
        print("self.clamp_max", self.clamp_max)

    def clamp(self, batch_dict):
        if self.clamp_max == "tanh":
            batch_dict["points"][..., 4] = torch.tanh(batch_dict["points"][..., 4])
            if "voxels" in batch_dict:
                batch_dict["voxels"][..., 3] = torch.tanh(batch_dict["voxels"][..., 3])
            if "det_voxels" in batch_dict:
                batch_dict["det_voxels"][..., 3] = torch.tanh(batch_dict["det_voxels"][..., 3])
            # print("use tanh")
        elif self.clamp_max > 0.0:
            batch_dict["points"][..., 4] = torch.clamp(batch_dict["points"][..., 4], min=0.0, max=self.clamp_max)
            if "voxels" in batch_dict:
                batch_dict["voxels"][..., 3] = torch.clamp(batch_dict["voxels"][..., 3], min=0.0, max=self.clamp_max)
            if "det_voxels" in batch_dict:
                batch_dict["det_voxels"][..., 3] = torch.clamp(batch_dict["det_voxels"][..., 3], min=0.0, max=self.clamp_max)
        return batch_dict

    def forward(self, batch_dict):
        bind = 0
        if self.clamp_max is not None:
            batch_dict = self.clamp(batch_dict)

        use_occ_prob = [True for i in range(batch_dict["batch_size"])]  # False
        prob = np.random.uniform(size=batch_dict["batch_size"], high=0.9999)
        if batch_dict["is_train"]:
            use_occ_prob = prob <= self.percentage
        batch_dict["use_occ_prob"] = use_occ_prob
        if self.occ_nograd:
            with torch.no_grad():
                for cur_module in self.occ_module_list:
                    batch_dict = cur_module(batch_dict)
        else:
            for cur_module in self.occ_module_list:
                batch_dict = cur_module(batch_dict)

        if self.det_nograd:
            with torch.no_grad():
                for cur_module in self.det_module_list:
                    batch_dict = cur_module(batch_dict)
        else:
            for cur_module in self.det_module_list:
                batch_dict = cur_module(batch_dict)

        if not batch_dict["is_train"]:
            self.eval_count+=1
        tb_dict, pc_dict = self.get_vis(batch_dict, bind)

        if self.training:
            loss, det_tb_dict, disp_dict = self.get_training_loss(batch_dict)
            ret_dict = {
                'loss': loss
            }
            tb_dict.update(det_tb_dict)
            return ret_dict, tb_dict, disp_dict, pc_dict
        else:
            metric_dicts = {}
            if hasattr(self.model_cfg, "OCC") and hasattr(self.model_cfg.OCC, 'OCC_POST_PROCESSING'):
                occ_dicts, batch_dict = self.occ_post_processing(batch_dict)
                metric_dicts.update(occ_dicts)
            if hasattr(self.model_cfg, 'POST_PROCESSING'):
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                if self.model_cfg.get('OCC', None) is not None and self.eval_count % self.model_cfg.OCC.OCC_PNT_UPDATE.VIS.STEP_STRIDE == 0:
                    pc_dict.update(pred_dicts[bind])
                elif self.model_cfg.get('OCC', None) is None and self.eval_count % EVL_VIS == 0:
                    gt_points = self.filter_by_bind(batch_dict["points"][..., 0], bind, batch_dict["points"][..., 1:4])
                    pc_dict.update(pred_dicts[bind])
                    pc_dict.update({
                        "gt_points": gt_points,
                        "gt_boxes": batch_dict["gt_boxes"][bind, :batch_dict["gt_boxes_num"][bind], ...]
                    })
                metric_dicts.update(recall_dicts)
                return pred_dicts, metric_dicts, tb_dict, pc_dict
            else:
                return {'loss': torch.zeros(1)}, metric_dicts, {}, pc_dict


    def get_training_loss(self, batch_dict, no_occ=False):
        disp_dict = {}
        tb_dict = {}
        loss, det_loss_rpn, det_loss_point, det_loss_rcnn = 0,0,0,0
        if hasattr(self.occ_modules, 'occ_dense_head') and not no_occ:
            occ_loss_rpn, occ_tb_scalar_dict = self.occ_modules.occ_dense_head.get_loss(batch_dict)
            tb_dict.update({
                'loss_occ': occ_loss_rpn.item(),
                **occ_tb_scalar_dict
            })
            loss += occ_loss_rpn
            # print("loss", loss)
            
        if hasattr(self.det_modules, 'dense_head'):
            det_loss_rpn, det_tb_dict = self.det_modules.dense_head.get_loss()
            det_loss_rcnn, det_tb_dict = self.det_modules.roi_head.get_loss(det_tb_dict)
            tb_dict.update({
                'loss_rpn': det_loss_rpn.item(),
                **det_tb_dict
            })
        elif hasattr(self.det_modules, 'occ_point_head'):
            det_loss_rpn, det_tb_dict = self.det_modules.occ_point_head.get_loss()
            det_loss_rcnn, det_tb_dict = self.det_modules.roi_head.get_loss(det_tb_dict)
            tb_dict.update({
                'loss_rpn': det_loss_rpn.item(),
                **det_tb_dict
            })
            if hasattr(self.det_modules, 'point_head'):
                det_loss_point, det_tb_dict = self.det_modules.point_head.get_loss(det_tb_dict)
                tb_dict.update({
                    **det_tb_dict
                })
            # print("det loss", det_loss_rcnn)
        # else:
        #     print("SPG only!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("loss {} det_loss_rpn {} det_loss_point {} det_loss_rcnn {}".format(loss, det_loss_rpn, det_loss_point, det_loss_rcnn))
        loss = loss + det_loss_rpn + det_loss_point + det_loss_rcnn

        return loss, tb_dict, disp_dict


    def voxelpoints_filter_by_bind(self, voxel_coords, bind, voxels):
        repeat = list(voxels.shape)[1] if voxels.dim == 3 else 1
        real_b =  voxel_coords[..., :1].repeat(1,repeat)
        voxels_pnts = self.filter_by_bind(real_b.view(-1), bind, voxels)
        return voxels_pnts

    def get_vis(self, batch_dict, bind, no_occ=False):
        tb_dict = {}
        pc_dict = {}
        if no_occ:
            return tb_dict, pc_dict
        if self.model_cfg.get('OCC', None) is not None and self.model_cfg.OCC.get('OCC_PNT_UPDATE', None) is not None and (self.global_step+1) % self.model_cfg.OCC.OCC_PNT_UPDATE.VIS.STEP_STRIDE == 0 or self.model_cfg.get('OCC', None) is None and (self.global_step+1) % 1000 == 0 or self.eval_count % EVL_VIS == 0:
            raw_points = self.filter_by_bind(batch_dict["points"][..., 0], bind, batch_dict["points"][..., 1:4])
            voxels_pnts = self.voxelpoints_filter_by_bind(batch_dict['voxel_coords'], bind,  batch_dict['voxels']) # batch_dict['voxels'])
            voxels_pnts = voxels_pnts[voxels_pnts[...,-1]>0,:]
            # print("voxels_pnts", voxels_pnts.shape, batch_dict['voxel_features'].shape)
            pc_dict.update({
                "raw_points": raw_points,
                "gt_boxes": batch_dict["gt_boxes"][bind, :batch_dict["gt_boxes_num"][bind], ...],
                "augment_box_num": batch_dict["augment_box_num"],
                "frame_id": batch_dict["frame_id"],
                "voxels": voxels_pnts,
                "btc_miss_points": batch_dict["miss_points"][bind] if "miss_points" in batch_dict else None,
                "btc_self_points": batch_dict["self_points"][bind] if "self_points" in batch_dict else None,
                "btc_other_points": batch_dict["other_points"][bind] if "other_points" in batch_dict else None,
                "btc_miss_voxelpoints": batch_dict["miss_occ_points"][bind] if "miss_occ_points" in batch_dict else None,
                "btc_miss_full_voxelpoints": batch_dict["miss_full_occ_points"][bind] if "miss_full_occ_points" in batch_dict else None,
                "btc_self_voxelpoints": batch_dict["self_occ_points"][bind] if "self_occ_points" in batch_dict else None,
                "btc_self_limit_voxelpoints": batch_dict["self_limit_occ_mask"][bind] if "self_limit_occ_mask" in batch_dict else None,
                "btc_other_voxelpoints": batch_dict["other_occ_points"][bind] if "other_occ_points" in batch_dict else None,
                "btc_other_full_voxelpoints": batch_dict["other_full_occ_points"][bind] if "other_full_occ_points" in batch_dict else None,
            })

        if self.model_cfg.get('OCC', None) is not None and self.model_cfg.OCC.get('OCC_PNT_UPDATE', None) is not None and ((self.global_step+1) % self.model_cfg.OCC.OCC_PNT_UPDATE.VIS.STEP_STRIDE == 0 or self.eval_count % EVL_VIS == 0):
            occ_tb, occ_pc = self.occ_modules.occ_pnt_update.visualize(batch_dict, bind)
            tb_dict.update(occ_tb)
            pc_dict.update(occ_pc)
        if "conv_vis_dict" in batch_dict:
            pc_dict.update(batch_dict["conv_vis_dict"])
            batch_dict.pop("conv_vis_dict")
        return tb_dict, pc_dict