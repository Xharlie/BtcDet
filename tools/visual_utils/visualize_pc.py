import mayavi.mlab as mlab
import numpy as np
import torch
import visualize_utils as vu
import argparse
import os, sys
import io
sys.path.append('../../btcdet/datasets/waymo/')
import pickle
# from waymo_utils import *
# from waymo_open_dataset import dataset_pb2 as open_dataset
# from waymo_open_dataset.utils import frame_utils
sys.path.append('../../btcdet/')
from utils import coords_utils

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# import tensorflow.compat.v1 as tf
# tf.enable_eager_execution()

# RGB
clrs = {
    'gt_points': (0.5, 0.5, 0.5),
    'bm_points': (1.0, 0.2, 0.2),
    'forepnts': (1.0, 0.2, 0.2),
    'voxels': (1, 0.8, 1),
    'raw_points': (0.5, 0.5, 0.5),
    'combined_cloud_pnts': (1, 1, 0.1),
    'fore_gt_center': (1, 0.5, 0.5),
    'occ_center': (1, 0.5, 0.5),
    'bmvoxel_center': (1, 0.5, 0.5),
    'ocp_center': (1, 0.5, 0.5),
    'general_cls_loss_center': (120/255,200/255,200/255),
    'filter_center': (0.8, 0.8, 0),
    'boxvoxel_center': (1, 0.5, 0),
    'addpnt_view': (0.2, 1, 0.2),
    'proboccpoints': (1.0, 0.2, 0.2),
    'vis_map': (0.2, 1, 0.2),
    'occ_map': (1.0, 0.2, 0.2),
    'vicinity_map': (0.2, 0.6, 0.8),
    'drop_voxel_center': (0.8, 0, 0.8),
    'drop_det_voxel_point': (0.4, 0, 0.7),
    'drop_det_point_xyz': (0.5, 0, 0.6),

    "occ_fore_center": (1, 0.5, 0.5),
    "occ_mirr_center": (0.8, 0.5, 0),
    "occ_bm_center": (0.8, 0.8, 0),
    "occ_pos_center": (1.0, 0.4, 0), # (1.0, 0.2, 0.2), # ,
    "occ_neg_center": (0.2, 0.6, 0.8),

    "rois_raw_points_0": (0.2, 1, 0.2),
    "rois_raw_rot_points_0": (1.0, 0.2, 0.2),
    "rois_occ_points_0": (0.4, 0, 0.7),
    "rois_occ_rot_points_0": (0.3, 0, 0.8),

    "rois_global_grid_points": (0.3, 0, 0.8),
    "rois_conv_grid_points": (0.2, 0.6, 0.8),
    "rois_sparse_grid": (0.5, 0, 0.6),

    "btc_miss_points": (0, 165/255, 1.0), # (0.2, 0.6, 0.8),
    "btc_self_points": (123/255, 179/255, 46/255), # (1, 0.8, 0.1),
    "btc_other_points": (1.0, 0.2, 0.2),

    "btc_miss_voxelpoints": (0.2, 0.6, 0.8),
    "btc_miss_full_voxelpoints": (0.2, 0.6, 0.8),
    "btc_self_voxelpoints": (152/255, 251/255, 152/255), # (1, 0.8, 0.1),
    "btc_other_voxelpoints": (1.0, 0.2, 0.2),
    "btc_other_full_voxelpoints": (1.0, 0.2, 0.2),
    "btc_self_limit_voxelpoints": (152/255, 251/255, 152/255) #, (1, 0.8, 0.1),
}

opacities = {
    "btc_miss_points": 0.2,
    "btc_self_points": 0.3,
    "btc_other_points": 0.2,
    "occ_pos_center": 0.1,

    "btc_miss_voxelpoints": 0.01,
    "btc_miss_full_voxelpoints": 0.01,
    "btc_self_voxelpoints": 0.01,
    "btc_other_voxelpoints": 0.01,
    "btc_other_full_voxelpoints": 0.01,
    "btc_self_limit_voxelpoints": 0.01,
    "general_cls_loss_center": 0.03,
}

scales = {
    'gt_points': .1, #0.05
    'bm_points': .08,
    'forepnts': .08,
    'voxels': .05,
    'raw_points': .05,
    'combined_cloud_pnts': .05,
    'fore_gt_center': .1,
    'occ_center': .1,
    'bmvoxel_center': .1,
    'ocp_center': .1,
    'general_cls_loss_center': .1,
    'filter_center': .1,
    'boxvoxel_center': .1,
    'addpnt_view': .15,
    'proboccpoints': .05,
    'vis_map': .02,
    'occ_map': .04,
    'vicinity_map': .04,
    'drop_voxel_center': .1,
    'drop_det_voxel_point': .1,
    'drop_det_point_xyz': .1,
    "occ_fore_center": .1,
    "occ_mirr_center": .1,
    "occ_bm_center": .1,
    "occ_pos_center": .1,
    "occ_neg_center": .1,

    "rois_raw_points_0": .1,
    "rois_raw_rot_points_0": .1,
    "rois_occ_points_0": .1,
    "rois_occ_rot_points_0": .1,

    "rois_global_grid_points": .1,
    "rois_conv_grid_points": .1,
    "rois_sparse_grid": .1,

    "btc_miss_points": .08,
    "btc_self_points": .08,
    "btc_other_points": .08,
}

modes = {
    'gt_points': "sphere",
    'bm_points': "sphere",
    'forepnts': "sphere",
    'voxels': "sphere",
    'raw_points': "sphere",
    'combined_cloud_pnts': "sphere",
    'fore_gt_center': "sphere",
    'occ_center': "sphere",
    'bmvoxel_center': "sphere",
    'ocp_center': "sphere",
    'general_cls_loss_center': "sphere",
    'filter_center': "sphere",
    'boxvoxel_center': "sphere",
    'addpnt_view': "sphere",
    'proboccpoints': "sphere",
    'vis_map': "sphere",
    'occ_map': "sphere",
    'vicinity_map': "sphere",
    'drop_voxel_center': "sphere",
    'drop_det_voxel_point': "sphere",
    'drop_det_point_xyz': "sphere",
    "occ_fore_center": "sphere",
    "occ_mirr_center": "sphere",
    "occ_bm_center": "sphere",
    "occ_pos_center": "sphere",
    "occ_neg_center": "sphere",

    "rois_raw_points_0": "sphere",
    "rois_raw_rot_points_0": "sphere",
    "rois_occ_points_0": "sphere",
    "rois_occ_rot_points_0": "sphere",

    "rois_global_grid_points": "sphere",
    "rois_conv_grid_points": "sphere",
    "rois_sparse_grid": "sphere",

    "btc_miss_points": "sphere",
    "btc_self_points": "sphere",
    "btc_other_points": "sphere",
}


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pc', type=str, default=None, help='specify the config for training')
    parser.add_argument('--gt_pnt', type=str, default=None, help='od_val_+frame_id')

    args = parser.parse_args()

    return args

def main():
    voxel_point_key = None
    args = parse_config()
    # if args.pc is not None:
    dict = np.load(args.pc, allow_pickle=True).item()
    # dict = loads(args.pc).item()
    print(dict.keys())
    keys=[]
    voxel_point_keys = []
    # frame_id 006449

    # dict_keys(['gt_points', 'fore_gt_center', 'filter_center', 'boxvoxel_center', 'addpnt_view', 'drop_voxel_center','gt_boxes', 'pred_boxes', 'pred_scores', 'pred_labels', 'proboccpoints', 'ocp_center'])

    vu.visualize_pts(dict['gt_points'])
    # keys = ["gt_points",'occ_fore_center', 'occ_mirr_center', 'occ_bm_center']
    # keys = ["gt_points",'btc_miss_points', 'btc_self_points', 'btc_other_points']
    keys = ["gt_points"]
    # keys = ["gt_points", 'occ_center']
    # keys = ["gt_points", "general_cls_loss_center"] # gt_points
    # keys = ["gt_points", "fore_gt_center"] # gt_points
    # keys = ["gt_points", "bmvoxel_center"] # gt_points

    voxel_point_keys = ["occ_pos_center"] # gt_points
    # voxel_point_keys = ["general_cls_loss_center", "occ_pos_center"] # gt_points
    # voxel_point_keys = ["addpnt_view"] # gt_points

    # keys = ["gt_points", 'btc_other_points']
    # voxel_point_keys = ['btc_other_voxelpoints']

    # keys = ["gt_points", 'btc_miss_points']
    # voxel_point_keys = ['btc_miss_voxelpoints']

    # keys = ["gt_points", 'btc_miss_points', 'btc_self_points', 'btc_other_points']
    # voxel_point_keys = ['btc_self_voxelpoints']

    # keys = ["gt_points", "rois_global_grid_points", "rois_conv_grid_points"] # gt_points
    # keys = ["gt_points", "rois_raw_rot_points_0", "rois_raw_points_0", "rois_global_grid_points"] # gt_points

    if "occ_map" in keys:
        voxel_centers = coords_utils.creat_grid_coords([0, -39.68, -1.5, 69.12, 39.68, 1], [0.16, 0.16, 0.25])
        print("voxel_centers", voxel_centers.shape)
        occ_map = dict["occ_map"].data.cpu().numpy().astype(np.bool)
        if "vicinity_map" in keys:
            vicinity_map = dict["vicinity_map"].data.cpu().numpy().astype(np.int)
            occ_map = np.logical_and(occ_map, vicinity_map > 0.5)
            print(np.sum(occ_map > -5.5), np.sum(occ_map > 0.5))

            keys.remove("vicinity_map")
        x,y,z = np.nonzero(occ_map > 0.5)
        print(x.shape,y.shape,z.shape)
        print("occ_map", occ_map.shape, occ_map[0, 0, 0])
        # dict["vis_map"] = voxel_centers.reshape(-1,3)
        dict["occ_map"] = voxel_centers[x,y,z,:]
    elif "vicinity_map" in keys:
        voxel_centers = coords_utils.creat_grid_coords([0, -39.68, -1.4, 69.12, 39.68, 1], [0.16, 0.16, 0.16])
        vicinity_map = dict["vicinity_map"].data.cpu().numpy().astype(np.int) > 0.5
        x, y, z = np.nonzero(vicinity_map > 0.5)
        print(x.shape, y.shape, z.shape)
        print("vicinity_map", vicinity_map.shape, vicinity_map[0, 0, 0])
        print(np.sum(vicinity_map > -5.5), np.sum(vicinity_map > 0.5))
        dict["vicinity_map"] = voxel_centers[x, y, z, :]

    points_lst = [dict[key] for key in keys]
    colors_lst = [clrs[key] for key in keys]
    scales_lst = [scales[key] for key in keys]
    mode_lst = [modes[key] for key in keys]

    voxelpnts_lst = [dict[key] for key in voxel_point_keys]
    voxelpnts_colors_lst = [clrs[key] for key in voxel_point_keys]
    voxelpnts_op_lst = [opacities[key] for key in voxel_point_keys]

    aug_boxes = None
    gt_boxes = dict["gt_boxes"] if "gt_boxes" in dict else None
    if "frame_id" in dict:
        print("frame_id", dict["frame_id"])
    if "augment_box_num" in dict and gt_boxes is not None:
        if dict['augment_box_num'][0] != 0:
            aug_boxes = gt_boxes[-dict['augment_box_num'][0]:,...]
            gt_boxes = gt_boxes[:-dict['augment_box_num'][0],...]
            print("aug_boxes ", aug_boxes.shape)
        print("gt_boxes ", gt_boxes.shape)
    #                 'pred_boxes': final_boxes,
    #                 'pred_scores': final_scores,
    #                 'pred_labels': final_labels
    ref_boxes = dict["pred_boxes"] if "pred_boxes" in dict else None
    ref_scores = dict["pred_scores"] if "pred_scores" in dict else None
    ref_labels = dict["pred_labels"] if "pred_labels" in dict else None

    for i in range(len(keys)):
        if keys[i] =="bm_points":
            points_lst[i] += np.array([0.,0., 3.0])

    # ref_boxes = dict["afternms_pred_boxes"] if "afternms_pred_boxes" in dict else None
    # ref_scores = dict["afternms_pred_scores"] if "afternms_pred_scores" in dict else None
    # ref_labels = dict["afternms_pred_labels"] if "afternms_pred_labels" in dict else None

    # ref_boxes = dict["prenms_pred_boxes"] if "prenms_pred_boxes" in dict else None
    # ref_scores = dict["prenms_pred_scores"] if "prenms_pred_scores" in dict else None
    # ref_labels = dict["prenms_pred_labels"] if "prenms_pred_labels" in dict else None
    #
    # ref_boxes = dict["rois"] if "rois" in dict else None
    # ref_scores = dict["roi_scores"] if "roi_scores" in dict else None
    # ref_labels = dict["roi_labels"] if "roi_labels" in dict else None

    # ref_boxes = dict["anchors"].view(-1,7).cpu().numpy() if "anchors" in dict else None
    # ref_scores = dict["roi_scores"] if "roi_scores" in dict else None
    # ref_labels = np.ones_like(ref_boxes[...,0], dtype=np.int)
    ref_ious = dict["iou"] if "iou" in dict else None
    vu.draw_scenes_multi(points_lst, colors_lst, scales_lst, mode_lst, gt_boxes=None, aug_boxes=aug_boxes, ref_boxes=ref_boxes, ref_scores=ref_scores, ref_labels=ref_labels, ref_ious=ref_ious, voxelpnts_lst=voxelpnts_lst, voxelpnts_colors_lst=voxelpnts_colors_lst, voxelpnts_op_lst=voxelpnts_op_lst, axis=False)

    mlab.view(azimuth=159, elevation=75.0, distance=104.0, roll=93.5)
    # points_lst[1] = points_lst[1][-27:,:]
    # points_lst[2] = points_lst[2][-27*3*4*6:,:]
    # points_lst[3] = points_lst[3][-1:,:]
    # vu.draw_scenes_multi(points_lst, colors_lst, scales_lst, mode_lst, gt_boxes=gt_boxes, aug_boxes=aug_boxes, ref_boxes=dict["rois"][0,-1:,...], ref_scores=None, ref_labels=None, ref_ious=None)
        # mlab.show()
    #############################################################
    # if args.gt_pnt is not None:
    #     component = args.gt_pnt.split("_")
    #     set_len = len(component[0]) + len(component[1]) + 1
    #     dataset = args.gt_pnt[:set_len]
    #     frame_id = args.gt_pnt[set_len+1:]
    #     #
    #     # points, boxes = get_single_frame_tfrcd(dataset, frame_id)
    #     # bbox = torch.tensor([box['box_3d'] for box in boxes["boxes"]])
    #     # bbox_tp = torch.tensor([box['obj_type'] for box in boxes["boxes"]])
    #     # bbox_num_point = torch.tensor([box['num_pnts_inbox'] for box in boxes["boxes"]])
    #     # print(len(boxes["boxes"]), boxes["top_lidar_points_num"])
    #     # print(points.shape)
    #     # key = "raw_points"
    #     # points_lst = [points]
    #     # colors_lst = [clrs[key]]
    #     # scales_lst = [scales[key]]
    #     # mode_lst = [modes[key]]
    #     # vu.draw_scenes_multi(points_lst, colors_lst, scales_lst, mode_lst, gt_boxes=None, aug_boxes=None,
    #     #                      ref_boxes=bbox, ref_scores=bbox_num_point, ref_labels=bbox_tp)
    #     # mlab.show()
    #
    #
    #
    #
    #     lidar_file = os.path.join("/home/xharlie/dev/openbtcdet/data/waymo/",dataset, "points", frame_id+".bin")
    #     box_file = os.path.join("/home/xharlie/dev/openbtcdet/data/waymo/",dataset, "boxes", frame_id+".pkl")
    #     waymo_infos_file = os.path.join("/home/xharlie/dev/openbtcdet/data/waymo/", "waymo_infos_{}.pkl".format(dataset))
    #     info_file = os.path.join("/home/xharlie/dev/openbtcdet/data/waymo/",dataset, "frames", frame_id+".pkl")
    #     points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1,5)
    #     valid_boxes, invalid_boxes, num_valid, real_num_valid = get_boxes_from_waymoinfosfile(waymo_infos_file, frame_id)
    #     info = np.load(info_file, allow_pickle=True)
    #     # bbox = torch.tensor([box['box_3d'] for box in boxes["boxes"]])
    #     # bbox_tp = torch.tensor([box['obj_type'] for box in boxes["boxes"]])
    #     # bbox_num_point = torch.tensor([box['num_pnts_inbox'] for box in boxes["boxes"]])
    #     bbox = torch.tensor(valid_boxes["box_3d"])
    #     bbox_tp = torch.tensor(valid_boxes["obj_type"])
    #     bbox_num_point = torch.tensor(valid_boxes["num_pnts_inbox"])
    #     bbox_real_num_point = torch.tensor(valid_boxes["real_num_pnts_inbox"])
    #
    #     print(points.shape)
    #     print(info)
    #     key = "raw_points"
    #     points_lst = [points]
    #     colors_lst = [clrs[key]]
    #     scales_lst = [scales[key]]
    #     mode_lst = [modes[key]]

        # vu.draw_scenes_multi(points_lst, colors_lst, scales_lst, mode_lst, gt_boxes=invalid_boxes["box_3d"], aug_boxes=None, ref_boxes=bbox, ref_scores=bbox_real_num_point, ref_labels=bbox_tp)
    mlab.show()



def get_boxes_from_waymoinfosfile(src_file, frame_id):
    with open(src_file, "rb") as f:
        infos = pickle.load(f)
    for info in infos:
        sample_idx, top_lidar_points_num, is_od = info['point_cloud']['lidar_idx'], info['point_cloud'][
            'top_lidar_points_num'], info['point_cloud']['is_od']
        if sample_idx != frame_id:
            continue
        valid_boxes = {"box_3d":[], "obj_type":[]}
        invalid_boxes = {"boxes":[]}
        if 'annos' in info:
            annos = info['annos']
            num_valid = annos['num_valid']
            real_num_valid = annos['real_num_valid']
            if num_valid > 0:
                if real_num_valid > 0:
                    valid_boxes["box_3d"] = annos['gt_boxes_lidar'][:real_num_valid, ...]
                    valid_boxes["obj_type"] = annos['type'][:real_num_valid, ...]
                    valid_boxes["num_pnts_inbox"] = annos['num_pnts_inbox'][:real_num_valid, ...]
                    valid_boxes["real_num_pnts_inbox"] = annos['real_num_pnts_inbox'][:real_num_valid, ...]

                if num_valid - real_num_valid > 0:
                    invalid_boxes["box_3d"] = annos['gt_boxes_lidar'][real_num_valid:, ...]
                    invalid_boxes["obj_type"] = annos['type'][real_num_valid:, ...]
                    invalid_boxes["num_pnts_inbox"] = annos['num_pnts_inbox'][real_num_valid:, ...]
                    invalid_boxes["real_num_pnts_inbox"] = annos['real_num_pnts_inbox'][real_num_valid:, ...]
            return valid_boxes, invalid_boxes, num_valid, real_num_valid

def get_single_frame_tfrcd(dataset, frame_id):
    time_id = frame_id.split("_")[-1]
    time_id_len = len(time_id) + 1
    src_path = os.path.join("/hdd_extra1/datasets/waymo/", dataset, "segment-"+frame_id[:-time_id_len]+".tfrecord")
    time_id=int(time_id)
    print("starting new tfrcord {}".format(src_path))
    dataset = tf.data.TFRecordDataset(src_path, compression_type='')
    indx = 0
    for data in dataset:
        if indx == time_id:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            points, boxes = trans_single_frame_custom(frame, pose_name=None) #
            return points, boxes
        else:
            indx+=1
            print("indx", indx)



def trans_single_frame_custom(frame, pose_name=None):
    # if pose_name is not None:
    #     range_images, range_image_poses = parse_range_image_one_return_one_pose(frame, pose_name)
    # else:
    #     range_images, range_image_poses = parse_range_image_one_return_all_pose(frame)
    #
    # points, pose_names, top_index = convert_range_image_to_point_cloud(
    #     frame,
    #     range_images,
    #     range_image_poses)
    # points[0], points[top_index] = points[top_index], points[0]
    # pose_names[0], pose_names[top_index] = pose_names[top_index], pose_names[0]
    # top_lidar_points_num = points[0].shape[0]
    # points_all = np.concatenate(points, axis=0)
    (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
    points, _ = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose, 0)
    points = np.concatenate(points, axis=0)
    top_lidar_points_num = points[0].shape[0]

    box_point_info = parse_box_labels(frame, top_lidar_points_num)
    print("tf top_lidar_points_num", top_lidar_points_num)
    return points, box_point_info


# class MappedUnpickler(pickle.Unpickler):
#     def __init__(self, *args, map_location='cpu', **kwargs):
#         self._map_location = map_location
#         super().__init__(*args, **kwargs)
#
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(BytesIO(b), map_location=self._map_location)
#         else:
#             return super().find_class(module, name)
#
# def mapped_loads(s, map_location='cpu'):
#     bs = BytesIO(s)
#     unpickler = MappedUnpickler(bs, map_location==map_location)
#     return unpickler.load()
#
#
# def loads(x):
#     bs = io.BytesIO(x)
#     unpickler = MappedUnpickler(bs)
#     return unpickler.load()

if __name__ == '__main__':
    main()
