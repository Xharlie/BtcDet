import copy
import pickle
import sys
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from skimage import io
import mayavi.mlab as mlab
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from ..ops.roiaware_pool3d import roiaware_pool3d_utils
from ..utils import box_utils, calibration_kitti, common_utils, object3d_kitti, point_box_utils
from .dataset import DatasetTemplate
import torch
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
sys.path.append('/home/xharlie/dev/match2det/tools/visual_utils')
import visualize_utils as vu
from PIL import ImageColor
from ..ops.chamfer_distance import ChamferDistance
from ..ops.iou3d_nms import iou3d_nms_utils
chamfer_dist = ChamferDistance()


NUM_POINT_FEATURES = 4
def extract_allpnts(root_path=None, splits=['train','val']):
    all_db_infos_lst = []
    box_dims_lst = []
    pnts_lst = []
    mirrored_pnts_lst = []
    for split in splits:
        db_info_save_path = Path(root_path) / ('kitti_dbinfos_%s.pkl' % split)
        with open(db_info_save_path, 'rb') as f:
            all_db_infos = pickle.load(f)['Car']
        for k in range(len(all_db_infos)):
            info = all_db_infos[k]
            obj_type = info['name']
            if obj_type != "Car":
                continue
            gt_box = info['box3d_lidar']
            all_db_infos_lst.append(info)
    return all_db_infos_lst



if __name__ == '__main__':
    PNT_THRESH = 400
    ROOT_DIR = (Path(__file__).resolve().parent / '../../').resolve()
    print("ROOT_DIR", ROOT_DIR)
    path = ROOT_DIR / 'data' / 'kitti' / 'detection3d'
    match_info_save_path = path / "match_maxdist_10extcrdsnum_info_car.pkl"
    cluster_num = 20
    voxel_size = [0.08, 0.08, 0.08]
    all_db_infos_lst = extract_allpnts(
        root_path=path, splits=['train','val']
    )
    range_all = np.zeros([18, 3])
    x_all = np.zeros([18, 3])
    y_all = np.zeros([18, 3])
    diff_count = np.array([0,0,0])
    diff_dist = np.array([0,0,0])

    # db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
    #            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
    #            'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}

    for info in all_db_infos_lst:
        diff = info['difficulty']
        if diff > -1:
            box = info['box3d_lidar']
            dist = np.linalg.norm(box[:3])
            ind = int(dist/5)
            xind = int(box[0]/5)
            yind = int((box[1] + 40) / 5)
            # print("diff", diff, "dist", dist, "ind", ind)
            range_all[ind, diff] = range_all[ind, diff] + 1
            x_all[xind, diff] = x_all[xind, diff] + 1
            y_all[yind, diff] = y_all[yind, diff] + 1
            diff_count[diff] += 1
            diff_dist[diff] += dist

    print("avg: ", diff_dist/diff_count)
    print("breakdown: ", range_all)
    print("x breakdown: ", x_all)
    print("y breakdown: ", y_all)




