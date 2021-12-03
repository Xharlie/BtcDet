import pickle
import sys
import numpy as np
import copy
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, coords_utils, point_box_utils

class SupGTSampling(object):
    def __init__(self, root_path, sampler_cfg, class_names, db_infos, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = db_infos

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        self.det_height_shift = sampler_cfg.get('DETECT_HEIGHT_SHIFT', 0.0)
        self.no_stucking = sampler_cfg.get('NO_STUCKING', False)
        self.drop_rate = sampler_cfg.get('DROP_RATE', 0.0)

        self.mlt_bm_root = self.root_path.resolve() / sampler_cfg.get('MLT_BM_ROOT', None)
        self.bm_num_point_features = sampler_cfg.get('BM_NUM_POINT_FEATURES', 3)
        self.num_point_features = sampler_cfg.get('NUM_POINT_FEATURES', 4)

        if sampler_cfg.get('GT_SMP', None) is not None:
            self.gt_smp_cfg ={
                "sample_groups": {},
                "box_range_jitter": sampler_cfg.GT_SMP.get('BOX_RANGE_JITTER', None),
                "box_rot_jitter": sampler_cfg.GT_SMP.get('BOX_ROT_JITTER', None),
                "box_yaw_jitter": sampler_cfg.GT_SMP.get('BOX_YAW_JITTER', None),
                "yaw_type": sampler_cfg.GT_SMP.get('YAW_TYPE', None),
                "remove_yz_expansion": sampler_cfg.GT_SMP.get('RMV_YZ_EXPSN', 0),
                "dp_rate": sampler_cfg.GT_SMP.get('DROP_RATE', 0),
            }
            for x in sampler_cfg.GT_SMP.SAMPLE_GROUPS:
                class_name, sample_num = x.split(':')
                if class_name not in class_names:
                    continue
                self.gt_smp_cfg["sample_groups"][class_name] = int(sample_num)

        if sampler_cfg.get('MLT_BM', None) is not None:
            self.mlt_bm_cfg ={
                "sample_groups": {},
                "box_range_jitter": sampler_cfg.MLT_BM.get('BOX_RANGE_JITTER', None),
                "box_rot_jitter": sampler_cfg.MLT_BM.get('BOX_ROT_JITTER', None),
                "box_yaw_jitter": sampler_cfg.MLT_BM.get('BOX_YAW_JITTER', None),
                "yaw_type": sampler_cfg.MLT_BM.get('YAW_TYPE', None),
                "remove_yz_expansion": sampler_cfg.MLT_BM.get('RMV_YZ_EXPSN', 0),
                "dp_rate": sampler_cfg.MLT_BM.get('DROP_RATE', 0),
            }
            for x in sampler_cfg.MLT_BM.SAMPLE_GROUPS:
                class_name, sample_num = x.split(':')
                if class_name not in class_names:
                    continue
                self.mlt_bm_cfg["sample_groups"][class_name] = int(sample_num)


        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

        # self.sphere_coords_min = np.asarray([[0, -42.0, -16.9]])
        self.sphere_coords_res = np.asarray([[0.2, 0.0875*2, 0.4203125]])

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos
        # print("!!!!!!!sys.getsizeof, new_db_infos", get_size(1), get_size(db_infos))
        return db_infos


    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict


    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height


    def add_sampled_boxes_best_match_points_to_scene(self, data_dict, sampled_gt_boxes, boxes_oriyaw, total_valid_sampled_dict, aug_boxes_image_idx, aug_boxes_gt_idx, valid_boxes_type):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        data_dict["gt_boxes_inds"] = data_dict["gt_boxes_inds"][gt_boxes_mask]
        points = data_dict['points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')
        gt_smp_points_list = []
        gt_smp_bm_points_list = []
        bm_points_list = []
        final_bm_points_list = []
        final_total_valid_sampled_dict = []
        final_sampled_gt_boxes = []
        final_aug_boxes_image_idx = []
        final_aug_boxes_gt_idx = []
        for idx, info in enumerate(total_valid_sampled_dict):
            type = valid_boxes_type[idx]
            smp_box = sampled_gt_boxes[idx]
            ori_yaw = boxes_oriyaw[idx]
            file_path = self.root_path / info['path']
            bm_file_path = self.mlt_bm_root / "{}_{}.pkl".format(aug_boxes_image_idx[idx], aug_boxes_gt_idx[idx])
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape([-1, self.num_point_features])
            with open(bm_file_path, 'rb') as f:
                bm_obj_points = pickle.load(f)
            bm_obj_points = bm_obj_points.reshape([-1, self.bm_num_point_features]).astype(np.float32)
            gtrotation = point_box_utils.get_yaw_rotation(smp_box[6])
            bm_obj_points = np.einsum("nj,ij->ni", bm_obj_points, gtrotation) + smp_box[:3]
            if type > 0:
                gtrotation = point_box_utils.get_yaw_rotation(smp_box[6]-ori_yaw[0])
                obj_points[..., :3] = np.einsum("nj,ij->ni", obj_points[..., :3], gtrotation) + smp_box[:3]
                gt_smp_points_list.append(obj_points)
                gt_smp_bm_points_list.append(bm_obj_points)
            else:
                avg_feature = np.mean(obj_points[:,3:], axis=0, keepdims=True)
                bm_obj_points = np.concatenate([bm_obj_points, np.tile(avg_feature,(bm_obj_points.shape[0],1))], axis=-1)
                bm_points_list.append(bm_obj_points)
        if not self.no_stucking:
            large_sampled_gt_boxes = box_utils.enlarge_box3d(
                sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
            )
            points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)

        if len(gt_smp_points_list) > 0:
            smp_points = np.concatenate(gt_smp_points_list, axis=0)
            smp_inds = (valid_boxes_type > 0).nonzero()[0]
            if self.gt_smp_cfg["remove_yz_expansion"] > 0:
                smp_points, pnt_num_mask = self.remove_occ(points, smp_points, sampled_gt_boxes[smp_inds,:], self.gt_smp_cfg["remove_yz_expansion"], self.gt_smp_cfg["dp_rate"])
                smp_inds = smp_inds[pnt_num_mask]
                final_bm_points_list.extend([gt_smp_bm_points_list[i] for i in range(len(gt_smp_bm_points_list)) if pnt_num_mask[i]])
                bm_points_list = [bm_points_list[i] for i in range(len(bm_points_list)) if pnt_num_mask[i]]
            else:
                final_bm_points_list.extend(gt_smp_bm_points_list)
            points = np.concatenate([points, smp_points], axis=0)
            final_sampled_gt_boxes.append(sampled_gt_boxes[smp_inds,:])
            final_aug_boxes_image_idx.append(aug_boxes_image_idx[smp_inds])
            final_aug_boxes_gt_idx.append(aug_boxes_gt_idx[smp_inds])
            final_total_valid_sampled_dict.extend([total_valid_sampled_dict[smp_inds[i]] for i in range(len(smp_inds))])

        if len(bm_points_list) > 0:
            bm_points = np.concatenate(bm_points_list, axis=0)
            bm_inds = (valid_boxes_type == 0).nonzero()[0]
            ### remove occ, drop ratio
            if self.mlt_bm_cfg["remove_yz_expansion"] > 0:
                bm_points, pnt_num_mask = self.remove_occ(points, bm_points, sampled_gt_boxes[bm_inds,:], self.mlt_bm_cfg["remove_yz_expansion"], self.mlt_bm_cfg["dp_rate"])
                bm_inds = bm_inds[pnt_num_mask]
                final_bm_points_list.extend([bm_points_list[i] for i in range(len(bm_points_list)) if pnt_num_mask[i]])
            else:
                final_bm_points_list.extend(bm_points_list)
            points = np.concatenate([points, bm_points], axis=0)
            final_sampled_gt_boxes.append(sampled_gt_boxes[bm_inds, :])
            final_aug_boxes_image_idx.append(aug_boxes_image_idx[bm_inds])
            final_aug_boxes_gt_idx.append(aug_boxes_gt_idx[bm_inds])
            final_total_valid_sampled_dict.extend([total_valid_sampled_dict[bm_inds[i]] for i in range(len(bm_inds))])

        if len(final_bm_points_list) > 0:
            sampled_gt_names = np.array([x['name'] for x in final_total_valid_sampled_dict])
            data_dict['bm_points'] = [bm_points_raw[:,:3] for bm_points_raw in final_bm_points_list]
            del bm_points_list, gt_smp_bm_points_list, gt_smp_points_list
            if gt_boxes.ndim != 2 or gt_boxes.shape[0] == 0:
                # print("prefilter gt name:", data_dict['gt_names'], " filtered gt name:", gt_names)
                gt_names = sampled_gt_names
                gt_boxes = final_sampled_gt_boxes[0] if len(final_sampled_gt_boxes)==1 else np.concatenate(final_sampled_gt_boxes, axis=0)
            else:
                gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
                gt_boxes = np.concatenate([gt_boxes, *final_sampled_gt_boxes], axis=0)
            # print("points", len(points), len(data_dict['points']), len(final_bm_points_list))
            data_dict['points'] = points
            data_dict["augment_box_num"] = sampled_gt_names.shape[0]
            data_dict['aug_boxes_image_idx'] = final_aug_boxes_image_idx[0] if len(final_aug_boxes_image_idx)==1 else np.concatenate(final_aug_boxes_image_idx, axis=0)
            data_dict['aug_boxes_gt_idx'] = final_aug_boxes_gt_idx[0] if len(final_aug_boxes_gt_idx)==1 else np.concatenate(final_aug_boxes_gt_idx, axis=0)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        return data_dict


    def expand_voxel(self, bm_occ_coords, yz_epansion=2):
        y_ind = np.arange(-1, yz_epansion)
        z_ind = np.arange(-1, yz_epansion)
        y, z = np.meshgrid(y_ind, z_ind)
        x = np.ones_like(y)
        xyz_shift = np.stack([x, y, z], axis=-1).reshape(1, 9, 3)
        # print("xyz_shift", y.shape, y, xyz_shift.shape, xyz_shift[0,0], xyz_shift[0,1], xyz_shift[0,2])
        expand_bm_occ_coords = np.clip(np.expand_dims(bm_occ_coords, axis=1) + xyz_shift, a_min=0, a_max=None).reshape(-1, 3)
        return expand_bm_occ_coords


    def remove_occ(self, points, bm_points, sampled_gt_boxes, yz_epansion, drop_rate):
        if yz_epansion > 1:
            occ_coords_points = coords_utils.absxyz_2_spherexyz_np(points)[...,:3]
            bm_occ_coords_points = coords_utils.absxyz_2_spherexyz_np(bm_points)[...,:3]

            # v_res = 0.4203125, h_res = 0.0875, dist_res = 0.2, v_fov = (-24.9, 4), h_fov = (-60, 60), dist_fov = (0, 80)
            sphere_coords_min = np.min(np.concatenate([occ_coords_points, bm_occ_coords_points], axis=0), axis=0, keepdims=True)
            occ_coords = np.floor_divide(occ_coords_points - sphere_coords_min, self.sphere_coords_res).astype(np.int32)
            bm_occ_coords = np.floor_divide(bm_occ_coords_points - sphere_coords_min, self.sphere_coords_res).astype(np.int32)
            expand_bm_occ_coords = self.expand_voxel(bm_occ_coords, yz_epansion=yz_epansion)
            occ_coords = np.concatenate([occ_coords, expand_bm_occ_coords], axis=0)
            nx, ny, nz = list(np.max(occ_coords, axis=0))
            nx, ny, nz = nx+1, ny+1, nz+1
            # print("nx",nx,ny,nz)
            voxelwise_mask = np.zeros([nx, ny, nz], dtype=np.uint8)
            voxelwise_mask[occ_coords[...,0], occ_coords[...,1], occ_coords[...,2]] = np.ones_like(occ_coords[...,0], dtype=np.uint8)
            voxelwise_mask = np.cumsum(voxelwise_mask, axis=0) < 1.5
            bm_occ_coords_mask = voxelwise_mask[bm_occ_coords[..., 0], bm_occ_coords[...,1], bm_occ_coords[...,2]]
            bm_points = bm_points[bm_occ_coords_mask, :]
            bm_occ_coords = bm_occ_coords[bm_occ_coords_mask,:]
            # return bm_points[bm_occ_coords_mask,:]
            closeind = np.argsort(bm_points[...,2])
            bm_points, bm_occ_coords = bm_points[closeind,:], bm_occ_coords[closeind,:]
            _, indices = np.unique(bm_occ_coords, axis=0, return_index=True)
            bm_points = bm_points[indices,:]
        if drop_rate > 0:
            post_drop_mask = np.ones(bm_points.shape[0], dtype=int)
            post_drop_mask[:int(bm_points.shape[0] * drop_rate)] = 1
            np.random.shuffle(post_drop_mask)
            post_drop_mask = post_drop_mask.astype(bool)
            bm_points = bm_points[post_drop_mask, :]

        point_belong_mask = box_utils.boxes3d_contain_points(bm_points, sampled_gt_boxes)
        # point_box_ind = np.amax(point_belong_mask, axis=0)
        pnt_num_box_mask = np.sum(point_belong_mask, axis=1) >= 5
        if np.sum(pnt_num_box_mask) > 0:
            point_mask = np.sum(point_belong_mask[pnt_num_box_mask,:], axis=0) > 0
            bm_points = bm_points[point_mask, :]
        else:
            bm_points = np.zeros([0, bm_points.shape[1]], dtype=bm_points.dtype)
        # xavg = np.bincount(rinds, bm_points[...,0]) / counts
        # yavg = np.bincount(rinds, bm_points[...,1]) / counts
        # zavg = np.bincount(rinds, bm_points[...,2]) / counts
        # bm_points = np.concatenate([np.stack([xavg, yavg, zavg], axis=-1), bm_points[indices,:][...,3:]],axis=-1)
        return bm_points, pnt_num_box_mask # bm_points


    def add_box_jitter(self, sampled_boxes, points, existed_boxes, cfg):
        # occ_coords_points = coords_utils.absxyz_2_cylinxyz_np(points)
        centers = coords_utils.absxyz_2_cylinxyz_np(sampled_boxes[..., :3])
        range_scale = [min(5.3, np.min(centers[..., 0])), max(67.00, np.max(centers[..., 0]))]
        rot_scale = [min(-40.6944, min(centers[..., 1])), max(40.6944, np.max(centers[..., 1]))]
        jitters = np.random.uniform(low=0.0, high=1.0, size=(3, centers.shape[0]))
        center_base_min = np.clip(centers[..., 0] - cfg['box_range_jitter'], a_min=range_scale[0], a_max=range_scale[1])
        center_change = np.clip(centers[..., 0] + cfg['box_range_jitter'], a_min=range_scale[0], a_max=range_scale[1]) - center_base_min
        center_range = center_base_min + jitters[0,:] * center_change
        rot_base_min = np.clip(centers[..., 1] - cfg['box_rot_jitter'], a_min=rot_scale[0], a_max=rot_scale[1])
        rot_change = np.clip(centers[..., 1] + cfg['box_rot_jitter'], a_min=rot_scale[0], a_max=rot_scale[1]) - rot_base_min
        rot_range = rot_base_min + jitters[1, :] * rot_change
        if cfg['yaw_type'] == "main" and existed_boxes.shape[0] > 0:
            indx = np.random.randint(0, high=existed_boxes.shape[0], size=sampled_boxes.shape[0], dtype=int)
            base_yaw = existed_boxes[indx, 6]
        else:
            base_yaw = sampled_boxes[..., 6] - (rot_range - centers[..., 1]) * np.pi / 180
        yaw_base_min = base_yaw - cfg['box_yaw_jitter']
        yaw_change = base_yaw + cfg['box_yaw_jitter'] - yaw_base_min
        yaw_range = yaw_base_min + jitters[2, :] * yaw_change
        sampled_boxes[:, :3] = coords_utils.uvd2absxyz_np(center_range, rot_range, centers[..., 2], "cylinder")
        sampled_boxes = np.concatenate([sampled_boxes, sampled_boxes[..., 6:7]], axis=-1)
        sampled_boxes[:, 6] = yaw_range
        return sampled_boxes


    def individual_no_stucking(self):
        det_boxes = copy.deepcopy(valid_sampled_boxes[:, 0:7])
        det_boxes[:, 2] += self.det_height_shift
        clean_mask = \
        box_utils.boxes3d_contain_point_thresh(data_dict['points'], det_boxes, thresh=0, smaller=True).nonzero()[0]


        valid_boxes_type = valid_boxes_type[clean_mask]
        valid_sampled_dict = [valid_sampled_dict[x] for x in clean_mask]
        valid_sampled_boxes = valid_sampled_boxes[clean_mask]
        valid_sampled_boxes_image_idx = valid_sampled_boxes_image_idx[clean_mask]
        valid_sampled_boxes_gt_idx = valid_sampled_boxes_gt_idx[clean_mask]


    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)
                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)
                sampled_boxes_image_idx = np.stack([x['image_idx'] for x in sampled_dict], axis=0).astype(np.int32)
                sampled_boxes_gt_idx = np.stack([x['gt_idx'] for x in sampled_dict], axis=0).astype(np.int32)
                sampled_boxes_lst= []
                sampled_boxes_type_lst= []
                gt_num, bm_num = 0, 0
                if self.sampler_cfg.get('GT_SMP', None) is not None and self.gt_smp_cfg['sample_groups'][class_name] > 0:
                    gt_num = min(self.gt_smp_cfg['sample_groups'][class_name], len(sampled_dict))
                    sampled_boxes_lst.append(self.add_box_jitter(sampled_boxes[:gt_num,:], data_dict['points'], existed_boxes, self.gt_smp_cfg))
                    sampled_boxes_type_lst.append(np.ones([gt_num]))
                    bm_num = len(sampled_dict) - gt_num
                if self.sampler_cfg.get('MLT_BM', None) is not None and bm_num > 0:
                    sampled_boxes_lst.append(self.add_box_jitter(sampled_boxes[len(sampled_boxes)-bm_num:,:], data_dict['points'], existed_boxes, self.mlt_bm_cfg))
                    sampled_boxes_type_lst.append(np.zeros([bm_num]))
                sampled_boxes = sampled_boxes_lst[0] if len(sampled_boxes_lst) == 1 else np.concatenate(sampled_boxes_lst, axis=0)
                sampled_boxes_type = sampled_boxes_type_lst[0] if len(sampled_boxes_type_lst) == 1 else np.concatenate(sampled_boxes_type_lst, axis=0)
                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                valid_mask = self.remove_collide_boxes(sampled_boxes, existed_boxes)
                valid_boxes_type = sampled_boxes_type[valid_mask]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]
                valid_sampled_boxes_image_idx = sampled_boxes_image_idx[valid_mask]
                valid_sampled_boxes_gt_idx = sampled_boxes_gt_idx[valid_mask]
                if self.no_stucking:
                    det_boxes = copy.deepcopy(valid_sampled_boxes[:, 0:7])
                    det_boxes[:, 2] += self.det_height_shift
                    clean_mask = box_utils.boxes3d_contain_point_thresh(data_dict['points'], det_boxes, thresh=0, smaller=True).nonzero()[0]
                    valid_boxes_type = valid_boxes_type[clean_mask]
                    valid_sampled_dict = [valid_sampled_dict[x] for x in clean_mask]
                    valid_sampled_boxes = valid_sampled_boxes[clean_mask]
                    valid_sampled_boxes_image_idx = valid_sampled_boxes_image_idx[clean_mask]
                    valid_sampled_boxes_gt_idx = valid_sampled_boxes_gt_idx[clean_mask]
                valid_sampled_boxes, valid_sampled_boxes_oriyaw = valid_sampled_boxes[...,:7], valid_sampled_boxes[...,7:8]
                # print("valid_mask", len(valid_mask), len(clean_mask))
                if existed_boxes.ndim != 2 or existed_boxes.shape[0] == 0:
                    existed_boxes = valid_sampled_boxes
                else:
                    existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        if total_valid_sampled_dict.__len__() > 0:
            data_dict = self.add_sampled_boxes_best_match_points_to_scene(data_dict, sampled_gt_boxes, valid_sampled_boxes_oriyaw, total_valid_sampled_dict, valid_sampled_boxes_image_idx, valid_sampled_boxes_gt_idx, valid_boxes_type)
            data_dict['pre_aug_bm'] = True
        data_dict.pop('gt_boxes_mask')
        return data_dict

    def remove_collide_boxes(self, sampled_boxes, existed_boxes):
        if existed_boxes.ndim != 2 or existed_boxes.shape[0] == 0:
            print("existed_boxes: skip iou ", existed_boxes.shape)
            iou1 = None
        else:
            iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
        valid_inds = (iou1.max(axis=1) == 0).nonzero()[0]
        iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
        iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
        iou2 = iou2[valid_inds,:][:,valid_inds]
        for i in range(len(valid_inds)):
            if iou2.max() == 0:
                break
            else:
                max_ind = np.argmax(iou2.sum(axis=1))
                valid_inds = np.delete(valid_inds, max_ind)
                iou2 = np.delete(np.delete(iou2, max_ind, axis=0), max_ind, axis=1)
        # iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
        # iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
        # iou1 = None
        # if existed_boxes.ndim != 2 or existed_boxes.shape[0] == 0:
        #     print("existed_boxes: skip iou ", existed_boxes.shape)
        # else:
        #     iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
        # iou1 = iou1 if iou1 is not None and (iou1.shape[1] > 0) else iou2
        # valid_inds = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
        return valid_inds

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size