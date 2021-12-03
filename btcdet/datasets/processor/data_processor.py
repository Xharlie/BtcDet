from functools import partial

import numpy as np

from ...utils import box_utils, common_utils, point_box_utils, coords_utils

class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, **kwargs):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.occ_config = kwargs["occ_config"]
        self.det_point_cloud_range = kwargs["det_point_cloud_range"]
        self.data_processor_queue = []
        self.occ_dim = None
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            print("cur_cfg.NAME", cur_cfg.NAME)
            self.data_processor_queue.append(cur_processor)


    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        mask = common_utils.mask_points_by_range(data_dict['points'], self.det_point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        if 'pre_rot_points' in data_dict:
            data_dict['pre_rot_points'] = data_dict['pre_rot_points'][mask]
        # print("self.point_cloud_range", self.point_cloud_range)
        # print("points", np.min(data_dict['points'], axis=0), np.max(data_dict['points'], axis=0))
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.det_point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]

        # print(data_dict['gt_boxes'])
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = np.expand_dims(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = np.arange(max_num, dtype=np.int).reshape(max_num_shape)
        paddings_indicator = actual_num > max_num
        return paddings_indicator

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            # print("max_voxels", self.mode, config.MAX_NUMBER_OF_VOXELS[self.mode])
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.occ_grid_size = np.round(grid_size).astype(np.int64)
            self.occ_voxel_size = config.VOXEL_SIZE
            self.det_grid_size = self.occ_grid_size
            self.det_voxel_size = self.occ_voxel_size
            self.max_points_per_voxel = config.MAX_POINTS_PER_VOXEL
            self.voxel_array = np.arange(self.max_points_per_voxel)
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator)

        points = data_dict['pre_rot_points'] if 'pre_rot_points' in data_dict else data_dict['points']
        voxel_output = voxel_generator.generate(points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output
        # print("points", np.min(voxels.reshape(-1,4), axis=0), np.max(voxels.reshape(-1,4), axis=0))
        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        if 'pre_rot_points' in data_dict:
            noise_rotation = data_dict['rot_z'] * np.pi / 180
            voxels = common_utils.rotate_points_along_z(voxels, np.array([noise_rotation]))
            data_dict.pop('pre_rot_points')

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict


    def transform_points_to_sphere_voxels(self, data_dict=None, config=None, voxel_generator=None):
        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            # print("max_voxels", self.mode, config.MAX_NUMBER_OF_VOXELS[self.mode])
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.occ_grid_size = np.round(grid_size).astype(np.int64)
            self.occ_voxel_size = config.VOXEL_SIZE
            # self.det_grid_size = self.occ_grid_size
            # self.det_voxel_size = self.occ_voxel_size
            self.max_points_per_voxel = config.MAX_POINTS_PER_VOXEL
            self.voxel_array = np.arange(self.max_points_per_voxel)
            return partial(self.transform_points_to_sphere_voxels, voxel_generator=voxel_generator)

        points = data_dict['pre_rot_points'] if 'pre_rot_points' in data_dict else data_dict['points']
        if self.occ_config.COORD_TYPE == "sphere":
            occ_coords_points = coords_utils.absxyz_2_spherexyz_np(points)
        elif self.occ_config.COORD_TYPE == "cylinder":
            occ_coords_points = coords_utils.absxyz_2_cylinxyz_np(points)
        else:
            assert False, "{}!!!".format(self.occ_config.COORD_TYPE)

        voxel_output = voxel_generator.generate(occ_coords_points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        # print("leng x", np.min(voxels[..., 0]), np.max(voxels[..., 0])-np.min(voxels[..., 0]))
        # print("leng y", np.min(voxels[..., 1]), np.max(voxels[..., 1])-np.min(voxels[..., 1]))
        # print("leng z", np.min(voxels[..., 2]), np.max(voxels[..., 2]))
        if 'pre_rot_points' in data_dict:
            voxels[..., 1] = voxels[..., 1] - data_dict['rot_z']
            data_dict.pop('pre_rot_points')
        # print("voxels", voxels.shape) # waymo < 19429
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    
    def det_transform_points_to_voxels(self, data_dict=None, config=None, det_voxel_generator=None):
        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            det_voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.det_point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            grid_size = (self.det_point_cloud_range[3:6] - self.det_point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.det_grid_size = np.round(grid_size).astype(np.int64)
            self.det_voxel_size = config.VOXEL_SIZE
            return partial(self.det_transform_points_to_voxels, det_voxel_generator=det_voxel_generator)

        points = data_dict['points']
        voxel_output = det_voxel_generator.generate(points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['det_voxels'] = voxels
        data_dict['det_voxel_coords'] = coordinates
        data_dict['det_voxel_num_points'] = num_points
        return data_dict


    def gen_pnt_label(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.gen_pnt_label, config=config)
        voxel_points = data_dict['voxels']
        V, VP, C = voxel_points.shape
        # print("V, VP, C", V, VP, C)
        voxel_num_points = data_dict['voxel_num_points']
        mask = self.get_paddings_indicator(voxel_num_points, VP, axis=0)
        inds = mask.nonzero()
        point_xyz = voxel_points[inds[0], inds[1], :3]
        point_label = point_box_utils.points_in_box_3d_label(point_xyz, data_dict['gt_boxes'], slack=1.0)
        voxel_points_label = np.zeros((V, VP), dtype=np.float32)
        voxel_points_label[inds[0], inds[1]] = point_label
        data_dict["voxel_points_label"] = voxel_points_label
        # print(np.max(voxel_points_label))
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
