from functools import partial
import pickle
import numpy as np

from ...utils import common_utils
from . import augmentor_utils, database_sampler, best_match_querier, multi_best_match_querier, sup_gt_sampling

SPECIAL_NAMES = ["bm_points", "miss_points", "self_points", "other_points", "miss_occ_points", "self_occ_points", "other_occ_points", "self_limit_occ_mask", "miss_full_occ_points", "other_full_occ_points"]
class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST
        self.db_infos = {}
        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            if (cur_cfg.NAME in ["waymo_gt_sampling", "gt_sampling", "add_best_match", "sup_gt_sampling"]) and len(self.db_infos.keys()) == 0:
                for class_name in class_names:
                    self.db_infos[class_name] = []

                for db_info_path in cur_cfg.DB_INFO_PATH:
                    db_info_path = self.root_path.resolve() / db_info_path
                    with open(str(db_info_path), 'rb') as f:
                        infos = pickle.load(f)
                        [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]
                        print("self.db_infos", self.db_infos.keys())

            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)


    def sup_gt_sampling(self, config=None):
        db_sampler = sup_gt_sampling.SupGTSampling(
            root_path=self.root_path,
            sampler_cfg=config,
            db_infos=self.db_infos,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            db_infos=self.db_infos,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def waymo_gt_sampling(self, config=None):
        db_sampler = waymo_database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            db_infos=self.db_infos,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def waymo_obj_gt_sampling(self, config=None):
        db_sampler = waymo_obj_database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def add_best_match(self, config=None):
        bm_querier = best_match_querier.BestMatchQuerier(
            root_path=self.root_path,
            querier_cfg=config,
            class_names=self.class_names,
            db_infos=self.db_infos,
            logger=self.logger
        )
        return bm_querier

    def add_multi_best_match(self, config=None):
        bm_querier = multi_best_match_querier.MltBestMatchQuerier(
            root_path=self.root_path,
            querier_cfg=config,
            class_names=self.class_names,
            db_infos=self.db_infos,
            logger=self.logger
        )
        return bm_querier

    def add_waymo_multi_best_match(self, config=None):
        bm_querier = waymo_multi_best_match_querier.MltBestMatchQuerier(
            root_path=self.root_path,
            querier_cfg=config,
            class_names=self.class_names,
            db_infos=self.db_infos,
            logger=self.logger
        )
        return bm_querier

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
   
    def random_world_flip(self, data_dict=None, config=None, enable=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config, enable=enable)
        gt_boxes, points, bm_points, miss_points, self_points, other_points = data_dict['gt_boxes'], data_dict['points'], data_dict['bm_points'] if "bm_points" in data_dict else None, data_dict['miss_points'] if "miss_points" in data_dict else None, data_dict['self_points'] if "self_points" in data_dict else None, data_dict['other_points'] if "other_points" in data_dict else None
        special_points_lst = [data_dict[pt_key] for pt_key in SPECIAL_NAMES if pt_key in data_dict]
        special_name_lst = [pt_key for pt_key in SPECIAL_NAMES if pt_key in data_dict]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points, special_points_lst = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points, special_points_lst=special_points_lst, enable=enable
            )
        for name,val in zip(special_name_lst, special_points_lst):
            data_dict[name] = val
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def abs_world_flip(self, data_dict=None, config=None):
        return self.random_world_flip(data_dict=data_dict, config=config, enable=True)


    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        pre_rot_points = data_dict['points']
        special_points_lst = [data_dict[pt_key] for pt_key in SPECIAL_NAMES if pt_key in data_dict]
        special_name_lst = [pt_key for pt_key in SPECIAL_NAMES if pt_key in data_dict]
        gt_boxes, points, noise_rotation, special_points_lst  = augmentor_utils.global_rotation(data_dict['gt_boxes'], pre_rot_points, rot_range=rot_range, special_points_lst=special_points_lst)

        for name, val in zip(special_name_lst, special_points_lst):
            data_dict[name] = val

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        if config.get("SAVE_PRE_ROT", False):
            data_dict['pre_rot_points'] = pre_rot_points
            data_dict['rot_z'] = noise_rotation * 180 / np.pi
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        special_points_lst = [data_dict[pt_key] for pt_key in SPECIAL_NAMES if pt_key in data_dict]
        special_name_lst = [pt_key for pt_key in SPECIAL_NAMES if pt_key in data_dict]
        gt_boxes, points, special_points_lst = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], special_points_lst=special_points_lst)
        for name, val in zip(special_name_lst, special_points_lst):
            data_dict[name] = val

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict, validation=False):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        data_dict["gt_boxes_inds"] = np.arange(list(data_dict["gt_boxes_mask"].shape)[0])
        for cur_augmentor in self.data_augmentor_queue:
            if not validation or type(cur_augmentor).__name__ in ["BestMatchQuerier", "MltBestMatchQuerier"]:
                data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        # if 'calib' in data_dict:
        #     data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if "obj_ids" in data_dict:
                data_dict['obj_ids'] = data_dict['obj_ids'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        data_dict.pop('gt_boxes_inds', None)
        return data_dict
