import numpy as np


class PointFeatureEncoder(object):
    def __init__(self, config, point_cloud_range=None):
        super().__init__()
        self.point_encoding_config = config
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
        self.used_feature_list = self.point_encoding_config.used_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list
        self.point_cloud_range = point_cloud_range

    @property
    def num_point_features(self):
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        # data_dict['points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
        #     data_dict['points']
        # )
        if self.point_encoding_config.encoding_type == "absolute_coordinates_encoding":
            data_dict['use_lead_xyz'] = True
        else:
            data_dict['use_lead_xyz'] = False
        # print("points_intensity", np.min(data_dict['points'][:,3]), np.max(data_dict['points'][:,3]))
        return data_dict

    def absolute_coordinates_encoding(self, points=None):
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx+1])
        function = self.point_encoding_config.get("function", None)
        if function is not None:
            pos = int(function[0]) - 2
            if function[1] == "tanh":
                point_feature_list[pos] = np.tanh(point_feature_list[pos])
            elif function[1].startswith("clip"):
                min, max = function[1][5:].split("-")
                min, max = float(min), float(max)
                point_feature_list[pos] = np.tanh(np.clip(point_feature_list[pos], min, max))
        point_features = np.concatenate(point_feature_list, axis=1)
        return point_features, True