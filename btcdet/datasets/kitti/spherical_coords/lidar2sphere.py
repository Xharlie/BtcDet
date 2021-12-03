import numpy as np
import matplotlib.pyplot as plt
from ....utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from skimage import io

try:
    from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
except:
    from spconv.utils import VoxelGenerator



def load_from_bin(root_split_path, idx):
    bin_path = root_split_path  / 'velodyne'/ ('%s.bin' % idx)
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return obj[:,:3]


def normalize_depth(val, min_v, max_v):
    """
    print 'nomalized depth value'
    nomalize values to 0-255 & close distance value has high value. (similar to stereo vision's disparity map)
    """
    return (((max_v - val) / (max_v - min_v)) * 255).astype(np.uint8)


def normalize_val(val, min_v, max_v):
    """
    print 'nomalized depth value'
    nomalize values to 0-255 & close distance value has low value.
    """
    return (((val - min_v) / (max_v - min_v)) * 255).astype(np.uint8)


def in_h_range_points(m, n, fov):
    """ extract horizontal in-range points """
    return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), \
                          np.arctan2(n, m) < (-fov[0] * np.pi / 180))


def in_v_range_points(m, n, fov):
    """ extract vertical in-range points """
    return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), \
                          np.arctan2(n, m) > (fov[0] * np.pi / 180))


def fov_setting(points, x, y, z, dist, h_fov, v_fov):
    """ filter points based on h,v FOV  """

    if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points

    if h_fov[1] == 180 and h_fov[0] == -180:
        return points[in_v_range_points(dist, z, v_fov)]
    elif v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points[in_h_range_points(x, y, h_fov)]
    else:
        h_points = in_h_range_points(x, y, h_fov)
        v_points = in_v_range_points(dist, z, v_fov)
        return points[np.logical_and(h_points, v_points)]


def velo_points_2_pano(points, v_res, h_res, v_fov, h_fov, depth=False):
    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    print("z",z.shape)
    xydist = np.sqrt((abs(x)) ** 2 + y ** 2)

    points = fov_setting(points, x, y, z, xydist, h_fov, v_fov)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    xydist = np.sqrt((abs(x)) ** 2 + y ** 2)

    y_angel = np.arctan2(z, xydist) * 180 / np.pi

    # project point cloud to 2D point map
    # x_img = np.arctan2(-y, x) / (h_res * (np.pi / 180))
    # x_offset = (h_fov[0]) / h_res
    # x_img = np.trunc(x_img - x_offset).astype(np.int32)
    #
    # y_img = -(np.arctan2(z, xydist) / (v_res * (np.pi / 180)))
    # y_offset = v_fov[1] / v_res
    # y_img = np.trunc(y_img + y_offset).astype(np.int32)

    x_img = np.trunc((np.arctan2(-y, x) * 180 / np.pi - h_fov[0]) / h_res ).astype(np.int32)
    y_img = np.trunc((v_fov[1] - np.arctan2(z, xydist) * 180 / np.pi ) / v_res ).astype(np.int32)


    """ filter points based on h,v FOV  """


    x_size = int(np.ceil((h_fov[1] - h_fov[0]) / h_res))
    y_size = int(np.ceil((v_fov[1] - v_fov[0]) / v_res))

    # shift negative points to positive points (shift minimum value to 0)



    if depth == True:
        # nomalize distance value & convert to depth map
        dist = normalize_depth(dist, min_v=0, max_v=120)
    else:
        dist = normalize_val(dist, min_v=0, max_v=120)

    # array to img
    img = np.zeros([y_size, x_size], dtype=np.uint8)
    print("x_img",max(x_img), min(x_img))
    print("y_img",max(y_img), min(y_img))
    print("y_angel",max(y_angel), min(y_angel), max(y_angel)-min(y_angel))
    img[y_img, x_img] = dist
    dup(y_img, x_img)
    return img

def velo_points_2_spherical_voxel(points, v_res, h_res, dist_res, v_fov, h_fov, depth=False, dist_fov=80):
    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    print("z",z.shape)
    get_fov(x, y, z)
    xydist = np.sqrt((abs(x)) ** 2 + y ** 2)

    """ filter points based on h,v FOV  """
    points = fov_setting(points, x, y, z, xydist, h_fov, v_fov)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    xydist = np.sqrt((abs(x)) ** 2 + y ** 2)

    # project point cloud to 2D point map
    y_axis = np.trunc((np.arctan2(-y, x) * 180 / np.pi - h_fov[0]) / h_res).astype(np.int32)
    z_axis = np.trunc((np.arctan2(z, xydist) * 180 / np.pi - v_fov[0]) / v_res).astype(np.int32)
    x_axis = np.trunc(dist / dist_res).astype(np.int32)

    y_size = int(np.ceil((h_fov[1] - h_fov[0]) / h_res))
    z_size = int(np.ceil((v_fov[1] - v_fov[0]) / v_res))
    x_size = int(dist_fov / dist_res)
    print("x_size", x_size, " y_size", y_size, " z_size", z_size)
    dup3d(y_axis, x_axis, z_axis)



def dup3d(y_axis, x_axis, z_axis):
    mx, my, mz = np.max(x_axis) + 1, np.max(y_axis) + 1, np.max(z_axis) + 1
    hash = x_axis * my * mz + z_axis * my + y_axis
    u, c = np.unique(hash, return_counts=True)
    dupxzy = u[c > 1]
    dupx = dupxzy // (my * mz)
    dupy = dupxzy % my
    dupz = (dupxzy - dupy - dupx * my * mz) / my
    # print("dupx", dupx, dupx.shape)
    # print("dupy", dupy, dupy.shape)
    # print("dupz", dupz, dupz.shape)
    print("3d dup shape", dupz.shape)

    hash = x_axis * my + y_axis
    u, c = np.unique(hash, return_counts=True)
    dupxy = u[c > 1]
    dupx = dupxy // my
    dupy = dupxy % my
    print("2d dup shape", dupy.shape)


def dup(y_img, x_img):
    hash = x_img * 100 + y_img
    u, c = np.unique(hash, return_counts=True)
    dupyx = u[c > 1]
    dupx = dupyx // 100
    dupy = dupyx % 100
    uniquex = u // 100
    uniquey = u % 100
    print("dupx",dupx, dupx.shape)
    print("dupy",dupy)

    print("uniquex",uniquex, uniquex.shape)
    print("uniquey",uniquey)
    uuniquex, cuniquex = np.unique(uniquex, return_counts=True)
    uuniquey, cuniquey = np.unique(uniquey, return_counts=True)
    # print("uuniquex", uuniquex)
    print("uuniquey", uuniquey)
    print("cuniquey", cuniquey)

def get_fov(x,y,z):
    print("x:", min(x), max(x))
    print("y:", min(y), max(y))
    print("z:", min(z), max(z))
    angle = np.arcsin(y / np.sqrt(x**2 + y**2))
    # print("angle:", min(angle), max(angle))


def filter_fov(points, img_shape, root_split_path, idx):
    calib = get_calib(root_split_path, idx)
    pts_rect = calib.lidar_to_rect(points[:, 0:3])
    fov_flag = get_fov_flag(pts_rect, img_shape, calib)
    return points[fov_flag]


def get_fov_flag(pts_rect, img_shape, calib):
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    return pts_valid_flag


def get_calib(root_split_path, idx):
    calib_file = root_split_path / 'calib' / ('%s.txt' % idx)
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)


def get_image_n_shape(root_split_path, idx):
    img_file = root_split_path / 'image_2' / ('%s.png' % idx)
    assert img_file.exists()
    img = io.imread(img_file)
    return img, np.array(img.shape[:2], dtype=np.int32)


def velo_points_2_spherical_sparse(points, voxel_generator, v_res, h_res, dist_res, v_fov, h_fov, dist_fov, depth=False):
    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    print("z", z.shape)
    get_fov(x, y, z)
    xydist = np.sqrt((abs(x)) ** 2 + y ** 2)

    """ filter points based on h,v FOV  """
    points = fov_setting(points, x, y, z, xydist, h_fov, v_fov)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    xydist = np.sqrt((abs(x)) ** 2 + y ** 2)

    # y_axis = np.trunc((np.arctan2(-y, x) * 180 / np.pi - h_fov[0]) / h_res).astype(np.int32)
    # z_axis = np.trunc((np.arctan2(z, xydist) * 180 / np.pi - v_fov[0]) / v_res).astype(np.int32)
    # x_axis = np.trunc(dist / dist_res).astype(np.int32)

    sphere_x = dist
    sphere_y = np.arctan2(-y, x) * 180 / np.pi
    sphere_z = np.arctan2(z, xydist) * 180 / np.pi
    sphere_coords_points = np.stack([sphere_x, sphere_y, sphere_z], axis=-1)
    print("sphere_x", np.min(sphere_x), np.max(sphere_x))
    print("sphere_y", np.min(sphere_y), np.max(sphere_y))
    print("sphere_z", np.min(sphere_z), np.max(sphere_z))
    print("sphere_coords_points", sphere_coords_points.shape)

    voxel_output = voxel_generator.generate(sphere_coords_points)
    if isinstance(voxel_output, dict):
        voxel_features, voxel_coords, num_points = \
            voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
    else:
        voxel_features, voxel_coords, num_points = voxel_output

    return voxel_features, voxel_coords, num_points


def vis_pano(root_split_path, idx, v_res=0.4203125, h_res=0.0875, dist_res=0.2, v_fov=(-24.9, 4), h_fov=(-60, 60), dist_fov = (0,80), FOV=True):
    velo_points = load_from_bin(root_split_path, idx)
    img, img_shape = get_image_n_shape(root_split_path, idx)
    if FOV:
        velo_points = filter_fov(velo_points, img_shape, root_split_path, idx)
    print(velo_points.shape)
    pano_img = velo_points_2_pano(velo_points, v_res=v_res, h_res=h_res, v_fov=v_fov, h_fov=h_fov, depth=False)
    # velo_points_2_spherical_voxel(velo_points, v_res=v_res, h_res=h_res, dist_res=dist_res, v_fov=v_fov, h_fov=h_fov, depth=False)
    # display result image
    fig, (ax1,ax2) = plt.subplots(2, figsize=(15, 5))
    ax1.set_title("Result of Vertical FOV ({} , {}) & Horizontal FOV ({} , {})".format(v_fov[0], v_fov[1], h_fov[0], h_fov[1]))
    ax1.imshow(pano_img)
    ax2.imshow(img)
    plt.show()
    print(pano_img.shape)


def preprocess_sparse(root_split_path, idx, voxel_generator, v_res=0.4203125, h_res=0.0875, dist_res=0.2, v_fov=(-24.9, 4), h_fov=(-60, 60), dist_fov = (0,80), FOV=True):
    velo_points = load_from_bin(root_split_path, idx)
    img, img_shape = get_image_n_shape(root_split_path, idx)
    if FOV:
        velo_points = filter_fov(velo_points, img_shape, root_split_path, idx)
    print(velo_points.shape)
    voxel_features, voxel_coords, num_points = velo_points_2_spherical_sparse(velo_points, voxel_generator, v_res, h_res, dist_res, v_fov, h_fov, dist_fov, depth=False)
    print(voxel_coords.shape, np.min(num_points), np.max(num_points))


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'vis_pano':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../../').resolve()
        print("ROOT_DIR", ROOT_DIR)
        idx, v_res, h_res, dist_res = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        root_split_path = ROOT_DIR / 'data' / 'kitti' / 'detection3d' / 'training'
        v_fov, h_fov, dist_fov = (-24.9, 4), (-80, 80), (0, 80)
        voxel_generator = VoxelGenerator(
            voxel_size=[float(dist_res), float(h_res), float(v_res)],
            point_cloud_range=[dist_fov[0],  h_fov[0], v_fov[0], dist_fov[1],  h_fov[1], v_fov[1]],
            max_num_points=10,
            max_voxels=20000
        )
        print("voxel_size",[float(dist_res), float(h_res), float(v_res)])
        print('point_cloud_range',[dist_fov[0],  h_fov[0], v_fov[0], dist_fov[1],  h_fov[1], v_fov[1]])
        # preprocess_sparse(root_split_path, idx, voxel_generator, v_res=float(v_res), h_res=float(h_res), dist_res=float(dist_res), v_fov=v_fov, h_fov=h_fov, dist_fov = dist_fov, FOV=True)

        vis_pano(root_split_path, idx, v_res=float(v_res), h_res=float(h_res),
                 dist_res=float(dist_res), v_fov=v_fov, h_fov=h_fov, dist_fov=dist_fov, FOV=True)

        # python -m btcdet.datasets.kitti.spherical_coords.lidar2sphere vis_pano 001488.bin 0.42 0.35 0.02