import mayavi.mlab as mlab
import numpy as np
import torch

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(1200, 1200), draw_origin=False, scale=1.0, mode="sphere"):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode=mode,
                          colormap='gnuplot', scale_factor=scale, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='sphere', color=fgcolor,
                          colormap='gnuplot', scale_factor=scale, figure=fig)

    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig


def draw_sphere_pts(pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(600, 600))

    if isinstance(color, np.ndarray) and color.shape[0] == 1:
        color = color[0]
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    if isinstance(color, np.ndarray):
        pts_color = np.zeros((pts.__len__(), 4), dtype=np.uint8)
        pts_color[:, 0:3] = color
        pts_color[:, 3] = 255
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], np.arange(0, pts_color.__len__()), mode='sphere',
                          scale_factor=scale_factor, figure=fig)
        G.glyph.color_mode = 'color_by_scalar'
        G.glyph.scale_mode = 'scale_by_vector'
        G.module_manager.scalar_lut_manager.lut.table = pts_color
    else:
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='sphere', color=color,
                      colormap='gnuplot', scale_factor=scale_factor, figure=fig)

    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
    mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), line_width=3, tube_radius=None, figure=fig)

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None):

    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    fig = visualize_pts(points)
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    if gt_boxes is not None:
        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = boxes_to_corners_3d(ref_boxes)
        if ref_labels is None:
            fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(box_colormap[k % len(box_colormap)])
                mask = (ref_labels == k)
                fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100)
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return fig


def draw_scenes_multi(points_lst, colors_lst, scales_lst, mode_lst, gt_boxes=None, aug_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None, ref_ious=None, bgcolor=(1,1,1), voxelpnts_lst=[], voxelpnts_colors_lst=[], voxelpnts_op_lst=[], axis=False):
    for points in points_lst:
        if not isinstance(points, np.ndarray):
            points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if aug_boxes is not None and not isinstance(aug_boxes, np.ndarray):
        aug_boxes = aug_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_ious is not None and not isinstance(ref_ious, np.ndarray):
        ref_ious = ref_ious.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    # def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
    #                   show_intensity=False, size=(600, 600), draw_origin=True):
    fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1200, 1200))
    if axis:
        draw_xyz_axis(fig)

    for voxelpnts, voxelpnts_colors, opacity in zip(voxelpnts_lst, voxelpnts_colors_lst, voxelpnts_op_lst):
        draw_spherical_voxels(voxelpnts, voxelpnts_colors, opacity)

    for pts, clr, scale, mode in zip(points_lst, colors_lst, scales_lst, mode_lst):
        print(pts.shape, clr, scale, mode)
        fig = visualize_pts(pts, fig=fig, fgcolor=clr, scale=scale, mode=mode)
    # fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    if gt_boxes is not None:
        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0.2, 0.6, 0.8), max_num=100)
    if aug_boxes is not None:
        corners3d_aug = boxes_to_corners_3d(aug_boxes)
        fig = draw_corners3d(corners3d_aug, fig=fig, color=(0, 1, 1), max_num=100)

    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = boxes_to_corners_3d(ref_boxes)
        if ref_labels is None:
            fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(box_colormap[k % len(box_colormap)])
                mask = (ref_labels == k)

                fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], ious=ref_ious[mask] if ref_ious is not None else None, max_num=100)
    # mlab.view(azimuth=149, elevation=77.0, distance=104.0, roll=90.0)
    return fig


def draw_xyz_axis(fig):
    ax = [[0, 0.5, 1, 1.5], [0, 0, 0, 0]]
    x_ax_x = ax[0]
    x_ax_y = ax[1]
    x_ax_z = ax[1]

    y_ax_x = ax[1]
    y_ax_y = ax[0]
    y_ax_z = ax[1]

    z_ax_x = ax[1]
    z_ax_y = ax[1]
    z_ax_z = ax[0]

    mlab.plot3d(x_ax_x, x_ax_y, x_ax_z, tube_radius=0.1, color=(1, 0, 0))
    mlab.plot3d(y_ax_x, y_ax_y, y_ax_z, tube_radius=0.1, color=(0, 1, 0))
    mlab.plot3d(z_ax_x, z_ax_y, z_ax_z, tube_radius=0.1, color=(0, 0, 1))


def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, ious=None, tag='', max_num=500, tube_radius=None):
    """
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    import mayavi.mlab as mlab
    num = min(max_num, len(corners3d))
    for n in range(num):
        b = corners3d[n]  # (8, 3)

        if cls is not None:
            if isinstance(cls, np.ndarray):
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
            else:
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)

        if ious is not None:
            if isinstance(ious, np.ndarray):
                mlab.text3d(b[5, 0], b[5, 1], b[5, 2], 'iou: %.2f' % ious[n], scale=(0.3, 0.3, 0.3), color=(1, 1, 0.7), figure=fig)
            else:
                mlab.text3d(b[5, 0], b[5, 1], b[5, 2], 'iou: %s' % ious[n], scale=(0.3, 0.3, 0.3), color=(1, 1, 0.7), figure=fig)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        i, j = 0, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig



def absxyz_2_spherexyz_np(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    dist = np.linalg.norm(points[:,:3], axis=1)# np.sqrt(x ** 2 + y ** 2 + z ** 2)
    xydist = np.linalg.norm(points[:,:2], axis=1) # np.sqrt((abs(x)) ** 2 + y ** 2)
    sphere_x = dist
    sphere_y = np.arctan2(-y, x) * 180. / np.pi
    sphere_z = np.arctan2(z, xydist) * 180. / np.pi
    if points.shape[1] > 3:
        return np.stack([sphere_x, sphere_y, sphere_z, points[:, 3]], axis=-1)
    else:
        return np.stack([sphere_x, sphere_y, sphere_z], axis=-1)


def draw_spherical_voxels(point_xyz, color, opacity, draw_origin=True):

    if not isinstance(point_xyz, np.ndarray):
        point_xyz = point_xyz.cpu().numpy()
    offset = np.array([-1.43, 0.0, -2.184])
    spherical_range = np.array([2.24, -40.6944, -16.5953125, 70.72, 40.6944, 4.0])
    # spherical_voxel_size = np.array([0.32, 0.5184, 0.4203125])
    # spherical_voxel_size = np.array([0.32 * 1, 0.5184 * 1, 0.4203125 * 1])
    # spherical_voxel_size = np.array([0.32 * 2, 0.5184 * 2, 0.4203125 * 2])
    # spherical_voxel_size = np.array([0.32 * 3, 0.5184 * 3, 0.4203125 * 3])
    # spherical_voxel_size = np.array([0.32 * 4, 0.5184 * 4, 0.4203125 * 4])
    spherical_voxel_size = np.array([0.32 * 5, 0.5184 * 5, 0.4203125 * 5])
    # point_xyz = point_xyz[point_xyz[:,2] > 0.05]
    point_xyz += offset
    sphere_xyz = absxyz_2_spherexyz_np(point_xyz)
    sphere_coords_floor = np.floor((sphere_xyz - spherical_range[:3].reshape(1, 3)) / spherical_voxel_size.reshape(1, 3)).astype(np.int)
    sphere_coords_floor = np.unique(sphere_coords_floor, axis=0)
    sphere_coords_ceil = sphere_coords_floor + np.array([[1.0, 1.0, 1.0]])
    # print(sphere_coords_floor.dtype, spherical_voxel_size.reshape(1, 3).dtype, spherical_range[:3].reshape(1, 3).dtype,)
    sphere_floor = sphere_coords_floor * spherical_voxel_size.reshape(1, 3) + spherical_range[:3].reshape(1, 3)
    sphere_ceil = sphere_coords_ceil * spherical_voxel_size.reshape(1, 3) + spherical_range[:3].reshape(1, 3)
    sphere_range = np.concatenate([sphere_floor, sphere_ceil], axis=-1)

    # color = (253 / 255, 185 / 255, 200 / 255)
    # color = (255 / 255, 140 / 255, 0 / 255)
    # opacity = 0.01
    xyz_lst, tries_lst, base = [], [], 0
    for i in range(len(sphere_range)):
        xyz, tries, base = sphere_voxel(base, res=5j, sphere_range=sphere_range[i])
        tries_lst.extend(tries)
        xyz_lst.append(xyz)
    xyz = np.concatenate(xyz_lst, axis=0)
    xyz -= offset
    mlab.triangular_mesh(xyz[..., 0], xyz[..., 1], xyz[..., 2], tries_lst, color=color, opacity=opacity, representation="surface")
    if draw_origin:
        mlab.points3d(-offset[0], -offset[1], -offset[2], color=(1, 0, 0), mode='sphere', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)


def sphere_voxel(base, res=3j, sphere_range=[3.84, -40.6944, -16.5953125, 70.72, 40.6944, 4.0]):
    xyz1, tries1, base = xfix(sphere_range[0], [sphere_range[4], sphere_range[2], sphere_range[1], sphere_range[5]], base, res=res)
    xyz2, tries2, base = xfix(sphere_range[3], [sphere_range[4], sphere_range[2], sphere_range[1], sphere_range[5]], base, res=res)

    xyz3, tries3, base = yfix(sphere_range[1], [sphere_range[0], sphere_range[5], sphere_range[3], sphere_range[2]], base, res=res)
    xyz4, tries4, base = yfix(sphere_range[4], [sphere_range[0], sphere_range[2], sphere_range[3], sphere_range[5]], base, res=res)

    xyz5, tries5, base = zfix(sphere_range[2], [sphere_range[0], sphere_range[1], sphere_range[3], sphere_range[4]], base, res=res)
    xyz6, tries6, base = zfix(sphere_range[5], [sphere_range[3], sphere_range[1], sphere_range[0], sphere_range[4]], base, res=res)

    xyz = np.concatenate([xyz1, xyz2, xyz3, xyz4, xyz5, xyz6], axis=0)
    tries_lst = tries1 + tries2 + tries3 + tries4 + tries5 + tries6
    return xyz, tries_lst, base

def xfix(sphere_x, yz_range, base, res=3j):
    # [3.84, -40.6944, -16.5953125, 70.72, 40.6944, 4.0]
    sphere_y, sphere_z = np.mgrid[yz_range[0]:yz_range[2]:res, yz_range[1]:yz_range[3]:res]

    xydist = sphere_x * np.cos(sphere_z * np.pi / 180.)
    x = xydist * np.cos(sphere_y * np.pi / 180.)
    y = -xydist * np.sin(sphere_y * np.pi / 180.)
    z = sphere_x * np.sin(sphere_z * np.pi / 180.)
    tries = []
    lw, lh = x.shape
    for i in range(lw - 1):
        for j in range(lh - 1):
            tries.extend([(base + i * lh + j, base + i * lh + j + 1, base + i * lh + j + lh),
                          (base + i * lh + j + 1, base + i * lh + j + lh, base + i * lh + j + lh + 1)])
    base += len(x.reshape(-1))
    return np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=-1), tries, base

def yfix(sphere_y, xz_range, base, res=3j):
    # [3.84, -40.6944, -16.5953125, 70.72, 40.6944, 4.0]
    sphere_x, sphere_z = np.mgrid[xz_range[0]:xz_range[2]:2j, xz_range[1]:xz_range[3]:res]

    xydist = sphere_x * np.cos(sphere_z * np.pi / 180.)
    x = xydist * np.cos(sphere_y * np.pi / 180.)
    y = -xydist * np.sin(sphere_y * np.pi / 180.)
    z = sphere_x * np.sin(sphere_z * np.pi / 180.)
    tries = []
    lw, lh = x.shape
    for i in range(lw - 1):
        for j in range(lh - 1):
            tries.extend([(base + i * lh + j, base + i * lh + j + 1, base + i * lh + j + lh),
                          (base + i * lh + j + 1, base + i * lh + j + lh, base + i * lh + j + lh + 1)])
    base += len(x.reshape(-1))
    return np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=-1), tries, base

def zfix(sphere_z, xy_range, base, res=3j):
    # [3.84, -40.6944, -16.5953125, 70.72, 40.6944, 4.0]
    sphere_x, sphere_y = np.mgrid[xy_range[0]:xy_range[2]:2j, xy_range[1]:xy_range[3]:res]

    xydist = sphere_x * np.cos(sphere_z * np.pi / 180.)
    x = xydist * np.cos(sphere_y * np.pi / 180.)
    y = -xydist * np.sin(sphere_y * np.pi / 180.)
    z = sphere_x * np.sin(sphere_z * np.pi / 180.)
    tries = []
    lw, lh = x.shape
    for i in range(lw - 1):
        for j in range(lh - 1):
            tries.extend([(base + i * lh + j, base + i * lh + j + 1, base + i * lh + j + lh),
                          (base + i * lh + j + 1, base + i * lh + j + lh, base + i * lh + j + lh + 1)])
    base += len(x.reshape(-1))
    return np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=-1), tries, base