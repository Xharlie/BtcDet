# import mayavi.mlab as mlab
import numpy as np
import torch
from skimage.draw import line_aa

def draw_lidars_box3d_on_birdview(lidars1, lidars2, colors, gt_box3d_center, input_map_h, input_map_w, xrange_min, xrange_max, yrange_min, yrange_max, vw=0.1, vh=0.1, bv_log_factor=1):

    image = lidar_to_bird_view_img(lidars1, input_map_h, input_map_w, xrange_min, xrange_max, yrange_min, yrange_max, vw, vh, bv_log_factor, color=colors[0])
    image = image.copy()
    image = lidar_to_bird_view_img(lidars2, input_map_h, input_map_w, xrange_min, xrange_max, yrange_min, yrange_max, vw, vh, bv_log_factor, color=colors[1], birdview=image)
    img = image.copy()
    # color = np.array(80, 255, 80)
    gt_box3d_corner = center_to_corner_box3d(gt_box3d_center)
    color = np.array([255, 0, 255])
    for box in gt_box3d_corner:
        x0, y0 = lidar_to_bird_view(
            box[0, 0], box[0, 1], xrange_min, xrange_max, yrange_min, yrange_max, vw, vh, factor=bv_log_factor
        )
        x1, y1 = lidar_to_bird_view(
            box[1, 0], box[1, 1], xrange_min, xrange_max, yrange_min, yrange_max, vw, vh, factor=bv_log_factor
        )
        x2, y2 = lidar_to_bird_view(
            box[2, 0], box[2, 1], xrange_min, xrange_max, yrange_min, yrange_max, vw, vh, factor=bv_log_factor
        )
        x3, y3 = lidar_to_bird_view(
            box[3, 0], box[3, 1], xrange_min, xrange_max, yrange_min, yrange_max, vw, vh, factor=bv_log_factor
        )
        _draw_line(img, [int(y0), int(x0)], [int(y1), int(x1)], color, max_value=255)
        _draw_line(img, [int(y1), int(x1)], [int(y2), int(x2)], color, max_value=255)
        _draw_line(img, [int(y2), int(x2)], [int(y3), int(x3)], color, max_value=255)
        _draw_line(img, [int(y3), int(x3)], [int(y0), int(x0)], color, max_value=255)
    return img.astype(np.uint8)


def _draw_line(image, p1, p2, color, max_value):
    assert len(image.shape) == 3
    image_h, image_w, channels = image.shape
    assert channels in [1, 3]
    assert color.shape == (3,)
    assert all(0 <= c <= 255 for c in color)
    rr, cc, val = line_aa(
        int(round(p1[0])), int(round(p1[1])), int(round(p2[0])), int(round(p2[1]))
    )
    in_bound = (rr >= 0) & (rr < image_h) & (cc >= 0) & (cc < image_w)
    rr = rr[in_bound]
    cc = cc[in_bound]
    val = val[in_bound].reshape([-1, 1])
    color = np.array(color if channels == 3 else [color[0]]) / 255.0
    image[rr, cc] = val * color * max_value


def lidar_to_bird_view(x, y, xrange_min, xrange_max, yrange_min, yrange_max, vw, vh, factor=1):
    a = (x - xrange_min) / vw * factor
    b = (y - yrange_min) / vh * factor
    a = np.clip(a, a_max=(xrange_max - xrange_min) / vw * factor, a_min=0)
    b = np.clip(b, a_max=(yrange_max - yrange_min) / vh * factor, a_min=0)
    return a, b

def lidar_to_bird_view_img(lidar, input_map_h=794, input_map_w=692, xrange_min=0, xrange_max=69.12, yrange_min=-39.68, yrange_max=39.68, vw=0.1, vh=0.1, factor=1, color=(255, 255, 255), birdview=None):
    if birdview is None:
        birdview = np.zeros(
            (int(input_map_h * factor), int(input_map_w * factor), 3), dtype=np.uint8)
    lidar_mask = np.where(
        np.logical_and(
            np.logical_and(lidar[:, 0] >= xrange_min, lidar[:, 0] < xrange_max),
            np.logical_and(lidar[:, 1] >=yrange_min, lidar[:, 1] < yrange_max)
        )
    )[0]
    for point in lidar[lidar_mask]:
        x, y = point[0:2]
        x, y = int((x - xrange_min)/ vw * factor), int((y - yrange_min) / vh * factor)
        birdview[y, x, :] = color
    return birdview

def center_to_corner_box3d(boxes_center, bottom_center=False):
    batch_size = boxes_center.shape[0]
    ret = np.zeros((batch_size, 8, 3), dtype=np.float32)
    for i in range(batch_size):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[6]]
        l, w, h = size[0], size[1], size[2]
        # print("l, w, h", l, w, h)
        mat = np.array([
            [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
            [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
            [0, 0, 0, 0, h, h, h, h]
        ]) if bottom_center else np.array([
            [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
            [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
            [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
        ])
        yaw = rotation[2]
        rot_mat = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                            [np.sin(yaw), np.cos(yaw), 0.0],
                            [0.0, 0.0, 1.0]])
        corner_pos_in_cam = np.dot(rot_mat, mat) + np.tile(translation, (8, 1)).T
        box3d = corner_pos_in_cam.transpose()
        ret[i] = box3d
    return ret