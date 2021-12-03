import mayavi.mlab as mlab
import numpy as np
import torch
import visualize_utils as vu
import argparse
from matplotlib import pyplot as plt
import os
# RGB
clrs = {
    'gt_points': (1, 1, 1),
    'fore_gt_center': (1, 0.5, 0.5),
    'filter_center': (0.8, 0.8, 0),
    'boxvoxel_center': (1, 0.5, 0),
    'addpnt_view': (0.2, 1, 0.2),
    'drop_voxel_center': (0.3, 0, 0.8),
}

scales = {
    'gt_points': .01,
    'fore_gt_center': .1,
    'filter_center': .1,
    'boxvoxel_center': .1,
    'addpnt_view': .1,
    'drop_voxel_center': .1,
}

modes = {
    'gt_points': "sphere",
    'fore_gt_center': "sphere",
    'filter_center': "sphere",
    'boxvoxel_center': "sphere",
    'addpnt_view': "sphere",
    'drop_voxel_center': "sphere",
}

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dir1', type=str, default=None, help='specify the config for training')
    parser.add_argument('--dir2', type=str, default=None, help='specify the config for training')
    parser.add_argument('--pad', type=int, default=0, help='specify the config for training')

    args = parser.parse_args()

    return args


def main():
    args = parse_config()
    detailfilepath = os.path.join(args.dir1, "pc_rc.pkl")
    dict = np.load(detailfilepath, allow_pickle=True)
    base_dict = None
    if args.dir2 is not None:
        base_filepath = os.path.join(args.dir2, "pc_rc.pkl")
        base_dict = np.load(base_filepath, allow_pickle=True)
    print(dict.keys())
    for metric, value in dict.items():
        for curcls, content in value.items():
            for diff, mval in content.items():
                for num in [11,40]:
                    fig = plt.figure()
                    # print("R11_rc",mval["R11_rc"])
                    # print("R11_pc", mval["R11_pc"])
                    filename = "{}_{}_{}_R{}".format(curcls,metric,diff,num)
                    fig.suptitle(filename, fontsize=14, fontweight='bold')

                    ax = fig.add_subplot(111)
                    fig.subplots_adjust(top=0.90)

                    ax.set_xlabel('Recall')
                    ax.set_ylabel('Precision')
                    i = (mval["R{}_rc".format(num)]==0).argmax(axis=0)
                    if args.pad > 0:
                        rc = np.concatenate([mval["R{}_rc".format(num)][:i],np.ones(len(mval["R{}_rc".format(num)])-i)])
                        pc = mval["R{}_pc".format(num)]
                    else:
                        rc =mval["R{}_rc".format(num)][:i]
                        pc = mval["R{}_pc".format(num)][:i]
                    # i = len(mval["R{}_rc".format(num)])
                    ax.plot(rc, pc, marker='.', label='spg+pp R{}'.format(num))
                    if base_dict is not None:
                        bval = base_dict[metric][curcls][diff]
                        i = (bval["R{}_rc".format(num)] == 0).argmax(axis=0)
                        if args.pad > 0:
                            rc = np.concatenate(
                                [bval["R{}_rc".format(num)][:i], np.ones(len(bval["R{}_rc".format(num)]) - i)])
                            pc = bval["R{}_pc".format(num)]
                        else:
                            rc = bval["R{}_rc".format(num)][:i]
                            pc = bval["R{}_pc".format(num)][:i]

                        # i = len(mval["R{}_rc".format(num)])
                        ax.plot(rc, pc, marker='.', label='pp R{}'.format(num))

                    ax.legend()
                    plt.savefig('{}.png'.format(filename))


if __name__ == '__main__':
    main()

# python visualize_pcrc_curve.py --dir1 ../../output/kitti_models/occ_pointpillar_car/sur0.5_40000_0.8_realdp25_gpu2b4_160_pcrc/eval/epoch_147/val/default --dir2 ../../output/kitti_models/occ_pointpillar_car/pp_car/eval/epoch_76/val/default/