# import mayavi.mlab as mlab
import numpy as np
import torch
# import visualize_utils as vu
import argparse
import os, sys
# sys.path.append('../../btcdet/datasets/waymo/')
import pickle
# from waymo_utils import *
# from waymo_open_dataset import dataset_pb2 as open_dataset
# from waymo_open_dataset.utils import frame_utils
# import tensorflow.compat.v1 as tf
import glob


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dir', type=str, default=None, help='specify the config for training')

    args = parser.parse_args()

    return args


def main():
    args = parse_config()
    files = glob.glob(args.dir + "/*npy")
    for file in files:
        new_dict = {}
        dict = np.load(file, allow_pickle=True).item()
        for key, val in dict.items():
            if not isinstance(val, np.ndarray):
                val = val.cpu().numpy()
            new_dict[key] = val
            print(val)
        np.save(file, new_dict)



if __name__ == '__main__':
    main()
