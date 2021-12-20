# Behind the Curtain: Learning Occluded Shapes for 3D Object Detection (AAAI-2022)

## Reference

Please cite our paper if you are interested to use this implementation,  
 <strong>Behind the Curtain: Learning Occluded Shapes for 3D Object Detection</strong>.
```
@article{xu2021behind,
  title={Behind the Curtain: Learning Occluded Shapes for 3D Object Detection},
  author={Xu, Qiangeng and Zhong, Yiqi and Neumann, Ulrich},
  journal={arXiv preprint arXiv:2112.02205},
  year={2021}
}
```

The implementatin is also inspired by the ICCV-2021 paper,  
 <strong>SPG: Unsupervised domain adaptation for 3d object detection via semantic point generation</strong>.
``` 
@inproceedings{xu2021spg,
  title={Spg: Unsupervised domain adaptation for 3d object detection via semantic point generation},
  author={Xu, Qiangeng and Zhou, Yin and Wang, Weiyue and Qi, Charles R and Anguelov, Dragomir},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15446--15456},
  year={2021}
}
``` 


## Acknowledgement
Our model, BtcDet, is implemented based on [`[OpenPcdet 0.3.0]`](https://github.com/open-mmlab/OpenPCDet). We thank Shaohuai Shi for the discussion during our implementation.
  
  
     
## Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 16.04, should be able to work on 18.04)
* Python 3.6+
* PyTorch 1.1 or higher (tested on PyTorch 1.7, 1.8.1, 1.9, 1.10)
* CUDA 9.0 or higher (PyTorch 1.3+ needs CUDA 9.2+, tested on CUDA 10.2)
* [`spconv v1.2.1 (commit fad3000249d27ca918f2655ff73c41f39b0f3127)`](https://github.com/traveller59/spconv/commit/fad3000249d27ca918f2655ff73c41f39b0f3127)


### Install
b. Install the dependent libraries as follows:

* Install the dependent python libraries: 
```
pip install -r requirements.txt 
```

* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 
    * The very recent spconv 2.+ has significant interface changes and some issues in operators. Therefore, we do not support it for now.
    * If you use PyTorch 1.1, then make sure you install the `spconv v1.0` with ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)) instead of the latest one.
    * If you use PyTorch 1.3+, then you need to install the `spconv v1.2`. As mentioned in v1.2.1 of [`spconv (https://github.com/traveller59/spconv/tree/v1.2.1)`](https://github.com/traveller59/spconv/tree/v1.2.1)
    ```
    git clone -b v1.2.1  https://github.com/traveller59/spconv.git --recursive
    
    cd spconv
    
    sudo apt-get install libboost-all-dev
    
    python setup.py bdist_wheel
    
    cd ./dist 
    ```
    Inside the `dist` directory, use pip to install generated whl file.
    ```
    pip install spconv-1.2.1-{your system info}.whl
    ```
    After that, you should first get out of the spconv directory, then do python import spconv to see if you installed it correctly.
    

c. Install the operators in `BtcDet` by running the following command:
```shell
cd BtcDet
python setup.py develop
```

## Preparation

### Use Our Preprocessed Data: 
you can use our generated kitti's data including the generated complete object points, download it [[here (about 31GBs)]](https://drive.google.com/drive/folders/1mK4akt3Qro9nbw_NRfP__p2nb3a_rzxv?usp=sharing)  and put the zip file inside data/kitti/ and unzip it as detection3d directory.

### Alternatively, Generate Data by Yourself:
####KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):
```
BtcDet
├── data
│   ├── kitti
    │   │   │──detection3d  │── ImageSets
                    │   │   │── training
                    │   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
                    │   │   │── testing
                    │   │   │   ├──calib & velodyne & image_2
```

* Generate the data infos by running the following command: 
```python 
python -m btcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
#### Generate Approximated complete object points:
(Under `BtcDet` directory, execute:)
```python 
python -m btcdet.datasets.multifindbestfit
```





## Run training:
```
cd tools/
```
Single gpu training
```
mkdir output

mkdir output/kitti_car

python train.py --cfg_file ./cfgs/model_configs/btcdet_kitti_car.yaml --output_dir ../output/kitti_car/ --batch_size 2 --gpu_str "0"
```

Multi gpu training, assuming you have 4 gpus:
```
bash scripts/dist_train.sh 4  --batch_size 8 --gpu_str "0,1,2,3" --cfg_file ./cfgs/model_configs/btcdet_kitti_car.yaml --output_dir ../output/kitti_car/
```

## Run training:
```
cd tools/
```
Single gpu testing for all saved checkpoints, assuming you have 4 gpus:
```
python test.sh 4 --eval-all --cfg_file ./cfgs/model_configs/btcdet_kitti_car.yaml --gpu_str "0" --batch_size 2 --output_dir ../output/kitti_car/ --ckpt_dir  ../output/kitti_car/ckpt/
```

Multi gpu testing for all saved checkpoints, assuming you have 4 gpus:
```
bash scripts/dist_test.sh 4 --eval-all --cfg_file ./cfgs/model_configs/btcdet_kitti_car.yaml --gpu_str "0,1,2,3" --batch_size 8 --output_dir ../output/kitti_car/ --ckpt_dir  ../output/kitti_car/ckpt/
```

Multi gpu testing a specific checkpoint, assuming you have 4 gpus and checkpoint_39 is your best checkpoint :
```
bash scripts/dist_test.sh 4  --cfg_file ./cfgs/model_configs/btcdet_kitti_car.yaml --gpu_str "0,1,2,3" --batch_size 8 --output_dir ../output/kitti_car/ --ckpt  ../output/kitti_car/ckpt/checkpoint_epoch_39.pth
```
