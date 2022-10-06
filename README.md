# RGB Road Scene Material Segmentation, ACCV2022
This repository provides an inplementation of our paper RGB Road Scene Material Segmentation in ACCV2022.  

If you use our code and data, please cite our paper (to be released).

The implementation of the encoder Mix-Transformer (MiT) and the corresponding ImageNet pre-trained model are adopted from the official project: [SegFormer](https://github.com/NVlabs/SegFormer) .

The dataloader, utils, files for training and evaluation are adopted/modified from the related github project: [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception).


## Download data
Our KITTI-Materials dataset is available at: [Google Drive](https://drive.google.com/drive/u/0/folders/1a5geigz8PKRlOYv-L2ePCfh0FlymW37H) .
Uncompress the folder and move it to 'data/KITTI_Materials/'
(As it has been organized and checked, no extra organization is required).
Please note that the list_folder1 and list_folder2 in the dataset folder denote the split-1 and split-2.

## Download trained models and ImageNet pre-trained weight
The trained rmsnet weights for KITTI-Materials and the ImageNet pre-trained weight for MiT-B2 encoder can be found at: [Google Drive](https://drive.google.com/drive/u/0/folders/1a5geigz8PKRlOYv-L2ePCfh0FlymW37H) . 

## Usage
### Requirements
Ubuntu 18.04 LTS

python 3.8

(Please use pip/pip3 or conda to install subsequent dependencies)

pytorch==1.11.0 (other recent versions may also be okay, but not sure)

torchvision==0.12.0 (it needs to match the pytorch version if different pytorch version is used)

opencv_contrib_python==4.5.2 (other recent versions may also be okay, but not sure)

tqdm

functools

einops

timm

### Guidance
To use our code, please 

(1) put the uncompressed 'KITTI_Materials' dataset into data/ (i.e., the data path is expected to be 'data/KITTI_Materials/')

(2) put the pre-trained weight 'mit_b2.pth' into 'weights/init/'

(3) put the trained weights for our RMSNet into 'weights/rmsnet/' (only for evaluation, no need to do so if training from scratch)

(4) use python or ipython to run test.py with trained weights (e.g., for python, insert python test.py)

(5) use python or ipython to run train.py to train RMSNet from scratch (e.g., for python, insert python train.py)

Note that, if you want to train with your customized settings, please directly change the corresponding hyperparameters (e.g., learning rate, epochs, Sync BN, and etc.) in the train.py, instead of using argparse from outside.
