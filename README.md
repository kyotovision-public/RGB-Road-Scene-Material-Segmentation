# RGB Road Scene Material Segmentation, ACCV2022
This repository provides an inplementation of our paper RGB Road Scene Material Segmentation in ACCV2022.  
If you use our code and data, please cite our paper (wait to be released).
The implementation of the encoder Mix-Transformer (MiT) and the corresponding ImageNet pre-trained model are adopted from the official project: https://github.com/NVlabs/SegFormer .

## Download data
Our KITTI-Materials dataset is available at: https://drive.google.com/drive/u/0/folders/1a5geigz8PKRlOYv-L2ePCfh0FlymW37H .
Uncompress the folder and move it to 'data/KITTI_Materials/'
(As it has been organized and checked, no extra organization is required).
Please note that the ``list_folder1'' and ``list_folder2'' in the dataset folder denote the ``split-1'' and ``split-2.''

## Download trained models and ImageNet pre-trained weight
The trained rmsnet weights for KITTI-Materials and the ImageNet pre-trained weight for MiT-B2 encoder can be found at: https://drive.google.com/drive/u/0/folders/1a5geigz8PKRlOYv-L2ePCfh0FlymW37H . 

## Usage
### Prerequisites

Requirements:
Ubuntu 18.04 LTS
python 3.8
torch==1.11.0 (other recent versions may also be okay, but not sure)
torchvision==0.12.0 (it needs to match the torch version, if different torch version is used)
opencv_contrib_python==4.5.2 (other recent versions may also be okay, but not sure)
tqdm
functools
einops (recent version)
timm (recent version)
