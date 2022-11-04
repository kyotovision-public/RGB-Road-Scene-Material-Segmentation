# RGB Road Scene Material Segmentation, ACCV2022

This repository provides an inplementation of our paper RGB Road Scene Material Segmentation in ACCV2022.
If you use our code and data please cite our paper.

Please note that this is research software and may contain bugs or other issues – please use it at your own risk. If you experience major problems with it, you may contact us, but please note that we do not have the resources to deal with all issues.

```
@InProceedings{Cai_2022_ACCV,
    author    = {Sudong Cai and Ryosuke Wakaki and Shohei Nobuhara and Ko Nishino},
    title     = {RGB Road Scene Material Segmentation},
    booktitle = {Proceedings of Asian Conference on Computer Vision (ACCV)},
    month     = {Dec},
    year      = {2022}
}
```

The implementation of the encoder Mix-Transformer (MiT) and the corresponding ImageNet pre-trained model are adopted from the original project: [SegFormer](https://github.com/NVlabs/SegFormer).

The dataloader, utils, files for training and evaluation are adopted/modified from the related github project: [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception).


## KITTI-Materials dataset

Our KITTI-Materials dataset is available as `data.zip` at: [Google Drive](https://drive.google.com/drive/u/0/folders/1a5geigz8PKRlOYv-L2ePCfh0FlymW37H).
Uncompress the folder and move it to 'data/KITTI_Materials/'
Please note that the `list_folder1` and `list_folder2` in the dataset folder denote the `split-1` and `split-2` respectively.

## Pretrained models

The pretrained rmsnet weights for KITTI-Materials can be found as `rmsnet_split1.pth` and `rmsnet_split2.pth` at: [Google Drive](https://drive.google.com/drive/u/0/folders/1a5geigz8PKRlOYv-L2ePCfh0FlymW37H). 

The ImageNet pre-trained weight for MiT-B2 encoder `mit_b2.pth` can be found at: [mit_b2](https://drive.google.com/file/d/1m8fsG812o6KotF1NVo0YuiSfSn18TAOA/view?usp=sharing) .

## How to use the code

### Requirements

We tested our code with Python 3.8 on Ubuntu 18.04 LTS using the following packages.
Other recent versions may also be okay in general, but not sure.
You can also use [`env.def`](env.def) to create a singularity container with these packages.

* pytorch==1.11.0
* torchvision==0.12.0
* opencv_contrib_python==4.5.2.54
* tqdm==4.62.3
* einops==0.5.0
* timm==0.6.11
* matplotlib==3.6.2
* tensorboardx==2.5.1
* pillow==9.0.1


Put the dataset and pretrained weights as follows.

1. Uncompress 'KITTI_Materials' dataset into `data/` (i.e., the data path is expected to be `data/KITTI_Materials/`).
2. Put the pretrained weight `mit_b2.pth` into `weights/init/`.
3. Put the pretrained weights for our RMSNet into `weights/rmsnet/` (only for evaluation, unnecessary if training from scratch).


### Test with pretrained weights

Run `test.py` with trained weights.  It should output
```
Validation:
[Epoch: 0, numImages:   200]
Acc:0.8500841899671052, Acc_class:0.6338839179588864, mIoU:0.46823551505456135, fwIoU: 0.7583153107992938
Loss: 120.971
```

### Train from scrath

Run `train.py` to train RMSNet from scratch.  It requires a GPU with 40GB or more.

Note that, if you want to train with your customized settings, please directly change the corresponding hyperparameters (e.g., learning rate, epochs, Sync BN, and etc.) in the `train.py`, instead of using argparse from outside.
