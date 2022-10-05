import argparse
import os
import numpy as np
from tqdm import tqdm
import random
import torch

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.rmsnet.rmsnet import RMSNet as Network # test rmsnet
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from torch.optim.lr_scheduler import _LRScheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 



### standard colors for material annotation
def get_my_labels():
    " r,g,b"
    return np.array([
        [ 44, 160,  44], # asphalt
        [ 31, 119, 180], # concrete
        [255, 127,  14], # metal
        [214,  39,  40], # road marking
        [140,  86,  75], # fabric, leather
        [127, 127, 127], # glass
        [188, 189,  34], # plaster
        [255, 152, 150], # plastic
        [ 23, 190, 207], # rubber
        [174, 199, 232], # sand
        [196, 156, 148], # gravel
        [197, 176, 213], # ceramic
        [247, 182, 210], # cobblestone
        [199, 199, 199], # brick
        [219, 219, 141], # grass
        [158, 218, 229], # wood
        [ 57,  59, 121], # leaf
        [107, 110, 207], # water
        [156, 158, 222], # human body
        [ 99, 121,  57]]) # sky

###########


class TrainerAdv(object):
    def __init__(self, args):
        self.args = args
        self.warm = 10 # warm-up rounds
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = Network(num_classes=self.nclass,
                        backbone='segformer',
                        encoder_id=2,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        ### define a save path
        self.save_path = ''
        
        train_params = [{'params': model.backbone.parameters(), 'lr': args.lr},
                        {'params': model.decoder.parameters(), 'lr': args.lr * 10}]
        
        optimizer = torch.optim.AdamW(train_params, betas=(0.9, 0.999), weight_decay=0.01)


        iter_per_epoch = len(self.train_loader)
        
        # balanced weights (no need to be applied in the eval phase)
        #weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
        #weight = torch.from_numpy(weight.astype(np.float32))
        #self.weight = weight
        
        # Define Criterion
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader), warmup_epochs=self.warm)

        # Using cuda
        if args.cuda:
            device = torch.device(0)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
            self.model.to(device)
            
        ### reload param and optimizer
        train_params = [{'params': self.model.module.backbone.parameters(), 'lr': args.lr},
                        {'params': self.model.module.decoder.parameters(), 'lr': args.lr * 10}]
        
        self.optimizer = torch.optim.AdamW(train_params, betas=(0.9, 0.999), weight_decay=0.01)
        
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader), warmup_epochs=self.warm)

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        
        
        # observing box
        self.box_pred = []
        self.box_target = []
        self.seg_pred= []


        # re-load self.args, in case args changed in above sections
        self.args = args

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        
        # train loader re-creation with each epoch (to shuffle the train data)
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, _, _, _ = make_data_loader(self.args, **kwargs)

        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            ### training            
            image_cat, target, mask = sample['image_cat'], sample['label'], sample['mask']
            
            if self.args.cuda:
                image_cat, target, mask = image_cat.cuda(), target.cuda(), mask.cuda()
            
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image_cat)
            loss = self.criterion(output, target, mask) # ce has input "mask"
            #loss = self.criterion(output, target) # focal has no input "mask"
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                if args.positional_encoding:
                   self.summary.visualize_image(self.writer, self.args.dataset, image_cat[:,-5:-2], target, output, global_step)
                else:
                   self.summary.visualize_image(self.writer, self.args.dataset, image_cat[:,-3:], target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image_cat.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image_cat, target = sample['image_cat'], sample['label']
            if self.args.cuda:
                image_cat, target = image_cat.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image_cat)
            loss = self.criterion(output, target)
            
            # put predictions into the box
            self.box_pred.append(output)
            self.box_target.append(target)
            
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            
            # record pred
            self.seg_pred.append(pred)
            
            

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image_cat.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


            

###############################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet', 'resnet_adv', 'resnet_adv_wave', 'xception_wave'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes', 'kitti', 'kitti_advanced', 'kitti_advanced_manta', 'handmade_dataset', 'handmade_dataset_stereo'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal', 'original'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--batch-size-val', type=int, default=None,
                        metavar='N', help='input batch size for \
                                validation (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # propagation and positional encoding option
    parser.add_argument('--propagation', type=int, default=0,
                        help='image propagation length (default: 0)')
    parser.add_argument('--positional-encoding', action='store_true', default=False,
                        help='use positional encoding')
    parser.add_argument('--use-aolp', action='store_true', default=False,
                        help='use aolp')
    parser.add_argument('--use-dolp', action='store_true', default=False,
                        help='use dolp')
    parser.add_argument('--use-nir', action='store_true', default=False,
                        help='use nir')
    parser.add_argument('--use-pretrained-resnet', action='store_true', default=False,
                        help='use pretrained resnet101')
    parser.add_argument('--list-folder', type=str, default='list_folder1')

    args = parser.parse_args()
    
    ### args inputs
    args.lr = 0 
    args.workers = 1
    args.epochs = 1
    args.batch_size = 1
    args.batch_size_val = 8
    args.gpu_ids = "0" 
    args.backbone = "mit_b2" #
    args.checkname = "new"
    args.eval_interval = 1
    args.loss_type = "ce"
    args.dataset = "kitti_advanced"
    args.propagation = 0 # int value
    args.sync_bn = False #True
    args.list_folder = "list_folder1" # split-1: list_folder1; split-2: list_folder2
    args.lr_scheduler = 'cos' # choices=['poly', 'step', 'cos']
    args.use_balanced_weights = False # give weight masks to the objective function (no need for eval phase)
    
    
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'kitti': 50,
            'kitti_advanced': 50
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'kitti' : 0.01,
            'kitti_advanced' : 0.01
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)

    torch.manual_seed(args.seed)


    trainer = TrainerAdv(args)
    
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    trainer.save_path = 'weights/rmsnet/' 
    trainer.model.module.load_state_dict(torch.load(trainer.save_path+'rmsnet_split1.pth')) # for split-1
    #trainer.model.module.load_state_dict(torch.load(trainer.save_path+'rmsnet_split2.pth')) # for split-2
    trainer.args.start_epoch = 0
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.validation(epoch)

    trainer.writer.close()
    print(args)
