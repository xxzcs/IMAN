
import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Optional
from hausdorff import hausdorff_distance

from numpy.core.fromnumeric import nonzero
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vision_transformer import SwinUnet as Swin_ViT_seg
# from networks.DpRAN import DPRAN as DPRAN
import breastdatasets
import albumentations as albu
import cv2
import numpy as np
# import segmentation_models_pytorch as smp
from gitmodules.smp import segmentation_models_pytorch as smp

import timm
import torch
import torch.backends.cudnn as cudnn
import torchvision
from sklearn.metrics import cohen_kappa_score
from tensorboardX import SummaryWriter
from timm import optim
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler, accuracy, get_state_dict
from tqdm import tqdm
from turbojpeg import (TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE, TJFLAG_PROGRESSIVE,
                       TJPF_GRAY, TJPF_RGB, TJSAMP_GRAY, TurboJPEG)
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
import SimpleITK as sitk

from medpy import metric
import losses
import utils
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.utils import base
#from datasets import ceus_dataset, us_dataset
'''

https://github.com/qubvel/segmentation_models.pytorch
https://github.com/CSAILVision/semantic-segmentation-pytorch

'''


class DiceScore(base.Metric):

    def __init__(self, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, y_pr, y_gt):
        # 1-dice_loss
        y_pr = self.activation(y_pr)
        dice_score = 1-self._dice_loss(y_pr[:, 0, ...], y_gt[:, 0, ...]) #切片意义
        return dice_score
class HDScore(base.Metric):

    def __init__(self, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)

    def _hd_score(self, score, target):
        target = target.float()
        hd_score = 0
        num = score.shape[0] #16
        
        for i in range(num):

            scorei = score[i, ...]
            targeti = target[i, ...]

            hd_score += hausdorff_distance(targeti.squeeze().detach().cpu().numpy(),scorei.squeeze().detach().cpu().numpy(),distance='euclidean')

        hd_score = hd_score / num

        return hd_score

    def forward(self, y_pr, y_gt):
        # 1-dice_loss
        y_pr = self.activation(y_pr)
        hd_score = self._hd_score(y_pr[:, 0, ...], y_gt[:, 0, ...]) #切片意义
        hd_score_tensor = torch.tensor(hd_score).cuda()    
        return hd_score_tensor

class HausdorffScore(base.Metric):
    def __init__(self, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.hausdorffcomputer = sitk.HausdorffDistanceImageFilter()

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)

        b, _, _, _ = y_pr.shape
        hd = []
        for i in range(b):
            labelPred = sitk.GetImageFromArray(y_pr[i].squeeze().detach().cpu().numpy(), isVector=False)
            labelTrue = sitk.GetImageFromArray(y_gt[i].squeeze().detach().cpu().numpy(), isVector=False)
            self.hausdorffcomputer.Execute(labelTrue, labelPred)
            hd.append(self.hausdorffcomputer.GetHausdorffDistance())
        return torch.Tensor([np.mean(hd)]).cuda()


class Hausdorff95Score(base.Metric):
    def __init__(self, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        device = y_pr.device
        y_pr = y_pr.detach().cpu().numpy()
        y_gt = y_gt.detach().cpu().numpy()

        hd95score = metric.binary.hd95(y_pr, y_gt)
        return torch.tensor(hd95score).to(device)


class AvgHausdorffScore(base.Metric):
    def __init__(self, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.hausdorffcomputer = sitk.HausdorffDistanceImageFilter()

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)

        labelPred = sitk.GetImageFromArray(y_pr, isVector=False)
        labelTrue = sitk.GetImageFromArray(y_gt, isVector=False)

        self.hausdorffcomputer.Execute(labelTrue, labelPred)
        avghd = self.hausdorffcomputer.GetAverageHausdorffDistance()
        return avghd


def get_args_parser():
    parser = argparse.ArgumentParser('Gamma Task 3 training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--modality', default='US', type=str)
    parser.add_argument('--suffix', default='', type=str)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--thresh', default=0.5, type=float)
    parser.add_argument('--kernel_size', default=25, type=int)
    parser.add_argument('--use_mmg', action='store_true', help='Use margin mask gen')
    parser.add_argument('--filter', choices=['None', 'gaussian', 'median', 'mean',
                        'bilateral'], default='None', help='Use margin mask gen')

    # save checkpoint
    parser.add_argument('--save_checkpoint', default=5, type=int)

    parser.add_argument('--in_channels', default=3, type=int, help='images input channel number')
    # Dataset parameters
    parser.add_argument('--data-path', default='../datasets/us_small', type=str,
                        help='dataset path')
    parser.add_argument('--task',
                        type=str, help='Task number for gamma')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    # Model parameters
    parser.add_argument('--model_name', default='Unet', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--criterion_name', default='DiceLoss', type=str,
                        help='Name of criterion to train')
    parser.add_argument('--outprefx', default='', type=str)
    parser.add_argument('--det_name', default='', type=str)

    # parser.set_defaults(polar_input=True)
    parser.add_argument('--loss_alpha', default=0.5, type=float,
                        help='Alpha')
    parser.add_argument('--encoder', default='resnet50', type=str,
                        help='Name of encoder in the chosen model')
    parser.add_argument('--encoder_weight_name', default='imagenet', type=str,
                        help='Name of encoder weight')
    parser.add_argument('--activation', default=None,
                        type=str,
                        help='activatiuon func to apply after the final layer')

    parser.add_argument('--encoder_depth', default=5, type=int, help='model encoder depth')
    parser.add_argument('--input-size', default=320, type=int, help='images input size')
    parser.add_argument('--nb_classes', default=1, type=int, help='class num')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Model weight update with ema
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-4, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    # parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    parser.add_argument('--test-data-path', default='/data/public-datasets/gamma/val/val_data/multi-modality_images', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument(
        '--resume', default='task3/Key_7.71321_20210804091841_Unet_resnet34_DiceLoss__bs16/checkpoint_min.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--infer', action='store_true', help='Perform infer only')
    parser.add_argument('--test', action='store_true', help='Perform test only')
    parser.add_argument('--tta', action='store_true', help='Perform test time augmentation')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--prefetch_factor', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def setup_seed(seed):
    # torch.manual_seed(seed) #为CPU设置随机种子
    # torch.cuda.manual_seed_all(seed) #为所有GPU设置随机种子
    # np.random.seed(seed) #
    # random.seed(seed)  #若数据读取过程中采用了随机预处理，则python、numpy的随机数生成器也需要设置种子
    random.seed(seed)
    np.random.seed(seed) #
    torch.manual_seed(seed) #为CPU设置随机种子
    torch.cuda.manual_seed_all(seed) #为所有GPU设置随机种子
    
    
    torch.backends.cudnn.deterministic = True


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing_US(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_preprocessing_CEUS(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        # albu.Lambda(image=preprocessing_fn), # why
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def main(args):
    # print(args)
    utils.init_distributed_mode(args)
    if args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    cudnn.benchmark = True  # deterministic will be false

    ######################################## CREATE DATASET ########################################

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        args.encoder, args.encoder_weight_name) if args.encoder in smp.encoders.get_encoder_names() else None
    if args.modality == 'CEUS':
        preprocessing_fn = None
        dataset = breastdatasets.ceus_dataset(image_file=os.path.join(args.data_path, 'train', 'img'),
                                              gt_path=os.path.join(args.data_path, 'train', 'labelcol'),
                                              frame_num=args.in_channels,
                                              augmentation=breastdatasets.get_ceus_train_aug(args),
                                              preprocessing=get_preprocessing_CEUS(preprocessing_fn),
                                              use_mmg=args.model_name if args.use_mmg else None,
                                              filter=args.filter,
                                              kernel_size=args.kernel_size,
                                              suffix=args.suffix
                                              #   preprocessing=None
                                              )
        test_dataset = breastdatasets.ceus_dataset(image_file=os.path.join(args.data_path, 'test', 'img'),
                                                   gt_path=os.path.join(args.data_path, 'test', 'labelcol'),
                                                   frame_num=args.in_channels,
                                                   augmentation=breastdatasets.get_ceus_val_aug(args),
                                                   preprocessing=get_preprocessing_CEUS(preprocessing_fn),
                                                   use_mmg=args.model_name if args.use_mmg else None,
                                                   filter=args.filter,
                                                   kernel_size=args.kernel_size,
                                                   suffix=args.suffix
                                                   )
    else:
        dataset = breastdatasets.us_dataset(image_file=os.path.join(args.data_path, 'train', 'img'),
                                            gt_path=os.path.join(args.data_path, 'train', 'labelcol'),
                                            augmentation=breastdatasets.get_us_train_aug(args.input_size),
                                            preprocessing=get_preprocessing_US(preprocessing_fn),
                                            use_mmg=args.model_name if args.use_mmg else None,
                                            filter=args.filter,
                                            kernel_size=args.kernel_size,
                                            suffix=args.suffix
                                            )
        test_dataset = breastdatasets.us_dataset(image_file=os.path.join(args.data_path, 'test', 'img'),
                                                 gt_path=os.path.join(args.data_path, 'test', 'labelcol'),
                                                 augmentation=breastdatasets.get_us_val_aug(args.input_size),
                                                 preprocessing=get_preprocessing_US(preprocessing_fn),
                                                 use_mmg=args.model_name if args.use_mmg else None,
                                                 filter=args.filter,
                                                 kernel_size=args.kernel_size,
                                                 suffix=args.suffix
                                                 )
    train_size = int(0.8*len(dataset))
    val_size = len(dataset)-train_size
    test_size = 0

    dataset_train, dataset_val, _ = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        prefetch_factor=args.prefetch_factor # loading prefetch_factor*num_works (64) data before training
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=int(1.0 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        prefetch_factor=args.prefetch_factor
    )

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=int(1.0 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        prefetch_factor=args.prefetch_factor
    )

    ######################################## CREATE DATASET ########################################

    ######################################### CREATE MODEL #########################################
    if args.model_name == 'Unet':
        model = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weight_name,
            in_channels=args.in_channels,
            classes=args.nb_classes, # 1
            encoder_depth=args.encoder_depth,
            activation=args.activation
        )
    elif args.model_name == 'FPN':
        model = smp.FPN(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weight_name,
            in_channels=args.in_channels,
            classes=args.nb_classes,
            encoder_depth=args.encoder_depth,
            activation=args.activation
        )
    elif args.model_name == 'PSPNet':
        model = smp.PSPNet(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weight_name,
            in_channels=args.in_channels,
            classes=args.nb_classes,
            # encoder_depth=args.encoder_depth,
            activation=args.activation
        )
    elif args.model_name == 'DeepLabV3':
        model = smp.DeepLabV3(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weight_name,
            in_channels=args.in_channels,
            classes=args.nb_classes,
            encoder_depth=args.encoder_depth,
            activation=args.activation
        )
    elif args.model_name == 'PAN':
        model = smp.PAN(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weight_name,
            in_channels=args.in_channels,
            classes=args.nb_classes,
            # encoder_depth=args.encoder_depth,
            activation=args.activation
        )
    elif args.model_name == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weight_name,
            in_channels=args.in_channels,
            classes=args.nb_classes,
            encoder_depth=args.encoder_depth,
            activation=args.activation
        )
    elif args.model_name == 'TransUnet':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']      

        config_vit.n_classes = args.nb_classes
        config_vit.n_skip = 3
        config_vit.in_channel = args.in_channels # 3
        config_vit.patches.grid = (int(args.input_size / 16),
                                   int(args.input_size / 16))
        # print("test the config of transunet!!!!!!!!!!")                           
        # print(config_vit)
        model = ViT_seg(config_vit, img_size=args.input_size,
                        num_classes=config_vit.n_classes, in_channels=args.in_channels)
        model.load_from(weights=np.load(config_vit.pretrained_path))
        print("test transunet!!!!!!!!!!!!!!!!")
        print(config_vit)
    elif args.model_name == 'SwinUnet':
        config_vit = CONFIGS_ViT_seg['SwinViT']    

        config_vit.n_classes = args.nb_classes
        # config_vit.n_skip = 3
        config_vit.in_channels = args.in_channels + 1# 3
 
        # print("test the config of transunet!!!!!!!!!!")                           
        # print(config_vit)
        model = Swin_ViT_seg(config_vit, img_size=args.input_size,
                        num_classes=config_vit.n_classes)
        model.load_from(config_vit)
        print("test swinunet!!!!!!!!!!!!!!!!")
        print(config_vit)
    # elif args.model_name == 'DpRAN':
    #     config_vit = CONFIGS_ViT_seg['DpRAN']    

    #     config_vit.n_classes = args.nb_classes
    #     # config_vit.n_skip = 3
    #     config_vit.in_channels = args.in_channels + 1# 3
 
    #     # print("test the config of transunet!!!!!!!!!!")                           
    #     # print(config_vit)
    #     model = DPRAN(config_vit, img_size=args.input_size,
    #                     num_classes=config_vit.n_classes)
    #     model.load_from(config_vit)
    #     print("test swinunet!!!!!!!!!!!!!!!!")
    #     print(config_vit)
    
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)
    ######################################### CREATE MODEL #########################################

    ######################################## CREATE OPT CRI ########################################
    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = getattr(losses, args.criterion_name)()
    # 这个loss 有问题 无法收敛 loss 为负数
    # criterion = smp.utils.losses.DiceLoss()

    metrics = [
        DiceScore(), 
        # smp.utils.metrics.IoU(threshold=0.5),
        # HDScore()
        # HausdorffScore(),
        # AvgHausdorffScore(),
        # Hausdorff95Score()
    ]

    ######################################## CREATE OPT CRI ########################################
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=criterion,
        metrics=metrics,
        optimizer=optimizer,
        device=args.device,
        verbose=True, # show details during the training process
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=criterion,
        metrics=metrics,
        device=args.device,
        verbose=True,
    )

    test_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=criterion,
        metrics=metrics,
        device=args.device,
        verbose=True,
    )

    max_score = 0
    min_dice = np.inf

    ##############################################################################
    output_dir = Path(args.output_dir)
    ##############################################################################
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        print("\nEpoch: {}".format(epoch))

        train_logs = train_epoch.run(data_loader_train)
        valid_logs = valid_epoch.run(data_loader_val)

        lr_scheduler.step(epoch)

        torch.cuda.synchronize()

        if min_dice > valid_logs[criterion.__name__]:
        # if max_score < valid_logs['iou_score']:
            # when get better model perform test run
            test_logs = test_epoch.run(data_loader_test)
            min_dice = valid_logs[criterion.__name__]
            # max_score = valid_logs['iou_score']
            if args.save_checkpoint < epoch:
                checkpoint_path = output_dir / ('checkpoint_min_dice_ep%d.pth' % epoch)
                # checkpoint_path = output_dir / ('checkpoint_max_iou_ep%d.pth' % epoch)
                torch.save(model, checkpoint_path)
                print('* Model saved!')
        if (epoch+1) % 500 == 0:
            checkpoint_path = output_dir / ('checkpoint_ep%d.pth' % epoch)
            torch.save(model, checkpoint_path)
            print('* Model saved!')

        ################################################save image to tensorboard##################################################
        model_without_ddp.eval()
        images, masks = next(iter(data_loader_train))
        masks_pred = model_without_ddp(images.to(args.device))

        masks_pred[masks_pred >= 0.5] = 1
        masks_pred[masks_pred < 0.5] = 0
        if args.modality == 'CEUS':
            images = images[0].unsqueeze(1)
            masks = masks[0].unsqueeze(1)
            masks_pred = masks_pred[0].unsqueeze(1)
        writer.add_images("input_images", images, epoch)
        writer.add_images("mask_ground_truth", masks, epoch)
        writer.add_images("mask_prediction", masks_pred.int(), epoch)

        print('TFBorad saved!')
        ################################################save image to tensorboard##################################################

        # #############################tensorboard##########################
        for k, v in train_logs.items():
            writer.add_scalar('train/%s' % k, v, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], epoch)
        for k, v in valid_logs.items():
            writer.add_scalar('val/%s' % k, v, epoch)
        for k, v in test_logs.items():
            writer.add_scalar('test/%s' % k, v, epoch)

        writer.add_scalar('best/dice_loss', min_dice, epoch)
        # ##################################################################


@torch.no_grad()
def eval():
    '''
    task and save task pred image
    '''
    print('* EVAL')

    model_paths = [i for i in os.listdir(args.resume) if i.endswith('.pth') if str(args.epochs) in i]
    for model_path in model_paths:
        print(model_path)
        # load best saved checkpoint
        # model = torch.load(Path(args.resume) / ('checkpoint_min_dice_ep%d.pth' % args.epochs))
        model = torch.load(Path(args.resume) / model_path)
        model.to(args.device)
        model.eval()
        print("* MODEL LOAD")
        thresh = args.thresh
        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            args.encoder, args.encoder_weight_name) if args.encoder in smp.encoders.get_encoder_names() else None
        if args.modality == 'CEUS':
            preprocessing_fn = None
            test_dataset = breastdatasets.ceus_dataset(image_file=os.path.join(args.data_path, 'test', 'img'),
                                                       gt_path=os.path.join(args.data_path, 'test', 'labelcol'),
                                                       frame_num=args.in_channels,
                                                       augmentation=breastdatasets.get_ceus_val_aug(args),
                                                       preprocessing=get_preprocessing_CEUS(preprocessing_fn),
                                                       #    mode='val'
                                                       use_mmg=args.model_name if args.use_mmg else None,
                                                       filter=args.filter,
                                                       kernel_size=args.kernel_size,
                                                       suffix=args.suffix
                                                       )

            data_loader_test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=int(1.0 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
                prefetch_factor=args.prefetch_factor
            )
            criterion = getattr(losses, args.criterion_name)()

            metrics = [
                DiceScore(),
                # smp.utils.metrics.IoU(threshold=0.5),
                # HDScore(),
                # Hausdorff95Score()
            ]
            test_epoch = smp.utils.train.ValidEpoch(
                model,
                loss=criterion,
                metrics=metrics,
                device=args.device,
                verbose=True,
            )
            test_logs = test_epoch.run(data_loader_test)
            print(test_logs)

            # repeat with above
            test_dataset = breastdatasets.ceus_dataset(image_file=os.path.join(args.data_path, 'test', 'img'),
                                                       gt_path=os.path.join(args.data_path, 'test', 'labelcol'),
                                                       frame_num=args.in_channels,
                                                       augmentation=breastdatasets.get_ceus_val_aug(args),
                                                       preprocessing=get_preprocessing_CEUS(preprocessing_fn),
                                                       mode='val',
                                                       use_mmg=args.model_name if args.use_mmg else None,
                                                       filter=args.filter,
                                                       kernel_size=args.kernel_size,
                                                       suffix=args.suffix
                                                       )
        else:
            test_dataset = breastdatasets.us_dataset(image_file=os.path.join(args.data_path, 'test', 'img'),
                                                     gt_path=os.path.join(args.data_path, 'test', 'labelcol'),
                                                     augmentation=breastdatasets.get_us_val_aug(args.input_size),
                                                     preprocessing=get_preprocessing_US(preprocessing_fn),
                                                     #  mode='val',
                                                     use_mmg=args.model_name if args.use_mmg else None,
                                                     filter=args.filter,
                                                     kernel_size=args.kernel_size,
                                                     suffix=args.suffix
                                                     )

            data_loader_test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=int(1.0 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
                prefetch_factor=args.prefetch_factor
            )
            criterion = getattr(losses, args.criterion_name)()
            # 这个loss 有问题 无法收敛 loss 为负数
            # criterion = smp.utils.losses.DiceLoss()

            metrics = [
                DiceScore(),
                # smp.utils.metrics.IoU(threshold=0.5),
                # HDScore(),
                # HausdorffScore(),
                # AvgHausdorffScore(),
                Hausdorff95Score()
            ]
            test_epoch = smp.utils.train.ValidEpoch(
                model,
                loss=criterion,
                metrics=metrics,
                device=args.device,
                verbose=True,
            )
            test_logs = test_epoch.run(data_loader_test)
            print(test_logs)
            test_dataset = breastdatasets.us_dataset(image_file=os.path.join(args.data_path, 'test', 'img'),
                                                     gt_path=os.path.join(args.data_path, 'test', 'labelcol'),
                                                     augmentation=breastdatasets.get_us_val_aug(args.input_size),
                                                     preprocessing=get_preprocessing_US(preprocessing_fn),
                                                     mode='val',
                                                     use_mmg=args.model_name if args.use_mmg else None,
                                                     filter=args.filter,
                                                     kernel_size=args.kernel_size,
                                                     suffix=args.suffix
                                                     )

        device = args.device
        test_dataloader = torch.utils.data.DataLoader(test_dataset)
        print("* DATA LOAD")
        # evaluate model on test set
        dice_scores = []
        hd1_scores = []
        hd2_scores = []

        # image save
        outpath = '%s_%s_th%f_%s_%d_%s_xxz' % (args.task, args.model_name, args.thresh, args.filter, args.kernel_size, args.outprefx)
        if not os.path.exists(outpath):
            os.mkdir(outpath)

        for i, (image, mask_gt, real_ind) in enumerate(test_dataloader):
            image = image.to(device)
            mask_pred = model(image)

            mask_pred_c = Activation(None)(mask_pred) # why 'None' is added
            # mask_pred_c = Activation(mask_pred)
            #####################################################################
            num = mask_gt.shape[0]
            for j in range(num):
                target = mask_gt[j, 0, ...]
                score = mask_pred_c[j, 0, ...].cpu()
                # new added
                # target.resize(320,320)
                # score.resize(320,320)
                #####
                smooth = 1e-5
                intersect = torch.sum(score * target)
                y_sum = torch.sum(target * target)
                z_sum = torch.sum(score * score)
                dice_score = (2 * intersect + smooth) / (z_sum + y_sum + smooth)

                dice_scores.append(dice_score.item())
                hd = hausdorff_distance(target.squeeze().numpy(), score.squeeze().numpy(), distance='euclidean')
                hd1_scores.append(hd)

            #####################################################################

            mask_pred = mask_pred.detach().cpu().squeeze().numpy()
            mask_gt = mask_gt.squeeze().numpy()
            # print(dice_scores)
            # print(hd1_scores)
            # print("###########################")
            # print(mask_pred)
            # print(mask_pred.shape)
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(mask_gt)
            # print(mask_gt.shape)
            # save
            # input_img = image.cpu().squeeze().permute(1, 2, 0).numpy().astype(np.uint8)*255
            mask_gt_img = (mask_gt*255).astype(np.uint8)
            mask_pred_img = ((mask_pred > thresh)*255).astype(np.uint8)

            cv2.imwrite(os.path.join(outpath, '%s_mask_gt.png' % real_ind), mask_gt_img)
            cv2.imwrite(os.path.join(outpath, '%s_mask_pred.png' % real_ind), mask_pred_img)

            labelPred = sitk.GetImageFromArray(mask_pred_img, isVector=False)
            labelTrue = sitk.GetImageFromArray(mask_gt_img, isVector=False)
            hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
            scores = {}
            try:
                hausdorffcomputer.Execute(labelTrue, labelPred)
                scores["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
            except:
                scores["avgHausdorff"] = 10000
            hd2_scores.append(scores["avgHausdorff"])
            # dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
            # try:
            #     dicecomputer.Execute(labelTrue, labelPred)
            #     scores["dice"] = dicecomputer.GetDiceCoefficient()
            # except:
            #     scores["dice"] = 0.1
            print(real_ind, dice_score, hd, scores)
            # s2.append(scores["dice"])
            # dice_score.append(dcscore)
            # hd_score.append(hd95score)
            # break
        # print(sum(dice_score), len(dice_score))
        # print(sum(hd_score), len(hd_score))
        print('* LENGTH: %d' % len(dice_scores), ' DICE: %f' %
              (sum(dice_scores)/len(dice_scores)), ' HD: %f' % (sum(hd1_scores)/len(hd1_scores)),
              ' AVG HD: %f' % (sum(hd2_scores)/len(hd2_scores)))

        print('%f\t%f\t%f\t%s\t%d\t%d\t%s' % ((sum(dice_scores)/len(dice_scores)), (sum(hd1_scores)/len(hd1_scores)),
                                              (sum(hd2_scores)/len(hd2_scores)), args.resume.split('/')[-1], args.epochs, args.kernel_size, args.filter)
              )

'''
@torch.no_grad()
def infer(args):
    
    # 未调用 
    # 需要修改
    
    device = args.device
    data_path = args.test_data_path
    jpeg = TurboJPEG()

    thresh = 0.5

    # load model
    model = build_model(args)
    model.to(device)
    checkpoint = torch.load(os.path.join(args.resume, 'checkpoint_min_dice.pth'), map_location=args.device)
    model.load_state_dict(checkpoint['model'])

    # switch to evaluation mode
    model.eval()
    # TTA
    if args.tta:
        import ttach as tta
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Rotate90(angles=[0, 180]),
                tta.Scale(scales=[1, 2, 4]),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )

        model = tta.SegmentationTTAWrapper(model, transforms)
    file_list = os.listdir(data_path)
    for file in tqdm(file_list):
        print(file)
        if '%s.bmp' % file in os.listdir(args.output_dir):
            continue

        img_path = os.path.join(data_path, file, '%s.jpg' % file)
        if args.det_name != "":
            root_path = img_path.split("/")
            root_path.pop()
            root_path = '/'.join(root_path)
            file_name = [i for i in os.listdir(root_path) if args.det_name in i]
            assert len(file_name), "get too much preprocess files"
            img_path = os.path.join(root_path, file_name[0])

        in_file = open(img_path, 'rb')
        image = jpeg.decode(in_file.read(), pixel_format=TJPF_RGB)
        h, w, _ = image.shape
        in_file.close()

        # trans (W,H,C) to (C,W,H)
        def to_tensor(x, **kwargs):
            return x.transpose(2, 0, 1).astype('float32')[np.newaxis, :, ...] 

        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            args.encoder, args.encoder_weight_name) if args.encoder in smp.encoders.get_encoder_names() else None
        train_transform = [
            albu.Resize(args.input_size, args.input_size),
            albu.CLAHE(p=1),
            albu.RandomBrightnessContrast(p=1),
            albu.MotionBlur(blur_limit=3, p=1),

            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
        trans = albu.Compose(train_transform)
        image = torch.Tensor(trans(image=image)['image']).to(device)
        # pred
        pred = model(image)
        pred = pred.squeeze().cpu().numpy().transpose(1, 2, 0)

        # def normalization(data):
        #     _range = np.max(data) - np.min(data)
        #     return (data - np.min(data)) / _range

        # pred = (normalization(pred)*255).astype(np.uint8)
        # cv2.imwrite('disc_%s.bmp' % file, pred[:, :, 0])
        # print(pred[:, :, 1].shape)
        # cv2.imwrite('cup_%s.bmp' % file, pred[:, :, 1])
        # print(pred.shape, pred.max(), pred.min())

        # sys.exit(111)

        if args.det_name == "":
            inverse_trans = albu.Compose([albu.Resize(h, w)])
            pred = inverse_trans(image=pred)['image']
        else:
            if args.polar_input:
                # inverse polar
                pred = cv2.warpPolar(src=pred, dsize=(args.input_size, args.input_size),
                                     center=(args.input_size//2, args.input_size//2), maxRadius=args.input_size//2,
                                     flags=cv2.WARP_FILL_OUTLIERS+cv2.WARP_POLAR_LINEAR+cv2.WARP_INVERSE_MAP)
            # reshape
            inverse_trans = albu.Compose([albu.Resize(h, w)])
            pred = inverse_trans(image=pred)['image']
            # revise margin pix val
            pred[pred == 255] = 0
            patch_h = h
            patch_w = w

        pred[pred >= thresh] = 255
        pred[pred < thresh] = 0
        # cup
        cup = pred[:, :, 0].astype(np.uint8)
        # cup = utils.KeepMaxArea(cup.astype(np.uint8))
        # cv2.imwrite('cup_%s.bmp' % file, cup)
        # disc
        disc = pred[:, :, 1].astype(np.uint8)
        # disc = utils.FillHole(disc.astype(np.uint8))
        # disc = utils.KeepMaxArea(disc)
        disc[disc > 126] = 127
        # cv2.imwrite('disc_%s.bmp' % file, disc)

        res = 255-np.max(np.stack([cup, disc], axis=2), axis=2)
        if args.det_name != "":
            # fill ori input
            print(img_path)
            path_words = img_path.split('/')
            file_words = path_words[-1].split('_')
            x, y = int(file_words[-2]), int(file_words[-1].replace('.jpg', ''))
            path_words[-1] = path_words[-1].split('_')[0]+'.jpg'
            ori_img_path = '/'.join(path_words)
            in_file = open(ori_img_path, 'rb')
            ori_inp = jpeg.decode(in_file.read(), pixel_format=TJPF_RGB)
            h, w, _ = ori_inp.shape
            in_file.close()
            pred = (np.ones((h, w))*255).astype(np.uint8)
            pred[y:y+patch_h, x:x+patch_w] = res
            res = pred
        cv2.imwrite(os.path.join(args.output_dir, '%s.bmp' % file), res)
        # break
    print('* INFER DONE')
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EXP1 training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # fix random seed
    setup_seed(args.seed)
    
    if args.modality == 'CEUS':
        args.data_path = args.data_path.replace('/us_small', '/ceus_frames')
    elif args.modality == 'US':
        args.data_path = args.data_path.replace('/ceus_frames', '/us_small')

    if args.test:
        import json
        with open(os.path.join(args.resume, 'config.log'), 'r') as f:
            dic = json.load(f)

        args.model_name = dic['model_name']
        args.encoder = dic['encoder']
        args.in_channels = dic['in_channels']
        args.modality = dic['modality']
        if 'kernel_size' in dic:
            args.kernel_size = dic['kernel_size']
        if 'filter' in dic:
            args.filter = dic['filter']
        if 'use_mmg' in dic:
            args.use_mmg = dic['use_mmg']
        if 'frame_num' in dic:
            args.frame_num = dic['frame_num']
        args.task = dic['task']

        args.batch_size = 64

        print(args)
        eval()
        sys.exit(0)
    '''
    if args.infer:
        import json
        with open(os.path.join(args.resume, 'config.log'), 'r') as f:
            dic = json.load(f)

        args.model_name = dic['model_name']
        args.encoder = dic['encoder']
        args.det_name = dic['det_name'] if 'det_name' in dic else ""
        args.polar_input = dic['polar_input'] if 'polar_input' in dic else False
        # args.activation = dic['activation']
        # for unet plus efficient
        # args.activation = 'sigmoid'
        if dic['activation']:
            args.activation = dic['activation']
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        infer(args)
        sys.exit(0)
    '''
    if not args.output_dir:
        args.output_dir = args.outprefx+'%s/%s_%s_%s_%s_img%d_bs%d_fn%d_kn%d_%s' % (args.task, time.strftime(
            "%Y%m%d%H%M%S", time.localtime()), args.model_name, args.encoder, args.criterion_name, args.input_size, args.batch_size, args.in_channels, args.kernel_size, args.filter)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(args.output_dir)

    # write run config
    with open(os.path.join(args.output_dir, 'config.log'), 'a+') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args)
