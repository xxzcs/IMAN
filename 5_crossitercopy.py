
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
import gc
from numpy.core.fromnumeric import nonzero
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vision_transformer import SwinUnet as Swin_ViT_seg
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
from datasets import ceus_dataset, us_dataset

import warnings
warnings.filterwarnings("ignore")

'''

https://github.com/qubvel/segmentation_models.pytorch
https://github.com/CSAILVision/semantic-segmentation-pytorch
多模态大迭代
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
        dice_score = 1-self._dice_loss(y_pr[:, 0, ...], y_gt[:, 0, ...])
        return dice_score


class HausdorffScore(base.Metric):

    __name__ = 'hd_score'

    def __init__(self, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)

    def _hd(self, target, score):
        hd = hausdorff_distance(target, score, distance='euclidean')
        return hd

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        targets = y_gt[:].cpu()
        scores = y_pr.cpu()
        b, _, _, _ = y_pr.shape
        hds = []
        for n in range(b):
            hds.append(self._hd(targets[n].squeeze().numpy(), scores[n].squeeze().numpy()))
        return torch.tensor(sum(hds)/len(hds)).cuda()

class HDScore(base.Metric):

    def __init__(self, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)

    def _hd_score(self, score, target):
        target = target.float()
        hd_score = 0
        # print("test in method!!!!!!!!!!!!!!!!!!!!")
        # print(score)
        # print(score.shape) # torch.Size([16, 224, 224])
        # print(target)
        # print(target.shape) # torch.Size([16, 224, 224])
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
        # print(y_pr.shape) # torch.Size([16, 1, 224, 224])
        # print(y_gt.shape) # torch.Size([16, 1, 224, 224])
        hd_score = self._hd_score(y_pr[:, 0, ...], y_gt[:, 0, ...]) #切片意义
        # hd_score = self._hd_score(y_pr, y_gt) #切片意义

        hd_score_tensor = torch.tensor(hd_score).cuda()    
        return hd_score_tensor

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
        return torch.Tensor([hd95score]).to(device)


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
    parser.add_argument('--batch-size', default=16, type=int)
    # parser.add_argument('--modality', default='US', type=str)
    parser.add_argument('--suffix', default='', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--thresh', default=0.5, type=float)
    parser.add_argument('--total_rounds', default=10, type=int)
    parser.add_argument('--earlystop_interval', default=50, type=int)
    # parser.add_argument('--kernel_size', default=75, type=int)
    parser.add_argument('--use_mmg', action='store_true', help='Use margin mask gen')
    parser.add_argument('--filter', default='gaussian', choices=['None', 'gaussian', 'median', 'mean',
                        'bilateral'], help='Use margin mask gen')
    parser.add_argument('--us_resume', default='xxx', help='resume from checkpoint')
    parser.add_argument('--ceus_resume', default='xxx', help='resume from checkpoint')

    # save checkpoint
    parser.add_argument('--save_checkpoint', default=0, type=int)

    # parser.add_argument('--in_channels', default=3, type=int, help='images input channel number')
    # Dataset parameters
    parser.add_argument('--data-path', default='/data/wutianhao/datasets/us_small', type=str,
                        help='dataset path')
    parser.add_argument('--task', type=str, help='Task number for gamma')
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
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
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
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
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

    # parser.add_argument('--test-data-path', default='/data/public-datasets/gamma/val/val_data/multi-modality_images', type=str,
    #                     help='dataset path')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    # parser.add_argument(
    #     '--resume', default='us/20210907183055_TransUnet_resnext50_32x4d_DiceLoss_bs16/checkpoint_min_dice_ep110.pth', help='resume from checkpoint')
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
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
        # albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


num = 0
@torch.no_grad()
def gen_mmg(modality='US', root_path='../datasets/us_small', resume_path='', round_num=0, model_name='', model=None, suffix=''):
    # define call times of the modality
    global num
    # num = 0
    if modality == 'US':
        '''
        mmg mask
        '''
        kernel_size = 100 #75
        thresh = 0.1
        frame_num = 5 #9 #13 #5

        base_path = os.path.join(root_path, 'us_img_all')
        out_dir = os.path.join('../datasets/ceus_frames', 'ceus_mmg_%s_th%s_kn%d_fn%d_rd%d%s' %
                               (model_name, str(thresh), kernel_size, frame_num, round_num, suffix))
        parser = argparse.ArgumentParser('EXP2 margin mask gen ', parents=[get_args_parser()])
        args = parser.parse_args()
        # args.activation = 'tanh'
        setup_seed(args.seed)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if model is None:
            # print('* USISNG LOADING MODEL TO GEN ON', out_dir)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            print('* MODEL LOADING', resume_path, ' TO GEN ON', out_dir)
            model = torch.load(resume_path)
        else:
            print('* USING MODEL TO GEN ON', out_dir)
            model = model
        model.cuda()
        model.eval()
        print('* MMG START')

        jpeg = TurboJPEG()

        for file in tqdm(os.listdir(base_path)):
            in_file = open(os.path.join(base_path, file), 'rb')
            img = jpeg.decode(in_file.read(), pixel_format=TJPF_RGB)
            img_c = img.copy()
            in_file.close()

            h, w, _ = img.shape

            trans = breastdatasets.get_us_val_aug(args.input_size)

            preprocessing_fn = smp.encoders.get_preprocessing_fn(
                'resnet50', 'imagenet') if 'resnet50' in smp.encoders.get_encoder_names() else None
            process = get_preprocessing_US(preprocessing_fn)

            img = trans(image=img)['image']
            img = process(image=img)['image']
            # new added !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # if round_num == 0:
            #     img = img
            # else:

            b = img[0,:,:]
            b = b[np.newaxis,...]
            # print(b)
            # print(b.shape)           

            img = np.concatenate([img, b], axis=0)

            img = torch.Tensor(img).cuda().unsqueeze(0)
            # print("test_img_shape")
            # print(round_num)
            # print(img.shape) #[1, 4, 224, 224]
            # print("test!!!!!!!!!!!!!!!!!!!!!!")
            pred = model(img)
            # print("after test!!!!!!!!!!!!!!")
            pred = pred.squeeze().detach().cpu().numpy()

            mask_pred_img = ((pred > thresh)*255).astype(np.uint8)

            inverse_trans = albu.Compose([albu.Resize(h, w)])
            mask_pred_img = inverse_trans(image=mask_pred_img)['image']
            # print(mask_pred_img.shape)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            o1 = cv2.morphologyEx(mask_pred_img, cv2.MORPH_OPEN, kernel, iterations=1)

            # cv2.imwrite('open1.jpg', o1)
            inner = utils.erode(o1, kernel_size)
            outer = utils.close(o1, kernel_size)
            # margin
            inner = (255-inner)
            inner[np.isnan(inner)] = 1
            inner[inner == 255] = 1

            margin = outer * inner
            margin[np.isnan(margin)] = 255
            margin = margin.astype(np.uint8)
            of = int(file.replace('.jpg', '').replace('.png', ''))
            cv2.imwrite(os.path.join(out_dir, '%d.png' % of), margin)
        print('* %s -- GEN MMG ON --%s-- AT ROUND %d' % (modality, out_dir, round_num))
    elif modality == 'CEUS':
        base_path = os.path.join(root_path, 'ceus_img_all')

        parser = argparse.ArgumentParser('EXP2 margin mask gen ', parents=[get_args_parser()])
        args = parser.parse_args()
        # args.activation = 'tanh'
        setup_seed(args.seed)

        kernel_size = 75 #100
        thresh = 0.1
        frame_num = 5 #9 #5
        time_step = 3

        out_dir = os.path.join('../datasets/us_small', 'us_mmg_%s_th%s_kn%d_rd%d%s' %
                               (model_name, str(thresh), kernel_size, round_num, suffix))

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if model is None:
            # print('* USISNG LOADING MODEL TO GEN CEUS MMG ON', out_dir)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            print('* MODEL LOADING', resume_path, ' TO GEN ON', out_dir)
            model = torch.load(resume_path)
        else:
            print('* USING MODEL TO GEN ON', out_dir)
            model = model
        model.cuda()
        model.eval()
        print('* MMG START')

        jpeg = TurboJPEG()

        for file in tqdm(os.listdir(base_path)):

            img_list = os.listdir(os.path.join(base_path, file))
            if frame_num < len(img_list):
                images = np.array(img_list)
                center_index = len(images)//2
                start_index = center_index-(frame_num - 1)//2*time_step
                stop_index = center_index+(frame_num - 1)//2*time_step
                indexs = np.linspace(start_index, stop_index, frame_num, dtype=np.int)
                img_list = images[indexs].tolist()
            all_img = []
            # all_mask = []
            for img_name in img_list:
                img = cv2.imread(os.path.join(base_path, file, img_name), 0)[..., np.newaxis]
                all_img.append(img)

            image = np.concatenate(all_img, axis=2)
            # new_added!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # print("test the times called of this method!!!!!!!!!!!!")
            # print(num)
            if round_num == 0 and num < 169:
                image = image
                # print(image.shape)
                args.in_channels = 5 #9
                num += 1
            else: #if (round_num == 0 and num != 0) or round_num != 0:
                args.in_channels = 6 #10
                mid = image[:,:,2]
                mid = mid[...,np.newaxis]

                image = np.concatenate([mid, image], axis = 2)
            # print("after this method!!!!!!!!!!!!")
            # print(num)
            # print(image.shape)
            

            img_c = image.copy()
            # args.activation = 'softmax'
            h, w, _ = img.shape
            # args.in_channels = 13 #5
            trans = breastdatasets.get_ceus_val_aug(args)
            # print(trans, img)
            preprocessing_fn = None
            process = get_preprocessing_CEUS(preprocessing_fn)

            img = trans(image=image)['image']
            img = process(image=img)['image']
            img = torch.Tensor(img).cuda().unsqueeze(0)
            pred = model(img)
            pred = pred.squeeze().detach().cpu().numpy()

            mask_pred_img = ((pred > thresh)*255).astype(np.uint8)

            inverse_trans = albu.Compose([albu.Resize(h, w)])
            mask_pred_img = inverse_trans(image=mask_pred_img)['image']

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            o1 = cv2.morphologyEx(mask_pred_img, cv2.MORPH_OPEN, kernel, iterations=1)

            o1[o1 > 10] = 255
            o1[o1 <= 10] = 0
            inner = utils.erode(o1, kernel_size)
            outer = utils.close(o1, kernel_size)
            # margin
            inner = (255-inner)
            inner[np.isnan(inner)] = 1
            inner[inner == 255] = 1

            margin = outer * inner
            margin[np.isnan(margin)] = 255
            margin = margin.astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, '%05d.jpg' % (int(file))), margin)
        print('* %s -- GEN MMG ON --%s-- AT ROUND %d' % (modality, out_dir, round_num))

# ep_offset represents what?
def one_round(round_num, ep_offset, model, metrics, modality):
    print("*******"*20)
    print(round_num, ep_offset, args)
    print("*******"*20)
    # each round use last round pred to gen mmg
    ######################################## CREATE DATASET ########################################

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        args.encoder, args.encoder_weight_name) if args.encoder in smp.encoders.get_encoder_names() else None
    if modality == 'CEUS':
        preprocessing_fn = None
        args.in_channels = 5 #9
        args.filter = 'mean'#'gaussian'
        dataset = breastdatasets.ceus_dataset(image_file=os.path.join(args.data_path, 'train', 'img'),
                                              gt_path=os.path.join(args.data_path, 'train', 'labelcol'),
                                              frame_num=args.in_channels,
                                              augmentation=breastdatasets.get_ceus_train_aug(args),
                                              preprocessing=get_preprocessing_CEUS(preprocessing_fn),
                                              use_mmg=args.model_name if args.use_mmg else None,
                                              filter=args.filter,
                                              kernel_size=args.kernel_size,
                                              round_num=round_num,
                                              suffix=args.suffix
                                              )
        test_dataset = breastdatasets.ceus_dataset(image_file=os.path.join(args.data_path, 'test', 'img'),
                                                   gt_path=os.path.join(args.data_path, 'test', 'labelcol'),
                                                   frame_num=args.in_channels,
                                                   augmentation=breastdatasets.get_ceus_val_aug(args),
                                                   preprocessing=get_preprocessing_CEUS(preprocessing_fn),
                                                   use_mmg=args.model_name if args.use_mmg else None,
                                                   filter=args.filter,
                                                   kernel_size=args.kernel_size,
                                                   round_num=round_num,
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
                                            round_num=round_num,
                                            suffix=args.suffix
                                            )
        test_dataset = breastdatasets.us_dataset(image_file=os.path.join(args.data_path, 'test', 'img'),
                                                 gt_path=os.path.join(args.data_path, 'test', 'labelcol'),
                                                 augmentation=breastdatasets.get_us_val_aug(args.input_size),
                                                 preprocessing=get_preprocessing_US(preprocessing_fn),
                                                 use_mmg=args.model_name if args.use_mmg else None,
                                                 filter=args.filter,
                                                 kernel_size=args.kernel_size,
                                                 round_num=round_num,
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
        prefetch_factor=args.prefetch_factor
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

    model_without_ddp = model
    # print(model)
    ######################################## CREATE OPT CRI ########################################

    # param = [
    #     # {'params': model.encoder.parameters(), 'lr': args.lr},
    #     {'params': model.decoder.parameters(), 'lr': args.lr*10},
    # ]
    param = model.parameters()
    optimizer = optim.AdamW(param, args.lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    # optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = getattr(losses, args.criterion_name)()
    # 这个loss 有问题 无法收敛 loss 为负数
    # criterion = smp.utils.losses.DiceLoss()

    ######################################## CREATE OPT CRI ########################################
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=criterion,
        metrics=metrics,
        optimizer=optimizer,
        device=args.device,
        verbose=False,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=criterion,
        metrics=metrics,
        device=args.device,
        verbose=False,
    )

    test_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=criterion,
        metrics=metrics,
        device=args.device,
        verbose=False,
    )

    max_score = 0
    min_dice = np.inf

    ##############################################################################
    output_dir = Path(args.output_dir)
    ##############################################################################
    print(f"* TRAIN ROUND %d" % round_num)
    cnt = 0
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        # print("\nEpoch: {}".format(epoch))

        train_logs = train_epoch.run(data_loader_train)
        valid_logs = valid_epoch.run(data_loader_val)

        lr_scheduler.step(epoch)

        torch.cuda.synchronize()

        if min_dice > valid_logs[criterion.__name__]:
            # when get better model perform test run
            test_logs = test_epoch.run(data_loader_test)
            print(test_logs)
            min_dice = valid_logs[criterion.__name__]
            if args.save_checkpoint <= epoch:
                checkpoint_path = output_dir / ('%s_checkpoint_min_dice_ep%d_rd%d.pth' %
                                                (modality.lower(), epoch+ep_offset, round_num))
                torch.save(model, checkpoint_path)
                print('* Model saved!')
            cnt = 0
        else:
            cnt += 1

        ################################################save image to tensorboard##################################################
        model_without_ddp.eval()
        images, masks = next(iter(data_loader_train))
        masks_pred = model_without_ddp(images.to(args.device))

        masks_pred[masks_pred >= 0.5] = 1
        masks_pred[masks_pred < 0.5] = 0
        if modality == 'CEUS':
            images = images[0].unsqueeze(1)
            masks = masks[0].unsqueeze(1)
            masks_pred = masks_pred[0].unsqueeze(1)
        writer.add_images("%s_input_images" % modality, images, (epoch+ep_offset))
        writer.add_images("%s_mask_ground_truth" % modality, masks, (epoch+ep_offset))
        writer.add_images("%s_mask_prediction" % modality, masks_pred.int(), (epoch+ep_offset))
        writer.add_scalar('%s_best/restart' % modality, round_num,  (epoch+ep_offset))

        # print('* TFBorad saved!')
        ################################################save image to tensorboard##################################################

        # #############################tensorboard##########################
        for k, v in train_logs.items():
            writer.add_scalar('%s_train/%s' % (modality, k), v, (epoch+ep_offset))
        writer.add_scalar('%s_train/lr' % modality, optimizer.param_groups[0]["lr"], (epoch+ep_offset))
        for k, v in valid_logs.items():
            writer.add_scalar('%s_val/%s' % (modality, k), v, (epoch+ep_offset))
        for k, v in test_logs.items():
            writer.add_scalar('%s_test/%s' % (modality, k), v, (epoch+ep_offset))

        writer.add_scalar('%s_best/dice_loss' % modality, min_dice, (epoch+ep_offset))
        # ##################################################################

        ################################################ early stop ################################################
        if cnt >= args.earlystop_interval:
            print('* ROUND %d\tEARLY STOP AT %d\t RETURN %d' % (round_num, epoch, epoch+ep_offset+1))
            return epoch+ep_offset+1
            break

        ################################################ early stop ################################################
    # args.lr = optimizer.param_groups[0]["lr"]
    print('* ROUND %d\tNO EARLY STOP RUN TOTAL %d\t RETURN %d' % (round_num, args.epochs-1, args.epochs+ep_offset))
    return args.epochs+ep_offset


@torch.no_grad()
def eval():
    '''
    task and save task pred image
    '''
    print('* EVAL')

    # load best saved checkpoint
    pth_name = [i for i in os.listdir(args.resume) if '_ep%d_' % args.epochs in i][0]
    print(pth_name)
    round_num = int(pth_name.split('_')[-1].replace('.pth', '').replace('rd', ''))

    ################################################
    if round_num != 0:
        print('* GEN MMG FOR ROUND ', round_num)

        mmg_model_path = [i for i in os.listdir(args.resume) if 'rd%d' % (round_num-1) in i]
        mmg_model_path.sort(key=lambda x: int(x.split('_')[-2].replace('ep', '')))
        model = torch.load(os.path.join(args.resume, mmg_model_path[-1]))
        print("* MMG WITH", mmg_model_path[-1])
        model.cuda()
        model.eval()
        resume_path = ""
        gen_mmg(modality=args.modality, root_path=args.data_path,
                resume_path=resume_path, round_num=round_num, model_name=args.model_name, model=model, suffix=args.suffix)

    ################################################

    model = torch.load(os.path.join(args.resume, pth_name))
    model.to(args.device)
    model.eval()
    print("* MODEL LOAD")
    thresh = args.thresh
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        args.encoder, args.encoder_weight_name) if args.encoder in smp.encoders.get_encoder_names() else None
    if args.modality == 'CEUS':
        preprocessing_fn = None
        args.in_channels = 5 #9
        args.filter = 'mean'#'gaussian'
        test_dataset = breastdatasets.ceus_dataset(image_file=os.path.join(args.data_path, 'test', 'img'),
                                                   gt_path=os.path.join(args.data_path, 'test', 'labelcol'),
                                                   frame_num=args.in_channels,
                                                   augmentation=breastdatasets.get_ceus_val_aug(args),
                                                   preprocessing=get_preprocessing_CEUS(preprocessing_fn),
                                                   use_mmg=args.model_name if args.use_mmg else None,
                                                   filter=args.filter,
                                                   kernel_size=args.kernel_size,
                                                   round_num=round_num,
                                                   suffix=args.suffix
                                                   )
        print(len(test_dataset))
        data_loader_test = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=int(1.0 * args.batch_size),
            # batch_size=int(16),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            prefetch_factor=args.prefetch_factor
        )
        criterion = getattr(losses, args.criterion_name)()

        metrics = [
            DiceScore(),
            HausdorffScore(),
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
        test_dataset = breastdatasets.ceus_dataset(image_file=os.path.join(args.data_path, 'test', 'img'),
                                                   gt_path=os.path.join(args.data_path, 'test', 'labelcol'),
                                                   frame_num=args.in_channels,
                                                   augmentation=breastdatasets.get_ceus_val_aug(args),
                                                   preprocessing=get_preprocessing_CEUS(preprocessing_fn),
                                                   use_mmg=args.model_name if args.use_mmg else None,
                                                   filter=args.filter,
                                                   mode='val',
                                                   kernel_size=args.kernel_size,
                                                   round_num=round_num,
                                                   suffix=args.suffix
                                                   )
    else:
        test_dataset = breastdatasets.us_dataset(image_file=os.path.join(args.data_path, 'test', 'img'),
                                                 gt_path=os.path.join(args.data_path, 'test', 'labelcol'),
                                                 augmentation=breastdatasets.get_us_val_aug(args.input_size),
                                                 preprocessing=get_preprocessing_US(preprocessing_fn),
                                                 use_mmg=args.model_name if args.use_mmg else None,
                                                 filter=args.filter,
                                                 kernel_size=args.kernel_size,
                                                 round_num=round_num,
                                                 suffix=args.suffix
                                                 #  suffix='lrddcnewmodel'
                                                 #  / mnt/nfs-storage/wutianhao/datasets/us_small/us_mmg_TransUnet_th0.1_kn75_rd9lrddcnewmodel
                                                 )
        print(len(test_dataset))
        data_loader_test = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=int(1.0 * args.batch_size),
            # batch_size=int(16),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            prefetch_factor=args.prefetch_factor
        )
        criterion = getattr(losses, args.criterion_name)()

        metrics = [
            DiceScore(),
            HausdorffScore(),
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
        # sys.exit(0)
        test_dataset = breastdatasets.us_dataset(image_file=os.path.join(args.data_path, 'test', 'img'),
                                                 gt_path=os.path.join(args.data_path, 'test', 'labelcol'),
                                                 augmentation=breastdatasets.get_us_val_aug(args.input_size),
                                                 preprocessing=get_preprocessing_US(preprocessing_fn),
                                                 mode='val',
                                                 use_mmg=args.model_name if args.use_mmg else None,
                                                 filter=args.filter,
                                                 kernel_size=args.kernel_size,
                                                 round_num=round_num,
                                                 suffix=args.suffix
                                                 )
    device = args.device
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  num_workers=8,
                                                  pin_memory=True)
    print("* DATA LOAD")
    # evaluate model on test set
    dice_scores = []
    hd1_scores = []
    hd2_scores = []

    # image save
    outpath = './%s_%s_th%f' % (args.task, args.model_name, args.thresh)
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    for i, (image, mask_gt, real_ind) in enumerate(test_dataloader):
        print(image.shape)
        image = image.to(device)
        mask_pred = model(image)

        mask_pred_c = Activation(None)(mask_pred)
        #####################################################################
        num = mask_gt.shape[1]
        print(mask_gt.shape)
        for j in range(num):

            target = mask_gt[j, 0, ...]
            score = mask_pred_c[j, 0, ...].cpu()
            smooth = 1e-5
            intersect = torch.sum(score * target)
            y_sum = torch.sum(target * target)
            z_sum = torch.sum(score * score)
            dice_score = (2 * intersect + smooth) / (z_sum + y_sum + smooth)

            dice_scores.append(dice_score.item())
            hd = hausdorff_distance(target.squeeze().numpy(), score.squeeze().numpy(), distance='euclidean')
            hd1_scores.append(hd)

        #####################################################################

        # mask_pred = mask_pred.detach().cpu().squeeze().numpy()
        # mask_gt = mask_gt.squeeze().numpy()

        # # save
        # # input_img = image.cpu().squeeze().permute(1, 2, 0).numpy().astype(np.uint8)*255
        # mask_gt_img = (mask_gt*255).astype(np.uint8)
        # mask_pred_img = ((mask_pred > thresh)*255).astype(np.uint8)

        # cv2.imwrite(os.path.join(outpath, '%s_mask_gt.png' % real_ind), mask_gt_img)
        # cv2.imwrite(os.path.join(outpath, '%s_mask_pred.png' % real_ind), mask_pred_img)

        # labelPred = sitk.GetImageFromArray(mask_pred_img, isVector=False)
        # labelTrue = sitk.GetImageFromArray(mask_gt_img, isVector=False)
        # hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
        # scores = {}
        # try:
        #     hausdorffcomputer.Execute(labelTrue, labelPred)
        #     scores["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
        # except:
        #     scores["avgHausdorff"] = 10000
        # hd2_scores.append(scores["avgHausdorff"])

        # print(real_ind, dice_score, hd, scores)

    # print(sum(dice_score), len(dice_score))
    # print(sum(hd_score), len(hd_score))
    print('* LENGTH: %d' % len(dice_scores), ' DICE: %f' %
          (sum(dice_scores)/len(dice_scores)), ' HD: %f' % (sum(hd1_scores)/len(hd1_scores)),
          #   ' AVG HD: %f' % (sum(hd2_scores)/len(hd2_scores))
          )

    print('%f\t%f\t%f\t%s\t%d\t%d\t%s' % ((sum(dice_scores)/len(dice_scores)), (sum(hd1_scores)/len(hd1_scores)),
                                          #   (sum(hd2_scores)/len(hd2_scores)),
                                          0,
                                          args.resume.split('/')[-1], args.epochs, args.kernel_size, args.filter)
          )


def create_model(args):
    if args.model_name == 'Unet':
        model = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weight_name,
            in_channels=args.in_channels,
            classes=args.nb_classes,
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
        config_vit.in_channel = args.in_channels
        config_vit.patches.grid = (int(args.input_size / 16),
                                   int(args.input_size / 16))
        model = ViT_seg(config_vit, img_size=args.input_size,
                        num_classes=config_vit.n_classes, in_channels=args.in_channels)
        model.load_from(weights=np.load(config_vit.pretrained_path))
    elif args.model_name == 'SwinUnet':
        config_vit = CONFIGS_ViT_seg['SwinViT']    

        config_vit.n_classes = args.nb_classes
        # config_vit.n_skip = 3
        if args.modality == 'US':
            args.in_channels = 4
        else:
            args.in_channels = 6 #10
        config_vit.in_channels = args.in_channels #+ 1 # 3
 
        # print("test the config of transunet!!!!!!!!!!")                           
        # print(config_vit)
        model = Swin_ViT_seg(config_vit, img_size=args.input_size,
                        num_classes=config_vit.n_classes)
        model.load_from(config_vit)
        print("test swinunet!!!!!!!!!!!!!!!!")
        print(config_vit)
    return model


def main(args):

    metrics = [
        DiceScore(),
    ]

    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    cudnn.benchmark = True
    n_gpu = torch.cuda.device_count()

    us_ep_offset = 0
    ceus_ep_offset = 0
    for r in range(args.total_rounds):

        print('* ITER AT ROUND %d' % r)

        ######################################### GEN MMG #########################################
        if r == 0:
            # args.modality = 'US'
            # args.data_path = '/mnt/nfs-storage/wutianhao/datasets/us_small'
            # args.resume_path = args.us_resume
            # args.kernel_size = 75
            # args.in_channels = 3
            # gen_mmg(modality=args.modality, root_path=args.data_path,
            #         resume_path=args.us_resume, round_num=r, model_name=args.model_name, suffix=args.suffix)
            args.modality = 'CEUS'
            args.data_path = '../datasets/ceus_frames'
            args.resume_path = args.ceus_resume
            args.kernel_size = 100 #75 # 100
            args.in_channels = 5 #9 #5
            gen_mmg(modality=args.modality, root_path=args.data_path,
                    resume_path=args.ceus_resume, round_num=r, model_name=args.model_name, suffix=args.suffix)

        ######################################################## US ########################################################
        args.modality = 'US'
        args.data_path = '../datasets/us_small'
        args.resume_path = args.us_resume
        args.kernel_size = 75 #100 #75
        args.in_channels = 3

        if r == 0:
            ######################################### CREATE MODEL #########################################
            model = create_model(args).to(device)
        else:
            ########################################## LOAD MODEL ##########################################
            models_list = [int(i.replace('us_checkpoint_min_dice_ep', '').replace('_rd%d.pth' % (r-1), ''))
                           for i in os.listdir(args.output_dir) if (i.endswith('rd%d.pth' % (r-1)) and i.startswith('us_'))]
            max_ep = max(models_list)
            model_path = os.path.join(args.output_dir, 'us_checkpoint_min_dice_ep%d_rd%d.pth' % (max_ep, r-1))
            model = torch.load(model_path).to(device)
            # gen_mmg(modality=args.modality, root_path=args.data_path,
            #         resume_path='', round_num=r, model_name=args.model_name, model=model, suffix=args.suffix)
            print('* LOADING MODEL', model_path, ' ', args.modality, r)
        us_ep_offset = one_round(r, us_ep_offset, model, metrics, modality=args.modality)

        gen_mmg(modality=args.modality, root_path=args.data_path,
                model=model, round_num=r, model_name=args.model_name, suffix=args.suffix)

        ####################################################### CEUS #######################################################
        args.modality = 'CEUS'
        args.data_path = '../datasets/ceus_frames'
        args.resume_path = args.ceus_resume
        args.kernel_size = 100 #75 #50 #100
        args.in_channels = 5 #9 #5

        if r == 0:
            ######################################### CREATE MODEL #########################################
            model = create_model(args).to(device)
            # ceus_model = create_model(args).to('cuda:1')
            ######################################### CREATE MODEL #########################################
        else:
            ########################################## LOAD MODEL ##########################################
            models_list = [int(i.replace('ceus_checkpoint_min_dice_ep', '').replace('_rd%d.pth' % (r-1), ''))
                           for i in os.listdir(args.output_dir) if (i.endswith('rd%d.pth' % (r-1)) and i.startswith('ceus_'))]
            max_ep = max(models_list)
            model_path = os.path.join(args.output_dir, 'ceus_checkpoint_min_dice_ep%d_rd%d.pth' % (max_ep, r-1))
            model = torch.load(model_path).to(device)
            # gen_mmg(modality=args.modality, root_path=args.data_path,
            #         resume_path='', round_num=r, model_name=args.model_name, model=model, suffix=args.suffix)
            print('* LOADING MODEL', model_path, ' ', args.modality, r)
        ceus_ep_offset = one_round(r, ceus_ep_offset, model, metrics, modality=args.modality)

        gen_mmg(modality=args.modality, root_path=args.data_path,
                model=model, round_num=(r+1), model_name=args.model_name, suffix=args.suffix)
        # adjust lr
        # args.lr = args.lr * 0.5
        # args.warmup_lr = args.lr
        # args.min_lr = args.min_lr * 0.1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EXP3 training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # fix random seed
    setup_seed(args.seed)

    # num = 0
    # if args.modality == 'CEUS':
    #     args.data_path = args.data_path.replace('/us_small', '/ceus_frames')
    # elif args.modality == 'US':
    #     args.data_path = args.data_path.replace('/ceus_frames', '/us_small')

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
        args.task = dic['task']
        if 'use_mmg' in dic:
            args.use_mmg = dic['use_mmg']

        args.batch_size = 64

        print(args)
        eval()
        sys.exit(0)

    if not args.output_dir:
        args.output_dir = args.outprefx+'%s/%s_%s_%s_%s_bs%d_%s' % (args.task, time.strftime(
            "%Y%m%d%H%M%S", time.localtime()), args.model_name, args.encoder, args.criterion_name, args.batch_size, args.suffix)
    print("* OUTDIR ", args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(args.output_dir)

    # write run config
    with open(os.path.join(args.output_dir, 'config.log'), 'a+') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args)
