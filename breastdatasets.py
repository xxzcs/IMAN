import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
import torch.nn as nn
import torch.nn.functional as F

# from gitmodules.volumentations import volumentations as volu
from turbojpeg import (TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE, TJFLAG_PROGRESSIVE,
                       TJPF_GRAY, TJPF_RGB, TJSAMP_GRAY, TurboJPEG)
import albumentations as albu
from torch.utils.data import DataLoader


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def get_us_train_aug(input_size):
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Resize(input_size, input_size),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.GaussNoise(p=0.2),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_us_val_aug(input_size):
    val_transform = [
        albu.Resize(input_size, input_size),
        albu.CLAHE(p=1),
        albu.Sharpen(p=1),
    ]
    return albu.Compose(val_transform)


def get_ceus_train_aug(args):
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Resize(args.input_size, args.input_size),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.GaussNoise(p=0.2),

        albu.OneOf(
            [
                # albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1), #using gamma transformation
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                # albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        # mean and std for gray
        # https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        albu.Normalize(mean=[0.44531]*args.in_channels, std=[0.26924]*args.in_channels, always_apply=True),
        # albu.Normalize(mean=[0.44531]*(args.in_channels-1), std=[0.26924]*(args.in_channels-1), always_apply=True),
    ]
    return albu.Compose(train_transform)


def get_ceus_val_aug(args):

    val_transform = [
        albu.Resize(args.input_size, args.input_size),
        # albu.CLAHE(p=1),
        albu.Sharpen(p=1),
        albu.Normalize(mean=[0.44531]*args.in_channels, std=[0.26924]*args.in_channels, always_apply=True),
        # albu.Normalize(mean=[0.44531]*(args.in_channels-1), std=[0.26924]*(args.in_channels-1), always_apply=True),
    ]
    return albu.Compose(val_transform)
    # pad_size = (1, int(args.input_size*1.2), int(args.input_size*1.2))
    # patch_size = (1, int(args.input_size*1), int(args.input_size*1))
    # val_transform = [
    #     volu.Resize(pad_size, always_apply=True),
    #     volu.CenterCrop(patch_size, always_apply=True),
    #     volu.Normalize(always_apply=True),
    # ]
    # return volu.Compose(val_transform)


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class ceus_dataset(Dataset):
    def __init__(self, image_file, gt_path=None,  mode='train', augmentation=None,
                 preprocessing=None, frame_num=1, use_mmg=None, filter=None, kernel_size=25, round_num=0, suffix=''):
        super(ceus_dataset, self).__init__()
        self.filter = filter
        self.mode = mode
        self.image_path = image_file
        # image_idxs = os.listdir(image_file)
        self.gt_path = gt_path

        # self.img_path = os.path.join(dataset_path, mode, 'img')
        # self.mask_path = os.path.join(dataset_path, mode, 'labelcol')
        self.class_values = [1]

        self.file_list = os.listdir(self.image_path) # no shuffle in ceus

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.frame_num = frame_num
        self.jpeg = TurboJPEG()
        self.time_step = 3
        self.use_mmg = use_mmg
        if self.use_mmg:
            self.mmg_mask_path = image_file.replace(
                # 'train/img', 'ceus_mmg_%s_th0.1_kn%d_fn%d%s' % (self.use_mmg, kernel_size, frame_num, suffix))
                # 'train/img', 'ceus_mmg_%s_th0.1_kn%d_fn13%s' % (self.use_mmg, kernel_size, suffix))
                'train/img', 'ceus_mmg_%s_th0.1_kn%d_fn%d_rd%d%s' % (self.use_mmg, kernel_size, frame_num, round_num, suffix))
            self.mmg_mask_path = self.mmg_mask_path.replace(
                # 'test/img', 'ceus_mmg_%s_th0.1_kn%d_fn%d%s' % (self.use_mmg, kernel_size, frame_num, suffix))
                # 'test/img', 'ceus_mmg_%s_th0.1_kn%d_fn13%s' % (self.use_mmg, kernel_size, suffix))
                'test/img', 'ceus_mmg_%s_th0.1_kn%d_fn%d_rd%d%s' % (self.use_mmg, kernel_size, frame_num, round_num, suffix))
            print("* Use Margin Mask Generator", self.mmg_mask_path)
        print("* FRAME NUM", self.frame_num)

    def __getitem__(self, idx):
        real_ind = int(self.file_list[idx])
        mask = cv2.imread(os.path.join(self.gt_path, '%05d.png' % real_ind), 0)
        h, w = mask.shape
        mask[mask > 0] = 1
        # convert mask to the ont-hot encoding
        masks = [mask] # e.g., 578*469*2
        # mask = mask[..., np.newaxis]
        # print("test!!!!!!!!!!!!!!!!!!!!!!!1")
        # print(mask) 
        # print(mask.shape) #578*469
        # print(masks)
        # arr_masks = np.array(masks) 
        # print(arr_masks.shape) # 578*469*1
        if self.use_mmg:
            
            # print(os.path.join(self.mmg_mask_path, '%d.png' % real_ind))
            margin_mask = cv2.imread(os.path.join(self.mmg_mask_path, '%d.png' % real_ind), 0)
            # print('xxxxx', os.path.join(self.mmg_mask_path, '%d.png' % real_ind))
            # margin_mask = cv2.resize(margin_mask, mask.shape, interpolation=cv2.INTER_LINEAR)
            # print(os.path.join(self.mmg_mask_path, '%d.png' % real_ind))
            margin_mask[margin_mask < 10] = 0
            margin_mask[margin_mask >= 10] = 1
            margin_mask = cv2.resize(margin_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            # print(margin_mask)
            # print(margin_mask.shape)
            # margin_mask = F.interpolate(margin_mask, size=(h, w), mode='bilinear', align_corners=False)
            masks.append(margin_mask)
            # print("masksssssssssssssssssss")
            # print(masks)
            # arr_masks = np.array(masks)
            # print(arr_masks.shape)
        mask = np.stack(masks, axis=-1).astype(np.uint8)
        # print("maskkkkkkkk after stack")
        # print(mask)
        # print(mask.shape)  # 578*469*2

        img_list = os.listdir(os.path.join(self.image_path, str(real_ind)))
        # print("all imagesssssssssssssssss1")
        # print(img_list)
        # print(len(img_list))
        if self.frame_num < len(img_list):
            images = np.array(img_list)
            center_index = len(images)//2
            start_index = center_index-(self.frame_num - 1)//2*self.time_step
            stop_index = center_index+(self.frame_num - 1)//2*self.time_step
            indexs = np.linspace(start_index, stop_index, self.frame_num, dtype=np.int)
            img_list = images[indexs].tolist()

        # print("all imagesssssssssssssssss2")
        # print(img_list)

        all_img = []
        # all_mask = []
        for img_name in img_list:
            img = cv2.imread(os.path.join(self.image_path, str(real_ind), img_name), 0)[..., np.newaxis]
            all_img.append(img)
            # all_mask.append(mask)
        image = np.concatenate(all_img, axis=2)
        
        # mask = np.concatenate(all_mask, axis=2).astype(np.uint8)[np.newaxis, ...]
        # designed for the frame_num=1
        # if image.shape[2] == 1:
        #     image = np.concatenate([image,image,image], axis=2)
        # apply augmentations
       
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        # print(image.shape, mask.shape)
        # apply preprocessing (to tensor)
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        # print("mask after aug and preprocessing...")
        # print(mask)

        if self.use_mmg:
            margin = mask[1, ...]
            mask = mask[:1, ...]
            # print("margin of maskssssssss")
            # print(margin)
            # print(margin.shape)
            # print(mask)
            # print(mask.shape)
            if self.filter == 'None':
                pass
            elif self.filter == 'median':
                # 中值
                margin = cv2.medianBlur(margin, 75)
            elif self.filter == 'bilateral':
                # 双边
                margin = cv2.bilateralFilter(margin, 75, 75, 75)
            elif self.filter == 'gaussian':
                # 高斯
                margin = cv2.GaussianBlur(margin, (75, 75), 0)
            elif self.filter == 'mean':
                # 均值
                margin = cv2.blur(margin, (75, 75))
            margin = margin[np.newaxis, ...]
            image = np.concatenate([image, margin], axis=0)
        # print(image.shape) #(14,224,224)
        # print(mask.shape) #(1,224,224)

        if self.mode == 'val':
            return image, mask, real_ind
        if self.mode == 'test':
            # During the testing process,
            # the sample returns fundus image, sample name,
            # height and width of the original image
            return image, real_ind, h, w
        if self.mode == 'train':
            # During the training process, the sample returns fundus image and its corresponding ground truth
            return image, mask

    def __len__(self):
        return len(self.file_list)


class us_dataset(Dataset):
    def __init__(self, image_file, gt_path=None, filelists=None, mode='train', augmentation=None,
                 preprocessing=None, frame_num=1, use_mmg=None, filter=None, kernel_size=25,
                 round_num=0, suffix=''):
        super(us_dataset, self).__init__()
        self.filter = filter
        self.mode = mode
        self.image_path = image_file
        # 0001, fundus_img in the folder 0001
        image_idxs = [i for i in os.listdir(self.image_path) if '.jpg' in i or '.png' in i]
        self.gt_path = gt_path

        self.class_values = [1]

        self.file_list = [image_idxs[i] for i in range(len(image_idxs))]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item in filelists]
        # print(len(self.file_list))

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.frame_num = frame_num
        self.jpeg = TurboJPEG()
        self.use_mmg = use_mmg
        if self.use_mmg:
            self.mmg_mask_path = image_file.replace(
                # 'train/img', 'us_mmg_%s_th0.1_kn%d%s' % (self.use_mmg, kernel_size, suffix))
                'train/img', 'us_mmg_%s_th0.1_kn%d_rd%d%s' % (self.use_mmg, kernel_size, round_num, suffix))
            self.mmg_mask_path = self.mmg_mask_path.replace(
                # 'test/img', 'us_mmg_%s_th0.1_kn%d%s' % (self.use_mmg, kernel_size, suffix))
                'test/img', 'us_mmg_%s_th0.1_kn%d_rd%d%s' % (self.use_mmg, kernel_size, round_num, suffix))
            print("* LOADING Margin Mask Generator", self.mmg_mask_path)
        print("* FRAME NUM", self.frame_num)

    def __getitem__(self, idx):
        real_index = self.file_list[idx]
        img_path = os.path.join(self.image_path, real_index)
        # image = cv2.imread(img_path)
        in_file = open(img_path, 'rb')
        image = self.jpeg.decode(in_file.read(), pixel_format=TJPF_RGB)
        in_file.close()
        h, w, c = image.shape

        gt_tmp_path = os.path.join(self.gt_path, str(real_index)).replace('.jpg', '.png')
        # print("########################## ori mask")
        mask = cv2.imread(gt_tmp_path, 0)
        # print(gt_tmp_path)
        # print(mask)
        # print(mask.shape) #(224, 224)
        # in_file = open(gt_tmp_path, 'rb')
        # mask = self.jpeg.decode(in_file.read(), pixel_format=TJPF_GRAY)
        # in_file.close()
        mask[mask == 255] = 1
        masks = [(mask == v) for v in self.class_values]
        # print(masks)
        arr_masks = np.array(masks)
        # print(arr_masks.shape) #(1, 224, 224)
        if self.use_mmg:
            margin_mask = cv2.imread(os.path.join(self.mmg_mask_path, real_index), 0)
            margin_mask = cv2.resize(margin_mask, mask.shape, interpolation=cv2.INTER_LINEAR)
            margin_mask[margin_mask < 10] = 0
            margin_mask[margin_mask >= 10] = 1
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! use mmg")
            # print(margin_mask) 
            # print(margin_mask.shape) #(224, 224)
            masks.append(margin_mask)
        mask = np.stack(masks, axis=-1).astype(np.uint8)
        # print(mask)
        # print(mask.shape) #(224, 224, 2)
        # mask = masks[0]
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        # print(image.shape, mask.shape)  # (224, 224, 3) (224, 224, 2)
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # print('TR', image.shape, mask.shape) # (3, 224, 224) (2, 224, 224)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! after preprocessing")
        # print(mask)
        if self.use_mmg:
            margin = mask[1, ...]
            mask = mask[:1, ...]
            # print(margin) 
            # print(margin.shape) #(224, 224)
            # print(mask)
            # print(mask.shape) # (1, 224, 224)

            if self.filter == 'None':
                pass
            elif self.filter == 'median':
                # 中值
                margin = cv2.medianBlur(margin, 75)
            elif self.filter == 'bilateral':
                # 双边
                margin = cv2.bilateralFilter(margin, 75, 75, 75)
            elif self.filter == 'gaussian':
                # 高斯
                margin = cv2.GaussianBlur(margin, (75, 75), 0)
            elif self.filter == 'mean':
                # 均值
                margin = cv2.blur(margin, (75, 75))
            # print("!!!!!!!!!!!!!!!!!!!!!! after filter")
            # print(mask.shape) #(1, 224, 224)
            # print(image.shape) #(3, 224, 224)
            # print(margin)
            # print(margin.shape) #(224, 224)
            margin = margin[np.newaxis, ...]
            image = np.concatenate([image, margin], axis=0)

            # margin_3channel = np.concatenate([margin, margin, margin], axis=0)
            # image = cv2.multiply(image, margin_3channel)

            # print("!!!!!!!!!!!!!!!!!!!!!! margin")
            # # print(margin)
            # print(margin.shape) #(1, 224, 224)
            # print(image.shape) #(4, 224, 224)
        # print('pro', image.shape, mask.shape)
        if self.mode == 'val':
            return image, mask, real_index
        if self.mode == 'test':
            return image, real_index, h, w
        if self.mode == 'train':
            return image, mask

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    dataset =us_dataset(image_file='/data/wutianhao/datasets/us_small/train/img')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for i_batch, batch_data in enumerate(dataloader):
        print(i_batch)  # 打印batch编号
        print(batch_data[0].shape)
        print(batch_data[1].shape)
        break

    print(batch_data[1].max(), batch_data[1].min())

    for i in range(4):
        pic = (batch_data[1][i].squeeze().numpy()*255).astype(np.uint8)
        import cv2
        cv2.imwrite('TASK2_HMP_%d.jpg' % i, pic)
