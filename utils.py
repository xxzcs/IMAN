from segmentation_models_pytorch.utils import functional as F
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.base.modules import Activation
import torch.distributed as dist
import segmentation_models_pytorch as smp
import cv2
from collections import defaultdict, deque
import time
import os
import io
import datetime
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    return metric_list


"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""


class DiceScore(base.Metric):

    def __init__(self, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.dice_loss = smp.utils.losses.DiceLoss()
        self.activation = Activation(activation)

    def forward(self, y_pr, y_gt):
        # 1-dice_loss
        y_pr = self.activation(y_pr)
        dice_score = 1-self.dice_loss(y_pr[:, 0, ...], y_gt[:, 0, ...])
        return dice_score


class DiceCupScore(base.Metric):

    def __init__(self, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.dice_loss = smp.utils.losses.DiceLoss()
        self.activation = Activation(activation)

    def forward(self, y_pr, y_gt):
        # 1-dice_loss
        y_pr = self.activation(y_pr)
        dice_cup = 1-self.dice_loss(y_pr[:, 0, ...], y_gt[:, 0, ...])
        return dice_cup


class DiceDiscScore(base.Metric):

    def __init__(self, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.dice_loss = smp.utils.losses.DiceLoss()
        self.activation = Activation(activation)

    def forward(self, y_pr, y_gt):
        # 1-dice_loss
        y_pr = self.activation(y_pr)
        dice_disc = 1-self.dice_loss(y_pr[:, 1, ...], y_gt[:, 1, ...])
        return dice_disc


class MaeScore(base.Metric):

    def __init__(self, activation=None, ** kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)

        # thresh
        y_pr[y_pr >= 0.5] = 1
        y_pr[y_pr < 0.5] = 0
        try:
            pr = torch.max(y_pr, dim=-1).values.count_nonzero(dim=-1)
            pr_ratio = pr[:, 0, ...]/pr[:, 1, ...]
            gt = torch.max(y_gt, dim=-1).values.count_nonzero(dim=-1)
            gt_ratio = gt[:, 0, ...]/gt[:, 1, ...]

            nan_count = torch.isnan(pr_ratio).sum()
            nan_ratio = nan_count/pr_ratio.shape[0]
            nan_ind = ~pr_ratio.isnan()
            pr_ratio = pr_ratio[nan_ind]
            gt_ratio = gt_ratio[nan_ind]
            res = 1/((pr_ratio - gt_ratio).abs().mean()+0.1)*(1-nan_ratio)
        except:
            res = torch.tensor(np.nan)
        return torch.tensor(0).to('cuda') if torch.isnan(res) else res


class Task3Score(base.Metric):

    def __init__(self, activation=None, ** kwargs):
        super().__init__(**kwargs)
        self.dice_loss = smp.utils.losses.DiceLoss()
        self.activation = Activation(activation)

    def forward(self, y_pr, y_gt):
        # 1-dice_loss
        y_pr = self.activation(y_pr)
        dice_cup = 1-self.dice_loss(y_pr[:, 0, ...], y_gt[:, 0, ...])
        dice_disc = 1-self.dice_loss(y_pr[:, 0, ...], y_gt[:, 0, ...])

        y_pr = (y_pr)
        # thresh
        y_pr[y_pr >= 0.5] = 1
        y_pr[y_pr < 0.5] = 0
        try:
            pr = torch.max(y_pr, dim=-1).values.count_nonzero(dim=-1)
            pr_ratio = pr[:, 0, ...]/pr[:, 1, ...]
            gt = torch.max(y_gt, dim=-1).values.count_nonzero(dim=-1)
            gt_ratio = gt[:, 0, ...]/gt[:, 1, ...]

            nan_count = torch.isnan(pr_ratio).sum()
            nan_ratio = nan_count/pr_ratio.shape[0]
            nan_ind = ~pr_ratio.isnan()
            pr_ratio = pr_ratio[nan_ind]
            gt_ratio = gt_ratio[nan_ind]
            mae_score = 1/((pr_ratio - gt_ratio).abs().mean()+0.1)*(1-nan_ratio)
            mae_score = 0 if torch.isnan(mae_score) else mae_score
        except:
            mae_score = 0
        return 0.25*dice_cup*10+0.35*dice_disc*10+0.4*mae_score


def draw_keypoint(images, points_gt, points_pred, radius=5):
    b, _, h, w = images.shape
    device = images.device
    # color_gt = colors[0]
    # color_pred = colors[1]
    hw = torch.Tensor([h, w]).to(device)
    points_gt_offsets = points_gt * hw
    points_pred_offsets = points_pred * hw
    images = []
    # print(points_pred, points_gt)
    # print(points_pred_offsets, points_gt_offsets)
    for ind in range(b):
        gt_x, gt_y = points_gt_offsets[ind].int().tolist()
        # images[ind, :, max(0, gt_x-radius//2):min(gt_x+radius//2, h),
        #        max(0, gt_y-radius//2):min(gt_y+radius//2, w)] = color_gt

        pred_x, pred_y = points_pred_offsets[ind].int().tolist()
        if np.nan in points_pred_offsets[ind].int().tolist():
            continue
        # images[ind, :, max(0, pred_x-radius//4):min(pred_x+radius//4, h),
        #        max(0, pred_y-radius//4):min(pred_y+radius//4, w)] = color_pred
        img = np.ones((h, w, 3))*255
        cv2.line(img, (int(gt_x) - radius, int(gt_y)), (int(gt_x) + radius, int(gt_y)), (0, 255, 0), radius)
        cv2.line(img, (int(gt_x), int(gt_y) - radius), (int(gt_x), int(gt_y) + radius), (0, 255, 0), radius)
        cv2.line(img, (int(pred_x) - radius, int(pred_y)), (int(pred_x) + radius, int(pred_y)), (255, 0, 0), radius)
        cv2.line(img, (int(pred_x), int(pred_y) - radius), (int(pred_x), int(pred_y) + radius), (255, 0, 0), radius)
        # images[ind] = torch.Tensor(img).float()
        images.append(img)
    images = torch.Tensor(images).float().permute((0, 3, 1, 2))
    return images/255


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print("* INIT DIST, get world size", int(os.environ['WORLD_SIZE']))
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.device = torch.device('cuda', args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def FillHole(mask, pix_val=255):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    # assert len_contour < 10, "Got too much contour"
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (pix_val, pix_val, pix_val), -1)
        contour_list.append(img_contour)
    out = sum(contour_list)
    if type(out) == int:
        out = np.zeros_like(mask, np.uint8)
        print("This get no mask segmentation")
    return out


def cal_cohen_kappa(pred, targets):
    labels = pred.argmax(1)
    if len(targets.shape) > 1:
        targets = targets.argmax(1)
    p0 = (labels == targets).sum()/labels.shape[0]
    batch = labels.shape[0]

    pe = (labels == 0).sum() * (targets == 0).sum()/(batch**2)+(labels == 1).sum() * \
        (targets == 1).sum()/(batch**2)+(labels == 2).sum() * (targets == 2).sum()/(batch**2)

    return (p0-pe)/(1-pe)


def KeepMaxArea(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    max_area = 0
    assert len_contour > 100, "Got too much contour"
    if len_contour != 1:
        print(len_contour)
    for i in range(len_contour):
        area = cv2.contourArea(contours[i])
        if area > max_area:
            max_area = area
            max_ind = i

    drawing = np.zeros_like(mask, np.uint8)  # create a black image
    if len_contour > 0:
        img_contour = cv2.drawContours(drawing, contours, max_ind, (255, 255, 255), -1)
    else:
        img_contour = drawing
    return img_contour


def gaussian_kernel(size_w, size_h, center_x, center_y, sigma):
    grid_y, grid_x = np.mgrid[0: size_h, 0: size_w]
    D2 = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2

    return np.exp(-D2 / 2.0 / sigma / sigma)


def GenHeatMap(point, out_shape, sigma=5):
    '''
        :sigma control heatpoint radius
    '''
    # Generate heatmaps
    # heatmap = np.zeros(out_shape, np.float32)
    # # Anatomical landmarks
    # pts = np.array(point, dtype=np.float32).reshape(-1, 2)
    # tpts = np.array(t_keypoints, dtype=np.float32).reshape(-1, 2) / self.stride

    kernel = gaussian_kernel(size_h=out_shape[0], size_w=out_shape[1],
                             center_x=int(point[0]*out_shape[1]), center_y=int(point[1]*out_shape[0]),
                             sigma=sigma)
    kernel[kernel > 1] = 1
    kernel[kernel < 0.01] = 0
    # heatmap[:, :, i + 1] = kernel

    # Generate the heatmap of background
    # heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)
    return kernel


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def tensor_unsqueeze(image):
    close_tensor = torch.tensor(image).unsqueeze(0)
    numpy_array = np.array(close_tensor)
    numpy_array = numpy_array.transpose(1, 2, 0)
    return numpy_array


def close(image,kernel_size):
    '''
    outer
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    close = cv2.dilate(image, kernel)
    # close3 = tensor_unsqueeze(close2)
    # print(close.shape)
    kernal = gauss(15, 15)
    close = imgAverageFilter(close.squeeze(), kernal)
    close = normalization(close)
    close[close > 0] = 255
    return close


def erode(image,kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # close1 = cv2.dilate(image, kernel, iterations=3)
    close = cv2.erode(image, kernel)
    kernal = gauss(15, 15)
    close = imgAverageFilter(close, kernal)
    close = normalization(close)
    close[close > 0] = 255
    return close


# def dilate(image):
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
#     # close1 = cv2.dilate(image, kernel, iterations=3)
#     close1 = cv2.dilate(image, kernel)
#     close2 = tensor_unsqueeze(close1)
#     # print(close.shape)
#     return close2


def gauss(kernel_size, sigma):
    '''
    gen gauss kernel
    '''
    kernel = np.zeros([kernel_size, kernel_size])
    center = kernel_size // 2

    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = 2 * (sigma ** 2)
    sum_val = 0
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]
            # /(np.pi * s)
    sum_val = 1 / sum_val
    return kernel * sum_val


def imgConvolve(image, kernel):
    '''
    卷积
    :param image: 图片矩阵
    :param kernel: 滤波窗口
    :return:卷积后的矩阵
    '''
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    kernel_h = int(kernel.shape[0])
    kernel_w = int(kernel.shape[1])
    # padding
    padding_h = int((kernel_h - 1) / 2)
    padding_w = int((kernel_w - 1) / 2)

    convolve_h = int(img_h + 2 * padding_h)
    convolve_W = int(img_w + 2 * padding_w)

    # 分配空间
    img_padding = np.zeros((convolve_h, convolve_W))
    # 中心填充图片
    img_padding[padding_h:padding_h + img_h, padding_w:padding_w + img_w] = image[:, :]
    # 卷积结果
    image_convolve = np.zeros(image.shape)
    # 卷积
    for i in range(padding_h, padding_h + img_h):
        for j in range(padding_w, padding_w + img_w):
            image_convolve[i - padding_h][j - padding_w] = int(
                np.sum(img_padding[i - padding_h:i + padding_h+1, j - padding_w:j + padding_w+1]*kernel))
            # print(image_convolve[i - padding_h][j - padding_w])

    return image_convolve


def imgAverageFilter(image, kernel):
    '''
    均值滤波
    :param image: 图片矩阵
    :param kernel: 滤波窗口
    :return:均值滤波后的矩阵
    '''
    return imgConvolve(image, kernel) * (1.0 / kernel.size)
