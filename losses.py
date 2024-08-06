"""
Implements the knowledge distillation loss
"""
import numpy as np
import math
import time
import cv2
import torch
from segmentation_models_pytorch import losses as smp_losses
# from segmentation_models_pytorch.utils.losses import DiceLoss
from sklearn.metrics.pairwise import euclidean_distances
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss


class DiceLoss(nn.Module):
    def __init__(self, n_classes=1, alpha=0.5, **kwargs):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha

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

    def forward(self, inputs, target, softmax=False):
        '''
        weight: cof for different classes
        '''
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        weight = [self.alpha, 1-self.alpha]

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes

    @property
    def __name__(self):
        return 'dice_loss'


class DiceEdgeLoss(nn.Module):
    def __init__(self, n_classes=1, alpha=0.5, **kwargs):
        super(DiceEdgeLoss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha

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

    def _edge_loss(self, score, target):
        target = target.float()

        score_smooth = nn.MaxPool2d(3, stride=1, padding=1)(score)
        edge_pred = torch.abs(score-score_smooth)
        target_smooth = nn.MaxPool2d(3, stride=1, padding=1)(score)
        edge_gt = torch.abs(target-target_smooth)

        loss = nn.MSELoss()(edge_pred, edge_gt)
        return loss

    def forward(self, inputs, target, softmax=False):
        '''
        weight: cof for different classes
        '''
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        weight = [self.alpha, 1-self.alpha]

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        dice_loss = self._dice_loss(inputs[:, 0], target[:, 0])
        edge_loss = self._edge_loss(inputs[:, 0], target[:, 0])

        loss = dice_loss * weight[0] + edge_loss * weight[1]
        return loss

    @property
    def __name__(self):
        return 'dice_edge_loss'


from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve
"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
"""
class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform
    https://github.com/PatRyg99/HausdorffLoss/blob/master/hausdorff_loss.py
    """

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.detach().cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.detach().cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss

    @property
    def __name__(self):
        return 'hd_dt_loss'

class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion
    https://github.com/PatRyg99/HausdorffLoss/blob/master/hausdorff_loss.py
    """

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        if debug:
            eroted, erosions = self.perform_erosion(
                pred.cpu().numpy(), target.cpu().numpy(), debug
            )
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(
                self.perform_erosion(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), debug)
            ).float()

            loss = eroted.mean()

            return loss
    @property
    def __name__(self):
        return 'hd_er_loss'



if __name__ == "__main__":
    loss_func = AdaptiveWingLoss()
    y = torch.ones(2, 2)
    y_hat = torch.zeros(2, 2)
    y_hat.requires_grad_(True)
    loss = loss_func(y_hat, y)
    loss.backward()
    print(loss)
