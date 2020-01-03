"""
 # Copyright 2019 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it. If you have received this file from a source other than Adobe,
 # then your use, modification, or distribution of it requires the prior
 # written permission of Adobe. 
 
"""

import torch
from torch import nn
import numpy as np
from .layers import bn, VarianceLayer, GrayscaleLayer
from .downsampler import * 
from torch.nn import functional


class StdLoss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))


class ExclusionLoss(nn.Module):

    def __init__(self, level=3):
        """
        Loss on the gradient. based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            # alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            # alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))
            alphay = 1
            alphax = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            # gradx_loss.append(torch.mean(((gradx1_s ** 2) * (gradx2_s ** 2))) ** 0.25)
            # grady_loss.append(torch.mean(((grady1_s ** 2) * (grady2_s ** 2))) ** 0.25)
            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        ch = grad1_s.size(1)
        for i in range(ch):
            for j in range(ch):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :] + 1e-6
        grady = img[:, :, :, 1:] - img[:, :, :, :-1] + 1e-6
        return gradx, grady


class ExtendedL1Loss(nn.Module):
    """
    also pays attention to the mask, to be relative to its size
    """
    def __init__(self):
        super(ExtendedL1Loss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, a, b, mask):
        normalizer = self.l1(mask, torch.zeros(mask.shape).cuda())
        # if normalizer < 0.1:
        #     normalizer = 0.1
        c = self.l1(mask * a, mask * b) / normalizer
        return c


class NonBlurryLoss(nn.Module):
    def __init__(self):
        """
        Loss on the distance to 0.5
        """
        super(NonBlurryLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x):
        return 1 - self.mse(x, torch.ones_like(x) * 0.5)


class GrayscaleLoss(nn.Module):
    def __init__(self):
        super(GrayscaleLoss, self).__init__()
        self.gray_scale = GrayscaleLayer()
        self.mse = nn.MSELoss().cuda()

    def forward(self, x, y):
        x_g = self.gray_scale(x)
        y_g = self.gray_scale(y)
        return self.mse(x_g, y_g)


class GrayLoss(nn.Module):
    def __init__(self):
        super(GrayLoss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, x):
        y = torch.ones_like(x) / 2.
        return 1 / self.l1(x, y)


class GradientLoss(nn.Module):
    """
    L1 loss on the gradient of the picture
    """
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)

class FreqGradLoss(nn.Module):
    """
    L1 loss on the Frequency consistency
    """
    def __init__(self):
        super(FreqGradLoss, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
        return torch.mean(gradient_a_x)

class MaskGradLoss(nn.Module):
    """
    L1 loss on the Frequency consistency
    """
    def __init__(self):
        super(MaskGradLoss, self).__init__()

    def forward(self, a):
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        return torch.mean(gradient_a_y)


class FreqVarLoss(nn.Module):
    """
    Std loss on the the Freq-time Map
    """

    def __init__(self):
        super(FreqVarLoss, self).__init__()

    def forward(self, x):

        std = torch.var(x, dim=-1)
        return torch.mean(std)

class MaskVarLoss(nn.Module):
    """
    Std loss on the the Freq-time Map
    """

    def __init__(self):
        super(MaskVarLoss, self).__init__()

    def forward(self, x):

        std = torch.var(x, dim=-2)
        return torch.mean(std)


class MaskTCLoss(nn.Module):
    """
     Mask loss on the temporal consistency
    """
    def __init__(self):
        super(MaskTCLoss, self).__init__()

    def forward(self, a):
        a = a.permute(0, 1, 3, 2) # 512x192 --- 192x512
        a0 = torch.abs(a)
        a1 = torch.abs(a-1)
        a0 = torch.mean(a0, dim=-1)
        a1 = torch.mean(a1, dim =-1)
        x = torch.stack([a0, a1], dim=-1)
        x = torch.min(x, dim=-1).values
        return torch.mean(x)

class ProjLoss(nn.Module):
    """
     Mask loss on the temporal consistency
    """
    def __init__(self):
        super(ProjLoss, self).__init__()

    def forward(self, s1, s2):
        a = torch.max(s1, dim=-1).values
        b = torch.max(s2, dim=-1).values

        return torch.mean(torch.exp(-torch.abs(a-b)))

class DissLoss(nn.Module):
    """
     Mask loss on the temporal consistency
    """
    def __init__(self):
        super(DissLoss, self).__init__()

    def forward(self, s1, s2):

        return torch.mean(torch.exp(-torch.abs(s1-s2)))

class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, m):

        return torch.mean(1.0/(1e-6 + torch.mean(torch.abs(m-0.5), (1, 2, 3))))

class NonZeroLoss(nn.Module):
    def __init__(self):
        super(NonZeroLoss, self).__init__()

    def forward(self, x, y, weight):
        a = torch.mean(x, (1,2,3))
        b = torch.mean(y, (1,2,3))
        return torch.mean(torch.exp(-torch.min(a, b)))

class NonZeroMaskLoss(nn.Module):
    def __init__(self):
        super(NonZeroMaskLoss, self).__init__()

    def forward(self, x, y):
        min_value = torch.clamp(x+y, max=1.0) + 1e-6
        l = torch.mean(1.0/min_value)
        return l

class NonZeroMaskLoss_v2(nn.Module):
    def __init__(self):
        super(NonZeroMaskLoss_v2, self).__init__()

    def forward(self, x, y, weight):
        min_value = torch.clamp(x+y, max=1.0) + 1e-6

        l = torch.mean(torch.mul(weight, 1.0/min_value))
        return l

class MaskDeactivationLoss(nn.Module):
    def __init__(self):
        super(MaskDeactivationLoss, self).__init__()

    def forward(self, x, mask_dea):
        diff = x - (1. - mask_dea)
        v = torch.mul(diff, mask_dea)
        l = torch.mean(v)

        return l







