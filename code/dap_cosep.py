"""
 # Copyright 2019 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it. If you have received this file from a source other than Adobe,
 # then your use, modification, or distribution of it requires the prior
 # written permission of Adobe. 
 
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from net import skip, skip_mask_vec
from net.losses import ExclusionLoss, plot_image_grid, \
    StdLoss, NonZeroLoss, FreqVarLoss, MaskVarLoss, FreqGradLoss, \
    MaskGradLoss, MaskTCLoss, ProjLoss, BinaryLoss, DissLoss, NonZeroMaskLoss_v2
from net.noise import get_noise, get_video_noise
from utils.image_io import *
from utils.audio_io import *
from utils.audio_proc import *
from skimage.measure import compare_mse
import numpy as np
import torch
import cv2
import torch.nn as nn
from collections import namedtuple
from torch.autograd import Variable, Function
from torch.optim.lr_scheduler import StepLR
import argparse

parser = argparse.ArgumentParser(description='BSS')

# Data specifications
parser.add_argument('--input1', type=str, default='data/cosep/audiojungle/01.mp3',
                    help='input sound 1')
parser.add_argument('--input2', type=str, default='data/cosep/audiojungle/02.mp3',
                    help='input sound 2')
parser.add_argument('--input3', type=str, default='data/cosep/audiojungle/03.mp3',
                    help='input sound 3')
parser.add_argument('--output', type=str, default='output/cosep',
                    help='results')
args = parser.parse_args()

class Separation(object):
    def __init__(self, image_name, output_path, amp_mix1, amp_mix2, amp_mix3, phase_mix1, phase_mix2,
                 phase_mix3, audRate, numSeg, plot_during_training=True, show_every=500,
                 ckpt="ckpt", num_iter=5000,
                 original_mask1=None, original_mask2=None,original_mask3=None, original_mask4=None,
                 original_sound1=None, original_sound2=None, original_sound3=None, original_sound4=None):
        self.output_path = output_path
        self.image1 = amp_mix1
        self.image2 = amp_mix2
        self.image3 = amp_mix3
        self.phase1 = phase_mix1
        self.phase2 = phase_mix2
        self.phase3 = phase_mix3
        self.audRate = audRate
        self.plot_during_training = plot_during_training
        self.psnrs = []
        self.show_every = show_every
        self.image_name = image_name
        self.num_iter = num_iter
        self.loss_function = None
        self.parameters = None
        self.ckpt = ckpt
        self.learning_rate = 0.0003
        self.input_depth = 1
        self.N = numSeg
        self.mask1_net_inputs = None
        self.mask2_net_inputs = None
        self.mask3_net_inputs = None
        self.mask4_net_inputs = None

        self.sound1_net_inputs = None
        self.sound2_net_inputs = None
        self.sound3_net_inputs = None
        self.sound4_net_inputs = None

        self.original_mask1 = original_mask1
        self.original_mask2 = original_mask2
        self.original_mask3 = original_mask3
        self.original_mask4 = original_mask4
        self.original_sound1 = original_sound1
        self.original_sound2 = original_sound2
        self.original_sound3 = original_sound3
        self.original_sound4 = original_sound4
        self.mask1_net = None
        self.mask2_net = None
        self.mask3_net = None
        self.mask4_net = None
        self.sound1_net = None
        self.sound2_net = None
        self.sound3_net = None
        self.sound4_net = None
        self.total_loss = None
        self.mask1_out = None
        self.mask2_out = None
        self.mask3_out = None
        self.mask4_out = None
        self.sound1_out = None
        self.sound2_out = None
        self.sound3_out = None
        self.sound4_out = None
        self.current_result = None
        self.best_result = None
        self._init_all()

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        self.images1 = self.image1
        self.images2 = self.image2
        self.images3 = self.image3
        self.images_torch1 = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.image1]
        self.images_torch2 = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.image2]
        self.images_torch3 = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.image3]

    def mask_input(self, x, var, N, data_type):
        B, C, W, H = x[0].size()
        mask_input = [None for n in range(N)]
        for n in range(N):
            mask_input[n] = var * torch.rand_like(x[n], requires_grad=False).type(
                data_type).detach()
            mask_input[n] = (mask_input[n]).unsqueeze(1)
        mask_input = torch.cat(mask_input, dim=1).view(-1, C, W, H)  # + uniform_noise
        return mask_input

    def _init_inputs(self):

        input_type = 'noise'
        data_type = torch.cuda.FloatTensor

        self.B, self.C, self.W, self.H = self.images_torch1[0].size()
        N = len(self.images_torch1)

        self.mask1_input = self.mask_input(self.images_torch1, 1./30, N, data_type)
        self.mask2_input = self.mask_input(self.images_torch1, 1./30, N, data_type)
        self.mask3_input = self.mask_input(self.images_torch1, 1./30, N, data_type)
        self.mask4_input = self.mask_input(self.images_torch1, 1./30, N, data_type)

        self.uniform_noise1 = get_video_noise(self.input_depth, input_type, self.N,
                                         (self.images_torch1[0].shape[2],
                                          self.images_torch1[0].shape[3]), var=1. / 2000).type(
                                        torch.cuda.FloatTensor).detach()
        self.uniform_noise2 = get_video_noise(self.input_depth, input_type, self.N,
                                              (self.images_torch1[0].shape[2],
                                               self.images_torch1[0].shape[3]), var=1. / 2000).type(
                                                torch.cuda.FloatTensor).detach()

        self.uniform_noise3 = get_video_noise(self.input_depth, input_type, self.N,
                                              (self.images_torch1[0].shape[2],
                                               self.images_torch1[0].shape[3]), var=1. / 2000).type(
                                            torch.cuda.FloatTensor).detach()
        self.uniform_noise4 = get_video_noise(self.input_depth, input_type, self.N,
                                              (self.images_torch1[0].shape[2],
                                               self.images_torch1[0].shape[3]), var=1. / 2000).type(
                                            torch.cuda.FloatTensor).detach()

        g1_noise = torch.rand_like(self.images_torch1[0], requires_grad=False).type(data_type).detach()
        g1_input = [None for n in range(N)]
        for n in range(N):
            g1_input[n] = g1_noise
            g1_input[n] = (g1_input[n]).unsqueeze(1)
        self.g_noise1 = torch.cat(g1_input, dim=1).view(-1, self.C, self.W, self.H)

        g2_noise = torch.rand_like(self.images_torch1[0], requires_grad=False).type(data_type).detach()
        g2_input = [None for n in range(N)]
        for n in range(N):
            g2_input[n] = g2_noise
            g2_input[n] = (g2_input[n]).unsqueeze(1)
        self.g_noise2 = torch.cat(g2_input, dim=1).view(-1, self.C, self.W, self.H)

        g3_noise = torch.rand_like(self.images_torch1[0], requires_grad=False).type(data_type).detach()
        g3_input = [None for n in range(N)]
        for n in range(N):
            g3_input[n] = g3_noise
            g3_input[n] = (g3_input[n]).unsqueeze(1)
        self.g_noise3 = torch.cat(g3_input, dim=1).view(-1, self.C, self.W, self.H)

        g4_noise = torch.rand_like(self.images_torch1[0], requires_grad=False).type(data_type).detach()
        g4_input = [None for n in range(N)]
        for n in range(N):
            g4_input[n] = g4_noise
            g4_input[n] = (g4_input[n]).unsqueeze(1)
        self.g_noise4 = torch.cat(g4_input, dim=1).view(-1, self.C, self.W, self.H)

        x = [None for n in range(N)]
        for n in range(N):
            x[n] = torch.rand_like(self.images_torch1[n]).type(data_type).detach()
            x[n] = (x[n]).unsqueeze(1)
        self.g_dynamic_n1 = torch.cat(x, dim=1).view(-1, self.C, self.W, self.H)

        x = [None for n in range(N)]
        for n in range(N):
            x[n] = torch.rand_like(self.images_torch1[n]).type(data_type).detach()
            x[n] = (x[n]).unsqueeze(1)
        self.g_dynamic_n2 = torch.cat(x, dim=1).view(-1, self.C, self.W, self.H)

        x = [None for n in range(N)]
        for n in range(N):
            x[n] = torch.rand_like(self.images_torch1[n]).type(data_type).detach()
            x[n] = (x[n]).unsqueeze(1)
        self.g_dynamic_n3 = torch.cat(x, dim=1).view(-1, self.C, self.W, self.H)

        x = [None for n in range(N)]
        for n in range(N):
            x[n] = torch.rand_like(self.images_torch1[n]).type(data_type).detach()
            x[n] = (x[n]).unsqueeze(1)
        self.g_dynamic_n4 = torch.cat(x, dim=1).view(-1, self.C, self.W, self.H)

        self.images_torch1 = torch.cat(self.images_torch1, dim=1).view(-1, self.C, self.W, self.H)
        self.images_torch2 = torch.cat(self.images_torch2, dim=1).view(-1, self.C, self.W, self.H)
        self.images_torch3 = torch.cat(self.images_torch3, dim=1).view(-1, self.C, self.W, self.H)

    def _init_parameters(self):
        self.parameters = [p for p in self.mask1_net.parameters()] + \
                          [p for p in self.mask2_net.parameters()] + \
                          [p for p in self.mask3_net.parameters()] + \
                          [p for p in self.mask4_net.parameters()] + \
                          [p for p in self.sound1_net.parameters()] + \
                          [p for p in self.sound2_net.parameters()] + \
                          [p for p in self.sound3_net.parameters()] + \
                          [p for p in self.sound4_net.parameters()]

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'
        mask1_net = skip_mask_vec(
            self.input_depth, self.images1[0].shape[0],
            num_channels_down=[16, 32, 64],
            num_channels_up=[16, 32, 64],
            num_channels_skip=[0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask1_net = mask1_net.type(data_type)

        mask2_net = skip_mask_vec(
            self.input_depth, self.images1[0].shape[0],
            num_channels_down=[16, 32, 64],
            num_channels_up=[16, 32, 64],
            num_channels_skip=[0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask2_net = mask2_net.type(data_type)

        mask3_net = skip_mask_vec(
            self.input_depth, self.images1[0].shape[0],
            num_channels_down=[16, 32, 64],
            num_channels_up=[16, 32, 64],
            num_channels_skip=[0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask3_net = mask3_net.type(data_type)

        mask4_net = skip_mask_vec(
            self.input_depth, self.images1[0].shape[0],
            num_channels_down=[16, 32, 64],
            num_channels_up=[16, 32, 64],
            num_channels_skip=[0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask4_net = mask4_net.type(data_type)

        sound1_net = skip(
            self.input_depth, self.images1[0].shape[0],
            num_channels_down=[16, 32, 64],
            num_channels_up=[16, 32, 64],
            num_channels_skip=[0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=False, need_relu=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.sound1_net = sound1_net.type(data_type)

        sound2_net = skip(
            self.input_depth, self.images1[0].shape[0],
            num_channels_down=[16, 32, 64],
            num_channels_up=[16, 32, 64],
            num_channels_skip=[0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=False, need_relu=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.sound2_net = sound2_net.type(data_type)

        sound3_net = skip(
            self.input_depth, self.images1[0].shape[0],
            num_channels_down=[16, 32, 64],
            num_channels_up=[16, 32, 64],
            num_channels_skip=[0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=False, need_relu=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.sound3_net = sound3_net.type(data_type)

        sound4_net = skip(
            self.input_depth, self.images1[0].shape[0],
            num_channels_down=[16, 32, 64],
            num_channels_up=[16, 32, 64],
            num_channels_skip=[0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=False, need_relu=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.sound4_net = sound4_net.type(data_type)

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.l1_loss = nn.L1Loss().type(data_type)
        self.exclusion_loss = ExclusionLoss().type(data_type)
        self.freq_var_loss = FreqVarLoss().type(data_type)
        self.mask_var_loss = MaskVarLoss().type(data_type)
        self.freq_grad_loss = FreqGradLoss().type(data_type)
        self.mask_grad_loss = MaskGradLoss().type(data_type)
        self.mask_tc_loss = MaskTCLoss().type(data_type)
        self.proj_loss = ProjLoss().type(data_type)
        self.binary_loss = BinaryLoss().type(data_type)
        self.nonzero_loss = NonZeroLoss().type(data_type)
        self.diss_loss = DissLoss().type(data_type)
        self.nonzero_mask_loss = NonZeroMaskLoss_v2().type(data_type)

    def adjust_learning_rate(self, optimizer, iter_num):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.learning_rate * (0.1 ** (iter_num // 2000))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)

        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result(j)
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()

    def _optimization_closure(self, step):

        weight1 = torch.log1p(self.images_torch1)
        weight1 = torch.clamp(weight1, 0, 1)

        weight2 = torch.log1p(self.images_torch2)
        weight2 = torch.clamp(weight2, 0, 1)

        weight3 = torch.log1p(self.images_torch3)
        weight3 = torch.clamp(weight3, 0, 1)

        if step < 2000:
            self.sound1_input = self.uniform_noise1 + self.g_noise1
            self.sound2_input = self.uniform_noise2 + self.g_noise2
            self.sound3_input = self.uniform_noise3 + self.g_noise3
            self.sound4_input = self.uniform_noise4 + self.g_noise4
        elif step < 4000:
            a = step/4000.
            self.sound1_input = self.uniform_noise1 + a * self.g_dynamic_n1 + (1 - a) * self.g_noise1
            self.sound2_input = self.uniform_noise2 + a * self.g_dynamic_n2 + (1 - a) * self.g_noise2
            self.sound3_input = self.uniform_noise3 + a * self.g_dynamic_n3 + (1 - a) * self.g_noise3
            self.sound4_input = self.uniform_noise4 + a * self.g_dynamic_n4 + (1 - a) * self.g_noise4
        else:
            self.sound1_input = self.uniform_noise1 + self.g_dynamic_n1
            self.sound2_input = self.uniform_noise2 + self.g_dynamic_n2
            self.sound3_input = self.uniform_noise3 + self.g_dynamic_n3
            self.sound4_input = self.uniform_noise4 + self.g_dynamic_n4

        mask1_out = (self.mask1_net(self.mask1_input))
        mask2_out = (self.mask2_net(self.mask2_input))
        mask3_out = (self.mask1_net(self.mask3_input))
        mask4_out = (self.mask2_net(self.mask4_input))

        B, C, W, H = mask1_out.size()

        mask1_out = (torch.sigmoid(torch.mean(mask1_out, dim=-2).unsqueeze(-2)))
        mask2_out = (torch.sigmoid(torch.mean(mask2_out, dim=-2).unsqueeze(-2)))
        mask3_out = (torch.sigmoid(torch.mean(mask3_out, dim=-2).unsqueeze(-2)))
        mask4_out = (torch.sigmoid(torch.mean(mask4_out, dim=-2).unsqueeze(-2)))

        self.mask1_out = mask1_out.repeat(1, 1, W, 1)
        self.mask2_out = mask2_out.repeat(1, 1, W, 1)
        self.mask3_out = mask3_out.repeat(1, 1, W, 1)
        self.mask4_out = mask4_out.repeat(1, 1, W, 1)

        self.sound1_out = self.sound1_net(self.sound1_input)
        self.sound2_out = self.sound2_net(self.sound2_input)
        self.sound3_out = self.sound1_net(self.sound3_input)
        self.sound4_out = self.sound2_net(self.sound4_input)

        self.sound1 = self.mask1_out * self.sound1_out
        self.sound2 = self.mask2_out * self.sound2_out
        self.sound3 = self.mask3_out * self.sound3_out
        self.sound4 = self.mask4_out * self.sound4_out

        self.l11 = self.l1_loss(self.sound1 + self.sound2, self.images_torch1)
        self.l12 = self.l1_loss(self.sound1 + self.sound3, self.images_torch2)
        self.l13 = self.l1_loss(self.sound1 + self.sound4, self.images_torch3)

        self.l21 = self.freq_grad_loss(self.sound1_out)
        self.l22 = self.freq_grad_loss(self.sound2_out)
        self.l23 = self.freq_grad_loss(self.sound3_out)
        self.l24 = self.freq_grad_loss(self.sound4_out)

        self.l31 = self.exclusion_loss(self.sound1, self.sound2)
        self.l32 = self.exclusion_loss(self.sound1, self.sound3)
        self.l33 = self.exclusion_loss(self.sound1, self.sound4)

        self.l41 = self.binary_loss(self.mask1_out) * 0.01
        self.l42 = self.binary_loss(self.mask2_out) * 0.01
        self.l43 = self.binary_loss(self.mask3_out) * 0.01
        self.l44 = self.binary_loss(self.mask4_out) * 0.01

        self.l51 = self.nonzero_mask_loss(self.mask1_out, self.mask2_out, weight1)
        self.l52 = self.nonzero_mask_loss(self.mask1_out, self.mask3_out, weight2)
        self.l53 = self.nonzero_mask_loss(self.mask1_out, self.mask4_out, weight3)

        if step <= 4500:
            self.total_loss = self.l11 + self.l12 + self.l13 + self.l21 + self.l22 + \
                self.l23 + self.l24 + self.l31 + self.l32 + self.l33 + self.l51 + self.l52 + self.l53
        else:
            self.total_loss = self.l11 + self.l12 + self.l13 + self.l21 + self.l22 + \
                              self.l23 + self.l24 + self.l31 + self.l32 + self.l33 + self.l51 \
                              + self.l52 + self.l53 + self.l41 + self.l42 + self.l43 + self.l44

        self.total_loss.backward()

    def _obtain_current_result(self, step):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        if step == self.num_iter - 1 or step % 8 == 0:

            sound1_np = torch_to_np(torch.cat(torch.chunk(self.sound1_out, self.N, dim=0), dim=-1))
            sound2_np = torch_to_np(torch.cat(torch.chunk(self.sound2_out, self.N, dim=0), dim=-1))
            sound3_np = torch_to_np(torch.cat(torch.chunk(self.sound3_out, self.N, dim=0), dim=-1))
            sound4_np = torch_to_np(torch.cat(torch.chunk(self.sound4_out, self.N, dim=0), dim=-1))

            sound1_out_np = torch_to_np(torch.cat(torch.chunk(self.sound1, self.N, dim=0), dim=-1))
            sound2_out_np = torch_to_np(torch.cat(torch.chunk(self.sound2, self.N, dim=0), dim=-1))
            sound3_out_np = torch_to_np(torch.cat(torch.chunk(self.sound3, self.N, dim=0), dim=-1))
            sound4_out_np = torch_to_np(torch.cat(torch.chunk(self.sound4, self.N, dim=0), dim=-1))

            mask1_out_np = (torch_to_np(torch.cat(torch.chunk(self.mask1_out, self.N, dim=0), dim=-1)) * 255.).astype(
                np.uint8)
            mask2_out_np = (torch_to_np(torch.cat(torch.chunk(self.mask2_out, self.N, dim=0), dim=-1)) * 255.).astype(
                np.uint8)
            mask3_out_np = (torch_to_np(torch.cat(torch.chunk(self.mask3_out, self.N, dim=0), dim=-1)) * 255.).astype(
                np.uint8)
            mask4_out_np = (torch_to_np(torch.cat(torch.chunk(self.mask4_out, self.N, dim=0), dim=-1)) * 255.).astype(
                np.uint8)

            im_np1 = torch_to_np(torch.cat(torch.chunk(self.images_torch1, self.N, dim=0), dim=-1))
            im_np2 = torch_to_np(torch.cat(torch.chunk(self.images_torch2, self.N, dim=0), dim=-1))
            im_np3 = torch_to_np(torch.cat(torch.chunk(self.images_torch3, self.N, dim=0), dim=-1))
            mse1 = compare_mse(im_np1, sound1_out_np + sound2_out_np)
            mse2 = compare_mse(im_np2, sound1_out_np + sound3_out_np)
            mse3 = compare_mse(im_np3, sound1_out_np + sound4_out_np)

            mse = (mse1 + mse2 + mse3)/3
            self.psnrs.append(mse)
            self.current_result = SeparationResult(mask1=mask1_out_np[0], mask2=mask2_out_np[0],
                                                   mask3=mask3_out_np[0], mask4=mask4_out_np[0],
                                                   sound1=sound1_out_np,  sound2=sound2_out_np,
                                                   sound3=sound3_out_np, sound4=sound4_out_np,
                                                   sound1_out=sound1_np, sound2_out=sound2_np,
                                                   sound3_out=sound3_np, sound4_out=sound4_np,
                                                   mse=mse)
            if self.best_result is None or self.best_result.mse > self.current_result.mse:
                self.best_result = self.current_result

    def _plot_closure(self, step):
        print('Iteration {:5d}    Loss {:5f}  MSE: {:f}'.format(step, self.total_loss.item(),
                                                                self.current_result.mse), '\r', end='')
        if step % self.show_every == self.show_every - 1:
            x_1 = magnitude2heatmap(self.current_result.sound1)
            x_2 = magnitude2heatmap(self.current_result.sound2)
            x_3 = magnitude2heatmap(self.current_result.sound3)
            x_4 = magnitude2heatmap(self.current_result.sound4)

            x = np.concatenate([x_1, x_2, x_3, x_4], axis=-3)
            save_image("s1+s2+s3+s4_{}".format(step), x, self.output_path)

            x_1 = magnitude2heatmap(self.current_result.sound1_out)
            x_2 = magnitude2heatmap(self.current_result.sound2_out)
            x_3 = magnitude2heatmap(self.current_result.sound3_out)
            x_4 = magnitude2heatmap(self.current_result.sound4_out)
            x = np.concatenate([x_1, x_2, x_3, x_4], axis=-3)
            save_image("s1_out+s2_out+s3_out+s4_out_{}".format(step), x, self.output_path)

            x_1 = (self.current_result.mask1)
            x_2 = (self.current_result.mask2)
            x_3 = (self.current_result.mask3)
            x_4 = (self.current_result.mask4)

            x = np.concatenate([x_1, x_2, x_3, x_4], axis=-2)
            save_image("mask1+mask2+mask3+mask4_{}".format(step), x, self.output_path)

            a1_wav = istft_reconstruction(self.current_result.sound1[0], self.phase1)
            librosa.output.write_wav(os.path.join(self.output_path, 'sound_sep1_{}.wav'.format(step)), a1_wav, self.audRate)

            a2_wav = istft_reconstruction(self.current_result.sound2[0], self.phase1)
            librosa.output.write_wav(os.path.join(self.output_path, 'sound_sep2_{}.wav'.format(step)), a2_wav, self.audRate)

            a3_wav = istft_reconstruction(self.current_result.sound3[0], self.phase2)
            librosa.output.write_wav(os.path.join(self.output_path, 'sound_sep3_{}.wav'.format(step)), a3_wav, self.audRate)

            a4_wav = istft_reconstruction(self.current_result.sound4[0], self.phase3)
            librosa.output.write_wav(os.path.join(self.output_path, 'sound_sep4_{}.wav'.format(step)), a4_wav, self.audRate)

    def finalize(self):
        save_graph(self.image_name + "_psnr", self.psnrs, self.output_path)
        save_image(self.image_name + "_sound1", magnitude2heatmap(self.best_result.sound1), self.output_path)
        save_image(self.image_name + "_sound2", magnitude2heatmap(self.best_result.sound2), self.output_path)
        save_image(self.image_name + "_sound3", magnitude2heatmap(self.best_result.sound3), self.output_path)
        save_image(self.image_name + "_sound4", magnitude2heatmap(self.best_result.sound4), self.output_path)

        a1_wav = istft_reconstruction(self.best_result.sound1[0], self.phase1)
        librosa.output.write_wav(os.path.join(self.output_path, 'sound_sep1.wav'), a1_wav, self.audRate)
        a2_wav = istft_reconstruction(self.best_result.sound2[0], self.phase1)
        librosa.output.write_wav(os.path.join(self.output_path, 'sound_sep2.wav'), a2_wav, self.audRate)
        a3_wav = istft_reconstruction(self.best_result.sound3[0], self.phase2)
        librosa.output.write_wav(os.path.join(self.output_path, 'sound_sep3.wav'), a3_wav, self.audRate)
        a4_wav = istft_reconstruction(self.best_result.sound4[0], self.phase3)
        librosa.output.write_wav(os.path.join(self.output_path, 'sound_sep4.wav'), a4_wav, self.audRate)

        save_image(self.image_name + "_sound1_out", magnitude2heatmap(self.best_result.sound1_out), self.output_path)
        save_image(self.image_name + "_sound2_out", magnitude2heatmap(self.best_result.sound2_out), self.output_path)
        save_image(self.image_name + "_mask1", self.best_result.mask1, self.output_path)
        save_image(self.image_name + "_mask2", self.best_result.mask2, self.output_path)
        save_image(self.image_name + "_sound3_out", magnitude2heatmap(self.best_result.sound3_out), self.output_path)
        save_image(self.image_name + "_sound4_out", magnitude2heatmap(self.best_result.sound4_out), self.output_path)
        save_image(self.image_name + "_mask3", self.best_result.mask3, self.output_path)
        save_image(self.image_name + "_mask4", self.best_result.mask4, self.output_path)

        save_image(self.image_name + "_mix1", magnitude2heatmap(np.concatenate(self.images1, axis=-1)), self.output_path)
        save_image(self.image_name + "_mix2", magnitude2heatmap(np.concatenate(self.images2, axis=-1)), self.output_path)
        save_image(self.image_name + "_mix3", magnitude2heatmap(np.concatenate(self.images3, axis=-1)), self.output_path)


SeparationResult = namedtuple("SeparationResult",
                              ['mask1', 'mask2', 'mask3', 'mask4',
                               'sound1', 'sound2', 'sound3', 'sound4',
                               'sound1_out', 'sound2_out','sound3_out', 'sound4_out', 'mse'])

if __name__ == "__main__":

    # params
    audRate = 11000
    audLen = 12
    seg_num = 24
    start_time = 0

    # load three sounds with audiojungle watermaker
    audio1 = args.input1
    audio2 = args.input2
    audio3 = args.input3
    path_out = args.output
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    amp_mix1, phase_mix1, mix_wav1 = audio_process(audio1, audRate, audLen, seg_num, start_time)
    amp_mix2, phase_mix2, mix_wav2 = audio_process(audio2, audRate, audLen, seg_num, start_time)
    amp_mix3, phase_mix3, mix_wav3 = audio_process(audio3, audRate, audLen, seg_num, start_time)

    librosa.output.write_wav('output/cosep/mixture1.wav', mix_wav1, audRate)
    librosa.output.write_wav('output/cosep/mixture2.wav', mix_wav2, audRate)
    librosa.output.write_wav('output/cosep/mixture3.wav', mix_wav3, audRate)
    # Sep
    for i in range(seg_num):
        amp_mix1[i] = np.expand_dims(amp_mix1[i], axis=0)
        amp_mix2[i] = np.expand_dims(amp_mix2[i], axis=0)
        amp_mix3[i] = np.expand_dims(amp_mix3[i], axis=0)

    s = Separation('sounds', path_out, amp_mix1, amp_mix2, amp_mix3, phase_mix1, phase_mix2, phase_mix3, audRate, seg_num)
    s.optimize()
    s.finalize()


