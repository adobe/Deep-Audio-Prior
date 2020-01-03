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
    MaskGradLoss, MaskTCLoss, ProjLoss, BinaryLoss, DissLoss, NonZeroMaskLoss
from net.noise import get_noise, get_video_noise
from utils.image_io import *
from utils.audio_io import *
from skimage.measure import compare_mse
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from torch.autograd import Variable, Function
import argparse

parser = argparse.ArgumentParser(description='DAP audio synthesis')

# Data specifications
parser.add_argument('--input', type=str, default='data/synthesis/water.wav',
                    help='input sound')
parser.add_argument('--output', type=str, default='output/sysnthesis',
                    help='results')

args = parser.parse_args()

class Separation(object):
    def __init__(self, image_name, output_path, image, phase, audRate, numSeg, plot_during_training=True, show_every=500,
                 num_iter=5000,
                 original_mask1=None, original_sound1=None):
        self.image = image
        self.output_path = output_path
        self.phase = phase
        self.audRate = audRate
        self.plot_during_training = plot_during_training
        self.psnrs = []
        self.show_every = show_every
        self.image_name = image_name
        self.num_iter = num_iter
        self.loss_function = None
        # self.ratio_net = None
        self.parameters = None
        self.learning_rate = 0.0003
        self.input_depth = 1
        self.N = numSeg
        self.mask1_net_inputs = None
        self.sound1_net_inputs = None
        self.original_mask1 = original_mask1
        self.original_sound1 = original_sound1
        self.mask1_net = None
        self.sound1_net = None
        self.total_loss = None
        self.mask1_out = None
        self.sound1_out = None
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
        self.images = self.image
        self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.image]

    def _init_inputs(self):
        input_type = 'noise'
        # input_type = 'meshgrid'
        data_type = torch.cuda.FloatTensor

        self.B, self.C, self.W, self.H = self.images_torch[0].size()
        N = len(self.images_torch)

        mask1_input = [None for n in range(N)]
        m_noise = torch.rand_like(self.images_torch[0], requires_grad=False).type(data_type).detach()
        uniform_noise = get_video_noise(self.input_depth, input_type, self.N,
                                        (self.images_torch[0].shape[2],
                                         self.images_torch[0].shape[3]), var=1. / 2000).type(
            torch.cuda.FloatTensor).detach()

        for n in range(N):
            mask1_input[n] = m_noise
            mask1_input[n] = (mask1_input[n]).unsqueeze(1)
        self.mask1_input = torch.cat(mask1_input, dim=1).view(-1, self.C, self.W, self.H) + uniform_noise

        uniform_noise1 = get_video_noise(self.input_depth, input_type, self.N,
                                         (self.images_torch[0].shape[2],
                                          self.images_torch[0].shape[3]), var=1. / 500).type(
            torch.cuda.FloatTensor).detach()

        g1_noise = torch.rand_like(self.images_torch[0], requires_grad=False).type(data_type).detach()
        sound1_input = [None for n in range(N)]
        for n in range(N):
            sound1_input[n] = g1_noise  # torch.rand_like(self.images_torch[n]).type(data_type).detach()
            sound1_input[n] = (sound1_input[n]).unsqueeze(1)
        self.g_noise1 = torch.cat(sound1_input, dim=1).view(-1, self.C, self.W, self.H)  # + uniform_noise1

        self.uniform_noise1 = uniform_noise1
        self.sound1_input = uniform_noise1

        x = [None for n in range(N)]
        for n in range(N):
            x[n] = torch.rand_like(self.images_torch[n]).type(data_type).detach()
            x[n] = (x[n]).unsqueeze(1)
        self.g_dynamic_n1 = torch.cat(x, dim=1).view(-1, self.C, self.W, self.H)

        self.images_torch = torch.cat(self.images_torch, dim=1).view(-1, self.C, self.W, self.H)

    def _init_parameters(self):
        self.parameters = [p for p in self.mask1_net.parameters()] + \
                          [p for p in self.sound1_net.parameters()]

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'
        mask1_net = skip_mask_vec(
            self.input_depth, self.images[0].shape[0],
            num_channels_down=[16, 32, 64],
            num_channels_up=[16, 32, 64],
            num_channels_skip=[0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask1_net = mask1_net.type(data_type)

        sound1_net = skip(
            self.input_depth, self.images[0].shape[0],
            num_channels_down=[16, 32, 64],
            num_channels_up=[16, 32, 64],
            num_channels_skip=[0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=False, need_relu=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.sound1_net = sound1_net.type(data_type)

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
        self.nonzero_mask_loss = NonZeroMaskLoss().type(data_type)

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
        self.sound1_input = self.uniform_noise1 + self.g_noise1

        mask1_out = (self.mask1_net(self.mask1_input))

        B, C, W, H = mask1_out.size()

        mask1_out = (torch.sigmoid(torch.mean(mask1_out, dim=-2).unsqueeze(-2)))
        self.mask1_out = mask1_out.repeat(1, 1, W, 1)
        self.sound1_out = self.sound1_net(self.sound1_input)
        self.sound1 = self.sound1_out

        self.l1 = self.l1_loss(self.sound1[::2], self.images_torch[::2])

        # freq
        self.l2 = self.freq_grad_loss(self.sound1_out[::2])

        self.l3 = self.binary_loss(self.mask1_out[::2]) * 0.01  # 0.1

        self.total_loss = self.l1 + self.l2
        self.total_loss.backward()

    def _obtain_current_result(self, step):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        if step == self.num_iter - 1 or step % 8 == 0:

            sound1_np = torch_to_np(torch.cat(torch.chunk(self.sound1_out, self.N, dim=0), dim=-1))
            sound1_out_np = torch_to_np(torch.cat(torch.chunk(self.sound1, self.N, dim=0), dim=-1))
            mask1_out_np = (torch_to_np(torch.cat(torch.chunk(self.mask1_out, self.N, dim=0), dim=-1)) * 255.).astype(
                np.uint8)
            im_np = torch_to_np(torch.cat(torch.chunk(self.images_torch[::2], self.N, dim=0), dim=-1))
            mse = compare_mse(im_np, torch_to_np(torch.cat(torch.chunk(self.sound1[::2], self.N, dim=0), dim=-1)))

            self.psnrs.append(mse)
            self.current_result = SeparationResult(mask1=mask1_out_np[0],  sound1=sound1_out_np,
                                                   sound1_out=sound1_np,
                                                   mse=mse)
            if self.best_result is None or self.best_result.mse > self.current_result.mse:
                self.best_result = self.current_result

    def _plot_closure(self, step):
        print('Iteration {:5d}    Loss {:5f}  MSE: {:f}'.format(step, self.total_loss.item(),
                                                                self.current_result.mse), '\r', end='')
        if step % self.show_every == self.show_every - 1:
            x_l = magnitude2heatmap(self.current_result.sound1)
            save_image("s_{}".format(step), x_l, self.output_path)

            x_l = magnitude2heatmap(self.current_result.sound1_out)
            save_image("s_out_{}".format(step), x_l, self.output_path)

            x_l = (self.current_result.mask1)
            save_image("mask_{}".format(step), x_l, self.output_path)

            a1_wav = istft_reconstruction(self.current_result.sound1[0], self.phase)
            librosa.output.write_wav(os.path.join(self.output_path, 'sound_{}.wav'.format(step)), a1_wav, self.audRate)


    def finalize(self):
        save_graph(self.image_name + "_psnr", self.psnrs, self.output_path)
        save_image(self.image_name + "_sound", magnitude2heatmap(self.best_result.sound1), self.output_path)
        a1_wav = istft_reconstruction(self.best_result.sound1[0], self.phase)
        librosa.output.write_wav(os.path.join(self.output_path, 'sound.wav'), a1_wav, self.audRate)

        save_image(self.image_name + "_sound_out", magnitude2heatmap(self.best_result.sound1_out), self.output_path)
        save_image(self.image_name + "_mask", self.best_result.mask1, self.output_path)
        save_image(self.image_name + "_original", magnitude2heatmap(np.concatenate(self.images, axis=-1)), self.output_path)
        for i in range(self.N):
            if i % 2 != 0:
                self.images[i] *= 0
        save_image(self.image_name + "_input", magnitude2heatmap(np.concatenate(self.images, axis=-1)), self.output_path)
        x = np.concatenate(self.images, axis=-1)
        in_wav = istft_reconstruction(x[0], self.phase)
        librosa.output.write_wav(os.path.join(self.output_path, 'sound_input.wav'), in_wav, self.audRate)


SeparationResult = namedtuple("SeparationResult",
                              ['sound1', 'sound1_out', 'mask1', 'mse'])

if __name__ == "__main__":
    # params
    audRate = 11000
    start_time = 0
    audLen = 12
    seg_num = 24

    test_audio = args.input
    path_out = args.output
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    y, sr = librosa.load(test_audio)
    audio = librosa.resample(y, sr, audRate)

    if audRate * audLen > audio.shape[0]:
        n = int(audLen * audRate / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio_seg = audio[start_time * audRate: (start_time + audLen) * audRate]
    amp, phase = stft(audio_seg)
    mag = magnitude2heatmap(amp)
    mix_wav = istft_reconstruction(amp, phase)
    amp = np.hsplit(amp, seg_num)
    for i in range(seg_num):
        amp[i] = np.expand_dims(amp[i], axis=0)
    s = Separation('sounds', path_out, amp, phase, audRate, seg_num)
    s.optimize()
    s.finalize()
    librosa.output.write_wav(os.path.join(path_out, 'original.wav'), mix_wav, audRate)
