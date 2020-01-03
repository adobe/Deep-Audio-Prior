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
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from net import skip, skip_mask_vec
from net.losses import ExclusionLoss, plot_image_grid,\
    StdLoss, NonZeroLoss, FreqVarLoss, MaskVarLoss, FreqGradLoss,\
    MaskGradLoss,MaskTCLoss,ProjLoss,BinaryLoss,DissLoss,NonZeroMaskLoss_v2,\
    MaskDeactivationLoss
from net.noise import get_noise, get_video_noise
from utils.image_io import *
from utils.audio_io import *
from skimage.measure import compare_mse
import numpy as np
import torch
import cv2
import torch.nn as nn
from collections import namedtuple
from torch.autograd import Variable, Function
from torch.optim.lr_scheduler import StepLR
import time
import argparse

parser = argparse.ArgumentParser(description='DAP mask interaction')

# Data specifications
parser.add_argument('--input_mix', type=str, default='data/mask/violin_dog.wav',
                    help='input sound mixture')
parser.add_argument('--output', type=str, default='output/mask',
                    help='results')
parser.add_argument('--dea_map', type=str, default='data/mask/ckpt/mask2_dea.npy',
                    help='deactivation binary map')
parser.add_argument('--dea_map_id', type=str, default=2,
                    help='deactivation binary map id')

args = parser.parse_args()


class Separation(object):
    def __init__(self, image_name, output_path, dea_map, dea_map_id, image, phase, audRate, numSeg, plot_during_training=True, show_every=10, ckpt='data/mask/ckpt', num_iter=100,
                 original_mask1=None, original_mask2=None, original_sound1=None, original_sound2=None):
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
        self.parameters = None
        self.ckpt = ckpt
        if not os.path.exists(self.ckpt):
            os.makedirs(self.ckpt)
        self.learning_rate = 0.0003
        self.input_depth = 1
        self.N = numSeg
        self.mask1_net_inputs = None
        self.mask2_net_inputs = None
        self.sound1_net_inputs = None
        self.sound2_net_inputs = None
        self.original_mask1 = original_mask1
        self.original_mask2 = original_mask2
        self.original_sound1 = original_sound1
        self.original_sound2 = original_sound2
        self.mask1_net = None
        self.mask2_net = None
        self.sound1_net = None
        self.sound2_net = None
        self.total_loss = None
        self.mask1_out = None
        self.mask2_out = None
        self.sound1_out = None
        self.sound2_out = None
        self.current_result = None
        self.best_result = None
        self._init_all()


    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._load_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        self.images = self.image
        self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.image]

    def _init_inputs(self):

        self.B, self.C, self.W, self.H = self.images_torch[0].size()

        self.mask1_input = torch.load(os.path.join(self.ckpt, 'mask1_input.pt'))
        self.mask2_input = torch.load(os.path.join(self.ckpt, 'mask2_input.pt'))
        self.sound1_input = torch.load(os.path.join(self.ckpt, 'sound1_input.pt'))
        self.sound2_input = torch.load(os.path.join(self.ckpt, 'sound2_input.pt'))

        self.images_torch = torch.cat(self.images_torch, dim=1).view(-1, self.C, self.W, self.H)

        #load deactivate mask for mask1
        mask_dea = np.load(dea_map)
        mask_dea = np.hsplit(mask_dea, self.N)
        self.mask_dea = [np_to_torch(x).type(torch.cuda.FloatTensor) for x in mask_dea]
        self.mask_dea = torch.cat(self.mask_dea, dim=1).view(-1, self.C, self.W, self.H)
        self.dea_map_id = dea_map_id


    def _load_nets(self):
        self.mask1_net.load_state_dict(torch.load(os.path.join(self.ckpt, 'mask1_net.pt')))
        self.mask2_net.load_state_dict(torch.load(os.path.join(self.ckpt, 'mask2_net.pt')))
        self.sound1_net.load_state_dict(torch.load(os.path.join(self.ckpt, 'sound1_net.pt')))
        self.sound2_net.load_state_dict(torch.load(os.path.join(self.ckpt, 'sound2_net.pt')))


    def _init_parameters(self):
        self.parameters = [p for p in self.mask1_net.parameters()] + \
                          [p for p in self.mask2_net.parameters()] + \
                          [p for p in self.sound1_net.parameters()] + \
                          [p for p in self.sound2_net.parameters()]

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

        mask2_net = skip_mask_vec(
            self.input_depth, self.images[0].shape[0],
            num_channels_down=[16, 32, 64],
            num_channels_up=[16, 32, 64],
            num_channels_skip=[0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask2_net = mask2_net.type(data_type)

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

        sound2_net = skip(
            self.input_depth, self.images[0].shape[0],
            num_channels_down=[16, 32, 64],
            num_channels_up=[16, 32, 64],
            num_channels_skip=[0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=False, need_relu=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.sound2_net = sound2_net.type(data_type)
        self.relu = nn.ReLU()

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.l1_loss = nn.L1Loss().type(data_type)
        self.exclusion_loss = ExclusionLoss().type(data_type)
        self.freq_grad_loss = FreqGradLoss().type(data_type)
        self.binary_loss = BinaryLoss().type(data_type)
        self.nonzero_loss = NonZeroLoss().type(data_type)
        self.nonzero_mask_loss = NonZeroMaskLoss_v2().type(data_type)
        self.mask_deactivation_loss = MaskDeactivationLoss().type(data_type)

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

        weight = torch.log1p(self.images_torch)
        weight = torch.clamp(weight, 1e-3, 10)

        mask1_out = (self.mask1_net(self.mask1_input))
        mask2_out = (self.mask2_net(self.mask2_input))

        B, C, W, H = mask1_out.size()

        mask1_out = (torch.sigmoid(torch.mean(mask1_out, dim=-2).unsqueeze(-2)))
        mask2_out = (torch.sigmoid(torch.mean(mask2_out, dim=-2).unsqueeze(-2)))
        self.mask1_out = mask1_out.repeat(1, 1, W, 1)
        self.mask2_out = mask2_out.repeat(1, 1, W, 1)
        self.sound1_out = self.sound1_net(self.sound1_input)
        self.sound2_out = self.sound2_net(self.sound2_input)
        self.sound1 = self.mask1_out * self.sound1_out
        self.sound2 = self.mask2_out * self.sound2_out

        self.l1 = self.l1_loss(self.sound1 + self.sound2, self.images_torch)
        self.l2 = self.freq_grad_loss(self.sound1_out)#*10
        self.l3 = self.freq_grad_loss(self.sound2_out)#*10
        self.l4 = self.exclusion_loss(self.sound1, self.sound2)
        self.l5 = self.binary_loss(self.mask1_out) * 0.01
        self.l6 = self.binary_loss(self.mask2_out) * 0.01
        self.l7 = self.nonzero_mask_loss(self.mask1_out, self.mask2_out, weight)

        # mask interaction
        if self.dea_map_id == 1:
            self.l8 = self.mask_deactivation_loss(self.mask1_out, self.mask_dea)
        else:
            self.l8 = self.mask_deactivation_loss(self.mask2_out, self.mask_dea)


        self.total_loss = self.l1 + self.l2 + self.l3 + self.l4 + self.l5 + self.l6 + self.l7 + self.l8
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
            mask1_out = self.mask1_out
            mask2_out = self.mask2_out
            mask1_out[mask1_out<0.1] = 0
            mask2_out[mask2_out<0.1] = 0
            sound1_out_np = torch_to_np(torch.cat(torch.chunk(self.sound1_out*mask1_out, self.N, dim=0), dim=-1))
            sound2_out_np = torch_to_np(torch.cat(torch.chunk(self.sound2_out*mask2_out, self.N, dim=0), dim=-1))

            mask1_out_np  = (torch_to_np(torch.cat(torch.chunk(self.mask1_out, self.N, dim=0), dim=-1))*255.).astype(np.uint8)
            mask2_out_np  = (torch_to_np(torch.cat(torch.chunk(self.mask2_out, self.N, dim=0), dim=-1))*255.).astype(np.uint8)

            im_np = torch_to_np(torch.cat(torch.chunk(self.images_torch, self.N, dim=0), dim=-1))
            mse = compare_mse(im_np, sound1_out_np+sound2_out_np)


            self.psnrs.append(mse)
            self.current_result = SeparationResult(mask1=mask1_out_np[0], mask2=mask2_out_np[0], sound1=sound1_out_np,
                                                   sound2=sound2_out_np, sound1_out=sound1_np, sound2_out=sound2_np,
                                                   mse=mse)
            if self.best_result is None or self.best_result.mse > self.current_result.mse:
                self.best_result = self.current_result

    def _plot_closure(self, step):
        print(
        'Iteration {:5d}    Loss {:5f}  MSE: {:f}'.format(step, self.total_loss.item(), self.current_result.mse), '\r',
        end='')
        if step % self.show_every == self.show_every - 1:
            x_l = magnitude2heatmap(self.current_result.sound1)
            x_r = magnitude2heatmap(self.current_result.sound2)

            x = np.concatenate([x_l, x_r], axis=-3)
            save_image("s1+s2_{}".format(step), x, self.output_path)

            x_l = magnitude2heatmap(self.current_result.sound1_out)
            x_r = magnitude2heatmap(self.current_result.sound2_out)
            x = np.concatenate([x_l, x_r], axis=-3)
            save_image("s1_out+s2_out_{}".format(step), x, self.output_path)

            x_l = (self.current_result.mask1)
            x_r = (self.current_result.mask2)

            x = np.concatenate([x_l, x_r], axis=-2)
            save_image("mask1+mask2_{}".format(step), x, self.output_path)

            a1_wav = istft_reconstruction(self.current_result.sound1[0], self.phase)
            librosa.output.write_wav(os.path.join(self.output_path, 'sound_sep1_{}.wav'.format(step)), a1_wav,
                                     self.audRate)
            a2_wav = istft_reconstruction(self.current_result.sound2[0], self.phase)
            librosa.output.write_wav(os.path.join(self.output_path, 'sound_sep2_{}.wav'.format(step)), a2_wav,
                                     self.audRate)

    def finalize(self):
        save_graph(self.image_name + "_psnr", self.psnrs, self.output_path)
        save_image(self.image_name + "_sound1", magnitude2heatmap(self.best_result.sound1), self.output_path)
        save_image(self.image_name + "_sound2", magnitude2heatmap(self.best_result.sound2), self.output_path)
        a1_wav = istft_reconstruction(self.best_result.sound1[0], self.phase)
        librosa.output.write_wav(os.path.join(self.output_path, 'sound_sep1.wav'), a1_wav, self.audRate)
        a2_wav = istft_reconstruction(self.best_result.sound2[0], self.phase)
        librosa.output.write_wav(os.path.join(self.output_path, 'sound_sep2.wav'), a2_wav, self.audRate)

        save_image(self.image_name + "_sound1_out", magnitude2heatmap(self.best_result.sound1_out), self.output_path)
        save_image(self.image_name + "_sound2_out", magnitude2heatmap(self.best_result.sound2_out), self.output_path)
        save_image(self.image_name + "_mask1", self.best_result.mask1, self.output_path)
        save_image(self.image_name + "_mask2", self.best_result.mask2, self.output_path)
        save_image(self.image_name + "_original", magnitude2heatmap(np.concatenate(self.images, axis=-1)),
                   self.output_path)

SeparationResult = namedtuple("SeparationResult", ['mask1', 'mask2', 'sound1', 'sound2', 'sound1_out', 'sound2_out', 'mse'])

if __name__ == "__main__":

    # params
    audRate = 11000
    start_time = 0
    audLen = 12
    seg_num = 24  #

    test_mix = args.input_mix
    path_out = args.output
    dea_map = args.dea_map
    dea_map_id = args.dea_map_id
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    y, sr = librosa.load(test_mix, mono=False)
    audio = librosa.resample(y, sr, audRate)
    if audRate * audLen > audio.shape[1]:
        n = int(audLen * audRate / audio.shape[1]) + 1
        audio = np.tile(audio, n)
    audio_seg1 = audio[0, start_time * audRate: (start_time + audLen) * audRate]
    audio_seg2 = audio[1, start_time * audRate: (start_time + audLen) * audRate]

    amp_s1, phase = stft(audio_seg1)
    amp_s2, phase = stft(audio_seg2)
    mag1 = magnitude2heatmap(amp_s1)
    mag2 = magnitude2heatmap(amp_s2)
    audio_mix = audio_seg1 + audio_seg2

    amp_mix, phase_mix = stft(audio_mix)
    mag_mix = magnitude2heatmap(amp_mix)
    mix_wav = istft_reconstruction(amp_mix, phase_mix)
    amp_mix = np.hsplit(amp_mix, seg_num)

    # Sep
    for i in range(seg_num):
        amp_mix[i] = np.expand_dims(amp_mix[i], axis=0)

    t1 = time.time()
    s = Separation('sounds', path_out, dea_map, dea_map_id, amp_mix, phase_mix, audRate, seg_num)
    s.optimize()
    t2 = time.time()
    print("running time: ", t2-t1)

    s.finalize()

    librosa.output.write_wav(os.path.join(path_out, 'gt1.wav'), audio_seg1, audRate)
    librosa.output.write_wav(os.path.join(path_out, 'gt2.wav'), audio_seg2, audRate)
    librosa.output.write_wav(os.path.join(path_out, 'mixture.wav'), mix_wav, audRate)
    cv2.imwrite(os.path.join(path_out, 'spec_a1.jpg'), mag1)
    cv2.imwrite(os.path.join(path_out, 'spec_a2.jpg'), mag2)