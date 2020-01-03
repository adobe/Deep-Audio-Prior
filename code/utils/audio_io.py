"""
 # Copyright 2019 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it. If you have received this file from a source other than Adobe,
 # then your use, modification, or distribution of it requires the prior
 # written permission of Adobe. 
 
"""

import numpy as np
import librosa
import cv2



def load_audio_file( path):

    audio_raw, rate = librosa.load(path, sr=None, mono=True)
    return audio_raw, rate


def load_audio(path, audLen, audSec, audRate, nearest_resample=False):
    audio = np.zeros(audLen, dtype=np.float32)

    # load audio
    audio_raw, rate = load_audio_file(path)

    # repeat if audio is too short
    #if audio_raw.shape[0] < rate * audSec:
     #   n = int(rate * audSec / audio_raw.shape[0]) + 1
      #  audio_raw = np.tile(audio_raw, n)
    # resample
    if rate > audRate:
        # print('resmaple {}->{}'.format(rate, self.audRate))
        if nearest_resample:
            audio_raw = audio_raw[::rate // audRate]
        else:
            audio_raw = librosa.resample(audio_raw, rate, audRate)
    audio = audio_raw[:audLen]
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.

    return audio

def stft(audio, n_fft=1022, hop_length=172):

    spec = librosa.stft(
    audio, n_fft=n_fft, hop_length=hop_length)
    amp = np.abs(spec)
    phase = np.angle(spec)

    return amp, phase
    #return torch.from_numpy(amp), torch.from_numpy(phase)

def istft_reconstruction(mag, phase, hop_length=172):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length)
    return np.clip(wav, -1., 1.)

def magnitude2heatmap(mag, log=True, scale=200.):
    if log:
        mag = np.log10(mag + 1.)
    mag *= scale
    mag[mag > 255] = 255
    mag = mag.astype(np.uint8)
    if mag.shape[0] == 1:
        mag = mag[0]
    mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    #mag_color = mag_color[:, :, ::-1]
    return mag_color