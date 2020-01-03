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
from .audio_io import *

def audio_process(snd, audRate, audLen, seg_num, start_time):

    y, sr = librosa.load(snd)
    audio = librosa.resample(y, sr, audRate)
    audio_mix = audio[start_time * audRate: (start_time + audLen) * audRate]

    amp_mix, phase_mix = stft(audio_mix)
    mix_wav = istft_reconstruction(amp_mix, phase_mix)
    #print("map size:", amp_mix.shape)
    amp_mix = np.hsplit(amp_mix, seg_num)

    return amp_mix, phase_mix, mix_wav
