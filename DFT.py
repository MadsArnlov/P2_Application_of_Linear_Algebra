# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:37:12 2019

@author: lasse
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path
start = time.time()


# =============================================================================
# Data Generation
# =============================================================================
def data_generator(J = 18, freq1 = 0, freq2 = 0, freq3 = 0, freq4 = 0, phase1 = 0, 
                   phase2 = 0, phase3 = 0, phase4 = 0, imp_freq = 0):
    N = 2**J
    t = np.arange(1 , N+1)
    A = 2 * np.pi * t / N
    x1 = np.sin(A * freq1 + phase1)
    x2 = np.sin(A * freq2 + phase2)
    x3 = np.sin(A * freq3 + phase3)
    x4 = np.sin(A * freq4 + phase4)
    x_imp = np.zeros(N)
    if imp_freq != 0:
        for i in range(int(N/imp_freq), len(x_imp), int(N/(imp_freq+1))):
            x_imp[i] = 1
            x_imp[i+1] = -1
    x_sum = x1 + x2 + x3 + x4 + x_imp
    return x_sum


# =============================================================================
# Fourier
# =============================================================================
def fft(x_sum, energy):
    x_fft = np.fft.rfft(x_sum, norm = 'ortho')
    x_fft = x_fft[0:24000]
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    plt.plot(x_sum, 'r-')
    plt.subplot(2, 1, 2)
    plt.plot(range(24000), x_fft, 'b-')
    plt.show()
    
    frequencies = []
    frequencies.append(np.where(x_fft >= energy))
    print(frequencies[0][0])
    return x_fft, frequencies



def new(x_fft1, x_fft2, frequencies1, frequencies2, threshold):
    x_fft3 = x_fft2 - x_fft1
    plt.figure(figsize=(12, 3.5))
    plt.plot(x_fft3, 'b-')
    plt.show()
    for i in range(len(frequencies1)):
        if abs(frequencies1[i][0] - frequencies2[i][0]) >= threshold:
            print(frequencies1[i][0], '->', frequencies2[i][0])

# =============================================================================
# Data
# =============================================================================
data_folder = Path("Test_recordings/Without_noise/1000-500Hz_speaker2_uden_støj/")
file_to_open = [data_folder / "Test_recording microphone{:d}_1000-500Hz_speaker2_uden_støj.wav".format(i) for i in range(1,4)]

sampling_frequency, data1 = wavfile.read(file_to_open[0])
sampling_frequency, data2 = wavfile.read(file_to_open[1])
sampling_frequency, data3 = wavfile.read(file_to_open[2])

data_s = sampling_frequency * 10         # start value for data interval
data_m1 = data_s + 200000
data_m2 = data_s + 2**19-200000
data_e = data_s + 2**19                  # end value for data interval

x_prior = [data1[data_s:data_m1], data2[data_s:data_m1], data3[data_s:data_m1]]
x_fault = [data1[data_m2:data_e], data2[data_m2:data_e], data3[data_m2:data_e]]


# =============================================================================
# Execution
# =============================================================================
x_fft0_prior, frequencies1_p = fft(x_prior[0], 3000)
x_fft0_fault, frequencies1_f = fft(x_fault[0], 3000)
#x_fft1_prior, frequencies2_p = fft(x_prior[1], 5000)
#x_fft1_fault, frequencies2_f = fft(x_fault[1], 5000)
#x_fft2_prior, frequencies3_p = fft(x_prior[2], 5000)
#x_fft2_fault, frequencies3_f = fft(x_fault[2], 5000)


end = time.time()