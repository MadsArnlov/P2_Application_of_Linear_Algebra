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
from data_manipulation import zpad, hann, hamming, recw, fsinew, sinew
start = time.time()


# =============================================================================
# Fourier
# =============================================================================
def fft(signal, fs = 1, highest_frequency = 1250):
    N = len(signal)
    duration = N / fs
    spectrum = int(highest_frequency*duration)
    frequencies = np.arange(0, N // 2) / duration
    x_fft = np.abs(np.fft.fft(signal, norm = 'ortho'))[0:N // 2]
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    plt.plot(signal, 'r,')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(frequencies[:spectrum], x_fft[:spectrum], 'b-')
    plt.grid()
    plt.show()
    return x_fft, frequencies, spectrum


def new_freq(x_fft1, x_fft2, frequencies, spectrum):
    x_fft3 = x_fft2 - x_fft1
    plt.figure(figsize=(12, 3.5))
    plt.grid()
    plt.plot(frequencies[:spectrum], x_fft3[:spectrum], 'b-')
    plt.show()

    new_signal = np.sin((2 * np.pi * np.arange(0, 48000) / 48000) * np.argmax(x_fft3)/5.46)
    plt.figure(figsize=(12, 3.5))
    plt.plot(range(0, spectrum), new_signal[:spectrum], 'k.')
    plt.show()
    return new_signal, x_fft3


# =============================================================================
# Data
# =============================================================================
data_folder = Path("Test_recordings/Without_noise/1000-500Hz_speaker2_uden_støj/")
file_to_open = [data_folder / "Test_recording microphone{:d}_1000-500Hz_speaker2_uden_støj.wav".format(i) for i in range(1,4)]

sampling_frequency, data1 = wavfile.read(file_to_open[0])
sampling_frequency, data2 = wavfile.read(file_to_open[1])
sampling_frequency, data3 = wavfile.read(file_to_open[2])

data_s = sampling_frequency * 7
data_m1 = data_s + 2**18
data_e = len(data1) - sampling_frequency * 7
data_m2 = data_e - 2**18

x_prior = [data1[data_s:data_m1], data2[data_s:data_m1], data3[data_s:data_m1]]
x_fault = [data1[data_m2:data_e], data2[data_m2:data_e], data3[data_m2:data_e]]


# =============================================================================
# Execution
# =============================================================================
#dft1 = fft(fsinew(18, freq1 = 2000, freq2 = 1500, freq3 = 500))
#dft2 = fft(fsinew(18, freq1 = 2000, freq2 = 1500, freq3 = 760))
#dft_new = dft1-dft2
#for i in range(len(dft_new)):
#    if dft_new[i] > 0:
#        dft_new[i] = 0

x_fft0_prior, frequencies, spectrum = fft(x_prior[0], sampling_frequency)
x_fft0_fault, frequencies, spectrum = fft(x_fault[0], sampling_frequency)
new_signal, x_fft_new = new_freq(x_fft0_prior, x_fft0_fault, frequencies, spectrum)
fft(new_signal, sampling_frequency)

#x_fft1_prior = fft(x_prior[1])
#x_fft1_fault = fft(x_fault[1])
#x_fft2_prior = fft(x_prior[2])
#x_fft2_fault = fft(x_fault[2])


end = time.time()