# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:43:46 2019

@author: lasse
"""

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
    x_fft = np.fft.fft(signal, norm = 'ortho')[0:N // 2]

#    plt.figure(figsize=(12, 7))
#    plt.subplot(2, 1, 1)
#    plt.plot(signal, 'r,')
#    plt.grid()
#    plt.subplot(2, 1, 2)
#    plt.plot(frequencies[:spectrum], np.abs(x_fft)[:spectrum], 'b-')
#    plt.grid()
#    plt.show()
    return x_fft, duration, spectrum, frequencies


def new_freq(x_fft1, x_fft2, frequencies, spectrum, duration):
    x_fft_difference = np.abs(x_fft2) - np.abs(x_fft1)
#    plt.figure(figsize=(12, 3.5))
#    plt.grid()
#    plt.plot(frequencies[:spectrum], x_fft_difference[:spectrum], 'b-')
#    plt.show()

    t = (2 * np.pi * np.arange(0, 48000) / 48000)
    phase = np.arctan2(np.real(x_fft2[np.argmax(x_fft_difference)]),
                       np.imag(x_fft2[np.argmax(x_fft_difference)]))
    frequency = np.argmax(x_fft_difference) / duration
    new_signal = 5000 * np.sin(t * frequency + phase)
#    plt.figure(figsize=(12, 3.5))
#    plt.plot(range(0, spectrum), new_signal[:spectrum], 'k.')
#    plt.show()
    return new_signal, frequency


# =============================================================================
# Cross Correlation
# =============================================================================
def cross_corr(signal1, signal2, samples = 0):
    if samples != 0:
        signal2 = signal2[len(signal2) // 2 - samples:len(signal2) // 2 + samples]
#    plt.figure(figsize=(14, 5))
#    plt.subplot(2, 2, 1)
#    plt.plot(signal1, 'r,')
#    plt.subplot(2, 2, 2)
#    plt.plot(signal2, 'b,')
#    plt.show()
    correlation = np.correlate(signal1, signal2, 'full')
#    plt.figure(figsize=(14, 4))
#    plt.title("Cross Correlation", fontsize=18)
#    plt.plot(correlation, 'g', np.argmax(correlation), max(correlation), 'kx')
#    plt.show()
#    print("Signal 2 is shifted in time with", len(signal1) - (np.argmax(correlation) + 1), "samples")
    return len(signal1) - (np.argmax(correlation) + 1)


# =============================================================================
# Time Delay Estimation
# =============================================================================
def sample_delay(time1, time2, time3, frequency):
    delay = []
    delay.append(time1 - time2)
    delay.append(time3 - time2)
    delay.append(time3 - time1)
    frequency = int(frequency*10)*10**-1
    period_samples = 48000/frequency
    for i in range(len(delay)):
        delay[i] = delay[i] % period_samples
        if delay[i] >= 56:
            delay[i] = delay[i] - period_samples
    print(delay, frequency)


# =============================================================================
# Data
# =============================================================================
data_folder = Path("Test_recordings/With_noise/240-480Hz_speaker4/")
file_to_open = [data_folder / "Test_recording microphone{}_240-480Hz_speaker4.wav".format(i) for i in range(1,4)]

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
x_fft0_prior, duration, spectrum, frequencies = fft(x_prior[0], sampling_frequency)
x_fft0_fault, duration, spectrum, frequencies = fft(x_fault[0], sampling_frequency)
new_signal, new_frequency = new_freq(x_fft0_prior, x_fft0_fault, frequencies, spectrum, duration)
#fft(new_signal, sampling_frequency)

#new_signal = new_signal[:12000]

time1 = cross_corr(new_signal, data1[sampling_frequency*20:], samples = 48000)
time2 = cross_corr(new_signal, data2[sampling_frequency*20:], samples = 48000)
time3 = cross_corr(new_signal, data3[sampling_frequency*20:], samples = 48000)

sample_delay(time1, time2, time3, new_frequency)

end = time.time()