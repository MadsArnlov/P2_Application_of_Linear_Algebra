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
from scipy import signal
start = time.time()


# =============================================================================
# Fourier
# =============================================================================
def fft(signal, fs = 1, highest_frequency = 1250):
    N = len(signal)
    duration = N / fs
    spectrum = int(highest_frequency*duration)
    frequencies = np.arange(0, N // 2) / duration
    x_fft = np.fft.fft(signal)

    t = np.arange(0, len(signal)) / 48000
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal.real, 'r,')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(frequencies[:spectrum], np.abs(x_fft)[:spectrum], 'k-')
    plt.grid()
    plt.show()
    return x_fft, duration, spectrum, frequencies


def new_freq(x_fft1, x_fft2, frequencies, spectrum, duration):
    x_fft_difference = np.abs(x_fft2) - np.abs(x_fft1)
    plt.figure(figsize=(12, 3.5))
    plt.xlabel("Frequency [Hz]", fontsize=14)
    plt.plot(frequencies[:spectrum], x_fft_difference[:spectrum], 'k-')
    plt.show()
    
    i = np.argmax(x_fft_difference[:spectrum])
    X = np.zeros(len(x_fft1), dtype=complex)
    X[i] = x_fft2[i]
    X[-i] = x_fft2[-i]
    new_signal = np.fft.ifft(X)
    frequency = i / duration
    frequency = int(frequency*10)*10**-1
    
    plt.figure(figsize=(12, 3.5))
    plt.plot(new_signal.real[:2000], 'k,')
    plt.grid()
    plt.show()
    return new_signal, frequency


# =============================================================================
# Cross Correlation
# =============================================================================
def cross_corr(signal1, signal2):
    correlation = signal.correlate(signal1, signal2, mode='full', method='direct')
    plt.figure(figsize=(12, 4))
    plt.title("Cross Correlation", fontsize=18)
    plt.plot(correlation, 'g', np.argmax(correlation), max(correlation), 'kx')
    plt.show()
    return len(signal1) - (np.argmax(correlation) + 1)


# =============================================================================
# Time Delay Estimation
# =============================================================================
def sample_delay(time1, time2, time3, frequency):
    delay = []
    delay.append(time1)
    delay.append(time2)
    delay.append(time3)
    period_samples = 48000/frequency
    for i in range(len(delay)):
        delay[i] = delay[i] % period_samples
        if delay[i] >= 56:
            delay[i] = delay[i] - period_samples
    return delay


# =============================================================================
# Data
# =============================================================================
data_folder = Path("Test_recordings/With_noise/737-368.5Hz_speaker3")
file_to_open = [data_folder / "Test_recording microphone{}_737-368.5Hz_speaker3.wav".format(i) for i in range(1,4)]

sampling_frequency, data1 = wavfile.read(file_to_open[1])
sampling_frequency, data2 = wavfile.read(file_to_open[0])
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
x_fft0_prior, duration, spectrum, frequencies = fft(x_prior[0], sampling_frequency, sampling_frequency//2)
x_fft0_fault, duration, spectrum, frequencies = fft(x_fault[0], sampling_frequency, sampling_frequency//2)
new_signal0, new_frequency0 = new_freq(x_fft0_prior, x_fft0_fault, frequencies, spectrum, duration)

x_fft1_prior, duration, spectrum, frequencies = fft(x_prior[1], sampling_frequency, sampling_frequency//2)
x_fft1_fault, duration, spectrum, frequencies = fft(x_fault[1], sampling_frequency, sampling_frequency//2)
new_signal1, new_frequency1 = new_freq(x_fft1_prior, x_fft1_fault, frequencies, spectrum, duration)

x_fft2_prior, duration, spectrum, frequencies = fft(x_prior[2], sampling_frequency, sampling_frequency//2)
x_fft2_fault, duration, spectrum, frequencies = fft(x_fault[2], sampling_frequency, sampling_frequency//2)
new_signal2, new_frequency2 = new_freq(x_fft2_prior, x_fft2_fault, frequencies, spectrum, duration)

p1 = sampling_frequency
p2 = int(sampling_frequency - sampling_frequency//new_frequency0)


sample1 = cross_corr(new_signal0[:p1], np.hstack((new_signal1[:p2], np.zeros(p1-p2))))
sample2 = cross_corr(new_signal0[:p1], np.hstack((new_signal2[:p2], np.zeros(p1-p2))))
sample3 = cross_corr(new_signal1[:p1], np.hstack((new_signal2[:p2], np.zeros(p1-p2))))

print(sample1, sample2, sample3)
print(new_frequency0, new_frequency1, new_frequency2)
delay = sample_delay(sample1, sample2, sample3, new_frequency0)

print(56*"-", "\n", delay[0], delay[1], delay[2])


# =============================================================================
# Print of execution time
# =============================================================================
end = time.time()
print('The code is executed in', end - start, "seconds")