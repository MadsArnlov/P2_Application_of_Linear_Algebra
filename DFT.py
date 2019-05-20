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
import acoustics
#start = time.time()


# =============================================================================
# Fourier
# =============================================================================
def fft(signal, fs = 1, highest_frequency = 1250):
    N = len(signal)
    duration = N / fs
    spectrum = int(highest_frequency*duration)
    frequencies = np.arange(0, N // 2) / duration
    x_fft = np.fft.fft(signal)[0:N // 2]

    t = np.arange(0, len(signal)) / N
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
#    plt.subplot(2,1,1)
    plt.grid()
    plt.xlabel("Frequency [Hz]", fontsize=14)
    plt.plot(frequencies[:spectrum], x_fft_difference[:spectrum], 'k-')
    plt.show()

    X = np.zeros(len(x_fft1)*2, dtype=complex)
    X[np.argmax(x_fft_difference)] = x_fft2[np.argmax(x_fft_difference)]
    new_signal = np.fft.ifft(X)
#    t = (np.arange(0, 48000*30) / 48000)
#    phase = np.arctan2(np.real(x_fft2[np.argmax(x_fft_difference)]),
#                       np.imag(x_fft2[np.argmax(x_fft_difference)]))
    frequency = np.argmax(x_fft_difference) / duration
#    new_signal = np.sin(t * 2*np.pi*frequency + phase)
    plt.figure(figsize=(12, 3.5))
#    plt.subplot(2,1,2)
    t = np.arange(0,len(new_signal)) / 48000
    plt.plot(t[:int(len(new_signal)/(duration*frequency))], new_signal.real[:int(len(new_signal)/(duration*frequency))], 'k-')
    plt.xlabel("Time [s]", fontsize=14)
    plt.grid()
    plt.savefig("frequency_spectrum.pdf")
    plt.show()
    return new_signal, frequency


# =============================================================================
# Cross Correlation
# =============================================================================
def cross_corr(signal1, signal2):
#    plt.figure(figsize=(14, 5))
#    plt.subplot(2, 2, 1)
#    plt.plot(signal1, 'k,')
#    plt.subplot(2, 2, 2)
#    plt.plot(signal2, 'k,')
#    plt.show()
#    signal1[:1000] = 0
#    signal1[2**19-1000:] = 0
#    signal2[:7000] = 0
#    signal2[2**19-7000:] = 0
#    plt.figure(figsize=(14, 5))
#    plt.subplot(2, 2, 1)
#    plt.plot(signal1, 'b,')
#    plt.subplot(2, 2, 2)
#    plt.plot(signal2, 'b,')
#    plt.show()
    
    correlation = np.correlate(signal1, signal2, 'full')
    plt.figure(figsize=(14, 4))
    plt.title("Cross Correlation", fontsize=18)
    plt.plot(correlation, 'g', np.argmax(correlation), max(correlation), 'kx')
    plt.show()
#    print("Signal 2 is shifted in time with", len(signal1) - (np.argmax(correlation) + 1), "samples")
    return len(signal1) - (np.argmax(correlation) + 1)

# =============================================================================
# Time Delay Estimation
# =============================================================================
def sample_delay(time1, time2, time3, frequency):
    delay = []
    delay.append(time1 - time2)
    delay.append(time1 - time3)
    delay.append(time2 - time3)
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
data_folder = Path("Test_recordings/With_noise/1000-500Hz_speaker2/")
file_to_open = [data_folder / "Test_recording microphone{}_1000-500Hz_speaker2.wav".format(i) for i in range(1,4)]

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
#x_fft0_prior, duration, spectrum, frequencies = fft(x_prior[0], sampling_frequency)
#x_fft0_fault, duration, spectrum, frequencies = fft(x_fault[0], sampling_frequency)
#new_signal0, new_frequency0 = new_freq(x_fft0_prior, x_fft0_fault, frequencies, spectrum, duration)
##fft(new_signal, sampling_frequency)
#
#x_fft1_prior, duration, spectrum, frequencies = fft(x_prior[1], sampling_frequency)
#x_fft1_fault, duration, spectrum, frequencies = fft(x_fault[1], sampling_frequency)
#new_signal1, new_frequency1 = new_freq(x_fft1_prior, x_fft1_fault, frequencies, spectrum, duration)
#
#x_fft2_prior, duration, spectrum, frequencies = fft(x_prior[2], sampling_frequency)
#x_fft2_fault, duration, spectrum, frequencies = fft(x_fault[2], sampling_frequency)
#new_signal2, new_frequency2 = new_freq(x_fft2_prior, x_fft2_fault, frequencies, spectrum, duration)
#
#sample1 = cross_corr(new_signal0.imag[:48000], new_signal1.imag[:48000])
#sample2 = cross_corr(new_signal0.imag[:48000], new_signal2.imag[:48000])
#sample3 = cross_corr(new_signal1.imag[:48000], new_signal2.imag[:48000])
#
#print(sample1, sample2, sample3)

#time2 = cross_corr(new_signal0[:len(new_signal0)//3], data1[sampling_frequency*18:sampling_frequency*18 + int(duration*sampling_frequency)//2])
#time1 = cross_corr(new_signal0[:len(new_signal0)//3], data2[sampling_frequency*18:sampling_frequency*18 + int(duration*sampling_frequency)//2])
#time3 = cross_corr(new_signal0[:len(new_signal0)//3], data3[sampling_frequency*18:sampling_frequency*18 + int(duration*sampling_frequency)//2])
#
#sample_delay(time1, time2, time3, new_frequency0)

#end = time.time()

# =============================================================================
# Synthetic: Identifying new frequency
# =============================================================================
#signal1 = fsinew(J=18,freq1 = 0, freq2 = 0, freq3 = 10, freq4 = 25, phase1 = 0, 
#                   phase2 = 0, phase3 = 0, phase4 = 0)
#signal2 = fsinew(J=18,freq1 = 0, freq2 = 0, freq3 = 10, freq4 = 50, phase1 = 0, 
#                   phase2 = 0, phase3 = 0, phase4 = np.pi)
#noise1 = acoustics.generator.brown(2**18)
#noise2 = acoustics.generator.brown(2**18)
#signal1_dft = np.fft.fft(signal1)[:len(signal1)//2]
#X1 = np.zeros(len(signal1_dft)*2, dtype=complex)
#X1[np.argmax(np.abs(signal1_dft))] = signal1_dft[np.argmax(np.abs(signal1_dft))]
#new_signal1 = np.fft.ifft(X1)

#signal2_dft = np.fft.fft(signal2)[:len(signal2)//2]
#X2 = np.zeros(len(signal2_dft)*2, dtype=complex)
#X2[np.argmax(np.abs(signal2_dft))] = signal2_dft[np.argmax(np.abs(signal2_dft))]
#new_signal2 = np.fft.ifft(X2)
#
#samples = cross_corr(1j*new_signal1[:2*48000], 1j*new_signal2[:2*48000])

#x_fft0_prior, duration, spectrum, frequencies = fft(signal1+noise1/4, len(signal1), 70)
#x_fft0_fault, duration, spectrum, frequencies = fft(signal2+noise2/4, len(signal2), 70)
#new_signal0, new_frequency0 = new_freq(x_fft0_prior, x_fft0_fault, frequencies, spectrum, duration)

#t = (1 * np.arange(0, len(signal1)) / len(signal2))
#plt.figure(figsize=(16, 9))
#plt.subplot(2, 2, 1)
#plt.plot(t, signal1+noise1, 'r--')
#plt.grid()
#plt.xlabel("Time [s]", fontsize=14)
#plt.subplot(2, 2, 2)
#plt.plot(t, signal2+noise2, 'b--')
#plt.grid()
#plt.xlabel("Time [s]", fontsize=14)
#plt.subplot(2,2,3)
#plt.stem(frequencies[:spectrum], np.abs(x_fft0_prior)[:spectrum], 'k-', 'ko')
#plt.grid()
#plt.xlabel("Frequency [Hz]", fontsize=14)
#plt.subplot(2,2,4)
#plt.stem(frequencies[:spectrum], np.abs(x_fft0_fault)[:spectrum], 'k-', 'ko')
#plt.grid()
#plt.xlabel("Frequency [Hz]", fontsize=14)
#plt.savefig("identify_new_frequency.pdf")
#plt.show()
