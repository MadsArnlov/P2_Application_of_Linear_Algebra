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
import scipy as sp
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path
from data_manipulation import zpad, hann, hamming, recw, fsinew, sinew
import acoustics
#start = time.time()


# =============================================================================
# Fourier
# =============================================================================
def fft(signal, fs, highest_frequency = 1250):
    N = len(signal)
    duration = N / fs
    spectrum = int(highest_frequency*duration)
    frequencies = np.arange(0, N // 2) / duration
    x_fft = np.fft.fft(signal)

    "Plot of amplitude spectrum"
#    t = np.arange(0, len(signal)) / 48000
#    plt.figure(figsize=(12, 7))
#    plt.subplot(2, 1, 1)
#    plt.plot(t, signal.real, 'r,')
#    plt.grid()
#    plt.subplot(2, 1, 2)
#    plt.plot(frequencies[:spectrum], np.abs(x_fft)[:spectrum], 'k-')
#    plt.grid()
#    plt.show()
    return x_fft, duration, spectrum, frequencies


def new_freq(x_fft1, x_fft2, frequencies, spectrum, duration):
    x_fft_difference = np.abs(x_fft2) - np.abs(x_fft1)
#    plt.figure(figsize=(12, 3.5))
##    plt.subplot(2,1,1)
#    plt.grid()
#    plt.xlabel("Frequency [Hz]", fontsize=14)
#    plt.plot(frequencies[:spectrum], x_fft_difference[:spectrum], 'k-')
#    plt.show()
    i = np.argmax(x_fft_difference[:spectrum])
    X = np.zeros(len(x_fft_difference), dtype=complex)
    X[i] = x_fft2[i]
    X[-i] = x_fft2[-i]
    new_signal = np.fft.ifft(X)
#    plt.figure(figsize=(14,8))
#    plt.plot(np.angle(X))
#    print(np.angle(X[i]), np.angle(X[-i]))
#    plt.show()
#    t = (np.arange(0, 48000*30) / 48000)
#    phase = np.arctan2(np.real(x_fft2[i]),
#                       np.imag(x_fft2[i]))
    phase = np.angle(X[i])
    frequency = i / duration
#    frequency = int(frequency*10)*10**-1
#    new_signal = np.sin(t * 2*np.pi*frequency + phase)
#    plt.figure(figsize=(12, 3.5))
##    plt.subplot(2,1,2)
#    t = np.arange(0,len(new_signal)) / 48000
#    plt.plot(new_signal.real[:p1], 'k-')
#    plt.xlabel("Time [s]", fontsize=14)
#    plt.grid()
##    plt.savefig("frequency_spectrum.pdf")
#    plt.show()
    return new_signal, frequency, phase


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
    plt.figure(figsize=(12, 4))
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
    delay.append(time1)
    delay.append(time2)
    delay.append(time3)
    period_samples = 48000/frequency
    for i in range(len(delay)):
        if delay[i] < 0:
            while delay[i] <= -56:
                delay[i] += period_samples
        elif delay[i] > 0:
            while delay[i] >= 56:
                delay[i] -= period_samples
#        delay[i] = delay[i] % period_samples
#        if delay[i] >= 56:
#            delay[i] = delay[i] - period_samples
    return delay


# =============================================================================
# Data
# =============================================================================
data_folder = Path("Test_recordings/Without_noise/1000-500Hz_speaker2_uden_støj")
file_to_open = [data_folder / "Test_recording microphone{}_1000-500Hz_speaker2_uden_støj.wav".format(i) for i in range(1,4)]

sampling_frequency, data1 = wavfile.read(file_to_open[1])
sampling_frequency, data2 = wavfile.read(file_to_open[0])
sampling_frequency, data3 = wavfile.read(file_to_open[2])


data1, data2, data3 = data1/sp.std(data1), data2/sp.std(data2), data3/sp.std(data3)


data_s = sampling_frequency * 0
data_m1 = 96*1000#data_s + sampling_frequency*14
data_e = sampling_frequency*16 + 96*1000#sampling_frequency * 16
data_m2 = sampling_frequency*16

x_prior = [data1[data_s:data_m1], data2[data_s:data_m1], data3[data_s:data_m1]]
x_fault = [data1[data_m2:data_e], data2[data_m2:data_e], data3[data_m2:data_e]]

# =============================================================================
# Execution 1 
# =============================================================================
x_prior, x_fault = [hamming(x_prior[i]) for i in range(len(x_prior))], [hamming(x_fault[i]) for i in range(len(x_fault))]
x_fft0_prior, duration, spectrum, frequencies = fft(x_prior[0], sampling_frequency, sampling_frequency//2)
x_fft0_fault, duration, spectrum, frequencies = fft(x_fault[0], sampling_frequency, sampling_frequency//2)
new_signal0, new_frequency0, angle0 = new_freq(x_fft0_prior, x_fft0_fault, frequencies, spectrum, duration)

x_fft1_prior, duration, spectrum, frequencies = fft(x_prior[1], sampling_frequency, sampling_frequency//2)
x_fft1_fault, duration, spectrum, frequencies = fft(x_fault[1], sampling_frequency, sampling_frequency//2)
new_signal1, new_frequency1, angle1 = new_freq(x_fft1_prior, x_fft1_fault, frequencies, spectrum, duration)

x_fft2_prior, duration, spectrum, frequencies = fft(x_prior[2], sampling_frequency, sampling_frequency//2)
x_fft2_fault, duration, spectrum, frequencies = fft(x_fault[2], sampling_frequency, sampling_frequency//2)
new_signal2, new_frequency2, angle2 = new_freq(x_fft2_prior, x_fft2_fault, frequencies, spectrum, duration)

p1 = 10*int(sampling_frequency/new_frequency0)
p2 = 9*int(sampling_frequency/new_frequency0)

new_signal0, new_signal1, new_signal2 = new_signal0/sp.std(new_signal0), new_signal1/sp.std(new_signal1), new_signal2/sp.std(new_signal2)

plt.figure(figsize=(14,10))
plt.subplot(3,1,1)
plt.plot(new_signal0[:p1], 'r-', np.hstack((new_signal1[:p2], np.zeros(p1-p2))), 'b-')
plt.grid()
plt.subplot(3,1,2)
plt.plot(new_signal0[:p1], 'r-', np.hstack((new_signal2[:p2], np.zeros(p1-p2))), 'b-')
plt.grid()
plt.subplot(3,1,3)
plt.plot(new_signal1[:p1], 'r-', np.hstack((new_signal2[:p2], np.zeros(p1-p2))), 'b-')
plt.grid()
plt.show()

sample1 = cross_corr(new_signal0[:p1], np.hstack((new_signal1[:p2], np.zeros(p1-p2))))
sample2 = cross_corr(new_signal0[:p1], np.hstack((new_signal2[:p2], np.zeros(p1-p2))))
sample3 = cross_corr(new_signal1[:p1], np.hstack((new_signal2[:p2], np.zeros(p1-p2))))

print(sample1, sample2, sample3)
print(new_frequency0, new_frequency1, new_frequency2)
delay = sample_delay(sample1, sample2, sample3, new_frequency0)

print(18*"-", "\n", delay[0], delay[1], delay[2])
# =============================================================================
# Execution 2
# =============================================================================
#x_fft0_prior, duration, spectrum, frequencies = fft(x_prior[0], sampling_frequency, sampling_frequency//2)
#x_fft0_fault, duration, spectrum, frequencies = fft(x_fault[0], sampling_frequency, sampling_frequency//2)
#new_signal0, new_frequency0 = new_freq(x_fft0_prior, x_fft0_fault, frequencies, spectrum, duration)
#
#p1 = 1*1000
#p2 = 800
#new_signal0 = new_signal0/sp.std(new_signal0)
#
##x_fault[0], x_fault[1], x_fault[2] = x_fault[0]/sp.std(x_fault[0]), x_fault[1]/sp.std(x_fault[1]), x_fault[2]/sp.std(x_fault[2])
#
#
#plt.figure(figsize=(14,10))
#plt.subplot(3,1,1)
#plt.plot(data1[900000:900000+p2], 'b-', np.hstack((new_signal0[:p2], np.zeros(800-p2))), 'r-')
#plt.grid()
#plt.subplot(3,1,2)
#plt.plot(data2[900000:900000+p2], 'b-', np.hstack((new_signal0[:p2], np.zeros(800-p2))), 'r-')
#plt.grid()
#plt.subplot(3,1,3)
#plt.plot(data3[900000:900000+p2], 'b-', np.hstack((new_signal0[:p2], np.zeros(800-p2))), 'r-')
#plt.grid()
#plt.show()
#
##for i in range(1,2):
#p1 = 1000
#p2 = 800
#sample1 = cross_corr(data1[900000:900000+p2], np.hstack((new_signal0[:p2], np.zeros(800-p2))))
#sample2 = cross_corr(data2[900000:900000+p2], np.hstack((new_signal0[:p2], np.zeros(800-p2))))
#sample3 = cross_corr(data3[900000:900000+p2], np.hstack((new_signal0[:p2], np.zeros(800-p2))))
#print(sample1, sample2, sample3)    
#samples = sample_delay(sample1, sample2, sample3, new_frequency0)
#print(samples, samples[0] - samples[1], samples[0] - samples[2], samples[1] - samples[2])
#print(15*"-")

#print(new_frequency0)
#delay = sample_delay(sample1, sample2, sample3, new_frequency0)
#
#print(18*"-", "\n", delay[0], delay[1], delay[2])
# =============================================================================
# Synthetic: Identifying new frequency
# =============================================================================
#signal1 = fsinew(J=18,freq1 = 0, freq2 = 0, freq3 = 0, freq4 = 100, phase1 = 0, 
#                   phase2 = 0, phase3 = 0, phase4 = 0)
#signal2 = fsinew(J=18,freq1 = 0, freq2 = 0, freq3 = 0, freq4 = 100, phase1 = 0, 
#                   phase2 = 0, phase3 = 0, phase4 = 2*np.pi*1/4)
#noise1 = acoustics.generator.brown(2**18)
#noise2 = acoustics.generator.brown(2**18)
#signal1 = signal1 + noise1/4
#signal2 = signal2 + noise2/4
#signal1_dft = np.fft.fft(signal1)
#X1 = np.zeros(len(signal1_dft), dtype=complex)
#i = np.argmax(np.abs(signal1_dft))
#X1[i] = signal1_dft[i]
#X1[-i] = signal1_dft[-i]
#new_signal1 = np.fft.ifft(X1)
#
#signal2_dft = np.fft.fft(signal2)
#X2 = np.zeros(len(signal2_dft), dtype=complex)
#i = np.argmax(np.abs(signal2_dft))
#X2[i] = signal2_dft[i]
#X2[-i] = signal2_dft[-i]
#new_signal2 = np.fft.ifft(X2)
#
#period = int((2**18)/100)
#
#plt.figure(figsize=(14, 10))
#plt.subplot(2,1,1)
#plt.plot(new_signal1[:10*period], 'r-', new_signal2[:9*period], 'b--')
#plt.grid()
#plt.subplot(2,1,2)
#plt.plot(new_signal2[:period])
#plt.grid()
#plt.show()
#
#
#samples = cross_corr(new_signal1[:10*period], new_signal2[:9*period])
#
#period_samples = 2**18 / 10
#
#samples = samples % period_samples
#if samples >= 2**18/10:
#    samples = samples - period_samples
#
#print(samples)

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
