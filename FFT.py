# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path
from data_manipulation import zpad, hann, hamming, recw, fsinew, sinew

# =============================================================================
# FFT
# =============================================================================
def identify_f(x1, x2):
    X1 = np.fft.fft(x1)
    X2 = np.fft.fft(x2)
    X_diff = np.abs(X2) - np.abs(X1)

    index = np.argmax(X_diff[:fs//2])
    X_identity = np.zeros(len(X_diff), dtype=complex)
    X_identity[index] = X2[index]
    X_identity[-index] = X2[-index]
    X = [X1, X2, X_identity]

    frequency = index / duration
    phase = np.angle(X_identity[index])
    return phase, frequency, X


def sample_delay(samples, frequency):
    delay = []
    delay.append(samples[0] - samples[1])
    delay.append(samples[0] - samples[2])
    delay.append(samples[1] - samples[2])
    period_samples = fs/frequency
    for i in range(len(delay)):
        if delay[i] < 0:
            while delay[i] <= -56:
                delay[i] += period_samples
        elif delay[i] > 0:
            while delay[i] >= 56:
                delay[i] -= period_samples
    return delay


# =============================================================================
# Data
# =============================================================================
data_folder = Path("Test_recordings/With_noise/1000-500Hz_speaker2")
file_to_open = [data_folder / "Test_recording microphone{}_1000-500Hz_speaker2.wav".format(i) for i in range(1,4)]

fs, data1 = wavfile.read(file_to_open[1])
fs, data2 = wavfile.read(file_to_open[0])
fs, data3 = wavfile.read(file_to_open[2])


data1, data2, data3 = data1/sp.std(data1), data2/sp.std(data2), data3/sp.std(data3)

"Set frequency of the new frequency"
f_new = 500

data_s = fs * 0
data_m1 = int(fs/f_new *1000)#data_s + fs*14
data_e = int(fs*16 + fs/f_new*1000)#fs* 16
data_m2 = fs*16

x_prior = [data1[data_s:data_m1], data2[data_s:data_m1], data3[data_s:data_m1]]
x_fault = [data1[data_m2:data_e], data2[data_m2:data_e], data3[data_m2:data_e]]
# =============================================================================
# Execution
# =============================================================================
"Artificial x-axis for amplitude spectrum"
#N = len(x_fault[0])
#duration = N / fs
#fmax = 1250
#upper_limit = int(fmax*duration)
#xfrequencies = np.arange(0, N // 2) / duration

phase0, f0, X0 = identify_f(x_prior[0], x_fault[0])
phase1, f1, X1= identify_f(x_prior[1], x_fault[1])
phase2, f2, X2 = identify_f(x_prior[2], x_fault[2])

f = (f0 + f1 + f2)/3

w = 2*np.pi*f

samples = [phase0/w * fs, phase1/w * fs, phase2/w * fs]
delay = sample_delay(samples, f)

plt.figure(figsize=(16,9))
plt.subplot(3,1,1)
plt.plot(np.fft.ifft(X0[2])[:int(fs/f)], 'r-')
plt.grid()
plt.subplot(3,1,2)
plt.plot(np.fft.ifft(X1[2])[:int(fs/f)], 'r-')
plt.grid()
plt.subplot(3,1,3)
plt.plot(np.fft.ifft(X2[2])[:int(fs/f)], 'r-')
plt.grid()
plt.show()

for i in range(len(delay)):
    print("The sample delay is", "{:4d}".format(int(delay[i])), "samples.")

