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
    """
    Two signals x1 and x2 are compared in the frequency domain to determine
    if x2 contains a new frequency.

    X_diff is used to identify the frequency by the difference of modulus of
    X1 and X2.

    X_diff is then used to find the index of the new frequency.

    All other indices than that of the new frequency, the mirrored frequency
    and 0-frequency are set to 0 in X_identity.

    Lastly the frequency and phase are calculated.
    The phase is computed by using the duration of the signals x1 and x2,
    given by len(x1)/fs.
    """
    X1 = np.fft.fft(x1)
    X2 = np.fft.fft(x2)
    X_diff = np.abs(X2) - np.abs(X1)

    index = np.argmax(X_diff[:fs//2])
    X_identity = np.zeros(len(X_diff), dtype=complex)
    X_identity[index] = X2[index]
    X_identity[-index] = X2[-index]
    X_identity[0] = X2[0]
    X = [X1, X2, X_identity]

    frequency = index / duration
    phase = np.angle(X_identity[index])
    return phase, frequency, X


def sample_delay(samples, frequency):
    delay = []
    delay.append(samples[1] - samples[0])
    delay.append(samples[2] - samples[0])
    delay.append(samples[2] - samples[1])
    period_samples = fs/frequency
    for i in range(len(delay)):
        if delay[i] < 0:
            while delay[i] <= -56:
                delay[i] += period_samples
        elif delay[i] > 0:
            while delay[i] >= 56:
                delay[i] -= period_samples
    return delay


def corr(x1, x2):
    correlation = np.correlate(x1, x2, 'full')
    d = len(x1) - (np.argmax(correlation) + 1)
    return d
# =============================================================================
# Data
# =============================================================================
data_folder = Path("Test_recordings/With_noise/240-480Hz_speaker4")
file_to_open = [data_folder / "Test_recording microphone{}_240-480Hz_speaker4.wav".format(i) for i in range(1,4)]

fs, data1 = wavfile.read(file_to_open[1])
fs, data2 = wavfile.read(file_to_open[0])
fs, data3 = wavfile.read(file_to_open[2])


data1, data2, data3 = data1/sp.std(data1), data2/sp.std(data2), data3/sp.std(data3)

"Set frequency of the new frequency"
f_new = 500

data_s =  100000
data_m1 = data_s + 2**19
data_m2 = 800000
data_e = data_m2 + 2**19

x_prior = [data1[data_s:data_m1], data2[data_s:data_m1], data3[data_s:data_m1]]
x_fault = [data1[data_m2:data_e], data2[data_m2:data_e], data3[data_m2:data_e]]
# =============================================================================
# Execution
# =============================================================================
"Artificial x-axis for amplitude spectrum"
N = len(x_fault[0])
duration = N / fs
fmax = 1250
upper_limit = int(fmax*duration)
xfrequencies = np.arange(0, N // 2) / duration

"Determine phase, frequency and FFT's of signal"
phase0, f0, X0 = identify_f(x_prior[0], x_fault[0])
phase1, f1, X1= identify_f(x_prior[1], x_fault[1])
phase2, f2, X2 = identify_f(x_prior[2], x_fault[2])

"IFFT"
x0 = np.fft.ifft(X0[2]).real
x1 = np.fft.ifft(X1[2]).real
x2 = np.fft.ifft(X2[2]).real

f = (f0 + f1 + f2)/3
w = 2*np.pi*f
t = np.arange(0, N) / fs

s10 = corr(x1[:10*int(fs/f)], x0[:9*int(fs/f)])
s20 = corr(x2[:10*int(fs/f)], x0[:9*int(fs/f)])
s21 = corr(x2[:10*int(fs/f)], x1[:9*int(fs/f)])

s = [s10, s20, s21]

samples = [phase0/w * fs, phase1/w * fs, phase2/w * fs]
delay = sample_delay(samples, f)

plt.figure(figsize=(16,9))
plt.subplot(3,1,1)
plt.plot(x1[:10*int(fs/f)], 'r-', x0[:9*int(fs/f)], 'b-')
plt.grid()
plt.legend(())
plt.subplot(3,1,2)
plt.plot(x2[:10*int(fs/f)], 'r-', x0[:9*int(fs/f)], 'b-')
plt.grid()
plt.legend(())
plt.subplot(3,1,3)
plt.plot(x2[:10*int(fs/f)], 'r-', x1[:9*int(fs/f)], 'b-')
plt.grid()
plt.legend(())
#plt.savefig("idft.png")
plt.show()

for i in range(len(delay)):
    print("The sample delay is", "{:4d}".format(int(delay[i])), "samples.")

print(s)