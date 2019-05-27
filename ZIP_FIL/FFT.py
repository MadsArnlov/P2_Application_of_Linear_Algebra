# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path


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

    Parameters
    ----------
    x1: array_like
        input signal prior to a fault
    x2: array_like
        input signal after a fault

    Returns
    -------
    phase: float
        argument of X_identity to index, returned in radians
    frequency: float
        identified frequency
    X: array_like
        one-dimensional array, consisting of DFT of x1 and x2, as well as
        X_identity
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
    """
    Calculates the sample delay pairwise for n microphones.

    If the sample delays are shifted by one or more periods of the identified
    frequency, then the sample delays are shifted by a period until it is
    within the 56 maximum samples.

    Parameters
    ----------
    samples: array_like
        array consisting of the sample delays of n microphones
    frequency: float
        identified frequency of the speaker emitting the fault sound

    Returns
    -------
    delay: array_like
        pairwise sample delays of n microphones
    """
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


# =============================================================================
# Data
# =============================================================================
"""
To load another file, specify recording by changing data folder to either of:
    speakerA_frequencies        speakerA_impulse
    speakerB_frequencies        speakerB_impulse
    speakerC_frequencies        speakerC_impulse
    speakerD_frequencies        speakerD_impulse
"""
data_folder = Path("Test_recordings/speakerB_frequencies")
file_to_open = [data_folder / "microphone{}.wav".format(i) for i in range(1,4)]

fs, data1 = wavfile.read(file_to_open[1])
fs, data2 = wavfile.read(file_to_open[0])
fs, data3 = wavfile.read(file_to_open[2])

"The data is standardised"
data1, data2, data3 = data1/sp.std(data1), data2/sp.std(data2), data3/sp.std(data3)


"The data is segmented"
data_s =  100000
data_m1 = data_s + 2**19
data_m2 = 800000
data_e = data_m2 + 2**19

x_prior = [data1[data_s:data_m1], data2[data_s:data_m1], data3[data_s:data_m1]]
x_fault = [data1[data_m2:data_e], data2[data_m2:data_e], data3[data_m2:data_e]]
# =============================================================================
# Execution
# =============================================================================
"Used for plots"
N = len(x_fault[0])
duration = N / fs
fmax = 1500
upper_limit = int(fmax*duration)
xfrequencies = np.arange(0, N // 2) / duration

"Compute phase, frequency and FFT's of signal"
phase0, f0, X0 = identify_f(x_prior[0], x_fault[0])
phase1, f1, X1= identify_f(x_prior[1], x_fault[1])
phase2, f2, X2 = identify_f(x_prior[2], x_fault[2])

"Compute limits of axis for plots"
m0 = max(np.abs(X0[0][:upper_limit]))
m1 = max(np.abs(X1[0][:upper_limit]))
m2 = max(np.abs(X2[0][:upper_limit]))

f = (f0 + f1 + f2)/3
w = 2*np.pi*f
t = np.arange(0, N) / fs

"The sample delays are computed and printed"
samples = [phase0/w * fs, phase1/w * fs, phase2/w * fs]
delay = sample_delay(samples, f)
for i in range(len(delay)):
    print("The sample delay is", "{:4d}".format(int(delay[i])), "samples.")


# =============================================================================
# Plots
# =============================================================================
"DFT of x_prior and x_fault"
#plt.figure(figsize=(16,9))
#plt.subplot(3,2,1)
#plt.plot(xfrequencies[:upper_limit], np.abs(X0[0][:upper_limit]), 'k-', label="$|\mathcal{F}(\mathbf{x}_{prior,\u03B1})|$")
#plt.ylim((0, m0*1.1))
#plt.legend(fontsize = 'x-large')
#plt.subplot(3,2,2)
#plt.plot(xfrequencies[:upper_limit], np.abs(X0[1][:upper_limit]), 'k-', label="$|\mathcal{F}(\mathbf{x}_{fault,\u03B1})|$")
#plt.ylim((0, m0*1.1))
#plt.legend(fontsize = 'x-large')
#plt.subplot(3,2,3)
#plt.plot(xfrequencies[:upper_limit], np.abs(X1[0][:upper_limit]), 'k-', label="$|\mathcal{F}(\mathbf{x}_{prior,\u03B2})|$")
#plt.ylim((0, m1*1.1))
#plt.legend(fontsize = 'x-large')
#plt.subplot(3,2,4)
#plt.plot(xfrequencies[:upper_limit], np.abs(X1[1][:upper_limit]), 'k-', label="$|\mathcal{F}(\mathbf{x}_{fault,\u03B2})|$")
#plt.ylim((0, m1*1.1))
#plt.legend(fontsize = 'x-large')
#plt.subplot(3,2,5)
#plt.plot(xfrequencies[:upper_limit], np.abs(X2[0][:upper_limit]), 'k-', label="$|\mathcal{F}(\mathbf{x}_{prior,\u03B3})|$")
#plt.ylim((0, m2*1.1))
#plt.legend(fontsize = 'x-large')
#plt.xlabel("Frequency [Hz]", fontsize=14)
#plt.subplot(3,2,6)
#plt.plot(xfrequencies[:upper_limit], np.abs(X2[1][:upper_limit]), 'k-', label="$|\mathcal{F}(\mathbf{x}_{fault,\u03B3})|$")
#plt.ylim((0, m2*1.1))
#plt.legend(fontsize = 'x-large')
#plt.xlabel("Frequency [Hz]", fontsize=14)
#plt.show()


"Identify frequencies from prior and fault"
#plt.figure(figsize=(16,9))
#plt.subplot(3,1,1)
#plt.plot(xfrequencies[:upper_limit], np.abs(X0[1][:upper_limit]) - np.abs(X0[0][:upper_limit]), 'k-', label="$|\mathcal{F}(\mathbf{x}_{fault,\u03B1})| - |\mathcal{F}(\mathbf{x}_{prior,\u03B1})|$")
#plt.ylim((-m0*0.8, m0*0.8))
#plt.legend(fontsize = 'x-large')
#plt.subplot(3,1,2)
#plt.plot(xfrequencies[:upper_limit], np.abs(X1[1][:upper_limit]) - np.abs(X1[0][:upper_limit]), 'k-', label="$|\mathcal{F}(\mathbf{x}_{fault,\u03B2})| - |\mathcal{F}(\mathbf{x}_{prior,\u03B2})|$")
#plt.ylim((-m1*0.6, m1*0.6))
#plt.legend(fontsize = 'x-large')
#plt.subplot(3,1,3)
#plt.plot(xfrequencies[:upper_limit], np.abs(X2[1][:upper_limit]) - np.abs(X2[0][:upper_limit]), 'k-', label="$|\mathcal{F}(\mathbf{x}_{fault,\u03B3})| - |\mathcal{F}(\mathbf{x}_{prior,\u03B3})|$")
#plt.ylim((-m2*1.1, m2*1.1))
#plt.legend(fontsize = 'x-large')
#plt.xlabel("Frequency [Hz]", fontsize=14)
#plt.show()

"Plot of sine with phases"
#plt.figure(figsize=(16,9))
#plt.subplot(3,1,1)
#plt.plot(t[:5*int(fs/f)], np.sin(w*t + phase0)[:5*int(fs/f)])
#plt.subplot(3,1,2)
#plt.plot(t[:5*int(fs/f)], np.sin(w*t + phase1)[:5*int(fs/f)])
#plt.subplot(3,1,3)
#plt.plot(t[:5*int(fs/f)], np.sin(w*t + phase2)[:5*int(fs/f)])
#plt.show()
