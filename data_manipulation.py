# -*- coding: utf-8 -*-
import numpy as np
# =============================================================================
# Data Manipulation
# =============================================================================
def zpad(signal):
    """
    Zero-padded at end of signal to the next power of 2
    """
    if np.log2(len(signal)) - int(np.log2(len(signal))) != 0.0:
        signal = np.hstack((signal, np.zeros(2**(int(np.log2(len(signal))) + 1) - len(signal))))
    return signal


# =============================================================================
# Window Functions
# =============================================================================
def hann(signal):
    """
    Hann-window
    """
    signal = [signal[i] * 1/2 *(1 - np.cos(2*np.pi*i/len(signal))) for i in range(len(signal))]
    return signal


def hamming(signal):
    """
    Hamming-window
    """
    signal = [signal[i] * 25/46 *(1 - np.cos(2*np.pi*i/len(signal))) for i in range(len(signal))]
    return signal


def recw(signal, size = 15):
    """
    Rectangular-window with specific size cut off from the edges
    """
    signal[:size] = 0
    signal[-size:] = 0
    return signal


# =============================================================================
# Data Generation
# =============================================================================
def fsinew(J = 18, freq1 = 13, freq2 = 20, freq3 = 40, freq4 = 1000, freq5 = 0, phase1 = 0, 
                   phase2 = 0, phase3 = 0, phase4 = 0, phase5 = 0, imp_freq = 0, scaling1 = 1):
    """
    Data generator for a signal consisting of four sine waves as well as
    impulses
    """
    N = 2**J
    t = np.arange(1, N+1)
    A = 2 * np.pi * t / N
    x1 = np.sin(A * freq1 + phase1)*scaling1
    x2 = np.sin(A * freq2 + phase2)
    x3 = np.sin(A * freq3 + phase3)
    x4 = np.sin(A * freq4 + phase4)
    x5 = np.sin(A * freq5 + phase5)
    x_imp = np.zeros(N)
    if imp_freq != 0:
        for i in range(int(N/imp_freq), len(x_imp), int(N/(imp_freq+1))):
            x_imp[i] = 1
            x_imp[i+1] = -1
    x_sum = x1 + x2 + x3 + x4 + x5 + x_imp
    return x_sum


def sinew(J = 18, freq = 10, phase = 0):
    """
    Single sine wave with specific points, frequency and phase
    """
    N = 2**J
    t = np.arange(1 , N+1)
    A = 2 * np.pi * t / N
    wave = np.sin(A * freq + phase)
    return wave

