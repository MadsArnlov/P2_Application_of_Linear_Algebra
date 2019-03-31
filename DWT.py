# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 12:10:44 2019

@author: Lasse
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import ndimage
start = time.time()

# =============================================================================
# Filters
# =============================================================================
haar = [[1/2, 1/2], [1/2, -1/2]]

# =============================================================================
# Data generation
# =============================================================================
def data_generator(J = 10, freq1 = 10, freq2 = 15, freq3 = 20, phase1 = 0, phase2 = 0, phase3 = 0):
    N = 2**J
    t = np.arange(1 , N+1)
    A = 2 * np.pi * t / N
    x1 = np.sin(A * freq1 + phase1)
    x2 = np.sin(A * freq2 + phase2)
    x3 = np.sin(A * freq3 + phase3)
    x_sum = x1 + x2 + x3
    return x_sum

def wave_generator(J = 10, freq = 10, phase = 0):
    N = 2**J
    t = np.arange(1 , N+1)
    wave = np.sin(2 * np.pi * t / N * freq + phase)
    return wave

# =============================================================================
# Data manipulation
# =============================================================================
#def zeropadding(signal):

# =============================================================================
# Convolution
# =============================================================================
def cir_conv_downs(signal, filt):
    h = sp.ndimage.convolve1d(signal, filt[0], output = 'float', mode = 'wrap', origin = -1)
    g = sp.ndimage.convolve1d(signal, filt[1], output = 'float', mode = 'wrap', origin = -1)
    h = h[0:len(h):2]
    g = g[0:len(g):2]
    return h, g

# =============================================================================
# Multiresolution
# =============================================================================
def multiresolution(signal, filt, path = [0]):
    multires = []
    multires.append(signal)
    for i in range(len(path)):
        if i == 0:
            signal = cir_conv_downs(signal, filt)
            multires.append(signal)
        elif path[i] == 0:
            signal = cir_conv_downs(signal[0], filt)
            multires.append(signal)
        elif path[i] == 1:
            signal = cir_conv_downs(signal[1], filt)
            multires.append(signal)
    
    
    plt.figure(figsize=(13, 7))
    t = np.arange(1 , len(multires[0])+1)
    plt.subplot(2, 1, 1)
    plt.plot(t, multires[0], 'b.')
    plt.ylabel('Original signal')

    t1 = np.arange(1 , (len(multires[1][0])+1))
    plt.subplot(2, 2, 3)
    plt.plot(t1, multires[1][0], 'r.')
    plt.ylabel('h')

    plt.subplot(2, 2, 4)
    plt.plot(t1, multires[1][1], 'r.')
    plt.ylabel('g')

    plt.xlabel('Samples')
    plt.show()

# =============================================================================
# Execution
# =============================================================================
multiresolution(data_generator(), haar)

end = time.time()
print(end - start)