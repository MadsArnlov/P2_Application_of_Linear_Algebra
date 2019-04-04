# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 12:10:44 2019

@author: Lasse
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
start = time.time()

# =============================================================================
# Filters
# =============================================================================
haar = [[1/np.sqrt(2), 1/np.sqrt(2)],
        [1/np.sqrt(2), -1/np.sqrt(2)]]

db4 = [[-0.0105974018, 0.0328830117, 0.0308413818, -0.1870348117,
        -0.0279837694, 0.6308807679, 0.7148465706, 0.2303778133],
       [-0.2303778133, 0.7148465706, -0.6308807679, -0.0279837694,
        0.1870348117, 0.0308413818, -0.0328830117, -0.0105974018]]

bo13 = [[-0.0883883476, 0.0883883476, 0.7071067812, 0.7071067812, 0.0883883476, -0.0883883476],
        [0, 0, -0.7071067812, 0.7071067812, 0, 0]]

# =============================================================================
# Inverse Filters
# =============================================================================
inv_haar = [[1/np.sqrt(2), 1/np.sqrt(2)], 
            [-1/np.sqrt(2), 1/np.sqrt(2)]]

inv_db4 = [[0.23037781330885523, 0.7148465705525415, 0.6308807679295904, -0.02798376941698385, 
            -0.18703481171888114, 0.030841381835986965, 0.032883011666982945, -0.010597401784997278], 
           [-0.010597401784997278, -0.032883011666982945, 0.030841381835986965, 0.18703481171888114,
            -0.02798376941698385, -0.6308807679295904, 0.7148465705525415, -0.23037781330885523]]

inv_bo13 = [[0, 0, 0.7071067812, 0.7071067812, 0, 0],
            [-0.0883883476, -0.0883883476, 0.7071067812, -0.7071067812, 0.0883883476, 0.0883883476]]

# =============================================================================
# Data Generation
# =============================================================================
def data_generator(J = 18, freq1 = 13, freq2 = 20, freq3 = 40, phase1 = 0, phase2 = 0, phase3 = 0, imp_freq = 0, scaling = 1):
    N = 2**J
    t = np.arange(1 , N+1)
    A = 2 * np.pi * t / N
    x1 = np.sin(A * freq1 + phase1)*scaling
    x2 = np.sin(A * freq2 + phase2)
    x3 = np.sin(A * freq3 + phase3)
    x_imp = np.zeros(N)
    if imp_freq != 0:
        for i in range(int(N/imp_freq), len(x_imp), int(N/(imp_freq+1))):
            x_imp[i] = 1
            x_imp[i+1] = -1
    x_sum = x1 + x2 + x3 + x_imp
    return x_sum

def wave_generator(J = 18, freq = 10, phase = 0):
    N = 2**J
    t = np.arange(1 , N+1)
    A = 2 * np.pi * t / N
    wave = np.sin(A * freq + phase)
    return wave

# =============================================================================
# Data Manipulation
# =============================================================================
def zero_padding(signal):
    if np.log2(len(signal)) - int(np.log2(len(signal))) != 0.0:
        signal = np.hstack((signal, np.zeros(2**(int(np.log2(len(signal))) + 1) - len(signal))))
    return signal


# =============================================================================
# Convolution and Multiresolution
# =============================================================================
def cir_conv_downs(signal, filt):
    h = ndimage.convolve1d(signal, filt[0], output = 'float', mode = 'wrap', origin = -1)
    g = ndimage.convolve1d(signal, filt[1], output = 'float', mode = 'wrap', origin = -1)
    h = h[0:len(h):2]
    g = g[0:len(g):2]
    return h, g

def cir_conv_ups(sub_signal, inv_filt, path):
    zeros = np.zeros(len(sub_signal)*2)
    for i in range(len(sub_signal)):
        zeros[i*2] = sub_signal[i]
    sub_signal_ups = zeros
    if path == 0:
        inv = ndimage.convolve1d(sub_signal_ups, inv_filt[0], output = 'float', mode = 'wrap', origin = -1)
    elif path == 1: 
        inv = ndimage.convolve1d(sub_signal_ups, inv_filt[1], output = 'float', mode = 'wrap', origin = -1)
    return inv
            
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
    
    plt.figure(figsize=(14, 10))
    plt.subplot(len(multires), 1, 1)
    plt.plot(multires[0], 'b-')
    plt.axis([0, len(multires[0]), min(multires[0]), max(multires[0])])
    for i in range(len(path)):
        plt.subplot(len(multires), 2, 3+(i*2))
        plt.plot(multires[i+1][0], 'r-')
        #plt.axis([0, len(multires[0]), min(multires[i+1][0]), max(multires[i+1][0])])
        plt.subplot(len(multires), 2, 4+(i*2))
        plt.plot(multires[i+1][1], 'r-')
        #plt.axis([0, len(multires[0]), min(multires[i+1][1]), max(multires[i+1][1])])
    plt.show()
    
    return multires, path
    
def inv_multiresolution(inv_filt, multires, path):
    inv_multires = []
    for i in range(len(path)):
        if i == 0:
            if path[-1-i] == 0:
                inv_multires.append(cir_conv_ups(multires[-1][0], inv_filt, path[-1-i]))
            elif path[-1-i] == 1:
                inv_multires.append(cir_conv_ups(multires[-1][1], inv_filt, path[-1-i]))    
        else:
            if path[-1-i] == 0:
                inv_multires.append(cir_conv_ups(inv_multires[-1], inv_filt, path[-1-i]))
            elif path[-1-i] == 1:
                inv_multires.append(cir_conv_ups(inv_multires[-1], inv_filt, path[-1-i]))
    
    plt.figure(figsize=(14, 10))
    for i in range(len(inv_multires)):
        plt.subplot(len(inv_multires), 1, i+1)
        plt.plot(inv_multires[i], 'k-')
        plt.axis([0, len(inv_multires[-1]), min(inv_multires[i]), max(inv_multires[i])])
    plt.show()
    return inv_multires[-1]

# =============================================================================
# Cross Correlation
# =============================================================================
def cross_corr(signal1, signal2):
    correlation = np.correlate(signal1, signal2, 'full')
    plt.plot(correlation, 'g')
    plt.show()
    print("Signal 2 er forskudt med", len(signal1) - (np.argmax(correlation) + 1), "samples")

# =============================================================================
# Execution
# =============================================================================
import Simple_sine_with_impulses as file

path = np.zeros(8)
filt = bo13
inv_filt = inv_bo13

shifted_signal = np.hstack([np.zeros(file.shift), data_generator(file.J, file.freq1,
                            file.freq2, file.freq3, file.phase1, file.phase2,
                            file.phase3, file.imp_freq, file.scaling)
                            [0:2**file.J-file.shift]])

multires, path = multiresolution(data_generator(file.J, file.freq1, file.freq2,
                                                file.freq3, file.phase1,
                                                file.phase2, file.phase3,
                                                file.imp_freq, file.scaling),
                                                filt, path)
inv_multires = inv_multiresolution(inv_filt, multires, path)

multires, path = multiresolution(shifted_signal, filt, path)
inv_multires2 = inv_multiresolution(inv_filt, multires, path)

cross_corr(inv_multires, inv_multires2)

end = time.time()
print('Koden eksekveres på', end - start, "sekunder")