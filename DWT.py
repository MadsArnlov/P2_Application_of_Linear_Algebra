# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 12:10:44 2019

@author: Lasse
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.io import wavfile
from pathlib import Path
from data_manipulation import zpad, hann, hamming, recw, fsinew, sinew
start = time.time()


# =============================================================================
# Filters
# =============================================================================
def filters(name = "db4"):
    name = name.upper()
    if name == "HAAR":
        filt = [[1/np.sqrt(2), 1/np.sqrt(2)],
                [1/np.sqrt(2), -1/np.sqrt(2)]]
        inv_filt = [[1/np.sqrt(2), 1/np.sqrt(2)], 
                    [-1/np.sqrt(2), 1/np.sqrt(2)]]
    elif name == "DB4":
        filt = [[-0.0105974018, 0.0328830117, 0.0308413818, -0.1870348117,
                 -0.0279837694, 0.6308807679, 0.7148465706, 0.2303778133],
                [-0.2303778133, 0.7148465706, -0.6308807679, -0.0279837694,
                 0.1870348117, 0.0308413818, -0.0328830117, -0.0105974018]]
        inv_filt = [[0.23037781330885523, 0.7148465705525415, 0.6308807679295904, -0.02798376941698385, 
                     -0.18703481171888114, 0.030841381835986965, 0.032883011666982945, -0.010597401784997278], 
                    [-0.010597401784997278, -0.032883011666982945, 0.030841381835986965, 0.18703481171888114,
                     -0.02798376941698385, -0.6308807679295904, 0.7148465705525415, -0.23037781330885523]]
    elif name == "BO13":
        filt = [[-0.0883883476, 0.0883883476, 0.7071067812, 0.7071067812, 0.0883883476, -0.0883883476],
                [0, 0, -0.7071067812, 0.7071067812, 0, 0]]
        inv_filt = [[0, 0, 0.7071067812, 0.7071067812, 0, 0],
                    [-0.0883883476, -0.0883883476, 0.7071067812, -0.7071067812, 0.0883883476, 0.0883883476]]
    elif name == "SYM5":
        filt = [[0.027333068345077982, 0.029519490925774643, -0.039134249302383094,
                 0.1993975339773936, 0.7234076904024206, 0.6339789634582119,
                 0.01660210576452232, -0.17532808990845047, -0.021101834024758855,
                 0.019538882735286728],
                [-0.019538882735286728, -0.021101834024758855, 0.17532808990845047,
                 0.01660210576452232, -0.6339789634582119, 0.7234076904024206,
                 -0.1993975339773936, -0.039134249302383094, -0.029519490925774643,
                 0.027333068345077982]]
        inv_filt = [[0.019538882735286728, -0.021101834024758855, -0.17532808990845047,
                     0.01660210576452232, 0.6339789634582119, 0.7234076904024206,
                     0.1993975339773936, -0.039134249302383094, 0.029519490925774643,
                     0.027333068345077982],
                    [0.027333068345077982, -0.029519490925774643, -0.039134249302383094,
                     -0.1993975339773936, 0.7234076904024206, -0.6339789634582119,
                     0.01660210576452232, 0.17532808990845047, -0.021101834024758855,
                     -0.019538882735286728]]
    return filt, inv_filt


def plot_filter(filtername):
    filt, inv_filt = filters(filtername)
    plt.figure(figsize=(14, 10)).suptitle("Filter Coefficients for {:s}".format(filtername), fontsize=18, y=0.95)
    for i in range(2):
        plt.subplot(2, 2, i+1)
        plt.stem(filt[i], 'c-')
        if i == 0:
            plt.title("Low-pass Decomposition", fontsize=16)
        else:
            plt.title("High-pass Decomposition", fontsize=16)
        plt.axis([-0.1, len(filt[i]) + 0.1, -max(filt[i]) - 0.1, max(filt[i]) + 0.1])
    for i in range(2):
        plt.subplot(2, 2, i+3)
        plt.stem(inv_filt[i], 'c-')
        if i == 0:
            plt.title("Low-pass Reconstruction", fontsize=16)
        else:
            plt.title("High-pass Reconstruction", fontsize=16)
        plt.axis([-0.1, len(inv_filt[i]) + 0.1, -max(inv_filt[i]) - 0.1, max(inv_filt[i])+ 0.1])    
    plt.show()


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

    plt.figure(figsize=(14, 10)).suptitle("Multiresolution Analysis with {:d} levels".format(len(path)), fontsize=18, y=0.93)
    plt.subplot(len(multires), 1, 1)
    plt.plot(multires[0], 'b-')
    #plt.axis([0, len(multires[0]), min(multires[0]), max(multires[0])])
    for i in range(len(path)):
        plt.subplot(len(multires), 2, 3+(i*2))
        plt.plot(multires[i+1][0], 'r,')
        #plt.axis([0, len(multires[0]), min(multires[i+1][0]), max(multires[i+1][0])])
        plt.subplot(len(multires), 2, 4+(i*2))
        plt.plot(multires[i+1][1], 'm,')
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

    plt.figure(figsize=(14, 10)).suptitle("Multiresolution Synthesis with {:d} levels".format(len(path)), fontsize=18, y=0.93)
    for i in range(len(inv_multires)):
        plt.subplot(len(inv_multires), 1, i+1)
        plt.plot(inv_multires[i], 'k,')
        #plt.axis([0, len(inv_multires[-1]), min(inv_multires[i]), max(inv_multires[i])])
    plt.show()
    return inv_multires[-1]


def packet_decomposition(signal, filt, levels, energylevels, plot = 0):
    packets = []
    signal = cir_conv_downs(signal, filt)
    packets.append(signal)
    for i in range(levels-1):
        signal = []
        tuple_signal = ()
        for j in range(2**(len(packets))):
            tuple_signal = cir_conv_downs(packets[len(packets)-1][j], filt)
            signal.append(tuple_signal[0])
            signal.append(tuple_signal[1])
        packets.append(signal)
    
    if plot == 1:
        for n in range(len(packets)):
            plt.figure(figsize=(14,10))
            for m in range(2**(n+1)):
                plt.subplot(1, 2**(n+1), m+1)
                plt.plot(packets[n][m], range(len(packets[n][m])), 'k,')
            plt.show()
    
    packets_energy = [packets[-1][i]**2 for i in range(len(packets[-1]))]
    packets_energy_sum = []
    for i in range(len(packets_energy)):
        packets_energy_sum.append(sum(packets_energy[i]))
    
    index_max_energy = []
    for i in range(energylevels):
        index_max_energy.append(np.argmax(packets_energy_sum))
        packets_energy_sum[np.argmax(packets_energy_sum)] = 0
    
    list_path_max_energy = []
    for k in range(energylevels):
        path_max_energy = []
        a = index_max_energy[k] + 1
        for i in range(levels):
            if a % 2 == 0:
                path_max_energy.append(1)
                if a != 0:
                    a = a / 2
            elif a % 2 == 1:
                path_max_energy.append(0)
                a = (a + 1) / 2
        path_max_energy = path_max_energy[::-1]
        list_path_max_energy.append(path_max_energy)
    
    list_freq_spec = []
    for i in range(len(list_path_max_energy)):
        freq_spec = [0, 48000/2]
        for j in range(len(list_path_max_energy[i])):
            if  list_path_max_energy[i][j] == 1:
                freq_spec_temp = freq_spec[0]
                freq_spec[0] = freq_spec[1]   
                freq_spec[1] = freq_spec_temp + (freq_spec[0] - freq_spec_temp)/2
            elif  list_path_max_energy[i][j] == 0:
                freq_spec[1] = freq_spec[1] + (freq_spec[0] - freq_spec[1])/2
        list_freq_spec.append(freq_spec)

    for l in range(energylevels):
        print('        Index   Path           Spectrum')
        print(l,'        {}   {}   {}'.format(index_max_energy[l], list_path_max_energy[l], list_freq_spec[l]))
    return packets, list_path_max_energy


# =============================================================================
# Threshold Denoisning
# =============================================================================
def threshold_denoising(multires, path):
    multires[-1][int(path[-1])][abs(multires[-1][int(path[-1])]) < (max(multires[-1][int(path[-1])]) * 0)] = 0
    return multires


# =============================================================================
# Cross Correlation
# =============================================================================
def cross_corr(signal1, signal2):
    plt.figure(figsize=(14, 5))
    plt.subplot(2, 2, 1)
    plt.plot(signal1, 'r,')
    plt.subplot(2, 2, 2)
    plt.plot(signal2, 'b,')
    plt.show()
#    signal1[:300] = 0
#    signal1[2**19-300:] = 0
    
    correlation = np.correlate(signal1, signal2, 'full')
    plt.figure(figsize=(14, 4))
    plt.title("Cross Correlation", fontsize=18)
    plt.plot(correlation, 'g', np.argmax(correlation), max(correlation), 'kx')
    plt.show()
    print("Signal 2 is shifted in time with", len(signal1) - (np.argmax(correlation) + 1), "samples")
    return len(signal1) - (np.argmax(correlation) + 1)


# =============================================================================
# Data
# =============================================================================
data_folder = Path("Test_recordings/Without_noise/impuls300pr.min_speaker2_uden_støj")
file_to_open = [data_folder / "Test_recording microphone{}_impuls_speaker2_uden_støj.wav".format(i) for i in range(1,4)]

sampling_frequency, data1 = wavfile.read(file_to_open[0])
sampling_frequency, data2 = wavfile.read(file_to_open[1])
sampling_frequency, data3 = wavfile.read(file_to_open[2])

data_s = sampling_frequency * 10         # start value for data interval
data_e = data_s + 2**19                  # end value for data interval

x = [data1[data_s:data_e], data2[data_s:data_e], data3[data_s:data_e]]

x_fault = [data1[800000:800000+2**19], data2[800000:800000+2**19], data3[800000:800000+2**19]]

# =============================================================================
# Plot of Data
# =============================================================================
#t = np.linspace(10, 10+len(x[0])/sampling_frequency, len(x[0]))
#
#plt.figure(figsize=(14,10))
#for i in range(3):
#    plt.subplot(3, 1, 1+i)
#    plt.title("Lydsignal for mikrofon {}".format(1+i), fontsize=16)
#    plt.plot(t, x[i], 'r,')
#    plt.grid()
#    plt.subplot(3, 1, 1+i)
#    plt.plot(t, hamming_window(x[i]), 'b,')
#    plt.grid
#plt.show()


# =============================================================================
# Denoise
# =============================================================================
#filt, inv_filt = filters("db4")
#wave = sinew(J = 14)
#noise = np.random.normal(0, 1, 2**14)
#signal = wave + noise
#
#plots = []
#plots.append(signal)
#for i in range(4,9):
#    packets, path = packet_decomposition(signal, filt, i)
#    multires, path = multiresolution(signal, filt, path)
#    inv_multires = inv_multiresolution(inv_filt, multires, path)
#    plots.append(inv_multires)
#
#plt.figure(figsize=(14, 8))
#plt.subplot(2, 1, 1)
#plt.plot(wave, 'b-')
#plt.grid()
#plt.subplot(2, 1, 2)
#plt.plot(signal, 'r--')
#plt.grid()
#plt.savefig("Signal_with_noise.pdf")
#plt.show()
#
#plt.figure(figsize=(14, 10))
#for i in range(len(plots)-1):
#    plt.subplot(len(plots)-1, 1, i+1)
#    plt.plot(plots[i+1], 'k-')
#    plt.grid()
#plt.savefig("Denoise_signal.pdf")
#plt.show()

# =============================================================================
# Execution
# =============================================================================
filt, inv_filt = filters("db4")
x = [hamming(x[i]) for i in range(len(x))]

packets, list_path = packet_decomposition(x_fault[0], filt, 13, 100)

#path = list_path[24]
#
#multires, path = multiresolution(x_fault[0], filt, path)
#inv_multires = inv_multiresolution(inv_filt, multires, path)
#
#multires, path = multiresolution(x_fault[1], filt, path)
#inv_multires2 = inv_multiresolution(inv_filt, multires, path)
#
#cross1 = cross_corr(inv_multires, inv_multires2)
#time_shift1 = sampling_frequency/cross1
#
#multires, path = multiresolution((x_fault[0]), filt, path)
#inv_multires = inv_multiresolution(inv_filt, multires, path)
#
#multires, path = multiresolution((x_fault[2]), filt, path)
#inv_multires2 = inv_multiresolution(inv_filt, multires, path)
#
#cross2 = cross_corr(inv_multires, inv_multires2)
#time_shift2 = sampling_frequency/cross2
#
#multires, path = multiresolution((x_fault[1]), filt, path)
#inv_multires = inv_multiresolution(inv_filt, multires, path)
#
#multires, path = multiresolution((x_fault[2]), filt, path)
#inv_multires2 = inv_multiresolution(inv_filt, multires, path)
#
#cross3 = cross_corr(inv_multires, inv_multires2)
#time_shift3 = sampling_frequency/cross3


# =============================================================================
# Synthetic Analysis
# =============================================================================
#path = np.ones(7)
#filt, inv_filt = filters("db4")
#plot_filter("haar")
#
#import Simple_sine_with_impulses as file
#import Wave_high_frequencies as file
#import Synthetic_signal as file

#shifted_signal = np.append([sinew(np.log2(file.shift))], [fsinew(file.J, file.freq1,
#                            file.freq2, file.freq3, file.freq4, file.phase1, file.phase2,
#                            file.phase3, file.phase4, file.imp_freq, file.scaling1)
#                            [0:2**file.J-file.shift]])
#
#multires, path = multiresolution(fsinew(file.J, file.freq1, file.freq2,
#                                                file.freq3, file.freq4, file.phase1,
#                                                file.phase2, file.phase3, file.phase4,
#                                                file.imp_freq, file.scaling1),
#                                                filt, path)
#inv_multires = inv_multiresolution(inv_filt, multires, path)
#
#multires, path = multiresolution(shifted_signal, filt, path)
#inv_multires2 = inv_multiresolution(inv_filt, multires, path)
#
#cross_corr(inv_multires, inv_multires2)


# =============================================================================
# Print of execution time
# =============================================================================
end = time.time()
print('The code is executed in', end - start, "seconds")
