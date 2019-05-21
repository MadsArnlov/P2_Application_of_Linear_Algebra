# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 12:10:44 2019

@author: Lasse
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
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
    elif name == "DB13":
        filt = [[5.2200350984548e-07, -4.700416479360808e-06, 1.0441930571407941e-05, 3.067853757932436e-05, -0.0001651289885565057,
                 4.9251525126285676e-05, 0.000932326130867249, -0.0013156739118922766, -0.002761911234656831, 0.007255589401617119, 
                 0.003923941448795577, -0.02383142071032781, 0.002379972254052227, 0.056139477100276156, -0.026488406475345658, 
                 -0.10580761818792761, 0.07294893365678874, 0.17947607942935084, -0.12457673075080665, -0.31497290771138414, 
                 0.086985726179645, 0.5888895704312119, 0.6110558511587811, 0.3119963221604349, 0.08286124387290195, 0.009202133538962279],
                [-0.009202133538962279, 0.08286124387290195, -0.3119963221604349, 0.6110558511587811, -0.5888895704312119, 0.086985726179645, 
                 0.31497290771138414, -0.12457673075080665, -0.17947607942935084, 0.07294893365678874, 0.10580761818792761, -0.026488406475345658, 
                 -0.056139477100276156, 0.002379972254052227, 0.02383142071032781, 0.003923941448795577, -0.007255589401617119, -0.002761911234656831, 
                 0.0013156739118922766, 0.000932326130867249, -4.9251525126285676e-05, -0.0001651289885565057, -3.067853757932436e-05, 1.0441930571407941e-05, 
                 4.700416479360808e-06, 5.2200350984548e-07]]
        inv_filt = [[0.009202133538962279, 0.08286124387290195, 0.3119963221604349, 0.6110558511587811, 0.5888895704312119, 0.086985726179645, 
                     -0.31497290771138414, -0.12457673075080665, 0.17947607942935084, 0.07294893365678874, -0.10580761818792761, -0.026488406475345658, 
                     0.056139477100276156, 0.002379972254052227, -0.02383142071032781, 0.003923941448795577, 0.007255589401617119, -0.002761911234656831, 
                     -0.0013156739118922766, 0.000932326130867249, 4.9251525126285676e-05, -0.0001651289885565057, 3.067853757932436e-05,
                     1.0441930571407941e-05, -4.700416479360808e-06, 5.2200350984548e-07], 
                    [5.2200350984548e-07, 4.700416479360808e-06, 1.0441930571407941e-05, -3.067853757932436e-05, -0.0001651289885565057, 
                     -4.9251525126285676e-05, 0.000932326130867249, 0.0013156739118922766, -0.002761911234656831, -0.007255589401617119, 
                     0.003923941448795577, 0.02383142071032781, 0.002379972254052227, -0.056139477100276156, -0.026488406475345658, 
                     0.10580761818792761, 0.07294893365678874, -0.17947607942935084, -0.12457673075080665, 0.31497290771138414, 0.086985726179645, 
                     -0.5888895704312119, 0.6110558511587811, -0.3119963221604349, 0.08286124387290195, -0.009202133538962279]]
    elif name == "BO13":
        filt = [[-0.0883883476, 0.0883883476, 0.7071067812, 0.7071067812, 0.0883883476, -0.0883883476],
                [0, 0, -0.7071067812, 0.7071067812, 0, 0]]
        inv_filt = [[0, 0, 0.7071067812, 0.7071067812, 0, 0],
                    [-0.0883883476, -0.0883883476, 0.7071067812, -0.7071067812, 0.0883883476, 0.0883883476]]
    elif name == "DB16":
        filt = np.array([[-2.1093396300980412e-08, 2.3087840868545578e-07, -7.363656785441815e-07,
                          -1.0435713423102517e-06, 1.133660866126152e-05, -1.394566898819319e-05,
                          -6.103596621404321e-05, 0.00017478724522506327, 0.00011424152003843815,
                          -0.0009410217493585433, 0.00040789698084934395, 0.00312802338120381,
                          -0.0036442796214883506, -0.006990014563390751, 0.013993768859843242,
                          0.010297659641009963, -0.036888397691556774, -0.007588974368642594,
                          0.07592423604445779, -0.006239722752156254, -0.13238830556335474,
                          0.027340263752899923, 0.21119069394696974, -0.02791820813292813,
                          -0.3270633105274758, -0.08975108940236352, 0.44029025688580486,
                          0.6373563320829833, 0.43031272284545874, 0.1650642834886438,
                          0.03490771432362905, 0.0031892209253436892],
                         [-0.0031892209253436892, 0.03490771432362905, -0.1650642834886438,
                          0.43031272284545874, -0.6373563320829833, 0.44029025688580486,
                          0.08975108940236352, -0.3270633105274758, 0.02791820813292813,
                          0.21119069394696974, -0.027340263752899923, -0.13238830556335474,
                          0.006239722752156254, 0.07592423604445779, 0.007588974368642594,
                          -0.036888397691556774, -0.010297659641009963, 0.013993768859843242,
                          0.006990014563390751, -0.0036442796214883506, -0.00312802338120381,
                          0.00040789698084934395, 0.0009410217493585433, 0.00011424152003843815,
                          -0.00017478724522506327, -6.103596621404321e-05, 1.394566898819319e-05,
                          1.133660866126152e-05, 1.0435713423102517e-06, -7.363656785441815e-07,
                          -2.3087840868545578e-07, -2.1093396300980412e-08]])
        inv_filt = np.array([[0.0031892209253436892,0.03490771432362905,0.1650642834886438,
0.43031272284545874,0.6373563320829833,0.44029025688580486,-0.08975108940236352,
-0.3270633105274758,-0.02791820813292813,0.21119069394696974,0.027340263752899923,
-0.13238830556335474,-0.006239722752156254,0.07592423604445779,-0.007588974368642594,
-0.036888397691556774,0.010297659641009963,0.013993768859843242,-0.006990014563390751,
-0.0036442796214883506,0.00312802338120381,0.00040789698084934395,-0.0009410217493585433,
0.00011424152003843815,0.00017478724522506327,-6.103596621404321e-05,-1.394566898819319e-05,
1.133660866126152e-05,-1.0435713423102517e-06,-7.363656785441815e-07,2.3087840868545578e-07,
-2.1093396300980412e-08],
                         [-2.1093396300980412e-08,-2.3087840868545578e-07,-7.363656785441815e-07,
1.0435713423102517e-06,1.133660866126152e-05,1.394566898819319e-05,-6.103596621404321e-05,
-0.00017478724522506327,0.00011424152003843815,0.0009410217493585433,0.00040789698084934395,
-0.00312802338120381,-0.0036442796214883506,0.006990014563390751,0.013993768859843242,
-0.010297659641009963,-0.036888397691556774,0.007588974368642594,0.07592423604445779,
0.006239722752156254,-0.13238830556335474,-0.027340263752899923,0.21119069394696974,
0.02791820813292813,-0.3270633105274758,0.08975108940236352,0.44029025688580486,-0.6373563320829833,
0.43031272284545874,-0.1650642834886438,0.03490771432362905,-0.0031892209253436892]])
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


def matrix_filters(filt, size):
    filt = filters(filt)
    zero_matrix = np.zeros((size, size))
    for k in range(size//len(filt[0][0])):
        for l in range(len(filt[0][0])):
            zero_matrix[k][l + k * len(filt[0])] = filt[0][0][-1 + l]
    for m in range(size//len(filt[0][1])):
        for j in range(len(filt[0][1])):
            zero_matrix[m + size//2][j + m * len(filt[0])] = filt[0][1][-1 + j]
    inv_zero_matrix = np.matrix.transpose(zero_matrix)
    return inv_zero_matrix


def plot_filter(filtername):
    filt, inv_filt = filters(filtername)
    plt.figure(figsize=(14, 10))#.suptitle("Filter Coefficients for {:s}".format(filtername), fontsize=18, y=0.95)
    for i in range(2):
        plt.subplot(2, 2, i+1)
        plt.stem(filt[i], 'c-')
        if i == 0:
            plt.title("Low-pass Decomposition", fontsize=16)
        else:
            plt.title("High-pass Decomposition", fontsize=16)
        plt.axis([-0.1, len(filt[i]) + 0.1, -max(np.abs(filt[i])) - 0.1, max(np.abs(filt[i])) + 0.1])
    for i in range(2):
        plt.subplot(2, 2, i+3)
        plt.stem(inv_filt[i], 'c-')
        if i == 0:
            plt.title("Low-pass Reconstruction", fontsize=16)
        else:
            plt.title("High-pass Reconstruction", fontsize=16)
        plt.axis([-0.1, len(inv_filt[i]) + 0.1, -max(np.abs(inv_filt[i])) - 0.1, max(np.abs(inv_filt[i]))+ 0.1])    
    plt.savefig("db16.pdf")
    plt.show()


# =============================================================================
# Convolution and Multiresolution
# =============================================================================
def cir_conv_downs(signal, filt):
    h = ndimage.convolve1d(signal, filt[0], output = 'float', mode = 'wrap', origin = 0)
    g = ndimage.convolve1d(signal, filt[1], output = 'float', mode = 'wrap', origin = 0)
    h = h[0:len(h):2]
    g = g[0:len(g):2]
    return h, g


def cir_conv_ups(sub_signal, inv_filt, path):
    zeros = np.zeros(len(sub_signal)*2)
    for i in range(len(sub_signal)):
        zeros[i*2] = sub_signal[i]
    sub_signal_ups = zeros
    if path == 0:
        inv = ndimage.convolve1d(sub_signal_ups, inv_filt[0], output = 'float', mode = 'wrap', origin = 0)
    elif path == 1:
        inv = ndimage.convolve1d(sub_signal_ups, inv_filt[1], output = 'float', mode = 'wrap', origin = 0)
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


def perfect_reconstruction(packets, filt):
    new_packets = packets.copy()
    recon_packets = []
    for i in range(len(new_packets)):
        if i != 0:
            new_packets[-i] = recon_packets
        for j in range(len(new_packets[-i])//2):
            hgstack = np.hstack((new_packets[-i][j*2], new_packets[-i][j*2+1]))
            recon_packets.append(np.dot(matrix_filters(filt, len(hgstack)), hgstack))
    return recon_packets


# =============================================================================
# Threshold Denoisning
# =============================================================================
#def threshold_denoising(multires, path):
#    multires[-1][int(path[-1])][abs(multires[-1][int(path[-1])]) < (max(multires[-1][int(path[-1])]) * 0)] = 0
#    return multires

def threshold_denoising(signal, threshold):
    signal = np.where(signal < threshold * np.amax(signal), signal, 0)
    return signal

# =============================================================================
# Cross Correlation
# =============================================================================
def cross_corr(signal1, signal2):
    plt.figure(figsize=(14, 5))
    plt.subplot(2, 2, 1)
    plt.plot(signal1, 'c,')
    plt.subplot(2, 2, 2)
    plt.plot(signal2, 'c,')
    plt.show()
#    signal1[:1000] = 0
#    signal1[2**19-1000:] = 0
#    signal2[:7000] = 0
#    signal2[2**19-7000:] = 0

    correlation = np.correlate(signal1, signal2, 'full')
    plt.figure(figsize=(14, 4))
    plt.title("Cross Correlation", fontsize=18)
    plt.plot(correlation, 'g', np.argmax(correlation), max(correlation), 'kx')
    plt.show()
    print("Signal 2 is shifted in time with", len(signal2) - (np.argmax(correlation) + 1), "samples")
    return len(signal2) - (np.argmax(correlation) + 1)


# =============================================================================
# Data
# =============================================================================
data_folder = Path("Test_recordings/With_noise/impuls300pr.min_speaker2")
file_to_open = [data_folder / "Test_recording microphone{}_impuls_speaker2.wav".format(i) for i in range(1,4)]

sampling_frequency, data1 = wavfile.read(file_to_open[0])
sampling_frequency, data2 = wavfile.read(file_to_open[1])
sampling_frequency, data3 = wavfile.read(file_to_open[2])

data_s = sampling_frequency * 10         # start value for data interval
data_e = data_s + 2**19                  # end value for data interval

x = [data1[data_s:data_e], data2[data_s:data_e], data3[data_s:data_e]]

x_fault = [data1[800000:800000+2**19], data2[800000:800000+2**19], data3[800000:800000+2**19]]
x_fault_norm = [x_fault[0]/scipy.std(x_fault[0]), x_fault[1]/scipy.std(x_fault[1]), x_fault[2]/scipy.std(x_fault[2])]
x_fault_norm = [hamming(x_fault_norm[i]) for i in range(len(x_fault_norm))]


# =============================================================================
# Plot of Data
# =============================================================================
#plt.figure(figsize=(14, 10))
#plt.subplot(311)
#plt.plot(data1, 'r,', label = "Microphone \u03B1")
#plt.legend()
#plt.ylabel('Voltage [mV]')
#plt.axvline(x=800001, linewidth = 2, color = 'k', linestyle = "--")
#plt.axvline(x=1324289, linewidth = 2, color = 'k', linestyle = "--")
#
#plt.subplot(312)
#plt.plot(data2, 'r,', label = "Microphone \u03B2")
#plt.legend(loc='upper right')
#plt.ylabel('Voltage [mV]')
#plt.axvline(x=800001, linewidth = 2, color = 'k', linestyle = "--")
#plt.axvline(x=1324289, linewidth = 2, color = 'k', linestyle = "--")
#
#plt.subplot(313)
#plt.plot(data3, 'r,', label = "Microphone \u03B3")
#plt.legend()
#plt.xlabel("Samples")
#plt.ylabel('Voltage [mV]')
#plt.axvline(x=800001, linewidth = 2, color = 'k', linestyle = "--")
#plt.axvline(x=1324289, linewidth = 2, color = 'k', linestyle = "--")
#
#plt.savefig('soundsignals_of_microphones_experiment_7.png')
#plt.show()


# =============================================================================
# Time Delay Estimation and Triangulation
# =============================================================================
signal1 = fsinew(12, 30, 0, 0, 0)
signal2 = np.hstack((np.zeros(50), fsinew(12, 30, 0, 0, 0)))
signal3 = np.hstack((np.zeros(100), fsinew(12, 30, 0, 0, 0)))

plt.figure(figsize=(14,9))
plt.subplot(311)
plt.plot(signal1, 'k-', label = '$\mathbf{x}_1$')
plt.legend(loc='upper right')
plt.subplot(312)
plt.plot(signal2, 'k-', label = '$\mathbf{x}_2$')
plt.legend(loc='upper right')
plt.subplot(313)
plt.plot(signal3, 'k-', label = '$\mathbf{x}_3$')
plt.legend(loc='upper right')
plt.xlabel("Samples", fontsize = 14)
plt.savefig('time_delay_signals.pdf')
plt.show()

plt.figure(figsize=(14,9))
plt.subplot(311)
correlation = np.correlate(signal1, signal2, 'full')
plt.plot(np.linspace(-4046, 4196, 8241), correlation, 'g-', label = '$\mathbf{x}_1\star\mathbf{x}_2$')
plt.plot(np.argmax(correlation)-4046, max(correlation), 'kx')
plt.legend(loc='upper right')
plt.axvline(x=50, linewidth = 2, color = 'k', linestyle = "--")
plt.subplot(312)
correlation = np.correlate(signal1, signal3, 'full')
plt.plot(np.linspace(-3996, 4296, 8291), correlation, 'g-', label = '$\mathbf{x}_1\star\mathbf{x}_3$')
plt.plot(np.argmax(correlation)-3996, max(correlation), 'kx')
plt.legend(loc='upper right')
plt.axvline(x=100, linewidth = 2, color = 'k', linestyle = "--")
plt.subplot(313)
correlation = np.correlate(signal2, signal3, 'full')
plt.plot(np.linspace(-4096, 4246, 8341), correlation, 'g-', label ='$\mathbf{x}_2\star\mathbf{x}_3$')
plt.plot(np.argmax(correlation)-4096, max(correlation), 'kx')
plt.legend(loc='upper right')
plt.axvline(x=50, linewidth = 2, color = 'k', linestyle = "--")
plt.xlabel("Samples", fontsize = 14)
plt.savefig('time_delay_signals_cross-correlated.pdf')
plt.show()


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
filt, inv_filt = filters("haar")

#packets, list_path = packet_decomposition(x_fault_norm[0], filt, 7, 128)
#
#path = np.ones(4)
#
#multires, path = multiresolution(x_fault_norm[0], filt, path)
#inv_multires0 = inv_multiresolution(inv_filt, multires, path)
#
#multires, path = multiresolution(x_fault_norm[1], filt, path)
#inv_multires1 = inv_multiresolution(inv_filt, multires, path)
#
#multires, path = multiresolution((x_fault_norm[2]), filt, path)
#inv_multires2 = inv_multiresolution(inv_filt, multires, path)
#
#cross1 = cross_corr(inv_multires0, inv_multires1)
#time_shift1 = sampling_frequency/cross1
#
#cross2 = cross_corr(inv_multires0, inv_multires2)
#time_shift2 = sampling_frequency/cross2
#
#cross3 = cross_corr(inv_multires1, inv_multires2)
#time_shift3 = sampling_frequency/cross3


# =============================================================================
# Synthetic Analysis
# =============================================================================
#x_synthetic = sinew(10, 10)
#
#packets, list_path = packet_decomposition(x_synthetic, filt, 4, 0)
#recon_packets = perfect_reconstruction(packets, 'haar')
#
#path = np.zeros(3)
#
#multires, path = multiresolution(x_synthetic, filt, path)
#inv_multires = inv_multiresolution(inv_filt, multires, path)
#
#multires, path = multiresolution(x_synthetic[1], filt, path)
#inv_multires2 = inv_multiresolution(inv_filt, multires, path)
#
#cross1 = cross_corr(inv_multires, recon_packets[0])
#time_shift1 = sampling_frequency/cross1
#
#cross1 = cross_corr(x_synthetic[0], x_synthetic[1])
#time_shift1 = sampling_frequency/cross1


# =============================================================================
# Print of execution time
# =============================================================================
end = time.time()
print('The code is executed in', end - start, "seconds")