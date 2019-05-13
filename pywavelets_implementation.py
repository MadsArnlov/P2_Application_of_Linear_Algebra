# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:11:16 2019

@author: bergl
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.io import wavfile
from pathlib import Path
from data_manipulation import zpad, hann, hamming, recw, fsinew, sinew
import pywt

# =============================================================================
# Fomalia til pywt packet decomposition
# =============================================================================
def format_array(a):
    """Consistent array representation across different systems"""
    a = np.where(np.abs(a) < 1e-5, 0, a)
    return np.array2string(a, precision=5, separator=' ', suppress_small=True)

# =============================================================================
# Import af data
# =============================================================================
data_folder = Path("Test_recordings/Without_noise/737-368.5Hz_speaker3_uden_støj/")
file_to_open = [data_folder / "Test_recording microphone{:d}_737-368.5Hz_speaker3_uden_støj.wav".format(i) for i in range(1,4)]

sampling_frequency, data1 = wavfile.read(file_to_open[0])
sampling_frequency, data2 = wavfile.read(file_to_open[1])
sampling_frequency, data3 = wavfile.read(file_to_open[2])

data_s = sampling_frequency * 10         # start value for data interval
data_e = data_s + 2**19                  # end value for data interval

x = [data1[data_s:data_e], data2[data_s:data_e], data3[data_s:data_e]]
x_1 = np.array([1,2,3,4,5,6,7,8])

# =============================================================================
# implementatering af pywt packet decomposition
# =============================================================================
def packets_selection(signal, packets, threshold, freq1, freq2, freq3, freq4):
    path_list = []
#    threshold = np.sum(signal**2)
    for i in wavelet_packet.get_leaf_nodes(False):
        if np.sum((wavelet_packet[i.path].data)**2) < threshold:
            del wavelet_packet[i.path]
        else:
            path_list.append(i.path)

        list_freq_spec = []
    for i in range(len(path_list)):
        freq_spec = [0, 48000/2]
        for j in range(len(path_list[i])):
            if  path_list[i][j] == 'b':
                freq_spec_temp = freq_spec[0]
                freq_spec[0] = freq_spec[1]   
                freq_spec[1] = freq_spec_temp + (freq_spec[0] - freq_spec_temp)/2
            elif  path_list[i][j] == 'a':
                freq_spec[1] = freq_spec[1] + (freq_spec[0] - freq_spec[1])/2
        list_freq_spec.append(freq_spec)
    
    remove_index_list = []
    for i in range(len(path_list)):
        if freq1 >= list_freq_spec[i][0] - 10 and freq1 <= list_freq_spec[i][1] + 10:
            remove_index_list.append(i)
        elif freq2 >= list_freq_spec[i][0] - 10 and freq2 <= list_freq_spec[i][1] + 10:
            remove_index_list.append(i)
        elif freq3 >= list_freq_spec[i][0] - 10 and freq3 <= list_freq_spec[i][1] + 10:
            remove_index_list.append(i)
        elif freq4 >= list_freq_spec[i][0] - 10 and freq4 <= list_freq_spec[i][1] + 10:
            remove_index_list.append(i)
    for i in remove_index_list:
         del wavelet_packet[path_list[i]]
         del list_freq_spec[i]
    return wavelet_packet, list_freq_spec

max_level = pywt.dwt_max_level(len(x_1), "haar")

wavelet_packet = pywt.WaveletPacket(x_1,"haar")

level_decomposition = wavelet_packet.get_level(2)


#for i in wavelet_packet.get_leaf_nodes(False):
#    print(i.path,format_array(i.data))


for i in wavelet_packet.get_leaf_nodes(False):
    if (np.sum(np.abs(wavelet_packet[i.path].data)))**2 < 100:
        del wavelet_packet[i.path]
    else:
        print(i.path,format_array(i.data))

#for i in wavelet_packet.get_leaf_nodes():
#    print(i.path, format_array(i.data))
        
# =============================================================================
# synthesis of orignal signal with packets of most energy
# =============================================================================
print(wavelet_packet.reconstruct())