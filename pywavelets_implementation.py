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
import scipy

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
data_folder = Path("C:\\Users\\bergl\\OneDrive\\Documents\\GitHub\\P2_Application_of_Linear_Algebra\\Test_recordings\\Without_noise\\impuls300pr.min_speaker1_uden_støj")
file_to_open = [data_folder / "Test_recording microphone{:d}_impuls_speaker1_uden_støj.wav".format(i) for i in range(1,4)]

sampling_frequency, data1 = wavfile.read(file_to_open[0])
sampling_frequency, data2 = wavfile.read(file_to_open[1])
sampling_frequency, data3 = wavfile.read(file_to_open[2])

data_s = sampling_frequency * 21         # start value for data interval
data_e = data_s + 2**19                  # end value for data interval

x = [data1[data_s:data_e], data2[data_s:data_e], data3[data_s:data_e]]
x_1 = np.array([1,2,3,4,5,6,7,8])

x[0] = x[0]/scipy.std(x[0])
x[1] = x[1]/scipy.std(x[1])
x[2] = x[2]/scipy.std(x[2])

# =============================================================================
# Cross correlation
# =============================================================================
def cross_corr(signal1, signal2):
    plt.figure(figsize=(14, 5))
    plt.subplot(2, 2, 1)
    plt.plot(signal1, 'r,')
    plt.subplot(2, 2, 2)
    plt.plot(signal2, 'b,')
    plt.show()
    signal1[:5000] = 0
    signal1[2**19-5000:] = 0
    
    correlation = np.correlate(signal1, signal2, 'full')
    plt.figure(figsize=(14, 4))
    plt.title("Cross Correlation", fontsize=18)
    plt.plot(correlation, 'g', np.argmax(correlation), max(correlation), 'kx')
    plt.show()
    print("Signal 2 is shifted in time with", len(signal1) - (np.argmax(correlation) + 1), "samples")
    return len(signal1) - (np.argmax(correlation) + 1)
# =============================================================================
# 



#levels=2
# 
# max_level = pywt.dwt_max_level(len(x_1),"haar")
# 
#wp = pywt.WaveletPacket(x[0],"haar")
#
#level_decomposition = wp.get_level(13)
##
#for i in wp.get_leaf_nodes(False):
#    print(i.path,format_array(i.data))
# 
# delete_list=[]
# 
# for i in wp.get_leaf_nodes(False):
#     if (np.sum((wp[i.path].data)**2)) < 10:
#        delete_list.append(i.path)
# 
# #wp["aa"]=[0,0]
# #for j in delete_list: 
# #    del wp[j]        
#     
# for i in wp.get_leaf_nodes():
#     print(i.path, format_array(i.data))
#   
# print(wp.reconstruct())

#wp = pywt.WaveletPacket(x_1, "haar")
#level_decomposition = wp.get_level(3)

#wp = pywt.WaveletPacket(x[0], "db16")
#level_decomposition = wp.get_level(8)
#decomposition_list=[]
#usefull_path_list=["aaaaaaaa"]
#for i in level_decomposition:
#    decomposition_list.append(i.path)
#    if i.path in usefull_path_list: 
#        "Do Nothing"
#    else:
#        wp[i.path].data = np.zeros(len(wp[i.path].data))
#print(wp["aaaaaaaa"].data)

# =============================================================================
def packets_selection(signal, level, threshold=0, freq1=737, freq2=240, freq3=1000, freq4=125, filters="db16"):
    
#    max_level = pywt.dwt_max_level(len(signal), filters)

    wp = pywt.WaveletPacket(signal, filters)
    level_decomposition = wp.get_level(level)
   
    path_list = []
    limit = (np.sum(signal**2)/(2**level))*threshold
    for i in range(2**level):
        if np.sum((level_decomposition[i].data)**2) < limit:
            level_decomposition[i].data = np.zeros(len(level_decomposition[i].data))
        else:
            path_list.append(level_decomposition[i].path)

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
        if freq1 >= list_freq_spec[i][0] + 10  and freq1 <= list_freq_spec[i][1] + 10:
            remove_index_list.append(i)
        elif freq2 >= list_freq_spec[i][0] + 10  and freq2 <= list_freq_spec[i][1] + 10:
            remove_index_list.append(i)
        elif freq3 >= list_freq_spec[i][0] + 10  and freq3 <= list_freq_spec[i][1] + 10:
            remove_index_list.append(i)
        elif freq4 >= list_freq_spec[i][0] + 10  and freq4 <= list_freq_spec[i][1] + 10:
            remove_index_list.append(i)
    for i in remove_index_list:
        level_decomposition[i].data = np.zeros(len(level_decomposition[i].data))    
    

    usefull_path_list=[]
    for i in level_decomposition:
        if i.data.any() != 0:
            print(i.path, type(i.path))
            usefull_path_list.append(i.path)
            
    print(wp.reconstruct())
    
    
    synthesis = wp.reconstruct()
    return synthesis, usefull_path_list


def reconstruct_from_packet(signal, level, path_list, filters="db16"):
    wp = pywt.WaveletPacket(signal, filters)
    level_decomposition = wp.get_level(level)
    
    decomposition_list=[]
    for i in level_decomposition:
        decomposition_list.append(i.path)
        if i.path in path_list: 
            "Do Nothing"
        else:
            wp[i.path].data = np.zeros(len(wp[i.path].data))
    print(len(decomposition_list))
    synthesis = wp.reconstruct()
    return synthesis
          
    
        
    
    
    
# =============================================================================
# possibility of hamming window
# =============================================================================
#x_h0 = np.array(hamming(x[0]))
#x_h1 = np.array(hamming(x[1]))
#x_h2 = np.array(hamming(x[2]))


#data_norm_by_std0 = x[0]/scipy.std(x[0])
#data_norm_by_std1 = x[1]/scipy.std(x[1])
#data_norm_by_std2 = x[2]/scipy.std(x[2])
synthesis1, usefull_path_list = packets_selection(x[0], 8)
synthesis2 = reconstruct_from_packet(x[1], 8, usefull_path_list)
synthesis3 = reconstruct_from_packet(x[2], 8, usefull_path_list)
cross_corr(synthesis1[300000:400000]/scipy.std(synthesis1), synthesis2[300000:400000]/scipy.std(synthesis2))

plt.figure(figsize=(14,7))
plt.subplot(4,1,1)
plt.plot(x[0])
plt.subplot(4,1,2)
plt.plot(synthesis1[100000:400000]/scipy.std(synthesis1))
plt.subplot(4,1,3)
plt.plot(synthesis2[100000:400000]/scipy.std(synthesis2))
plt.subplot(4,1,4)
plt.plot(synthesis3[100000:400000]/scipy.std(synthesis3))



