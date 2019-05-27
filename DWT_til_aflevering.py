# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:11:16 2019

@author: bergl
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path
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
data_folder = Path("Test_recordings\\With_noise\\impuls300pr.min_speaker2")
file_to_open = [data_folder / "Test_recording microphone{:d}_impuls_speaker2.wav".format(i) for i in range(1,4)]

sampling_frequency, data1 = wavfile.read(file_to_open[0])
sampling_frequency, data2 = wavfile.read(file_to_open[1])
sampling_frequency, data3 = wavfile.read(file_to_open[2])

#data_s = sampling_frequency * 21         # start value for data interval'
data_s=800000
data_e = data_s + 2**19                 # end value for data interval

x = [data1[data_s:data_e], data2[data_s:data_e], data3[data_s:data_e]]
x_1 = np.array([1,2,3,4,5,6,7,8])

x[0] = x[0]/scipy.std(x[0])
x[1] = x[1]/scipy.std(x[1])
x[2] = x[2]/scipy.std(x[2])

# =============================================================================
# Cross correlation
# =============================================================================
def cross_corr(signal1, signal2):
    """
        Calculates time delay    
    
        Parameters:
            signal1,signal2
    
        Returns: 
            Delay between the two signal in samples and a plot 
            containing the two signals and their cross-correlation
    """
    plt.figure(figsize=(14, 5))
    plt.subplot(2, 2, 1)
    plt.plot(signal1, 'k,')
    plt.subplot(2, 2, 2)
    plt.plot(signal2, 'k,')
    plt.show()
    
    correlation = np.correlate(signal1, signal2, 'full')
    plt.figure(figsize=(14, 4))
    plt.title("Cross Correlation", fontsize=18)
    plt.plot(correlation, 'g', np.argmax(correlation), max(correlation), 'kx')
    plt.show()
    print("Signal 2 is shifted in time with", len(signal1) - (np.argmax(correlation) + 1), "samples")
    return len(signal1) - (np.argmax(correlation) + 1), correlation


# =============================================================================
# Choosing Packets
# =============================================================================
def packets_selection(signal, level, safe_zone = 1000, threshold=0, freq1=737, freq2=240, freq3=1000, freq4=125, filters="db16"):
    """
        Function for selecting the packets from which to synthesize the signal
        
        parameters:
            signal, level, safe_zone = 1000, threshold=0, freq1=737, freq2=240, freq3=1000, freq4=125, filters="db16"
            
        Returns:
            Array with all packets satisfying the criteria
    """    
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
            if  path_list[i][j] == 'd':
                freq_spec_temp = freq_spec[0]
                freq_spec[0] = freq_spec[1]   
                freq_spec[1] = freq_spec_temp + (freq_spec[0] - freq_spec_temp)/2
            elif  path_list[i][j] == 'a':
                freq_spec[1] = freq_spec[1] + (freq_spec[0] - freq_spec[1])/2
        list_freq_spec.append(freq_spec)
    
    
    remove_index_list = []
    for i in range(len(path_list)):
        if freq1 >= list_freq_spec[i][0] - safe_zone  and freq1 <= list_freq_spec[i][1] + safe_zone:
            remove_index_list.append(i)
        elif freq2 >= list_freq_spec[i][0] - safe_zone  and freq2 <= list_freq_spec[i][1] + safe_zone:
            remove_index_list.append(i)
        elif freq3 >= list_freq_spec[i][0] - safe_zone and freq3 <= list_freq_spec[i][1] + safe_zone:
            remove_index_list.append(i)
        elif freq4 >= list_freq_spec[i][0] - safe_zone   and freq4 <= list_freq_spec[i][1] + safe_zone:
            remove_index_list.append(i)
        elif freq1 <= list_freq_spec[i][0] + safe_zone  and freq1 >= list_freq_spec[i][1] - safe_zone:
            remove_index_list.append(i)
        elif freq2 <= list_freq_spec[i][0] + safe_zone  and freq2 >= list_freq_spec[i][1] - safe_zone:
            remove_index_list.append(i)
        elif freq3 <= list_freq_spec[i][0] + safe_zone  and freq3 >= list_freq_spec[i][1] - safe_zone:
            remove_index_list.append(i)
        elif freq4 <= list_freq_spec[i][0] + safe_zone  and freq4 >= list_freq_spec[i][1] - safe_zone:
            remove_index_list.append(i)
    for i in remove_index_list:
        level_decomposition[i].data = np.zeros(len(level_decomposition[i].data))    
    

    usefull_path_list=[]
    for i in level_decomposition:
        if i.data.any() != 0:
            usefull_path_list.append(i.path)
            
    print(wp.reconstruct())
    synthesis = wp.reconstruct()
    return synthesis, usefull_path_list



# =============================================================================
# Synthesis
# =============================================================================
def reconstruct_from_packet(signal, level, path_list, filters="db16"):
    """
        Function to reconstruct signal from multiple predefined packets
        
        Aguments:
            Signal, level, path_list, filters="db16"
            
        Returns:
            synthesized signal from predefined packets
    
    """
    wp = pywt.WaveletPacket(signal, filters)
    level_decomposition = wp.get_level(level)
    
    decomposition_list=[]
    for i in level_decomposition:
        decomposition_list.append(i.path)
        if i.path in path_list: 
            "Do Nothing"
        else:
            wp[i.path].data = np.zeros(len(wp[i.path].data))
    synthesis = wp.reconstruct()
    return synthesis


# =============================================================================
# calling packets_selection and Reconstruct_from_packet
# =============================================================================
synthesis1, usefull_path_list = packets_selection(x[1], 10)
synthesis2 = reconstruct_from_packet(x[0], 10, usefull_path_list)
synthesis3 = reconstruct_from_packet(x[2], 10, usefull_path_list)


# =============================================================================
# calling Cross_corr
# =============================================================================
sample_delay_1_2, correlation1 = cross_corr(synthesis1[300000:400000]/scipy.std(synthesis1), synthesis2[300000:400000]/scipy.std(synthesis2))
sample_delay_1_3, correlation2 = cross_corr(synthesis1[300000:400000]/scipy.std(synthesis1), synthesis3[300000:400000]/scipy.std(synthesis3))
sample_delay_2_3, correlation3 = cross_corr(synthesis2[300000:400000]/scipy.std(synthesis2), synthesis3[300000:400000]/scipy.std(synthesis3))


# =============================================================================
# Plot of Cross-correlations
# =============================================================================
plt.figure(figsize=(14, 10))
plt.subplot(311)
plt.plot(np.linspace(-99999, 100000, 199999), correlation1, 'g-', label = "$\~\mathbf{x}_{\u03B1}\star\~\mathbf{x}_{\u03B2}$")
plt.plot(np.argmax(correlation1)-99999, max(correlation1), 'kx')
plt.legend(loc="upper right", fontsize = 'x-large')


plt.subplot(312)
plt.plot(np.linspace(-99999, 100000, 199999), correlation2, 'g-', label = "$\~\mathbf{x}_{\u03B1}\star\~\mathbf{x}_{\u03B3}$")
plt.plot(np.argmax(correlation2)-99999, max(correlation2), 'kx')
plt.legend(loc='upper right', fontsize = 'x-large')


plt.subplot(313)
plt.plot(np.linspace(-99999, 100000, 199999), correlation3, 'g-', label = "$\~\mathbf{x}_{\u03B2}\star\~\mathbf{x}_{\u03B3}$")
plt.plot(np.argmax(correlation3)-99999, max(correlation3), 'kx')
plt.legend(loc="upper right", fontsize = 'x-large')
plt.xlabel("Samples")

plt.savefig('cross-correlation_experiment_7.png')
plt.show()


