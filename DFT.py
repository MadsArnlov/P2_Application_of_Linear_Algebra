# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:37:12 2019

@author: lasse
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
start = time.time()

def data_generator(J = 18, freq1 = 0, freq2 = 0, freq3 = 0, freq4 = 0, phase1 = 0, 
                   phase2 = 0, phase3 = 0, phase4 = 0, imp_freq = 0):
    N = 2**J
    t = np.arange(1 , N+1)
    A = 2 * np.pi * t / N
    x1 = np.sin(A * freq1 + phase1)
    x2 = np.sin(A * freq2 + phase2)
    x3 = np.sin(A * freq3 + phase3)
    x4 = np.sin(A * freq4 + phase4)
    x_imp = np.zeros(N)
    if imp_freq != 0:
        for i in range(int(N/imp_freq), len(x_imp), int(N/(imp_freq+1))):
            x_imp[i] = 1
            x_imp[i+1] = -1
    x_sum = x1 + x2 + x3 + x4 + x_imp
    return x_sum, N

def fft(x_sum, N):
    t = np.arange(N)
    
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    plt.plot(t, x_sum, 'r.')
#    plt.subplot(2, 1, 2)
#    plt.plot(x_fft, 'b.')
#    plt.show()

x_sum, N = data_generator(18, 30, 100)
fft(x_sum, N)

end = time.time()