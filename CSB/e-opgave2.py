# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:44:26 2019

@author: bergl
"""


import numpy as np
import matplotlib.pyplot as plt


a=np.arange(-100,100)

def f(x):
    """
    input: array
    """
    y=np.zeros(len(x))
    for i in range(0,len(x)):
        if x[i] <= np.sqrt(2):
            y[i]=-(x[i]-np.sqrt(2))**2
        else:
            y[i]=(x[i]-np.sqrt(2))**2
    return y

plt.figure(figsize=(14,9))
plt.plot(f(a),a, "k-")
plt.show()