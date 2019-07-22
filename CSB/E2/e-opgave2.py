# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


a=np.linspace(0, 3, 1000)

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
plt.plot(a, f(a), "k-")
plt.show()