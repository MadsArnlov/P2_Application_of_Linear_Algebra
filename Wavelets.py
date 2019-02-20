# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 09:42:00 2019

@author: arnlo
"""

import numpy as np


def liftingsteps(s, j, n):
    for i in range(n):
        s = lifting(s, j - i)
    return s


def lifting(s, j):
    for i in list(range(0, 2**(j), 2)):
        s[i + 1] = (s[i] + s[i + 1])/2
        s[i] = s[i] - s[i + 1]
    s[range(2**(j))] = s[list(range(1, 2**(j) + 1, 2)) +
      list(range(0, 2**(j), 2))]
    return s


def predict(s):
    return s


def update(d, k):
    return d[k]/2


sj = np.array([56, 40, 8, 24, 48, 48, 40, 16])
j = int(np.log2(len(sj)))
sj = liftingsteps(sj, j, 3)

# =============================================================================
# Tried to implement code from ``Ripples in Mathematics''
# =============================================================================
"""
Signal = np.array([56, 40, 8, 24, 48, 48, 40, 16])

def dwtHaar(s):
    N = len(s)
    s_new = np.zeros(N//2)
    d_new = np.copy(s_new)
    for n in range(N//2):
        s_new[n] = 1/2 * (s[2*n] + s[2*n + 1])
        d_new[n] = s[2*n] - s_new[n]
    return np.hstack((s_new, d_new))
print(dwtHaar(Signal))

def w_decomp(s):
    j = int(np.log2(len(s)))
    T = np.zeros((j, len(s)))
    T[0,:] = s
    for n in range(j):
        L = 2**(j - n - 1) - 1
        print(L)
        T[j, 0:L] = dwtHaar(T[j, 1:L])
        T[j, L: len(s)] = T[j, L: len(s)]
    return T

print(w_decomp(Signal))
"""
