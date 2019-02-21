# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 09:42:00 2019

@author: arnlo
"""

import numpy as np


def Wa(s, n):
    for i in range(n):
        s = Ta(s, j - i)
    return s


def Ta(s, j):
    for i in list(range(0, 2**(j), 2)):
        s[i + 1] = (s[i] + s[i + 1])/2
        s[i] = s[i] - s[i + 1]
    s[range(2**(j))] = s[list(range(1, 2**(j) + 1, 2)) +
                         list(range(0, 2**(j), 2))]
    return s


def Ws(s, n):
    for i in range(1, n + 1):
        s = Ts(s, i)
    return s


def Ts(s, j):
    s[list(range(1, 2**(j) + 1, 2)) + list(range(0, 2**(j), 2))] = s[range(2**(j))]
    for i in list(range(0, 2**(j), 2)):
        s[i] = s[i] + s[i + 1]
        s[i + 1] = 2*s[i + 1] - s[i]
    return s


def predict(s):
    return s


def update(d, k):
    return d[k]/2


def zeroPadding(s):
    if np.log2(len(s)) - int(np.log2(len(s))) != 0.0:
        s = np.hstack((s, np.zeros(2**(int(np.log2(len(s))) + 1) - len(s))))
    return s

sj = np.array([56, 40, 8, 24, 48, 48, 40, 16])
sj = zeroPadding(sj)
j = int(np.log2(len(sj)))
sj = Wa(sj, 3)
sj = Ws(sj, 3)

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
