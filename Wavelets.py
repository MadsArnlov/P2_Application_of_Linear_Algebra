# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 09:42:00 2019

@author: arnlo
"""

import numpy as np
import matplotlib.pyplot as plt


def Wa(s, n):
    for i in range(n):
        s = Ta(s, j - i)
    return s


def Ta(s, j):
    for i in list(range(0, 2**(j), 2)):
        s[i + 1] = predict(s[i] + s[i + 1])
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


def predict(component):
    return component/2


def update(d, k):
    return d[k]/2


def zeroPadding(s):
    if np.log2(len(s)) - int(np.log2(len(s))) != 0.0:
        s = np.hstack((s, np.zeros(2**(int(np.log2(len(s))) + 1) - len(s))))
    return s


def multiresolution(s, n):
    s = Wa(s, n)
    plt.figure(figsize=(12, 8))
    for sequence in range(n + 1):
        s_plot = np.hstack((np.zeros(int(2**(n - sequence - 1))),
                            s[int(2**(n - sequence - 1)):]))
        s_plot = Ws(s_plot, n)
        plt.subplot(n + 1, 1, sequence + 1)
        plt.plot(s_plot)
    plt.show()


def testFunction(x):
    value = np.log(2 + np.sin(3*np.pi*np.sqrt(x)))
    for k in range(1, len(value) + 1):
        if k % 32 == 1:
            value[k - 1] = value[k - 1] + 2
    return value


sj = testFunction(np.linspace(0, 1, 2**10))

#sj = np.array([56, 40, 8, 24, 48, 48, 40, 16])
#sj = zeroPadding(sj)
#j = int(np.log2(len(sj)))
#multiresolution(sj, j)

np.random.seed(5)


def dataGenerator(l=32, r1=10, r2=13):
    """
    Returns a synthetic signal in time representing data.

    Parameters
    ----------
    l:  int
        The length of the signal.
    r1: int
        Start value for range of non-zero values in frequency signal.
    r2: int
        End value for range of non-zero values in frequency signal.
    """
    t = np.arange(l)
    n = np.zeros((l,), dtype=complex)
    n[r1:r2] = np.exp(1j*np.random.uniform(0, 2*np.pi, (r2-r1,)))
    return np.fft.ifft(n), t

s, t = dataGenerator(32, 10, 13)
"Deviation"
s[25] += 0.1
#b = 1
#for i in range(60, 100):
#    s[i] /= np.math.factorial(b)
#    b += 1

"Plot of the generated signal in time"
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, abs(s), 'b.')
plt.subplot(3, 1, 2)
plt.plot(t, s)
plt.subplot(3, 1, 3)
plt.plot(t, s.real, 'b-', t, s.imag, 'r--')
plt.legend(('real', 'imaginary'))
plt.show()

"Multiresolution plot of the signal"
sj = zeroPadding(abs(s))
j = int(np.log2(len(sj)))
multiresolution(sj, j)

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
