# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 09:42:00 2019

@author: arnlo
"""

import numpy as np
import matplotlib.pyplot as plt


def Wa(s, n, transform='haar'):
    for i in range(n):
        s = Ta(s, j - i, transform)
    return s


def Ta(S, j, transform='haar'):
    transform = transform.upper()
    if transform == 'HAAR':
        for i in list(range(0, 2**(j), 2)):
            S[i + 1] = predict(S[i] + S[i + 1])
            S[i] = S[i] - S[i + 1]
        S[range(2**(j))] = S[list(range(1, 2**(j) + 1, 2)) +
                             list(range(0, 2**(j), 2))]
        return S
    if transform == 'CDF':
        s = np.zeros(len(S)//2)
        d = np.zeros_like(s)
        K = 4/np.sqrt(2)
        N = len(S)
        s[0] = S[0] - S[1]
        for n in range(5):
            s[n+1] = \
            S[2*(n+1)]
            -(S[2*(n+1)-1]
            +S[2*(n+1)+1])/4
            d[n] = S[2*n+1] - s[n] - s[n+1]
        s[0] += d[0]/2
        s[1] -= (-3*d[0]-3*d[1])/16
        s[2] -= (5*d[0]-29*d[1]-29*d[2]+5*d[3])/128
        for n in range(3, N//2-3):
            s[n+3] = S[2*(n+3)] - (S[2*(n+3)-1] + S[2*(n+3)+1])/4
            d[n+2] = S[2* (n+2) +1] - s[n+2] - s[n+3]
            s[n] += (35*d[n-3]-265*d[n-2]+998*d[n-1]+998*d[n]-265*d[n+1]+35*d[n+2])/4096
            s[n-3] *= K
            d[n-3] /= K
        N = N//2
        d[N-1] = S[2*N-1] - s[N-1]
        s[N-3] += (35*d[N-6]-265*d[N-5]+998*d[N-4]+998*d[N-3]-265*d[N-2]+35*d[N-1])/4096
        s[N-2] -= (5*d[N-4]-29*d[N-3]-29*d[N-2]+5*d[N-1])/128
        s[N-1] -= (-3*d[N-2]-3*d[N-1])/16
        for n in range(6, 0, -1):
            s[N-n] *= K
            d[N-n] /= K
        return np.hstack((s, d))


def Ws(s, n, transform='haar'):
    for i in range(1, n + 1):
        s = Ts(s, i, transform)
    return s


def Ts(S, j, transform='haar'):
    transform = transform.upper()
    if transform == 'HAAR':
        S[list(range(1, 2**(j) + 1, 2)) + list(range(0, 2**(j), 2))] = S[range(2**(j))]
        for i in list(range(0, 2**(j), 2)):
            S[i] = S[i] + S[i + 1]
            S[i + 1] = 2*S[i + 1] - S[i]
            return S
    if transform == 'CDF':
        s = np.zeros(len(S)//2)
        d = np.zeros_like(s)
        K = 4/np.sqrt(2)
        N = len(S)
        for k in range(6):
            d[N//2-k] *= K
            s[N//2-k] /= K
        s[N//2] = s[N//2] + 1/16*(-3*d[N//2-1]-3*d[N/2])
        s[N//2-1] = s[N/2-1]+1/128*(5*d[N//2-3]-29*d[N//2-2]
        -29*d[N//2-1]+5*d[N//2])
        s[N//2-2] = s[N//2-2]+1/4096*(-35*d[N//2-5]+265*d[N//2-4]
        -998*d[N//2-3]-998*d[N//2-2]+265*d[N//2-1]-35*d[N//2])
        S[N] = d[N//2]+s[N//2]
        for n in range(N/2-3, 4, -1):
            d[n-3] *=  K
            s[n-3] /= K
            s[n] = s[n]+1/4096*(-35*d[n-3]+265*d[n-2]-998*d[n-1]
            -998*d[n]+265*d[n+1]-35*d[n+2])
            S[2*(n+2)] = d[n+2]+s[n+2]+s[n+3]
            S[2*(n+3)-1] = s[n+3]+1/4*(S[2*(n+3)-2]+S[2*(n+3)])
        s[3] = s[3]+1/128*(5*d[1]-29*d[2]-29*d[3]+5*d[4])
        s[2] = s[2]+1/16*(-3*d[1]-3*d[2])
        s[1] = s[1]-1/2*d[1]
        for n in range(5, 1, -1):
            S[2*n] = d[n]+s[n]+s[n+1]
            S[2*(n+1)-1] = s[n+1]+1/4*(S[2*(n+1)-2]+S[2*(n+1)])
        S[1] = s[1]+S[2]
        return S


def predict(component):
    return component/2


def update(d, k):
    return d[k]/2


def multiresolution(s, n, transform='haar'):
    s = Wa(s, n, transform)
    plt.figure(figsize=(12, 12))
    for sequence in range(n + 1):
        s_plot = np.hstack((np.zeros(int(2**(n - sequence - 1))),
                            s[int(2**(n - sequence - 1)):]))
        s_plot = Ws(s_plot, n)
        plt.subplot(n + 1, 1, sequence + 1)
        plt.plot(s_plot)
    plt.show()


def zeroPadding(s):
    if np.log2(len(s)) - int(np.log2(len(s))) != 0.0:
        s = np.hstack((s, np.zeros(2**(int(np.log2(len(s))) + 1) - len(s))))
    return s


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


np.random.seed(6)
s1, t1 = dataGenerator(64, 57, 64)
t1 = np.arange(256)
s2, t2 = dataGenerator(64, 0, 4)
s1 = np.hstack((s1, s1, s1, s1))
s2 = np.hstack((s2, s2, s2, s2))
"Deviation"
#a = np.random.randint(0, 64*2)
#b = np.random.randint(0, 64*2)
#s1[a:] *= 2
#s2[b:] *= 2
#print(a, b)
#b = 1
#for i in range(50, len(s1)):
#    s1[i] += b
#    s2[i] += b
#    b += 1

"Plot of the generated signal in time"
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t1, abs(s1), 'b.')
plt.subplot(2, 1, 2)
plt.plot(t1, abs(s2), 'r.')
plt.show()

"Multiresolution plot of the signal"
s1 = zeroPadding(abs(s1))
j = int(np.log2(len(s1)))
multiresolution(s1, j, 'haar')

s2 = zeroPadding(abs(s2))
multiresolution(s2, j, 'haar')

