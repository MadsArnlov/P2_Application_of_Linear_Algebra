# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:32:50 2019

@author: arnlambo
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x, h0, lamb):
    return h0 + lamb*np.cosh(x/lamb)


h0 = [0, 5, 25]
lamb = [10, 17, 25]

x = np.linspace(-30, 30, 2000)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(x, f(x, h0[0], lamb[0]), 'k-', label="$h_0 = {:.2f},\lambda = {:.2f}$".format(h0[0], lamb[0]))
plt.plot(x, f(x, h0[0], lamb[1]), 'b-', label="$h_0 = {:.2f},\lambda = {:.2f}$".format(h0[0], lamb[1]))
plt.plot(x, f(x, h0[0], lamb[2]), 'r-', label="$h_0 = {:.2f},\lambda = {:.2f}$".format(h0[0], lamb[2]))
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x, f(x, h0[0], lamb[1]), 'k-', label="$h_0 = {:.2f},\lambda = {:.2f}$".format(h0[0], lamb[1]))
plt.plot(x, f(x, h0[1], lamb[1]), 'b-', label="$h_0 = {:.2f},\lambda = {:.2f}$".format(h0[1], lamb[1]))
plt.plot(x, f(x, h0[2], lamb[1]), 'r-', label="$h_0 = {:.2f},\lambda = {:.2f}$".format(h0[2], lamb[1]))
plt.grid()
plt.legend()
plt.savefig("kædelinie.pdf")
plt.show()
