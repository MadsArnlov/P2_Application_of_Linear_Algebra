# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:32:50 2019

@author: arnlambo
"""

import numpy as np
import matplotlib.pyplot as plt


def bisect(fcn, a, b, eps):
    """
    Bisection method for solution of the equation `fcn(x)=0` to an
    accuracy `eps`, in the interval [a,b].
    """
    if fcn(a) * fcn(b) > 0:
        print('Two endpoints have same sign')
        return

    count = 0
    while abs(b - a) >= eps:
        m = (a + b) / 2
        if fcn(a) * fcn(m) <= 0:
            b = m
        else:
            a = m
        count += 1
    return m, count


def newton(f, df, x0, eps):
    old = x0 + 1        # Ensure iteration starts
    count = 0
    while abs(x0-old) > eps:
        old = x0
        x0 = old - f(old)/df(old)
        count += 1
    return x0, count


def secant(f, x0, x1, eps):
    count = 0
    while abs(x0 - x1) > eps:
        x_temp = x1-(f(x1)*(x1-x0))/(f(x1)-f(x0))
        x0 = x1
        x1 = x_temp
        count += 1
    return x1, count


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
