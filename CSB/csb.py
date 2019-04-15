# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:32:50 2019

@author: arnlambo
"""

import numpy as np
import matplotlib.pyplot as plt


def bisect(f, a, b, eps):
    """
    Bisection method for solution of the equation `fcn(x)=0` to an
    accuracy `eps`, in the interval [a,b].
    """
    if f(a) * f(b) > 0:
        print('Two endpoints have same sign')
        return

    count = 0
    while abs(b - a) >= eps:
        m = (a + b) / 2
        if f(a) * f(m) <= 0:
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


def f_iter(g, x0, solution, eps, N):
    x = [x0]
    for k in range(1, N+1):
        if abs(x[-1] - solution) >= eps:
            x.append(g(x[k - 1]))
    return x


def f(x, h0, lamb):
    return h0 + lamb*np.cosh(x/lamb)


def f_4(lamb):
    return lamb*np.cosh(75/lamb) - lamb - 15


def df_4(lamb):
    return np.cosh(75/lamb) - 75*np.sinh(75/lamb)/lamb - 1


def g1(lamb):
    return (lamb+15)/np.cosh(75/lamb)


def g2(lamb):
    return lamb*np.cosh(75/lamb) - 15


a, b, eps, N = 180, 210, 1E-2, 100

m, count_B = bisect(f_4, a, b, eps)
x0, count_N = newton(f_4, df_4, a, eps)
x1, count_S = secant(f_4, a, b, eps)

solution = 189.94865405998303

lamb_test = f_iter(g2, a, solution, eps, N)


#h0 = [0, 5, 25]
#lamb = [10, 17, 25]

#lamb = np.linspace(180, 210, 10000)
##
#plt.figure(figsize=(14, 6))
#plt.plot(lamb,f_4(lamb))
##plt.subplot(1, 2, 1)
##plt.plot(x, f(x, h0[0], lamb[0]), 'k-', label="$h_0 = {:.2f},\lambda = {:.2f}$".format(h0[0], lamb[0]))
##plt.plot(x, f(x, h0[0], lamb[1]), 'b-', label="$h_0 = {:.2f},\lambda = {:.2f}$".format(h0[0], lamb[1]))
##plt.plot(x, f(x, h0[0], lamb[2]), 'r-', label="$h_0 = {:.2f},\lambda = {:.2f}$".format(h0[0], lamb[2]))
##plt.grid()
##plt.legend()
##plt.subplot(1, 2, 2)
##plt.plot(x, f(x, h0[0], lamb[1]), 'k-', label="$h_0 = {:.2f},\lambda = {:.2f}$".format(h0[0], lamb[1]))
##plt.plot(x, f(x, h0[1], lamb[1]), 'b-', label="$h_0 = {:.2f},\lambda = {:.2f}$".format(h0[1], lamb[1]))
##plt.plot(x, f(x, h0[2], lamb[1]), 'r-', label="$h_0 = {:.2f},\lambda = {:.2f}$".format(h0[2], lamb[1]))
#plt.grid()
##plt.legend()
#plt.savefig("funktion.pdf")
#plt.show()

