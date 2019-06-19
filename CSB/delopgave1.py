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
    return lamb*(np.cosh(75/lamb)) - lamb - 15


def df_4(lamb):
    return np.cosh(75/lamb) - 75*np.sinh(75/lamb)/lamb - 1


def g1(lamb):
    return (lamb+15)/np.cosh(75/lamb)


def dg1(lamb):
    return 1/np.cosh(75/lamb) + (75*(lamb+15)*np.sinh(75/lamb))/(np.cosh(75/lamb)**2 * lamb**2)


def g2(lamb):
    return lamb*np.cosh(75/lamb) - 15


def dg2(lamb):
    return np.cosh(75/lamb) - (75*np.sinh(75/lamb))/lamb


a, b, eps, N = 180, 210, 1E-8, 100

m, count_B = bisect(f_4, a, b, eps)
x0, count_N = newton(f_4, df_4, a, eps)
x1, count_S = secant(f_4, a, b, eps)

solution = 189.94865405998324

fejl = [abs(solution - m), abs(solution - x0), abs(solution - x1)]

iterations = [count_B, count_N, count_S]

print("Bisektionsmetoden:", "\n", "Fejl: {}".format(fejl[0]), "\n", "Iterationer: {}".format(iterations[0]))
print("Newtons metode:", "\n", "Fejl: {}".format(fejl[1]), "\n", "Iterationer: {}".format(iterations[1]))
print("Sekant metoden:", "\n", "Fejl: {}".format(fejl[2]), "\n", "Iterationer: {}".format(iterations[2]))



#lamb_test = f_iter(g1, a, solution, eps, N)
x = np.linspace(a, b, 500)
plt.plot(x, f_4(x))


#h0 = [0, 5, 25]
#lamb = [10, 17, 25]
#
#lamb = np.linspace(180, 210, 10000)
#
#plt.figure(figsize=(14, 6))
#plt.subplot(2, 2, 1)
#plt.plot(lamb, g1(lamb), 'r-', label='$g_1(\lambda)$')
#plt.grid()
#plt.legend()
#plt.subplot(2,2,2)
#plt.plot(lamb, g2(lamb), 'b-', label='$g_2(\lambda)$')
#plt.grid()
#plt.legend()
#plt.subplot(2,2,3)
#plt.plot(lamb, dg1(lamb), 'r-', label="$g_1'(\lambda)$")
#plt.grid()
#plt.legend()
#plt.subplot(2, 2, 4)
#plt.plot(lamb, dg2(lamb), 'b-', label="$g_2'(\lambda)$")
#plt.grid()
#plt.legend()
#plt.savefig("diff.pdf")
#plt.show()

