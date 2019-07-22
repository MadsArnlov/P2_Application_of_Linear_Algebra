# -*- coding: utf-8 -*-
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


def newton(f, df, x0, eps, solution):
    old = x0 + 1        # Ensure iteration starts
    count = 0
    error = []
    while abs(x0-old) > eps:
        old = x0
        x0 = old - f(old)/df(old)
        error.append(solution - x0)
        count += 1
    return x0, count, error


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
    s = x0 + 1
    count = 0
    while abs(s - solution) >= eps:
        s = g(s)
        count += 1
#    for k in range(1, N+1):
#        if abs(x[-1] - solution) >= eps:
#            x.append(g(x[k - 1]))
    return s, count


def f(x, h0, lamb):
    return h0 + lamb*np.cosh(x/lamb)

"Plot af kædelinjen"
x = np.linspace(-30, 30, 1000)

plt.figure(figsize=(16, 9))
plt.subplot(1, 2, 1)
plt.plot(x, f(x, 0, 10), 'k-')
plt.plot(x, f(x, 0, 17), 'b-')
plt.plot(x, f(x, 0, 25), 'r-')
plt.subplot(1, 2, 2)
plt.plot(x, f(x, 0, 17), 'k-')
plt.plot(x, f(x, 5, 17), 'b-')
plt.plot(x, f(x, 25, 17), 'r-')
plt.show()


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


a, b, eps, N = 180, 210, 1E-8, 200

solution = 189.94865405998324

m, count_B = bisect(f_4, a, b, eps)
x0, count_N, error_N = newton(f_4, df_4, a, eps, solution)
x1, count_S = secant(f_4, a, b, eps)
x2, count_F = f_iter(g2, a, solution, eps, 0)

fejl = [abs(m - solution), abs(x0 - solution), abs(x1 - solution), abs(x2 - solution)]

iterations = [count_B, count_N, count_S, count_F]

orden_N = [error_N[i-1]/error_N[i] for i in range(1, count_N-1)]


print("Bisektionsmetoden:", "\n", "Approksimation: {}".format(m), "\n", "Fejl: {}".format(fejl[0]), "\n", "Iterationer: {}".format(iterations[0]))
print("Funktionsiterationsmetoden:", "\n", "Approksimation: {}".format(x2), "\n", "Fejl: {}".format(fejl[3]), "\n", "Iterationer: {}".format(iterations[3]))
print("Newtons metode:", "\n", "Approksimation: {}".format(x0), "\n", "Fejl: {}".format(fejl[1]), "\n", "Iterationer: {}".format(iterations[1]))
print("Sekant metoden:", "\n", "Approksimation: {}".format(x1), "\n", "Fejl: {}".format(fejl[2]), "\n", "Iterationer: {}".format(iterations[2]))


#print('-'*34)
#print('             Orden            ')
#print('-'*34)
#print('  N       Fejl       Orden')
#print('-'*34)
#print("{}      {:.4f}          ".format(1, error_N[0]))
#for i in range(1, count_N):
#    print("{}      {:.4f}        {}".format(i, error_N[i], orden_N[i]))


#x = np.linspace(10, b, 500)
#plt.plot(x, f_4(x))


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
##plt.savefig("diff.pdf")
#plt.show()
