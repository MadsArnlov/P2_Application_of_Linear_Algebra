# -*- coding: utf-8 -*-

# solving an autonomous syste of ODEs
# using the Runge-Kutta 4 method
# the predator-prey model


import math
import numpy as np
import matplotlib.pyplot as plt

# one step RK4 adapted to system
def rk4(f, t, x, h):
    k1 = f(t, x)
    k2 = f(t + 0.5*h, x + 0.5*h*k1)
    k3 = f(t + 0.5*h, x + 0.5*h*k2)
    k4 = f(t + h, x + h*k3)
    xp = x + h*(k1 + 2.0*(k2 + k3) + k4)/6.0
    return xp, t + h

def Euler(f, t, x, h):
    xp = x + h*f(t, x)
    return xp, t + h

# time parameters
t_start = 0.0
t_stop = 500
N = 500
t_step = (t_stop - t_start)/float(N)

# equation parameters
alpha = 0.7
beta = 0.005
gamma = 0.2
delta = 0.001
mu = 0.003

# initial conditions
x10 = 300
x20 = 60

# The system itself
def fun(t, x):
    return np.array([alpha*x[0] - beta*x[0]*x[1],
                     -gamma*x[1] + delta*x[0]*x[1]])

def fun2(t, x):
    return np.array([alpha*x[0] - beta*x[0]*x[1] - mu*x[0]**2,
                     -gamma*x[1] + delta*x[0]*x[1]])


X1 = np.zeros(N + 1)
X2 = np.zeros(N + 1)

X1[0] = x10
X2[0] = x20

X3 = np.zeros(N + 1)
X4 = np.zeros(N + 1)

X3[0] = x10
X4[0] = x20

# Time variable
t = 0.0
t1 = t
for k in range(N):
    Xp1, t = rk4(fun2, t, np.array([X1[k], X2[k]]), t_step)
    Xp2, t1 = Euler(fun2, t1, np.array([X1[k], X2[k]]), t_step)
    X1[k+1] = Xp1[0]
    X2[k+1] = Xp1[1]
    X3[k+1] = Xp2[0]
    X4[k+1] = Xp2[1]

plt.figure(figsize=(16, 9))
plt.subplot(1, 1, 1)
plt.plot(X1, X2, 'r-', [x10], [x20], 'k.')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
#plt.subplot(2, 1, 2)
#plt.plot(X3, X4, 'b-', [x10], [x20], 'k.')
#plt.xlabel('$x_1$')
#plt.ylabel('$x_2$')
plt.show()
