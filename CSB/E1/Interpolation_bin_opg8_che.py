# -*- coding: utf-8 -*-

import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt

# Interpolating polynomial
def lagrange(x, x_values, y_values): 
    
    from functools import reduce
    # Lagrange basis polynomials
    def l(k,x):
        temp = [(x-x_values[j])/(x_values[k]-x_values[j])\
              for j in range(len(x_values)) if j != k]
        result = reduce(lambda x, y: x*y, temp)
        return result
    
    # Lagrange interpolation polynomial
    p = []
    for i in range(len(x)):
        temp = [y_values[k]*l(k,x[i]) for k in range(len(x_values))]
        p.append(sum(temp))
    
    return p

# Parameters for the experiment
a = -2
b = 3

def f(x):
    return x**2-sin(10*x)

# Error bound for equidistant points for the function f(x)=x^2-sin(10x)
def equi_bound(h, N):
    M = 10**(N+1)
    return 0.25*h**(N+1)*M

## Print the results
#print('-'*34)
#print('      Lagrange interpolation ')
#print('-'*34)
#print('  N     Error bound       Error')
#print('-'*34)
#for N in range(10,101,10):
#    
#    # Parameters for Lagrange interpolation
#    h = abs(b-a)/N
#    x_values = [a+k*h for k in range(N+1)] 
#    y_values = [f(x_values[k]) for k in range(N+1)]
#   
#    # New points for calculating the error max|f(x)-p_N(x)|
#    N_test =  797
#    h_test = abs(b-a)/N_test
#    x_test = [3*(1 - cos(k*pi/N))/2 for k in range(N_test+1)]
#     
#    # Calculate the approximation and the solution
#    approx = lagrange(x_test, x_values, y_values)
#    y_test = [f(x_test[k]) for k in range(N_test+1)]
#    
#    # Calculate the error
#    from operator import sub
#    temp1 = list(map(sub, y_test, approx))
#    temp2 = list(map(abs, temp1))
#    error = max(temp2)
#    
#    # Print a table
#    print('{:3d}  {:14.5E}  {:13.5E}'.format(N, \
#          equi_bound(h,N), error))
# =============================================================================
# Opgave 8
# =============================================================================
N = [5, 15, 100]
plt.figure(figsize=(13, 11))
for i in range(len(N)):
    #Chebyshev
    h_che = np.arange(0, N[i]+1)
    x_values_che = ((b-a)*(1-np.cos(h_che*np.pi/N[i]))/2)+a
    y_values_che = [f(x_values_che[k]) for k in range(N[i]+1)]
    
    h = abs(b-a)/N[i]
    x_values = [a+k*h for k in range(N[i]+1)] 
    y_values = [f(x_values[k]) for k in range(N[i]+1)]
    N_test =  797
    h_test = abs(b-a)/N_test
    x_test = [a+k*h_test for k in range(N_test+1)]
    approx = lagrange(x_test, x_values_che, y_values_che)
    f_values = [f(x_test[k]) for k in range(N_test+1)]
        
    plt.subplot(3,1,1+i)
    plt.plot(x_values_che, y_values_che, 'kd', label="Knudepunkter")
    plt.plot(x_test, approx, 'b--', label="$p_N(x)$")
    plt.plot(x_test, f_values, 'r-', label="$f(x)$")
    plt.legend()
    plt.grid()
    plt.title("Plot af funktionen og lagrange polynomiumet med N = {:d}".format(N[i]))
plt.savefig("lagrange_plot_che.pdf")
plt.show()

# =============================================================================
# Opgave 8 fejl
# =============================================================================
for N in range(5, 30, 5):
    #Chebyshev parameters for Lagrange polynomial
    h_che = np.arange(0, N+1)
    x_values_che = 3*(1-np.cos(h_che*np.pi/N))/2
    y_values_che = [f(x_values_che[k]) for k in range(N+1)]
   
    # New points for calculating the error max|f(x)-p_N(x)|
    N_test =  797
    h_test = abs(b-a)/N_test
    x_test = [a+k*h_test for k in range(N_test+1)]
     
    # Calculate the approximation and the solution
    approx = lagrange(x_test, x_values_che, y_values_che)
    y_test = [f(x_test[k]) for k in range(N_test+1)]
    
    # Calculate the error
    from operator import sub
    temp1 = list(map(sub, y_test, approx))
    temp2 = list(map(abs, temp1))
    error = max(temp2)
    
    # Print a table
    print('{:3d}  {:14.5E}  {:13.5E}'.format(N, \
          equi_bound(h,N), error))

