# -*- coding: utf-8 -*-

# quadrature
# midpoint rule
# composite formula

import numpy as np

def midpointrule(f,a,b,N):
    h = (b - a)/N
    xL = a - h
    xR = a
    I = 0
    for k in range(N):
        xL = xL + h
        xR = xR + h
        I = I + h*f((xL+xR)/2.0)
    return I
    
def fun2(x):
    return (x - np.sqrt(2))**2


def fun1(x):
    return -(x - np.sqrt(2))**2
  
def fun(x):
    if x < np.sqrt(2):
        y = -(x - np.sqrt(2))**2
    else:
        y = (x - np.sqrt(2))**2
    return y    

Iexact = 15 - ((31 * np.sqrt(2)) / 3)
    
iter_max = 15

L0 = [0.0 for i in range(iter_max)]
L1 = [0.0 for i in range(iter_max)]
L2 = [0.0 for i in range(iter_max)]
L3 = [0.0 for i in range(iter_max)]
a = 0.0
c = np.sqrt(2)
b = 3.0


for k in range(iter_max):
    N = 2**(k+1)
    L0[k] = N
    J = midpointrule(fun, a, b, N)
    L1[k] = J
    L2[k] = abs(J - Iexact)
    if k>0:
        L3[k] = np.log2(L2[k-1]/L2[k])
    
    

L = zip(L0,L1,L2,L3)

header_fmt='{0:^7} {1:^16} {2:^9} {3:^8}'
print(header_fmt.format('N', 'Approksimation', 'Fejl', 'Orden'))
print(header_fmt.format('-'*7, '-'*16, '-'*9, '-'*8))

for n, Z, fejl, ord in L:
    print('{0:>7} {1:<11.10E} {2:4.3E} {3:4.3E}'.format(n, Z, fejl, ord))

