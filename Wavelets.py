# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 09:42:00 2019

@author: arnlo
"""

import numpy as np


def lifting(s, j):
    s_new = np.ones(2**(j-1))
    d_new = np.copy(s_new)
    for k in range(2**(j-1)):
        s_new[k] = (s[2*k] + s[2*k + 1])/2
        d_new[k] = s_new[k] - s[2*k + 1]
    s = np.hstack((s_new, d_new))
    return s

def liftingsteps():
    


j = 3

sj = np.array([56, 40, 8, 24, 48, 48, 40, 16])


#sj_1, dj_1 = lifting(sj, j-1)
#sj_2, dj_2 = lifting(sj_1, j-2)
#sj_3, dj_3 = lifting(sj_2, j-3)
#
#sj = np.hstack((sj_3, dj_3, dj_2,dj_1))