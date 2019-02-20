# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 09:42:00 2019

@author: arnlo
"""

import numpy as np

j = 3

sj = np.array([56, 40, 8, 24, 48, 48, 40, 16])


sj_1 = np.ones(2**(j-1))
for component in sj_1:
    component = (sj[2*component] + sj[2*component + 1])/2

dj_1 = 5
