# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:01:01 2026

@author: callum

various common bits
"""

import numpy as np
from scipy.constants import h, c, e, Boltzmann

base_dir = "out"
hcnm = (h * c) / (1.0E-9)
dg_h2o = -5.10 # in eV
dg_co2 = -3.85 # in eV
xlim = [400.0, 2000.0]
nx = int(xlim[1] - xlim[0])

def kbt_ev(T):
    return T * (Boltzmann / e)

@np.vectorize
def ev_nm(x):
    ''' convert eV to nm and vice-versa '''
    return (hcnm / e) * (1.0 / x)

base_rates = {
"cs"     : 1.0 / 10.0E-12,
"decay"  : 1.0 / 1.0E-9,
"ox"     : 1.0 / 1.0E-3,
"lin"    : 1.0 / 10.0E-3,
"cyc"    : 1.0 / 10.0E-3,
"red"    : 1.0 / 10.0E-3,
"recomb" : 1.0,
"inhib"  : 1.0,
}
