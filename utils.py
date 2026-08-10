# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:01:01 2026

@author: callum

various common bits
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c, e, Boltzmann

base_dir = "out"
hcnm = (h * c) / (1.0E-9)
dg_h2o = -5.10 # in eV
dg_co2 = -3.85 # in eV
xlim = [400.0, 4000.0]
increment = 10
nx = int(xlim[1] - xlim[0])//increment + 1 # + 1 for linspace

def kbt_ev(T):
    return T * (Boltzmann / e)

@np.vectorize
def ev_nm(x):
    ''' convert eV to nm and vice-versa '''
    return (hcnm / e) * (1.0 / x)

diff_rates = {
"cs"     : 1.0 / 10.0E-12,
"decay"  : 1.0 / 1.0E-9,
"ox"     : 1.0 / 1.0E-3,
"lin"    : 1.0 / 10.0E-3,
"cyc"    : 1.0 / 10.0E-3,
"red"    : 1.0 / 10.0E-3,
"recomb" : 1.0,
"inhib"  : 1.0,
}

base_rates = {
"cs"     : 1.0,
"decay"  : 1.0,
"ox"     : 1.0,
"lin"    : 1.0,
"cyc"    : 1.0,
"red"    : 1.0,
"recomb" : 1.0,
"inhib"  : 1.0,
}

def strip_name(name):
    '''
    antimony or roadrunner or whatever output the column names as
    [modelLabel_speciesName]. we do not want the brackets or model name
    so get rid of them
    '''
    return name[name.find("_") + 1:name.find("]")]   

def get_latex(colname):
    '''
    take a species name and return a properly latex'd version.
    We name all the species with a letter e.g. D, C, A, and so on,
    and then a single letter telling us whether it's oxidised or reduced.
    '''
    td = {'o': "ox", "r": "red", "s": "s"}
    # cut the leading model label and underscore from the column name
    name = strip_name(colname)
    if name == "Estar":
        lstr = r'$ E^{*} $'
    else:
        if len(name) == 2 and name[1] in td:
            lstr = f"$ {name[0]}_{{\\text{{{td[name[1]]}}}}} $"
        else:
            lstr = name
    return lstr

def simple_plot(result, cols):
    fig, ax = plt.subplots()
    labels = [get_latex(c) for c in cols]
    for cn, cl in zip(cols, labels):
        ax.plot(result['time'], result[cn], label=cl)
    ax.legend()
    return fig, ax

if __name__ == "__main__":
    names = ['[A_P]', '[A_P1]', "[A_P1s]", "[A_Do]", "[A_Estar]", "[A_Cr]", "[A_Qo]"]
    for n in names:
        print(n, strip_name(n), get_latex(n))