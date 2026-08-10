# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:33:16 2026

@author: callum

this is some code to set up and simulate a bunch of different
chemical reaction networks that aim to describe possible ways to do
photosynthesis. basically we think about all the ways we could use
light energy to drive carbon reduction that are *thermodynamically*
feasible, and then set up our little networks and see how well they do
at reducing carbon, etc.

at the moment it's a mess, everything's in one big horrible monolithic
file, etc. part of that is hard to avoid because we need to format
the big string that contains the model with stuff like the stoichiometry
of various things, so we can't just put it in a separate text file and
have antimony read it in because it'll fail. also it breaks on a load
of other stuff to do with SMBL conversion and i can't be arsed to figure
out why. i will maybe fix this at some point.
"""

import os
import pathlib
import zipfile
import roadrunner
import tellurium as te
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import light
import utils

tmax = 1000
cycles = 10 # only relevant if day/night cycle is turned on
period = tmax // cycles
# print out the intermediate python f-string as a file. useful for debugging
write_fstr = True
# write out the final antimony model as a file. also for debugging
write_sbml = True
# make a time-series plot of quantities to check convergence
# if not using steady state methods
make_time_series_plots = False

dg_p = utils.dg_co2 - utils.dg_h2o

'''
various scheme-dependent parameters.
note that scheme_to_call is there because some of the schemes
are actually identical apart from the stoichiometry of the
dark carbon reduction reactions. sadly we can't have a number as
a parameter to an antimony function (there's another comment about
this below) so we need to f-string it in, and I didn't want to leave
the stoichiometries as defined constants like they were before because
then if you forget to change them when changing scheme you'll get
incorrect results. so now they are here in a big dict of dicts instead.
maybe a bit inelegant but it does the trick.
also note that deltaG_gamma all have two entries: this is because
later two-photon schemes require it and they're called
in the f-string as exc_rate[0], exc_rate[1], so that list
must have two members
'''
params = {
    "one-photon": {# reddest usable wavelength of light
        "1a" : { 
            'scheme_to_call': "one_p_1a",
            'deltaG_gamma': [utils.ev_nm(611.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0, # not used for scheme 1
            },
        "1b" : { 
            'scheme_to_call': "one_p_1b",
            'deltaG_gamma': [utils.ev_nm(541.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0,
            },
        "1c" : { 
            'scheme_to_call': "one_p_1c",
            'deltaG_gamma': [utils.ev_nm(541.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0,
            },
        "2a" : { 
            'scheme_to_call': "one_p_2a",
            'deltaG_gamma': [utils.ev_nm(611.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
            },
        "2b" : { 
            'scheme_to_call': "one_p_2b",
            'deltaG_gamma': [utils.ev_nm(541.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
            },
        "2c" : { 
            'scheme_to_call': "one_p_2c",
            'deltaG_gamma': [utils.ev_nm(541.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
            },
        "3a" : { 
            'scheme_to_call': "one_p_3a",
            'deltaG_gamma': [utils.ev_nm(700.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
            },
        "3b" : { 
            'scheme_to_call': "one_p_3b",
            'deltaG_gamma': [utils.ev_nm(611.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
            },
        "3c" : { 
            'scheme_to_call': "one_p_3c",
            'deltaG_gamma': [utils.ev_nm(611.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
            },
        "4a"  : { 
            'scheme_to_call': "one_p_3a",
            'deltaG_gamma': [utils.ev_nm(1192.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 4,
            },
        "4b"  : { 
            'scheme_to_call': "one_p_3b",
            'deltaG_gamma': [utils.ev_nm(954.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 4,
            },
        "4c"  : { 
            'scheme_to_call': "one_p_3c",
            'deltaG_gamma': [utils.ev_nm(954.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 4,
            },
        "4d"  : { 
            'scheme_to_call': "one_p_4d",
            'deltaG_gamma': [utils.ev_nm(954.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 4,
            },
        "4e"  : { 
            'scheme_to_call': "one_p_4e",
            'deltaG_gamma': [utils.ev_nm(954.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 4,
            },
        "5a"  : { 
            'scheme_to_call': "one_p_5a",
            'deltaG_gamma': [utils.ev_nm(1192.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0,
            },
        "5b"  : { 
            'scheme_to_call': "one_p_5b",
            'deltaG_gamma': [utils.ev_nm(1192.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0,
            },
        "5c"  : { 
            'scheme_to_call': "one_p_5c",
            'deltaG_gamma': [utils.ev_nm(1192.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 5,
            },
        "5d"  : { 
            'scheme_to_call': "one_p_5d",
            'deltaG_gamma': [utils.ev_nm(1192.0), np.nan],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 5,
            },
    },
    "uncoupled": {
        "1a" : { 
            'scheme_to_call': "two_p_1a",
            'deltaG_gamma': [utils.ev_nm(611.0), utils.ev_nm(611.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0, # not used for scheme 1
            },
        "1b" : { 
            'scheme_to_call': "two_p_1b",
            'deltaG_gamma': [utils.ev_nm(611.0), utils.ev_nm(541.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0, # not used for scheme 1
            },
        "1c" : { 
            'scheme_to_call': "two_p_1c",
            'deltaG_gamma': [utils.ev_nm(541.0), utils.ev_nm(541.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0, # not used for scheme 1
            },
        "2a" : { 
            'scheme_to_call': "two_p_2a",
            'deltaG_gamma': [utils.ev_nm(611.0), utils.ev_nm(611.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0, # not used for scheme 1
            },
        "2b" : { 
            'scheme_to_call': "two_p_2b",
            'deltaG_gamma': [utils.ev_nm(611.0), utils.ev_nm(611.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0, # not used for scheme 1
            },
        "2c" : { 
            'scheme_to_call': "two_p_2c",
            'deltaG_gamma': [utils.ev_nm(611.0), utils.ev_nm(541.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0, # not used for scheme 1
            },
        "2d" : { 
            'scheme_to_call': "two_p_2d",
            'deltaG_gamma': [utils.ev_nm(541.0), utils.ev_nm(541.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0, # not used for scheme 1
            },
        "3a" : { 
            'scheme_to_call': "one_p_3a",
            'deltaG_gamma': [utils.ev_nm(700.0), utils.ev_nm(700.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
            },
        "4a" : { 
            'scheme_to_call': "one_p_3a",
            'deltaG_gamma': [utils.ev_nm(1192.0), utils.ev_nm(1192.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 4,
            },    
        "5a" : { 
            'scheme_to_call': "two_p_5a",
            'deltaG_gamma': [utils.ev_nm(611.0), utils.ev_nm(1192.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0,
            },
        "5b" : { 
            'scheme_to_call': "two_p_5b",
            'deltaG_gamma': [utils.ev_nm(700.0), utils.ev_nm(1192.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
            },
        "5c" : { 
            'scheme_to_call': "two_p_5c",
            'deltaG_gamma': [utils.ev_nm(1192.0), utils.ev_nm(1192.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 4,
            },
        "6a" : { 
            'scheme_to_call': "two_p_6a",
            'deltaG_gamma': [utils.ev_nm(1192.0), utils.ev_nm(1192.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 5,
            },
        "7a" : { 
            'scheme_to_call': "two_p_7a",
            'deltaG_gamma': [utils.ev_nm(611.0), utils.ev_nm(700.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
            },
        "7b" : { 
            'scheme_to_call': "two_p_7a",
            'deltaG_gamma': [utils.ev_nm(611.0), utils.ev_nm(1192.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 4,
            },
        "8a" : { 
            'scheme_to_call': "two_p_8a",
            'deltaG_gamma': [utils.ev_nm(611.0), utils.ev_nm(611.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
            },
        "8b" : { 
            'scheme_to_call': "two_p_8a",
            'deltaG_gamma': [utils.ev_nm(611.0), utils.ev_nm(954.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 4,
            },
        "9a" : { 
            'scheme_to_call': "two_p_9a",
            'deltaG_gamma': [utils.ev_nm(541.0), utils.ev_nm(700.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
            },
        "9b" : { 
            'scheme_to_call': "two_p_9a",
            'deltaG_gamma': [utils.ev_nm(541.0), utils.ev_nm(1192.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 4,
            },
        "10a" : { 
            'scheme_to_call': "two_p_10a",
            'deltaG_gamma': [utils.ev_nm(541.0), utils.ev_nm(611.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
            },
        "10b" : { 
            'scheme_to_call': "two_p_10a",
            'deltaG_gamma': [utils.ev_nm(541.0), utils.ev_nm(954.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 4,
            },
        "11a" : { 
            'scheme_to_call': "two_p_11a",
            'deltaG_gamma': [utils.ev_nm(611.0), utils.ev_nm(1192.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0,
            },
        "11b" : { 
            'scheme_to_call': "two_p_11b",
            'deltaG_gamma': [utils.ev_nm(611.0), utils.ev_nm(1192.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0,
            },
        "11c" : { 
            'scheme_to_call': "two_p_11c",
            'deltaG_gamma': [utils.ev_nm(541.0), utils.ev_nm(1192.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0,
            },
        "11d" : { 
            'scheme_to_call': "two_p_11d",
            'deltaG_gamma': [utils.ev_nm(541.0), utils.ev_nm(1192.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0,
            },
    },
    "coupled": {
        "1a": {
            'scheme_to_call': "two_p_c_1a",
            'deltaG_gamma': [utils.ev_nm(883.0), utils.ev_nm(883.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0,
        },
        "1b": {
            'scheme_to_call': "two_p_c_1b",
            'deltaG_gamma': [utils.ev_nm(745.0), utils.ev_nm(883.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0,
        },
        "1c": {
            'scheme_to_call': "two_p_c_1c",
            'deltaG_gamma': [utils.ev_nm(883.0), utils.ev_nm(745.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0,
        },
        "1d": {
            'scheme_to_call': "two_p_c_1d",
            'deltaG_gamma': [utils.ev_nm(745.0), utils.ev_nm(745.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 0,
        },
        "2a": {
            'scheme_to_call': "two_p_c_2a",
            'deltaG_gamma': [utils.ev_nm(883.0), utils.ev_nm(883.0)],
            'split_lin_cyc': True,
            'resp_stoi': 5,
            'dcr_stoi': 0,
        },
        "2b": {
            'scheme_to_call': "two_p_c_2b",
            'deltaG_gamma': [utils.ev_nm(745.0), utils.ev_nm(883.0)],
            'split_lin_cyc': True,
            'resp_stoi': 5,
            'dcr_stoi': 0,
        },
        "2c": {
            'scheme_to_call': "two_p_c_2c",
            'deltaG_gamma': [utils.ev_nm(883.0), utils.ev_nm(745.0)],
            'split_lin_cyc': True,
            'resp_stoi': 5,
            'dcr_stoi': 0,
        },
        "2d": {
            'scheme_to_call': "two_p_c_2d",
            'deltaG_gamma': [utils.ev_nm(745.0), utils.ev_nm(745.0)],
            'split_lin_cyc': True,
            'resp_stoi': 5,
            'dcr_stoi': 0,
        },
        "3a": {
            'scheme_to_call': "two_p_c_3a",
            'deltaG_gamma': [utils.ev_nm(973.0), utils.ev_nm(973.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
        },
        "3b": {
            'scheme_to_call': "two_p_c_3b",
            'deltaG_gamma': [utils.ev_nm(808.0), utils.ev_nm(973.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
        },
        "3c": {
            'scheme_to_call': "two_p_c_3c",
            'deltaG_gamma': [utils.ev_nm(973.0), utils.ev_nm(808.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
        },
        "3d": {
            'scheme_to_call': "two_p_c_3d",
            'deltaG_gamma': [utils.ev_nm(808.0), utils.ev_nm(808.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 1,
        },
        "4a": {
            'scheme_to_call': "two_p_c_4a",
            'deltaG_gamma': [utils.ev_nm(973.0), utils.ev_nm(973.0)],
            'split_lin_cyc': True,
            'resp_stoi': 5,
            'dcr_stoi': 1,
        },
        "4b": {
            'scheme_to_call': "two_p_c_4b",
            'deltaG_gamma': [utils.ev_nm(808.0), utils.ev_nm(973.0)],
            'split_lin_cyc': True,
            'resp_stoi': 5,
            'dcr_stoi': 1,
        },
        "4c": {
            'scheme_to_call': "two_p_c_4c",
            'deltaG_gamma': [utils.ev_nm(973.0), utils.ev_nm(808.0)],
            'split_lin_cyc': True,
            'resp_stoi': 5,
            'dcr_stoi': 1,
        },
        "4d": {
            'scheme_to_call': "two_p_c_4d",
            'deltaG_gamma': [utils.ev_nm(808.0), utils.ev_nm(808.0)],
            'split_lin_cyc': True,
            'resp_stoi': 5,
            'dcr_stoi': 1,
        },
        "5a": {
            'scheme_to_call': "two_p_c_3a",
            'deltaG_gamma': [utils.ev_nm(1192.0), utils.ev_nm(1192.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 4,
        },
        "5b": {
            'scheme_to_call': "two_p_c_3b",
            'deltaG_gamma': [utils.ev_nm(954.0), utils.ev_nm(1192.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 4,
        },
        "5c": {
            'scheme_to_call': "two_p_c_3c",
            'deltaG_gamma': [utils.ev_nm(1192.0), utils.ev_nm(954.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 4,
        },
        "5d": {
            'scheme_to_call': "two_p_c_3d",
            'deltaG_gamma': [utils.ev_nm(954.0), utils.ev_nm(954.0)],
            'split_lin_cyc': False,
            'resp_stoi': 5,
            'dcr_stoi': 4,
        },
        "6a": {
            'scheme_to_call': "two_p_c_4a",
            'deltaG_gamma': [utils.ev_nm(1192.0), utils.ev_nm(1192.0)],
            'split_lin_cyc': True,
            'resp_stoi': 5,
            'dcr_stoi': 4,
        },
        "6b": {
            'scheme_to_call': "two_p_c_4b",
            'deltaG_gamma': [utils.ev_nm(954.0), utils.ev_nm(1192.0)],
            'split_lin_cyc': True,
            'resp_stoi': 5,
            'dcr_stoi': 4,
        },
        "6c": {
            'scheme_to_call': "two_p_c_4c",
            'deltaG_gamma': [utils.ev_nm(1192.0), utils.ev_nm(954.0)],
            'split_lin_cyc': True,
            'resp_stoi': 5,
            'dcr_stoi': 4,
        },
        "6d": {
            'scheme_to_call': "two_p_c_4d",
            'deltaG_gamma': [utils.ev_nm(954.0), utils.ev_nm(954.0)],
            'split_lin_cyc': True,
            'resp_stoi': 5,
            'dcr_stoi': 4,
        },
    }
}

simulation_params = {
    "light"         : "switch",
    "intensity"     : "high",
    "maintenance_C" : False,
    "k_resp"        : 1,
    "k_resp_back"   : 1,
    "k1"            : 1,
    "k2"            : 1,
    "k3"            : 1,
    "k4"            : 1,
    "k_day"         : 1,
    "k_night"       : 1,
    "k_maintenance" : 1,
    "k_cr"          : 1,
    "k_cr_back"     : 1,
    "phi_lin_cyc"   : 0.5,
}

cols_to_log = ['[A_Cr]', '[A_Estar]', '[A_Do]', '[A_Ar]', '[A_Br]', '[A_Qr]']
cols_to_plot = ['[A_Cr]', '[A_Estar]']

def make_outdir(basedir, scheme_set): 
    output_dir = os.path.join(basedir,
                f"{scheme_set}",
                f"maintenance_Cred_{simulation_params['maintenance_C']}",
                f"{simulation_params['intensity']}_intensity")
    if scheme_set == "coupled":
        output_dir = os.path.join(output_dir, f"phi_{simulation_params['phi_lin_cyc']}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def run(scheme, Tstar, scheme_set, sim_params=simulation_params):
    '''
    wrapper to set up all the antimony models, pick the one
    you want with the parameters you want, simulate, plot
    the results and output relevant data.
    having one massive f-string is ugly, i know. the reason we can't
    just save it all to one txt file and pull the relevant schemes
    at runtime is the stoichiometries for Estar (see models
    respiration and DCR_E) - these must be integers or antimony
    will break, but they might differ depending on scheme, so we need
    to insert them here. This is really annoying but currently it's
    most flexible usage situation i can come up with.
    
    arguments
    ---------

    `scheme` should be a *string* corresponding to a key in
    `scheme_params` and a named scheme in the long f-string below.
    `Tstar` should match one of the stellar temperatures in light.py,
    which are a bunch of effective stellar temperatures taken from the
    isochrones of Baraffe et al.
    `params` is a dict of the light type, stoichiometries and rates.
    light type should be "periodic", "switch" or "constant"

    outputs
    -------

    NB: haven't fixed what these will be yet
    '''
    scheme_params = params[scheme_set]
    if scheme not in scheme_params.keys():
        raise KeyError(f"Incorrect scheme provided to run(). Scheme = {scheme}, allowed schemes are {params.keys()}")
    if Tstar not in light.phz_stars['Tstar']:
        raise ValueError(f"Incorrect stellar temperature provided to run(). Tstar = {Tstar}, allowed values are {light.phz_stars['Tstar']}")
    print(f"Ts = {Tstar}, Scheme {scheme}")
    spectrum, out_pref = light.stellar_hz_midpoint(Tstar)
    total_flux = light.micromole_in_region(spectrum,
            utils.xlim[0], utils.ev_nm(dg_p))
    sc_p = scheme_params[scheme]
    scheme_to_call = sc_p["scheme_to_call"]

    output_dir = make_outdir(utils.base_dir, scheme_set)
    
    fig, ax = plt.subplots()
    ax.plot(spectrum[:, 0], spectrum[:, 1])
    ax.set_title(f"T = {Tstar}")
    plt.grid(visible=True)
    fig.savefig(os.path.join(output_dir, f"{out_pref}_spectrum.pdf"))
    plt.close()
    np.savetxt(os.path.join(output_dir, f"{out_pref}_spectrum.csv"), spectrum)

    exc_rate = np.zeros(len(sc_p['deltaG_gamma']), dtype=float)
    for i, dgg in enumerate(sc_p['deltaG_gamma']):
        uf = light.micromole_in_region(spectrum, 
                utils.xlim[0], utils.ev_nm(dgg))
        exc_rate[i] = uf/total_flux
    
    if sim_params["maintenance_C"]:
        maint_str = " * Cr"
    else:
        maint_str = ""

    if sc_p['split_lin_cyc']:
        pstr = f"A.P1 = {2 * sim_params['phi_lin_cyc']}; " +\
        f"A.P2 = {2. * (1. - sim_params['phi_lin_cyc'])}"
    else:
        pstr = "A.P1 = 1; A.P2 = 1"
        
    if sim_params["intensity"] == "high":
        exc_rate *= 10
    elif sim_params["intensity"] == "medium":
        exc_rate *= 1 # for clarity
    elif sim_params["intensity"] == "low":
        exc_rate *= 0.1
    else:
        raise ValueError(f"params['intensity'] value = {params['intensity']} not recognised. Should be low, medium or high")
    
    print(f"Total flux = {total_flux} muE. exc_rate = {exc_rate}")
    if sim_params["light"] == "periodic":
        gamma_string = [f"square_wave(time, {period}, {er})" for er in exc_rate]
    elif sim_params["light"] == "switch":
        gamma_string = [f"switch_off(time, {tmax//2}, {er})" for er in exc_rate]
    elif sim_params["light"] == "constant":
        gamma_string = [f"constant(time, {er})" for er in exc_rate]
    else:
        raise ValueError(f"sim_params['light'] value = {sim_params['light']} not recognised. Should be periodic, switch or constant")

    antstr = f"""
    // three functions for light input
    function square_wave(t, period, intensity)
      intensity * (1 - (floor(t / period) - 2 * floor(floor(t / period) / 2)))
    end
    
    function switch_off(t, off_time, intensity)
        piecewise(intensity, (t - off_time) <= 0, 0)
    end

    function constant(t, intensity)
        intensity
    end

    /* the submodels defined below here are for common parts
    * of reactions. The convention is that the rates go first in the
    * argument list, followed by the species, and all the species are
    * explicitly declared as species in the models. Then they're called
    * with all the arguments in the same way in the main schemes. This
    * allows use to reuse them (and the variable definitions, and species
    * initialisations, and ensures that the species in the submodels always
    * correspond to the ones in the main schemes) */

    model respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
        fw: Do + Cr + {sc_p["resp_stoi"]} $E -> $Dr + $Co + {sc_p["resp_stoi"]} Estar; kr*Do*Cr*($E^{sc_p["resp_stoi"]});
        bw: $Dr + $Co + {sc_p["resp_stoi"]} Estar -> Do + Cr + {sc_p["resp_stoi"]} $E; kb*(Estar^{sc_p["resp_stoi"]});
    end

    model bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
        species Do, Dr, Co, Cr, E, Estar
        kar = 1
        kbr = 1
        kqr = 1
        Resp   : respiration(kr, kb, Do, $Dr, $Co, Cr, $E, Estar)
        Edrain : Estar -> $E  ; km * Estar {maint_str}
        Cdrain : Cr    -> $Co ; kd * Cr
        Ddrain : Do    -> $Dr ; kn * Do
        Adrain : Ar    -> $Ao ; kar * Ar
        Bdrain : Br    -> $Bo ; kbr * Br
        Qdrain : Qr    -> $Qo ; kqr * Qr
    end
    
    // reversible dark carbon reduction with no energy production
    model DCR_noE(kcr, kcrb, Ar, Ao, Cr, Co)
        species Ar, Ao, Cr, Co
        fw: Ar + $Co -> Ao + Cr; kcr * Ar
        bw: Ao + Cr -> Ar + $Co; kcrb * Ao * Cr
    end

    // dark carbon reduction with energy production
    model DCR_E(kcr, kcrb, Ar, Ao, Cr, Co, E, Estar)
        // this is currently irreversible, per Chris
        // to make reversible just uncomment the backward reaction
        species Ar, Ao, Cr, Co, E, Estar
        fw: Ar + Co + {sc_p["dcr_stoi"]} Estar -> Ao + Cr + {sc_p["dcr_stoi"]} E; kcr * Ar * Co * (Estar^{sc_p["dcr_stoi"]})
        // bw: Ao + Cr + {sc_p["dcr_stoi"]} E -> Ar + Co + {sc_p["dcr_stoi"]} Estar; kcrb * Ao * Cr * (E^{sc_p["dcr_stoi"]})
    end

    // just acceptor reduction and donor oxidation
    model photo(gam, k1, k2, P, Ps, Po, Dr, Do, Ar, Ao, E, Estar)
       species P, Ps, Po, Dr, Do, Ar, Ao, E, Estar
       PE   : P       -> Ps      ; gam * P
       Ared : Ps + Ao -> Po + Ar ; k1 * Ps * Ao
       Dox  : Dr + Po -> Do + P  ; k2 * Po * Dr
    end

    // same but with energy production at the acceptor side
    model photoEA(gam, k1, k2, P, Ps, Po, Dr, Do, Ar, Ao, E, Estar)
       species P, Ps, Po, Dr, Do, Ao, Ar, E, Estar
       PE   : P           -> Ps              ; gam * P
       Ared : Ps + Ao + E -> Po + Ar + Estar ; k1 * Ps * Ao * E
       Dox  : Dr + Po     -> Do + P          ; k2 * Po * Dr
    end

    // same again but with energy production at the donor side
    model photoED(gam, k1, k2, P, Ps, Po, Dr, Do, Ar, Ao, E, Estar)
       species P, Ps, Po, Dr, Do, Ao, Ar, E, Estar
       PE   : P            -> Ps             ; gam * P
       Ared : Ps + Ao      -> Po + Ar        ; k1 * Ps * Ao
       Dox  : Dr + Po + E  -> Do + P + Estar ; k2 * Po * Dr * E
    end

    /* NB: every model must have a name of the form `scheme_SCHEMELABEL`,
    * have all of the rates that are specified at the bottom as arguments,
    * in the right order, and include the species definitions as below.
    * the rate arguments are so that we can just set the rates in the
    * final bit, sim_{scheme}, and the species are so we can initialise
    * all the populations there as well.
    * This minimises the amount of per-scheme code we have to write
    * by eliminating the need to duplicate the rate declarations and
    * species initialisations for each scheme at the cost of this
    * bit of boilerplate, but I think it's a decent tradeoff. */

    model one_p_1a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
       PC : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
       B  : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end

    model one_p_1b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
       PC : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
       B  : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end

    model one_p_1c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
       PC : photoED(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
       B  : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end

    model one_p_2a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
       PC : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
       D  : DCR_noE(kcr, kcrb, Ar, Ao, Cr, $Co)
       B  : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end

    model one_p_2b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
       PC : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
       D  : DCR_noE(kcr, kcrb, Ar, Ao, Cr, $Co)
       B  : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end

    model one_p_2c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
       PC : photoED(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
       D  : DCR_noE(kcr, kcrb, Ar, Ao, Cr, $Co)
       B  : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end

    model one_p_3a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
       PC : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
       D  : DCR_E(kcr, kcrb, Ar, Ao, Cr, $Co, $E, Estar)
       B  : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end

    model one_p_3b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
       PC : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
       D  : DCR_E(kcr, kcrb, Ar, Ao, Cr, $Co, $E, Estar)
       B  : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end

    model one_p_3c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
       PC : photoED(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
       D  : DCR_E(kcr, kcrb, Ar, Ao, Cr, $Co, $E, Estar)
       B  : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model one_p_4d(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
       PC  : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
       D   : DCR_E(kcr, kcrb, Ar, Ao, Cr, $Co, $E, Estar)
       B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model one_p_4e(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
       PC  : photoED(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
       D   : DCR_E(kcr, kcrb, Ar, Ao, Cr, $Co, $E, Estar)
       B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model one_p_5a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;  
        PC : photo(gam1, k1, k2, P1, P1s, P1o, Ar, Ao, Ar, Ao, $E, Estar)
        B  : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model one_p_5b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC : photoED(gam1, k1, k2, P1, P1s, P1o, Ar, Ao, Ar, Ao, $E, Estar)
        B  : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model one_p_5c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC : photoEA(gam1, k1, k2, P1, P1s, P1o, Ar, Ao, Ar, Ao, $E, Estar)
        D  : DCR_E(kcr, kcrb, $Dr, Do, Cr, $Co, $E, Estar)
        B  : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model one_p_5d(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC : photoED(gam1, k1, k2, P1, P1s, P1o, Ar, Ao, Ar, Ao, $E, Estar)
        D  : DCR_E(kcr, kcrb, $Dr, Do, Cr, $Co, $E, Estar)
        B  : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_1a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
        PC2 : photo(gam2, k3, k4, P2, P2s, P2o, $Dr, Do, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
        
    model two_p_1b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, $Dr, Do, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_1c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, $Dr, Do, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
        
    model two_p_2a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photo(gam2, k3, k4, P2, P2s, P2o, $Dr, Do, Ar, Ao, $E, Estar)
        D   : DCR_noE(kcr, kcrb, Ar, Ao, Cr, $Co)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_2b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photo(gam2, k3, k4, P2, P2s, P2o, $Dr, Do, Br, $Bo, $E, Estar)
        D1  : DCR_noE(kcr, kcrb, Ar, Ao, Cr, $Co)
        D2  : DCR_noE(kcr, kcrb, Br, $Bo, Cr, $Co)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_2c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, $Dr, Do, Ar, Ao, $E, Estar)
        D1  : DCR_noE(kcr, kcrb, Ar, Ao, Cr, $Co)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
       
    model two_p_2d(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, $Dr, Do, Ar, Ao, $E, Estar)
        D1  : DCR_noE(kcr, kcrb, Ar, Ao, Cr, $Co)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_3a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photo(gam2, k3, k4, P2, P2s, P2o, $Dr, Do, Ar, Ao, $E, Estar)
        Aox : Ar + Do        -> Ao + $Dr       ; k3 * Ar * Do
        D1  : DCR_E(kcr, kcrb, Ar, Ao, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_5a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, Br, $Bo, Br, $Bo, $E, Estar)
        D1  : DCR_noE(kcr, kcrb, Ar, Ao, Cr, $Co)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_5b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, Br, $Bo, Br, $Bo, $E, Estar)
        D1  : DCR_E(kcr, kcrb, Ar, Ao, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_5c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, Br, $Bo, Br, $Bo, $E, Estar)
        Aox : Ar + Do        -> Ao + $Dr       ; k3 * Ar * Do
        D1  : DCR_E(kcr, kcrb, Ar, Ao, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_6a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photoEA(gam1, k1, k2, P1, P1s, P1o, Ar, Ao, Ar, Ao, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, Br, $Bo, Br, $Bo, $E, Estar)
        D1  : DCR_E(kcr, kcrb, $Dr, Do, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_7a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
        PC2 : photo(gam2, k3, k4, P2, P2s, P2o, $Dr, Do, Ar, Ao, $E, Estar)
        D1  : DCR_E(kcr, kcrb, Ar, Ao, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_8a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, $Dr, Do, Ar, Ao, $E, Estar)
        D1  : DCR_E(kcr, kcrb, Ar, Ao, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_9a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
        PC2 : photo(gam2, k3, k4, P2, P2s, P2o, $Dr, Do, Ar, Ao, $E, Estar)
        D1  : DCR_E(kcr, kcrb, Ar, Ao, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_10a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, $Dr, Do, Ar, Ao, $E, Estar)
        D1  : DCR_E(kcr, kcrb, Ar, Ao, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
        
    model two_p_11a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Ar, Ao, $E, Estar)
        D1  : DCR_E(kcr, kcrb, Ar, Ao, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
            
    model two_p_11b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Ar, Ao, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
            
    model two_p_11c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Ar, Ao, $E, Estar)
        D1  : DCR_E(kcr, kcrb, Ar, Ao, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
            
    model two_p_11d(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Ar, Ao, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_c_1a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photo(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_c_1b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photo(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
        
    model two_p_c_1c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
        
    model two_p_c_1d(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
                
    model two_p_c_2a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1  : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2l : photo(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Cr, $Co, $E, Estar)
        PC2c : photoEA(gam2, k3, k4, P2, P2s, P2o, Br, Bo, Br, Bo, $E, Estar)
        B    : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    model two_p_c_2b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1  : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2l : photo(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Cr, $Co, $E, Estar)
        PC2c : photoEA(gam2, k3, k4, P2, P2s, P2o, Br, Bo, Br, Bo, $E, Estar)
        B    : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
             
    model two_p_c_2c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1  : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2l : photoEA(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Cr, $Co, $E, Estar)
        PC2c : photoEA(gam2, k3, k4, P2, P2s, P2o, Br, Bo, Br, Bo, $E, Estar)
        B    : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
                 
    model two_p_c_2d(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1  : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2l : photoEA(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Cr, $Co, $E, Estar)
        PC2c : photoEA(gam2, k3, k4, P2, P2s, P2o, Br, Bo, Br, Bo, $E, Estar)
        B    : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
                     
    model two_p_c_3a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photo(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Br, Bo, $E, Estar)
        D1  : DCR_E(kcr, kcrb, Br, Bo, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
                         
    model two_p_c_3b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photo(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Br, Bo, $E, Estar)
        D1  : DCR_E(kcr, kcrb, Br, Bo, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
                         
    model two_p_c_3c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Br, Bo, $E, Estar)
        D1  : DCR_E(kcr, kcrb, Br, Bo, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
                         
    model two_p_c_3d(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1 : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2 : photoEA(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Br, Bo, $E, Estar)
        D1  : DCR_E(kcr, kcrb, Br, Bo, Cr, $Co, $E, Estar)
        B   : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
                             
    model two_p_c_4a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1  : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2l : photo(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Br, Bo, $E, Estar)
        PC2c : photoEA(gam2, k3, k4, P2, P2s, P2o, Qr, Qo, Qr, Qo, $E, Estar)
        D1   : DCR_E(kcr, kcrb, Br, Bo, Cr, $Co, $E, Estar)
        B    : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
                             
    model two_p_c_4b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1  : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2l : photo(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Br, Bo, $E, Estar)
        PC2c : photoEA(gam2, k3, k4, P2, P2s, P2o, Qr, Qo, Qr, Qo, $E, Estar)
        D1   : DCR_E(kcr, kcrb, Br, Bo, Cr, $Co, $E, Estar)
        B    : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
                                 
    model two_p_c_4c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1  : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2l : photoEA(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Br, Bo, $E, Estar)
        PC2c : photoEA(gam2, k3, k4, P2, P2s, P2o, Qr, Qo, Qr, Qo, $E, Estar)
        D1   : DCR_E(kcr, kcrb, Br, Bo, Cr, $Co, $E, Estar)
        B    : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
                                     
    model two_p_c_4d(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, E, Estar, Ar, Ao, Br, Bo, Qr, Qo;
        PC1  : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, Ao, $E, Estar)
        PC2l : photoEA(gam2, k3, k4, P2, P2s, P2o, Ar, Ao, Br, Bo, $E, Estar)
        PC2c : photoEA(gam2, k3, k4, P2, P2s, P2o, Qr, Qo, Qr, Qo, $E, Estar)
        D1   : DCR_E(kcr, kcrb, Br, Bo, Cr, $Co, $E, Estar)
        B    : bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Ar, Ao, Br, Bo, Qr, Qo)
    end
    
    // pull in whichever scheme you want from above and run that model
    model sim()
        // Variable initialization:
        gam1 := {gamma_string[0]}
        gam2 := {gamma_string[1]}
        k1    = {sim_params["k1"]}
        k2    = {sim_params["k2"]}
        k3    = {sim_params["k3"]}
        k4    = {sim_params["k4"]}
        km    = {sim_params["k_maintenance"]}
        kr    = {sim_params["k_resp"]} // this controls how much Cr you accumulate
        kb    = {sim_params["k_resp_back"]}
        kd    = {sim_params["k_day"]}
        kn    = {sim_params["k_night"]}
        kcr   = {sim_params["k_cr"]}
        kcrb  = {sim_params["k_cr_back"]}
        A: {scheme_to_call}(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        // Species initializations:
        var species A.P1, A.P1s, A.P1o, A.P2, A.P2s, A.P2o, A.Cr, A.Ar, A.Br, A.Qr, A.Do, A.Estar;
        const species A.Co, A.Dr, A.E, A.Ao, A.Bo, A.Qo

        {pstr}
        A.P1s = 0; A.P1o = 0;
        A.P2s = 0; A.P2o = 0;
        A.Cr = 0;    A.Co = 1;
        A.Ar = 0;    A.Ao = 1;
        A.Br = 0;    A.Bo = 1;
        A.Qr = 0;    A.Qo = 1;
        A.Do = 0;    A.Dr = 1;
        A.Estar = 0; A.E = 1;
    end
    """
    output_files = []
    if write_fstr:
        r_file = os.path.join(output_dir,
                          f"{out_pref}_scheme_{scheme}_fstr.txt")
        output_files.append(r_file)
        with open(r_file, "w") as f:
            f.write(antstr)
    sbml_str = te.antimonyToSBML(antstr)
    rr = roadrunner.RoadRunner(sbml_str)
    ss = rr.steadyState()
    vals = rr.getSteadyStateValuesNamedArray()
    d = {}
    for cc in vals.colnames:
        d[cc] = vals[cc][0] # vals[cc] returns a len(1) array
    d['sum_residuals'] = ss
    print(d)
    
    if write_sbml:
        sbml_file = os.path.join(output_dir,
                          f"{out_pref}_scheme_{scheme}_SBML.txt")
        output_files.append(sbml_file)
        with open(sbml_file, "w") as ff:
            ff.write(rr.getCurrentSBML())
    return d, out_pref, output_files

def df_plot(df, columns, output_file):
    '''
    plot the relevant columns as defined above from dataframe df
    '''
    fig, ax = plt.subplots()
    for cn, cl in zip(columns, [utils.get_latex(cc) for cc in columns]):
        ax.plot(df.index.to_numpy(), df[cn].to_numpy(),
                label=cl, marker='o', ms=5.0)
    ax.legend()
    ax.set_xlabel(df.index.name)
    ax.set_ylabel("concentration")
    plt.grid(visible=True)
    plt.show()
    plot_file = os.path.splitext(output_file)[0] + ".pdf"
    fig.savefig(plot_file)
    plt.close()

def loop(constant, scheme_set):
    '''
    for a given constant Tstar or scheme, loop over all schemes or Tstars
    and run the simulations, returning a dataframe of steady state results
    '''
    scheme_params = params[scheme_set]
    output_dir = make_outdir(utils.base_dir, scheme_set)
    output_files = []
    if constant in light.phz_stars['Tstar']:
        loop_var = scheme_params.keys()
        loop_str = "scheme"
        idx = pd.Index(scheme_params.keys(), name="scheme")
    elif constant in scheme_params.keys():
        loop_var = light.phz_stars['Tstar']
        loop_str = "Tstar"
        idx = pd.Index(light.phz_stars['Tstar'], name="Tstar")
    else:
        raise ValueError("incorrect value {constant} passed to schemes.loop_over")
    data = []
    for vv in loop_var:
        try:
            if loop_str == "Tstar":
                res_dict, out_pref, ofs = run(constant, vv, scheme_set)
            elif loop_str == "scheme":
                res_dict, out_pref, ofs = run(vv, constant, scheme_set)
            output_files.extend(ofs)
        except RuntimeError:
            print("Runtime error in r.simulate() for {constant}, {vv}")
            continue
        data.append(res_dict)

    if loop_str == "scheme":
        output_file = os.path.join(output_dir,
                                   f"{out_pref}_steady_state.csv")
    else:
        output_file = os.path.join(output_dir,
                                   f"scheme_{constant}_steady_state.csv")

    df = pd.DataFrame(data, index = idx)
    df.to_csv(output_file)
    df_plot(df, cols_to_log, output_file)
    return df, output_files

def loop_over_all(scheme_set):
    scheme_params = params[scheme_set]
    aa = {}
    output_files = []
    for Tstar in light.phz_stars['Tstar']:
        aa[Tstar], ofs = loop(Tstar, scheme_set)
        output_files.extend(ofs)
    df = pd.concat(aa, names=['Tstar', 'scheme'])
    idx_s = pd.Index(scheme_params.keys(), name="scheme")
    idx_t = pd.Index(light.phz_stars['Tstar'], name="Tstar")
    idx_ts = pd.MultiIndex.from_product([idx_t, idx_s])
    df.set_index(idx_ts)
    output_dir = make_outdir(utils.base_dir, scheme_set)
    output_file = os.path.join(output_dir, "all_steady_state.csv")
    df.to_csv(output_file)
    
    # zip up other output files
    zipfilename = os.path.join(output_dir, "other_files.zip")
    with zipfile.ZipFile(zipfilename, mode="w",
            compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in output_files:
            if os.path.isfile(filename):
                archive.write(filename, arcname = os.path.basename(filename))
    # delete the original files to save space
    for filename in output_files:
        pathlib.Path.unlink(filename, missing_ok = True)
    # plot - this is ugly but whatever
    fig, axes = plt.subplots(ncols=len(cols_to_plot),
                             figsize=((10*len(cols_to_plot), 8)))
    for ii, qq in enumerate(cols_to_plot):
        sdf = df[qq]
        for Tstar, new_df in sdf.groupby(level=0):
            ndf = new_df.droplevel(0)
            axes[ii].plot(ndf.index.to_numpy(), ndf.values, label=f"T = {Tstar}K", marker='o')
        axes[ii].set_title(utils.get_latex(qq))
        axes[ii].legend()
        axes[ii].set_xlabel("scheme")
        axes[0].set_ylabel("value at light off")
        axes[ii].grid(visible=True)
    title = f"{scheme_set} schemes, {simulation_params['intensity']} intensity"
    if scheme_set == "coupled":
        title += r', $ \phi $ = ' + f"{simulation_params['phi_lin_cyc']}"
    plt.suptitle(title)
    fig.savefig(os.path.splitext(output_file)[0] + ".pdf")
    plt.show()
    plt.close()
    return df

if __name__ == "__main__":
    scheme_set = "one-photon"
    df = loop_over_all(scheme_set)
