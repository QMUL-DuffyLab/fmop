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
import tellurium as te
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import roadrunner
from scipy.constants import h, c, e, Boltzmann
import light

output_dir = "out"
Temperature = 300
hcnm = (h * c) / (1.0E-9)
beta_cm = 1.0 / (Boltzmann * Temperature * (1. / (100.0 * c))) # in wavenumbers
beta_ev = 1.0 / (Boltzmann * Temperature / e) # in eV
ten_kbt = 0.26

xlim = [400.0, 2000.0]
nx = int(xlim[1] - xlim[0])

tmax = 500
cycles = 10 # only relevant if day/night cycle is turned on
period = tmax // cycles

def estar_stoi(dg_donor, dg_acceptor, unit):
    return np.round(-(dg_donor - dg_acceptor) / unit)

@np.vectorize
def ev_nm(x):
    ''' convert eV to nm and vice-versa '''
    return (hcnm / e) * (1.0 / x)

dg_h2o = -5.10 # in eV
dg_co2 = -3.85 # in eV
dg_p = dg_co2 - dg_h2o

# note that these all have two entries: this is because
# later two-photon schemes require it and they're called
# in the f-string as exc_rate[0], exc_rate[1], so that list
# must have two members
dg_gamma = { # reddest usable wavelength of light
    "1a": [ev_nm(611.0), np.nan],
    "1b": [ev_nm(541.0), np.nan],
    "1c": [ev_nm(541.0), np.nan],
    "2a": [ev_nm(611.0), np.nan],
    "2b": [ev_nm(541.0), np.nan],
    "2c": [ev_nm(541.0), np.nan],
    "3a": [ev_nm(611.0), np.nan],
    "3b": [ev_nm(611.0), np.nan],
    "4": [ev_nm(1192.0), np.nan],
    "5a": [5 * ten_kbt, np.nan],
    "5b": [5 * ten_kbt, np.nan],
    "6": [6.8 * ten_kbt, 4 * ten_kbt],
    "7a": [ev_nm(882.0), ev_nm(882.0)],
    "7b": [ev_nm(744.0), ev_nm(882.0)],
    "7c": [ev_nm(744.0), ev_nm(744.0)],
    "7d": [4 * ten_kbt, dg_p + 3 * ten_kbt],
    "8a": [ev_nm(969.0), ev_nm(969.0)],
    "8b": [ev_nm(805.0), ev_nm(969.0)],
    "8c": [ev_nm(805.0), ev_nm(805.0)],
    }

params = {
    "light"         : "switch",
    "estoi"         : 5,
    "dcr_stoi"      : 5,
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
}

def run(scheme, Tstar, params=params, dg=dg_gamma,
        cycles=cycles, period=period):
    '''
    wrapper to set up all the antimony models, pick the one
    you want with the parameters you want, simulate, plot
    the results and output relevant data.
    i know this is ugly as fuck, believe me. but because of the
    multiple schemes, the parameters that need to be set at runtime
    and some of the ways that antimony works, this is currently the
    most flexible usage situation i can come up with.
    
    arguments
    ---------

    `scheme` should be a *string* corresponding to a key in
    `dg` and a named scheme in the long f-string below.
    `params` is a dict of the light type, stoichiometries and rates.
    `dg` is a dict of the photon energy gaps for each scheme
    light type should be "periodic", "switch" or "constant"

    outputs
    -------

    NB: haven't fixed what these will be yet
    '''
    print(f"Ts = {Tstar}, Scheme {scheme}")
    spectrum, out_pref = light.stellar_hz_midpoint(Tstar)
    total_flux = light.micromole_in_region(spectrum, xlim[0], ev_nm(dg_p))

    exc_rate = []
    for dgg in dg_gamma[scheme]:
        uf = light.micromole_in_region(spectrum, xlim[0], ev_nm(dgg))
        exc_rate.append(uf/total_flux)
        
    print(f"Excitation rate(s): {exc_rate}")

    if params["light"] == "periodic":
        gamma_string = [f"square_wave(time, {period}, {er})" for er in exc_rate]
    elif params["light"] == "switch":
        gamma_string = [f"switch_off(time, {tmax//2}, {er})" for er in exc_rate]
    elif params["light"] == "constant":
        gamma_string = [f"constant(time, {er})" for er in exc_rate]

    r = te.loada(f"""
    function square_wave(t, period, intensity)
      intensity * (1 - (floor(t / period) - 2 * floor(floor(t / period) / 2)))
    end
    
    function switch_off(t, off_time, intensity)
        piecewise(intensity, (t - off_time) <= 0, 0)
    end

    function constant(t, intensity)
        intensity
    end

    model storage(kn, kd)
       // i dunno if this will actually work like this,
       // but we're not using these reactions anyway atm
       fw: Cr -> Cstore; kd * Cr
       bw: Cstore -> Cr; kn * Cstore
    end
    
    model scheme_1a()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       gam := {gamma_string[0]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]} // this basically controls how much Cr you accumulate
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
      
       // Reactions:
       photo: P -> Ps; gam * P
       Cred: Ps + $Co -> Po + Cr; k1 * Ps
       Dox: $Dr + Po -> Do + P; k2 * Po
       M: Estar -> E; k_m * Estar * Cr
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});

       // Species initializations:
       P = 1; Ps = 0; Po = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Do = 0; Dr = 1;
       E = 1; Estar = 0;
    end

    model scheme_1b()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       gam := {gamma_string[0]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]} // this basically controls how much Cr you accumulate
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
      
       // Reactions:
       photo: P -> Ps; gam * P
       Cred: Ps + $Co + E -> Po + Cr + Estar; k1 * Ps * E
       Dox: $Dr + Po -> Do + P; k2 * Po
       M: Estar -> E; k_m * Estar * Cr
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});

       // Species initializations:
       P = 1; Ps = 0; Po = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Do = 0; Dr = 1;
       E = 1; Estar = 0;
    end

    model scheme_1c()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       gam := {gamma_string[0]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]} // this basically controls how much Cr you accumulate
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
      
       // Reactions:
       photo: P -> Ps; gam * P
       Cred: Ps + $Co -> Po + Cr; k1 * Ps
       Dox: $Dr + Po + E -> Do + P + Estar; k2 * Po * E
       M: Estar -> E; k_m * Estar * Cr
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});

       // Species initializations:
       P = 1; Ps = 0; Po = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Do = 0; Dr = 1;
       E = 1; Estar = 0;
    end

    model scheme_2a()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       gam := {gamma_string[0]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]} // this basically controls how much Cr you accumulate
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo: P -> Ps; gam * P
       Ared: Ps + Ao -> Po + Ar; k1 * Ps
       Dox: $Dr + Po -> Do + P; k2 * Po

       M: Estar -> E; k_m * Estar * Cr

       // dark carbon reduction
       DCRfw: Ar + $Co -> Ao + Cr; kcr * Ar
       DCRbw: Ao + Cr -> Ar + $Co; kcrb * Ao

       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});

       // Species initializations:
       P = 1; Ps = 0; Po = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    model scheme_2b()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       gam := {gamma_string[0]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]} // this basically controls how much Cr you accumulate
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo: P -> Ps; gam * P
       Ared: Ps + Ao + E -> Po + Ar + Estar; k1 * Ps * E
       Dox: $Dr + Po -> Do + P; k2 * Po

       M: Estar -> E; k_m * Estar * Cr

       // dark carbon reduction
       DCRfw: Ar + $Co -> Ao + Cr; kcr * Ar
       DCRbw: Ao + Cr -> Ar + $Co; kcrb * Ao

       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});

       // Species initializations:
       P = 1; Ps = 0; Po = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    model scheme_2c()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       gam := {gamma_string[0]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]} // this basically controls how much Cr you accumulate
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo: P -> Ps; gam * P
       Ared: Ps + Ao -> Po + Ar; k1 * Ps
       Dox: $Dr + Po + E -> Do + P + Estar; k2 * Po * E

       M: Estar -> E; k_m * Estar * Cr

       // dark carbon reduction
       DCRfw: Ar + $Co -> Ao + Cr; kcr * Ar
       DCRbw: Ao + Cr -> Ar + $Co; kcrb * Ao

       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});

       // Species initializations:
       P = 1; Ps = 0; Po = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    model scheme_3a()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       gam := {gamma_string[0]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]} // this basically controls how much Cr you accumulate
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo: P -> Ps; gam * P
       Ared: Ps + Ao + E -> Po + Ar + Estar; k1 * Ps * E
       Dox: $Dr + Po -> Do + P; k2 * Po

       // dark carbon reduction
       DCRfw: Ar + $Co + {params["dcr_stoi"]} Estar -> Ao + Cr + {params["dcr_stoi"]} E; kcr * Ar * (Estar^{params["dcr_stoi"]})
       DCRbw: Ao + Cr + {params["dcr_stoi"]} E -> Ar + $Co + {params["dcr_stoi"]} Estar; kcrb * Ao * (E^{params["dcr_stoi"]})

       M: Estar -> E; k_m * Estar * Cr

       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});

       // Species initializations:
       P = 1; Ps = 0; Po = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Ar = 0; Ao = 1;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    model scheme_3b()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       gam := {gamma_string[0]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]} // this basically controls how much Cr you accumulate
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo: P -> Ps; gam * P
       Ared: Ps + Ao -> Po + Ar; k1 * Ps
       Dox: $Dr + Po + E -> Do + P + Estar; k2 * Po * E

       // dark carbon reduction
       DCRfw: Ar + $Co + {params["dcr_stoi"]} Estar -> Ao + Cr + {params["dcr_stoi"]} E; kcr * Ar * (Estar^{params["dcr_stoi"]})
       DCRbw: Ao + Cr + {params["dcr_stoi"]} E -> Ar + $Co + {params["dcr_stoi"]} Estar; kcrb * Ao * (E^{params["dcr_stoi"]})

       M: Estar -> E; k_m * Estar * Cr

       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});

       // Species initializations:
       P = 1; Ps = 0; Po = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Ar = 0; Ao = 1;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    model scheme_4()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       gam := {gamma_string[0]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]} // this basically controls how much Cr you accumulate
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo: P -> Ps; gam * P
       Ared: Ps + Ao -> Po + Ar; k1 * Ps * Ao
       Dox: $Dr + Po -> Do + P; k2 * Po
       Cred: Ar + Co + {params["estoi"]} Estar -> Ao + Cr + {params["estoi"]} E; kcr * Ar * Co * (Estar^{params["estoi"]})
       M: Estar -> E; k_m * Estar * Cr
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});

       // Species initializations:
       P = 1; Ps = 0; Po = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Ar = 0; Ao = 1;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    model scheme_5a()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       gam := {gamma_string[0]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]} // this basically controls how much Cr you accumulate
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo: P -> Ps; gam * P
       Ared: Ps + Ao + E -> Po + Ar + Estar; k1 * Ps * Ao * E
       Dox: $Dr + Po -> Do + P; k2 * Po
       Cred: Ar + Co + {params["estoi"]} Estar -> Ao + Cr + {params["estoi"]} E; kcr * Ar * Co * (Estar^{params["estoi"]})
       M: Estar -> E; k_m * Estar * Cr
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});

       // Species initializations:
       P = 1; Ps = 0; Po = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Ar = 0; Ao = 1;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    model scheme_5b()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       gam := {gamma_string[0]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]} // this basically controls how much Cr you accumulate
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo: P -> Ps; gam * P
       Ared: Ps + Ao -> Po + Ar; k1 * Ps * Ao
       Dox: $Dr + Po + E -> Do + P + Estar; k2 * Po * E
       Cred: Ar + Co + {params["estoi"]} Estar -> Ao + Cr + {params["estoi"]} E; kcr * Ar * Co * (Estar^{params["estoi"]})
       M: Estar -> E; k_m * Estar * Cr
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});

       // Species initializations:
       P = 1; Ps = 0; Po = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Ar = 0; Ao = 1;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    model scheme_6()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       k3 = {params["k3"]}
       k4 = {params["k4"]}
       gam1 = {gamma_string[0]}
       gam2 = {gamma_string[1]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]}
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo1: P1 -> P1s; gam1 * P1
       Bred: P1s + Bo -> P1o + Br; k1 * P1s * Bo
       Dox: P1o + $Dr -> P1 + Do; k2 * P1o
       
       photo2: P2 -> P2s; gam2 * P2
       Ared: P2s + Ao + E -> P2o + Ar + Estar; k3 * P2s * Ao
       Aox: Ar + P2o -> Ao + P2; k4 * P2o * Ar
       
       M: Estar -> E; k_m * Estar * Cr
       DCRfw: Ar + $Co + {params["dcr_stoi"]} Estar -> Ao + Cr + {params["dcr_stoi"]} E; kcr * Ar * (Estar^{params["dcr_stoi"]})
       DCRbw: Ao + Cr + {params["dcr_stoi"]} E -> Ar + $Co + {params["dcr_stoi"]} Estar; kcrb * Ao * (E^{params["dcr_stoi"]})
       
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});
       

       // Species initializations:
       P1 = 1; P1s = 0; P1o = 0;
       P2 = 1; P2s = 0; P2o = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Ar = 0; Ao = 1;
       Bo = 1; Br = 0;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    model scheme_7a()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       k3 = {params["k3"]}
       k4 = {params["k4"]}
       gam1 = {gamma_string[0]}
       gam2 = {gamma_string[1]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]}
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo1: P1 -> P1s; gam1 * P1
       Ared: P1s + Ao -> P1o + Ar; k1 * P1s * Ao
       Dox: P1o + $Dr -> P1 + Do; k2 * P1o
       
       photo2: P2 -> P2s; gam2 * P2
       Cred: P2s + $Co -> P2o + Cr; k3 * P2s * Co
       Aox: Ar + P2o -> Ao + P2; k4 * P2o * Ar
       
       M: Estar -> E; k_m * Estar * Cr
       
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});
       

       // Species initializations:
       P1 = 1; P1s = 0; P1o = 0;
       P2 = 1; P2s = 0; P2o = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Ar = 0; Ao = 1;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    model scheme_7b()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       k3 = {params["k3"]}
       k4 = {params["k4"]}
       gam1 = {gamma_string[0]}
       gam2 = {gamma_string[1]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]}
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo1: P1 -> P1s; gam1 * P1
       Ared: P1s + Ao + E -> P1o + Ar + Estar; k1 * P1s * Ao * E
       Dox: P1o + $Dr -> P1 + Do; k2 * P1o
       
       photo2: P2 -> P2s; gam2 * P2
       Cred: P2s + $Co -> P2o + Cr; k3 * P2s * Co
       Aox: Ar + P2o -> Ao + P2; k4 * P2o * Ar
       
       M: Estar -> E; k_m * Estar * Cr
       
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});
       

       // Species initializations:
       P1 = 1; P1s = 0; P1o = 0;
       P2 = 1; P2s = 0; P2o = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Ar = 0; Ao = 1;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    model scheme_7c()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       k3 = {params["k3"]}
       k4 = {params["k4"]}
       gam1 = {gamma_string[0]}
       gam2 = {gamma_string[1]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]}
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo1: P1 -> P1s; gam1 * P1
       Ared: P1s + Ao + E -> P1o + Ar + Estar; k1 * P1s * Ao * E
       Dox: P1o + $Dr -> P1 + Do; k2 * P1o
       
       photo2: P2 -> P2s; gam2 * P2
       Cred: P2s + $Co + E -> P2o + Cr + Estar; k3 * P2s * Co
       Aox: Ar + P2o -> Ao + P2; k4 * P2o * Ar
       
       M: Estar -> E; k_m * Estar * Cr
       
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});
       

       // Species initializations:
       P1 = 1; P1s = 0; P1o = 0;
       P2 = 1; P2s = 0; P2o = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Ar = 0; Ao = 1;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    model scheme_7d()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       k3 = {params["k3"]}
       k4 = {params["k4"]}
       gam1 = {gamma_string[0]}
       gam2 = {gamma_string[1]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]}
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo1: P1 -> P1s; gam1 * P1
       Ared: P1s + Ao -> P1o + Ar; k1 * P1s * Ao
       Aoxcyc: P1o + Ar + E -> P1 + Ao + Estar; k2 * P1o * Ar * E
       
       photo2: P2 -> P2s; gam2 * P2
       Cred: P2s + $Co -> P2o + Cr; k3 * P2s * Co
       Dox: $Dr + P2o -> Do + P2; k4 * P2o * Dr
       
       M: Estar -> E; k_m * Estar * Cr
       
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});
       

       // Species initializations:
       P1 = 1; P1s = 0; P1o = 0;
       P2 = 1; P2s = 0; P2o = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Ar = 0; Ao = 1;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    model scheme_8a()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       k3 = {params["k3"]}
       k4 = {params["k4"]}
       gam1 = {gamma_string[0]}
       gam2 = {gamma_string[1]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]}
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo1: P1 -> P1s; gam1 * P1
       Ared: P1s + Ao -> P1o + Ar; k1 * P1s * Ao
       Dox: P1o + $Dr -> P1 + Do; k2 * P1o
       
       photo2: P2 -> P2s; gam2 * P2
       Bred: P2s + Bo -> P2o + Br; k3 * P2s * Bo
       Aox: Ar + P2o -> Ao + P2; k4 * P2o * Ar
       
       M: Estar -> E; k_m * Estar * Cr

       DCRfw: Ar + $Co + {params["dcr_stoi"]} Estar -> Ao + Cr + {params["dcr_stoi"]} E; kcr * Ar * (Estar^{params["dcr_stoi"]})
       DCRbw: Ao + Cr + {params["dcr_stoi"]} E -> Ar + $Co + {params["dcr_stoi"]} Estar; kcrb * Ao * (E^{params["dcr_stoi"]})
       
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});
       

       // Species initializations:
       P1 = 1; P1s = 0; P1o = 0;
       P2 = 1; P2s = 0; P2o = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Ar = 0; Ao = 1;
       Br = 0; Bo = 1;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    model scheme_8b()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       k3 = {params["k3"]}
       k4 = {params["k4"]}
       gam1 = {gamma_string[0]}
       gam2 = {gamma_string[1]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]}
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo1: P1 -> P1s; gam1 * P1
       Ared: P1s + Ao + E -> P1o + Ar + Estar; k1 * P1s * Ao * E
       Dox: P1o + $Dr -> P1 + Do; k2 * P1o
       
       photo2: P2 -> P2s; gam2 * P2
       Bred: P2s + Bo -> P2o + Br; k3 * P2s * Bo
       Aox: Ar + P2o -> Ao + P2; k4 * P2o * Ar
       
       M: Estar -> E; k_m * Estar * Cr

       DCRfw: Ar + $Co + {params["dcr_stoi"]} Estar -> Ao + Cr + {params["dcr_stoi"]} E; kcr * Ar * (Estar^{params["dcr_stoi"]})
       DCRbw: Ao + Cr + {params["dcr_stoi"]} E -> Ar + $Co + {params["dcr_stoi"]} Estar; kcrb * Ao * (E^{params["dcr_stoi"]})
       
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});
       

       // Species initializations:
       P1 = 1; P1s = 0; P1o = 0;
       P2 = 1; P2s = 0; P2o = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Ar = 0; Ao = 1;
       Br = 0; Bo = 1;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    model scheme_8c()
       // Variable initialization:
       k1 = {params["k1"]}
       k2 = {params["k2"]}
       k3 = {params["k3"]}
       k4 = {params["k4"]}
       gam1 = {gamma_string[0]}
       gam2 = {gamma_string[1]}
       k_m = {params["k_maintenance"]}    
       kr = {params["k_resp"]}
       kb = {params["k_resp_back"]}
       kd = {params["k_day"]}
       kn = {params["k_night"]}
       kcr = {params["k_cr"]}
       kcrb = {params["k_cr_back"]}
      
       // Reactions:
       photo1: P1 -> P1s; gam1 * P1
       Ared: P1s + Ao + E -> P1o + Ar + Estar; k1 * P1s * Ao * E
       Dox: P1o + $Dr -> P1 + Do; k2 * P1o
       
       photo2: P2 -> P2s; gam2 * P2
       Bred: P2s + Bo + E -> P2o + Br + Estar; k3 * P2s * Bo * E
       Aox: Ar + P2o -> Ao + P2; k4 * P2o * Ar
       
       M: Estar -> E; k_m * Estar * Cr

       DCRfw: Ar + $Co + {params["dcr_stoi"]} Estar -> Ao + Cr + {params["dcr_stoi"]} E; kcr * Ar * (Estar^{params["dcr_stoi"]})
       DCRbw: Ao + Cr + {params["dcr_stoi"]} E -> Ar + $Co + {params["dcr_stoi"]} Estar; kcrb * Ao * (E^{params["dcr_stoi"]})
       
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});
       

       // Species initializations:
       P1 = 1; P1s = 0; P1o = 0;
       P2 = 1; P2s = 0; P2o = 0;
       Cr = 0; Co = 1; Cstore = 0;
       Ar = 0; Ao = 1;
       Br = 0; Bo = 1;
       Do = 0; Dr = 1;
       Estar = 0; E = 1;
    end

    // pull in whichever scheme you want from above and run that model
    model sim()
        A: scheme_{scheme}()
    end
    """)
    result = r.simulate(0, cycles*period, cycles*period + 1)
    df_file = os.path.join(output_dir, f"{out_pref}_scheme_{scheme}.csv")
    df = pd.DataFrame(result, columns=result.colnames)
    df.to_csv(df_file)

    return r, result, df, out_pref

def simple_plot(result):
    fig, ax = plt.subplots()
    ax.plot(result['time'], result['[A_Cr]'], label='Cr', color='firebrick')
    ax.plot(result['time'], result['[A_Estar]'], label='E*', color='forestgreen')
    ax.plot(result['time'], result['[A_Do]'], label='Dox', color='gold')
    ax.legend()
    plt.show()
    return plt

scheme = "1a"
Tstar = 2343

r, result, df, out_pref = run(scheme, Tstar)
r.plot(result, logy=False, ylim=[0.001, 1.2 * np.max(result[:, 1:])])
# results is a roadrunner.NamedArray not a structured numpy array
# so the relevant object method is colnames not fields
# (check dir(object) to look at all the methods! useful)
plot_file = os.path.join(output_dir, f"{out_pref}_scheme_{scheme}.csv")
