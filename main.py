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
    "3a": [ev_nm(700.0), np.nan],
    "3b": [ev_nm(611.0), np.nan],
    "3c": [ev_nm(611.0), np.nan],
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
    "dcr_stoi"      : 1,
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

    // currently unused
    model storage(kn, kd, Cr, Cstore)
       species Cr, Cstore
       fw: Cr -> Cstore; kd * Cr
       bw: Cstore -> Cr; kn * Cstore
    end

    model maintenance(km, E, Estar, Cr)
        species E, Estar, Cr
        Estar -> E; km * Estar * Cr
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
        fw: Ar + $Co + {params["dcr_stoi"]} Estar -> Ao + Cr + {params["dcr_stoi"]} E; kcr * Ar * (Estar^{params["dcr_stoi"]})
        // bw: Ao + Cr + {params["dcr_stoi"]} E -> Ar + $Co + {params["dcr_stoi"]} Estar; kcrb * Ao * Cr * (E^{params["dcr_stoi"]})
    end

    model respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
       fw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       bw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});
    end
    
    /* NB: every model must have a name of the form `scheme_SCHEMELABEL`,
    * have all of the rates that are specified at the bottom as arguments,
    * in the right order, and include the species definitions as below.
    * This is to minimise the amount of per-scheme code we have to write;
    * it eliminates the need to duplicate the rate declarations and
    * species initialisations for each scheme at the cost of this
    * bit of boilerplate, but I think it's a decent tradeoff. */

    model scheme_1a(gam, k1, k2, k3, k4, km, kr, kb, krc, krcb, kd, kn)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       // List of reactions. form is "Label: Reactants -> Products; Rate"
       photo : P1        -> P1s      ; gam * P1
       Cred  : P1s + $Co -> P1o + Cr ; k1 * P1s
       Dox   : $Dr + P1o -> Do + P1  ; k2 * P1o
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
    end

    model scheme_1b(gam, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1            -> P1s              ; gam * P1
       Cred  : P1s + $Co + E -> P1o + Cr + Estar ; k1 * P1s * E
       Dox   : $Dr + P1o     -> Do + P1          ; k2 * P1o
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
    end

    model scheme_1c(gam, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1            -> P1s             ; gam * P1
       Cred  : P1s + $Co     -> P1o + Cr        ; k1 * P1s
       Dox   : $Dr + P1o + E -> Do + P1 + Estar ; k2 * P1o * E
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
    end

    model scheme_2a(gam, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1        -> P1s      ; gam * P1
       Ared  : P1s + Ao  -> P1o + Ar ; k1 * P1s
       Dox   : $Dr + P1o -> Do + P1  ; k2 * P1o
       D     : DCR_noE(kcr, kcrb, Ar, Ao, Cr, Co)
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
    end

    model scheme_2b(gam, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1           -> P1s              ; gam * P1
       Ared  : P1s + Ao + E -> P1o + Ar + Estar ; k1 * P1s * E
       Dox   : $Dr + P1o    -> Do + P1          ; k2 * P1o
       D     : DCR_noE(kcr, kcrb, Ar, Ao, Cr, Co)
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
    end

    model scheme_2c(gam, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1            -> P1s             ; gam * P1
       Ared  : P1s + Ao      -> P1o + Ar        ; k1 * P1s
       Dox   : $Dr + P1o + E -> Do + P1 + Estar ; k2 * P1o * E
       D     : DCR_noE(kcr, kcrb, Ar, Ao, Cr, Co)
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
    end

    model scheme_3a(gam, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1        -> P1s      ; gam * P1
       Ared  : P1s + Ao  -> P1o + Ar ; k1 * P1s * Ao
       Dox   : $Dr + P1o -> Do + P1  ; k2 * P1o
       D     : DCR_E(kcr, kcrb, Ar, Ao, Cr, Co, E, Estar)
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
    end

    model scheme_3b(gam, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1           -> P1s              ; gam * P1
       Ared  : P1s + Ao + E -> P1o + Ar + Estar ; k1 * P1s * Ao
       Dox   : $Dr + P1o    -> Do + P1          ; k2 * P1o
       D     : DCR_E(kcr, kcrb, Ar, Ao, Cr, Co, E, Estar)
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
    end

    model scheme_3c(gam, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1            -> P1s             ; gam * P1
       Ared  : P1s + Ao      -> P1o + Ar        ; k1 * P1s * Ao
       Dox   : $Dr + P1o + E -> Do + P1 + Estar ; k2 * P1o
       D     : DCR_E(kcr, kcrb, Ar, Ao, Cr, Co, E, Estar)
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
    end

    model scheme_4()
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       // Reactions:
       photo: P1 -> P1s; gam * P1
       Ared: P1s + Ao -> P1o + Ar; k1 * P1s * Ao
       Dox: $Dr + P1o -> Do + P1; k2 * P1o
       Cred: Ar + Co + {params["estoi"]} Estar -> Ao + Cr + {params["estoi"]} E; kcr * Ar * Co * (Estar^{params["estoi"]})
       M: Estar -> E; km * Estar * Cr
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});
    end

    model scheme_5a(gam, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       // Reactions:
       photo: P1 -> P1s; gam * P1
       Ared: P1s + Ao + E -> P1o + Ar + Estar; k1 * P1s * Ao * E
       Dox: $Dr + P1o -> Do + P1; k2 * P1o
       Cred: Ar + Co + {params["estoi"]} Estar -> Ao + Cr + {params["estoi"]} E; kcr * Ar * Co * (Estar^{params["estoi"]})
       M: Estar -> E; km * Estar * Cr
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});
    end

    model scheme_5b(gam, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       // Reactions:
       photo: P1 -> P1s; gam * P1
       Ared: P1s + Ao -> P1o + Ar; k1 * P1s * Ao
       Dox: $Dr + P1o + E -> Do + P1 + Estar; k2 * P1o * E
       Cred: Ar + Co + {params["estoi"]} Estar -> Ao + Cr + {params["estoi"]} E; kcr * Ar * Co * (Estar^{params["estoi"]})
       M: Estar -> E; km * Estar * Cr
       Rfw: Do + Cr + {params["estoi"]} E -> $Dr + $Co + {params["estoi"]} Estar; kr*Do*Cr*(E^{params["estoi"]});
       Rbw: $Dr + $Co + {params["estoi"]} Estar -> Do + Cr + {params["estoi"]} E; kb*(Estar^{params["estoi"]});
    end

    // pull in whichever scheme you want from above and run that model
    model sim()
        // Variable initialization:
        k1 = {params["k1"]}
        k2 = {params["k2"]}
        k3 = {params["k3"]}
        k4 = {params["k4"]}
        gam := {gamma_string[0]}
        km = {params["k_maintenance"]}
        kr = {params["k_resp"]} // this basically controls how much Cr you accumulate
        kb = {params["k_resp_back"]}
        kd = {params["k_day"]}
        kn = {params["k_night"]}
        kcr = {params["k_cr"]}
        kcrb = {params["k_cr_back"]}
        A: scheme_{scheme}(gam, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        // Species initializations:
        A.P1 = 1;    A.P1s = 0; A.P1o = 0;
        A.P2 = 1;    A.P2s = 0; A.P2o = 0;
        A.Cr = 0;    A.Co = 1;  A.Cstore = 0;
        A.Ar = 0;    A.Ao = 1;
        A.Br = 0;    A.Bo = 1;
        A.Do = 0;    A.Dr = 1;
        A.Estar = 0; A.E = 1;
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

scheme = "3a"
Tstar = 2343

r, result, df, out_pref = run(scheme, Tstar)
r.plot(result, logy=False, ylim=[0.001, 1.2 * np.max(result[:, 1:])])
# results is a roadrunner.NamedArray not a structured numpy array
# so the relevant object method is colnames not fields
# (check dir(object) to look at all the methods! useful)
plot_file = os.path.join(output_dir, f"{out_pref}_scheme_{scheme}.csv")
