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
import seaborn as sns
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

tmax = 5000
cycles = 10 # only relevant if day/night cycle is turned on
period = tmax // cycles

@np.vectorize
def ev_nm(x):
    ''' convert eV to nm and vice-versa '''
    return (hcnm / e) * (1.0 / x)

dg_h2o = -5.10 # in eV
dg_co2 = -3.85 # in eV
dg_p = dg_co2 - dg_h2o

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
scheme_params = { # reddest usable wavelength of light
    "1a" : { 
        'scheme_to_call': "1a",
        'deltaG_gamma': [ev_nm(611.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 0, # not used for scheme 1
        },
    "1b" : { 
        'scheme_to_call': "1b",
        'deltaG_gamma': [ev_nm(541.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 0,
        },
    "1c" : { 
        'scheme_to_call': "1c",
        'deltaG_gamma': [ev_nm(541.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 0,
        },
    "2a" : { 
        'scheme_to_call': "2a",
        'deltaG_gamma': [ev_nm(611.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 1,
        },
    "2b" : { 
        'scheme_to_call': "2b",
        'deltaG_gamma': [ev_nm(541.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 1,
        },
    "2c" : { 
        'scheme_to_call': "2c",
        'deltaG_gamma': [ev_nm(541.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 1,
        },
    "3a" : { 
        'scheme_to_call': "3a",
        'deltaG_gamma': [ev_nm(700.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 1,
        },
    "3b" : { 
        'scheme_to_call': "3b",
        'deltaG_gamma': [ev_nm(611.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 1,
        },
    "3c" : { 
        'scheme_to_call': "3c",
        'deltaG_gamma': [ev_nm(611.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 1,
        },
    "4a"  : { 
        'scheme_to_call': "3a",
        'deltaG_gamma': [ev_nm(1192.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 4,
        },
    "4b"  : { 
        'scheme_to_call': "3b",
        'deltaG_gamma': [ev_nm(954.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 4,
        },
    "4c"  : { 
        'scheme_to_call': "3c",
        'deltaG_gamma': [ev_nm(954.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 4,
        },
    }

params = {
    "light"         : "switch",
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

def run(scheme, Tstar, params=params):
    '''
    wrapper to set up all the antimony models, pick the one
    you want with the parameters you want, simulate, plot
    the results and output relevant data.
    having one massive f-string is ugly, i know. but because of the
    multiple schemes, the parameters that need to be set at runtime
    and some of the ways that antimony works, this is currently the
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
    if scheme not in scheme_params.keys():
        raise KeyError(f"Incorrect scheme provided to run(). Scheme = {scheme}, allowed schemes are {scheme_params.keys()}")
    if Tstar not in light.phz_stars['Tstar']:
        raise ValueError(f"Incorrect stellar temperature provided to run(). Tstar = {Tstar}, allowed values are {light.phz_stars['Tstar']}")
    print(f"Ts = {Tstar}, Scheme {scheme}")
    spectrum, out_pref = light.stellar_hz_midpoint(Tstar)
    total_flux = light.micromole_in_region(spectrum, xlim[0], ev_nm(dg_p))
    sc_p = scheme_params[scheme]
    scheme_to_call = sc_p["scheme_to_call"]

    exc_rate = []
    for dgg in sc_p['deltaG_gamma']:
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
    model storage(kd, kn, Cr, Cstore)
       species Cr, Cstore
       fw : Cr     -> Cstore ; kd * Cr
       //bw : Cstore -> Cr     ; kn * Cstore
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
        fw: Ar + $Co + {sc_p["dcr_stoi"]} Estar -> Ao + Cr + {sc_p["dcr_stoi"]} E; kcr * Ar * (Estar^{sc_p["dcr_stoi"]})
        // bw: Ao + Cr + {sc_p["dcr_stoi"]} E -> Ar + $Co + {sc_p["dcr_stoi"]} Estar; kcrb * Ao * Cr * (E^{sc_p["dcr_stoi"]})
    end

    model respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
       fw: Do + Cr + {sc_p["resp_stoi"]} E -> $Dr + $Co + {sc_p["resp_stoi"]} Estar; kr*Do*Cr*(E^{sc_p["resp_stoi"]});
       bw: $Dr + $Co + {sc_p["resp_stoi"]} Estar -> Do + Cr + {sc_p["resp_stoi"]} E; kb*(Estar^{sc_p["resp_stoi"]});
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

    model scheme_1a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       // List of reactions. form is "Label: Reactants -> Products; Rate"
       photo : P1        -> P1s      ; gam1 * P1
       Cred  : P1s + $Co -> P1o + Cr ; k1 * P1s
       Dox   : $Dr + P1o -> Do + P1  ; k2 * P1o
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
       S     : storage(kd, kn, C, Cstore)
    end

    model scheme_1b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1            -> P1s              ; gam1 * P1
       Cred  : P1s + $Co + E -> P1o + Cr + Estar ; k1 * P1s * E
       Dox   : $Dr + P1o     -> Do + P1          ; k2 * P1o
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
       S     : storage(kd, kn, C, Cstore)
    end

    model scheme_1c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1            -> P1s             ; gam1 * P1
       Cred  : P1s + $Co     -> P1o + Cr        ; k1 * P1s
       Dox   : $Dr + P1o + E -> Do + P1 + Estar ; k2 * P1o * E
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
       S     : storage(kd, kn, C, Cstore)
    end

    model scheme_2a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1        -> P1s      ; gam1 * P1
       Ared  : P1s + Ao  -> P1o + Ar ; k1 * P1s
       Dox   : $Dr + P1o -> Do + P1  ; k2 * P1o
       D     : DCR_noE(kcr, kcrb, Ar, Ao, Cr, Co)
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
       S     : storage(kd, kn, C, Cstore)
    end

    model scheme_2b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1           -> P1s              ; gam1 * P1
       Ared  : P1s + Ao + E -> P1o + Ar + Estar ; k1 * P1s * E
       Dox   : $Dr + P1o    -> Do + P1          ; k2 * P1o
       D     : DCR_noE(kcr, kcrb, Ar, Ao, Cr, Co)
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
       S     : storage(kd, kn, C, Cstore)
    end

    model scheme_2c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1            -> P1s             ; gam1 * P1
       Ared  : P1s + Ao      -> P1o + Ar        ; k1 * P1s
       Dox   : $Dr + P1o + E -> Do + P1 + Estar ; k2 * P1o * E
       D     : DCR_noE(kcr, kcrb, Ar, Ao, Cr, Co)
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
       S     : storage(kd, kn, C, Cstore)
    end

    model scheme_3a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1        -> P1s      ; gam1 * P1
       Ared  : P1s + Ao  -> P1o + Ar ; k1 * P1s * Ao
       Dox   : $Dr + P1o -> Do + P1  ; k2 * P1o
       D     : DCR_E(kcr, kcrb, Ar, Ao, Cr, Co, E, Estar)
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
       S     : storage(kd, kn, C, Cstore)
    end

    model scheme_3b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1           -> P1s              ; gam1 * P1
       Ared  : P1s + Ao + E -> P1o + Ar + Estar ; k1 * P1s * Ao
       Dox   : $Dr + P1o    -> Do + P1          ; k2 * P1o
       D     : DCR_E(kcr, kcrb, Ar, Ao, Cr, Co, E, Estar)
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
       S     : storage(kd, kn, C, Cstore)
    end

    model scheme_3c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       photo : P1            -> P1s             ; gam1 * P1
       Ared  : P1s + Ao      -> P1o + Ar        ; k1 * P1s * Ao
       Dox   : $Dr + P1o + E -> Do + P1 + Estar ; k2 * P1o
       D     : DCR_E(kcr, kcrb, Ar, Ao, Cr, Co, E, Estar)
       M     : maintenance(km, E, Estar, Cr);
       R     : respiration(kr, kb, Do, Dr, Co, Cr, E, Estar)
       S     : storage(kd, kn, C, Cstore)
    end

    // pull in whichever scheme you want from above and run that model
    model sim()
        // Variable initialization:
        k1 = {params["k1"]}
        k2 = {params["k2"]}
        k3 = {params["k3"]}
        k4 = {params["k4"]}
        gam1 := {gamma_string[0]}
        gam2 := {gamma_string[1]}
        km = {params["k_maintenance"]}
        kr = {params["k_resp"]} // this basically controls how much Cr you accumulate
        kb = {params["k_resp_back"]}
        kd = {params["k_day"]}
        kn = {params["k_night"]}
        kcr = {params["k_cr"]}
        kcrb = {params["k_cr_back"]}
        A: scheme_{scheme_to_call}(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
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
    # results is a roadrunner.NamedArray not a structured numpy array
    # so the relevant object method is colnames not fields
    # (check dir(object) to look at all the methods! useful)
    df = pd.DataFrame(result, columns=result.colnames)
    df.to_csv(df_file)

    return r, result, df, out_pref

def simple_plot(result):
    fig, ax = plt.subplots()
    ax.plot(result['time'], result['[A_Do]'], label='Dox', color='gold')
    ax.plot(result['time'], result['[A_Ar]'], label='Ared', color='black')
    ax.plot(result['time'], result['[A_Cr]'], label='Cr', color='firebrick')
    ax.plot(result['time'], result['[A_Cstore]'], label='Cstore', color='royalblue')
    ax.plot(result['time'], result['[A_Estar]'], label='E*', color='forestgreen')
    ax.legend()
    return fig, ax

def loop_over_all():
    for Tstar in light.phz_stars['Tstar']:
        for scheme in scheme_params.keys():
            try:
                r, result, df, out_pref = run(scheme, Tstar)
            except RuntimeError:
                print("Runtime error in r.simulate() for scheme {scheme}")
                continue
            fig, ax = simple_plot(result)
            plt.show()
            plot_file = os.path.join(output_dir, f"{out_pref}_scheme_{scheme}.pdf")
            fig.savefig(plot_file)
            plt.close()

#loop_over_all()

columns = ['C_red', 'D_ox', 'A_red', 'C_store', 'Estar']
col_legends = [r'$ C_{\text{red}} $', r'$ D_{\text{ox}} $',
               r'$ A_{\text{red}} $', r'$ C_{\text{store}} $',
               r'$ E_{*} $']
idx = pd.Index(scheme_params.keys(), name="scheme")
for Tstar in light.phz_stars['Tstar']:
    spectrum, out_pref = light.stellar_hz_midpoint(Tstar)
    data = np.zeros((len(scheme_params.keys()), len(columns)), dtype=float)
    output_file = os.path.join(output_dir, f"{out_pref}_steady_state.csv")
    for ii, scheme in enumerate(scheme_params.keys()):
        filename = os.path.join(output_dir, f"{out_pref}_scheme_{scheme}.csv")
        try:
            df = pd.read_csv(filename, index_col=0)
        except FileNotFoundError:
            print(f"File {filename} not found. oopsie")
            continue
        ss_vals = df.loc[tmax//2]
        arr = np.array([ss_vals['[A_Cr]'], ss_vals['[A_Do]'], 
                   ss_vals['[A_Ar]'], ss_vals['[A_Cstore]'],
                   ss_vals['[A_Estar]']])
        data[ii, :] = arr
    df = pd.DataFrame(data, columns = columns, index = idx)
    df.to_csv(output_file)
    fig, ax = plt.subplots()
    for cn, cl in zip(columns, col_legends):
        ax.plot(df.index.to_numpy(), df[cn].to_numpy(),
                label=cl, marker='o', ms=5.0)
    ax.legend()
    ax.set_xlabel("Scheme")
    ax.set_ylabel("concentration values")
    plt.grid(visible=True)
    plt.show()
    plot_file = os.path.splitext(output_file)[0] + ".pdf"
    print(plot_file)
    fig.savefig(plot_file)
    plt.close()
