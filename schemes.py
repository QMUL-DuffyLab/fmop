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
from scipy.constants import h, c, e, Boltzmann
import light
import utils

tmax = 10000
cycles = 10 # only relevant if day/night cycle is turned on
period = tmax // cycles

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
scheme_params = { # reddest usable wavelength of light
    "1a" : { 
        'scheme_to_call': "1a",
        'deltaG_gamma': [utils.ev_nm(611.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 0, # not used for scheme 1
        },
    "1b" : { 
        'scheme_to_call': "1b",
        'deltaG_gamma': [utils.ev_nm(541.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 0,
        },
    "1c" : { 
        'scheme_to_call': "1c",
        'deltaG_gamma': [utils.ev_nm(541.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 0,
        },
    "2a" : { 
        'scheme_to_call': "2a",
        'deltaG_gamma': [utils.ev_nm(611.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 1,
        },
    "2b" : { 
        'scheme_to_call': "2b",
        'deltaG_gamma': [utils.ev_nm(541.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 1,
        },
    "2c" : { 
        'scheme_to_call': "2c",
        'deltaG_gamma': [utils.ev_nm(541.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 1,
        },
    "3a" : { 
        'scheme_to_call': "3a",
        'deltaG_gamma': [utils.ev_nm(700.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 1,
        },
    "3b" : { 
        'scheme_to_call': "3b",
        'deltaG_gamma': [utils.ev_nm(611.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 1,
        },
    "3c" : { 
        'scheme_to_call': "3c",
        'deltaG_gamma': [utils.ev_nm(611.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 1,
        },
    "4a"  : { 
        'scheme_to_call': "3a",
        'deltaG_gamma': [utils.ev_nm(1192.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 4,
        },
    "4b"  : { 
        'scheme_to_call': "3b",
        'deltaG_gamma': [utils.ev_nm(954.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 4,
        },
    "4c"  : { 
        'scheme_to_call': "3c",
        'deltaG_gamma': [utils.ev_nm(954.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 4,
        },
    "4d"  : { 
        'scheme_to_call': "4d",
        'deltaG_gamma': [utils.ev_nm(954.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 4,
        },
    "4e"  : { 
        'scheme_to_call': "4e",
        'deltaG_gamma': [utils.ev_nm(954.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 4,
        },
    "5a"  : { 
        'scheme_to_call': "5a",
        'deltaG_gamma': [utils.ev_nm(1192.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 0,
        },
    "5b"  : { 
        'scheme_to_call': "5b",
        'deltaG_gamma': [utils.ev_nm(1192.0), np.nan],
        'resp_stoi': 5,
        'dcr_stoi': 0,
        },
    }

params = {
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
}

# for plotting steady state values
columns = ['C_red', 'D_ox', 'A_red', 'C_store', 'Estar']
col_legends = [r'$ C_{\text{red}} $', r'$ D_{\text{ox}} $',
               r'$ A_{\text{red}} $', r'$ C_{\text{store}} $',
               r'$ E_{*} $']
idx_s = pd.Index(scheme_params.keys(), name="scheme")
idx_t = pd.Index(light.phz_stars['Tstar'], name="Tstar")
idx_ts = pd.MultiIndex.from_product([idx_t, idx_s])
cols_to_plot = ['C_red', 'Estar']
labels = [r'$ C_{\text{red}} $', r'$ E^{*} $']

def make_outdir(basedir, params): 
    output_dir = os.path.join(basedir,
                f"maintenance_Cred_{params['maintenance_C']}",
                f"{params['intensity']}_intensity")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def run(scheme, Tstar, params=params):
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
    if scheme not in scheme_params.keys():
        raise KeyError(f"Incorrect scheme provided to run(). Scheme = {scheme}, allowed schemes are {scheme_params.keys()}")
    if Tstar not in light.phz_stars['Tstar']:
        raise ValueError(f"Incorrect stellar temperature provided to run(). Tstar = {Tstar}, allowed values are {light.phz_stars['Tstar']}")
    print(f"Ts = {Tstar}, Scheme {scheme}")
    spectrum, out_pref = light.stellar_hz_midpoint(Tstar)
    total_flux = light.micromole_in_region(spectrum,
            utils.xlim[0], utils.ev_nm(dg_p))
    sc_p = scheme_params[scheme]
    scheme_to_call = sc_p["scheme_to_call"]

    output_dir = make_outdir(utils.base_dir, params)
    
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
    
    if params["maintenance_C"]:
        maint_str = " * Cr"
    else:
        maint_str = ""
        
    if params["intensity"] == "high":
        exc_rate *= 10
    elif params["intensity"] == "medium":
        exc_rate *= 1 # for clarity
    elif params["intensity"] == "low":
        exc_rate *= 0.1
    else:
        raise ValueError(f"params['intensity'] value = {params['intensity']} not recognised. Should be low, medium or high")
    
    print(f"Total flux = {total_flux} muE. exc_rate = {exc_rate}")
    if params["light"] == "periodic":
        gamma_string = [f"square_wave(time, {period}, {er})" for er in exc_rate]
    elif params["light"] == "switch":
        gamma_string = [f"switch_off(time, {tmax//2}, {er})" for er in exc_rate]
    elif params["light"] == "constant":
        gamma_string = [f"constant(time, {er})" for er in exc_rate]
    else:
        raise ValueError(f"params['light'] value = {params['light']} not recognised. Should be periodic, switch or constant")

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

    model maintenance(km, E, Estar, Cr)
        species E, Estar, Cr
        Estar -> $E; km * Estar {maint_str}
    end

    model storage(kd, kn, Cr, Cstore)
        species Cr, Cstore
        fw : Cr     -> Cstore ; kd * Cr
        // bw : Cstore -> Cr     ; kn * Cstore
    end

    model bg(kr, kb, km, kd, kn, Dr, Do, Cr, Co, E, Estar, Cstore)
        species Do, Dr, Co, Cr, E, Estar, Cstore
        M     : maintenance(km, $E, Estar, Cr);
        R     : respiration(kr, kb, Do, $Dr, $Co, Cr, $E, Estar)
        S     : storage(kd, kn, Cr, Cstore)
    end
    
    // reversible dark carbon reduction with no energy production
    model DCR_noE(kcr, kcrb, Ar, Ao, Cr, Co)
        species Ar, Ao, Cr, Co
        fw: Ar + $Co -> $Ao + Cr; kcr * Ar
        bw: $Ao + Cr -> Ar + $Co; kcrb * $Ao * Cr
    end

    // dark carbon reduction with energy production
    model DCR_E(kcr, kcrb, Ar, Ao, Cr, Co, E, Estar)
        // this is currently irreversible, per Chris
        // to make reversible just uncomment the backward reaction
        species Ar, Ao, Cr, Co, E, Estar
        fw: Ar + $Co + {sc_p["dcr_stoi"]} Estar -> $Ao + Cr + {sc_p["dcr_stoi"]} $E; kcr * Ar * (Estar^{sc_p["dcr_stoi"]})
        // bw: $Ao + Cr + {sc_p["dcr_stoi"]} $E -> Ar + $Co + {sc_p["dcr_stoi"]} Estar; kcrb * $Ao * Cr * ($E^{sc_p["dcr_stoi"]})
    end

    model photo(gam, k1, k2, P, Ps, Po, Dr, Do, Ar, Ao, E, Estar)
       species P, Ps, Po, Dr, Do, Ar, Ao, E, Estar
       PE   : P        -> Ps      ; gam * P
       Ared : Ps + $Ao -> Po + Ar ; k1 * Ps
       Dox  : $Dr + Po -> Do + P  ; k2 * Po
    end

    model photoEA(gam, k1, k2, P, Ps, Po, Dr, Do, Ar, Ao, E, Estar)
       species P, Ps, Po, Dr, Do, Ao, Ar, E, Estar
       PE   : P             -> Ps              ; gam * P
       Ared : Ps + $Ao + $E -> Po + Ar + Estar ; k1 * Ps
       Dox  : $Dr + Po      -> Do + P          ; k2 * Po
    end

    model photoED(gam, k1, k2, P, Ps, Po, Dr, Do, Ar, Ao, E, Estar)
       species P, Ps, Po, Dr, Do, Ao, Ar, E, Estar
       PE   : P             -> Ps              ; gam * P
       Ared : Ps + $Ao      -> Po + Ar         ; k1 * Ps
       Dox  : $Dr + Po + $E -> Do + P + Estar  ; k2 * Po
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
       P : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
       B : bg(kr, kb, km, kd, kn, $Dr, Do, Cr, $Co, $E, Estar, Cstore)
    end

    model scheme_1b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       P : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
       B : bg(kr, kb, km, kd, kn, $Dr, Do, Cr, $Co, $E, Estar, Cstore)
    end

    model scheme_1c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       P : photoED(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Cr, $Co, $E, Estar)
       B : bg(kr, kb, km, kd, kn, $Dr, Do, Cr, $Co, $E, Estar, Cstore)
    end

    model scheme_2a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       P : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, $Ao, $E, Estar)
       D : DCR_noE(kcr, kcrb, Ar, $Ao, Cr, $Co)
       B : bg(kr, kb, km, kd, kn, $Dr, Do, Cr, $Co, $E, Estar, Cstore)
    end

    model scheme_2b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       P : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, $Ao, $E, Estar)
       D : DCR_noE(kcr, kcrb, Ar, $Ao, Cr, $Co)
       B : bg(kr, kb, km, kd, kn, $Dr, Do, Cr, $Co, $E, Estar, Cstore)
    end

    model scheme_2c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       P : photoED(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, $Ao, $E, Estar)
       D : DCR_noE(kcr, kcrb, Ar, $Ao, Cr, $Co)
       B : bg(kr, kb, km, kd, kn, $Dr, Do, Cr, $Co, $E, Estar, Cstore)
    end

    model scheme_3a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       P : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, $Ao, $E, Estar)
       D : DCR_E(kcr, kcrb, Ar, $Ao, Cr, $Co, $E, Estar)
       B : bg(kr, kb, km, kd, kn, $Dr, Do, Cr, $Co, $E, Estar, Cstore)
    end

    model scheme_3b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       P : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, $Ao, $E, Estar)
       D : DCR_E(kcr, kcrb, Ar, $Ao, Cr, $Co, $E, Estar)
       B : bg(kr, kb, km, kd, kn, $Dr, Do, Cr, $Co, $E, Estar, Cstore)
    end

    model scheme_3c(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       P : photoED(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, $Ao, $E, Estar)
       D : DCR_E(kcr, kcrb, Ar, $Ao, Cr, $Co, $E, Estar)
       B : bg(kr, kb, km, kd, kn, $Dr, Do, Cr, $Co, $E, Estar, Cstore)
    end
    
    model scheme_4d(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       P   : photo(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, $Ao, $E, Estar)
       Aox : Ar + Do        -> $Ao + $Dr       ; k3 * Ar * Do
       D   : DCR_E(kcr, kcrb, Ar, $Ao, Cr, $Co, $E, Estar)
       B   : bg(kr, kb, km, kd, kn, $Dr, Do, Cr, $Co, $E, Estar, Cstore)
    end
    
    model scheme_4e(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
       species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
       P   : photoED(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, $Ao, $E, Estar)
       Aox : Ar + Do        -> $Ao + $Dr       ; k3 * Ar * Do
       D   : DCR_E(kcr, kcrb, Ar, $Ao, Cr, $Co, $E, Estar)
       B   : bg(kr, kb, km, kd, kn, $Dr, Do, Cr, $Co, $E, Estar, Cstore)
    end
    
    model scheme_5a(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
        P : photoEA(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, $Ao, $E, Estar)
        B : bg(kr, kb, km, kd, kn, $Dr, Do, Cr, $Co, $E, Estar, Cstore)
    end
    
    model scheme_5b(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        species P1, P1s, P1o, P2, P2s, P2o, Do, Dr, Co, Cr, Cstore, E, Estar, Ar, Ao, Br, Bo;
        P : photoED(gam1, k1, k2, P1, P1s, P1o, $Dr, Do, Ar, $Ao, $E, Estar)
        B : bg(kr, kb, km, kd, kn, $Dr, Do, Cr, $Co, $E, Estar, Cstore)
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
    """
    r_file = os.path.join(output_dir,
                          f"{out_pref}_scheme_{scheme}_fstr.txt")
    with open(r_file, "w") as f:
        f.write(antstr)
    r = te.loada(antstr)
    r_file = os.path.join(output_dir,
                          f"{out_pref}_scheme_{scheme}_antimony.txt")
    r.exportToAntimony(r_file)
    result = r.simulate(0, cycles*period, cycles*period + 1)
    df_file = os.path.join(output_dir, f"{out_pref}_scheme_{scheme}.csv")
    # results is a roadrunner.NamedArray not a structured numpy array
    # so the relevant object method is colnames not fields
    # (check dir(object) to look at all the methods! useful)
    df = pd.DataFrame(result, columns=result.colnames)
    df.to_csv(df_file)

    return r, result, df, out_pref, output_dir

def simple_plot(result):
    fig, ax = plt.subplots()
    cols = ['[A_Cr]', '[A_Estar]']
    for cn, cl in zip(cols, labels):
        ax.plot(result['time'], result[cn], label=cl)
    ax.legend()
    return fig, ax

def df_plot(df, output_file):
    '''
    plot the relevant columns as defined above from dataframe df
    '''
    fig, ax = plt.subplots()
    for cn, cl in zip(columns, col_legends):
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

def loop(constant, params=params):
    '''
    for a given constant Tstar or scheme, loop over all schemes or Tstars
    and run the simulations, returning a dataframe of steady state results
    '''
    if constant in light.phz_stars['Tstar']:
        loop_var = scheme_params.keys()
        loop_str = "scheme"
    elif constant in scheme_params.keys():
        loop_var = light.phz_stars['Tstar']
        loop_str = "Tstar"
    else:
        raise ValueError("incorrect value {constant} passed to schemes.loop_over")
    data = np.full((len(loop_var), len(columns)), np.nan, dtype=float)
    for ii, vv in enumerate(loop_var):
        try:
            if loop_str == "Tstar":
                r, result, df, out_pref, output_dir = run(constant, vv)
            elif loop_str == "scheme":
                r, result, df, out_pref, output_dir = run(vv, constant)
        except RuntimeError:
            print("Runtime error in r.simulate() for {constant}, {vv}")
            continue
        ss_vals = df.loc[len(df)//2]
        arr = np.array([ss_vals['[A_Cr]'], ss_vals['[A_Do]'], 
                   ss_vals['[A_Ar]'], ss_vals['[A_Cstore]'],
                   ss_vals['[A_Estar]']])
        data[ii, :] = arr
        fig, ax = simple_plot(result)
        if loop_str == "scheme":
            ax.set_title(f"Scheme {vv}")
            plot_file = os.path.join(output_dir,
                        f"{out_pref}_scheme_{vv}.pdf")
            output_file = os.path.join(output_dir,
                    f"{out_pref}_steady_state.csv")
        elif loop_str == "Tstar":
            ax.set_title(f"Tstar {vv}")
            plot_file = os.path.join(output_dir,
                        f"{out_pref}_scheme_{constant}.pdf")
            output_file = os.path.join(output_dir,
                    f"scheme_{constant}_steady_state.csv")
        plt.show()
        fig.savefig(plot_file)
        plt.close()
    df = pd.DataFrame(data, columns = columns, index = idx_s)
    df.to_csv(output_file)
    df_plot(df, output_file)
    return df

def loop_over_all():
    aa = {}
    for Tstar in light.phz_stars['Tstar']:
        aa[Tstar] = loop(Tstar, params)
    df = pd.concat(aa, names=['Tstar', 'scheme'])
    df.set_index(idx_ts)
    output_dir = make_outdir(utils.base_dir, params)
    output_file = os.path.join(output_dir, "all_steady_state.csv")
    df.to_csv(output_file)
    # plot - this is ugly but whatever
    fig, axes = plt.subplots(ncols=len(cols_to_plot),
                             figsize=((10*len(cols_to_plot), 8)))
    for ii, (qq, ll) in enumerate(zip(cols_to_plot, labels)):
        sdf = df[qq]
        for Tstar, new_df in sdf.groupby(level=0):
            ndf = new_df.droplevel(0)
            axes[ii].plot(ndf.index.to_numpy(), ndf.values, label=f"T = {Tstar}K", marker='o')
        axes[ii].set_title(ll)
        axes[ii].legend()
        axes[ii].set_xlabel("scheme")
        axes[0].set_ylabel("value at light off")
    fig.savefig(os.path.splitext(output_file)[0] + ".pdf")
    plt.show()
    plt.close()
    return df

if __name__ == "__main__":
    df = loop_over_all()
