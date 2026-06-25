# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:13:16 2026

@author: callum

similar use of tellurium to probe limits (wavelength and intensity)
of oxygenic photosynthesis.
"""

import os
import tellurium as te
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import h, c, e, Boltzmann
import light
import utils

gammas = np.linspace(*utils.xlim, utils.nx)

def gap_calc(dg_gamma, e_donor=utils.dg_h2o,
        e_acceptor=utils.dg_co2, n_steps=5, temperature=273):
    '''
    for given incident light wavelength (in nm), base_rate (s^{-1}), 
    donor and acceptor energies (eV), number of steps and temperature (K),
    calculate the energy gap between adjacent reaction steps in eV
    for a set of n_steps intermediate reaction steps, each representing
    an equal drop in reduction potential, as well as the number of 
    units of energy (equal to the energy drop) required to reduce carbon.
    '''
    dg_ev         = utils.ev_nm(dg_gamma) # energy of photon in ev
    kte           = utils.kbt_ev(temperature) # k_B T for given T
    gap_ev        = dg_ev / n_steps # photon energy split into n_steps
    gap_kt        = gap_ev / kte # energy gap per step in units of kT
    reduction_gap = e_acceptor - (e_donor + gap_ev) # uphill gap for CR
    n             = np.ceil(reduction_gap / gap_ev)
    return gap_kt, n, reduction_gap / kte

def construct_model(gap_kt, n_estar, red_gap_kt, base_rates):
    '''
    use the output of rate_calc to construct an antimony model
    of our system. note that n_steps in that function will have to
    be changed if the structure of this reaction network changes; it's
    not clever enough to recognise if you add or remove a reaction
    '''
    k_recomb   = utils.base_rates["recomb"] * np.exp(-4.0 * gap_kt)
    k_cs_back  = utils.base_rates["trap"]   * np.exp(-1.0 * gap_kt)
    k_red_back = utils.base_rates["red"]    * np.exp(-2.0 * gap_kt)
    k_ox_back  = utils.base_rates["ox"]     * np.exp(-1.0 * gap_kt)
    k_cr_back  = utils.base_rates["red"]    * np.exp(-red_gap_kt)
    nu = 1
    
    '''
    NB: for this if we need multiple photosystems there's probably
    a way to do it such that the model takes parameters
    (gamma, P, Pstar, Ps, Po, Pi, Do, Dr, Ao, Ar, E, Estar)
    and then we can just chain photosystems as
    PS1 : photosystem(gamma1, P1, P1star, P1s, P1o, P1i, Dr, Do, A1r, A1o, E, Estar)
    PS2 : photosystem(gamma2, P2, P2star, P2s, P2o, P2i, A1r, A1o, A2r, A2o, E, Estar)
    PS3 : photosystem(gamma3, P3, P3star, P3s, P3o, P3i, A2r, A2o, A3r, A3o, E, Estar)
    D   : DCR_E(kcr, kcrb, A3r, A3o, Cr, Co, E, Estar)
    or something like that? think about it
    '''
    r = te.loada(f"""

    // dark carbon reduction
    model DCR_E(kcr, kcrb, Ar, Ao, Cr, Co, E, Estar)
        species Ar, Ao, Cr, Co, E, Estar
        fw: Ar + $Co + {n_estar} Estar -> $Ao + Cr + {n_estar} $E; kcr * Ar * (Estar^{n_estar})
        bw: $Ao + Cr + {n_estar} $E -> Ar + $Co + {n_estar} Estar; kcrb * $Ao * Cr * ($E^{n_estar})
    end

    model photoinhibition()
        species
        pi     : Po              -> Pi         ; kpi * Po
        repair : Pi + {nu} Estar -> P + {nu} E ; k_repair * Pi * (Estar^{nu})
    end

    model single(gam1, gam2, k1, k2, k3, k4, km, kr, kb, kd, kn, kcr, kcrb)
        // Variables
        gam1 = 1 // what's this gonna be
        k_decay  = {utils.base_rates["decay"]}
        kcsf     = {utils.base_rates["cs"]}
        kcsb     = {k_cs_back}
        k_recomb = {k_recomb}
        krf      = {utils.base_rates["red"]}
        krb      = {k_red_back}
        kof      = {utils.base_rates["ox"]}
        kob      = {k_ox_back}
        kcr      = {utils.base_rates["red"]}
        kcrb     = {k_cr_back}
        kpi      = {utils.base_rates["inhib"]}

        species P1, P1star, P1s, P1o, Do, Dr, Co, Cr, E, Estar, Ar, Ao;
        photo  : P1               -> P1star           ; gam1 * P1
        decay  : P1star           -> P1               ; k_decay * P1star
        csf    : P1star           -> P1s              ; kcsf * P1star
        csb    : P1s              -> P1star           ; kcsb * P1s
        recomb : P1s              -> P1               ; k_recomb * P1s
        Aredf  : P1s + Ao + E     -> P1o + Ar + Estar ; krf * P1s * Ao * E
        Aredb  : P1o + Ar + Estar -> P1s + Ao + E     ; krb * P1o * Ar * Estar
        Doxf   : $Dr + P1o        -> Do + P1          ; kof * P1o * $Dr
        Doxb   : Do + P1          -> $Dr + P1o        ; kob * P1 * Do
        DCR    : DCR_E(kcr, kcrb, Ar, Ao, Cr, Co, E, Estar)
    end
    """)
    return r
    
if __name__ == "__main__":
    ns    = np.zeros(utils.nx, dtype=int)
    gaps  = np.zeros(utils.nx, dtype=float)
    rates = np.zeros_like(gaps)
    base_rate = 1.
    n_gap = 1.0
    for ii, gg in enumerate(gammas):
        gaps[ii], ns[ii] = gap_calc(gg)
        rates[ii] = base_rate * np.exp(-n_gap * gaps[ii])
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(gammas, ns)
    ax[1].plot(gammas, rates)
    ax[1].set_xlabel(r'$ \Delta G^{\gamma} (\text{nm}) $')
    ax[0].set_ylabel(r'n $ (E^{*}) $')
    ax[1].set_ylabel(r'backward rate $ (\text{s}^{-1}) $')
    ax[1].set_yscale('log')
    ax[0].grid(visible=True)
    ax[1].grid(visible=True)
    plt.show()
    plt.close()
