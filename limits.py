# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:13:16 2026

@author: callum

similar use of tellurium to probe limits (wavelength and intensity)
of oxygenic photosynthesis.
"""

import os
import roadrunner
import tellurium as te
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import h, c, e, Boltzmann
import light
import utils

gammas = np.linspace(*utils.xlim, utils.nx, dtype=int)
output_dir = os.path.join("out", "limits")

def gap_calc(dg_gamma, e_donor=utils.dg_h2o,
        e_acceptor=utils.dg_co2, n_steps=3, temperature=273):
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
    k_cs_back  = utils.base_rates["cs"]  * np.exp(-1.0 * gap_kt)
    k_red_back = utils.base_rates["red"] * np.exp(-1.0 * gap_kt)
    k_cr_back  = utils.base_rates["red"] * np.exp(-red_gap_kt)
    
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
    antstr = f"""

    // dark carbon reduction
    model DCR_E(kcr, kcrb, Dr, Do, Cr, Co, E, Estar)
        species Do, Cr, Estar
        const species Dr, Co, E
        fw: $Dr + $Co + {n_estar} Estar -> Do + Cr + {n_estar} $E; kcr * $Dr * (Estar^{n_estar})
        bw: Do + Cr + {n_estar} $E -> $Dr + $Co + {n_estar} Estar; kcrb * Do * Cr * ($E^{n_estar})
    end

    model photosystem(gam, kd, kcsf, kcsb, krc, kef, keb, P, Pstar, Ps, E, Estar)
        var species P, Pstar, Ps, Estar;
        const species E;
        photo  : P         -> Pstar     ; gam * P
        decay  : Pstar     -> P         ; kd * Pstar
        csf    : Pstar     -> Ps        ; kcsf * Pstar
        csb    : Ps        -> Pstar     ; kcsb * Ps
        recomb : Ps        -> P         ; krc * Ps
        esgf   : Ps + $E   -> P + Estar ; kef * Ps * $E
        esgb   : P + Estar -> Ps + $E   ; keb * P * Estar
    end

    model sim()
        // Variables
        gam  = 1
        kdor = 1
        kcrr = 1
        kesr = 1
        kd   = {utils.base_rates["decay"]}
        kcsf = {utils.base_rates["cs"]}
        krc  = {utils.base_rates["recomb"]}
        kef  = {utils.base_rates["red"]}
        kcr  = {utils.base_rates["red"]}
        kcsb = {k_cs_back} // Ps -> Pstar
        keb  = {k_red_back} // Estar -> E
        kcrb = {k_cr_back} // Cr -> Co

        // common species
        var species Estar
        const species E

        Estar = 0; $E = 1;

        PS1: photosystem(gam, kd, kcsf, kcsb, krc, kef, keb, P, Pstar, Ps, E, Estar)
        DCR: DCR_E(kcr, kcrb, Dr, Do, Cr, Co, E, Estar)
        Doxr   : Do    -> $Dr ; kdor * Do
        Credr  : Cr    -> $Co ; kcrr * Cr
        Esr    : Estar -> $E  ; kesr * Estar

        PS1.P     = 1; PS1.Pstar = 0; PS1.Ps = 0;
        Do        = 0; $Dr       = 1;
        Cr        = 0; $Co       = 1;
    end
    """
    r = te.loada(antstr)
    r.exportToAntimony(os.path.join(output_dir, f"ant_{gg}nm.txt"))
    sbml_str = te.antimonyToSBML(antstr)
    rr = roadrunner.RoadRunner(sbml_str)
    return r, rr

def plot_df(df):
    x     = df['wavelength'].to_numpy()
    estar = df['E* steady state'].to_numpy()
    cr    = df['C_r steady state'].to_numpy()
    esg = estar * df['gap'].to_numpy()
    
    
    fig, ax = plt.subplots()
    ax.plot(x, cr/np.max(cr), label=r'$ C_{\text{red}} $')
    ax.set_xlabel(r'$ \Delta G^{\gamma} (\text{nm}) $')
    ax.set_ylabel(r'$ \left[ C_{\text{red}} \right] $ relative to max' )
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.axvline(x=1300, color='k', ls='--', lw=1.5)
    ax.set_ylim([0.8*np.min(cr),1.1])
    ax.grid(visible=True)
    ax.legend()
    plt.show()
    fig.savefig(os.path.join(output_dir, "c_r.pdf"))
    plt.close()
    
    fig, ax = plt.subplots()
    ax.plot(x, esg, marker='o', label=r'$ E^{*} \Delta G $')
    ax.set_xlabel(r'$ \lambda_p (\text{nm}) $')
    ax.set_ylabel(r'$ \left[E^{*}\right] \Delta G $ (kT)')
    ax.grid(visible=True)
    ax.legend()
    plt.show()
    fig.savefig(os.path.join(output_dir, "estar_gap.pdf"))
    plt.close()
    
    
if __name__ == "__main__":
    do_calc = False
    if do_calc:
        ns    = np.zeros(utils.nx, dtype=int)
        gaps  = np.zeros(utils.nx, dtype=float)
        rgkt  = np.zeros_like(gaps)
        rates = np.zeros_like(gaps)
        cr_ss = np.zeros_like(gaps)
        estar_ss = np.zeros_like(gaps)
        base_rate = 1.
        n_gap = 1.0
        os.makedirs(output_dir, exist_ok=True)
        for ii, gg in enumerate(gammas):
            gaps[ii], ns[ii], rgkt[ii] = gap_calc(gg)
            rates[ii] = base_rate * np.exp(-n_gap * gaps[ii])
            r, rr = construct_model(gaps[ii], ns[ii], rgkt[ii], utils.base_rates)
            '''
            this fails when n_estar > 1, i think, for some reason?
            ss = rr.steadyState()
            rr.steadyStateSelections = ['[Cr]','[Estar]']
            vals = rr.getSteadyStateValues()
            '''
            # hence, fallback to this
            result = r.simulate(0, 100, 101)
            df = pd.DataFrame(result, columns=result.colnames)
            final_vals = df.loc[len(df) - 1]
            fig, ax = utils.simple_plot(result, ['[Cr]','[Estar]'],
                    [r'$ C_{\text{red}} $', r'$ E^{*} $'])
            ax.set_title(r'$ \lambda_p = $' + f"{gg}nm, n = {ns[ii]}")
            plt.show()
            plt.close()
            cr_ss[ii]    = result['[Cr]'][-1]
            estar_ss[ii] = result['[Estar]'][-1]
            print(f"l_p = {gg}, n = {ns[ii]}, final = {final_vals}, cr = {cr_ss[ii]}, e* = {estar_ss[ii]}")
            
        df = pd.DataFrame({
            'wavelength'         : gammas,
            'gap'                : gaps,
            'n'                  : ns,
            'reduction gap (kt)' : rgkt,
            'C_r steady state'   : cr_ss,
            'E* steady state'    : estar_ss,
            })
        df_file = os.path.join(output_dir, "results.csv")
        df.to_csv(df_file, index=False)
        plot_df(df)
    
    else:
        df_file = os.path.join(output_dir, "results.csv")
        df = pd.read_csv(df_file)
        plot_df(df)
