# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:33:16 2026

@author: cuaml
"""

import tellurium as te
import numpy as np
import roadrunner

beta_ev = 0.24 # sort this out properly
e_donor    = -5.10 # in eV
e_acceptor = -3.85 # in eV
estoi = np.floor(-(e_donor + e_acceptor) * beta_ev)
print(estoi)

period = 1000
intensity = 0.7
cycles = 20

def detailed_balance(e1, e2, k12, k21):
    '''
    apply a Boltzmann penalty to an uphill rate, leaving the
    downhill rate unchanged. check this and also Beta definition
    '''
    fac = np.exp(beta_ev * (e2 - e1))
    if e1 > e2:
        k21 *= fac
    elif e2 > e1:
        k12 *= (1. / fac)
    return k12, k21

r = te.loada(f"""
function square_wave(t, period, intensity)
  intensity * (1 - (floor(t / period) - 2 * floor(floor(t / period) / 2)))
end

model resp()
    // Variables
    kr = 0.1
    kb = 0 // add detailed balance!!
    
    // Reactions
    fw: Do + Cr + {estoi} E -> $Dr + $Co + {estoi} Estar; kr*Do*Cr*(E^{estoi});
    bw: $Dr + $Co + {estoi} Estar -> Do + Cr + {estoi} E; kb*(Estar^{estoi});
    
    // Initial conditions
    Do = 0; 
    Cr = 0;
    E = 1; Estar = 0;
end

model scheme1_photochem()

  // Variable initialization:
  k1 = 1
  k2 = 1
  gam := square_wave(time, {period}, {intensity})
  
  // Reactions:
  photo: P -> Ps; gam * P
  Cred: Ps + $Co -> Po + Cr; k1 * Ps
  Dox: $Dr + Po -> Do + P; k2 * Po

  // Species initializations:
  P = 1; Ps = 0; Po = 0;
  Cr = 0; Co = 1;
  Do = 0; Dr = 1;
  
end

model storage()
    // variables
    gam := square_wave(time, {period}, 1)
    kd = 1
    kn = 1/{period}
    
    // reactions
    Cr -> Cstore; kd * Cr
    Cstore -> Cr; kn * (1 - gam) * Cstore
end

model maintenance()
    // variables
    km = 1
    
    //reactions
    Estar -> E; km * Estar * Cstore
end

model scheme1()
    P: scheme1_photochem()
    R: resp()
    M: maintenance()
    S: storage()
    P.Do is R.Do
    P.Cr is R.Cr
    M.E is R.E
    M.Estar is R.Estar
    S.Cr is R.Cr
    M.Cstore is S.Cstore
end

""")
result = r.simulate(0, cycles*period, cycles*period + 1)
r.plot(result, logy=True, ylim=[0.0001, 100])
