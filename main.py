# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:33:16 2026

@author: cuaml
"""

import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import roadrunner

'''
TO DO:
    - sort out the workflow: i think the intended behaviour here is that
    the user sets which photosynthetic scheme they want to investigate,
    possibly with a set of specific parameters to scan over, and then
    everything else follows from that (Estoi, gamma wavelength, etc.).
    - figure out how the required wavelength, light intensity, etc. all
    fit in here; at the moment they kind of don't affect anything
    - fix docstrings and tidy up code and so on
'''
beta_ev = 0.24 # sort this out properly
e_donor    = -5.10 # in eV
e_acceptor = -3.85 # in eV
estoi = np.floor(-(e_donor - e_acceptor) / beta_ev)
print(f"E* stoichiometry: {estoi}")

scheme = "scheme1"

period = 100
intensity = 10
cycles = 10

k_resp = 1
k_resp_back = 1
k1 = 1
k2 = 1
k_day = 1
k_night = 1
k_maintenance = 1

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
    kr = {k_resp} // this basically controls how much Cr you accumulate
    kb = {k_resp_back} // add detailed balance!!
    
    // Reactions
    fw: Do + Cr + {estoi} E -> $Dr + $Co + {estoi} Estar; kr*Do*Cr*(E^{estoi});
    bw: $Dr + $Co + {estoi} Estar -> Do + Cr + {estoi} E; kb*(Estar^{estoi});
    
    // Initial conditions
    Do = 0; 
    Cr = 0;
    E = 1; Estar = 0;
end

model storage()
    // variables
    gam := square_wave(time, {period}, 1)
    kd = {k_day}
    kn = {k_night}
    
    // reactions
    Cr -> Cstore; kd * Cr
    Cstore -> Cr; kn * (1 - gam) * Cstore
end

model maintenance()
    // variables
    km = {k_maintenance}
    
    //reactions
    Estar -> E; km * Estar
end

model scheme1_photochem()

  // Variable initialization:
  k1 = {k1}
  k2 = {k2}
  //gam := square_wave(time, {period}, {intensity})
  gam = {intensity}
  
  // Reactions:
  photo: P -> Ps; gam * P
  Cred: Ps + $Co -> Po + Cr; k1 * Ps
  Dox: $Dr + Po -> Do + P; k2 * Po

  // Species initializations:
  P = 1; Ps = 0; Po = 0;
  Cr = 0; Co = 1;
  Do = 0; Dr = 1;
  
end

model scheme2_photochem()
  // Variable initialization:
  k1 = {k1}
  k2 = {k2}
  //gam := square_wave(time, {period}, {intensity})
  gam = {intensity}
  
  // Reactions:
  photo: P -> Ps; gam * P
  Cred: Ps + $Co -> Po + Cr + Estar; k1 * Ps
  Dox: $Dr + Po -> Do + P; k2 * Po

  // Species initializations:
  P = 1; Ps = 0; Po = 0;
  Cr = 0; Co = 1;
  Do = 0; Dr = 1;
  Estar = 0;
  
end

model {scheme}()
    P: {scheme}_photochem()
    R: resp()
    M: maintenance()
    //S: storage()
    P.Do is R.Do
    P.Cr is R.Cr
    M.E is R.E
    M.Estar is R.Estar
    //S.Cr is R.Cr
    //M.Cr is R.Cr
end

""")
result = r.simulate(0, cycles*period, cycles*period + 1)
r.plot(result, logy=False, ylim=[0.001, 1.2 * np.max(result[:, 1:])])
'''
fig, ax = plt.subplots()
ax.plot(result['time'], result['[R_Cr]'], label='[R_Cr]', color='r')
ax.plot(result['time'], result['[R_Estar]'], label='[E*]', color='k')
ax.legend()
plt.show()
plt.close()
'''
