# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:33:16 2026

@author: callum

this code should eventually take some user-defined parameters
that correspond to one of the possible photosynthetic schemes
we've dreamt up, and then set up a tellurium model for that
scheme and run it. ultimately we want to rank these schemes in some way
and investigate whether terrestrial oxygenic photosynthesis is a "better"
or more robust method than others.

at the moment it does not really do this though, because the schemes
all take different parameters and i haven't sorted out all the details yet
"""

import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import roadrunner
from scipy.constants import h, c, e, Avogadro, Boltzmann
import light

Temperature = 300
hcnm = (h * c) / (1.0E-9)
beta_cm = 1.0 / (Boltzmann * Temperature * (1. / (100.0 * c))) # in wavenumbers
beta_ev = 1.0 / (Boltzmann * Temperature / e) # in eV
ten_kbt = 0.26

Tstar = 2343
spectrum, output_prefix = light.stellar_hz_midpoint(Tstar)

def estar_stoi(dg_donor, dg_acceptor):
    return np.floor(-(dg_donor - dg_acceptor) / beta_ev)

def photon_gap(dg_donor, dg_acceptor, n_steps):
    return dg_acceptor - dg_donor + n_steps * beta_ev

@np.vectorize
def ev_nm(x):
    ''' convert eV to nm and vice-versa '''
    return (hcnm / e) * (1.0 / x)

@np.vectorize
def nm_wvn(l):
    ''' convert nm to wavenumber in cm^{-1} '''
    return 1.0 / (1e-7 * l)

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

xlim = [400.0, 2000.0]
nx = int(xlim[1] - xlim[0])

dg_h2o = -5.10 # in eV
dg_co2 = -3.85 # in eV
dg_a = -4.475 # halfway between h2o and co2, for schemes with one
dg_b = -4.27 # 1/3 between
dg_c = -4.68 # 2/3 between

dg_p = { # limit for the integral to get excitation rate
    "1": [ev_nm(611.0)],
    "2": [ev_nm(541.0)],
    "3": [ev_nm(611.0)],
    "4": [ev_nm(1192.0)],
    "5": [ev_nm(611.0), ev_nm(1192.0)],
    "6": [ev_nm(611.0), ev_nm(1192.0)],
    "7": [ev_nm(611.0), ev_nm(1192.0)],
    "8": [ev_nm(611.0), ev_nm(1192.0)],
    }

dg_gamma = { # reddest usable wavelength of light
    "1": [ev_nm(611.0)],
    "2": [ev_nm(541.0)],
    "3": [ev_nm(611.0)],
    "4": [ev_nm(1192.0)],
    "5": [ev_nm(611.0), ev_nm(1192.0)],
    "6": [ev_nm(611.0), ev_nm(1192.0)],
    "7": [ev_nm(611.0), ev_nm(1192.0)],
    "8": [ev_nm(611.0), ev_nm(1192.0)],
    }

estoi = estar_stoi(dg_h2o, dg_co2)
print(f"E* stoichiometry: {estoi}")

scheme = "1"

period = 100
intensity = 5
cycles = 10

k_resp = 1
k_resp_back = 1
k1 = 1
k2 = 1
k_day = 1
k_night = 1
k_maintenance = 1

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
    /* gam here is just to tell us when it's day or night;
    we do not care about the magnitude, hence magnitude 1 */
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

model scheme_1_photochem()

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

model scheme_2_photochem()
  // Variable initialization:
  k1 = {k1}
  k2 = {k2}
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

model scheme_2b_photochem()
  // Variable initialization:
  k1 = {k1}
  k2 = {k2}
  gam = {intensity}
  
  // Reactions:
  photo: P -> Ps; gam * P
  Cred: Ps + $Co -> Po + Cr; k1 * Ps
  Dox: $Dr + Po + E -> Do + P + Estar; k2 * Po * E

  // Species initializations:
  P = 1; Ps = 0; Po = 0;
  Cr = 0; Co = 1;
  Do = 0; Dr = 1;
  E = 1; Estar = 0;
end

model scheme_3_photochem()
  // Variable initialization:
  k1 = {k1}
  k2 = {k2}
  gam = {intensity}
  
  // Reactions:
  photo: P -> Ps; gam * P
  Ared: Ps + Ao -> Po + Ar; k1 * Ps
  Aox: Ar + Po + E -> Ao + P + Estar; k2 * Po * E

  // Species initializations:
  P = 1; Ps = 0; Po = 0;
  Ar = 0; Ao = 1;
  Do = 0; Dr = 1;
  E = 1; Estar = 0;
end

model scheme_{scheme}()
    P: scheme_{scheme}_photochem()
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
