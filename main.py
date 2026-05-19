# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:33:16 2026

@author: callum

TO DO:
    - figure out the workflow. there are various parameters that are
    scheme-dependent (e.g. the photon energy in some schemes, where
    the number of reactions is already fixed), 
    and others that are free even within a scheme (e.g. the redox 
    potentials of intermediate donors or acceptors). How do I code that
    up in a coherent way?
"""

import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import roadrunner
from scipy.constants import h, c, Avogadro, Boltzmann

xlim = [400.0, 2000.0]
nx = int(xlim[1] - xlim[0])

# AU in metres for stellar spectra
AU = 149597870700

Rsol = 6.957E8
Msol = 1.989E30
Lsol = 3.828E26
# (Tstar, Lstar/Lsol, Rstar/Rsol) - 4Gyr isochrones from Isabelle Baraffe
phz_stars = [
 (2343,  -3.58,  0.099),
 (2811,  -3.06,  0.124),
 (3131,  -2.58,  0.174),
 (3416,  -1.97,  0.297),
 (3680,  -1.45,  0.465),
 (4418,  -0.85,  0.646),
 (5297,  -0.29,  0.850),
 (5697,  -0.03,  0.994),
 (6053,   0.43,  1.491)
 ]

def stellar(**kwargs):
    '''
    Generate irradiance for arbitrary star at arbitrary distance.
    kwargs:
    - Tstar:       temperature of star in K
    - Rstar:       radius of star in units of solar radius
    - a:           orbital distance (semi-major axis) in AU
    - attenuation: attenuation factor (should be in [0, 1])
    '''
    try:
        T = kwargs["Tstar"]
        R = kwargs["Rstar"] * Rsol
        a = kwargs["a"]
        att = kwargs["attenuation"]
    except KeyError:
        print("Invalid kwargs in light.stellar()")
        print("Expected: 'Tstar', 'Rstar', 'a', 'attenuation'")
        print(f"Got: {kwargs}")
        raise
    output_prefix = f"stellar_{T:4d}K_a_{a:6.4f}AU"
    a *= AU # convert to metres
    lambdas = np.linspace(*xlim, nx)
    # spectral irradiance (W m^{-2} nm^{-1})
    lambdas *= 1e-9 # convert to SI
    sp = 1.0E-9 * np.pi * ((2.0 * h * c**2 / lambdas**5.0)
    / (np.exp(h * c / (lambdas * Boltzmann * T)) -1.))
    # irradiance at distance a
    irr = (1.0 - att) * sp * (R / a)**2
    return np.column_stack((lambdas * 1e9, irr)), output_prefix

def micromole_in_region(spectrum, lower, upper):
    '''
    calculate the light intensity in micromoles over the wavelength
    range (lower, upper). assumes nm, but since all our spectra are
    given in nm that shouldn't matter. can use this to increase or
    decrease light intensity in a given range as much as we like
    '''
    muM = 0.0
    for row in spectrum:
        if row[0] >= lower and row[0] <= upper:
            e_per_photon = h * c / (row[0] * 1e-9)
            muM += row[1] / e_per_photon
    return 1e6 * (muM / Avogadro)

beta_ev = 0.24 # sort this out properly
e_donor    = -5.10 # in eV
e_acceptor = -3.85 # in eV
estoi = np.floor(-(e_donor - e_acceptor) / beta_ev)
print(f"E* stoichiometry: {estoi}")

scheme = "1"

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
