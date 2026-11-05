# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:33:16 2026

@author: callum
"""

import os
import pathlib
import zipfile
import roadrunner
import tellurium as te
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import light
import utils

# print out the intermediate python f-string as a file. useful for debugging
write_fstr = True
# write out the final antimony model as a file. also for debugging
write_sbml = True

dg_p = utils.dg_co2 - utils.dg_h2o
alpha = 15 # how many kBT does an Estar carry?
T = 300 # temp in kelvin
akt = alpha * utils.kbt_ev(T)

params = {
    "one-photon": {
        "opA" : { 
            'deltaG_gamma': 5 * akt + dg_p,
            },
        "opB" : { 
            'deltaG_gamma': 4 * akt + dg_p,
            },
        "opC" : { 
            'deltaG_gamma': 3 * akt + dg_p,
            },
        "opD" : { 
            'deltaG_gamma': 4 * akt,
            },
    },
    "two-photon": {
        "tpA" : { 
            'deltaG_gamma': 0.5 * (8 * akt + dg_p),
            },
        "tpB" : { 
            'deltaG_gamma': 0.5 * (7 * akt + dg_p),
            },
        "tpC" : { 
            'deltaG_gamma': 0.5 * (6 * akt + dg_p),
            },
        "tpD" : { 
           'deltaG_gamma': 0.5 * (8 * akt + dg_p),
           },
        "tpE" : { 
            'deltaG_gamma': 0.5 * (7 * akt + dg_p),
            },
        "tpF" : { 
            'deltaG_gamma': 0.5 * (6 * akt + dg_p),
            },
    },
}

cols_to_log = ['[A_Cr]', '[A_Est]', '[A_Do]', '[A_Ar]', '[A_Qr]']
cols_to_plot = ['[A_Cr]', '[A_Est]']

def make_outdir(basedir, scheme_set, sim_params): 
    output_dir = os.path.join(basedir,
                f"{scheme_set}",
                f"{sim_params['intensity']}_intensity")
    if scheme_set == "coupled":
        output_dir = os.path.join(output_dir,
                            f"phi_{sim_params['phi_lin_cyc']}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def run(scheme, Tstar, scheme_set, sim_params):
    '''
    wrapper to set up all the antimony models, pick the one
    you want with the parameters you want, simulate, plot
    the results and output relevant data.
    might be possible now to save the component models to a separate
    file and load them in? need to figure that out
    
    arguments
    ---------

    `scheme` should be a *string* corresponding to a key in params[scheme_set]
    `Tstar` should match one of the stellar temperatures in light.py,
    which are a bunch of effective stellar temperatures taken from the
    isochrones of Baraffe et al.
    `scheme_set` is which of the dicts of schemes you want to use
    `sim_params` is a dict of the light type, stoichiometries and rates.
    light type should be "periodic", "switch" or "constant"

    outputs
    -------

    NB: haven't fixed what these will be yet
    '''
    scheme_params = params[scheme_set]
    if scheme not in scheme_params.keys():
        raise KeyError(f"Incorrect scheme provided to run(). "
        f"Scheme = {scheme}, allowed schemes are {params.keys()}")
    if Tstar not in light.phz_stars['Tstar']:
        raise ValueError(f"Incorrect stellar temperature provided to run(). "
        f"Tstar = {Tstar}, allowed values are {light.phz_stars['Tstar']}")
    print(f"Ts = {Tstar}, Scheme {scheme}")
    spectrum, out_pref = light.stellar_hz_midpoint(Tstar)
    total_flux = light.micromole_in_region(spectrum,
            utils.xlim[0], utils.ev_nm(dg_p))
    sc_p = scheme_params[scheme]

    output_dir = make_outdir(utils.base_dir, scheme_set, sim_params)
    
    fig, ax = plt.subplots()
    ax.plot(spectrum[:, 0], spectrum[:, 1])
    ax.set_title(f"T = {Tstar}")
    plt.grid(visible=True)
    spec_file = os.path.join(output_dir, f"{out_pref}_spectrum.pdf")
    print(spec_file)
    fig.savefig(spec_file)
    plt.close()
    spec_file = os.path.splitext(spec_file)[0] + ".csv"
    np.savetxt(spec_file, spectrum)

    uf = light.micromole_in_region(spectrum, 
                utils.xlim[0], utils.ev_nm(sc_p['deltaG_gamma']))
    exc_rate = uf/total_flux

    if 'split_lin_cyc' in sc_p:
        pstr = f"A.P1   = {2. * sim_params['phi_lin_cyc']}; " +\
        f"A.P2   = {2. * (1. - sim_params['phi_lin_cyc'])}"
    else:
        pstr = "A.P1   = 1; A.P2   = 1"
        
    if sim_params["intensity"] == "high":
        exc_rate *= 1
    elif sim_params["intensity"] == "medium":
        exc_rate *= 0.1
    elif sim_params["intensity"] == "low":
        exc_rate *= 0.01
    else:
        raise ValueError(f"params['intensity'] value = {params['intensity']} "
                         "not recognised. Should be low, medium or high")
    
    print(f"Total flux = {total_flux} muE. exc_rate = {exc_rate}")
    gamma_string = f"{exc_rate}"

    antstr = f"""
    import "models.txt"

    // wrapper - the loop calls each model in term via {scheme}
    model sim()
        // Variable initialization:
        gam1 := {gamma_string}
        gam2 := {gamma_string}
        A: {scheme}(gam1, gam2)
        // Species initializations:
        var species A.P1, A.P1s, A.P1st, A.P1o, A.P2, A.P2s, A.P2st, A.P2o;
        var species A.Cr, A.Ao, A.Ar, A.Qr, A.Qo, A.Do, A.Est;
        const species A.Co, A.Dr, A.E

        {pstr}
        A.P1s  = 0; A.P2s  = 0;
        A.P1st = 0; A.P2st = 0;
        A.P1o  = 0; A.P2o  = 0;
        A.Cr   = 0; A.Co   = 1;
        A.Ar   = 0; A.Ao   = 1;
        A.Qr   = 0; A.Qo   = 1;
        A.Do   = 0; A.Dr   = 1;
        A.Est  = 0; A.E    = 1;
    end
    """
    output_files = []
    if write_fstr:
        r_file = os.path.join(output_dir,
                          f"{out_pref}_scheme_{scheme}_fstr.txt")
        output_files.append(r_file)
        with open(r_file, "w") as f:
            f.write(antstr)
    sbml_str = te.antimonyToSBML(antstr)
    
    if write_sbml:
        sbml_file = os.path.join(output_dir,
                          f"{out_pref}_scheme_{scheme}_SBML_initial.txt")
        output_files.append(sbml_file)
        with open(sbml_file, "w") as ff:
            ff.write(sbml_str)
    
    rr = roadrunner.RoadRunner(sbml_str)
    ss = rr.steadyState()
    vals = rr.getSteadyStateValuesNamedArray()
    d = {}
    for cc in vals.colnames:
        d[cc] = vals[cc][0] # vals[cc] returns a len(1) array
    d['sum_squared_residuals'] = ss
    d['gamma'] = exc_rate
    print(d)
    
    if write_sbml:
        sbml_file = os.path.join(output_dir,
                          f"{out_pref}_scheme_{scheme}_SBML_final.txt")
        output_files.append(sbml_file)
        with open(sbml_file, "w") as ff:
            ff.write(rr.getCurrentSBML())
            
    return d, out_pref, output_files

def df_plot(df, columns, output_file):
    '''
    plot the relevant columns as defined above from dataframe df
    '''
    fig, ax = plt.subplots()
    for cn, cl in zip(columns, [utils.get_latex(cc) for cc in columns]):
        ax.plot(df.index.to_numpy(), df[cn].to_numpy(),
                label=cl, marker='o', ms=5.0)
    ax.legend()
    ax.set_xlabel(df.index.name)
    ax.set_ylabel("concentration")
    ax.set_yscale('log')
    ax.set_ylim([1e-5, 1.0])
    plt.grid(visible=True)
    plt.show()
    plot_file = os.path.splitext(output_file)[0] + ".pdf"
    fig.savefig(plot_file)
    plt.close()

def loop(constant, scheme_set, sim_params):
    '''
    for a given constant Tstar or scheme, loop over all schemes or Tstars
    and run the simulations, returning a dataframe of steady state results
    '''
    scheme_params = params[scheme_set]
    output_dir = make_outdir(utils.base_dir, scheme_set, sim_params)
    output_files = []
    if constant in light.phz_stars['Tstar']:
        loop_var = scheme_params.keys()
        loop_str = "scheme"
        idx = pd.Index(scheme_params.keys(), name="scheme")
    elif constant in scheme_params.keys():
        loop_var = light.phz_stars['Tstar']
        loop_str = "Tstar"
        idx = pd.Index(light.phz_stars['Tstar'], name="Tstar")
    else:
        raise ValueError("incorrect value {constant} passed "
        "to schemes.loop_over")
    data = []
    for vv in loop_var:
        try:
            if loop_str == "Tstar":
                res_dict, out_pref, ofs = run(constant, vv, scheme_set,
                                              sim_params)
            elif loop_str == "scheme":
                res_dict, out_pref, ofs = run(vv, constant, scheme_set,
                                              sim_params)
            output_files.extend(ofs)
        except RuntimeError:
            print(f"Runtime error in r.simulate() for {constant}, {vv}")
            raise
        data.append(res_dict)

    if loop_str == "scheme":
        output_file = os.path.join(output_dir,
                                   f"{out_pref}_steady_state.csv")
    else:
        output_file = os.path.join(output_dir,
                                   f"scheme_{constant}_steady_state.csv")

    df = pd.DataFrame(data, index = idx)
    df.to_csv(output_file)
    df_plot(df, cols_to_log, output_file)
    return df, output_files

def loop_over_all(scheme_set, sim_params):
    scheme_params = params[scheme_set]
    aa = {}
    output_files = []
    for Tstar in light.phz_stars['Tstar']:
        aa[Tstar], ofs = loop(Tstar, scheme_set, sim_params)
        output_files.extend(ofs)
    df = pd.concat(aa, names=['Tstar', 'scheme'])
    idx_s = pd.Index(scheme_params.keys(), name="scheme")
    idx_t = pd.Index(light.phz_stars['Tstar'], name="Tstar")
    idx_ts = pd.MultiIndex.from_product([idx_t, idx_s])
    df.set_index(idx_ts)
    output_dir = make_outdir(utils.base_dir, scheme_set, sim_params)
    output_file = os.path.join(output_dir, "all_steady_state.csv")
    df.to_csv(output_file)
    
    # zip up other output files
    zipfilename = os.path.join(output_dir, "other_files.zip")
    with zipfile.ZipFile(zipfilename, mode="w",
            compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in output_files:
            if os.path.isfile(filename):
                archive.write(filename, arcname = os.path.basename(filename))
    # delete the original files to save space
    for filename in output_files:
        pathlib.Path.unlink(filename, missing_ok = True)
    # plot - this is ugly but whatever
    fig, axes = plt.subplots(ncols=len(cols_to_plot),
                             figsize=((10*len(cols_to_plot), 8)))
    for ii, qq in enumerate(cols_to_plot):
        sdf = df[qq]
        for Tstar, new_df in sdf.groupby(level=0):
            ndf = new_df.droplevel(0)
            axes[ii].plot(ndf.index.to_numpy(), ndf.values,
                          label=f"T = {Tstar}K", marker='o')
        axes[ii].set_title(utils.get_latex(qq))
        axes[ii].legend()
        axes[ii].set_xlabel("scheme")
        axes[0].set_ylabel("steady-state concentration")
        axes[ii].grid(visible=True)
    title = f"{scheme_set} schemes, {simulation_params['intensity']} intensity"
    if scheme_set == "coupled":
        title += r', $ \phi $ = ' + f"{simulation_params['phi_lin_cyc']}"
    plt.suptitle(title)
    fig.savefig(os.path.splitext(output_file)[0] + ".pdf")
    plt.show()
    plt.close()
    return df

if __name__ == "__main__":

    scheme_set = "two-photon"
    simulation_params = {
        "intensity"     : "high",
        "phi_lin_cyc"   : 0.5, # only used for two-photon schemes
    }

    df = loop_over_all(scheme_set, simulation_params)
