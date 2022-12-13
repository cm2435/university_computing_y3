import pandas as pd
import numpy as np 
from typing import Optional, List

import pymc3 as pm
import aesara_theano_fallback.tensor as tt

import pymc3_ext as pmx
from celerite2.theano import terms, GaussianProcess
import exoplanet as xo 


def fold_lightcurve(time, flux, error, period, verbose: bool = False):
    """
    Folds the lightcurve given a period.
    time: input time (same unit as period)
    flux: input flux
    error: input error
    period: period to be folded to, needs to same unit as time (i.e. days)
    returns: phase, folded flux, folded error
    """
    #Create a pandats dataframe from the 
    data = pd.DataFrame({'time': time, 'flux': flux, 'error': error})
    #create the phase 
    data['phase'] = data.apply(lambda x: ((x.time/ period) - np.floor(x.time / period)), axis=1)
    if verbose: 
        print(data.head(10))

    #Creates the out phase, flux and error
    phase_long = np.concatenate((data['phase'], data['phase'] + 1.0, data['phase'] + 2.0))
    flux_long = np.concatenate((flux, flux, flux))
    err_long = np.concatenate((error, error, error))
    
    return(data['time'], phase_long, flux_long, err_long)


def model_curve(x, d, transit_b, transit_e) -> float: 
    """
    Fit a qu
    """
    m = (16 * (1-d) / (transit_e - transit_b)**4) * (x - (transit_e+transit_b) / 2)**4 + d
    return m 


def fit_gp_model(x : np.ndarray, y : np.ndarray, yerr : np.ndarray, period_guess : float, t0_guess : float):
    depth_guess = 0.0057
    with pm.Model() as model:

        # Stellar parameters
        mean = pm.Normal("mean", mu=1.002, sigma=1)
        u = xo.QuadLimbDark("u")
        star_params = [mean, u]

        # Gaussian process noise model
        sigma = pm.InverseGamma("sigma", alpha=3.0, beta=2 * np.median(yerr))
        log_sigma_gp = pm.Normal("log_sigma_gp", mu=1.0, sigma=1)
        log_rho_gp = pm.Normal("log_rho_gp", mu=np.log(10.0), sigma=1)
        kernel = terms.SHOTerm(
            sigma=tt.exp(log_sigma_gp), rho=tt.exp(log_rho_gp), Q=1.0 / 3
        )
        noise_params = [sigma, log_sigma_gp, log_rho_gp]

        # Planet parameters
        log_ror = pm.Normal(
            "log_ror", mu=0.5 * np.log(depth_guess * 1e-3), sigma=1
        )
        ror = pm.Deterministic("ror", tt.exp(log_ror))

        # Orbital parameters
        log_period = pm.Normal("log_period", mu=np.log(period_guess), sigma=1.0)
        period = pm.Deterministic("period", tt.exp(log_period))
        t0 = pm.Normal("t0", mu=t0_guess, sigma=1.0)
        log_dur = pm.Normal("log_dur", mu=np.log(0.18), sigma=1)
        dur = pm.Deterministic("dur", tt.exp(log_dur))
        b = xo.distributions.ImpactParameter("b", ror=ror)

        # Set up the orbit
        orbit = xo.orbits.KeplerianOrbit(
            period=period, 
            duration=dur, 
            t0=t0, 
            b=b,
            )

        # We're going to track the implied density for reasons that will become clear later
        pm.Deterministic("rho_circ", orbit.rho_star)

        # Set up the mean transit model
        star = xo.LimbDarkLightCurve(u)
        lc_model = mean + 1e3 * tt.sum(
            star.get_light_curve(orbit=orbit, r=ror, t=x), axis=-1
        )

        # Finally the GP observation model
        gp = GaussianProcess(kernel, t=x, diag=yerr**2 + sigma**2)
        gp.marginal("obs", observed=y - lc_model)

        # Double check that everything looks good - we shouldn't see any NaNs!
        print(model.check_test_point())

        # Optimize the model
        map_soln = model.test_point
        map_soln = pmx.optimize(map_soln, [sigma])
        map_soln = pmx.optimize(map_soln, [ror, b, dur])
        map_soln = pmx.optimize(map_soln, noise_params)
        map_soln = pmx.optimize(map_soln, star_params)
        map_soln = pmx.optimize(map_soln)

    return model, map_soln, gp, lc_model

def fit_gp_model(
    x : np.ndarray, 
    y : np.ndarray, 
    yerr : np.ndarray, 
    period_guess : float, 
    t0_guess : float,
    depth_guess : float  = 0.0057):

    with pm.Model() as model_reduced:
        
        # Stellar parameters
        mean = pm.Normal("mean", mu=1.0, sigma=0.001)
        u = xo.QuadLimbDark("u")
        star_params = [mean, u]

        # Planet parameters
        log_ror = pm.Normal(
            "log_ror", mu=0.5 * np.log(depth_guess * 1e-3), sigma=1
        )
        ror = pm.Deterministic("ror", tt.exp(log_ror))

        # Orbital parameters
        log_period = pm.Normal("log_period", mu=np.log(period_guess), sigma=1)
        period = pm.Deterministic("period", tt.exp(log_period))
        t0 = pm.Normal("t0", mu=t0_guess, sigma=1.0)
        log_dur = pm.Normal("log_dur", mu=np.log(0.18), sigma=1)
        dur = pm.Deterministic("dur", tt.exp(log_dur))
        b = xo.distributions.ImpactParameter("b", ror=ror)

        # Set up the orbit
        orbit = xo.orbits.KeplerianOrbit(
            period=period, 
            duration=dur, 
            t0=t0, 
            b=b,
            )

        # We're going to track the implied density for reasons that will become clear later
        pm.Deterministic("rho_circ", orbit.rho_star)

        # Set up the mean transit model
        star = xo.LimbDarkLightCurve(u)
        lc_model = mean + 1e3 * tt.sum(
            star.get_light_curve(orbit=orbit, r=ror, t=x), axis=-1
        )


        # Double check that everything looks good - we shouldn't see any NaNs!
        print(model_reduced.check_test_point())

        # Optimize the model
        map_soln = model_reduced.test_point
        map_soln = pmx.optimize(map_soln, [ror, b, dur])
        map_soln = pmx.optimize(map_soln, star_params)
        map_soln = pmx.optimize(map_soln)

    return model_reduced, map_soln, lc_model


if __name__ == "__main__":
    pass
