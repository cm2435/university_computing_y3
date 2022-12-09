import pandas as pd
import numpy as np 
from typing import Optional, List

def fold_lightcurve(time, flux, error, period, found_phase : Optional[int] = 1  verbosebool = False):
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
    data['found_phase_delta'] = data['phase'] - found_phase

    time_to_return = data.sort_values('found_phase_delta')['time'].head(5)

    if verbose: 
        print(data.head(10))

    #Creates the out phase, flux and error
    phase_long = np.concatenate((data['phase'], data['phase'] + 1.0, data['phase'] + 2.0))
    flux_long = np.concatenate((flux, flux, flux))
    err_long = np.concatenate((error, error, error))
    
    return(data['time'], phase_long, flux_long, err_long, time_to_return)


def model_curve(x, d, transit_b, transit_e) -> float: 
    """
    """
    m = (16 * (1-d) / (transit_e - transit_b)**4) * (x - (transit_e+transit_b) / 2)**4 + d
    return m 



if __name__ == "__main__":
    pass
