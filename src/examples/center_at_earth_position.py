from lgwa_response.bilby_interface import LunarLikelihoodBilbyInjection
from lgwa_response.simple_waveforms import from_bilby
import numpy as np
from astropy.time import Time
from astropy.coordinates import ICRS, get_body
import astropy.units as u

mc = 30.0
ra = 3.0
dec = 0.4
injection_params = {
    "chirp_mass": mc,
    "mass_ratio": 0.95,
    "luminosity_distance": 500,
    "theta_jn": 2.5,
    "psi": np.pi / 2,
    "phase": np.pi,
    "ra": ra,
    "dec": dec,
    "time_at_center": 0.0,
    "time_at_center_baseline": 1200000000.0,
    "chi_1": 0.0,
    "chi_2": 0.0,
    "lambda_1": 0.0,
    "lambda_2": 0.0,
}

def arbitrary_grid(f_min, f_max, n_samples, exp=0.0):
    
    if exp == 0:
        return np.geomspace(f_min, f_max, n_samples)
    
    f_min_reduced, f_max_reduced = f_min ** exp, f_max ** exp
    
    scaled_grid = np.linspace(
        f_max_reduced,
        f_min_reduced,
        n_samples,
    )
    
    if exp > 0:
        return scaled_grid ** (1/exp)
    else:
        return scaled_grid[::-1] ** (1/exp)

def get_earth_position(gps_time):
    time = Time(gps_time, format="gps")
    
    earth = get_body('earth', time).transform_to(ICRS())
    earth.representation_type = 'cartesian'
    
    return np.squeeze(
        (earth.x.to(u.m).value, earth.y.to(u.m).value, earth.z.to(u.m).value)
    )

if __name__ == "__main__":

    
    freqs = arbitrary_grid(0.01, 3, n_samples=2000, exp=-0.5)
    likelihood = LunarLikelihoodBilbyInjection()
    likelihood.center = get_earth_position(injection_params["time_at_center_baseline"])

    likelihood.make_relbin_data(freqs, from_bilby(injection_params))