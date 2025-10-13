from lgwa_response.bilby_interface import LunarLikelihoodBilbyInjection
from lgwa_response.simple_waveforms import from_bilby
import bilby
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils import random
from bilby.core.prior import PriorDict
import logging

# logging.basicConfig(level=logging.INFO)

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
random.seed(123)

label = "two_parameter_regression"
outdir = "outdir"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

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
    "time_at_center": 1200000000.0,
    "chi_1": 0.0,
    "chi_2": 0.0,
    "lambda_1": 0.0,
    "lambda_2": 0.0,
}

priors = PriorDict(
    injection_params.copy()
    | {
        "ra": bilby.core.prior.Uniform(ra - 0.03, ra + 0.03),
        "dec": bilby.core.prior.Uniform(dec - 0.03, dec + 0.03),
    }
)

if __name__ == "__main__":
    freqs = np.geomspace(0.01, 3, num=2000)
    likelihood = LunarLikelihoodBilbyInjection()
    likelihood.compute_center(injection_params["time_at_center"])
    likelihood.make_relbin_data(freqs, from_bilby(injection_params))

    logging.basicConfig(level=logging.INFO)
    print(f"Optimal SNR: {likelihood.optimal_snr(freqs, from_bilby(injection_params))}")
    logging.basicConfig(level=logging.WARNING)

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        npoints=1000,
        injection_parameters=injection_params,
        outdir=outdir,
        label=label,
    )
    result.plot_corner()
