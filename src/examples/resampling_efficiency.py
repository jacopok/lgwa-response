from lgwa_response.bilby_interface import LunarLikelihoodBilbyInjection
from lgwa_response.simple_waveforms import from_bilby
import bilby
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils import random
from bilby.core.result import Result, get_weights_for_reweighting
from bilby.core.prior import PriorDict

from bilby_inference import priors, injection_params

label = "two_parameter_regression"
outdir = "outdir"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

freqs = np.geomspace(0.01, 3, num=10_000)
likelihood = LunarLikelihoodBilbyInjection()
likelihood.compute_center(injection_params["time_at_center"])
likelihood.make_relbin_data(freqs, from_bilby(injection_params))

result = Result.from_json(outdir + '/two_parameter_regression_result.json')

ln_weights, new_log_likelihood_array, new_log_prior_array, old_log_likelihood_array, old_log_prior_array = get_weights_for_reweighting(result, new_likelihood=likelihood, new_prior=priors)

w = np.exp(ln_weights - ln_weights.max()) 

print(f'Sample efficiency: {w.sum()**2 / (w**2).sum() / len(w)}')
