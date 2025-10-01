import bilby

from .likelihood import LunarLikelihood
from .simple_waveforms import from_bilby

DEFAULT_PARAMS = {
    "chirp_mass": None,
    "mass_ratio": None,
    "luminosity_distance": None,
    "theta_jn": None,
    "psi": None,
    "phase": None,
    "ra": None,
    "dec": None,
    "time_at_center": None,
    "chi_1": None,
    "chi_2": None,
    "lambda_1": None,
    "lambda_2": None,
}

class LunarLikelihoodBilbyInjection(LunarLikelihood, bilby.core.likelihood.Likelihood):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        bilby.core.likelihood.Likelihood.__init__(self, parameters=DEFAULT_PARAMS)
    
    def log_likelihood_ratio(self):
        return self.relbin_log_likelihood_ratio(from_bilby(self.parameters))

    def log_likelihood(self):
        return self.relbin_log_likelihood_ratio(from_bilby(self.parameters))
    
    def noise_log_likelihood(self):
        return 0.0