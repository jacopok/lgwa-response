import bilby
import numpy as np

from .likelihood import LunarLikelihood, relbin_log_likelihood_kernel_h_inner_h, relbin_log_likelihood_kernel_d_inner_h
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

        self.phase_marginalization = kwargs.pop('phase_marginalization', False)

        super().__init__(**kwargs)
        
        bilby.core.likelihood.Likelihood.__init__(self)

    def log_likelihood_ratio(self, parameters):
        if self.phase_marginalization:
            return self.relbin_log_likelihood_ratio_phase_marginalized(from_bilby(parameters))
        else:
            return self.relbin_log_likelihood_ratio(from_bilby(parameters))

    def log_likelihood(self, parameters):
        if self.phase_marginalization:
            return self.relbin_log_likelihood_ratio_phase_marginalized(from_bilby(parameters))
        else:
            return self.relbin_log_likelihood_ratio(from_bilby(parameters))

    def noise_log_likelihood(self):
        return 0.0

    def generate_phase_from_marginalized_sample(
            self, parameters):
        r"""
        Generate a single sample from the posterior distribution for phase when
        using a likelihood which explicitly marginalises over phase.

        See Eq. (C29-C32) of https://arxiv.org/abs/1809.02293

        From https://github.com/bilby-dev/bilby/blob/main/bilby/gw/likelihood/base.py#L733-L768
        
        Parameters
        ==========
        signal_polarizations: dict, optional
            Polarizations modes of the template.

        Returns
        =======
        new_phase: float
            Sample from the phase posterior.

        Notes
        =====
        This is only valid when assumes that mu(phi) \propto exp(-2i phi).
        """
        
        f_bin = self.relbin_frequencies

        r_bin = self.projected_waveform(f_bin, parameters) / self.h0_bin

        bin_widths = self.bin_widths
        r0 = (r_bin[:, 1:] + r_bin[:, :-1]) / 2.0
        r1 = (r_bin[:, 1:] - r_bin[:, :-1]) / bin_widths[np.newaxis, :]

        # self.relbin_summary_data has shape [n_channels, n_freqs-1, 4]
        # the last axis contains A0, A1, B0, B1 in this order

        summary_data = self.relbin_summary_data
        
        d_inner_h = relbin_log_likelihood_kernel_d_inner_h(
            r0, r1, np.asarray(summary_data, dtype=complex))
        h_inner_h = relbin_log_likelihood_kernel_h_inner_h(
            r0, r1, np.asarray(summary_data, dtype=complex))
        
        phases = np.linspace(0, 2 * np.pi, 101)
        phasor = np.exp(-2j * phases)
        phase_log_post = d_inner_h * phasor - h_inner_h / 2
        phase_post = np.exp(phase_log_post.real - max(phase_log_post.real))
        new_phase = bilby.core.prior.Interped(phases, phase_post).sample()
        return new_phase
