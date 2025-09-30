import numpy as np
from .likelihood import LunarLikelihood, noise_weighted_inner_product


class LunarLikelihoodAddingSamples(LunarLikelihood):

    def add_relbin_frequency(self, i_bin, n_local_grid=2**10):

        f1, f2 = self.relbin_frequencies[i_bin], self.relbin_frequencies[i_bin + 1]

        f_mid = (f1 + f2) / 2.0
        freqs = np.asarray([f1, f_mid, f2])

        self.relbin_summary_data = np.insert(
            self.relbin_summary_data, i_bin + 1, np.zeros((2, 4)), axis=1
        )
        self.relbin_frequencies = np.insert(self.relbin_frequencies, i_bin + 1, f_mid)
        # compute h0 at three frequencies to allow for t_of_f computation by phase differentiation
        self.h0_bin = np.insert(
            self.h0_bin,
            i_bin + 1,
            self.projected_waveform(freqs, self.h0_parameters)[:, 1],
            axis=1,
        )

        for i, (f_left, f_right) in enumerate(zip(freqs[:-1], freqs[1:])):
            f_avg = (f_right + f_left) / 2.0
            local_grid = np.geomspace(f_left, f_right, num=n_local_grid)
            local_psd = self.psd(local_grid)
            local_h0 = self.projected_waveform(local_grid, self.h0_parameters)
            for channel in range(2):

                A0 = noise_weighted_inner_product(
                    local_h0[channel], local_h0[channel], local_psd, local_grid
                )
                A1 = noise_weighted_inner_product(
                    local_h0[channel],
                    local_h0[channel] * (local_grid - f_avg),
                    local_psd,
                    local_grid,
                )

                self.relbin_summary_data[channel, i + i_bin, 0] = A0
                self.relbin_summary_data[channel, i + i_bin, 1] = A1
                self.relbin_summary_data[channel, i + i_bin, 2] = A0
                self.relbin_summary_data[channel, i + i_bin, 3] = A1

    def relbin_log_likelihood_error(self, parameters, n_midpoints=1, n_to_return=1):
        f_bin = self.relbin_frequencies
        f_mid = np.empty((self.n_bins, n_midpoints))
        for i in range(self.n_bins):
            f_mid[i] = np.linspace(f_bin[i], f_bin[i + 1], num=n_midpoints + 2)[1:-1]

        r_bin = self.projected_waveform(f_bin, parameters) / self.h0_bin
        r_mid = (
            self.projected_waveform(f_mid.flatten(), parameters)
            / self.projected_waveform(f_mid.flatten(), self.h0_parameters)
        ).reshape((2,) + f_mid.shape)

        # self.relbin_summary_data has shape [n_channels, n_freqs-1, 4]
        # the last axis contains A0, A1, B0, B1 in this order

        error, error_indices = relbin_log_likelihood_error_kernel(
            f_bin,
            f_mid,
            r_bin,
            r_mid,
            np.asarray(self.relbin_summary_data, dtype=complex),
            n_to_return,
        )
        indices = list(set(error_indices.flatten()))
        return error, indices
