from .lunar_coordinates import (
    generate_data_position,
    generate_data_response,
    spherical_to_cartesian,
    wave_frame_basis_cartesian,
)
from .simple_waveforms import Phif3hPN, Af3hPN, time_to_merger
from . import data_path
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from tqdm import tqdm
import yaml
from numba import njit, types, float64, complex128, int64
from pathlib import Path
import logging

SPEED_OF_LIGHT = 299792458.0
DETECTOR_LIFETIME = 10 * 3.16e7


@njit(
    float64[:, :](
        float64[:],
        float64[:],
        float64[:, :],
    )
)
def interp_position(times, stored_times, stored_positions):

    result = np.empty(times.shape + (3,))
    result[:, 0] = np.interp(times, stored_times, stored_positions[:, 0])
    result[:, 1] = np.interp(times, stored_times, stored_positions[:, 1])
    result[:, 2] = np.interp(times, stored_times, stored_positions[:, 2])

    return result


@njit(
    float64[:, :](
        float64[:],
        float64[:],
        float64[:, :],
    )
)
def interpolate_detector_frame(times, stored_times, stored_response):

    res = np.empty((6,) + times.shape)

    res[0] = np.interp(times, stored_times, stored_response[:, 0])
    res[1] = np.interp(times, stored_times, stored_response[:, 1])
    res[2] = np.interp(times, stored_times, stored_response[:, 2])
    res[3] = np.interp(times, stored_times, stored_response[:, 3])
    res[4] = np.interp(times, stored_times, stored_response[:, 4])
    res[5] = np.interp(times, stored_times, stored_response[:, 5])

    # n_ra, n_dec, x_ra, x_dec, y_ra, y_dec
    return res


@njit(
    float64(
        complex128[:, :],
        complex128[:, :],
        complex128[:, :, :],
    )
)
def relbin_log_likelihood_kernel(r0, r1, summary_data):

    ll_total = 0
    for channel in range(2):
        for i_bin in range(summary_data.shape[1]):

            ll_hd = np.real(
                summary_data[channel, i_bin, 0] * np.conj(r0[channel, i_bin])
                + summary_data[channel, i_bin, 1] * np.conj(r1[channel, i_bin])
            )

            ll_hh = np.real(
                summary_data[channel, i_bin, 2] * np.abs(r0[channel, i_bin]) ** 2
                + 2
                * summary_data[channel, i_bin, 3]
                * np.real(r0[channel, i_bin] * np.conj(r1[channel, i_bin]))
            )

            ll_total += ll_hd - 0.5 * ll_hh

    return ll_total


@njit(
    types.Tuple((float64, int64[:]))(
        float64[:],
        float64[:, :],
        complex128[:, :],
        complex128[:, :, :],
        complex128[:, :, :],
        int64,
    )
)
def relbin_log_likelihood_error_kernel(
    f_bin, f_mid, r_bin, r_mid, summary_data, n_to_return
):

    ll_error_total = 0
    n_mid = f_mid.shape[1]

    max_error_bin = np.zeros(r_bin.shape[0], dtype=int64)
    errors = np.empty((r_bin.shape[0], summary_data.shape[1]), dtype=complex128)

    for channel in range(r_bin.shape[0]):

        for i_bin in range(summary_data.shape[1]):

            r0_estimate = (r_bin[channel, i_bin + 1] + r_bin[channel, i_bin]) / 2.0
            r1_estimate = (r_bin[channel, i_bin + 1] - r_bin[channel, i_bin]) / (
                f_bin[i_bin + 1] - f_bin[i_bin]
            )

            # the first index is the original one
            # (0=constant, 1=linear)
            # the second index denotes the part of the bin we are considering

            r0j_estimate = np.empty(n_mid + 1, dtype=complex128)
            r1j_estimate = np.empty(n_mid + 1, dtype=complex128)

            r0j_estimate[0] = (r_mid[channel, i_bin, 0] + r_bin[channel, i_bin]) / 2.0
            r0j_estimate[n_mid] = (
                r_bin[channel, i_bin + 1] + r_mid[channel, i_bin, n_mid - 1]
            ) / 2.0
            r1j_estimate[0] = (r_mid[channel, i_bin, 0] - r_bin[channel, i_bin]) / (
                f_mid[i_bin, 0] - f_bin[i_bin]
            )
            r1j_estimate[n_mid] = (
                r_bin[channel, i_bin + 1] - r_mid[channel, i_bin, n_mid - 1]
            ) / (f_bin[i_bin + 1] - f_mid[i_bin, n_mid - 1])

            for j in range(1, n_mid):
                r0j_estimate[j] = (
                    r_mid[channel, i_bin, j] + r_mid[channel, i_bin, j - 1]
                ) / 2.0
                r1j_estimate[j] = (
                    r_mid[channel, i_bin, j] - r_mid[channel, i_bin, j - 1]
                ) / (f_mid[i_bin, j] - f_mid[i_bin, j - 1])

            error_this_bin = 0.0

            for j in range(n_mid + 1):

                r0j_error = r0j_estimate[j] - r0_estimate
                r1j_error = r1j_estimate[j] - r1_estimate

                # dh term, r0 part
                error_this_bin += np.real(
                    summary_data[channel, i_bin, 0] * np.conj(r0j_error)
                )
                # dh term, r1 part
                error_this_bin += np.real(
                    (summary_data[channel, i_bin, 1]) * np.conj(r1j_error)
                )
                # hh term, r0 part
                # we do error propagation on the square of r0
                # we need to remember  the 1/2 in front of l_hh
                error_this_bin -= (
                    0.5
                    * summary_data[channel, i_bin, 2]
                    * np.real(
                        (
                            r0j_estimate[j] * np.conj(r0j_error)
                            + r0j_error * np.conj(r0j_estimate[j])
                        )
                    )
                )
                # hh term, r1 part
                # the factor 2 in the term cancels with the 1/2 in front of l_hh
                error_this_bin -= summary_data[channel, i_bin, 3] * np.real(
                    r0j_estimate[j] * np.conj(r1j_error)
                    + r0j_error * np.conj(r1j_estimate[j])
                )
            errors[channel, i_bin] = error_this_bin

    # get the indices for which all channels are badly represented
    error_indices = np.argsort(-np.abs(np.sum(errors, axis=0)))[:n_to_return]

    return abs(np.sum(errors)) / (n_mid + 1), error_indices


def noise_weighted_inner_product(aa, bb, power_spectral_density, frequencies):
    """
    Calculate the noise weighted inner product between two arrays.

    Parameters
    ==========
    aa: array_like
        Array to be complex conjugated
    bb: array_like
        Array not to be complex conjugated
    power_spectral_density: array_like
        Power spectral density of the noise
    duration: float
        duration of the data

    Returns
    =======
    Noise-weighted inner product.
    """

    integrand = np.conj(aa) * bb / power_spectral_density
    return 4 * trapezoid(integrand, x=frequencies).real


class LunarLikelihood:

    def __init__(self, gps_time_range=(788572813.0, 2050876818.0), log_dir=None):
        # 2005 to 2045
        self.gps_time_range = gps_time_range
        self.cache_folder = data_path
        self.ensure_ephemeris_are_available()
        self.center = np.asarray([0, 0, 0])[np.newaxis, :]

        self.data = None

        if log_dir is None:
            self.log_dir = Path(".")

        psd_path = data_path / "LGWA_Si_psd.txt"
        self.psd_data = np.loadtxt(psd_path)

    def psd(self, f):
        return np.interp(f, self.psd_data[:, 0], self.psd_data[:, 1] / 2.0)

    def get_detector_frame(self, time):

        n_ra, n_dec, x_ra, x_dec, y_ra, y_dec = interpolate_detector_frame(
            time, self.times_response, self.data_response
        )

        return (
            spherical_to_cartesian(n_ra, n_dec),
            spherical_to_cartesian(x_ra, x_dec),
            spherical_to_cartesian(y_ra, y_dec),
        )

    def get_antenna_response(self, time, ra, dec, psi):

        n, x, y = self.get_detector_frame(time)

        u, v = wave_frame_basis_cartesian(ra, dec, psi)

        un = np.dot(n, u)
        ux = np.dot(x, u)
        uy = np.dot(y, u)
        vn = np.dot(n, v)
        vx = np.dot(x, v)
        vy = np.dot(y, v)

        hpx = un * ux - vn * vx
        hcx = un * vx + vn * ux
        hpy = un * uy - vn * vy
        hcy = un * vy + vn * uy

        return np.vstack((hpx, hpy, hcx, hcy))

    def get_detector_position_vector(self, times):
        return interp_position(times, self.times_position, self.data_position)

    def get_detector_position(self, time):
        x = np.interp(time, self.times_position, self.data_position[:, 0])
        y = np.interp(time, self.times_position, self.data_position[:, 1])
        z = np.interp(time, self.times_position, self.data_position[:, 2])

        return np.asarray([x, y, z])

    def compute_position_interpolant(self):
        logging.info("Computing position interpolant")
        # safety margin for GPS times
        gps_time_range = self.gps_time_range + np.array([-1e5, 1e5])
        n_points_position = int((gps_time_range[1] - gps_time_range[0]) / (60 * 10))
        self.times_position, self.data_position = generate_data_position(
            n_points_position, *gps_time_range
        )
        np.save(self.fname_times_position, self.times_position)
        np.save(self.fname_data_position, self.data_position)

    def compute_response_interpolant(self):
        logging.info("Computing response interpolant")
        # safety margin for GPS times
        gps_time_range = self.gps_time_range + np.array([-1e5, 1e5])
        n_points_response = int((gps_time_range[1] - gps_time_range[0]) / (60 * 200))
        self.times_response, self.data_response = generate_data_response(
            n_points_response, *gps_time_range
        )

        np.save(self.fname_times_response, self.times_response)
        np.save(self.fname_data_response, self.data_response)

    def ensure_ephemeris_are_available(self):

        self.fname_times_position = (self.cache_folder / "times_position").with_suffix(
            ".npy"
        )
        self.fname_times_response = (self.cache_folder / "times_response").with_suffix(
            ".npy"
        )
        self.fname_data_position = (self.cache_folder / "data_position").with_suffix(
            ".npy"
        )
        self.fname_data_response = (self.cache_folder / "data_response").with_suffix(
            ".npy"
        )

        if self.fname_data_position.exists() and self.fname_times_position.exists():
            self.times_position = np.load(self.fname_times_position)
            if (self.times_position[0] > self.gps_time_range[0]) or (
                self.times_position[-1] < self.gps_time_range[1]
            ):

                self.compute_position_interpolant()
            self.data_position = np.load(self.fname_data_position)
        else:
            self.compute_position_interpolant()

        if self.fname_data_response.exists() and self.fname_times_response.exists():
            self.times_response = np.load(self.fname_times_response)
            if (self.times_response[0] > self.gps_time_range[0]) or (
                self.times_response[-1] < self.gps_time_range[1]
            ):
                self.compute_response_interpolant()
            self.data_response = np.load(self.fname_data_response)
        else:
            self.compute_response_interpolant()

    def amp_phase(self, f, parameters):
        q = parameters["mass_ratio"]
        q = max(q, 1 / q)
        eta = np.clip(q / (1 + q) ** 2, 0, 0.25)

        total_mass = parameters["chirp_mass"] / eta ** (3 / 5)

        phase = Phif3hPN(
            f,
            total_mass,
            eta,
            parameters["spin_1z"],
            parameters["spin_2z"],
            parameters["lambda_eff"],
            parameters["d_lambda"],
        )

        amplitude = Af3hPN(
            f,
            total_mass,
            eta,
            parameters["spin_1z"],
            parameters["spin_2z"],
            parameters["lambda_eff"],
            parameters["d_lambda"],
            parameters["luminosity_distance"],
        )
        return amplitude, phase

    def t_of_f(self, f, parameters):
        amplitude, phase = self.amp_phase(f, parameters)
        
        t_baseline = parameters.pop("time_at_center_baseline", 0.)
        
        return time_to_merger(f, phase) + parameters["time_at_center"] + t_baseline

    def projected_waveform(self, f, parameters, parameters_for_amp_phase=None):

        if parameters_for_amp_phase is None:
            parameters_for_amp_phase = parameters
        amplitude, phase = self.amp_phase(f, parameters_for_amp_phase)
        
        t_baseline = parameters.pop("time_at_center_baseline", 0.)
        
        t_of_f = time_to_merger(f, phase) + parameters["time_at_center"] + t_baseline

        prop_unit_vector = -spherical_to_cartesian(
            parameters["right_ascension"], parameters["declination"]
        )

        detector_position = self.get_detector_position_vector(t_of_f) - self.center
        delay_phase = (
            np.dot(detector_position, prop_unit_vector) / SPEED_OF_LIGHT
            + parameters["time_at_center"]
        ) * (2 * np.pi * f)

        detector_exists = t_of_f > (parameters["time_at_center"] + t_baseline - DETECTOR_LIFETIME)

        logging.info(
            f"Detector exists from f >= {f[detector_exists][0]}"
        )
        
        cosiota = np.cos(parameters["inclination"])

        amplitude[~detector_exists] = 0.0

        total_phase = phase - delay_phase + parameters["phase"]

        h_plus = amplitude * 0.5 * (1 + cosiota**2) * np.exp(1j * (total_phase))
        h_cross = amplitude * cosiota * np.exp(1j * (total_phase + np.pi / 2))

        hpx, hpy, hcx, hcy = self.get_antenna_response(
            t_of_f,
            parameters["right_ascension"],
            parameters["declination"],
            parameters["polarization"],
        )

        h_x = h_plus * hpx + h_cross * hcx
        h_y = h_plus * hpy + h_cross * hcy

        # returns shape (n_channels, n_frequencies)
        return np.vstack((h_x, h_y))

    def compute_center(self, time):
        self.center = self.get_detector_position(time)[np.newaxis, :]

    def log_likelihood_ratio(self, f, parameters):

        hx, hy = self.projected_waveform(f, parameters)
        dx, dy = self.data

        psd = self.psd(f)

        return (
            +noise_weighted_inner_product(hx, dx, psd, f)
            + noise_weighted_inner_product(hy, dy, psd, f)
            - 0.5 * noise_weighted_inner_product(hx, hx, psd, f)
            - 0.5 * noise_weighted_inner_product(hy, hy, psd, f)
        )

    @property
    def bin_widths(self):
        return np.ediff1d(self.relbin_frequencies)

    @property
    def n_bins(self):
        return len(self.relbin_frequencies) - 1

    def set_reference_waveform(self, frequency_grid, parameters_h0):
        self.h0_parameters = parameters_h0
        h0 = self.projected_waveform(frequency_grid, parameters_h0)
        h0_mask_1d = abs(h0) > 0.0
        h0_mask = np.logical_and(*h0_mask_1d)
        self.relbin_frequencies = frequency_grid[h0_mask]
        self.h0_bin = h0[:, h0_mask]

        assert np.all(self.h0_bin != 0.0)

    def make_relbin_data(self, frequency_grid, parameters_h0, n_local_grid=2**10):

        meta_dict = parameters_h0 | {
            "n_freqs": len(frequency_grid),
            "f_min": float(min(frequency_grid)),
            "f_max": float(max(frequency_grid)),
        }

        if self.log_dir is not None:
            fname_data = self.log_dir / "relbin_data.npy"
            fname_meta = self.log_dir / "relbin_metadata.yaml"

            parameters_are_identical = True
            if fname_meta.exists():
                with open(fname_meta, "r") as f:
                    saved_meta = yaml.safe_load(f)
                    for key in saved_meta:
                        if not np.isclose(meta_dict[key], saved_meta[key]):
                            parameters_are_identical = False
            else:
                parameters_are_identical = False

            if parameters_are_identical:
                self.relbin_summary_data = np.load(fname_data)
                self.set_reference_waveform(frequency_grid, parameters_h0)

                return

            with open(fname_meta, "w") as f:
                yaml.dump(meta_dict, f)

        # here we work with an injection, so d == h0

        self.set_reference_waveform(frequency_grid, parameters_h0)

        self.relbin_summary_data = np.empty((2, self.n_bins, 4))

        for i, (f_left, f_right) in tqdm(
            enumerate(zip(self.relbin_frequencies[:-1], self.relbin_frequencies[1:])),
            total=self.n_bins,
        ):
            f_avg = (f_right + f_left) / 2.0
            local_grid = np.geomspace(f_left, f_right, num=n_local_grid)
            local_psd = self.psd(local_grid)
            local_h0 = self.projected_waveform(local_grid, parameters_h0)
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

                self.relbin_summary_data[channel, i, 0] = A0
                self.relbin_summary_data[channel, i, 1] = A1
                self.relbin_summary_data[channel, i, 2] = A0
                self.relbin_summary_data[channel, i, 3] = A1

        np.save(fname_data, self.relbin_summary_data)

    def relbin_log_likelihood_ratio(self, parameters):
        f_bin = self.relbin_frequencies

        r_bin = self.projected_waveform(f_bin, parameters) / self.h0_bin

        bin_widths = self.bin_widths
        r0 = (r_bin[:, 1:] + r_bin[:, :-1]) / 2.0
        r1 = (r_bin[:, 1:] - r_bin[:, :-1]) / bin_widths[np.newaxis, :]

        # self.relbin_summary_data has shape [n_channels, n_freqs-1, 4]
        # the last axis contains A0, A1, B0, B1 in this order

        summary_data = self.relbin_summary_data
        return relbin_log_likelihood_kernel(
            r0, r1, np.asarray(summary_data, dtype=complex)
        )

    def optimal_snr(self, f, parameters):
        h = self.projected_waveform(f, parameters)
        return np.sqrt(noise_weighted_inner_product(h, h, self.psd(f), f).sum())


if __name__ == "__main__":
    like = LunarLikelihood()
    t0 = 788572813.0
    # t0 = 1893024018.

    parameters = {
        "chirp_mass": 1.2,
        "mass_ratio": 1.0,
        "time_at_center": t0,
        "inclination": 0.5,
        "phase": 0.0,
        "right_ascension": 0.0,
        "declination": 0.0,
        "polarization": 0.0,
        "luminosity_distance": 40.0,
        "spin_1z": 0.0,
        "spin_2z": 0.0,
        "lambda_eff": 0.0,
        "d_lambda": 0.0,
    }

    like.compute_center(t0)

    f = np.geomspace(1e-1, 3, num=1000)

    like.make_relbin_data(f, parameters)

    masses = np.geomspace(1e-9, 1e-6, num=20) + 1.2
    errors = np.empty_like(masses)
    likes = np.empty_like(masses)

    full_f = np.geomspace(1e-1, 3, num=10_000_000)

    for i, mass in tqdm(enumerate(masses)):

        mod_parameters = parameters | {"chirp_mass": mass}
        relbin = like.relbin_log_likelihood_ratio(mod_parameters)
        like.data = like.projected_waveform(full_f, parameters)
        regular = like.log_likelihood_ratio(full_f, mod_parameters)
        errors[i] = np.abs(relbin - regular)
        likes[i] = regular

    plt.loglog(masses - 2.8, errors, label="error")
    newax = plt.gca().twinx()
    newax.semilogx(masses - 2.8, likes, label="likes")
    # hx, hy = like.projected_waveform(f, parameters)
    # plt.loglog(f, abs(hx))
    # plt.loglog(f, abs(hy))
    plt.legend()
    plt.show()
