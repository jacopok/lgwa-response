import numpy as np
from lgwa_response.simple_waveforms import from_bilby
from lgwa_response.likelihood import LunarLikelihood
from lgwa_response.add_relbin_samples import LunarLikelihoodAddingSamples
import pytest


@pytest.fixture
def injection():
    total_mass = 2.8
    distance = 40
    t0_moon = 1187008882.4
    q = 0.9
    injection_params = {
        "chirp_mass": total_mass * q ** (3 / 5) * (1 + q) ** (-6 / 5),
        "mass_ratio": q,
        "luminosity_distance": distance,
        "theta_jn": 2.545,
        "psi": np.pi / 2,
        "phase": np.pi,
        "ra": 3.44616,
        "dec": -0.408084,
        "time_at_center": t0_moon,
        "chi_1": 0.0,
        "chi_2": 0.0,
        "lambda_1": 400.0,
        "lambda_2": 400.0,
    }
    return from_bilby(injection_params)


def test_grid_refine_grid(injection):
    like = LunarLikelihoodAddingSamples()
    f = np.linspace(0.1, 0.10001, num=101)
    like.make_relbin_data(f, injection, n_local_grid=2**10)

    new_f = np.sqrt(f[51] * f[50])
    f2 = np.insert(f, 51, new_f)

    like.add_relbin_frequency(50, n_local_grid=2**10)

    data_after_insertion = like.relbin_summary_data.copy()

    like2 = LunarLikelihood()
    like2.make_relbin_data(f2, injection, n_local_grid=2**10)

    # not the exact same since the frequency grids differ (?)
    assert np.allclose(like.relbin_summary_data, like2.relbin_summary_data, rtol=1e-3)
    assert np.allclose(like.relbin_frequencies, like2.relbin_frequencies, rtol=1e-5)
    assert np.allclose(like.h0_bin, like2.h0_bin, rtol=1e-3, atol=0)


def test_likelihood_convergence_max(injection):

    fmin, fmax = 0.1, 4
    like = LunarLikelihood()

    llmax = []
    for n_grid in 2 ** np.arange(9, 11):
        f = np.linspace(fmin, fmax, num=n_grid)
        like.make_relbin_data(f, injection, n_local_grid=2**9)
        llmax.append(like.relbin_log_likelihood_ratio(injection))
    llmax = np.asarray(llmax)

    errors = llmax - llmax[-1]
    assert np.allclose(errors, 0, atol=1e-2)

    opt_snr = like.optimal_snr(np.geomspace(fmin, fmax, num=2**14), injection)

    assert np.allclose(llmax, opt_snr**2 / 2.0, rtol=2e-2)


def test_likelihood_convergence_off_max(injection):

    fmin, fmax = 0.5, 4
    like = LunarLikelihood()

    off_max_inj = injection.copy()
    off_max_inj["time_at_center"] += 0.1
    off_max_inj["chirp_mass"] += 1e-7
    off_max_inj["right_ascension"] += 1e-7

    llmax = []
    for n_grid in 2 ** np.arange(8, 11):
        f = np.linspace(fmin, fmax, num=n_grid)
        like.make_relbin_data(f, injection, n_local_grid=2**9)
        llmax.append(like.relbin_log_likelihood_ratio(off_max_inj))
    llmax = np.asarray(llmax)

    errors = llmax - llmax[-1]
    # we're using few samples for the test so the accuracy will not be great
    assert np.allclose(errors, 0, atol=5)
