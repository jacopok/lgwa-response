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

def test_detector_exists_nowhere_edge_case():
    t0 = 1.42087814e+09    
    mc = 31.27177785

    injection_params = {
    "chirp_mass": mc,
    "mass_ratio": 0.97828418,
    "luminosity_distance": 413.79263441,
    "theta_jn": 0.71793531,
    "psi": 1.32899451,
    "phase": 1.5664732,
    "ra": 2.33323452,
    "dec": 0.19024356,
    "time_at_center": 0,
    "time_at_center_baseline": t0,
    "chi_1": 0.0569203,
    "chi_2": 0.01825482,
    "lambda_1": 0.0,
    "lambda_2": 0.0,
    }
    
    weird_params = [{
        "chirp_mass": 3.12718176e+01,
        "mass_ratio": 9.99999999e-01,
        "luminosity_distance": 2.24819515e+02,
        "theta_jn": 1.97990526e+00,
        "psi": 4.43827337e-01,
        "phase": 1.07711245e+00,
        "ra": 5.34992216e+00,
        "dec": -1.86114762e-01,
        "time_at_center": 7.03045553e-02,
        "time_at_center_baseline": t0,
        "chi_1": 0.0569203,
        "chi_2": 0.01825482,
        "lambda_1": 0.0,
        "lambda_2": 0.0,
    },
    {
        "chirp_mass": 3.12717981e+01,
        "mass_ratio": 9.99999999e-01,
        "luminosity_distance": 4.96809788e+02,
        "theta_jn": 1.29917807e+00,
        "psi": 6.61506004e-01,
        "phase": 3.05986641e+00,
        "ra": 2.11695218e+00,
        "dec": -6.62047343e-03,
        "time_at_center": 1.58576570e+01,
        "time_at_center_baseline": t0,
        "chi_1": 0.0569203,
        "chi_2": 0.01825482,
        "lambda_1": 0.0,
        "lambda_2": 0.0,
    },
    {
        "chirp_mass": 3.12717718e+01,
        "mass_ratio": 9.99999989e-01,
        "luminosity_distance": 6.38652691e+02,
        "theta_jn": 2.53061975e+00,
        "psi": 9.49684955e-02,
        "phase": 4.50127937e+00,
        "ra": 2.45350809e+00,
        "dec": 1.20834395e-01,
        "time_at_center": -1.48847994e+01,
        "time_at_center_baseline": t0,
        "chi_1": 0.0569203,
        "chi_2": 0.01825482,
        "lambda_1": 0.0,
        "lambda_2": 0.0,
    }
    ]
    
    like = LunarLikelihood()
    f = np.geomspace(0.15, 3, num=40)
    like.make_relbin_data(f, from_bilby(injection_params))
    
    for param in weird_params:
        like.projected_waveform(f, from_bilby(param))
        like.relbin_log_likelihood_ratio(from_bilby(param))