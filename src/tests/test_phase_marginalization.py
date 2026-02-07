import numpy as np
import pytest
from lgwa_response.simple_waveforms import from_bilby
from lgwa_response.likelihood import LunarLikelihood
from scipy.special import logsumexp

@pytest.fixture
def injection_params():
    mc = 31.27177785
    t0 = 1.42087814e+09

    return from_bilby({
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
        'chi_1': -0.05063882, 
        'chi_2': 0.01304105,
        "lambda_1": 0.0,
        "lambda_2": 0.0,
    })

def test_likelihood_with_integral_over_phase(injection_params):
    
    
    like = LunarLikelihood()
    like.compute_center(injection_params['time_at_center_baseline'])
    
    freq = np.geomspace(0.01, 3, num=1000)
    
    # injection_params['phase'] = 0.
    
    like.make_relbin_data(freq, injection_params)
    
    new_params = injection_params.copy()
    new_params['chirp_mass'] += 3e-5
    
    ll_pm = like.relbin_log_likelihood_ratio_phase_marginalized(new_params)
    
    n = 1000
    phases = np.arange(0, n) / n * 2 * np.pi
    dphi = 2 * np.pi / n
    
    ll_by_phase = np.empty_like(phases)
    
    for i, phase in enumerate(phases):
        ll_by_phase[i] = like.relbin_log_likelihood_ratio(new_params | {'phase': phase})
    
    integrated_likelihood = logsumexp(ll_by_phase) - np.log(2*np.pi) + np.log(dphi)
    
    # import matplotlib.pyplot as plt
    # plt.plot(phases, np.exp(ll_by_phase-ll_pm))
    # plt.axhline(np.average(np.exp(ll_by_phase-ll_pm)))
    # plt.show()

    assert np.isclose(integrated_likelihood, ll_pm)