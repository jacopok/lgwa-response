from lgwa_response.lunar_coordinates import (
    wave_frame_basis_cartesian,
    spherical_to_cartesian,
    get_orthonormal_vectors,
    scalar_product_spherical,
)

from lunarsky import MoonLocation, SkyCoord, LunarTopo
from astropy.time import Time
from erfa import ufunc

from hypothesis import given, strategies as st
import numpy as np


@given(
    st.floats(0, 2 * np.pi),
    st.floats(-np.pi / 2, np.pi / 2.0),
    st.floats(0, 2 * np.pi),
)
def test_orthonormality_wave_frame(ra, dec, psi):
    u, v = wave_frame_basis_cartesian(ra, dec, psi)

    m = spherical_to_cartesian(ra, dec)

    assert np.isclose(np.dot(u, u), 1)
    assert np.isclose(np.dot(v, v), 1)
    assert np.isclose(np.dot(m, m), 1)
    assert np.isclose(np.dot(v, u), 0)
    assert np.isclose(np.dot(v, m), 0)
    assert np.isclose(np.dot(u, m), 0)


@given(
    st.floats(0, 2 * np.pi),
    st.floats(-np.pi / 2, np.pi / 2.0),
    st.floats(0, 2 * np.pi),
)
def test_psi_rotation(ra, dec, psi):
    u, v = wave_frame_basis_cartesian(ra, dec, psi)
    u2, v2 = wave_frame_basis_cartesian(ra, dec, psi + np.pi / 2)

    assert np.isclose(np.dot(u, u2), 0)
    assert np.isclose(np.dot(v, v2), 0)


@given(
    st.floats(1577491218.0, 1893024018.0),
)
def test_orthonormality_detector_frame(gps_time):
    time = Time(gps_time, format="gps")
    location = MoonLocation.from_selenodetic(
        lat=-88.5, 
        lon=15.
    )
    topo = LunarTopo(obstime=time, location=location)
    n_ra, n_dec, x_ra, x_dec, y_ra, y_dec = get_orthonormal_vectors(topo)

    n = spherical_to_cartesian(n_ra, n_dec)
    x = spherical_to_cartesian(x_ra, x_dec)
    y = spherical_to_cartesian(y_ra, y_dec)

    assert np.isclose(np.dot(n, n), 1)
    assert np.isclose(np.dot(x, x), 1)
    assert np.isclose(np.dot(y, y), 1)
    assert np.isclose(np.dot(x, y), 0)
    assert np.isclose(np.dot(x, n), 0)
    assert np.isclose(np.dot(n, y), 0)


@given(
    st.floats(0, 2 * np.pi),
    st.floats(-np.pi / 2, np.pi / 2.0),
)
def test_spherical_cartesian_conventions(ra, dec):
    their_definition = spherical_to_cartesian(ra, dec)

    explicit_definition = np.asarray(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    )

    assert np.allclose(their_definition, explicit_definition)


@given(
    st.floats(0, 2 * np.pi),
    st.floats(-np.pi / 2, np.pi / 2.0),
    st.floats(0, 2 * np.pi),
    st.floats(-np.pi / 2, np.pi / 2.0),
)
def test_spherical_scalar_products(ra1, dec1, ra2, dec2):
    v1 = spherical_to_cartesian(ra1, dec1)
    v2 = spherical_to_cartesian(ra2, dec2)

    sp_angles = scalar_product_spherical(ra1, dec1, ra2, dec2)
    assert np.isclose(sp_angles, np.dot(v1, v2))
