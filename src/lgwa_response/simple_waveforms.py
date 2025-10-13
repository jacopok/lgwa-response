import numpy as np
from numba import njit

SUN_MASS_SECONDS: float = 4.92549094830932e-6  # M_sun * G / c**3
EULER_GAMMA = 0.57721566490153286060

TF2_BASE: float = 3.668693487138444e-19
# ( Msun * G / c**3)**(5/6) * Hz**(-7/6) * c / Mpc / s


@njit(cache=True)
def compute_quadrupole_yy(lam: float) -> float:
    """Compute quadrupole coefficient from Lambda
    using the chi precessing spin parameter (for given 3-dim spin vectors)"""

    if lam <= 0.0:
        return 1.0
    loglam = np.log(lam)
    logCQ = (
        0.194
        + 0.0936 * loglam
        + 0.0474 * loglam**2
        - 4.21e-3 * loglam**3
        + 1.23e-4 * loglam**4
    )
    return np.exp(logCQ)


@njit(cache=True)
def compute_lambda_tilde(m1: float, m2: float, l1: float, l2: float) -> float:
    """Compute Lambda Tilde from masses and tides components
    --------
    m1 = primary mass component [solar masses]
    m2 = secondary mass component [solar masses]
    l1 = primary tidal component [dimensionless]
    l2 = secondary tidal component [dimensionless]
    """
    M = m1 + m2
    m1_4 = m1**4.0
    m2_4 = m2**4.0
    M5 = M**5.0
    comb1 = m1 + 12.0 * m2
    comb2 = m2 + 12.0 * m1
    return (16.0 / 13.0) * (comb1 * m1_4 * l1 + comb2 * m2_4 * l2) / M5


@njit(cache=True)
def compute_delta_lambda(m1: float, m2: float, l1: float, l2: float):
    """Compute delta Lambda Tilde from masses and tides components
    --------
    m1 = primary mass component [solar masses]
    m2 = secondary mass component [solar masses]
    l1 = primary tidal component [dimensionless]
    l2 = secondary tidal component [dimensionless]
    """
    M = m1 + m2
    q = m1 / m2
    eta = q / ((1.0 + q) * (1.0 + q))
    X = np.sqrt(1.0 - 4.0 * eta)
    m1_4 = m1**4.0
    m2_4 = m2**4.0
    M4 = M**4.0
    comb1 = (1690.0 * eta / 1319.0 - 4843.0 / 1319.0) * (m1_4 * l1 - m2_4 * l2) / M4
    comb2 = (6162.0 * X / 1319.0) * (m1_4 * l1 + m2_4 * l2) / M4
    return comb1 + comb2


@njit(cache=True)
def Phif3hPN(
    f: np.ndarray,
    M: float,
    eta: float,
    s1z: float = 0.0,
    s2z: float = 0.0,
    Lam: float = 0.0,
    dLam: float = 0.0,
) -> np.ndarray:
    """Compute post-Newtonian phase @ 3.5PN for compact binary coalescences
    including spins contributions and tidal effects @ 6PN (if Lam or dLam != 0)
    --------
    f = frequency series [Hz]
    M = binary mass [solar masses]
    s1z = primary spin component along z axis [dimensionless]

    s2z = secondary spin component along z axis [dimensionless]
    Lam = reduced tidal deformability parameter [dimensionless]
    dLam = asymmetric reduced tidal deformation parameter [dimensionless]
    --------
    Adapted from
    https://bitbucket.org/dailiang8/gwbinning/src/master/
    """
    s1x = 0.0
    s1y = 0.0
    s2x = 0.0
    s2y = 0.0
    vlso = 1.0 / np.sqrt(6.0)
    delta = np.sqrt(1.0 - 4.0 * eta)
    v = np.abs(np.pi * M * f * SUN_MASS_SECONDS) ** (1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v3 * v4
    v10 = v5 * v5
    v12 = v10 * v2
    eta2 = eta**2
    eta3 = eta**3

    m1M = 0.5 * (1.0 + delta)
    m2M = 0.5 * (1.0 - delta)
    chi1L = s1z
    chi2L = s2z
    chi1sq = s1x * s1x + s1y * s1y + s1z * s1z
    chi2sq = s2x * s2x + s2y * s2y + s2z * s2z
    chi1dotchi2 = s1x * s2x + s1y * s2y + s1z * s2z
    SL = m1M * m1M * chi1L + m2M * m2M * chi2L
    dSigmaL = delta * (m2M * chi2L - m1M * chi1L)

    # Phase correction due to spins
    sigma = eta * (721.0 / 48.0 * chi1L * chi2L - 247.0 / 48.0 * chi1dotchi2)
    sigma += 719.0 / 96.0 * (m1M * m1M * chi1L * chi1L + m2M * m2M * chi2L * chi2L)
    sigma -= 233.0 / 96.0 * (m1M * m1M * chi1sq + m2M * m2M * chi2sq)
    phis_15PN = 188.0 * SL / 3.0 + 25.0 * dSigmaL
    ga = (554345.0 / 1134.0 + 110.0 * eta / 9.0) * SL + (
        13915.0 / 84.0 - 10.0 * eta / 3.0
    ) * dSigmaL
    pn_ss3 = (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta * chi1L * chi2L
    pn_ss3 += (
        (
            (4703.5 / 8.4 + 2935.0 / 6.0 * m1M - 120.0 * m1M * m1M)
            + (-4108.25 / 6.72 - 108.5 / 1.2 * m1M + 125.5 / 3.6 * m1M * m1M)
        )
        * m1M
        * m1M
        * chi1sq
    )
    pn_ss3 += (
        (
            (4703.5 / 8.4 + 2935.0 / 6.0 * m2M - 120.0 * m2M * m2M)
            + (-4108.25 / 6.72 - 108.5 / 1.2 * m2M + 125.5 / 3.6 * m2M * m2M)
        )
        * m2M
        * m2M
        * chi2sq
    )
    phis_3PN = np.pi * (3760.0 * SL + 1490.0 * dSigmaL) / 3.0 + pn_ss3
    phis_35PN = (
        -8980424995.0 / 762048.0 + 6586595.0 * eta / 756.0 - 305.0 * eta2 / 36.0
    ) * SL - (
        170978035.0 / 48384.0 - 2876425.0 * eta / 672.0 - 4735.0 * eta2 / 144.0
    ) * dSigmaL

    # Point mass
    LO = 3.0 / 128.0 / eta / v5
    # pointmass = 1.0
    pointmass = (
        1
        + 20.0 / 9.0 * (743.0 / 336.0 + 11.0 / 4.0 * eta) * v2
        + (phis_15PN - 16.0 * np.pi) * v3
        + 10.0
        * (3058673.0 / 1016064.0 + 5429.0 / 1008.0 * eta + 617.0 / 144.0 * eta2 - sigma)
        * v4
        + (38645.0 / 756.0 * np.pi - 65.0 / 9.0 * eta * np.pi - ga)
        * (1.0 + 3.0 * np.log(v / vlso))
        * v5
        + (
            11583231236531.0 / 4694215680.0
            - 640.0 / 3.0 * np.pi**2
            - 6848.0 / 21.0 * (EULER_GAMMA + np.log(4.0 * v))
            + (-15737765635.0 / 3048192.0 + 2255.0 * np.pi**2 / 12.0) * eta
            + 76055.0 / 1728.0 * eta2
            - 127825.0 / 1296.0 * eta3
            + phis_3PN
        )
        * v6
        + (
            np.pi
            * (
                77096675.0 / 254016.0
                + 378515.0 / 1512.0 * eta
                - 74045.0 / 756.0 * eta**2
            )
            + phis_35PN
        )
        * v7
    )

    # Tidal correction to phase at 6PN
    # Eq.(1,4) [https://arxiv.org/abs/1310.8288]
    # Lam is the reduced tidal deformation parameter (\tilde\Lambda)
    # dLam is the asymmetric reduced tidal deformation parameter (\delta\tilde\Lambda)
    if Lam != 0.0 or dLam != 0.0:
        tidal = (
            Lam * v10 * (-39.0 / 2.0 - 3115.0 / 64.0 * v2) + dLam * 6595.0 / 364.0 * v12
        )
    else:
        tidal = 0.0 * v

    return LO * (pointmass + tidal)


@njit(cache=True)
def Af3hPN(
    f: np.ndarray,
    M: float,
    eta: float,
    s1z: float = 0.0,
    s2z: float = 0.0,
    Lam: float = 0.0,
    dLam: float = 0.0,
    Deff: float = 1.0,
) -> np.ndarray:
    """Compute post-Newtonian amplitude @ 3.5PN for compact binary coalescences
    --------
    f = frequency series [Hz]
    M = binary mass [solar masses]
    s1z = primary spin component along z axis [dimensionless]
    s2z = secondary spin component along z axis [dimensionless]
    Deff = luminosity distance
    --------
    Adapted from
    https://bitbucket.org/dailiang8/gwbinning/src/master/
    """
    s1x = 0.0
    s1y = 0.0
    s2x = 0.0
    s2y = 0.0

    Mchirp = M * np.power(np.abs(eta), 3.0 / 5.0)
    delta = np.sqrt(1.0 - 4.0 * eta)
    v = np.power(np.abs(np.pi * M * f * SUN_MASS_SECONDS), 1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v3 * v4
    eta2 = eta**2
    eta3 = eta**3

    # 0PN order
    A0 = (
        np.power(np.abs(Mchirp), 5.0 / 6.0)
        / np.power(np.abs(f), 7.0 / 6.0)
        / Deff
        / np.abs(np.pi) ** (2.0 / 3.0)
        * np.sqrt(5.0 / 24.0)
    )

    # Modulus correction due to aligned spins
    chis = 0.5 * (s1z + s2z)
    chia = 0.5 * (s1z - s2z)
    be = 113.0 / 12.0 * (chis + delta * chia - 76.0 / 113.0 * eta * chis)
    sigma = (
        chia**2 * (81.0 / 16.0 - 20.0 * eta)
        + 81.0 / 8.0 * chia * chis * delta
        + chis**2 * (81.0 / 16.0 - eta / 4.0)
    )
    eps = delta * chia * (502429.0 / 16128.0 - 907.0 / 192.0 * eta) + chis * (
        5.0 / 48.0 * eta2 - 73921.0 / 2016.0 * eta + 502429.0 / 16128.0
    )

    return (
        TF2_BASE
        * A0
        * (
            1.0
            + v2 * (11.0 / 8.0 * eta + 743.0 / 672.0)
            + v3 * (be / 2.0 - 2.0 * np.pi)
            + v4
            * (
                1379.0 / 1152.0 * eta2
                + 18913.0 / 16128.0 * eta
                + 7266251.0 / 8128512.0
                - sigma / 2.0
            )
            + v5 * (57.0 / 16.0 * np.pi * eta - 4757.0 * np.pi / 1344.0 + eps)
            + v6
            * (
                856.0 / 105.0 * EULER_GAMMA
                + 67999.0 / 82944.0 * eta3
                - 1041557.0 / 258048.0 * eta2
                - 451.0 / 96.0 * np.pi**2 * eta
                + 10.0 * np.pi**2 / 3.0
                + 3526813753.0 / 27869184.0 * eta
                - 29342493702821.0 / 500716339200.0
                + 856.0 / 105.0 * np.log(4.0 * v)
            )
            + v7
            * (
                -1349.0 / 24192.0 * eta2
                - 72221.0 / 24192.0 * eta
                - 5111593.0 / 2709504.0
            )
            * np.pi
        )
    )


def time_to_merger(f, phase):
    return np.gradient(phase, f, edge_order=2) / (2 * np.pi)


def time_to_merger_simple(f, mchirp):
    return -5.0 / (256.0 * np.pi ** (8 / 3)) / mchirp ** (5 / 3) / f ** (8 / 3)


def time_to_merger_simple_inverse(t, mchirp):
    """Input negative time in seconds, mchirp in seconds"""
    # t = -5./(256.*np.pi**(8/3))/mchirp**(5/3)/f**(8/3)
    # f = ((-5./(256.*np.pi**(8/3))/mchirp**(5/3)) / t)**(3/8)
    return ((-5.0 / (256.0 * np.pi ** (8 / 3)) / mchirp ** (5 / 3)) / t) ** (3 / 8)


def from_bilby(parameter_dict):
    res = {}

    res["chirp_mass"] = parameter_dict["chirp_mass"]
    res["mass_ratio"] = parameter_dict["mass_ratio"]

    q = res["mass_ratio"]
    eta = q / (1 + q) ** 2
    total_mass = res["chirp_mass"] / eta ** (3 / 5)
    m1 = total_mass * q / (1 + q)

    # clipping to prevent rounding errors when q=1
    m2 = np.clip(total_mass / (1 + q), 0, m1)

    res["phase"] = parameter_dict["phase"]
    res["luminosity_distance"] = parameter_dict["luminosity_distance"]
    res["time_at_center"] = parameter_dict["time_at_center"]
    res["time_at_center_baseline"] = parameter_dict.get("time_at_center_baseline", 0.)

    res["right_ascension"] = float(parameter_dict["ra"])
    res["declination"] = float(parameter_dict["dec"])
    res["inclination"] = parameter_dict[
        "theta_jn"
    ]  # this is not exactly true but it'll do for now
    res["polarization"] = parameter_dict["psi"]
    res["spin_1z"] = parameter_dict["chi_1"]
    res["spin_2z"] = parameter_dict["chi_2"]
    res["lambda_eff"] = compute_lambda_tilde(
        m1, m2, parameter_dict["lambda_1"], parameter_dict["lambda_2"]
    )
    res["d_lambda"] = compute_delta_lambda(
        m1, m2, parameter_dict["lambda_1"], parameter_dict["lambda_2"]
    )

    return res


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    q = 1
    eta = q / (1 + q) ** 2
    M = 2.8

    f = np.geomspace(1e-1, 3, num=10_000)

    phase = Phif3hPN(f, M, eta, 0.0, 0, 0.0, 0.0)

    mchirp = M * eta ** (3 / 5) * SUN_MASS_SECONDS

    dt = +time_to_merger_simple(f, mchirp) - time_to_merger(f, phase)
    t = -time_to_merger(f, phase) / 60

    plt.loglog(f, t, label="t")
    plt.loglog(f, dt, label="LO error")

    plt.legend()

    plt.show()
