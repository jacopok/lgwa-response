from lunarsky import MoonLocation, SkyCoord, LunarTopo
from astropy.time import Time
from astropy.coordinates import ICRS
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from tqdm import tqdm
from erfa import ufunc
from numba import njit

from . import data_path, lgwa_settings

LOCATION = MoonLocation.from_selenodetic(
    lon=lgwa_settings.LONGITUDE, lat=lgwa_settings.LATITUDE
)


def wave_frame_basis_cartesian(source_ra, source_dec, polarization_angle):
    theta = np.pi / 2.0 - source_dec
    phi = source_ra
    psi = polarization_angle

    u = np.asarray(
        [
            -np.sin(phi) * np.cos(psi) - np.cos(theta) * np.cos(phi) * np.sin(psi),
            +np.cos(phi) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi),
            +np.sin(theta) * np.sin(psi),
        ]
    )

    v = np.asarray(
        [
            -np.sin(phi) * np.sin(psi) + np.cos(theta) * np.cos(phi) * np.cos(psi),
            +np.cos(phi) * np.sin(psi) + np.cos(theta) * np.sin(phi) * np.cos(psi),
            -np.sin(theta) * np.cos(psi),
        ]
    )

    return u, v


def spherical_to_cartesian(ra, dec):
    return ufunc.s2c(ra, dec)


def scalar_product_spherical(ra1, dec1, ra2, dec2):
    return np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)


def get_orthonormal_vectors(topo):
    n = SkyCoord(alt=np.pi / 2 * u.rad, az=0 * u.rad, frame=topo).transform_to(ICRS())
    x = SkyCoord(alt=0 * u.rad, az=0 * u.rad, frame=topo).transform_to(ICRS())
    y = SkyCoord(alt=0 * u.rad, az=np.pi / 2 * u.rad, frame=topo).transform_to(ICRS())

    return (
        n.ra.to(u.rad).value,
        n.dec.to(u.rad).value,
        x.ra.to(u.rad).value,
        x.dec.to(u.rad).value,
        y.ra.to(u.rad).value,
        y.dec.to(u.rad).value,
    )


def get_detector_position(time):
    body = LOCATION.get_mcmf(time).transform_to(ICRS())
    body.representation_type = "cartesian"

    return np.squeeze(
        (body.x.to(u.m).value, body.y.to(u.m).value, body.z.to(u.m).value)
    )


def generate_data_response(
    n_points, gps_time_start=1577491218.0, gps_time_end=1893024018.0
):

    # 2030 to 2040
    times = Time(np.linspace(gps_time_start, gps_time_end, num=n_points), format="gps")

    saved_data = np.empty((n_points, 6))
    for i, time in tqdm(enumerate(times)):
        topo = LunarTopo(obstime=time, location=LOCATION)
        saved_data[i, :] = get_orthonormal_vectors(topo)

    for j in range(6):
        saved_data[:, j] = np.unwrap(saved_data[:, j])

    return times.value, saved_data


def generate_data_position(
    n_points, gps_time_start=1577491218.0, gps_time_end=1893024018.0
):

    # 2030 to 2040
    times = Time(np.linspace(gps_time_start, gps_time_end, num=n_points), format="gps")

    saved_data = np.empty((n_points, 3))
    for i, time in tqdm(enumerate(times)):
        saved_data[i, :] = get_detector_position(time)

    return times.value, saved_data


def test_interpolation_error_response(dataset_sizes):

    rng = np.random.default_rng(seed=1)

    all_errors = []
    for n_samples in dataset_sizes:
        print(f"Trying {n_samples} samples")

        times, data = generate_data_response(n_samples)

        these_errors = []
        for i in tqdm(range(1000)):
            time = Time(rng.uniform(times.value[0], times.value[-1]), format="gps")
            ra = rng.uniform(0, 2 * np.pi)
            dec = np.pi / 2 - np.arccos(rng.uniform(-1, 1))

            true_vecs = get_orthonormal_vectors(
                LunarTopo(obstime=time, location=LOCATION)
            )

            for j in [0, 2, 4]:
                true_scalar_product = scalar_product_spherical(
                    ra, dec, true_vecs[j], true_vecs[j + 1]
                )
                approx_ra = np.interp(time.value, times.value, data[:, j])
                approx_dec = np.interp(time.value, times.value, data[:, j + 1])
                approx_scalar_product = scalar_product_spherical(
                    ra, dec, approx_ra, approx_dec
                )
                these_errors.append(np.abs(approx_scalar_product - true_scalar_product))
        all_errors.append(these_errors)

    return all_errors


def test_interpolation_error_position(dataset_sizes):

    rng = np.random.default_rng(seed=1)

    all_errors = []
    for n_samples in dataset_sizes:
        print(f"Trying {n_samples} samples")

        times, data = generate_data_position(n_samples)

        these_errors = []
        for i in tqdm(range(1000)):
            time = Time(rng.uniform(times.value[0], times.value[-1]), format="gps")

            true_position = get_detector_position(time)
            approx_position = np.asarray(
                [np.interp(time.value, times.value, data[:, j]) for j in range(3)]
            )

            these_errors.append(np.linalg.norm(approx_position - true_position))

        all_errors.append(these_errors)

    return all_errors


def make_response_interpolation_plot():

    dataset_sizes = [1250, 2500, 5000, 10_000, 20_000]
    file_path = (
        data_path
        / "cache"
        / f'lunar_response_interpolation_{"_".join(map(str, dataset_sizes))}.npy'
    ).with_suffix(".npy")
    if file_path.exists():
        all_errors = np.load(file_path)
    else:
        all_errors = np.asarray(test_interpolation_error_response(dataset_sizes))
        np.save(file_path, all_errors)

    bplot_kwargs = {
        "patch_artist": True,
        "medianprops": dict(color="black"),
        "whis": (5, 95),
        "showfliers": False,
    }

    bplot = plt.boxplot(
        all_errors[:, ::3].T,
        tick_labels=[""] * len(dataset_sizes),
        widths=0.15 * len(dataset_sizes) * 0.3,
        positions=np.arange(1, len(dataset_sizes) + 1) - 0.25,
        label="Normal vector",
        **bplot_kwargs,
    )

    for patch in bplot["boxes"]:
        patch.set_facecolor("#D81B60")

    bplot = plt.boxplot(
        all_errors[:, 1::3].T,
        tick_labels=dataset_sizes,
        widths=0.15 * len(dataset_sizes) * 0.3,
        positions=np.arange(1, len(dataset_sizes) + 1),
        label="Horizontal vector x",
        **bplot_kwargs,
    )

    for patch in bplot["boxes"]:
        patch.set_facecolor("#FFC107")

    bplot = plt.boxplot(
        all_errors[:, 2::3].T,
        tick_labels=[""] * len(dataset_sizes),
        widths=0.15 * len(dataset_sizes) * 0.3,
        positions=np.arange(1, len(dataset_sizes) + 1) + 0.25,
        label="Horizontal vector y",
        **bplot_kwargs,
    )

    for patch in bplot["boxes"]:
        patch.set_facecolor("#1E88E5")

    plt.yscale("log")
    plt.xlabel("Number of ephemeris samples")
    plt.ylabel("Absolute error in scalar product [dimensionless]")

    seconds_ten_years = 315_576_000
    minutes_ten_years = seconds_ten_years / 60

    durations = minutes_ten_years / np.asarray(dataset_sizes)
    thirdax = plt.gca().secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    thirdax.set_xticks(
        np.arange(1, len(dataset_sizes) + 1), labels=[f"{d:.0f}" for d in durations]
    )
    thirdax.set_xlabel("Sampling interval [minutes]")

    plt.legend()
    plt.show()


def make_position_interpolation_plot():

    dataset_sizes = [10_000, 20_000, 40_000, 80_000, 160_000, 320_000]
    file_path = (
        data_path
        / "cache"
        / f'lunar_position_interpolation_{"_".join(map(str, dataset_sizes))}.npy'
    ).with_suffix(".npy")
    if file_path.exists():
        all_errors = np.load(file_path)
    else:
        all_errors = np.asarray(test_interpolation_error_position(dataset_sizes))
        np.save(file_path, all_errors)

    bplot_kwargs = {
        "patch_artist": True,
        "medianprops": dict(color="black"),
        "whis": (5, 95),
        "showfliers": False,
    }

    bplot = plt.boxplot(
        all_errors[:, 1::3].T,
        tick_labels=dataset_sizes,
        # widths=0.15*len(dataset_sizes)*.3,
        positions=np.arange(1, len(dataset_sizes) + 1),
        label="Position error",
        **bplot_kwargs,
    )

    for patch in bplot["boxes"]:
        patch.set_facecolor("#FFC107")

    plt.yscale("log")
    plt.xlabel("Number of ephemeris samples")
    plt.ylabel("Absolute error in position [meters]")
    # plt.title('Quartiles and 5-95%% confidence interval in the interpolation error')

    minimum_angular_wavelength = 299_792_458 / 3 / (2 * np.pi)

    secax = plt.gca().secondary_yaxis(
        "right",
        functions=(
            lambda x: x / minimum_angular_wavelength,
            lambda x: x * minimum_angular_wavelength,
        ),
    )
    secax.set_ylabel("Equivalent maximum phase error [rad]")

    seconds_ten_years = 315_576_000
    minutes_ten_years = seconds_ten_years / 60

    durations = minutes_ten_years / np.asarray(dataset_sizes)
    thirdax = plt.gca().secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    thirdax.set_xticks(
        np.arange(1, len(dataset_sizes) + 1), labels=[f"{d:.0f}" for d in durations]
    )
    thirdax.set_xlabel("Sampling interval [minutes]")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    make_response_interpolation_plot()
    make_position_interpolation_plot()

    # times, data = generate_data_response(4*240)

    # plt.plot(times.value, np.cos(data[:, 2]))
    # plt.plot(times.value, data[:, 3])
    # plt.show()
