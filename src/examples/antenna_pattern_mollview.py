from lgwa_response.likelihood import LunarLikelihood
import healpy
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path

def plot_mollview(time, psi, extra_name):
    
    like = LunarLikelihood(
        gps_time_range=(time-1e5, time+1e5),
        power_spectral_density_name="LGWA_Si_psd",
        lgwa_position={'longitude': 0., 'latitude': -87.5},
        log_dir_ephemeris='mollview_example',
    )

    NSIDE = 32
    NPIX = healpy.nside2npix(NSIDE)
    codec_array, ra_array = healpy.pix2ang(NSIDE, range(NPIX))
    dec_array = np.pi/2 - codec_array
    
    antenna_pattern = []

    for ra, dec in tqdm(zip(ra_array, dec_array)):
        hpx, hcx, hpy, hcy = like.get_antenna_response(
            np.array(time)[np.newaxis], 
            float(ra), 
            float(dec), 
            psi
    )
        antenna_pattern.append(hcx[0])

    cmap = plt.get_cmap('seismic')
    healpy.mollview(np.array(antenna_pattern), cmap=cmap, title=f'Antenna Pattern, psi = {psi:.2f}', min=-1, max=1)
    plt.savefig(Path('mollview_example') / extra_name)
    plt.close()

if __name__ == '__main__':
    for i, psi in enumerate(np.linspace(0, np.pi, 25)):
        plot_mollview(1e9, psi, f'test_{i:02}.png')
