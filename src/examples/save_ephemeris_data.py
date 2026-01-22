from lgwa_response.likelihood import LunarLikelihood
from pathlib import Path

if __name__ == '__main__':
    
    like = LunarLikelihood(
        lgwa_position={
            'latitude': 88.5,
            'longitude': 290.0,
            'azimuth': 0.0,
        },
        log_dir_ephemeris=Path('../lgwa_response/data/'),
        gps_time_range=(788572813.0, 2050876818.0),
    )
    # like.compute_position_interpolant()
    # like.compute_response_interpolant()