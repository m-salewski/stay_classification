import numpy as np

import warnings

from synthetic_data.trajectory import get_stay
from synthetic_data.trajectory import get_journey_path, get_segments
from synthetic_data.masking import get_mask_with_duplicates, get_adjusted_dup_mask
from synthetic_data.trajectory import get_stay_segs, get_adjusted_stays
from synthetic_data.noise import get_noisy_segs, get_noisy_path, get_noise_arr


def get_trajectory(stays, time, configs):
    """ 
    Creates a trajectory from a set of stays.

    :param stays: [dict] time as beginning of stay
    :param time:  np.array time at end of stay
    :param configs: dict unused in a stay; retained here for generality
    
    :return: np.array An array of time points
    :return: np.array An array of (raw) location points, without noise
    :return: np.array An array of noisy locations
    """
    
    threshold = configs['threshold']
    event_frac = configs['event_frac']
    duplicate_frac = configs['duplicate_frac']    
    noise_min = configs['noise_min']
    noise_max = configs['noise_max']
    
    t_segs, x_segs = get_stay_segs(stays)

    raw_journey = get_journey_path(time, get_segments(time, stays, threshold))

    dup_mask = get_mask_with_duplicates(time, event_frac, duplicate_frac)

    dup_mask = get_adjusted_dup_mask(time, stays, dup_mask)
    
    time_sub = time[dup_mask]
    raw_journey_sub = raw_journey[dup_mask]

    segments = get_segments(time, stays, threshold)
    new_stays = get_adjusted_stays(segments, time_sub)
    new_t_segs, new_x_segs = get_stay_segs(new_stays)      

    noises = get_noise_arr(noise_min, noise_max, len(segments))

    noise_segments = get_noisy_segs(segments, noises)

    noise_journey_sub = get_noisy_path(time_sub, raw_journey_sub, noise_segments)


    return time_sub, raw_journey_sub, noise_journey_sub