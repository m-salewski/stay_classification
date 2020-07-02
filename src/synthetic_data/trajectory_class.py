import numpy as np

import warnings

from synthetic_data.trajectory import get_stay
from synthetic_data.trajectory import get_journey_path, get_segments
from synthetic_data.masking import get_mask_with_duplicates, get_adjusted_dup_mask
from synthetic_data.trajectory import get_stay_segs, get_adjusted_stays
from synthetic_data.noise import get_noisy_segs, get_noisy_path, get_noise_arr

rand_range = lambda min_, max_, size: (max_-min_)*np.random.random_sample(size=size) + min_


def get_time_bounds(nr_stays, time_thresh = 1/6):
    
    # Check that the stays aren't too short, ie above the time thresh
    keep_running = True
    while keep_running:
        
        t_bounds = np.concatenate((np.array([0,24]),rand_range(0,24,2*nr_stays)))
        t_bounds = np.sort(t_bounds)
        
        keep_running = any(np.abs(t_bounds[:-1:2]-t_bounds[1::2])<time_thresh)
        
    return t_bounds


def get_xlocs(min_, max_, size, dist_thresh):
    
    
    keep_running = True
    
    while keep_running:
        
        xlocs = rand_range(min_, max_, size)
        
        keep_running = any(np.abs(xlocs[:-1]-xlocs[1:])<dist_thresh)
        
    return xlocs


def get_rand_stays(nr_stays=None):
    
    """ 
    Creates a random set of stays.

    :param nr_stays: int A proposed count for the number of stays*
    
    :return: [dict] list of time-ordered stays
    
    *Note, due to the stochasticity and error checking, 
    this quantity may be larger than the actual number of stays returned
    """
    
    # Create a random number of stays
    #TODO s: 
    # 1. make this follow a non-uniform distribution 
    # 2. iterate until the corrected number of stays matches the proposed
    #    (this will be important when the distribution is specified)
    if nr_stays == None:
        nr_stays = np.random.randint(10)
    
    # Create the ordered timepoints for the stays
    #TODO: give the time thresh as a param
    time_thresh = 1/6 # 10 mins
    t_bounds = get_time_bounds(nr_stays, time_thresh)
    
    # Create the sptaial locations for the stays 
    #TODO: these should be specified
    xlocs = rand_range(-2.0, 2.0, int(len(t_bounds)/2))
    
    # From the new times and locs, generate the stays
    #TODO: if checking against the proposed number of stays,
    #      apply the check above.
    stays = []
    for n in range(int(len(t_bounds)/2)):
        nn = 2*n
        stay = get_stay(t_bounds[nn], t_bounds[nn+1], xlocs[n])
        stays.append(stay)    

    return stays


def get_rand_traj(configs):
    """ 
    Creates a random trajectory.

    :param configs: dict unused in a stay; retained here for generality
    
    :return: np.array An array of time points
    :return: np.array An array of (raw) location points, without noise
    :return: np.array An array of noisy locations
    :return: [dict]   A list of ordered segments (stays and travels)
    """

    dsec = 1/3600.0
    time = np.arange(0,24,dsec)

    if 'event_frac' not in configs.keys():
        event_frac = rand_range(0.001,0.01,1)[0]
        configs['event_frac'] = event_frac        

    if 'duplicate_frac' not in configs.keys():      
        duplicate_frac = rand_range(0.05,0.3,1)[0]
        configs['duplicate_frac'] = duplicate_frac    

    stays  = get_rand_stays()

    time_arr, raw_arr, noise_arr = get_trajectory(stays, time, configs)
    segments = get_segments(time, stays, threshold=0.5)
    
    return time_arr, raw_arr, noise_arr, segments


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