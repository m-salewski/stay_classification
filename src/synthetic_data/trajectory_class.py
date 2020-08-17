import numpy as np

import warnings

from synthetic_data.trajectory import get_stay
from synthetic_data.trajectory import get_journey_path, get_segments
from synthetic_data.masking import get_mask_with_duplicates, get_adjusted_dup_mask
from synthetic_data.trajectory import get_stay_segs, get_adjusted_stays
from synthetic_data.noise import get_noisy_segs, get_noisy_path, get_noise_arr


rand_range = lambda min_, max_, size: (max_-min_)*np.random.random_sample(size=size) + min_

'''
TODO:
1. update all occurences of get_trajectory in NBs & scripts to include the `segments` output
'''

def get_time_bounds(nr_stays, time_thresh, m2m_flags=[True,True]):
        
    # Check that the stays aren't too short, ie above the time thresh
    keep_running = True
    while keep_running:
        
        
        # Get midnight-to-midnight trajectory (set to 80% --> 80% should have mid-to-mid)
        # TODO: adjust this so that the end points can also start/end at 00:00/23:59 independently
        t_bounds = rand_range(0,24,2*nr_stays)
        if m2m_flags[0]:
            t_bounds = np.concatenate((np.array([0]),t_bounds[1:]))
        if m2m_flags[1]:
            t_bounds = np.concatenate((t_bounds[:-1],np.array([24])))

            
        #t_bounds = np.concatenate((np.array([0,24]),rand_range(0,24,2*nr_stays)))
        t_bounds = np.sort(t_bounds)
        
        keep_running = any(np.abs(t_bounds[1::2]-t_bounds[:-1:2])<time_thresh)
        
    return t_bounds


def get_xlocs(min_, max_, size, dist_thresh):
    
    # Check that the stays aren't too close, ie within the dist thresh    
    keep_running = True
    while keep_running:
        
        xlocs = rand_range(min_, max_, size)
        
        keep_running = any(np.abs(xlocs[:-1]-xlocs[1:])<dist_thresh)
        
    return xlocs

get_slope = lambda t1,x1,t2,x2: (x1-x2)/(t1-t2)
get_t_to  = lambda t1, x1, x2 ,m : np.sign(x2-x1)*(x2 - x1)/m + t1
get_t_fro = lambda t2, x1, x2 ,m : np.sign(x2-x1)*(x1 - x2)/m + t2

def get_rand_stays(configs, nr_stays=None):      
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
    p,_ = np.histogram(np.random.lognormal(1.2,0.60,500),bins=np.arange(0,30,1), density=True)
    if nr_stays == None:
        nr_stays = np.random.choice(np.arange(p.size),size=1,p=p)[0]
    
    # flags to select whether the trajectory begins and ends on midnights
    # Originally: m2m_flags = np.random.choice(np.arange(p.size),size=2,p=p)[0:2]%5 > 0
    if p.size > 2: 
        m2m_flags = np.random.choice(np.arange(p.size),size=2,p=p)[0:2]%5 > 0
    else:
        m2m_flags = np.random.choice(np.arange(2),size=2)[0:2]%2 > 0
    
    # Create the ordered timepoints for the stays
    #TODO: give the time thresh as a param
    t_bounds = get_time_bounds(nr_stays, configs['time_thresh'], m2m_flags)
    
    # Create the sptaial locations for the stays 
    #TODO: these should be specified in a config-file
    xlocs = get_xlocs(-5.0, 5.0, int(len(t_bounds)/2), configs['dist_thresh'])
    endpoints_flag = np.random.choice(np.arange(p.size),size=1,p=p)[0]%3
    if nr_stays <= 2:
        endpoints_flag = 2
    if endpoints_flag == 0:
        xlocs[0] = xlocs[-1]
    elif endpoints_flag == 1:
        xlocs[-1] = xlocs[0]
    else:
        pass
    
    # Adjust the t-bounds to account for at least minimal walking speeds
    #TODO: make this configurable in speeds and turning off/on    
    min_speed = 3.6
    min_speed = rand_range(min_speed, 3*min_speed, 1)
    
    x_locs = []
    for xx in xlocs:
        x_locs.append(xx)
        x_locs.append(xx)
    
    for n in range(0,len(t_bounds)-2,2):
        
        m = get_slope(t_bounds[n+1], x_locs[n+1], t_bounds[n+2], x_locs[n+2])
        
        if abs(m) < 3.6:
            
            mid_t = (t_bounds[n+1] + t_bounds[n+2])/2.0
            mid_x = (x_locs[n+1] + x_locs[n+2])/2.0

            new_t1 = get_t_fro(mid_t, x_locs[n+1], mid_x, min_speed)
            new_t2  = get_t_to(mid_t, mid_x, x_locs[n+2], min_speed)            

            t_bounds[n+1] = new_t1
            t_bounds[n+2] = new_t2      

    
    # From the new times and locs, generate the stays
    #TODO: if checking against the proposed number of stays,
    #      apply the check above.
    stays = []
    for n in range(0,len(t_bounds)-1,2):
        #nn = 2*n
        stay = get_stay(t_bounds[n], t_bounds[n+1], x_locs[n+1])
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

    stays  = get_rand_stays(configs)

    time_arr, raw_arr, noise_arr, segments = get_trajectory(stays, time, configs)
    #segments = get_segments(time, stays, dist_thresh=configs['dist_thresh'])
    
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
    time_thresh = configs['time_thresh']
    dist_thresh = configs['dist_thresh']
    event_frac = configs['event_frac']
    duplicate_frac = configs['duplicate_frac']    
    noise_min = configs['noise_min']
    noise_max = configs['noise_max']
    
    #print("stays:",all([abs(d['end']-d['start'])>=time_thresh for d in stays]))

    
    t_segs, x_segs = get_stay_segs(stays)

    # Compute the segments
    segments = get_segments(time, stays, dist_thresh)
    
    # Compute the raw journey
    raw_journey = get_journey_path(time, segments)

    keep_running = True
    #n=0
    while keep_running:
        # Reduce the journey based on the event- and duplicate fractions
        dup_mask = get_mask_with_duplicates(time, event_frac, duplicate_frac)

        dup_mask = get_adjusted_dup_mask(time, stays, dup_mask)
        
        time_sub = time[dup_mask]
        raw_journey_sub = raw_journey[dup_mask]

        # Using the new time sub-array, get the _adjusted_ stays and segments
        stays_ = get_adjusted_stays(segments, time_sub)
        
        # Check whether to iterate again
        keep_running = any([abs(d['end']-d['start'])<time_thresh for d in stays_])
        #print(f"{n:3d} stays: keep_running =", keep_running)
        segments_ = get_segments(time, stays, dist_thresh)
        #n += 1
     
    #new_t_segs, new_x_segs = get_stay_segs(new_stays)      

    noises = get_noise_arr(noise_min, noise_max, len(segments_))

    noise_segments = get_noisy_segs(segments_, noises)

    noise_journey_sub = get_noisy_path(time_sub, raw_journey_sub, noise_segments)
    
    return time_sub, raw_journey_sub, noise_journey_sub, noise_segments


def pickle_trajectory(t_arr, x_arr, nx_arr, segs, path_to_file):
    
    import pickle

    trajectory = {}
    trajectory['segments'] = segs
    trajectory['time_arr'] = t_arr
    trajectory['raw_locs_arr'] = x_arr
    trajectory['nse_locs_arr'] = nx_arr
    
    pickle.dump( trajectory, open( path_to_file, "wb" ) )
