import numpy as np
import numpy.ma as ma
    
    
    
'''
sec_ext.py has from extract_sections import create_odm_sections, create_activity_sections, get_internal_sections

and extract_sections.py is in the same dir

It calls modules from subpackage
from extraction_algorithm.discard_invalid_raw_trips import discard_invalid_trips, remove_activities, discard_round_trips

This means box_classifier is the subpackage of stay_classification

it contains box_classifier.py which calls functions from module box_method, and others from module checks

'''

#import stay_classification.box_classifier
#from stay_classification.
#from box_classifier import box_method
from box_method import get_mask, make_box

# lambdas 
embedded = lambda t0,t1,pair: ((t0 >= pair[0]) & (t0 <= pair[1]) | (t1 >= pair[0]) & (t1 <= pair[1]))

def get_bounded_indices(sub_arr, center, offset, configs):
    """
    Get the endpoint-indices of a region with positions within a bounded mean

    :param x_sub_arr: np.array Subarray of trajectory array of locations
    :param mean: float Centreline used to define the box    
    :param offset: int distance threshold used to define the box    

    :return: int Lower (global) index of the region 
    :return: int Upper (global) index of the region 
    """
    
    dist_thresh = configs['dist_thresh']
    
    # TODO: this fails when x_sub_arr.size = 0
    mask =  get_mask(sub_arr, center, dist_thresh, offset)

    return mask.min(), mask.max(0)


# TODO: check if needed, and where --> maybe for debugging
def get_counts(sub_arr, mean, dist_thresh):
    """
    """
    
    return get_mask(sub_arr, mean, dist_thresh).size

#TODO: checks args ordering
def get_slope(t_sub_arr, x_sub_arr, mean, dist_thresh):
    """
    Get the slope for the region; excludes outliers using mean +/- dist_thresh

    :param t_sub_arr: np.array Subarray of trajectory array of times    
    :param x_sub_arr: np.array Subarray of trajectory array of locations
    :param mean: float Centreline used to define the box    
    :param mean: float distance threshold used to define the box    
    :param configs: dict containing the parameters (currently unused)

    :return: float New value for mean without the outliers
    """
            
    mask =  get_mask(x_sub_arr, mean, dist_thresh)
    
    ub_xdata = t_sub_arr[mask] - t_sub_arr[mask].mean()
    ub_ydata = x_sub_arr[mask] - x_sub_arr[mask].mean()
    
    return (ub_xdata.T.dot(ub_ydata))/(ub_xdata.T.dot(ub_xdata))

# NOTE: WIP! This is only included fo safe keeping
def box_classifier_core(t_arr, x_arr, start_ind, last_ind, timepoint, pairs, configs, verbose=False):
    """
    For a given timepoint, try to get a box

    :param t_arr: np.array Trajectory array of timepoints
    :param x_arr: np.array Trajectory array of locations
    :param timepoint: float Timepoint used to find the box    
    :param configs: dict containing the parameters used to define the box
    :param verbox: bool To select printing metrics to stdio
    
    :return: int Starting index (from t_arr) of the box
    :return: int Ending index of the box
    :return: bool Flag to tell the iterator to keep the box or discard it
    """
    
    # Set output flag to false
    keep=False
    mean = None

    # Get some configs
    time_thresh = configs['time_thresh']    
    dist_thresh = configs['dist_thresh']
    slope_time_thresh = configs['slope_time_thresh']
    slope_thresh = configs['slope_thresh']

    
    # If the current timepoint is less than the last box-end, skip ahead
    # TODO: this is useful but possibly, without refinement, misses some stays 
    # HOWEVER: without it, it doesn't work!
    if (t_arr[start_ind] <= timepoint) & (timepoint <= t_arr[last_ind]):
        if verbose: print('\t\t\talready processed, skip')                
        return start_ind, last_ind, mean, keep  
    else:
        #NOTE: this _was_ t0,t1 --> TODO: check if this is correct
        if verbose: print(f'\nStart at {timepoint:.3f}, dt = {t_diff:.3f}, {start_ind}, {last_ind}')  
    
    # Get a box for a given timepoint
    mean, start_ind, last_ind = make_box(\
        t_arr, x_arr, timepoint, configs, verbose)
    
    # Drop if a NAN was encountered --. failed to find a mean
    if np.isnan(mean):
        if verbose: print("\t\t\tmean = NaN, skip")
        return start_ind, last_ind, mean, keep
        
    # If the duration of the stay is too small, skip ahead
    if t_arr[last_ind]-t_arr[start_ind] < time_thresh:
        if verbose: print("\t\t\ttoo short, skip")        
        return start_ind, last_ind, mean, keep
    
    # Modify the potential box to get the endpoints within the converge and bounded box
    t0, t1 = get_bounded_indices(x_arr[start_ind:last_ind], mean, start_ind, configs)   

    # If the duration due to the thresholded box is too short, skip ahead
    if t_arr[t1]-t_arr[t0] < time_thresh:
        if verbose: print("\t\t\talso too short, skip")        
        return start_ind, last_ind, mean, keep  

    # If the stay is less than 1 hour, check the slope of the segement --> This isn't watertight
    if t_arr[t1]-t_arr[t0] < slope_time_thresh:         
        xdata = t_arr[t0:t1]
        ydata = x_arr[t0:t1]
        slope = get_slope(xdata, ydata, mean, dist_thresh)
        if verbose: print(f"\tAt {timepoint:.3f}, slope = {slope:.3f}")
        if abs(slope) > slope_thresh: 
            if verbose: print("\t\t\tslope is too big, skip")
            return start_ind, last_ind, mean, keep
    
    # If the stay is embedded with other stays --> This is tricky!
    if any([embedded(t0,t1, p) for p in pairs] + [embedded(p[0],p[1],[t0,t1]) for p in pairs]):
        if verbose: print("\t\t\tEmbedded, skip")   
        return start_ind, last_ind, mean, keep

    start_ind, last_ind, keep = t0, t1, True
    
    return start_ind, last_ind, mean, keep

