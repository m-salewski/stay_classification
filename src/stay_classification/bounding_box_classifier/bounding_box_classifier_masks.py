import numpy as np

from ..gap_tools import check_gaps

def bbox_mask(t_arr, x_arr, limits):
    """
    Just a wrapper for np.where
    """
    
    #NOTE: t_arr is included but no longer used
    mask = np.where(
        (x_arr >= limits[0]) & \
        (x_arr <= limits[1]))[0]
    
    return mask


def get_mask(loc_t_arr, loc_x_arr, loc, t_thresh, d_thresh, verbose=False):
    """
    """
    # Get the corresponding indices and ultimately adjust the bin
    mask = bbox_mask(loc_t_arr, loc_x_arr, [loc-d_thresh,loc+d_thresh])
        
    # check if empty    
    if mask.size <= 0: 
        if verbose: print('\t\tmask empty (1)')
        #print(loc_t_arr)
        #print(loc_x_arr, loc-d_thresh, loc, loc+d_thresh)
        return np.empty([])
    if verbose: print("\t\tmask 1:", mask[0], mask[-1])
    
    #tbounds = [loc_t_arr[mask].min(), loc_t_arr[mask].max()]
    #print(tbounds)
    #mask = np.where((loc_t_arr >= loc_t_arr[mask].min()) & \
    #                (loc_t_arr <= loc_t_arr[mask].max()))[0]

    
    # Check if the loc_t_arr comtains gaps > time_thresh; if so, split and repeat    
    gap_mask = check_gaps(loc_t_arr[mask], t_thresh, verbose)
    
    # Get those mask indices corresponding to the longest (sub)cluster
    mask = mask[gap_mask]
    
    # check if empty    
    if mask.size <= 0: 
        if verbose: print('\t\tmask empty (2)')
        return np.empty([])    
    if verbose: 
        print("\t\tmask 2:",mask[0],mask[-1])

    return mask


def update_global_mask(g_mask, arr, lims):
    """
    Update a Boolean array with keeps track of classified events.

    """
    
    #Note: this is easier than tracking & storing the indices, for now.
    
    # Get the indices of the events within the cluster limits
    mask1 = np.where((arr >= lims.min()) & (arr <= lims.max()))
    
    # Store those indices as False (removing the events from processing)
    g_mask[mask1] = False
    
    return g_mask


def get_mask_ends(loc_t_arr, loc_x_arr, loc, t_thresh, d_thresh, verbose=False):
    """
    """
    mask = np.where((loc_x_arr <= loc+1.0*d_thresh) & (loc_x_arr >= loc-1.0*d_thresh))[0]

    diffs = np.diff(loc_t_arr[mask])

    dmask = np.where(diffs>t_thresh)[0].tolist()

    ddmask = [-1] + dmask + [mask.size-1]

    return [[mask[p[0]+1],mask[p[1]]] for p in list(zip(ddmask[:-1], ddmask[1:]))]
