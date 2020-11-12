import numpy as np

def check_gaps(loc_t_arr, t_thresh, verbose=False):
    """
    """
    # If the masked array contains gaps larger than the thresh
    # split the mask and take the submask which corresponds to
    # the largest duration subarray.
    
    # If there's not enough evnets for a gap, exit
    if loc_t_arr.size < 2:
        if verbose: print("\t\t\tNo gaps: too short,",loc_t_arr.size)
        return list(range(loc_t_arr.size))
    
    # Check the time-gaps between points
    diffs = np.diff(loc_t_arr)
    
    # Check the time differences between all events
    if diffs.max() > t_thresh:
        # ... and split the chain based oan the largest t-diff
        if verbose: print(f"\t\t\tSplitting cluster")
        split_ind = np.where(diffs>t_thresh)[0].tolist()                
    else:
        if verbose: print(f"\t\t\tNo gaps")    
        return list(range(loc_t_arr.size)) 
        
    # Augment to get the beginning and ending indices
    split_ind = [-1] + split_ind + [loc_t_arr.size-1]
    
    # Get the durations of the subarrays
    duras = []
    for m,n in zip(split_ind[:-1],split_ind[1:]):
        diff = loc_t_arr[n]-loc_t_arr[m+1]
        if verbose: print(f"\t\t\t\t{diff:6.3f},[{m+1},{n}]")
        duras.append(diff)
    
    # Get the index of the subarray with the max duration
    # TODO: can get _all_ clusters > t_thresh, not just max?
    max_dura_ind = duras.index(max(duras))
    if verbose: print(f"\t\t\t\tlongest subcluster: {max(duras):6.3f}, index: {max_dura_ind}")
    
    # Get the starting index of the max. subarray 
    # (add 1 since the array starts -1 behind the gaps)
    output = np.arange(split_ind[max_dura_ind  ]+1,
                       split_ind[max_dura_ind+1]+1)
    return output


def check_gaps_check(loc_t_arr, t_thresh, verbose=False):
    """
    """
    # If there's not enough evnets for a gap, exit
    if loc_t_arr.size < 2:
        if verbose: print("\t\t\t\tNo gaps: too short,",loc_t_arr.size)
    
    # Check the time-gaps between points
    diffs = np.diff(loc_t_arr)
    
    # Check the time differences between all events
    duras = []    
    if np.any(diffs > t_thresh):
        #print(np.diff(loc_t_arr))
        # ... and split the chain based oan the largest t-diff
        gaps = np.where(diffs > t_thresh)[0]
        if verbose: print(f"There are {gaps.size} gaps: {diffs[gaps]}")
        split_ind = np.where(diffs>t_thresh)[0].tolist()                
            
        # Augment to get the beginning and ending indices
        split_ind = [-1] + split_ind + [loc_t_arr.size-1]
        
        # Get the durations of the subarrays
        duras = []
        for m,n in zip(split_ind[:-1],split_ind[1:]):
            diff = loc_t_arr[n]-loc_t_arr[m+1]
            if verbose: print(f"\t\t\t\t{diff:6.3f},[{m+1},{n}]")
            if diff > t_thresh:
                
                duras.append([m+1,n])

        #print(duras)
    else:
        if verbose: print("\t\t\t\tNo gaps")

    return duras


# TODO: check if and where this is still needed
def get_gap_metrics(c1, c2, t_arr, x_arr, time_thresh, min_speed=3.6):
    """
    Calculate some metrics for a gap: 
        * the duration of the gap, 
        * the minimal possible duration of the gap to include a stay
        * a location statistic of the gap
    """
    
    # Get the gap indices between adjacent clusters
    gap = list(range(c1[-1],c2[0]+1))
    gap_len = len(gap)
    interstay_dist_medi = abs(np.median(x_arr[c1])-np.median(x_arr[c2]))
    interstay_dist_mean = abs(np.mean(x_arr[c1])-np.mean(x_arr[c2]))
    
    # Returns
    ## Calulate the duration between adjacent clusters
    gap_time = get_gap_time(t_arr,c1,c2)

    ## Calculate the minimal time between two clusters to include two travels and a stay
    dur1 = get_gap_dist(x_arr,c1, gap)/min_speed
    dur2 = get_gap_dist(x_arr,c2, gap)/min_speed
    
    ## Calculate a statistic (ie the median) of the events contained in a gap
    gap_mean   = np.mean(x_arr[gap])
    gap_median = np.median(x_arr[gap])
    '''
    print(f"{gap_len:4d}, {gap_time:6.3f}, {interstay_dist_medi:6.3f}, {interstay_dist_mean:6.3f}, {gap_mean:6.3f}, {gap_median:6.3f}")
    '''
    warning = ""
    if gap_time < time_thresh: 
        warning = "WARNING: too short"
    
    return f"{gap_len:4d}, {gap_time:6.3f}, {interstay_dist_medi:6.3f}, {interstay_dist_mean:6.3f}, {gap_mean:6.3f}, {gap_median:6.3f}  {warning}"


# TODO: check if and where this is still needed
def get_clust_metrics(gap, t_arr, x_arr, time_thresh, min_speed=3.6):
    """
    Calculate some metrics for a gap: 
        * the duration of the gap, 
        * the minimal possible duration of the gap to include a stay
        * a location statistic of the gap
    """
    
    # Get the gap indices between adjacent clusters
    gap_len = len(gap)

    
    # Returns
    ## Calulate the duration between adjacent clusters
    gap_time = t_arr[gap[-1]]-t_arr[gap[0]]

    ## Calculate the minimal time between two clusters to include two travels and a stay
    #dur1 = get_gap_dist(x_arr,c1, gap)/min_speed
    #dur2 = get_gap_dist(x_arr,c2, gap)/min_speed
    
    ## Calculate a statistic (ie the median) of the events contained in a gap
    gap_mean   = np.mean(x_arr[gap])
    gap_median = np.median(x_arr[gap])
    
    return f"{gap_len:4d}, {gap_time:6.3f},   ----,   ----, {gap_mean:6.3f}, {gap_median:6.3f}"


# TODO: check if and where this is still needed
def print_metric(m, n, c1, c2, c_main, gap_time, t_thresh, dist, d_thresh):
    """
    """
    ints = f"{m:4d} {m:4d}\t"
    
    clusts = f"[{c1[0]:4d},{c1[-1]:5d}], [{c2[0]:4d},{c2[-1]:5d}], [{c_main[0]:4d},{c_main[-1]:5d}]"
    
    #dist = f"{dist:3d}" 
    #dist = f"{dist:5.3f}"
    print(ints, clusts, f"\t{gap_time:5.3f}\t{int(gap_time <= t_thresh):4d}\t", f"{dist:5.3f}\t{int(dist < d_thresh):4d}")


def get_gap_time(t_arr,c1,c2):
    """
    Gets the temporal difference between two neighboring gaps
    """
    return abs(t_arr[c2[0]]-t_arr[c1[-1]])


def get_gap_dist(x_arr,c1,c2):
    """
    Gets the spatial difference between two medians of neighboring gaps
    """
    return abs(np.median(x_arr[c1])-np.median(x_arr[c2]))
    

def get_gap_dist_generic(sub_arr, measurement):
    """
    Gets the spatial difference between a median and another measure of neighboring clusters
    """
    return abs(np.median(sub_arr)-measurement)


# NOTE: uses get_gap_time, get_gap_dist
def get_intercluster_metrics(c1, c2, t_arr, x_arr, t_thresh, min_speed=3.6):
    """
    Calculate some metrics for a gap: 
        * the duration of the gap, 
        * the minimal possible duration of the gap to include a stay
        * a location statistic of the gap
    """
    
    # Get the gap indices between adjacent clusters
    gap = list(range(c1[-1],c2[0]+1))
    
    # Returns
    ## Calulate the duration between adjacent clusters
    gap_time = get_gap_time(t_arr,c1,c2)

    ## Calculate the minimal time between two clusters to include two travels and a stay
    dur1 = get_gap_dist(x_arr,c1, gap)/min_speed
    dur2 = get_gap_dist(x_arr,c2, gap)/min_speed
    
    ## Calculate a statistic (ie the median) of the events contained in a gap
    x_stat = np.median(x_arr[gap])
    
    return gap_time, dur1+dur2+t_thresh, x_stat


# NOTE: uses get_gap_time, get_gap_dist
def gap_criterion_1(t_arr, x_arr, t_thresh, d_thresh):
    """
    Checks whether a spatiotemporal gap between clusters is sufficiently large
    based on the duration of the gap and the location statistics of the adjacent clusters
    """
    def meth(c1, c2):
    
        return ((get_gap_time(t_arr, c1, c2) <= t_thresh) and \
                (get_gap_dist(x_arr, c1, c2) <  d_thresh))
    
    return meth


# NOTE: uses get_gap_dist_generic, get_intercluster_metrics
def gap_criterion_2(t_arr, x_arr, t_thresh, d_thresh, min_speed=3.5):
    """
    Checks whether a spatiotemporal gap between adjacent clusters is large
    based on 
        1. the gap duration and duration of a potential travel-stay-travel
        2. the location statistics of the gap and adjacent clusters    
    """    
    def meth(c1, c2):
        
        gap_time, min_allowed, x_med = get_intercluster_metrics(c1, c2, t_arr, x_arr, t_thresh)
        
        # mark the gap for a merge when:
        # * the gap_time is less than allowed time for a travel-stay-travel and/or
        # * the gap is large enough and the medians are near enough, mark for a merge
        if ((get_gap_dist_generic(x_arr[c1], x_med) < d_thresh) & \
            (get_gap_dist_generic(x_arr[c2], x_med) < d_thresh)):
            return True        
        else:
            return False
    
    return meth


# NOTE: uses gap_criterion, gap_criterion_2
def gap_criterion_3(t_arr, x_arr, t_thresh, d_thresh, min_speed=3.5):
    """
    Checks whether a spatiotemporal gap between adjacent clusters is large
    based on 
        1. the gap duration and duration of a potential travel-stay-travel
        2. the location statistics of the gap and adjacent clusters    
    """    
    def meth(c1, c2):
        
        return gap_criterion_1(t_arr, x_arr, t_thresh, d_thresh)(c1,c2) \
            or gap_criterion_2(t_arr, x_arr, t_thresh, d_thresh)(c1,c2)

    return meth


# NOTE: uses gap_criterion, gap_criterion_2
def gap_criterion_4(t_arr, x_arr, t_thresh, d_thresh, min_speed=3.5):
    """
    Checks whether a spatiotemporal gap between adjacent clusters is large
    based on 
        1. the gap duration and duration of a potential travel-stay-travel
        2. the location statistics of the gap and adjacent clusters    
    """    
    def meth(c1, c2):
        
        return gap_criterion_1(t_arr, x_arr, t_thresh, d_thresh)(c1,c2) \
            and gap_criterion_2(t_arr, x_arr, t_thresh, d_thresh)(c1,c2)

    return meth
