import numpy as np


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


# get_gap_time, get_gap_dist
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


# get_gap_time, get_gap_dist
def gap_criterion_1(t_arr, x_arr, d_thresh, t_thresh):
    """
    Checks whether a spatiotemporal gap between clusters is sufficiently large
    based on the duration of the gap and the location statistics of the adjacent clusters
    """
    def meth(c1, c2):
    
        return ((get_gap_time(t_arr, c1, c2) <= t_thresh) and \
                (get_gap_dist(x_arr, c1, c2) <  d_thresh))
    
    return meth


# get_gap_dist_generic, get_intercluster_metrics
def gap_criterion_2(t_arr, x_arr, d_thresh, t_thresh, min_speed=3.5):
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


# gap_criterion, gap_criterion_2
def gap_criterion_3(t_arr, x_arr, d_thresh, t_thresh, min_speed=3.5):
    """
    Checks whether a spatiotemporal gap between adjacent clusters is large
    based on 
        1. the gap duration and duration of a potential travel-stay-travel
        2. the location statistics of the gap and adjacent clusters    
    """    
    def meth(c1, c2):
        
        return gap_criterion_1(t_arr, x_arr, d_thresh, t_thresh)(c1,c2) \
            or gap_criterion_2(t_arr, x_arr, d_thresh, t_thresh)(c1,c2)

    return meth


def merge_cluster_pair(clusts, gap_ind):
    """
    Merge two clusters which boarder a gap
    
    Notes:
    * the new cluster will be a continuous range of indices
    * The gap index "i" will merge clusters "i" and "i+1"
    """
    
    # Get the clusters up to the gap index
    new_clusts = clusts[:gap_ind].copy()
    
    # Add cluster as a range of indices from the merged clusters
    new_clusts.append(list(range(clusts[gap_ind][0],clusts[gap_ind+1][-1]+1)))
    
    # Add the remaining clusters
    new_clusts.extend(clusts[gap_ind+2:])
    
    return new_clusts


# gap_criterion_3, get_gap_dist, merge_cluster_pair
##def merge_clusters_combo(t_arr, x_arr, clusters, d_thresh,
# gap_criterion, get_gap_dist, merge_cluster_pair
##def merge_clusters(t_arr, x_arr, clusters, d_thresh, # gap_criterion_2, get_gap_dist, merge_cluster_pair
## def merge_clusters_2(t_arr, x_arr, clusters, d_thresh, 
# gap_criterion_2, get_gap_dist, merge_cluster_pair
def merge_clusters_gen(t_arr, x_arr, clusts, d_thresh, t_thresh, min_speed=3.5, verbose=False):
    """
    Iteratively merge clusters in a a list according to criteria regarding the gaps between them
    """
    
    # Using the gap criteria fnt from above
    gap_criterion_fnt = lambda c1, c2: gap_criterion_3(t_arr, x_arr, d_thresh, t_thresh)(c1,c2)
    '''
    NOTE:
    Gerenalized from `merge_clusters_combo`, etc. 
    '''
    
    new_clusts = clusts.copy()
    
    gaps = [gap_criterion_fnt(c1,c2) for c1, c2 in zip(new_clusts[:-1],new_clusts[1:])]
    
    while any(gaps):
        
        # Get spatial distances between all clusters (mean, median, etc.)
        dists = [get_gap_dist(x_arr,c1,c2) for c1, c2 in zip(new_clusts[:-1],new_clusts[1:])]

        # Get index of best (distance-wise) gap to merge
        gaps_add = np.array([100*int(not g) for g in gaps])
        dists_add = np.array(dists)+gaps_add
        min_index = np.argmin(dists_add)
        
        if verbose: print(gaps, "\n", [f"{d:5.3f}" for d in dists], "\n", min_index)
        
        # Merge the cluster with the optimal index
        new_clusts = merge_cluster_pair(new_clusts, min_index).copy()
        
        # Measure gaps and restart loop
        gaps = [gap_criterion_fnt(c1,c2) for c1, c2 in zip(new_clusts[:-1],new_clusts[1:])]        

    if verbose: print(gaps)
    
    return new_clusts
