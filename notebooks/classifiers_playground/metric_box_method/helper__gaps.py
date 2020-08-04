import numpy as np

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
    
    
def gap_criterion(d_thresh, t_thresh):
    """
    Checks whether a spatiotemporal gap between clusters is sufficiently large
    """
    def meth(t_arr,x_arr,c1,c2):
    
        return ((get_gap_time(t_arr, c1,c2) <= t_thresh) and \
                (get_gap_dist(x_arr,c1,c2) < d_thresh))
    
    return meth


def print_metric(m,n,c1, c2, c_main, gap_time, t_thresh, dist, d_thresh):
    
    ints = f"{m:4d} {m:4d}\t"
    
    clusts = f"[{c1[0]:4d},{c1[-1]:5d}], [{c2[0]:4d},{c2[-1]:5d}], [{c_main[0]:4d},{c_main[-1]:5d}]"
    
    #dist = f"{dist:3d}" 
    #dist = f"{dist:5.3f}"
    print(ints, clusts, f"\t{gap_time:5.3f}\t{int(gap_time <= t_thresh):4d}\t", f"{dist:5.3f}\t{int(dist < d_thresh):4d}")
    
    
def merge_cluster_pair(clusters, gap_index):
    """
    Merge two clusters from a specified index
    """
    new_clusters = []
    
    new_clusters.extend(clusters[:gap_index])
    
    new_clusters.append(list(range(clusters[gap_index][0],clusters[gap_index+1][-1]+1)))
    
    new_clusters.extend(clusters[gap_index+2:])
    
    return new_clusters
    

def merge_clusters(t_arr, x_arr, clusters, dist_thresh, time_thresh, verbose=False):
    """
    Iteratively merge clusters in a a list according to criteria regarding the gaps between them
    """
    
    new_clusters = clusters.copy()
    
    gaps = [gap_criterion(dist_thresh, time_thresh)(t_arr,x_arr,c1,c2) for c1, c2 in zip(new_clusters[:-1],new_clusters[1:])]
    
    while any(gaps):
        
        dists = [get_gap_dist(x_arr,c1,c2) for c1, c2 in zip(new_clusters[:-1],new_clusters[1:])]

        gaps_add = np.array([100*int(not g) for g in gaps])
        dists_add = np.array(dists)+gaps_add

        min_index = np.argmin(dists_add)
        
        if verbose: print(gaps, "\n", [f"{d:5.3f}" for d in dists], "\n", min_index)
        
        new_clusters = merge_cluster_pair(new_clusters, min_index).copy()
        
        gaps = [gap_criterion(dist_thresh, time_thresh)(t_arr,x_arr,c1,c2) for c1, c2 in zip(new_clusters[:-1],new_clusters[1:])]        

    if verbose: print(gaps)
    
    return new_clusters


def get_intercluster_metrics(c1, c2, t_arr, x_arr, time_thresh, min_speed=3.6):

    clust = list(range(c1[-1],c2[0]+1))
    
    #t_sub = t_arr[clust]
    #x_sub = x_arr[clust]
    x_med = np.median(x_arr[clust])
    
    gap_time = get_gap_time(t_arr,c1,c2)
    
    gap_d1 = get_gap_dist(x_arr,c1,clust)
    gap_d2 = get_gap_dist(x_arr,c2,clust)
    
    dur1 = gap_d1/min_speed
    dur2 = gap_d2/min_speed
    
    return gap_time, dur1+dur2+time_thresh, x_med


def gap_criterion_2(dist_thresh, time_thresh, min_speed=3.5):
    
    def meth(t_arr, x_arr, c1, c2):
        
        gap_time, min_allowed, x_med = get_intercluster_metrics(c1, c2, t_arr, x_arr, time_thresh)
        
        # if gap_time is less than allowed time for a recognizeable travel-stay-travel, mark for a merge
        # otherwise if the gap is large enough and the medians are near enough, mark for a merge
        if (gap_time < min_allowed) & \
           ((abs(np.median(x_arr[c1]) - x_med) < dist_thresh) & 
            (abs(np.median(x_arr[c2]) - x_med) < dist_thresh)):
            return True
        elif ((abs(np.median(x_arr[c1]) - x_med) < dist_thresh) & 
              (abs(np.median(x_arr[c2]) - x_med) < dist_thresh)):
            return True
        else:
            return False
    
    return meth


def merge_clusters_2(t_arr, x_arr, clusters, dist_thresh, time_thresh, min_speed=3.5, verbose=False):
    """
    Iteratively merge clusters in a a list according to criteria regarding the gaps between them
    """
    
    new_clusters = clusters.copy()
    
    gaps = [gap_criterion_2(dist_thresh, time_thresh)(t_arr,x_arr,c1,c2) for c1, c2 in zip(new_clusters[:-1],new_clusters[1:])]
    
    while any(gaps):
        
        dists = [get_gap_dist(x_arr,c1,c2) for c1, c2 in zip(new_clusters[:-1],new_clusters[1:])]

        gaps_add = np.array([100*int(not g) for g in gaps])
        dists_add = np.array(dists)+gaps_add

        min_index = np.argmin(dists_add)
        
        if verbose: print(gaps, "\n", [f"{d:5.3f}" for d in dists], "\n", min_index)
        
        new_clusters = merge_cluster_pair(new_clusters, min_index).copy()
        
        gaps = [gap_criterion_2(dist_thresh, time_thresh)(t_arr,x_arr,c1,c2) for c1, c2 in zip(new_clusters[:-1],new_clusters[1:])]        

    if verbose: print(gaps)
    
    return new_clusters
