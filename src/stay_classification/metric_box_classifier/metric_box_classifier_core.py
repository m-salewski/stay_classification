import numpy as np

#from synthetic_data.trajectory import get_stay_indices, get_adjusted_stays
#from sklearn.metrics import precision_score, recall_score, confusion_matrix


from .metric_box_classifier_gaps import gap_criterion_3, merge_cluster_pair

#from helper__3stays_v3_scripts import inter_bounds, contains, conta_bounds

get_cluster_ranges = lambda clusters: [list(range(c[0],c[-1]+1)) for c in clusters]
get_sorted_clusters = lambda clusters: sorted([sorted(c) for c in clusters]) 

'''
~~get_clusters_3 --> get_mini_clusters~~
print_clusts
get_extended_clusters
separate_clusters_hier
merge_clusters_combo
shift_box
get_iqr_trimmed_clusters
'''


get_err = lambda x1, x2: abs(x1-x2) #np.sqrt((x1-x2)**2)

#gap_criterion_3, merge_cluster_pair
def get_mini_clusters(t_arr, x_arr, d_thresh, t_thresh, verbose=False):
    """
    Get a list of cluster indices

    :param t_arr: np.array Trajectory array of timepoints
    :param t_arr: np.array Trajectory array of locations
    :param d_thresh: float temporal buffer between timepoints 
    :param t_thresh: float spatial buffer around clusters      
    
    :return: [int] indices of identified clusters
    
    TODOs: 
    * Update the debugging output
    * rename `gap_criterion_3` to `gap_criterion` (usw)
        * `gap_criterion` then calls `gap_criterion_1` & `gap_criterion_2`, etc. 
    """
    
    # Want to use this here to keep the scope variables
    def check_and_merge_clusters(clusts):

        # Check whether the new cluster is can be merged with the previous one
        c1, c2 = clusts[-2],clusts[-1]

        if gap_criterion_3(t_arr, x_arr, d_thresh, t_thresh)(c1, c2):            
            # Keep `min_index`? Here is it fixed
            min_index = len(clusts)-2
            return merge_cluster_pair(clusts, min_index).copy()  
        else:
            return clusts

        
    # Output list of indices: [[beg., end],[beg., end], ...] 
    clusters = []
    
    # Initialize working indices
    m = 0    
    new_cluster = [m]
    
    # Pass through the list of events    
    for n in range(1, x_arr.size):
        
        # Check: is the time within the time thresh?
        # NOTE: this will keep going until it finds a timepoint pair 
        # within the thresh. 
        # If there's an open cluster, that cluster will get closed
        # at the last added point once no new points can be added 
        # based on the distance criteria. 
        # This can allow already for some long-duration clusters
        # as long as they don't exceed the distance threshold.
        if abs(t_arr[n] - t_arr[n-1]) > t_thresh:
            if n >= x_arr.size-1:
                pass
            else:
                continue
        elif n == x_arr.size-1:
            # This ensures that will keep the last event
            pass
        
        # Get the last event's location
        if n < x_arr.size-1:
            new_x = x_arr[n].reshape(1,)
        else:
            new_x = x_arr[n-1].reshape(1,)
            
        # Get the current cluster's mean
        cluster_mean = np.mean(x_arr[m:n])

        # Get the potential cluster's mean    
        new_cluster_mean = np.mean(\
            np.concatenate([x_arr[m:n],new_x])\
                )        

        err1 = get_err(cluster_mean, new_x)
        err2 = get_err(cluster_mean, new_cluster_mean)
                
        # Checks: 
        # 1. new event is within dist. thresh of current clust.
        # 2. new mean - current mean is within dist. thresh.
        if  ((err1 < d_thresh) & (err2 < d_thresh)) & (n < x_arr.size):
            new_cluster.append(n)        
        else:
            # Save the current cluster and prepare restart
            if (len(new_cluster) >= 2):
                # Since this is an incomplete method (clusters will be merged after), 
                # can keep this; otherwise, would lose small clusters
                #if (abs(t_arr[new_cluster[-1]]-t_arr[new_cluster[0]]) > t_thresh):
                clusters.append(new_cluster)
                if verbose: print(f"\tAppended: [{new_cluster[0]:4d}, {new_cluster[-1]:4d}]")

                # Check whether the new cluster is can be merged with the previous one
                # NOTE: this combines the logic from `merge_clusters` function
                if len(clusters) > 1:
                    clusters = check_and_merge_clusters(clusters)
                    if (clusters[-1][0] != new_cluster[0]) & verbose:
                        print("\t\tMerged")
                        
            # Update starting point 
            # TODO: n or n+1?
            m=n
            
            # TODO: start with m?
            new_cluster = [m]
            
        # If there is an open cluster and the limit is reached, append it.
        if verbose: print(n, len(new_cluster))
        if (n >= x_arr.size-2) & (len(new_cluster) > 0):
            clusters.append(new_cluster)
            if verbose: print(f"\tFinal append: [{new_cluster[0]:4d}, {new_cluster[-1]:4d}]")
            # and check and merge it
            clusters = check_and_merge_clusters(clusters)
            if (clusters[-1][0] != new_cluster[0]) & verbose:
                print("\t\tFinal merge")
            new_cluster = []
            
    return clusters
