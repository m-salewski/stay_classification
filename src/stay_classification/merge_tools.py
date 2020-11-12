import numpy as np

from .gap_tools import gap_criterion_3, gap_criterion_4, get_gap_dist

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
def merge_clusters_gen(t_arr, x_arr, clusts, t_thresh, d_thresh, min_speed=3.5, verbose=False):
    """
    Iteratively merge clusters in a a list according to criteria regarding the gaps between them
    """
    
    # Using the gap criteria fnt from above
    gap_criterion_fnt = lambda c1, c2: gap_criterion_3(t_arr, x_arr, t_thresh, d_thresh)(c1,c2)
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


def merge_clusters_gen_xx(t_arr, x_arr, clusts, t_thresh, d_thresh, min_speed=3.5, verbose=False):
    """
    Iteratively merge clusters in a a list according to criteria regarding the gaps between them
    """
    #if verbose: print('\t\t\t\t\tMerge?')
    # Using the gap criteria fnt from above
    gap_criterion_fnt = lambda c1, c2: gap_criterion_4(t_arr, x_arr, t_thresh, d_thresh)(c1,c2)
    '''
    NOTE:
    Gerenalized from `merge_clusters_combo`, etc. 
    '''
    new_clusts = clusts.copy()
    
    if verbose: 
        n = 1
        for c1, c2 in zip(new_clusts[:-1],new_clusts[1:]):
            print(f'{f"Gap{n}":5s}: {(t_arr[c2[0]]-t_arr[c1[-1]]):7.3f}: {gap_criterion_fnt(c1,c2)}')
            n+=1
    
    gaps = [gap_criterion_fnt(c1,c2) for c1, c2 in zip(new_clusts[:-1],new_clusts[1:])]
    
    while any(gaps):
        
        # Get spatial distances between all clusters (mean, median, etc.)
        dists = [get_gap_dist(x_arr,c1,c2) for c1, c2 in zip(new_clusts[:-1],new_clusts[1:])]

        # Get index of best (distance-wise) gap to merge
        gaps_add = np.array([100*int(not g) for g in gaps])
        dists_add = np.array(dists)+gaps_add
        min_index = np.argmin(dists_add)

        if verbose: print("\t\t\t",gaps, "\n\t\t\t", [f"{d:5.3f}" for d in dists], "\n\t\t\t", min_index)
        
        # Merge the cluster with the optimal index
        new_clusts = merge_cluster_pair(new_clusts, min_index).copy()
        
        # Measure gaps and restart loop
        gaps = [gap_criterion_fnt(c1,c2) for c1, c2 in zip(new_clusts[:-1],new_clusts[1:])]        

    if verbose: print("\t\t\tMerged",gaps)
    
    return new_clusts
