import numpy as np

from .bounding_box_classifier_maxloc import get_max_loc_
from .bounding_box_classifier_masks import get_mask_ends, get_mask, update_global_mask
#from .bounding_box_classifier_gaps import get_gap_dist, merge_cluster_pair, gap_criterion_3, gap_criterion_4


def bounding_box_method(t_arr, x_arr, t_thresh, d_thresh, early_stopping=True, verbose=False):
    """
    """
    get_max_loc = lambda arr: get_max_loc_(d_thresh)(arr)
    
    # Initialize global mask array
    gmask = np.array(x_arr.size*[True])
    last_gmask_size = gmask[gmask].size

    blockarrow = '--> --> --> '
    n = 1
    nn = 0
    clusters = []

    stop_early_counter = 0
    # Loop until there are at most 30 clusters
    while nn < 31:

        if verbose: print(f'{n}: building cluster {nn}')

        # Get the location of the cluster with the max events
        loc = get_max_loc(x_arr[gmask])
        if verbose: print(f"\t{n}.1.1. Loc: {loc:6.3f}")
            
        #Get local mask based on bounding box
        mask = get_mask(t_arr[gmask], x_arr[gmask], loc, 2.5*t_thresh, 0.5*d_thresh, verbose)
        
        # These are the indices where the global mask is True;
        # ie., these promote the local mask to a global mask (see return)
        #print(mask)
        gmask_inds = np.where(gmask)[0]
        mask = gmask_inds[mask]
        
        # check if empty
        if mask.size > 0:
            if verbose: print(f"\t{n}.1.2. Mean: {np.mean(x_arr[mask]):6.3f}, median: {np.median(x_arr[mask]):6.3f}")
            if verbose: print(f"\t{n}.1.3. Mask: [{mask[0]:5d}, {mask[-1]:5d}], Mask size: {mask.size:2d}")

            if verbose: print(f"\t{n}.2. Bounds: {t_arr[mask].min():6.3f}, {t_arr[mask].max():6.3f}")                
            # Create the bounding box (needed? if wanting to optimize the box, do here)
            bounds = np.array([t_arr[mask].min(), t_arr[mask].max()])            
            
            # Check the duration, save the cluster if it is long enough
            if t_arr[mask].max()-t_arr[mask].min()>t_thresh:
                clusters.append(mask)
                # This is a fail safe in case it goes to far: 
                #    * no one will have 20 clusters --> make smaller!                
                nn+=1
                if verbose: print(f"{blockarrow}Appending cluster")
                stop_early_counter = 0
            else:
                if verbose: print(f"{blockarrow}Too short, no append: {(t_arr[mask].max()-t_arr[mask].min()):6.4f} < {t_thresh:6.4f}")
                stop_early_counter += 1
        else:
            # Need to add even a single point to the global mask; 
            # breaking would exit prematurely
            if verbose: print(f"\n{blockarrow}Too short, no append: single event")
            if verbose: print(f"{blockarrow}Breaking due to single-event mask")                
            break # Can't continue: otherwise would endlessly loop
            
        # Update the global mask
        gmask = update_global_mask(gmask, t_arr, bounds)
        
        if verbose: print(f"\t{n}.3. gmask:",gmask.size, "==", gmask[gmask].size,"+",gmask[gmask==False].size)
        
        # Stop when there are more than 3 consecutive append-skips
        if (early_stopping) & (stop_early_counter >= 3):
            if verbose: print(f"\n{blockarrow}Early stopping")
            break
        
        # Stop when there are no more events available
        if gmask[gmask==False].size == gmask.size: 
            if verbose: print(f"\n{blockarrow}Breaking due to saturation")
            break
            
        n+=1
        
        if verbose: print()
    
    return clusters


def bounding_box_method_v3(t_arr, x_arr, t_thresh, d_thresh, verbose = False):
    """
    """
    
    get_max_loc = lambda arr: get_max_loc_(d_thresh)(arr)

    # Initialize global mask array
    gmask = np.array(x_arr.size*[True])

    blockarrow = '--> --> --> '

    cluster_counter = 1
    failsafe_counter = 1
    loop_counter = 0

    clusters = []
    
    while loop_counter < 30:

        if verbose: print(f'{failsafe_counter}: Finding cluster {cluster_counter}')

        # Get the location of the cluster with the max events
        loc = get_max_loc(x_arr[gmask])
        if verbose: print(f"\t{failsafe_counter:3d}.1.   Loc: {loc:6.3f}")

        #Get local mask based on bounding box; 
        # split into sub mask endpoints based on temporal gaps in the mask
        mask_ends = get_mask_ends(t_arr[gmask],x_arr[gmask], loc, t_thresh, 0.5*d_thresh)
        
        # Change the mask endpoints into subsequences in the global indices, 
        # which are the indices where the global mask is True;
        # ie., these promote the local mask to a global mask (see return)
        gmask_inds = np.where(gmask)[0]    

        subclusts = []
        internal_count = 1
        for p in mask_ends:
            # create the subseq
            subc = list(range(gmask_inds[p[0]],gmask_inds[p[1]]+1))
            if verbose: print(f"\t{failsafe_counter:3d}.2.{internal_count}. Mask ends: [{gmask_inds[p[0]]:5d},{gmask_inds[p[1]]:5d}]")
            internal_count += 1                
            
            # store the subsequences
            subclusts.append(subc)

        clust_len0 = len(subclusts)
        # Check and merge the subclusters as needed
        subclusts = merge_clusters_gen_x(t_arr, x_arr, subclusts, d_thresh, t_thresh, 3.5, verbose)
        clust_len1 = len(subclusts)
        if verbose: print(f"\t{failsafe_counter:3d}.3.   Merged {clust_len0-clust_len1} subclusters ")
        
        # Add new clusters if long enough
        clust_len1 = len(clusters)
        internal_count=1
        appending = True
        for c in subclusts:
            # Rescind availability from the global index array
            ### NOTE: drop the global indices here or _before_ the merging?
            gmask[c] = False
     
            # Check the duration, save the cluster if it is long enough
            if t_arr[c].max()-t_arr[c].min()>t_thresh:

                if verbose: 
                    if appending: print(f"{blockarrow}Appending cluster(s)") 
                    print(f"\t{failsafe_counter:3d}.4.{internal_count}. New clust: [{c[0]:5d},{c[-1]:5d}], t = [{t_arr[c[0]]:7.3f},{t_arr[c[-1]]:7.3f}]")
                    print(f"\t\t Mean: {np.mean(x_arr[c]):6.3f}, " \
                                  f"median: {np.median(x_arr[c]):6.3f}")
                internal_count += 1
                clusters.append(c)    
                # This is a fail safe in case it goes to far: 
                #    * no one will have 20 clusters --> make smaller!                
                cluster_counter += 1
                appending = False

        if clust_len1 == len(clusters):
            if verbose: print(f"{blockarrow}No clusters appended")

            # Check the remaining subsequences; if they're too short, drop them?
            gdiffs = np.diff(gmask.astype(int))
            gtrans = np.where(gdiffs!=0)[0] + 1
            drop=False
            for m,mm in zip(gtrans[:-1:2],gtrans[1::2]):
                tdiff = t_arr[mm-1]-t_arr[m]
                if tdiff <= t_thresh:
                    gmask[m:mm]=False
                    drop=True
            dropped=""
            if drop:
                dropped=": dropped"
            if verbose: print(f"\tChecking for short subsequences"+dropped)

        if verbose: print(f"\t{failsafe_counter:3d}.5.   gmask:",gmask.size, "==", gmask[gmask].size,"+",gmask[gmask==False].size)
        # Stop when there are no more events available
        if gmask[gmask==False].size == gmask.size: 
            if verbose: print(f"\n{blockarrow}Breaking due to saturation")
            break

        # This is a fail safe in case it goes to far: 
        #    * no one will have 50 clusters --> make smaller!  
        loop_counter += 1
        failsafe_counter += 1

        if verbose: print()    
            
    return clusters

# NOTE: keeping this here as it uses `bounding_box_method`
def drop_mini_stays(t_arr, x_arr, clusts, t_thresh, d_thresh, frac=0.8, verbose=False):
    """
    """
    max_iterations = 10
    
    n = 0
    
    new_clusters = []
    
    for c in clusts:

        tdiff = abs(t_arr[c[-1]]-t_arr[c[0]])

        if tdiff < 2*t_thresh:

            if verbose: print('check',n, f'{tdiff:7.4f} [{c[0]:5d}, {c[-1]:5d}]')

            if (n > 0) & (n < len(clusts)-1):
                
                if verbose: print(clusts[n-1][-1],clusts[n+1][0])
                    
                mask = list(range(clusts[n-1][-1],clusts[n+1][0]+1))
            
                score = 0
                for _ in range(max_iterations):
                    
                    #print(len(mask), mask)
                    masko = np.sort(np.random.choice(mask, int(frac*len(mask)), replace=False)).tolist()
                    #print(len(mask), mask)
                    masko = bounding_box_method(t_arr[masko],x_arr[masko],t_thresh, d_thresh, True, False )
                    
                    score += int(len(masko)==0) # if masko is emtpy, +1
                    #print(len(masko), len(masko)==0, score, score>=3)
                n += 1

                if score>int(max_iterations/2): 
                    if verbose: print("Drop")
                    continue
                else:
                    if verbose: print("Okay",len(mask))

        new_clusters.append(c)

        n += 1

    return new_clusters
