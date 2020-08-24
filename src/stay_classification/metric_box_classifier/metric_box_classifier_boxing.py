import numpy as np
from ..box_classifier.box_method import get_directional_indices

def _get_iqr(data):
    
    q25 = np.quantile(data, 0.25, interpolation='lower')
    q75 = np.quantile(data, 0.75, interpolation='higher')
    return q25, q75


def get_iqr_mask_x(sub_arr, offset, iqr_bounds, iqr_fact = 1.5, within=True, verbose=False):
    """
    """
    
    #TODO: rename `x_arr` --> `sub_arr`, `cluster` --> `offset` 
    
    # Mask to include only events within the IQR
    if iqr_bounds != None:
        #if verbose: print("Getting input bounds", iqr_bounds)
        q25, q75 = iqr_bounds
    else:
        q25, q75 = _get_iqr(sub_arr)
    
    iqr = abs(q75 - q25)
        
    if within:
        mask = np.where((sub_arr >= (q25 - iqr_fact * iqr)) & (sub_arr <= (q75 + iqr_fact * iqr)))
        
    else:
        mask =  np.where((sub_arr < (q25 - iqr_fact * iqr)) | (sub_arr > (q75 + iqr_fact * iqr)))
    
    mask[0][:] += offset
    
    return mask


def get_time_ind(t_arr, index, t_thresh, direction, verbose=False):
    """
    Get the index of a region bounded by a timepoint +/- a buff

    :param t_arr: np.array Trajectory array of timepoints
    :param timepoint: float Timepoint
    :param t_thresh: float buff around timepoint  
    :param direction: int gets the min or max index of a region
    
    :return: int endpoint index of a region
    
    """
    
    timepoint = t_arr[index]
    
    indices = np.array([[]])
    n = 1

    within_limits = True
    while ((within_limits) & (indices[0].size == 0)):
        
        indices = get_directional_indices(t_arr, timepoint, n*t_thresh, direction)
        
        # Ensure the moving edge is within the time array
        if direction > 0:
            within_limits = timepoint + direction*n*t_thresh <= t_arr.max()
        else:
            within_limits = timepoint + direction*n*t_thresh >= t_arr.min()
       
        n+=1

    # if there are indices, get the min/max per direction
    if indices[0].size != 0:
        #if verbose: print("\t\tget_time_ind",indices[0].max(), indices[0].min())
        if direction == 1:
            return indices[0].max() 
        else: 
            return indices[0].min()
    else:
        #If size == 0 ==> while exited due to limits
        #if verbose: print("\t\tget_time_ind", "min", "max")
        if direction == 1:
            return t_arr.shape[0]-1
        else: 
            return 0


def extend_clusters(t_arr, x_arr, clusters, t_thresh, verbose=False):
    """
    """
    new_clusts = []

    i = 0 
    for c in clusters:

        cc = get_iqr_mask_x(x_arr[c], c[0], (x_arr[c].min(), x_arr[c].max()), 0, True, verbose)[0]
        
        # Get cluster
        if verbose: 
            
            if cc.size > 0:
                iqr_str = f"IQR: [{cc.min():4d}, {cc.max():4d}]"
            else:
                iqr_str = f"IQR: [ nan,  nan]"            
            
            print(f"Cluster #{i+1}\n\t",\
            f"indices: [{c[0]:4d}, {c[-1]:4d}], length: {len(c)}, ",\
            f"bounds: [{x_arr[c].min():6.3f}, {x_arr[c].max():6.3f}], ",\
            f"x-width: {abs(x_arr[c].max()-x_arr[c].min()):6.3f}",\
            f"{iqr_str}")
        
        # If cluster is too small, ignore it
        if len(c) < 2:
            continue

        # extend clust backwards w.r.t. time
        if verbose: print("    Backwards")

        ext_clust_bwd = np.array([])

        # Get proposed index
        work_ind = get_time_ind(t_arr, c[0], 2*t_thresh, -1, verbose)
        # Get previous   index
        prev_work_ind = c[0]
        if verbose: print(f"\t1.1. [{work_ind:4d}, {c[-1]+1:4d}], " + \
                          f"new: {work_ind:4d}, last: {prev_work_ind:4d}") 
        
        keep_going = True
        while keep_going:

            # Get the indices for the extended box
            cc = get_iqr_mask_x(x_arr[work_ind:c[-1]+1], work_ind, (x_arr[c].min(), x_arr[c].max()), 0, True)[0]
            ext_clust_bwd = cc.copy()
            
            #if verbose: print("\t2. clust size:", cc.size)
            if len(cc) < 1:
                if verbose: print("\tlength break")
                
                break
            
            if verbose: print(f"\t1.2. [{cc[0]:4d}, {cc[-1]:4d}], " + \
                              f"new: {work_ind:4d}, last: {prev_work_ind:4d}") 

            
            if cc[0] != prev_work_ind:
                if verbose: print(f"\t\tnot equal: {cc[0]:4d} =/= {prev_work_ind:4d}")
                prev_work_ind = cc[0]
            else:
                if verbose: print("\tno change break")                
                break
            work_ind = get_time_ind(t_arr, prev_work_ind, 2*t_thresh, -1, verbose)
            
        if ext_clust_bwd.size > 1:
            if verbose: print(f"\t1.3. [{ext_clust_bwd[0]:4d}, {ext_clust_bwd[-1]:4d}], " + \
                              f"new: {work_ind:4d}, last: {prev_work_ind:4d}")

        if len(cc) > 1:   
            if cc[-1]!=prev_work_ind:
                prev_work_ind = cc[-1]
        work_ind = get_time_ind(t_arr, work_ind, 2*t_thresh, 1)


        # extend clust forwards w.r.t. time
        if verbose: print("    Forwards")

        work_ind = get_time_ind(t_arr, c[-1], 2*t_thresh, 1, verbose)
        prev_work_ind = c[-1]
        if verbose: print(f"\t2.1. [{c[0]:4d},{c[-1]:4d}], " + \
            f"new: {work_ind:4d}, last: {prev_work_ind:4d}") 

        ext_clust_fwd = np.array([])

        keep_going = True
        while keep_going:

            cc = get_iqr_mask_x(x_arr[c[0]:work_ind+1], c[0], (x_arr[c].min(), x_arr[c].max()), 0, True)[0]    
            
            ext_clust_fwd = cc.copy()
            
            #if verbose: print("\t2. clust size:", cc.size)
            if len(cc) < 1:
                if verbose: print("\tlength break")
                break
            if verbose: print(f"\t2.2. [{cc[0]:4d},{cc[-1]:4d}], " + \
                              f"new: {work_ind:4d}, last: {prev_work_ind:4d}") 
     

            if cc[-1] != prev_work_ind:
                if verbose: print(f"\t\tnot equal: {cc[-1]:4d} =/= {prev_work_ind:4d}")
                prev_work_ind = cc[-1]
            else:
                if verbose: print("\tno change break")
                break

            work_ind = get_time_ind(t_arr, prev_work_ind, 2*t_thresh, 1, verbose)

        # Finalization
        new_clust = []
        
        #if verbose: print("sizes:", ext_clust_bwd.size, ext_clust_fwd.size)
        
        if (ext_clust_bwd.size > 0) and (ext_clust_fwd.size > 0):
            ext_clust_fwd = np.concatenate([ ext_clust_bwd.reshape(-1,), ext_clust_fwd.reshape(-1,)])
            new_clust = np.unique(ext_clust_fwd)            

            final_report = f"\nFinal clust: length = {new_clust.size:4d};"\
                           f" range = [{new_clust[0]:4d},{new_clust[-1]:4d}];"\
                           f" working index = {work_ind:4d};"
            dropped = ""
            new_clust = new_clust.tolist()
    
        else:
            final_report = f"\n\tFinal clust: length = {ext_clust_fwd.size:4d}"
            dropped = "(1)Dropped: "
            
        duration = 0
        if (len(new_clust) > 0):
            duration = abs(t_arr[new_clust[-1]]-t_arr[new_clust[0]])        
        final_report += f" duration = {duration:6.3f}\n"
        
        
        if (len(new_clust) > 0) and (duration > t_thresh):
            new_clusts.append(new_clust)
        else:
            dropped = f"(2)Dropped: "#{len(new_clust) > 0):4d} or duration > t_thresh = {duration > t_thresh}"

        if verbose: print(dropped+final_report)            
        
        i += 1
        #print()

    return new_clusts


def get_mask(t_arr, x_arr, clust, mod_t_thresh, direction):

    iqr_fact = 0 #1.5
    
    q25, q75 = _get_iqr(x_arr[clust])

    iqr = abs(q75 - q25)

    # Fwd-shifted bounds
    lower_ind = get_time_ind(t_arr, clust[ 0], mod_t_thresh, direction)
    upper_ind = get_time_ind(t_arr, clust[-1], mod_t_thresh, direction)

    sub_arr = x_arr[lower_ind:upper_ind+1].copy()
    
    mask = np.where((sub_arr >= (q25 - iqr_fact * iqr)) & (sub_arr <= (q75 + iqr_fact * iqr)))[0]

    return mask + lower_ind


def shift_cluster_box(t_arr, x_arr, clusts, t_thresh, d_thresh, verbose=False):
    """
    """

    t_thresh_fact = 1.0
    
    within=True
    
    iqr_fact = 0 #1.5
    
    new_clusts = []
    
    for c in clusts:
    
        #print(c[-1]-c[0], len(c)-1)
        offset = c[0]
            
        # Bwd-shifted bounds
        bwd_mask = get_mask(t_arr, x_arr, c, t_thresh_fact*t_thresh, -1)
        
        strg = f"[{c[0]:6d}, {c[-1]:6d}]: {np.mean(x_arr[c]):6.3f}\t"
        
        if bwd_mask.size > 0:        
            bwd_mean = np.mean(x_arr[bwd_mask])
        else:
            bwd_mean = np.nan
            
        #strg += f"BWD: [{lower_ind:6d}, {upper_ind:6d}]: {len(bwd_mask):5d}: {bwd_mean:6.3f}\t"
        strg += f"BWD: {len(bwd_mask):5d}: {bwd_mean:6.3f}\t"
        
        # Fwd-shifted bounds
        fwd_mask = get_mask(t_arr, x_arr, c, t_thresh_fact*t_thresh, 1)
        
        if fwd_mask.size > 0:
            fwd_mean = np.mean(x_arr[fwd_mask])
        else:
            fwd_mean = np.nan
        
        #strg += f"FWD: [{lower_ind:6d}, {upper_ind:6d}]: {len(fwd_mask):5d}: {fwd_mean:6.3f}\t\t{abs(bwd_mean - fwd_mean) < 0.5*d_thresh}"
        strg += f"FWD: {len(fwd_mask):5d}: {fwd_mean:6.3f}\t\t{abs(bwd_mean - fwd_mean) < 0.5*d_thresh}"
        if verbose: print(strg)
        
        if (len(bwd_mask) > 0 ) & (len(fwd_mask) > 0):
            if abs(bwd_mean - fwd_mean) < 0.5*d_thresh:
                new_clusts.append(c)
            
    return new_clusts
