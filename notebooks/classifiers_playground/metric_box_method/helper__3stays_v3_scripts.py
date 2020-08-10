import numpy as np
#from numpy import mean as np.mean

def get_clusters_x(t_arr, loc_arr, dist_thresh, time_thresh, verbose=False):
    """
    Get a list of cluster indices

    :param t_arr: np.array Trajectory array of timepoints
    :param timepoint: float Timepoint
    :param time_thresh: float buff around timepoint  
    :param direction: int gets the min or max index of a region
    
    :return: int endpoint index of a region
    
    """
    get_err = lambda x1, x2: abs(x1-x2) #np.sqrt((x1-x2)**2)
    
    # Output list of indices: [[beg., end],[beg., end], ...] 
    clusters = []

    m = 0

    # The current cluster indices: [n_0, n_1, ... ]
    new_cluster = []
    
    # Pass through the list of events
    for n in range(0,loc_arr.size-3):

        # Check: is the time within the time thresh?
        if abs(t_arr[n+1] - t_arr[n]) <= time_thresh:
            event_loc = loc_arr[n+1]
        else: 
            continue

        # Get the current cluster mean
        cluster_mean = np.mean(loc_arr[m:n+1])

        # Get the potential cluster mean    
        new_cluster_mean = np.mean(loc_arr[m:n+2])

        err1 = get_err(cluster_mean, event_loc)
        err2 = get_err(cluster_mean, new_cluster_mean)
        
        if verbose: print(f"{n:5d}, {err1:7.3f}, {err2:7.3f}, {dist_thresh:7.3f}")
 
        # Checks: 
        # 1. new event is within dist. thresh of current clust.
        # 2. new mean - current mean is within dist. thresh.
        if  (err1 < dist_thresh) & (err2 < dist_thresh) & \
            (n <= loc_arr.size-5):
            #if verbose: print("append")
            new_cluster.append(n)
        else:
            # Save the current cluster and prepare restart
            txt = f'Trying {n} '
            app = "Nope"
            if (len(new_cluster) >= 2):
                # Since this is an incomplete method (clusters will be merged after), 
                # can keep this; otherwise, would lose small clusters
                #if (abs(t_arr[new_cluster[-1]]-t_arr[new_cluster[0]]) > time_thresh):
                clusters.append(new_cluster)
                app = 'closed'
            #if verbose: print(txt+app)
            new_cluster = []

            # Update starting point
            m=n+1
            
    return clusters


def get_clusters_xx(t_arr, loc_arr, dist_thresh, time_thresh, verbose=False):
    """
    Get a list of cluster indices

    :param t_arr: np.array Trajectory array of timepoints
    :param timepoint: float Timepoint
    :param time_thresh: float buff around timepoint  
    :param direction: int gets the min or max index of a region
    
    :return: int endpoint index of a region
    
    """
    get_err = lambda x1, x2: abs(x1-x2) #np.sqrt((x1-x2)**2)
    
    # Output list of indices: [[beg., end],[beg., end], ...] 
    clusters = []

    m = 0

    # The current cluster indices: [n_0, n_1, ... ]
    new_cluster = []
    
    # Pass through the list of events
    #for n in range(0,loc_arr.size-3):
    n = 0
    while n < loc_arr.size-3:
        # Check: is the time within the time thresh?
        if abs(t_arr[n+1] - t_arr[n]) <= time_thresh:
            event_loc = loc_arr[n+1]
        else: 
            n+=1
            continue

        # Get the current cluster mean
        cluster_mean = np.mean(loc_arr[m:n+1])

        # Get the potential cluster mean    
        new_cluster_mean = np.mean(loc_arr[m:n+2])

        err1 = get_err(cluster_mean, event_loc)
        err2 = get_err(cluster_mean, new_cluster_mean)
        
        #if verbose: print(f"{n:5d}, {err1:7.3f}, {err2:7.3f}, {dist_thresh:7.3f}")
 
        # Checks: 
        # 1. new event is within dist. thresh of current clust.
        # 2. new mean - current mean is within dist. thresh.
        txt = f'Try {n:4d}: '
        app = ""
        if  (err1 < dist_thresh) & (err2 < dist_thresh):
            #if verbose: print("append")
            new_cluster.append(n)
        else:
            # Save the current cluster and prepare restart
            app = "breaking"
            if (len(new_cluster) >= 2):
                # Since this is an incomplete method (clusters will be merged after), 
                # can keep this; otherwise, would lose small clusters
                #if (abs(t_arr[new_cluster[-1]]-t_arr[new_cluster[0]]) > time_thresh):
                clusters.append(new_cluster)
                app = 'closing'
            #if verbose: print(txt+app)
            new_cluster = []

            # Update starting point
            m=n+1
                    
        if verbose: print(txt, f"err1 = {err1:6.3f}, err2 = {err2:6.3f}", app)
        n+=1
    
    n-=1
    #print(n, new_cluster)
    if len(new_cluster) > 0:
        m = new_cluster[0]

        while (n < loc_arr.size) & (new_cluster[0] < n):    
            # Get the current cluster mean
            cluster_mean = np.mean(loc_arr[m:n+1])

            # Get the potential cluster mean    
            new_cluster_mean = np.mean(loc_arr[m:n+1])

            err1 = get_err(cluster_mean, event_loc)
            err2 = get_err(cluster_mean, new_cluster_mean)    
            if len(new_cluster) != 0:
                txt = f'Try {n:4d}: '
                app = ""
                if  (err1 < dist_thresh) & (err2 < dist_thresh):
                    #if verbose: print("append")
                    new_cluster.append(n)
                else:
                    # Save the current cluster and prepare restart
                    app = "breaking"
                    if (len(new_cluster) >= 2):
                        # Since this is an incomplete method (clusters will be merged after), 
                        # can keep this; otherwise, would lose small clusters
                        #if (abs(t_arr[new_cluster[-1]]-t_arr[new_cluster[0]]) > time_thresh):
                        clusters.append(new_cluster)
                        app = 'closing'

            if verbose: print(txt, f"err1 = {err1:6.3f}, err2 = {err2:6.3f}", app)                

            n+=1
        if len(new_cluster) > 0:
            clusters.append(new_cluster)                
    
    
    if loc_arr.size-1 not in clusters[-1]:
        if verbose: print(f"{loc_arr.size:4d} is not in: [{clusters[-1][0]:4d},{clusters[-1][-1]:4d}]" )
    else:
        if verbose: print("Done")
            
    return clusters


subcluster_lengths = lambda cluster_list: [len(c) for c in cluster_list]


def switch_indices(clusters, max_ind):
    new_clusters = []
    for cc in clusters[::-1]:
        new_clusters.append([max_ind - 1 - c for c in cc[::-1]])
    return new_clusters

'''
def extend_cluster(t_arr, x_arr, cluster, configs, dist_thresh, verbose=False):
    
    results = _extend_edge(t_arr, x_arr, cluster[-1], cluster[0], [x_arr[cluster].mean()], configs, dist_thresh, verbose)

    cluster += results[1]

    results = _extend_edge(t_arr, x_arr, cluster[0], cluster[-1], [x_arr[cluster].mean()], configs, dist_thresh, verbose)

    cluster = results[1] + cluster

    return cluster
'''

def extend_cluster(t_arr, x_arr, c, time_thresh, verbose=False):
    """
    """
    i = 999
    # Get cluster
    if verbose: print(f"Cluster #{i}\n\tlength: {len(c)}, bounds: [{x_arr[c].min():6.3f}, {x_arr[c].max():6.3f}]")

    # If cluster is too small, ignore it
    if len(c) < 2:
        return []

    # extend cluster backwards w.r.t. time
    if verbose: print("Backwards")

    extended_cluster_ = np.empty([])

    work_ind = get_time_ind(t_arr, t_arr[c[0]], 2*time_thresh, -1)
    prev_work_ind = c[0]
    if verbose: print(f"\t1.1. [{c[0]:4d}, {c[-1]:4d}], new: {work_ind:4d}, last: {prev_work_ind:4d}") 

    keep_going = True
    while keep_going:

        # Get the indices for the extended box
        cc = get_iqr_mask_x(x_arr[work_ind:c[-1]+1], work_ind, (x_arr[c].min(), x_arr[c].max()), 0, True)[0]

        #if verbose: print("\t2. cluster size:", cc.size)
        if len(cc) < 1:
            break
        if verbose: print(f"\t1.2. [{cc[0]:4d}, {cc[-1]:4d}], new: {work_ind:4d}, last: {prev_work_ind:4d}") 

        extended_cluster_ = cc.copy()
        if cc[0] != prev_work_ind:
            if verbose: print(f"\t\tnot equal: {cc[0]:4d} \= {prev_work_ind:4d}")
            prev_work_ind = cc[0]
        else:
            break
        work_ind = get_time_ind(t_arr, t_arr[prev_work_ind], 2*time_thresh, -1)

    if extended_cluster_.size > 1:
        if verbose: print(f"\t1.3. [{extended_cluster_[0]}, {extended_cluster_[-1]}], {work_ind}")

    if len(cc) > 1:   
        if cc[-1]!=prev_work_ind:
            prev_work_ind = cc[-1]
    work_ind = get_time_ind(t_arr,t_arr[work_ind], 2*time_thresh, 1)


    # extend cluster forwards w.r.t. time
    if verbose: print("Forwards")

    work_ind = get_time_ind(t_arr,t_arr[c[-1]], 2*time_thresh, 1)
    prev_work_ind = c[-1]
    if verbose: print(f"\t2.1. [{c[0]:4d}, {c[-1]:4d}], new: {work_ind:4d}, last: {prev_work_ind:4d}") 

    extended_cluster = np.empty([])

    keep_going = True
    while keep_going:

        cc = get_iqr_mask_x(x_arr[c[0]:work_ind+1], c[0], (x_arr[c].min(), x_arr[c].max()), 0, True)[0]    
        
        extended_cluster = cc.copy()
        
        if len(cc) < 1:
            break
        if verbose: print(f"\t2.2. [{cc[0]:4d},{cc[-1]:4d}], new: {work_ind:4d}, last: {prev_work_ind:4d}")         

        if cc[-1] != prev_work_ind:
            if verbose: print(f"\t\tnot equal: {cc[-1]:4d} \= {prev_work_ind:4d}")
            prev_work_ind = cc[-1]
        else:
            break

        work_ind = get_time_ind(t_arr, t_arr[prev_work_ind], 2*time_thresh, 1)

    print(f"{extended_cluster}")

    if (extended_cluster_.size > 0) and (extended_cluster.size > 0):
        extended_cluster = np.concatenate([ extended_cluster_.reshape(-1,), extended_cluster.reshape(-1,)])
        if verbose: print(f"Final cluster: length = {extended_cluster.size:4d}; range = [{extended_cluster[0]:4d}, {extended_cluster[-1]:4d}],{work_ind:4d}")
        if verbose: print()
            
    return np.unique(extended_cluster).tolist()


def intersecting_bounds(a1,a2,b1,b2):
    """
    Check whether two ranges intersect
    """
    return (((a1 >= b1) & (a1 <= b2)) | 
            ((a2 >= b1) & (a2 <= b2)) | 
            ((b1 >= a1) & (b1 <= a2)) | 
            ((b2 >= a1) & (b2 <= a2)))    


def contains(a1,a2,b1,b2):
    """
    Check whether one range contains another
    """    
    return (((a1 >= b1) & (a2 <= b2)) | # a in b
            ((b1 >= a1) & (b2 <= a2)))  # b in a  

inter_bounds = lambda p1, p2: intersecting_bounds(p1[0],p1[-1],p2[0],p2[-1])
conta_bounds = lambda p1, p2: contains(p1[0],p1[-1],p2[0],p2[-1])


def extend_final_clusters(t_arr, x_arr, clusters, configs, verbose=False):
    """
    """
    from stay_classification.box_classifier.box_method import extend_edge

    clust = clusters[0]
    #_configs = configs.copy()
    #_configs['dist_thresh'] = get_iqr(x_arr[clust])
    new_clusters = [extend_cluster(t_arr, x_arr, clust.copy(), configs, dist_thresh, verbose)]    

    #dist_thresh = configs['dist_thresh']
    
    for clust in clusters[1:]:

        #_configs = configs.copy()
        #_configs['dist_thresh'] = get_iqr(x_arr[clust])        
        c = extend_cluster(t_arr, x_arr, clust.copy(), configs, dist_thresh, verbose)
        
        # check the IQR is within the allowed threshold
        dist_criterion = False
        if get_iqr(x_arr[c])<2*dist_thresh:
            dist_criterion = True
            
        c_last = new_clusters[-1]            
        
        # Check if new cluster overlaps with the previous one
        embed_criterion = False
        if len(new_clusters)>0:
            embed_criterion = inter_bounds(c,c_last)        
        
        print(f"[{ clust[0]:4d},{ clust[-1]:4d}]," + "\t"\
              f"{t_arr[clust[0]]:6.3f}...{t_arr[clust[-1]]:6.3f}" + "\t"\
              f"{x_arr[clust].mean():6.3f}," + "\t"\
              f"{get_iqr(x_arr[clust]):6.3f}," + "\t\t\t"\
              f"[{ c[0]:4d},{ c[-1]:4d}]," + "\t"\
              f"{t_arr[c[0]]:6.3f}...{t_arr[c[-1]]:6.3f}" + "\t"\
              f"{x_arr[c].mean():6.3f}," + "\t"\
              f"{get_iqr(x_arr[c]):6.3f},", \
              dist_criterion, embed_criterion)
        
        
        # check the IQR is within the allowed threshold
        dist_criterion0 = False
        if get_iqr(x_arr[clust])<2*dist_thresh:
            dist_criterion0 = True
                    
        # Check if new cluster overlaps with the previous one
        embed_criterion0 = False
        if len(new_clusters)>0:
            embed_criterion0 = inter_bounds(clust,c_last) 
            
        if dist_criterion & (not embed_criterion):
            new_clusters.append(c)
        elif dist_criterion0 & (not embed_criterion0) & (get_iqr(x_arr[clust]) < get_iqr(x_arr[c])):
            new_clusters.append(clust)
        '''if len(new_clusters)>0:
            embed_criterion = inter_bounds(c,c_last)
            print(f"[{ c[0]:4d},{ c[-1]:4d}]," + "\t"\
                  f"[{c_last[0]:4d},{c_last[-1]:4d}]")

        # ... if there is an overlap, get the one with the smaller IQR
        # ... 1. remove last, append new if dist_crit == True
        if dist_criterion & embed_criterion:
            if get_iqr(noise_arr[c])<get_iqr(noise_arr[c_last]):
                new_clusters[-1] = c       
            else:
                pass
        # ... if no overlap and dist_crit == True, append
        elif dist_criterion & (not embed_criterion):
            new_clusters.append(c)
        else:
            pass'''
    
    return new_clusters

from stay_classification.box_classifier.box_classify import box_classifier_core
from stay_classification.box_classifier.box_method import get_mask, make_box, get_directional_indices, get_thresh_mean, check_means, get_time_ind
from helper__metric_box__explore import _get_iqr

def _extend_edge(t_arr, x_arr, working_index, fixed_index, means, configs, *quantiles, verbose=False):
    """
    Extend the edge of a potential box to include more points. 

    :param t_arr: np.array Trajectory array of timepoints
    :param x_arr: np.array Trajectory array of locations
    :param working_index: np.array Timepoint index to be extended
    :param fixed_index: np.array Timepoint index to be fixed
    :param fixed_index: [float] List of means; only one is used
        
    
    :param configs: dict containing the parameters used to define the box
    :param verbose: bool To select printing metrics to stdio
    
    :return: [float] means for the extension.
    :return: [int] new indices included in the extension
    :return: bool Indicates whether the means have converged
    
    """
    
    count_thresh = configs['count_thresh']
    #dist_thresh = configs['dist_thresh']
    
    keep_running = (working_index > 1) & (working_index < len(x_arr)-1)
    
    indices = []
    
    if working_index < fixed_index: 
        # Go backwards in time
        direction = -1
    else: 
        # Go forwards in time
        direction = 1
        
    mean = means[-1]
    converged_mean = mean
    converged_mean_ind0 = working_index
    
    while keep_running:
        
        #print(mean, direction)
        # Update and store the working index
        working_index += direction*1
        indices.append(working_index)
        
        # Update and store the mean
        if direction == -1:
            mean = get_thresh_mean(x_arr[working_index:fixed_index], mean, dist_thresh)
        else:
            mean = get_thresh_mean(x_arr[fixed_index:working_index], mean, dist_thresh)
        
        means.append(mean)    
        
        if np.isnan(mean):
            #print(mean)
            break
        
        # Stopping criteria:
        # if the thresholded mean doesn't change upon getting new samples
        # * if the duration is too long and there are sufficient number of samples
        
        if mean != converged_mean:
            converged_mean = mean
            converged_mean_ind = working_index
            converged_mean_ind0 = working_index
        else:
            converged_mean_ind = working_index
        
        time_diff = abs(t_arr[fixed_index]-t_arr[working_index])
        ctime_diff = abs(t_arr[converged_mean_ind0]-t_arr[converged_mean_ind])                                                         
        if ((ctime_diff>1.0) & (mean == converged_mean)): 
            if verbose: print('cdrop', ctime_diff)
            break        
        
        
        # When the mean either converges or stops
        if ((len(indices)>count_thresh) | ((time_diff>0.5) & (len(indices)>5))):  
            #print(time_diff,len(indices))
            nr_events = min(len(indices), count_thresh)
            # see also: bug_check_means(means,nr_events,0.25)
            if check_means(means, configs, nr_events):
                if verbose: print('drop', time_diff)
                break       
                    
                    
        #print(f"{t_arr[working_index]:.3f} {time_diff:.3f} {ctime_diff:.3f}", \
        #      len(indices), fixed_index, working_index, converged_mean_ind0, converged_mean_ind, \
        #      f"\t{mean:.5f} {x_arr[working_index]:.3f} {mean+dist_thresh:.5f}",)#,[m == m0 for m in means[-count_thresh:]])            
                    
        keep_running = (working_index > 1) & (working_index < len(x_arr)-1)

    return means, indices, keep_running


def get_iqr_mask_x(sub_arr, offset, iqr_bounds, iqr_fact = 1.5, within=True, verbose=False):
    """
    """
    
    #TODO: rename `x_arr` --> `sub_arr`, `cluster` --> `offset` 
    
    # Mask to include only events within the IQR
    if iqr_bounds != None:
        if verbose: print(iqr_bounds)
        q25, q75 = iqr_bounds
    else:
        q25, q75 = _get_iqr(sub_arr)
    
    iqr = abs(q75 - q25)
        
    if within:
        mask = np.where((sub_arr > (q25 - iqr_fact * iqr)) & (sub_arr < (q75 + iqr_fact * iqr)))
        
    else:
        mask =  np.where((sub_arr <= (q25 - iqr_fact * iqr)) | (sub_arr >= (q75 + iqr_fact * iqr)))
    
    mask[0][:] += offset
    
    return mask


def get_bounded_events(x_arr, cluster, mininmum, maximum, within=True):
    """
    """    
    #TODO: rename `x_arr` --> `sub_arr`, `cluster` --> `offset` 
    
    # Mask to include only events within the IQR
    sub_arr = x_arr.copy()
    
    if within:
        mask = np.where((sub_arr > mininmum) & (sub_arr < maximum))
        
    else:
        mask =  np.where((sub_arr <= mininmum) | (sub_arr >= maximum))
    
    mask[0][:] += cluster[0]
    
    return mask[0]


def get_no_overlap(t_arr, clusters):
    """
    """
    final_clusters = [] 
    
    c1 = clusters[0]
    c2 = clusters[1]

    if c1[-1]>c2[0]:

        print(abs(t_arr[c1[-2]]-t_arr[c1[-1]]), abs(t_arr[c1[-1]]-t_arr[c2[0]]))
        if abs(t_arr[c1[-2]]-t_arr[c1[-1]]) < abs(t_arr[c1[-1]]-t_arr[c2[0]]):

            print("c2[0] is the outlier")
            pass
        else:
            print("c1[-1] is the outlier")   
            _ = c1.pop(-1)

            final_clusters.append(c1)
    else: 
        final_clusters.append(c1)
        
    for c1, c2 in zip( clusters[1:-1], clusters[2:]):

        print(c1[0],c1[-1], c2[0],c2[-1])
        if c1[-1]>c2[0]:

            print(abs(t_arr[c1[-2]]-t_arr[c1[-1]]), abs(t_arr[c1[-1]]-t_arr[c2[0]]))
            if abs(t_arr[c1[-2]]-t_arr[c1[-1]]) < abs(t_arr[c1[-1]]-t_arr[c2[0]]):
                pass
            else:
                print("c1[-1] is the outlier")   
                _ = c1.pop(-1)    
                final_clusters.append(c1)
        else: 
            final_clusters.append(c1)
                
    c1 = clusters[-2]
    c2 = clusters[-1]

    if c1[-1]>c2[0]:

        print(abs(t_arr[c1[-2]]-t_arr[c1[-1]]), abs(t_arr[c1[-1]]-t_arr[c2[0]]))
        if abs(t_arr[c1[-2]]-t_arr[c1[-1]]) < abs(t_arr[c1[-1]]-t_arr[c2[0]]):

            print("c2[0] is the outlier")
            pass
        else:
            print("c1[-1] is the outlier")   
            _ = c1.pop(-1)

            final_clusters.append(c1)   
    else: 
        final_clusters.append(c1)    
            
    return final_clusters


def get_extended_clusters(t_arr, x_arr, clusters, time_thresh, verbose=False):
    """
    """
    new_clusts = []

    i = 0 
    for c in clusters:

        # Get cluster
        if verbose: print(f"Cluster #{i}\n\tlength: {len(c)},  ",\
            f"bounds: [{x_arr[c].min():6.3f}, {x_arr[c].max():6.3f}], ",\
            f"x-width: {abs(x_arr[c].max()-x_arr[c].min()):6.3f}")
        
        # If cluster is too small, ignore it
        if len(c) < 2:
            continue

        # extend clust backwards w.r.t. time
        if verbose: print("Backwards")

        ext_clust_bwd = np.array([])

        work_ind = get_time_ind(t_arr, t_arr[c[0]], 2*time_thresh, -1)
        prev_work_ind = c[0]
        if verbose: print(f"\t1.1. [{c[0]:4d}, {c[-1]:4d}], new: {work_ind:4d}, last: {prev_work_ind:4d}") 
        
        keep_going = True
        while keep_going:

            # Get the indices for the ext box
            cc = get_iqr_mask_x(x_arr[work_ind:c[-1]+1], work_ind, (x_arr[c].min(), x_arr[c].max()), 0, True)[0]
            
            #if verbose: print("\t2. clust size:", cc.size)
            if len(cc) < 1:
                break
            if verbose: print(f"\t1.2. [{cc[0]:4d}, {cc[-1]:4d}], ",\
            f" new: {work_ind:4d}, last: {prev_work_ind:4d}") 

            ext_clust_bwd = cc.copy()
            
            if cc[0] != prev_work_ind:
                if verbose: print(f"\t\tnot equal: {cc[0]:4d} \= {prev_work_ind:4d}")
                prev_work_ind = cc[0]
            else:
                break
            work_ind = get_time_ind(t_arr, t_arr[prev_work_ind], 2*time_thresh, -1)
            
        if ext_clust_bwd.size > 1:
            if verbose: print(f"\t1.3. [{ext_clust_bwd[0]}, {ext_clust_bwd[-1]}], {work_ind}")

        if len(cc) > 1:   
            if cc[-1]!=prev_work_ind:
                prev_work_ind = cc[-1]
        work_ind = get_time_ind(t_arr,t_arr[work_ind], 2*time_thresh, 1)


        # extend clust forwards w.r.t. time
        if verbose: print("Forwards")

        work_ind = get_time_ind(t_arr,t_arr[c[-1]], 2*time_thresh, 1)
        prev_work_ind = c[-1]
        if verbose: print(f"\t2.1. [{c[0]:4d},{c[-1]:4d}], " \
            f" new: {work_ind:4d}, last: {prev_work_ind:4d}") 

        ext_clust_fwd = np.array([])

        keep_going = True
        while keep_going:

            cc = get_iqr_mask_x(x_arr[c[0]:work_ind+1], c[0], (x_arr[c].min(), x_arr[c].max()), 0, True)[0]    
            
            ext_clust_fwd = cc.copy()
            
            #if verbose: print("\t2. clust size:", cc.size)
            if len(cc) < 1:
                break
            if verbose: print(f"\t2.2. [{cc[0]:4d},{cc[-1]:4d}], ",\
            f" new: {work_ind:4d}, last: {prev_work_ind:4d}") 
     

            if cc[-1] != prev_work_ind:
                if verbose: print(f"\t\tnot equal: {cc[-1]:4d} \= {prev_work_ind:4d}")
                prev_work_ind = cc[-1]
            else:
                break

            work_ind = get_time_ind(t_arr, t_arr[prev_work_ind], 2*time_thresh, 1)

        new_clust = []
        
        if (ext_clust_bwd.size > 0) and (ext_clust_fwd.size > 0):
            ext_clust_fwd = np.concatenate([ ext_clust_bwd.reshape(-1,), ext_clust_fwd.reshape(-1,)])
            final_report = f"\nFinal clust: length = {ext_clust_fwd.size:4d};"\
                           f" range = [{ext_clust_fwd[0]:4d},{ext_clust_fwd[-1]:4d}]"\
                           f" working index = {work_ind:4d};"
            dropped = ""
            new_clust = np.unique(ext_clust_fwd).tolist()
        else:
            final_report = f"\nFinal clust: length = {ext_clust_bwd.size:4d}; "\
            f"  length = {ext_clust_fwd.size:4d}"
            dropped = "1Dropped: "

        duration = 0
        if (len(new_clust) > 0):
            duration = abs(t_arr[new_clust[-1]]-t_arr[new_clust[0]])

        if (len(new_clust) > 0) and (duration > time_thresh):
            final_report += f" duration = {duration:6.3f}\n"
            new_clusts.append(new_clust)
        else:
            final_report += f" duration = {duration:6.3f}\n"
            dropped = "2Dropped: "

        if verbose: print(dropped+final_report)            
        
        i += 1

    return new_clusts


def separate_clusters(clusters1, clusters2, verbose=False):
    """
    """
    new_clusts = []
    m = 0
    while m < len(clusters1):
        c1 = clusters1[m]
        n = 0
        sc1 = set(c1)
        new_clust = []
        while n < len(clusters2):
            c2 = clusters2[n]
            if verbose: print(f"{m:4d}: [{c1[0]:4d},{c1[-1]:4d}] and [{c2[0]:4d},{c2[-1]:4d}]")
            if inter_bounds(c1,c2):
                sc2 = set(c2)
                c = list(sc1.intersection(sc2))
                new_clust.extend(c)
                if verbose: print('yes')
            else:
                pass
            n+=1
        if len(new_clust)>0:
            new_clusts.append(new_clust)
        m+=1
        
    return new_clusts


from helper__gaps import merge_clusters, merge_clusters_2
from helper__metric_box__explore import eval_synth_data, get_iqr_mask


def switch_indices(clusters, max_ind):
    """
    Swap the ordering of indices in a list of lists which identify 
    clusters of a backwards array so that it corresponds to the 
    original ordering of the forwards array.
    """    
    new_clusters = []
    for cc in clusters[::-1]:
        new_clusters.append([max_ind - 1 - c for c in cc[::-1]])
    return new_clusters


def test_switch_indices(arr, clusters_bwd):
    """
    Unit test for the swicth_indices function. 
    """
    result = 0
    for m in list(range(-1,-1*len(clusters_bwd),-1)):
        n = -1*m-1
        result += sum(arr[::-1][clusters_bwd[m]]-arr[switch_indices(clusters_bwd, arr.shape[0])[n]])
        
    return_string = f"Function `{test_switch_indices.__name__}` "        
    if result == 0.0: 
        return_string += "passed"
        result = 0
    else:
        return_string += "failed"
        result = 1
    print(return_string)
    return result


def cluster_method(t_arr, x_arr, d_thresh, t_thresh, segments=None, verbose=False):
    """
    Find the stays of a given trajectory
    """
    
    # Get the minimal clusters, based on spatio-temporal nearness
    clusters_fwd = get_clusters_x(t_arr,       x_arr,       d_thresh, t_thresh)
    clusters_bwd = get_clusters_x(t_arr[::-1], x_arr[::-1], d_thresh, t_thresh)    
    ## Scoring
    prec, rec, conmat = eval_synth_data(segments, t_arr, clusters_fwd)
    if verbose: print_p_and_r(clusters_fwd, prec, rec, conmat, 'forward')
    prec, rec, conmat = eval_synth_data(segments, t_arr, clusters_bwd)
    if verbose: print_p_and_r(clusters_fwd, prec, rec, conmat, 'backward')
    
    # Merge the miniclusters based on the size of the gaps between them.
    clusters_fwd = merge_clusters(t_arr,       x_arr,       clusters_fwd, d_thresh, t_thresh).copy()
    clusters_bwd = merge_clusters(t_arr[::-1], x_arr[::-1], clusters_bwd, d_thresh, t_thresh).copy()
    ## Scoring
    prec, rec, conmat = eval_synth_data(segments, t_arr, clusters_fwd)
    if verbose: print_p_and_r(clusters_fwd, prec, rec, conmat, 'forward')
    prec, rec, conmat = eval_synth_data(segments, t_arr, clusters_bwd)
    if verbose: print_p_and_r(clusters_fwd, prec, rec, conmat, 'backward')
    

    # Merge clusters based on the potential of a gap to contain a travel-stay-travel
    clusters_fwd = merge_clusters_2(t_arr,       x_arr,       clusters_fwd, d_thresh, t_thresh).copy()
    clusters_bwd = merge_clusters_2(t_arr[::-1], x_arr[::-1], clusters_bwd, d_thresh, t_thresh).copy()
    ## Scoring
    prec, rec, conmat = eval_synth_data(segments, t_arr, clusters_fwd)
    if verbose: print_p_and_r(clusters_fwd, prec, rec, conmat, 'forward')
    prec, rec, conmat = eval_synth_data(segments, t_arr, clusters_bwd)
    if verbose: print_p_and_r(clusters_fwd, prec, rec, conmat, 'backward')
    
    # Merge the forward and backward clusters

    '''
    Get only those indices which occur in both for-/back-wards sets;
    --> store to a single set of clusters
    NOTE: the bias is towards the backwards set of clusters
    TODO: do this after the extension phase since the bias (but do the remap first)
    '''
    
    final_clusters = []
    total_mindices_bwd = []
    # Need to remap the reverse indices at this point
    # NOTE: the reverse indices don't go to the respective end point 
    #       (meaning there is no index = 0)
    clusters_bwd = switch_indices(clusters_bwd, t_arr.shape[0])
    for c in clusters_bwd:
        # get those indices which define the IQR of a cluster
        mask = get_iqr_mask(x_arr[c], c, 0, True)
        # ... and save them flattened
        total_mindices_bwd.extend(mask[0].tolist())

    total_mindices_fwd = []
    for c in clusters_fwd:
        # get those indices which define the IQR of a cluster    
        mask = get_iqr_mask(x_arr[c], c, 0, True)

        cluster = list(set(mask[0].tolist()).intersection(set(total_mindices_bwd)))
        if len(cluster)>5:
            final_clusters.append(sorted(cluster))

    # For the remaining clusters, extend in both directions to see if they are converged
    final_extended_clusters = get_extended_clusters(t_arr, x_arr, final_clusters, t_thresh, verbose)
    
    return final_extended_clusters
