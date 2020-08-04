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



def plot_cluster_boxplots(time_arr, raw_arr, noise_arr, t_segs, x_segs, clusters, clusters_rev, configs):

    from synthetic_data.plotting import plot_trajectory, add_plot_seg_boxes
    from helper__metric_box__explore import iqr_metrics, get_boxplot_quants, get_clusters_rev    


    dist_thresh = configs['dist_thresh']
    
    ax = plot_trajectory(time_arr, raw_arr, noise_arr, t_segs, x_segs, dist_thresh);
    add_plot_seg_boxes(t_segs, x_segs, dist_thresh, ax)

    ax.set_xlim([5.75,18.25])

    bp_data, labels, positions, widths = get_boxplot_quants(time_arr, noise_arr, clusters)

    axt = ax.twiny()
    _ = axt.boxplot(bp_data, labels=labels, positions=positions, boxprops=dict(color='red'), widths=widths)   

    for label in axt.get_xticklabels():
        label.set_rotation(90)
    axt.set_xticklabels(labels, visible=True, color='red')


    axt.set_xlim(ax.get_xlim())
    axt.legend(['forward clusters'], bbox_to_anchor=(1.15, 0.6), loc='center right', ncol=1);


    bp_data, labels, positions, widths = get_boxplot_quants(time_arr[::-1], noise_arr[::-1], clusters_rev)

    axt = ax.twiny()
    labels = list(map(lambda x: f"{x:.2f}", positions))
    _ = axt.boxplot(bp_data, labels=labels, positions=positions, boxprops=dict(color='blue'), widths=widths)
    axt.legend(['reverse clusters'], bbox_to_anchor=(1.15, 0.4), loc='center right', ncol=1);

    for label in axt.get_xticklabels():
        label.set_rotation(90)
        
    axt.set_xticklabels(labels, visible=False)

    axt.set_xlim(ax.get_xlim())
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper right', ncol=1)
    
    return ax


def extend_cluster(t_arr, x_arr, cluster, configs, dist_thresh, verbose=False):
    
    results = _extend_edge(t_arr, x_arr, cluster[-1], cluster[0], [x_arr[cluster].mean()], configs, dist_thresh, verbose)

    cluster += results[1]

    results = _extend_edge(t_arr, x_arr, cluster[0], cluster[-1], [x_arr[cluster].mean()], configs, dist_thresh, verbose)

    cluster = results[1] + cluster

    return cluster

    
def intersecting_bounds(a1,a2,b1,b2):
    
    return (((a1 >= b1) & (a1 <= b2)) | 
            ((a2 >= b1) & (a2 <= b2)) | 
            ((b1 >= a1) & (b1 <= a2)) | 
            ((b2 >= a1) & (b2 <= a2)))    

inter_bounds = lambda p1, p2: intersecting_bounds(p1[0],p1[-1],p2[0],p2[-1])


def extend_final_clusters(t_arr, x_arr, clusters, configs, verbose=False):
    
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
    
    new_clusters = []

    i = 0 
    for c in clusters:

        # Get cluster
        if verbose: print(f"Cluster #{i}\n\tlength: {len(c)}, bounds: [{x_arr[c].min():6.3f}, {x_arr[c].max():6.3f}]")

        if len(c) < 2:
            continue


        # extend time forwards
        if verbose: print("Backwards")

        extended_cluster_ = np.empty([])

        new_end = get_time_ind(t_arr, t_arr[c[0]], 2*time_thresh, -1)
        last_ind = c[0]
        if verbose: print(f"\t1.1. [{c[0]:4d}, {c[-1]:4d}], new: {new_end:4d}, last: {last_ind:4d}") 
        
        keep_going = True
        while keep_going:

            cc = get_iqr_mask_x(x_arr[new_end:c[1]], new_end, (x_arr[c].min(), x_arr[c].max()), 0, True)[0]
            
            #if verbose: print("\t2. cluster size:", cc.size)
            if len(cc) < 1:
                break
            if verbose: print(f"\t1.2. [{cc[0]:4d}, {cc[-1]:4d}], new: {new_end:4d}, last: {last_ind:4d}") 

            extended_cluster_ = cc.copy()
            if cc[0] != last_ind:
                if verbose: print(f"\t\tnot = [{cc[0]:4d}, {cc[-1]:4d}], new: {new_end:4d}, last: {last_ind:4d}")
                last_ind = cc[0]
            else:
                break
            new_end = get_time_ind(t_arr, t_arr[last_ind], 2*time_thresh, -1)
            
        if extended_cluster_.size > 1:
            if verbose: print(f"\t1.3. [{extended_cluster_[0]}, {extended_cluster_[-1]}], {new_end}")

        if len(cc) > 1:   
            if cc[-1]!=last_ind:
                last_ind = cc[-1]
        new_end = get_time_ind(t_arr,t_arr[new_end], 2*time_thresh, 1)

        # extend time forwards
        if verbose: print("Forwards")

        new_end = get_time_ind(t_arr,t_arr[c[-1]], 2*time_thresh, 1)
        last_ind = c[-1]
        if verbose: print(f"\t2.1. [{c[0]:4d},{c[-1]:4d}], new: {new_end:4d}, last: {last_ind:4d}") 

        extended_cluster = np.empty([])

        keep_going = True
        while keep_going:

            cc = get_iqr_mask_x(x_arr[c[0]:new_end], c[0], (x_arr[c].min(), x_arr[c].max()), 0, True)[0]    
            #if verbose: print("\t2. cluster size:", cc.size)
            if len(cc) < 1:
                break
            if verbose: print(f"\t2.2. [{cc[0]:4d},{cc[-1]:4d}], new: {new_end:4d}, last: {last_ind:4d}") 
     

            if verbose: print("\t", cc[0],cc[-1],new_end, last_ind)

            extended_cluster = cc.copy()
            if cc[-1] != last_ind:
                if verbose: print(f"\t\tnot = [{cc[0]:4d}, {cc[-1]:4d}], new: {new_end:4d}, last: {last_ind:4d}")
                last_ind = cc[-1]
            else:
                break

            new_end = get_time_ind(t_arr, t_arr[last_ind], 2*time_thresh, 1)

        if verbose: print(f"Final cluster: length = {extended_cluster.size:4d}; range = [{extended_cluster[0]:4d},{extended_cluster[-1]:4d}],{new_end:4d}")
        if verbose: print()

        if (extended_cluster_.size > 0) and (extended_cluster.size > 0):
            extended_cluster = np.concatenate([ extended_cluster_.reshape(-1,), extended_cluster.reshape(-1,)])

        new_clusters.append(np.unique(extended_cluster).tolist())

        i += 1


    return new_clusters


def eval_synth_data_clusters(segments, time_arr, clusters):

    """
    Evaluate based on individual clusters
    """
    
    expected_nr_of_stays = int((len(segments)+1)/2)
    predicted_nr_of_stays = len(clusters)
    
    if expected_nr_of_stays != predicted_nr_of_stays:
        return np.nan, np.nan, np.nan
    
    
    # Get the actual stay indices and create the labels for each event
    from synthetic_data.trajectory import get_stay_indices, get_adjusted_stays
    
    true_indices = get_stay_indices(get_adjusted_stays(segments, time_arr), time_arr)
    true_labels = np.zeros(time_arr.shape)

    for pair in true_indices:
        true_labels[pair[0]:pair[1]+1] = 1

    # Get the predicted labels for each event
    final_pairs = []
    for clust in clusters:
        final_pairs.append([clust[0],clust[-1]])
    pred_labels = np.zeros(time_arr.shape)

    for pair in final_pairs:
        pred_labels[pair[0]:pair[1]+1] = 1

    # Evaluate using prec. and rec.
    from sklearn.metrics import precision_score, recall_score, confusion_matrix

    prec = precision_score(true_labels, pred_labels)
    rec  = recall_score(true_labels, pred_labels)

    # Eval. using confuse. matrix
    conf_mat = confusion_matrix(true_labels, pred_labels)
    
    return prec, rec, conf_mat
