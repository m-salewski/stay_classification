import numpy as np
import matplotlib.pyplot as plt

def get_cluster_boxplots(old_clusts, new_clusts, t_arr, x_arr, ax, color, linestyle, legend_label, bbox_tuple):

    bp_data, labels, positions, widths = get_boxplot_quants(t_arr, x_arr, old_clusts)

    bplots = ax.boxplot(bp_data, labels=labels, positions=positions, widths=widths,
                        boxprops=dict(color=color, linewidth=1.0, linestyle='--', dashes=[3,2], alpha=0.5), 
                        capprops=dict(color=color, linewidth=1.0, linestyle='-', alpha=0.5),
                        whiskerprops=dict(color=color, linewidth=1.0, linestyle='--', dashes=[3,2], alpha=0.5),
                        flierprops=dict(color=color, markeredgecolor=color))

    ax.legend([legend_label], bbox_to_anchor=bbox_tuple, loc='center right', ncol=1);
    
    bp_data, labels, positions, widths = get_boxplot_quants(t_arr, x_arr, new_clusts)
    
    _ = ax.boxplot(bp_data, labels=labels, positions=positions, widths=widths, patch_artist=True, 
                         boxprops=dict(color=color, alpha=0.1, linewidth=0, facecolor=color),
                         capprops=dict(color=color, alpha=0.1),
                         whiskerprops=dict(color=color, alpha=0.1),
                         flierprops=dict(color=color, markeredgecolor=color, alpha=0.1))                   

    return ax

kwargs_unfilled = dict(color='red', linewidth=1.0, linestyle='--', dashes=[3,2], alpha=0.5, patch_artist=False)
kwargs_filled=dict(color='red', linewidth=0.0, linestyle='None', dashes='None', alpha=0.1, patch_artist=True)
                  
def get_cluster_boxplot(clusters, t_arr, x_arr, ax, **kwargs):
    
    color = kwargs['color']
    ls = kwargs['linestyle']
    lw = kwargs['linewidth']
    leg_lab = kwargs['legend_label']
    bbox_tup = kwargs['bbox_tuple']
    alpha = kwargs['alpha']
    dashes = kwargs['dashes']
    pat_art = kwargs['patch_artist']
    
    boxprops    =dict(color=color, linewidth=lw, linestyle=ls, alpha=alpha)
    whiskerprops=dict(color=color, linewidth=lw, linestyle=ls, alpha=alpha)  
    flierprops  =dict(color=color, markeredgecolor=color)
    if pat_art:
        boxprops['facecolor']=color
        flierprops['markerfacecolor']=color
    else:
        boxprops['dashes']=dashes
        whiskerprops['dashes']=dashes
        
    capprops    =dict(color=color, linewidth=lw, linestyle='-', alpha=alpha)
    
    bp_data, labels, positions, widths = get_boxplot_quants(t_arr, x_arr, clusters)

    bplots = ax.boxplot(bp_data, labels=labels, positions=positions, widths=widths, patch_artist=pat_art,
                        boxprops=boxprops, capprops=capprops, whiskerprops=whiskerprops, flierprops=flierprops)
    if leg_lab != None:
        ax.legend([leg_lab], bbox_to_anchor=bbox_tup, loc='center right', ncol=1)
    
    return ax


# ---

def _get_iqr(data):
    
    q25 = np.quantile(data, 0.25, interpolation='lower')
    q75 = np.quantile(data, 0.75, interpolation='higher')
    return q25, q75


def get_iqr(data):
    
    q25, q75 = _get_iqr(data)
    
    return abs(q75 - q25)


def iqr_metrics(data, iqr_fact=1.5):
    
    q25, q75 = _get_iqr(data)
    
    iqr = abs(q75 - q25)

    iqr_boost = iqr*iqr_fact
    
    full_range = q75 - q25 + 2*iqr_boost
    min_range = ys[np.where(ys < q75+iqr_boost )].max() - ys[np.where(ys > q25-iqr_boost )].min()
    
    return full_range, min_range


def get_iqr_mask(x_arr, cluster, iqr_fact = 1.5, within=True):

    #TODO: rename `x_arr` --> `sub_arr`, `cluster` --> `offset` 
    
    # Mask to include only events within the IQR
    q25, q75 = _get_iqr(x_arr)
    
    iqr = abs(q75 - q25)
    
    sub_arr = x_arr.copy()
    
    if within:
        mask = np.where((sub_arr > (q25 - iqr_fact * iqr)) & (sub_arr < (q75 + iqr_fact * iqr)))
        
    else:
        mask =  np.where((sub_arr <= (q25 - iqr_fact * iqr)) | (sub_arr >= (q75 + iqr_fact * iqr)))
    
    mask[0][:] += cluster[0]
    
    return mask


# --- 


def get_boxplot_quants(t_arr, x_arr, clusters):
    
    data = []
    labels = []
    positions = []
    widths = []

    for cl_nr, clust in enumerate(clusters):

        # Get the subseqs
        xs = t_arr[clust]
        ys = x_arr[clust]
        data.append(ys)
        widths.append(xs[-1]-xs[0])
        
        pos = (xs[-1]+xs[0])/2
        positions.append(pos)
        labels.append(f"{pos:.2f}")
    
    return data, labels, positions, widths


def get_boxplot_centers(t_arr, x_arr, clusters):
    
    data = []
    labels = []
    positions = []
    widths = []

    for cl_nr, clust in enumerate(clusters):

        # Get the subseqs
        xs = t_arr[clust]
        ys = x_arr[clust]

        # Mask to include only events within the IQR
        q5l = np.quantile(ys,0.5, interpolation='lower')
        q5h = np.quantile(ys,0.5, interpolation='higher')

        data.append((q5l+q5h)*0.5)
        positions.append((xs[-1]+xs[0])/2)
        
    return data, positions

def get_boxplot_lines(t_arr, x_arr, clusters):
    
    data = []
    labels = []
    positions = []
    widths = []

    for cl_nr, clust in enumerate(clusters):

        # Get the subseqs
        xs = t_arr[clust]
        ys = x_arr[clust]

        # Mask to include only events within the IQR
        q5l = np.quantile(ys,0.5, interpolation='lower')
        q5h = np.quantile(ys,0.5, interpolation='higher')

        data.append((q5l+q5h)*0.5)
        data.append((q5l+q5h)*0.5)
        
        positions.append(xs[0])
        positions.append(xs[-1])        
        
    return data, positions

def get_boxplot_iqr_midpoints(t_arr, x_arr, clusters):
    
    data = []
    labels = []
    positions = []
    widths = []

    for cl_nr, clust in enumerate(clusters):

        # Get the subseqs
        xs = t_arr[clust]
        ys = x_arr[clust]

        # Mask to include only events within the IQR
        q25 = np.quantile(ys,0.25, interpolation='lower')
        q75 = np.quantile(ys,0.75, interpolation='higher')

        data.append((q25+q75)*0.5)
        positions.append((xs[-1]+xs[0])/2)
        
    return data, positions

get_err = lambda x1, x2: abs(x1-x2) #np.sqrt((x1-x2)**2)

def get_clusters_rev(t_arr, loc_arr, dist_thresh, time_thresh, verbose=False):

    """
    Get a list of cluster indices

    :param t_arr: np.array Trajectory array of timepoints
    :param timepoint: float Timepoint
    :param time_thresh: float buff around timepoint  
    :param direction: int gets the min or max index of a region
    
    :return: int endpoint index of a region
    
    """
    
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
        
        if verbose: print(n, err1, err2, dist_thresh)
 
        # Checks: 
        # 1. new event is within dist. thresh of current clust.
        # 2. new mean - current mean is within dist. thresh.
        if  (err1 < dist_thresh) & (err2 < dist_thresh) & \
            (n <= loc_arr.size-5):
            new_cluster.append(n)
        else:
            # Save the current cluster and prepare restart
            txt = f'Trying {n} '
            app = "Nope"
            if (len(new_cluster) >= 2):
                if (abs(t_arr[new_cluster[-1]]-t_arr[new_cluster[0]]) > time_thresh):
                    clusters.append(new_cluster)
                    app = 'closed'
 
            new_cluster = []

            # Update starting point
            m=n+1
            
    return clusters

    
import os, sys
sys.path.append('/home/sandm/Notebooks/stay_classification/src/')
#from synthetic_data.trajectory import get_stay_segs, get_adjusted_stays
#from synthetic_data.trajectory_class import get_rand_traj
#from synthetic_data.plotting import plot_trajectory, add_plot_seg_boxes

def eval_synth_data(segments, t_arr, clusters):

    # Get the actual stay indices and create the labels for each event
    from synthetic_data.trajectory import get_stay_indices, get_adjusted_stays
    
    true_indices = get_stay_indices(get_adjusted_stays(segments, t_arr), t_arr)
    true_labels = np.zeros(t_arr.shape)

    for pair in true_indices:
        true_labels[pair[0]:pair[1]+1] = 1

    # Get the predicted labels for each event
    final_pairs = []
    for clust in clusters:
        final_pairs.append([clust[0],clust[-1]])
    pred_labels = np.zeros(t_arr.shape)

    for pair in final_pairs:
        pred_labels[pair[0]:pair[1]+1] = 1

    # Evaluate using prec. and rec.
    from sklearn.metrics import precision_score, recall_score, confusion_matrix

    prec = precision_score(true_labels, pred_labels)
    rec  = recall_score(true_labels, pred_labels)

    # Eval. using confuse. matrix
    conf_mat = confusion_matrix(true_labels, pred_labels)
    
    return prec, rec, conf_mat


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

# --- 

from stay_classification.box_classifier.box_classify import box_classifier_core
from stay_classification.box_classifier.box_method import get_mask, make_box,get_directional_indices, extend_edge

def extend_cluster(t_arr, x_arr, cluster, configs, verbose=False):
    
    results = extend_edge(t_arr, x_arr, cluster[-1], cluster[0], [x_arr[cluster].mean()], configs, verbose)

    cluster += results[1]

    results = extend_edge(t_arr, x_arr, cluster[0], cluster[-1], [x_arr[cluster].mean()], configs, verbose)

    cluster = results[1] + cluster

    return cluster

def intersecting_bounds(a1,a2,b1,b2):
    
    return (((a1 >= b1) & (a1 <= b2)) | 
            ((a2 >= b1) & (a2 <= b2)) | 
            ((b1 >= a1) & (b1 <= a2)) | 
            ((b2 >= a1) & (b2 <= a2)))    

inter_bounds = lambda p1, p2: intersecting_bounds(p1[0],p1[-1],p2[0],p2[-1])

def extend_clusters(t_arr, x_arr, clusters, configs, verbose=False):
    
    from stay_classification.box_classifier.box_method import extend_edge

    new_clusters = [extend_cluster(t_arr, x_arr, clusters[0].copy(), configs, verbose)]
    
    
    dist_thresh = configs['dist_thresh']
    
    for clust in clusters[1:]:
        
        c = extend_cluster(t_arr, x_arr, clust.copy(), configs, verbose)
        
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


"""
Okay, i want to share something with you. Like a birthday gift. of course, it's late. But i think i can tell it now.
You understood, at least in some light sense, that I have to cope with depression. During the lockdown, I restarted meditation. Usually, i start by letting my thoughts drift along while trying to remember what happened in the day, what was significant, good, not so good, etc. I call this my pre-meditation; it's like a warm up and allows some of the thoughts to be recognized and calmed. Then after this, I go to a more neutral meditation where I try to then focus and let all the stuff drift by while maintaining the focus: the puppy, but as I've said, it think it's more like a bunch of puppies all on different leashes, going everywhere.
So I was (and still am) doing this when you went back to PT. On one of the days after you arrived, during my pre-meditation, I was also having ideas about being biased towards negativity, and thinking about being happy, when the last time i was happy. And nothing really came up. But it was okay, there were other thoughts in the queue and I moved on. Then, I went into the usual meditation. ANd the thoughts were wandering, as they do, and it hit me that those nights we shared together, in the beginning, texting a for a week or so and then the museum and kblau; all that made me happy. I was happy then. And it brought me to a smile, more like a laugh cuz it made a noise and my eyes slightly squeezed; like a pure moment of smiling. And it totally broke into my meditation. Took me completely out of it so I couldn't go back to meditation that night. 
It's not much but it's a notion that you came in to my life and touched me in a nice way and we had a connection, no matter how short. 
"""
