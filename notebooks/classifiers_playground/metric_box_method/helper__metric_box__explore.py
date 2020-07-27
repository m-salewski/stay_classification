import numpy as np

def get_iqr(data):
    
    q25 = np.quantile(data, 0.25, interpolation='lower')
    q75 = np.quantile(data, 0.75, interpolation='higher')
    return abs(q75 - q25)

def iqr_metrics(data, iqr_fact=1.5):
    
    q25 = np.quantile(data, 0.25, interpolation='lower')
    q75 = np.quantile(data, 0.75, interpolation='higher')
    iqr = abs(q75 - q25)

    iqr_boost = iqr*iqr_fact
    
    full_range = q75 - q25 + 2*iqr_boost
    min_range = ys[np.where(ys < q75+iqr_boost )].max() - ys[np.where(ys > q25-iqr_boost )].min()
    
    return full_range, min_range


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


def get_iqr_mask(x_arr, cluster, within=True):

    # Mask to include only events within the IQR
    q25 = np.quantile(x_arr, 0.25, interpolation='lower')
    q75 = np.quantile(x_arr, 0.75, interpolation='higher')
    iqr = abs(q75 - q25)
    iqr_fact = 1.5
        
    if within:
        return np.where((x_arr > (q25 - iqr_fact * iqr)) & (x_arr < (q75 + iqr_fact * iqr)))
    else:
        return np.where((x_arr <= (q25 - iqr_fact * iqr)) | (x_arr >= (q75 + iqr_fact * iqr)))
    
    
import os, sys
sys.path.append('/home/sandm/Notebooks/stay_classification/src/')
from synthetic_data.trajectory import get_stay_segs, get_adjusted_stays
from synthetic_data.trajectory_class import get_rand_traj
from synthetic_data.plotting import plot_trajectory, add_plot_seg_boxes

def eval_synth_data(segments, time_arr, clusters):

    # Get the actual stay indices and create the labels for each event
    from synthetic_data.trajectory import get_stay_indices
    
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