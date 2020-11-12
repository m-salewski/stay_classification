import numpy as np

from synthetic_data.trajectory import get_stay_indices, get_adjusted_stays
    
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

from stay_classification.cluster_helper import inter_bounds

from .metrics_errors import get_segments_errs_core, print_total_errors, print_errors_result

#TODO
# [ ] check if any of these is still needed or relevant 

# Some older metrics (TODO: check if still needed and/or update, also in the NBs)
def eval_synth_data(t_arr, segments, clusters):
    """
    """
    # Get the actual stay indices and create the labels for each event
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
    prec = precision_score(true_labels, pred_labels)
    rec  = recall_score(true_labels, pred_labels)

    # Eval. using confuse. matrix
    conf_mat = confusion_matrix(true_labels, pred_labels)
    
    return prec, rec, conf_mat


def eval_synth_data_clusters(t_arr, segments, clusters, travels=False):
    """
    Evaluate based on individual clusters
    """
    
    expected_nr_of_stays = int((len(segments)+1)/2)
    predicted_nr_of_stays = len(clusters)
    
    if expected_nr_of_stays != predicted_nr_of_stays:
        return np.nan, np.nan, np.nan
    
    # Get the actual stay indices and create the labels for each event
    true_indices = get_stay_indices(get_adjusted_stays(segments, t_arr), t_arr)
    true_labels = np.zeros(t_arr.shape)

    for pair in true_indices:
        true_labels[pair[0]:pair[1]+1] = 1
    
    # Flip to get the values for the other class
    if travels:
        true_labels = 1 - true_labels
        
    # Get the predicted labels for each event
    final_pairs = []
    for clust in clusters:
        final_pairs.append([clust[0],clust[-1]])
    pred_labels = np.zeros(t_arr.shape)

    for pair in final_pairs:
        pred_labels[pair[0]:pair[1]+1] = 1

    # Flip to get the values for the other class    
    if travels:
        pred_labels = 1 - pred_labels
    
    # Evaluate using prec. and rec.    
    prec = precision_score(true_labels, pred_labels)
    rec  = recall_score(true_labels, pred_labels)

    # Eval. using confuse. matrix
    conf_mat = confusion_matrix(true_labels, pred_labels)
    
    return prec, rec, conf_mat


def eval_synth_data_clusters_long(t_arr, segments, clusters, travels=False):
    """
    Evaluate based on individual clusters
    """
    
    expected_nr_of_stays = int((len(segments)+1)/2)
    predicted_nr_of_stays = len(clusters)
    
    # Get the actual stay indices and create the labels for each event
    true_indices = get_stay_indices(get_adjusted_stays(segments, t_arr), t_arr)
    true_labels = np.zeros(t_arr.shape)

    for pair in true_indices:
        true_labels[pair[0]:pair[1]+1] = 1
    
    # Flip to get the values for the other class
    if travels:
        true_labels = 1 - true_labels
        
    # Get the predicted labels for each event
    final_pairs = []
    for clust in clusters:
        final_pairs.append([clust[0],clust[-1]])
    pred_labels = np.zeros(t_arr.shape)

    for pair in final_pairs:
        pred_labels[pair[0]:pair[1]+1] = 1

    # Flip to get the values for the other class    
    if travels:
        pred_labels = 1 - pred_labels
    
    # Evaluate using prec. and rec.    
    prec = precision_score(true_labels, pred_labels)
    rec  = recall_score(true_labels, pred_labels)

    # Eval. using confuse. matrix
    conf_mat = confusion_matrix(true_labels, pred_labels)
    
    return prec, rec, conf_mat


# NOTE: these are still useful; quicker and dirtier versions of the optimal methods
def eval_traj_stay(tr_stays, pr_stays, tr_labs, pr_labs, t_arr, time_weight_flag, verbose=False):
    """
    What it does not do:
    * if a base cluster overlaps with multiple clusters, it cannot distinguish these,
    _ie_ rec(true[tr_cl], pred[tr_cl]) will be larger if tr_cl overlaps with two pred clusters,
         eventhough this is incorrect.
    """
    from sklearn.metrics import precision_score, recall_score
    
    # Using the pred stays as the base: 
    # - all events of a pred stay are 1
    # - events from the pred cluster are correct stay events or incorrect stay events
    #   - hence, use precision(true, pred)    
    rec_score = get_score(precision_score, pr_stays, tr_labs, pr_labs, t_arr, time_weight_flag, verbose)
    if verbose: print(f"\tPred clusters score ={rec_score:6.3f}")    
    
    # Using the true stays as the base: 
    # - all events of a true stay are 1
    # - events from the pred cluster are correct stay events or incorrect travel events
    #   - hence, use recall(true, pred)
    prec_score = get_score(recall_score, tr_stays, tr_labs, pr_labs, t_arr, time_weight_flag, verbose)
    if verbose: print(f"\tTrue clusters score ={prec_score:6.3f}")
    if verbose: print()
    
    return accuracy_score(tr_labs,pr_labs), prec_score, rec_score


def get_score(fnc, clusts, tr_labs, pr_labs, t_arr, duration, verbose=False):
    """
    
    """     
    # Weighting (NOTE: doing here for debugging)
    if duration:
        # Weight by duration
        weight_func = lambda c: abs(t_arr[c[-1]]-t_arr[c[0]])
    else: 
        # Weight by segment count        
        weight_func = lambda c: len(c)            
        
    denom = 0 # accumlulator
    for n,c in enumerate(clusts):
        # Get segment length
        denom += weight_func(c)
    if duration:
        text = 'Time'
    else: 
        text = 'Size'
    if verbose: print(f"{text} weighting: denom = {denom:6.3f}") 
            
    # Iterate through the clusters
    final_score = 0        
    for n,c in enumerate(clusts):
        
        # Get segment duration
        tdiff = abs(t_arr[c[-1]]-t_arr[c[0]])
        
        #tdenom += tdiff
        if verbose: print(f"  seg. {n:2d}: c = [{c[0]:4d},{c[-1]:4d}], l = {len(c):5d}, dt = {tdiff:6.3f}")
        
        # Get the sub arrays
        tr_sub = tr_labs[c[0]:c[-1]+1]
        pr_sub = pr_labs[c[0]:c[-1]+1]
        if verbose: print(f"\t\t1: true ={sum(tr_sub):6.0f} vs pred ={sum(pr_sub):6.0f}")
        
        # Accuracy: this is equiv to
        # * recall_score(tr_sub, pr_sub) when clusts are trues
        # * precision_score(tr_sub, pr_sub) when clusts are preds
        # * in general, sum(tr_sub)/sum(pr_sub) since these are arrays of 0's and 1's
        #   * which should also be faster
        score = sum(tr_sub)/sum(pr_sub) #
        score = fnc(tr_sub,pr_sub)   
        if verbose: print(f"\t\t1: score ={score:6.3f}")
        #, precision_score(pr_sub, tr_sub), recall_score(pr_sub, tr_sub))              
        
        # Weighting     
        if duration:
            # Weight by duration
            numer = tdiff
        else:        
            # Weight by count: this is equiv to
            # * accuracy_score(tr_labs[tr_labs==1],pr_labs[tr_labs==1]) when clusts are trues
            # * accuracy_score(tr_labs[pr_labs==1],pr_labs[pr_labs==1]) when clusts are preds
            numer = float(len(c))
        # Final segment weight
        weight = numer/denom
        
        if verbose: print(f"\t\t2: numer = {numer:6.3f}; denom = {denom:6.3f}")        
        
        # Weight the segment score
        score = weight*score
        if verbose: print(f"\t\t2: score ={score:6.3f}\n")
        
        # Accumulate
        final_score += score
    
    return final_score
