import numpy as np

from synthetic_data.trajectory import get_stay_indices, get_adjusted_stays
    
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from stay_classification.cluster_helper import inter_bounds


def get_labels_from_clusters(clusters, shape):
    """
    Get the stay (1) and travel (0) labels from a set of clusters
    """
    
    # Loop through the clusters to get the end points;
    # create array of one & zeros (stays & travels) 
    labels = np.zeros(shape)
    for clust in clusters:            
        labels[clust[0]:clust[-1]+1] = 1

    return labels


def get_pred_labels(clusters, shape):
    """
    Get the stay (1) and travel (0) labels from a set of clusters
    """
    
    # Loop through the clusters to get the end points;
    # create array of one & zeros (stays & travels) 
    pred_labels = np.zeros(shape)
    for clust in clusters:            
        pred_labels[clust[0]:clust[-1]+1] = 1

    return pred_labels


def eval_synth_data(t_arr, segments, clusters):

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


def get_segments_scores(t_arr, segments, pred_clusters, verbose=False):
    """
    Evaluate the individual segments from a predicted classification
    """
    
    get_duration = lambda ind1, ind2: abs(t_arr[ind2]-t_arr[ind1])
    
    # Get the predicted and true number of clusters
    true_clusters = []
    for n in range(0,len(segments),2):
        
        true_clusters.append(list(get_stay_indices(get_adjusted_stays(segments[n:n+1], t_arr), t_arr)[0] ))
    
    precs, recs = get_segments_scores_core(t_arr, true_clusters, pred_clusters, verbose)
    
    return precs, recs


def get_segments_scores_core(t_arr, true_clusters, pred_clusters, verbose=False):
    """
    Evaluate the individual segments from a predicted classification
    """
    
    get_duration = lambda ind1, ind2: abs(t_arr[ind2]-t_arr[ind1])
    
    if verbose: print(f"Predicted {len(pred_clusters)} of {len(true_clusters)} true clusters\n")

    total_duration = get_duration(0,-1)
    true_durations = 0
    for true_clust in true_clusters:
        true_durations += get_duration(true_clust[0],true_clust[-1])
        
    pred_durations = 0
    for pred_clust in pred_clusters:
        pred_durations += get_duration(pred_clust[0],pred_clust[-1])
    
    durs = []
    precs = []
    recs = []
    w_precs = []
    w_recs = []
    
    
    # Loop through the pred. clusters, determine which true clusters they belong to and measure their p/rec scores    
    for nn, pred_clust in enumerate(pred_clusters):

        dur = get_duration(pred_clust[0], pred_clust[-1])

        if verbose: 
            print(f"Cluster {nn:3d}: [{pred_clust[0]:4d},{pred_clust[-1]:4d}];"\
                  f" dur: {dur:6.3f}, (frac. {(dur/pred_durations):6.3f})")
        
        overlapping_pred_clusters = []
        
        precs_ = []
        recs_ = []
        
        # Loop through the true clusters
        for n, true_clust in enumerate(true_clusters):
            
            true_labels = np.zeros(t_arr.shape)
            true_labels[true_clust[0]:true_clust[-1]+1] = 1
        
            # Check if there is an overlap
            true_clust_str = 13*" "+f"[{true_clust[0]:4d},{true_clust[-1]:4d}]"
            if inter_bounds(true_clust, pred_clust):    

                pair = [min(true_clust[0],pred_clust[0]), max(true_clust[-1],pred_clust[-1])]
                
                # Save overlap index of the true cluster
                overlapping_pred_clusters.append(n)
                
                
                pred_labels = np.zeros(t_arr.shape)
                pred_labels[pred_clust[0]:pred_clust[-1]+1] = 1      

                precs_.append(precision_score(true_labels, pred_labels))
                recs_.append(recall_score(true_labels, pred_labels))              
                
                if verbose: 
                    print(true_clust_str+", overlap")
            else:  
                if verbose: print(true_clust_str+", none")
                pass
        
        nr_overlaps = len(precs_)
        #print(len(overlapping_pred_clusters), len(precs), len(recs), nr_overlaps)
        if len(overlapping_pred_clusters) > 0:
            if verbose: print(f"\n\toverlaps with {len(overlapping_pred_clusters)} true cluster(s):")                
            for mm, m in enumerate(overlapping_pred_clusters):
                #mm = -1*nr_overlaps + mm
                if verbose: print(f"{m:11d}: [{true_clusters[m][0]:4d},{true_clusters[m][-1]:4d}];" \
                      f" prec.: {precs_[mm]:6.3f}; rec.: {recs_[mm]:6.3f}") 
                #print(f" prec.: {precs[mm]:6.3f}; rec.: {recs[mm]:6.3f}")                
        else:
            if verbose: 
                print(f"\n\tNo overlap")
                print(f"\t{19*' '} prec.: {0:6.3f}; rec.: {0:6.3f}")
                
        # When the 
        if nr_overlaps > 0:
            """"for nnn in range(len(precs_)):
                
                prec = precs_[nnn]/nr_overlaps
                rec = recs_[nnn]/nr_overlaps
                
                precs.append(prec)
                recs.append(rec)"""
            
            prec = sum(precs_)/nr_overlaps
            rec = sum(recs_)/nr_overlaps
            
            #print(nn,prec,rec)
            precs.append(prec)
            recs.append(rec)
            
            w_precs.append(prec*dur/pred_durations)
            w_recs.append(rec*dur/pred_durations)    
        else:
            precs.append(0.)
            recs.append(0.)

            w_precs.append(0.0)
            w_recs.append(0.0)
        
        if verbose: print()

    spacers = 4 
    if verbose: print(f"Stats: \n{spacers*' '}min. prec.: {min(precs):6.3f};{spacers*' '}min. rec.: {min(recs):6.3f}")
    if verbose: print(f"{spacers*' '}avg. prec.: {sum(precs)/len(precs):6.3f};{spacers*' '}avg. rec.: {sum(recs)/len(recs):6.3f}")
    if verbose: print(f"{(spacers-2)*' '}w-avg. prec.: {sum(w_precs):6.3f};{(spacers-2)*' '}w-avg. rec.: {sum(w_recs):6.3f}")
    
    true_labels = np.zeros(t_arr.shape)
    for n, true_clust in enumerate(true_clusters):    
        true_labels[true_clust[0]:true_clust[1]+1] = 1
        
    pred_labels = np.zeros(t_arr.shape)
    for nn, pred_clust in enumerate(pred_clusters):
        pred_labels[pred_clust[0]:pred_clust[-1]+1] = 1
                        
    prec = precision_score(true_labels, pred_labels)
    rec  = recall_score(true_labels, pred_labels)           
    if verbose: print(f"{(spacers)*' '}tot. prec.: {prec:6.3f};{(spacers)*' '}tot. rec.: {rec:6.3f}")    
        
    if verbose: print(f"\nDurations:",\
                      f"\ntot. trajectory duration: {total_duration:6.3f}",\
                      f"\ntot. true stays duration: {true_durations:6.3f}",\
                      f"({true_durations/total_duration:5.3f})",\
                      f"\ntot. pred stays duration: {pred_durations:6.3f}",\
                      f"({pred_durations/total_duration:5.3f})")
    
    return precs, recs


def get_segments_errs(t_arr, segments, pred_clusters, verbose=False):
    """
    Evaluate the individual segments from a predicted classification
    """
    
    get_duration = lambda ind1, ind2: abs(t_arr[ind2]-t_arr[ind1])
    
    # Get the predicted and true number of clusters
    true_clusters = []
    for n in range(0,len(segments),2):
        
        true_clusters.append(list(get_stay_indices(get_adjusted_stays(segments[n:n+1], t_arr), t_arr)[0] ))
    
    errs = get_segments_errs_core(t_arr, true_clusters, pred_clusters, verbose)
    
    return errs


def get_segments_errs_core(t_arr, true_clusters, pred_clusters, verbose=False):
    """
    Evaluate the individual segments from a predicted classification
    """
    get_err = lambda trues, preds: np.sum(abs(trues-preds))/trues.size
    
    get_duration = lambda ind1, ind2: abs(t_arr[ind2]-t_arr[ind1])
    
    if verbose: print(f"Predicted {len(pred_clusters)} of {len(true_clusters)} true clusters\n")

    total_duration = get_duration(0,-1)
    true_durations = 0
    for true_clust in true_clusters:
        true_durations += get_duration(true_clust[0],true_clust[-1])
        
    pred_durations = 0
    for pred_clust in pred_clusters:
        pred_durations += get_duration(pred_clust[0],pred_clust[-1])
    
    spacers = 4
    
    # Loop through the pred. clusters, and determine
    # which true clusters they overlap, then measure the err    
    durs = []
    errs = []
    w_errs = []
    for nn, pred_clust in enumerate(pred_clusters):

        # Get the labels for the current pred cluster        
        pred_labels = np.zeros(t_arr.shape)
        pred_labels[pred_clust[0]:pred_clust[-1]+1] = 1   
        
        # Get the duration for the curr pred cluster
        dur = get_duration(pred_clust[0], pred_clust[-1])

        if verbose: 
            print(f"Cluster {nn:3d}: [{pred_clust[0]:4d},{pred_clust[-1]:4d}];"\
                  f" dur: {dur:6.3f}, (frac. {(dur/pred_durations):6.3f})")
        

        # Loop through the true clusters        
        overlapping_pred_clusters = []        
        errs_ = []
        
        for n, true_clust in enumerate(true_clusters):
                
            # Get the labels for the current true cluster 
            true_labels = np.zeros(t_arr.shape)
            true_labels[true_clust[0]:true_clust[-1]+1] = 1

            true_clust_str = 13*' '+f'[{true_clust[0]:4d},{true_clust[-1]:4d}]'
            
            # Check if there is an overlap
            if inter_bounds(true_clust, pred_clust):

                #pair = [min(true_clust[0],pred_clust[0]), max(true_clust[-1],pred_clust[-1])]
            
            
                # Save index of the overlapping true cluster
                overlapping_pred_clusters.append(n)
                   
                
                errs_.append(get_err(true_labels, pred_labels))              
                
                if verbose: 
                    print(true_clust_str+', overlap')
            else:  
                if verbose: print(true_clust_str+', none')
                pass
        
        nr_overlaps = len(errs_)
        
        #if verbose: print('*',len(overlapping_pred_clusters), len(errs), nr_overlaps)
                
        # Average the errs when multiple overlaps
        if nr_overlaps > 0:
            """"for nnn in range(len(errs_)):
                
                err = errs_[nnn]/nr_overlaps
                
                errs.append(err)"""
            
            err = sum(errs_)/nr_overlaps
            
        else:
            err = get_err(0*pred_labels, pred_labels)
        
        w_err = err*dur/pred_durations
        errs.append(err)
        w_errs.append(w_err)
        
        if verbose:
            spacers = 1
            if len(overlapping_pred_clusters) > 0:
                print(f"\n\tOverlap with {len(overlapping_pred_clusters)} true cluster(s):")                
                for mm, m in enumerate(overlapping_pred_clusters):
                    #mm = -1*nr_overlaps + mm
                    print(f"{m:11d}; [{true_clusters[m][0]:4d},{true_clusters[m][-1]:4d}];" \
                          f"{spacers*' '}err.: {errs_[mm]:6.3f};" \
                          f"{(spacers)*' '}w-avg. err.: {errs_[mm]*dur/pred_durations:6.3f}")               
            else:
                spacers = 10
                print(f"\n\tNo overlap")
                print(f"{25*' '} err.: {err:6.3f};" \
                      f"{(1)*' '}w-avg. err.: {w_err:6.3f}")                
            print()

    spacers = 4
    if verbose: 
        print(f"Stats: \n{spacers*' '}max. err.: {max(errs):6.3f}")
        print(f"{spacers*' '}avg. err.: {sum(errs)/len(errs):6.3f}")
        print(f"{(spacers-2)*' '}w-avg. err.: {sum(w_errs):6.3f}")
    
    # Some additional measures of the error
    # NOTE: when there are no true stays shared among multiplt pred stays,
    # then the total err == N_{pred.stays}*(avg. err.)
    true_labels = get_labels_from_clusters(true_clusters, t_arr.shape)
    pred_labels = get_labels_from_clusters(pred_clusters, t_arr.shape)
    err  = get_err(true_labels, pred_labels)
    
    if verbose: print(f"{(spacers)*' '}tot. err.: {err:6.3f}")
    
    return errs


subcluster_lengths = lambda cluster_list: [len(c) for c in cluster_list]

def print_p_and_r(clusters, prec, rec, conmat):
    #if direction: print('forward')
    print(f"{len(clusters):5d} clusters, lengths:, {subcluster_lengths(clusters)}")
    print(f"\tprec.:{prec:6.3f}")
    print(f"\t rec.:{ rec:6.3f}")
    print(conmat)
