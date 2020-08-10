import numpy as np

from synthetic_data.trajectory import get_stay_indices, get_adjusted_stays
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from helper__3stays_v3_scripts import inter_bounds, contains, conta_bounds


def eval_synth_data_clusters(segments, time_arr, clusters):

    """
    Evaluate based on individual clusters
    """
    
    expected_nr_of_stays = int((len(segments)+1)/2)
    predicted_nr_of_stays = len(clusters)
    
    if expected_nr_of_stays != predicted_nr_of_stays:
        return np.nan, np.nan, np.nan
    
    # Get the actual stay indices and create the labels for each event
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
    prec = precision_score(true_labels, pred_labels)
    rec  = recall_score(true_labels, pred_labels)

    # Eval. using confuse. matrix
    conf_mat = confusion_matrix(true_labels, pred_labels)
    
    return prec, rec, conf_mat


def get_segments_scores(t_arr, segments, pred_clusters, verbose=False):
    
    true_clusters = []
    for n in range(0,len(segments),2):
        
        true_clusters.append(get_stay_indices(get_adjusted_stays(segments[n:n+1], t_arr), t_arr)[0] )
    
    print(f"Predicted {len(pred_clusters)} of {len(true_clusters)} true clusters ")
    
    precs = []
    recs = []
    
    for n, true_clust in enumerate(true_clusters):

        overlapping_pred_clusters = []
        for nn, pred_clust in enumerate(pred_clusters):
            
            true_labels = np.zeros(t_arr.shape)
            true_labels[true_clust[0]:true_clust[1]+1] = 1
        
            if inter_bounds(true_clust, pred_clust):    

                pair = [min(true_clust[0],pred_clust[0]), max(true_clust[-1],pred_clust[-1])]
                
                overlapping_pred_clusters.append(nn)
                #if verbose: print(true_clust, "overlaps", pair)

                pred_labels = np.zeros(t_arr.shape)
                pred_labels[pred_clust[0]:pred_clust[-1]+1] = 1      

                precs.append(precision_score(true_labels, pred_labels))
                recs.append(recall_score(true_labels, pred_labels)        )

                #if verbose: print(f"Cluster {nn:3d}, precision: {precs[-1]:6.3f}; recall: {recs[-1]:6.3f}")
            else:
                pass
                #if verbose: print("No overlap:", true_clust, "and", pair)
                
        if verbose: print(f"\tCluster {n:3d}, [{true_clust[0]:4d},{true_clust[-1]:4d}]")
        if len(overlapping_pred_clusters) > 0:
            if verbose: print(f"\t\toverlaps with {len(overlapping_pred_clusters)} pred_cluster(s):")                
            for m in overlapping_pred_clusters:
                print(f"\t\t\t[{pred_clusters[m][0]:4d},{pred_clusters[m][-1]:4d}]")
            if verbose: print(f"\t\t\tprecision: {precs[-1]:6.3f};\n\t\t\trecall: {recs[-1]:6.3f}")   
        else:
            if verbose: print(f"\t\tNo overlap")
        if verbose: print()


    if verbose: print(f"Stats: \n\tmin. precision: {min(precs):6.3f}; min. recall: {min(recs):6.3f}")
    if verbose: print(f"\tavg. precision: {sum(precs)/len(precs):6.3f}; avg. recall: {sum(recs)/len(recs):6.3f}")

    true_labels = np.zeros(t_arr.shape)
    for n, true_clust in enumerate(true_clusters):    
        true_labels[true_clust[0]:true_clust[1]+1] = 1
        
    pred_labels = np.zeros(t_arr.shape)
    for nn, pred_clust in enumerate(pred_clusters):
        pred_labels[pred_clust[0]:pred_clust[-1]+1] = 1
                        
    prec = precision_score(true_labels, pred_labels)
    rec  = recall_score(true_labels, pred_labels)           
    if verbose: print(f"\ttot. precision: {prec:6.3f}; tot. recall: {rec:6.3f}")    
    
    total_duration = t_arr[-1]-t_arr[0]
    true_stay_durations = 0
    for n, true_clust in enumerate(true_clusters):
        true_stay_durations += t_arr[true_clust[-1]]-t_arr[true_clust[0]]
        
    pred_stay_durations = 0
    for nn, pred_clust in enumerate(pred_clusters):
        pred_stay_durations += t_arr[pred_clust[-1]]-t_arr[pred_clust[0]]
    
    if verbose: print(f"\nDurations:",\
                      f"\ntot. trajectory duration: {total_duration:6.3f}",\
                      f"\ntot. true stays duration: {true_stay_durations:6.3f}",\
                      f"({true_stay_durations/total_duration:5.3f}%)",\
                      f"\ntot. pred stays duration: {pred_stay_durations:6.3f}",\
                      f"({pred_stay_durations/total_duration:5.3f}%)")
    
    return precs, recs


def print_p_and_r(clusters, prec, rec, conmat, direction):
    if direction: print('forward')
    print("\t", len(clusters), 'clusters:', subcluster_lengths(clusters))
    print(f"\t{prec:6.3f}")
    print(f"\t{ rec:6.3f}")
    #print(conmat)
