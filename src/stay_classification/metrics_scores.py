import numpy as np

from synthetic_data.trajectory import get_stay_indices, get_adjusted_stays
    
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

from stay_classification.cluster_helper import inter_bounds

from .metrics_cluster_tools import get_clust_duration, get_time_duration, get_clusters_duration
from .metrics_cluster_tools import get_subcluster_labels, get_labels_from_clusters
from .metrics_cluster_tools import get_clusters_length, get_clust_length, subcluster_lengths
from .metrics_cluster_tools import get_true_clusters

#TODO
# [ ] check all dependencies are maintained

def get_segments_scores_core(t_arr, base_clusts, test_clusts, pred_against_true=True, verbose=False):
    """
    Evaluate the individual segments from a predicted classification
    """
    get_time_dur = lambda ind1, ind2: get_time_duration(t_arr)(ind1, ind2)
    get_clust_dur = lambda clust: get_clust_duration(t_arr)(clust)
    
    # NOTE: this flag should align with the outer loop below
    # the default assumes check each test_clust (predictions) with 
    # each base cluster (trues)
    if pred_against_true:
        test_flag = 'pred'
        base_flag = 'true'
        fnt1 = precision_score
        fnt2 = recall_score
    else:
        test_flag = 'true'
        base_flag = 'pred'
        #NOTE: swap the functions when the inputs are swapped.
        # The reason is that always ensuring prec(true, pred)
        # _ie_ prec(true, pred) = prec(base_clust, test_clust)
        #      prec(true, pred) =  rec(base_clust, test_clust)
        fnt1 = recall_score
        fnt2 = precision_score
        
    if verbose: print(f"Comparing {len(test_clusts)} {test_flag} to {len(base_clusts)} {base_flag} clusters\n")

    total_duration = get_time_duration(t_arr)(0,-1)
    base_durations = get_clusters_duration(t_arr, base_clusts)
    test_durations = get_clusters_duration(t_arr, test_clusts) 
    
    base_lengths = get_clusters_length(t_arr, base_clusts)
    test_lengths = get_clusters_length(t_arr, test_clusts)
    
    durs = []
    precs = []
    recs = []
    a_precs = []
    a_recs = []    
    w_precs = []
    w_recs = []
    
    # Collect the overlapped true cluster indices
    all_overlapping_test_clusts = []
    
    # Loop through the test clusters,
    #  determine which base clusters they overlap with
    #  and measure their p/rec scores while ensuring p/rec(true, pred)
    for nn, test_clust in enumerate(test_clusts):

        dur = get_clust_dur(test_clust)
        cnt = get_clust_length(test_clust)

        if verbose: 
            print(f"Cluster {nn:3d}: [{test_clust[0]:4d},{test_clust[-1]:4d}];"\
                f"  count: {cnt:4d},  dur: {dur:6.3f}")
            print(f"{26*' '}(frac. {(cnt/test_lengths):5.3f})"
                f" (frac. {(dur/test_durations):5.3f})\n")            
        overlapping_test_clusts = []
        
        precs_ = []
        recs_  = []
        
        # Loop through the base clusters
        for n, base_clust in enumerate(base_clusts):
            
            # For shape-consistency, artificially inflate the base cluster
            # NOTE: this allows cluster comparisons
            base_labels = np.zeros(t_arr.shape)
            base_labels[base_clust[0]:base_clust[-1]+1] = 1
        
            # Check if there is an overlap
            base_clust_str = 13*" "+f"[{base_clust[0]:4d},{base_clust[-1]:4d}]"
            if inter_bounds(base_clust, test_clust):    

                pair = [min(base_clust[0],test_clust[0]), max(base_clust[-1],test_clust[-1])]
                
                # Save overlap index of the true cluster
                overlapping_test_clusts.append(n)
                
                # As above, artificially inflate the test cluster
                test_labels = np.zeros(t_arr.shape)
                test_labels[test_clust[0]:test_clust[-1]+1] = 1      
                
                # Careful! using precision(A,B) = recall(B,A)
                precs_.append(fnt1(base_labels, test_labels))
                recs_.append( fnt2(base_labels, test_labels))
                
                #first pass 
                #prec(true,pred)
                # rec(true, pred)
                #second pass
                # rec(pred,true) == prec(true,pred)
                #prec(pred,true) ==  rec(true,pred)
                
                if verbose: 
                    print(base_clust_str+", overlap")
            else:  
                if verbose: print(base_clust_str)#+", none")
                pass
        
        nr_overlaps = len(precs_)
        #print(len(overlapping_test_clusts), len(precs), len(recs), nr_overlaps)
        if len(overlapping_test_clusts) > 0:
            all_overlapping_test_clusts.extend(overlapping_test_clusts)
            if verbose: print(f"\n\tOverlaps with {len(overlapping_test_clusts)} {base_flag} cluster(s):")                
            for mm, m in enumerate(overlapping_test_clusts):
                #mm = -1*nr_overlaps + mm
                if verbose: print(f"{m:11d}: [{base_clusts[m][0]:4d},{base_clusts[m][-1]:4d}];" \
                      f" prec.: {precs_[mm]:6.3f}; rec.: {recs_[mm]:6.3f}") 
        else:
            if verbose: 
                print(f"\n\tNo overlap")
                print(f"\t{19*' '} prec.: {0:6.3f}; rec.: {0:6.3f}")
                
        # When there are overlaps, average the scores among the overlapping clusters 
        if nr_overlaps > 0:            
            prec = sum(precs_)/nr_overlaps
            rec = sum(recs_)/nr_overlaps
            
            #print(nn,prec,rec)
            precs.append(prec)
            recs.append(rec)
            
            a_precs.append(prec*cnt/test_lengths)
            a_recs.append(rec*cnt/test_lengths)  
            
            w_precs.append(prec*dur/test_durations)
            w_recs.append(rec*dur/test_durations)    
        else:
            precs.append(0.)
            recs.append(0.)
            
            a_precs.append(0.0)
            a_recs.append(0.0)
            
            w_precs.append(0.0)
            w_recs.append(0.0)
        
        if verbose: print()

    # Get the missed true clusters
    missing_base_clusts = list(set(range(len(base_clusts)))\
                     .difference(set(all_overlapping_test_clusts)))
    if len(missing_base_clusts) > 0:
        if verbose: print(f'{base_flag} clusters', missing_base_clusts, \
                          f'do not overlap any {test_flag} clusters\n')
        # This doesn't make sense to add to the average
        #precs.append(0.)
        #recs.append(0.)
        
        # This won't show up since the weights aren't accounted for
        #w_precs.append(0.0)
        #w_recs.append(0.0) 
        
    spacers = 4 
    if verbose: 
        print(f"Partial stats: \n{spacers*' '}min. prec.: {min(precs):6.3f};{spacers*' '}min. rec.: {min(recs):6.3f}")
        #print(f"{spacers*' '}avg. prec.: {sum(precs)/len(precs):6.3f};{spacers*' '}avg. rec.: {sum(recs)/len(recs):6.3f}")
        print(f"{(spacers-2)*' '}c-avg. prec.: {sum(a_precs):6.3f};{(spacers-2)*' '}c-avg. rec.: {sum(a_recs):6.3f}")
        print(f"{(spacers-2)*' '}t-avg. prec.: {sum(w_precs):6.3f};{(spacers-2)*' '}t-avg. rec.: {sum(w_recs):6.3f}")
    
    min_prec = min(precs)
    min_rec = min(recs)
    avg_prec = sum(a_precs)#/len(precs)
    avg_rec = sum(a_recs)#/len(recs)
    wavg_prec = sum(w_precs)
    wavg_rec = sum(w_recs)
    
    return min_prec, avg_prec, wavg_prec, min_rec, avg_rec, wavg_rec


def print_total_scores(t_arr, true_clusts, pred_clusts, verbose=False):
    """
    """
    spacers = 4
    
    true_labels = get_labels_from_clusters(true_clusts, t_arr.shape)
    pred_labels = get_labels_from_clusters(pred_clusts, t_arr.shape)
                        
    prec = precision_score(true_labels, pred_labels)
    rec  = recall_score(true_labels, pred_labels)           
    
    if verbose: print(f"\n{(spacers)*' '}tot. prec.: {prec:6.3f};{(spacers)*' '}tot. rec.: {rec:6.3f}\n")    
    
    return None


def print_scores_result(result, verbose=False):
    """
    """
    spacers = 4
    if verbose: 
        print(f"\nOverall stats: \n{spacers*' '}min. prec.: {result[0]:6.3f};"\
              f"{spacers*' '}min. rec.: {result[3]:6.3f}")
        print(f"{(spacers-2)*' '}c-avg. prec.: {result[2]:6.3f};"\
              f"{(spacers-2)*' '}c-avg. rec.: {result[5]:6.3f}")
        print(f"{(spacers-2)*' '}t-avg. prec.: {result[2]:6.3f};"\
              f"{(spacers-2)*' '}t-avg. rec.: {result[5]:6.3f}")   
        
    return None
