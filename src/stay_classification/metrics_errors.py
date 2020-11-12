import numpy as np
    
from sklearn.metrics import accuracy_score

from .cluster_helper import inter_bounds

from .metrics_cluster_tools import get_clust_duration, get_subcluster_labels, get_labels_from_clusters, get_time_duration, get_clust_length
from .metrics_cluster_tools import get_clusters_duration, get_clusters_length, get_true_clusters

#TODO
# [ ] follow-up get_segment_errs to check if it makes sense 

get_err = lambda trues, preds: np.sum(abs(trues-preds))/trues.size

def get_err_short(true_labels, pred_labels): 
    """
    """
    # Find the max size of overlap between clusters
    ranges = np.where(true_labels==1)[0]
    if ranges.size > 1:
        tmin,tmax = ranges[0],ranges[-1]+1
    else:
        tmin,tmax = len(true_labels),0
    
    ranges = np.where(pred_labels==1)[0]
    if ranges.size > 1:    
        pmin,pmax = ranges[0],ranges[-1]+1
    else:
        pmin,pmax = len(pred_labels),0
    
    # Use the length of the max overlap
    # TODO: div0 error!
    denom = max(tmax,pmax) - min(tmin,pmin)
    
    return np.sum(abs(true_labels-pred_labels))/denom


def get_segments_errs_core(t_arr, base_clusts, test_clusts, pred_against_true=True, verbose=False):
    """
    Evaluate the individual segments from a predicted classification
    """
    get_clust_dur = lambda clust: get_clust_duration(t_arr)(clust)
    get_sc_labels = lambda clust: get_subcluster_labels(t_arr)(clust)
    
    # NOTE: this flag should align with the outer loop below
    # the default assumes check each test_clust (predictions) with 
    # each base cluster (trues)
    if pred_against_true:
        test_flag = 'pred'
        base_flag = 'true'
    else:
        test_flag = 'true'
        base_flag = 'pred'

    if verbose: print(f"Comparing {len(test_clusts)} {test_flag} to {len(base_clusts)} {base_flag} clusters\n")

    total_test_duration = get_clusters_duration(t_arr, test_clusts)    
    total_test_length = get_clusters_length(t_arr, test_clusts)
    
    all_overlapping_test_clusts = []
    
    # Loop through the pred. clusters, and determine
    # which true clusters they overlap, then measure the err    
    durs = []
    errs = []
    c_errs = []
    t_errs = []
    
    for nn, test_clust in enumerate(test_clusts):
        # Get the labels for the current test cluster        
        test_labels = get_sc_labels(test_clust)
        
        # Get the duration for the curr test cluster
        dur = get_clust_dur(test_clust)
        cnt = get_clust_length(test_clust)
        
        if verbose:
            print(f"Cluster {nn:3d}: [{test_clust[0]:4d},{test_clust[-1]:4d}];"\
                f"  count: {cnt:4d},  dur: {dur:6.3f}")
            print(f"{26*' '}(frac. {(cnt/total_test_length):5.3f})"
                f" (frac. {(dur/total_test_duration):5.3f})\n")          

        # Loop through the true clusters        
        overlapping_test_clusts = []        
        errs_ = []
        
        for n, base_clust in enumerate(base_clusts):
            # Get the labels for the current true cluster 
            base_labels = get_sc_labels(base_clust)
            
            base_clust_str = 13*' '+f'[{base_clust[0]:4d},{base_clust[-1]:4d}]'
            
            # Check if there is an overlap
            if inter_bounds(base_clust, test_clust):

                #pair = [min(base_clust[0],test_clust[0]), max(base_clust[-1],test_clust[-1])]
                
                # Save index of the overlapping true cluster
                overlapping_test_clusts.append(n)
                
                errs_.append(get_err_short(base_labels, test_labels))              
                
                if verbose: 
                    print(base_clust_str+', overlap')
                    ##print(len(base_clust),len(test_clust))
                    
            else:  
                if verbose: print(base_clust_str)#+', none')
                pass
        
        nr_overlaps = len(errs_)
                
        # Average the errs when multiple overlaps
        if nr_overlaps > 0:
            """"for nnn in range(len(errs_)):
                
                err = errs_[nnn]/nr_overlaps
                
                errs.append(err)"""
            
            err = sum(errs_)/nr_overlaps
            
        else:
            err = 1.0#get_err(0*test_labels, test_labels)
        #print(err)
        c_err = err*cnt/total_test_length
        t_err = err*dur/total_test_duration
        errs.append(err)
        c_errs.append(c_err)
        t_errs.append(t_err)
        
        if verbose:
            spacers = 1
            if len(overlapping_test_clusts) > 0:
                all_overlapping_test_clusts.extend(overlapping_test_clusts)
                print(f"\n\tOverlaps with {len(overlapping_test_clusts)} {base_flag} cluster(s):")                
                for mm, m in enumerate(overlapping_test_clusts):
                    #mm = -1*nr_overlaps + mm
                    print(f"{m:11d}; [{base_clusts[m][0]:4d},{base_clusts[m][-1]:4d}];" \
                          f"{spacers*' '}err.: {errs_[mm]:6.3f}" \
                          f"{(spacers-2)*' '}; c-avg. err.: {errs_[mm]*cnt/total_test_length:6.3f}"
                          f"{(spacers-2)*' '}; t-avg. err.: {errs_[mm]*dur/total_test_duration:6.3f}")               
            else:
                spacers = 10
                print(f"\n\tNo overlap")
                print(f"{25*' '} err.: {err:6.3f};" \
                      f"{(1)*' '}c-avg. err.: {c_err:6.3f};" \
                      f"{(1)*' '}t-avg. err.: {t_err:6.3f}")
            print()
            
    # Get the missed true clusters
    missing_base_clusts = list(set(range(len(base_clusts)))\
                     .difference(set(all_overlapping_test_clusts)))
    if len(missing_base_clusts) > 0:
        if verbose: print(f'{base_flag} clusters', missing_base_clusts, \
                          f'do not overlap any {test_flag} clusters\n')
        
    spacers = 4
    if verbose: 
        print(f"Partial stats\n{spacers*' '}max. err.: {max(errs):6.3f}")
        print(f"{(spacers-2)*' '}c-avg. err.: {sum(c_errs):6.3f}")
        print(f"{(spacers-2)*' '}t-avg. err.: {sum(t_errs):6.3f}")

    max_err = max(errs)
    cavg_err = sum(c_errs)#/len(errs)
    tavg_err = sum(t_errs)

    return max_err, cavg_err, tavg_err


def print_total_errors(t_arr, true_clusts, pred_clusts, verbose=False):
    # Some additional measures of the error
    # NOTE: when there are no true stays shared among multiplt pred stays,
    # then the total err == N_{pred.stays}*(avg. err.)
    true_labels = get_labels_from_clusters(true_clusts, t_arr.shape)
    pred_labels = get_labels_from_clusters(pred_clusts, t_arr.shape)
    
    err  = get_err(true_labels, pred_labels)

    spacers = 4
    if verbose: 
        print(f"\n{(spacers)*' '}tot. err.: {err:6.3f}; "\
              f" 1-acc.: {(1-accuracy_score(true_labels, pred_labels)):6.3f}\n")
    
    return None


def print_errors_result(result, verbose=False):
    """
    """
    spacers = 4   
    if verbose: 
        print(f"\nTotal Stats\n{spacers*' '}max. err.: {result[0]:6.3f};")
        print(f"{(spacers-2)*' '}c-avg. err.: {result[1]:6.3f};")
        print(f"{(spacers-2)*' '}t-avg. err.: {result[2]:6.3f}")
        
    return None
