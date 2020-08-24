import numpy as np

from synthetic_data.trajectory import get_stay_indices, get_adjusted_stays
from sklearn.metrics import precision_score, recall_score, confusion_matrix

#from helper__3stays_v3_scripts import inter_bounds, contains, conta_bounds

from .metric_box_classifier_core import get_mini_clusters
from .metric_box_classifier_boxing import extend_clusters, shift_cluster_box
from .metric_box_classifier_gaps import merge_clusters_gen
from .metric_box_classifier_split import separate_clusters
from .metric_box_classifier_mask import get_iqr_filtered_clusters

from stay_classification.cluster_helper import print_clusts


get_cluster_ranges = lambda clusters: [list(range(c[0],c[-1]+1)) for c in clusters]

get_sorted_clusters = lambda clusters: sorted([sorted(c) for c in clusters]) 


def stay_classifier_testing(t_arr, x_arr, d_thresh, t_thresh, iqr_trim=True, verbose=False):
    
    all_clusters = []

    stage_nr = 1
    try:
        if verbose: print(f"Stage 1: get mini-clusters and merging")
        clusters = get_mini_clusters(t_arr, x_arr, d_thresh, t_thresh)
        clusters = get_sorted_clusters(clusters)        
        if verbose:
            print(len(clusters), "Clusters:")
            print_clusts(clusters);           
        all_clusters.append(clusters.copy())
        stage_nr +=1
    except:                    
        print("Failed at",stage_nr)
        return None
    
    try:
        if verbose: print(f"Stage {stage_nr}: extend clusters and IQR-filter")
        clusters = extend_clusters(t_arr, x_arr, clusters, t_thresh)
        clusters = get_sorted_clusters(clusters)        
        if verbose:
            print(len(clusters), "Clusters:")
            print_clusts(clusters);
        all_clusters.append(clusters.copy())
        stage_nr +=1
    except:                    
        print("Failed at",stage_nr)
        pass

    try:
        if verbose: print(f"Stage {stage_nr}: separate overlapping clusters")
        clusters = separate_clusters(clusters)
        clusters = get_sorted_clusters(clusters)        
        if verbose:
            print(len(clusters), "Clusters:")
            print_clusts(clusters);
        all_clusters.append(clusters.copy())
        stage_nr +=1
    except:                    
        print("Failed at",stage_nr)
        pass

    try:
        if verbose: print(f"Stage {stage_nr}: merge nearby clusters")
        clusters = merge_clusters_gen(t_arr, x_arr, clusters, d_thresh, t_thresh)
        clusters = get_sorted_clusters(clusters)        
        if verbose:
            print(len(clusters), "Clusters:")
            print_clusts(clusters); 
        all_clusters.append(clusters.copy())
        stage_nr +=1
    except:                    
        print("Failed at",stage_nr)
        pass

    try:    
        if verbose: print(f"Stage {stage_nr}: shift the boxes")
        clusters = shift_cluster_box(t_arr, x_arr, clusters, t_thresh, d_thresh)
        clusters = get_sorted_clusters(clusters)        
        if verbose:
            print(len(clusters), "Clusters:")
            print_clusts(clusters);
        all_clusters.append(clusters.copy())
        stage_nr +=1
    except:                    
        print("Failed at",stage_nr)
        pass

    try:
        if verbose: print(f"Stage {stage_nr}: filter regions by IQR")

        if iqr_trim:        
            clusters = get_iqr_filtered_clusters(x_arr, clusters, 1.5)
            all_clusters.append(clusters.copy())
            if verbose:
                print(len(clusters), "Clusters:")
                print_clusts(clusters);
        else: 
            if verbose: print("No IQR-trim")        
    except:                    
        print("Failed at",stage_nr)

    return all_clusters
    
