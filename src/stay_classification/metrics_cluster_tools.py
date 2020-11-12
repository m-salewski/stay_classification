import numpy as np

from synthetic_data.trajectory import get_stay_indices, get_adjusted_stays
    
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

from stay_classification.cluster_helper import inter_bounds

#TODO
# [x] move all generic functions related to measuring/manipulating clusters to cluster_helper
# [ ] update all dependencies of this submodule

def get_accumulated_result(fnt, iterable):
    """
    Iterate through clusters and accumulate the total length
    """
    accumulated_result = 0

    for iter_elem in iterable:
        accumulated_result += fnt(iter_elem)

    return accumulated_result

# TODO gather these into cluster_helper
get_time_duration = lambda t_arr: lambda ind1, ind2: abs(t_arr[ind2]-t_arr[ind1])
get_clust_duration = lambda t_arr: lambda clust: abs(t_arr[clust[-1]]-t_arr[clust[0]])
get_clust_length = lambda clust: len(clust)
subcluster_lengths = lambda cluster_list: [len(c) for c in cluster_list]


# def get_clusters_duration(get_clust_duration(t_arr),clusts)
def get_clusters_duration(t_arr, clusts):
    """
    Iterate through clusters and accumulate the total duration
    """
    accumulated_result = 0

    for clust in clusts:
        accumulated_result += get_clust_duration(t_arr)(clust)

    return accumulated_result


# def get_clusters_duration(get_clust_length,clusts)
def get_clusters_length(t_arr, clusts):
    """
    Iterate through clusters and accumulate the total length
    """
    accumulated_result = 0

    for clust in clusts:
        accumulated_result += get_clust_length(clust)

    return accumulated_result


def get_subcluster_labels(t_arr):
    """
    Get the stay (1) and travel (0) labels for a single cluster
    """
    def meth(clust):
        labels = np.zeros(t_arr.shape)
        labels[clust[0]:clust[-1]+1] = 1 
        return labels
    
    return meth


#TODO: this is a duplicate method; use a closure: get_all_labels(shape)(clusters)
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


#TODO: this is a duplicate method
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


def get_true_clusters(t_arr, segments):
    """
    Get the predicted and true number of clusters
    """
    true_clusters = []
    for n in range(0,len(segments),2):
        indices = get_stay_indices(get_adjusted_stays(segments[n:n+1], t_arr), t_arr)[0]
        true_clusters.append(list(range(indices[0],indices[-1]+1)))
        
    return true_clusters
