import numpy as np
#import pandas as pd

get_err = lambda x1, x2: np.sqrt((x1-x2)**2)

def get_clusters(t_arr, loc_arr, dist_thresh, time_thresh, verbose=False):

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
        if t_arr[n+1] <= t_arr[n]+time_thresh:
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
                if (t_arr[new_cluster[-1]]-t_arr[new_cluster[0]] > time_thresh):
                    clusters.append(new_cluster)
                    app = 'closed'
 
            new_cluster = []

            # Update starting point
            m=n+1
            
    return clusters


def get_iqr_masked(loc_arr, iqr_fact = 3):
    # Calculate first and third quartiles
    q25 = np.quantile(loc_arr,0.25, interpolation='lower')
    q75 = np.quantile(loc_arr,0.75, interpolation='higher')

    # Calculate the interquartile range (IQR)
    iqr = abs(q75 - q25)

    #print(m, nn, np.where((yyyy[m:nn] > (q25 - iqr_fact * iqr)) & (yyyy[m:nn] < (q75 + iqr_fact * iqr))))
    mask=np.where(  (loc_arr > (q25 - iqr_fact * iqr)) \
                  & (loc_arr < (q75 + iqr_fact * iqr)))    
    
    return mask

def get_iqr_std(loc_arr, iqr_fact = 3):
    
    mask=get_iqr_masked(loc_arr, iqr_fact)
    
    return np.std(loc_arr[mask])
'''
#NOTE:
This method is incorrect: it's very susceptible to drift since
the accumulation of clusters is based initially on time; 
this means temporal density will create clusters rather than
the spatial nearness. 
'''
def get_batch_clusters(t_arr, loc_arr, dist_thresh, time_thresh, iqr_fact=3.0, verbose=False):

    max_len = loc_arr.size-6
    
    clusters = []


    # Initializations
    cluster = [0]
    last_time_point = t_arr[cluster[0]]
    m = 0   

    ind = 0     
    while ind < max_len:

        # Set the time buffer around the last point of the current cluster
        nn=m+1
        # Build up a cluster based on temporal density:
        # from a given point in time, 
        # get the greatest time within the time threshold;
        # repeat until not possible.        
        # NOTE: this is not sensible!
        while (t_arr[nn]-last_time_point <= time_thresh) & (nn<=max_len):
            nn+=1
        
        # Measure the current cluster: std. dev.
        if loc_arr[m:nn].size > 1:
            cluster_std = np.std(loc_arr[m:nn])
            cluster_qstd = get_iqr_std(loc_arr[m:nn],iqr_fact)
        else:
            cluster_std = 0.0
            cluster_qstd = 0.0
        
        nnn=nn
        while ((loc_arr[m:nnn].size > 1) & ((cluster_std >= dist_thresh) | (cluster_qstd >= dist_thresh))):
            if verbose: print('\t\trefinement', nnn, cluster_std, cluster_qstd)
            # Get the current cluster std
            cluster_std = np.std(loc_arr[m:nnn])
            cluster_qstd = get_iqr_std(loc_arr[m:nnn],iqr_fact)
            nnn-=1

        # Check!
        if verbose: print('STD-testing at', nn, cluster_std, cluster_qstd, dist_thresh)
        # 
        new_cluster = list(range(m,nnn+1))

        # if the std's are good, keep the cluster, and update the final time point
        if  (cluster_std < dist_thresh) & (cluster_qstd < dist_thresh) & (nnn<max_len) & (nnn==nn) :

            last_time_point = t_arr[nnn]
            ind = nnn

        else:
            txt = f'Trying {len(new_cluster)} '
            app = "Nope"
            if (len(new_cluster) >= 2):
                if (t_arr[new_cluster[-1]]-t_arr[new_cluster[0]] > time_thresh):
                    clusters.append(new_cluster)
                    app = 'closed'
            if verbose: print(txt+app)

            new_cluster = []
            if (nnn==nn):
                ind=nn
                m=nn
            else:
                ind=nnn
                m=nnn
            
    return clusters

