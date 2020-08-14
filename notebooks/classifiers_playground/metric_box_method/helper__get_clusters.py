import numpy as np
#from numpy import mean as np.mean

from helper__gaps import merge_cluster_pair, gap_criterion_3

get_err = lambda x1, x2: abs(x1-x2) #np.sqrt((x1-x2)**2)
    
def get_clusters_1(t_arr, x_arr, d_thresh, t_thresh, verbose=False):
    """
    Get a list of cluster indices, v. 1

    :param t_arr: np.array Trajectory array of timepoints
    :param t_arr: np.array Trajectory array of locations
    :param d_thresh: float temporal buffer between timepoints 
    :param t_thresh: float spatial buffer around clusters      
    
    :return: [int] indices of identified clusters 
    
    """
    
    # Want to use this here to keep the scope variables
    def check_and_merge_clusters(clusters):

        # Check whether the new cluster is can be merged with the previous one
        c1, c2 = clusters[-2],clusters[-1]

        if gap_criterion_3(d_thresh, t_thresh)(t_arr, x_arr,c1,c2):            
            # Keep `min_index`? Here is it fixed
            min_index = len(clusters)-2
            return merge_cluster_pair(clusters, min_index).copy()  
        else:
            return clusters
        
    # Output list of indices: [[beg., end],[beg., end], ...] 
    clusters = []

    m = 0

    # The current cluster indices: [n_0, n_1, ... ]
    new_cluster = [m]
    
    # Pass through the list of events
    for n in range(0,x_arr.size-3):

        # Check: is the time within the time thresh?
        #
        #print(m,n,'pre')
        if abs(t_arr[n+1] - t_arr[n]) <= t_thresh:
            event_loc = x_arr[n+1]
        else: 
            print(m,n,'cont')
            continue

        if m == n:
            continue
        
        # Get the current cluster mean
        cluster_mean = np.mean(x_arr[m:n+1])

        # Get the potential cluster mean    
        new_cluster_mean = np.mean(\
            np.concatenate([x_arr[m:n+1],x_arr[n+1].reshape(1,)])\
                )

        err1 = get_err(cluster_mean, event_loc)
        err2 = get_err(cluster_mean, new_cluster_mean)
        
        if verbose: print(f"{m:5d}, {n:5d}, {err1:7.3f}, {err2:7.3f}, {d_thresh:7.3f}")
 
        # Checks: 
        # 1. new event is within dist. thresh of current clust.
        # 2. new mean - current mean is within dist. thresh.
        if  (err1 < d_thresh) & (err2 < d_thresh) & \
            (n <= x_arr.size-1):
            if verbose: print(f"\tappending {n:4d}")
            new_cluster.append(n)
        else:
            # Save the current cluster and prepare restart
            txt = f'Trying {n} '
            app = "Nope"
            if (len(new_cluster) >= 2):
                # Since this is an incomplete method (clusters will be merged after), 
                # can keep this; otherwise, would lose small clusters
                clusters.append(new_cluster)
                app = 'closed'

                if verbose: print("\t\tLast cluster:", new_cluster)
                clusters.append(new_cluster)
                clusters = check_and_merge_clusters(clusters)            
        
            # Update starting point
            m=n+1

            #if verbose: print(txt+app)
            new_cluster = [m]
            
    return clusters


def get_clusters_2(t_arr, x_arr, d_thresh, t_thresh, verbose=False):
    """
    Get a list of cluster indices, v. 2

    :param t_arr: np.array Trajectory array of timepoints
    :param t_arr: np.array Trajectory array of locations
    :param d_thresh: float temporal buffer between timepoints 
    :param t_thresh: float spatial buffer around clusters      
    
    :return: [int] indices of identified clusters 
    
    """
    
    get_err = lambda x1, x2: abs(x1-x2) #np.sqrt((x1-x2)**2)
    testdata_20200814/trajectory_3stays__prec0o949_rec0o597.pkl
    # Output list of indices: [[beg., end],[beg., end], ...] 
    clusters = []

    m = 0

    # The current cluster indices: [n_0, n_1, ... ]
    new_cluster = []
    
    # Pass through the list of events
    #for n in range(0,x_arr.size-3):
    n = 0
    while n < x_arr.size-3:
        # Check: is the time within the time thresh?
        if abs(t_arr[n+1] - t_arr[n]) <= t_thresh:
            event_loc = x_arr[n+1]
        else: 
            n+=1
            continue

        # Get the current cluster mean
        cluster_mean = np.mean(x_arr[m:n+1])

        # Get the potential cluster mean    
        new_cluster_mean = np.mean(x_arr[m:n+2])

        err1 = get_err(cluster_mean, event_loc)
        err2 = get_err(cluster_mean, new_cluster_mean)
        
        #if verbose: print(f"{n:5d}, {err1:7.3f}, {err2:7.3f}, {d_thresh:7.3f}")
 
        # Checks: 
        # 1. new event is within dist. thresh of current clust.
        # 2. new mean - current mean is within dist. thresh.
        txt = f'Try {n:4d}: '
        app = ""
        if  (err1 < d_thresh) & (err2 < d_thresh):
            #if verbose: print("append")
            new_cluster.append(n)
        else:
            # Save the current cluster and prepare restart
            app = "breaking"
            if (len(new_cluster) >= 2):
                # Since this is an incomplete method (clusters will be merged after), 
                # can keep this; otherwise, would lose small clusters
                #if (abs(t_arr[new_cluster[-1]]-t_arr[new_cluster[0]]) > t_thresh):
                clusters.append(new_cluster)
                app = 'closing'
            #if verbose: print(txt+app)
            new_cluster = []

            # Update starting point
            m=n+1
                    
        if verbose: print(txt, f"err1 = {err1:6.3f}, err2 = {err2:6.3f}", app)
        n+=1
    
    n-=1
    #print(n, new_cluster)
    if len(new_cluster) > 0:
        m = new_cluster[0]

        while (n < x_arr.size) & (new_cluster[0] < n):    
            # Get the current cluster mean
            cluster_mean = np.mean(x_arr[m:n+1])

            # Get the potential cluster mean    
            new_cluster_mean = np.mean(x_arr[m:n+1])

            err1 = get_err(cluster_mean, event_loc)
            err2 = get_err(cluster_mean, new_cluster_mean)    
            if len(new_cluster) != 0:
                txt = f'Try {n:4d}: '
                app = ""
                if  (err1 < d_thresh) & (err2 < d_thresh):
                    #if verbose: print("append")
                    new_cluster.append(n)
                else:
                    # Save the current cluster and prepare restart
                    app = "breaking"
                    if (len(new_cluster) >= 2):
                        # Since this is an incomplete method (clusters will be merged after), 
                        # can keep this; otherwise, would lose small clusters
                        #if (abs(t_arr[new_cluster[-1]]-t_arr[new_cluster[0]]) > t_thresh):
                        clusters.append(new_cluster)
                        app = 'closing'

            if verbose: print(txt, f"err1 = {err1:6.3f}, err2 = {err2:6.3f}", app)                

            n+=1
        if len(new_cluster) > 0:
            clusters.append(new_cluster)                
    
    
    if x_arr.size-1 not in clusters[-1]:
        if verbose: print(f"{x_arr.size:4d} is not in: [{clusters[-1][0]:4d},{clusters[-1][-1]:4d}]" )
    else:
        if verbose: print("Done")
            
    return clusters


def get_clusters_3(t_arr, x_arr, d_thresh, t_thresh, verbose=False):
    """
    Get a list of cluster indices, v. 3

    :param t_arr: np.array Trajectory array of timepoints
    :param t_arr: np.array Trajectory array of locations
    :param d_thresh: float temporal buffer between timepoints 
    :param t_thresh: float spatial buffer around clusters      
    
    :return: [int] indices of identified clusters
    
    TODOs: 
    * rename self to ``
    * Update the debugging output
    * rename `gap_criterion_3` to `gap_criterion` (usw)
        * `gap_criterion` then calls `gap_criterion_1` & `gap_criterion_2`, etc. 
    """
    get_err = lambda x1, x2: abs(x1-x2) #np.sqrt((x1-x2)**2)
    
    # Want to use this here to keep the scope variables
    def check_and_merge_clusters(clusters):

        # Check whether the new cluster is can be merged with the previous one
        c1, c2 = clusters[-2],clusters[-1]

        if gap_criterion_3(d_thresh, t_thresh)(t_arr, x_arr,c1,c2):            
            # Keep `min_index`? Here is it fixed
            min_index = len(clusters)-2
            return merge_cluster_pair(clusters, min_index).copy()  
        else:
            return clusters

        
    # Output list of indices: [[beg., end],[beg., end], ...] 
    clusters = []
    
    # Initialize working indices
    m = 0    
    new_cluster = [m]
    
    # Pass through the list of events    
    for n in range(1, x_arr.size):
        
        # Check: is the time within the time thresh?
        # NOTE: this will keep going until it finds a timepoint pair 
        # within the thresh. 
        # If there's an open cluster, that cluster will get closed
        # at the last added point once no new points can be added 
        # based on the distance criteria. 
        # This can allow already for some long-duration clusters
        # as long as they don't exceed the distance threshold.
        if abs(t_arr[n] - t_arr[n-1]) > t_thresh:
            if n >= x_arr.size-1:
                pass
            else:
                continue
        elif n == x_arr.size-1:
            # This ensures that will keep the last event
            pass
        
        # Get the last event's location
        if n < x_arr.size-1:
            new_x = x_arr[n].reshape(1,)
        else:
            new_x = x_arr[n-1].reshape(1,)
            
        # Get the current cluster's mean
        cluster_mean = np.mean(x_arr[m:n])

        # Get the potential cluster's mean    
        new_cluster_mean = np.mean(\
            np.concatenate([x_arr[m:n],new_x])\
                )        

        err1 = get_err(cluster_mean, new_x)
        err2 = get_err(cluster_mean, new_cluster_mean)
        
        #if verbose: print(f"{m:5d}, {n:5d}, {err1:6.3f}, {err2:6.3f}")
        
        # Checks: 
        # 1. new event is within dist. thresh of current clust.
        # 2. new mean - current mean is within dist. thresh.
        if  ((err1 < d_thresh) & (err2 < d_thresh)) & (n < x_arr.size):
            new_cluster.append(n)        
        else:
            # Save the current cluster and prepare restart
            if (len(new_cluster) >= 2):
                # Since this is an incomplete method (clusters will be merged after), 
                # can keep this; otherwise, would lose small clusters
                #if (abs(t_arr[new_cluster[-1]]-t_arr[new_cluster[0]]) > t_thresh):
                clusters.append(new_cluster)
                if verbose: print(f"\tAppended: [{new_cluster[0]:4d}, {new_cluster[-1]:4d}]")

                # Check whether the new cluster is can be merged with the previous one
                # NOTE: this combines the logic from `merge_clusters` function
                if len(clusters) > 1:
                    clusters = check_and_merge_clusters(clusters)
                    if (clusters[-1][0] != new_cluster[0]) & verbose:
                        print("\t\tMerged")
                        
            # Update starting point 
            # TODO: n or n+1?
            m=n
            
            # TODO: start with m?
            new_cluster = [m]
            
        # If there is an open cluster and the limit is reached, append it.
        if verbose: print(n, len(new_cluster))
        if (n >= x_arr.size-2) & (len(new_cluster) > 0):
            clusters.append(new_cluster)
            if verbose: print(f"\tFinal append: [{new_cluster[0]:4d}, {new_cluster[-1]:4d}]")
            # and check and merge it
            clusters = check_and_merge_clusters(clusters)
            if (clusters[-1][0] != new_cluster[0]) & verbose:
                print("\t\tFinal merge")
            new_cluster = []
            
    return clusters


def get_clusters_4(t_arr, x_arr, d_thresh, t_thresh, verbose=False):
    """
    Get a list of cluster indices, v. 4

    :param t_arr: np.array Trajectory array of timepoints
    :param t_arr: np.array Trajectory array of locations
    :param d_thresh: float temporal buffer between timepoints 
    :param t_thresh: float spatial buffer around clusters      
    
    :return: [int] indices of identified clusters 
    
    TODOs: 
    * rename self to ``
    * Update the debugging output
    * rename `gap_criterion_3` to `gap_criterion` (usw)
        * `gap_criterion` then calls `gap_criterion_1` & `gap_criterion_2`, etc. 
        
    ALGO:
    1. start with m,n as [m,..,n]
    2. Check if n+1 is within time-thresh
        2.1. F: 
            2.1.1. close and merge current (if current)
            2.1.2. update m,n
        2.2. T: Check if n+1 meets dist criteria
            2.2.1. F: 
                2.2.1.1. close and merge current (if current)
                2.2.1.2. update m,n        
            2.2.2. T:
                2.2.1.1. add to current
                2.2.1.2. update n        
            
    """
    
    # Want to use this here to keep the scope variables
    def check_and_merge_clusters(clusters):

        # Check whether the new cluster is can be merged with the previous one
        c1, c2 = clusters[-2],clusters[-1]

        if gap_criterion_3(d_thresh, t_thresh)(t_arr, x_arr,c1,c2):            
            # Keep `min_index`? Here is it fixed
            min_index = len(clusters)-2
            return merge_cluster_pair(clusters, min_index).copy()  
        else:
            return clusters

        
    # Output list of indices: [[beg., end],[beg., end], ...] 
    clusters = []

    # The current cluster indices: [n_0, n_1, ... ]
    new_cluster = []
    
    # Pass through the list of events
    #for n in range(0,x_arr.size-3):
    # Initialize working indices
    m, n = 0, 1
        
    new_cluster = [m]

    while n < x_arr.size-1:
        
        # Check time between consecutive events
        time_criterion = False
                
        # Check: is the time within the time thresh?
        if abs(t_arr[n+1] - t_arr[n]) <= t_thresh:
            if verbose: print(f"{m:5d}, {n:5d}\tTime passed")            
            new_x = x_arr[n+1]
            time_criterion = True

        # Check distance between consecutive events
        dist_criterion = False
        if time_criterion:
                    
            new_x = x_arr[n+1].reshape(1,)
            
            # Get the current cluster mean
            cluster_mean = np.mean(x_arr[m:n+1])

            # Get the potential cluster mean    
            new_cluster_mean = np.mean(\
                np.concatenate([x_arr[m:n+1],new_x])\
                    )

            err1 = get_err(cluster_mean, new_x[0])
            err2 = get_err(cluster_mean, new_cluster_mean)
            
            if  (err1 < d_thresh) & (err2 < d_thresh):
                dist_criterion = True
                if verbose: print(f"{m:5d}, {n:5d}\tDist passed: {err1:6.3f}, {err2:6.3f}")
 
        if dist_criterion:
            # both distance and time (implicitly) are okay
            if verbose: print("\t\tAdding event")
            new_cluster.append(n)
            n += 1
            continue
            
        elif time_criterion: 
            # the time is okay, but distance not --> close
            criterion = "dist. criterion failed"            
            
        else:
            # Both dist and time are exceeded --> close
            criterion = "time criterion failed"
            
        # Checks: 
        # 1. new event is within dist. thresh of current clust.
        # 2. new mean - current mean is within dist. thresh.

 
        if verbose: print(f"\n    Closing cluster: {criterion}, {len(new_cluster)}, {time_criterion}\n")
        if verbose: print(f"\tfrom {m:5d}, {n:5d}")

        # Save the current cluster and prepare restart            
        if ((len(new_cluster) >= 2) & (time_criterion)):
            # Since this is an incomplete method (clusters will be merged after), 
            # can keep this; otherwise, would lose small clusters
            if verbose: print("\t\tnew cluster:", new_cluster)
            clusters.append(new_cluster)
            m = n
        else:
            m += 1
        n = m+1
        
        if verbose: print(f"\t  to {m:5d}, {n:5d}")
            
        new_cluster = [m]

        # Check whether the new cluster is can be merged with the previous one
        # NOTE: this combines the logic from `merge_clusters` function
        if len(clusters) > 1:
            clusters = check_and_merge_clusters(clusters)

    # Save the last cluster if non-empty         
    if len(new_cluster) > 2:
        # Since this is an incomplete method (clusters will be merged after), 
        # can keep this; otherwise, would lose small clusters
        if verbose: print("\t\tLast cluster:", new_cluster)
        clusters.append(new_cluster)
        clusters = check_and_merge_clusters(clusters)
    
    if x_arr.size-1 not in clusters[-1]:
        print(new_cluster)
        if verbose: print(f"{x_arr.size:4d} is not in: [{clusters[-1][0]:4d},{clusters[-1][-1]:4d}]" )
    else:
        if verbose: print("Done")
            
    return clusters


subcluster_lengths = lambda cluster_list: [len(c) for c in cluster_list]

'''
Debugging output for get_clusters_3, near the end:
    # Since the last part of the trajectory array is possibly skipped,
    # need to close last potential cluster    
    print(f"Checking last cluster: [{clusters[-1][0]:4d}, {clusters[-1][-1]:4d}],",\
          f"{clusters[-1][-1]:4d} < {x_arr.size-1:4d}: {clusters[-1][-1]<x_arr.size-1}")
    
    indices = ""
    if len(new_cluster) > 0:
        indices = f"[{new_cluster[0]:4d}, {new_cluster[-1]:4d}], ",\
                  f"{clusters[-1][-1]:4d} < {x_arr.size-1:4d}: {clusters[-1][-1]<x_arr.size-1}"
    print(f"Checking for open cluster: {len(new_cluster):4d} > 0: {len(new_cluster)>0}" + indices)
'''
