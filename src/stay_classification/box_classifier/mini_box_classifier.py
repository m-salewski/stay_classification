import numpy as np
import numpy.ma as ma


get_err = lambda x1, x2: np.sqrt((x1-x2)**2) 


get_box_bounds = lambda sub_arr, eps: (np.mean(sub_arr), np.mean(sub_arr)+eps, np.mean(sub_arr)-eps)


def get_mini_box(time_ind_0, t_arr, loc_arr, dist_thresh, time_thresh, verbose=False):

    """
    Find the longest box of a given width to contain the maximum number of signaling events

    :param time_ind_0: int Starting timepoint
    :param time_ind: int Current timepoint    
    :param t_arr: np.array Trajectory array of timepoints
    :param loc_arr: np.array Trajectory array of locations
    :param dist_thresh: float Width of box
    :param time_thresh: float buff around timepoint      
    
    :return: int endpoint index of a region
    
    """
    # Set the (initial) metrics for the 'box' -- could update along the way
    time_ind = time_ind_0+1
    _, upper, lower = get_box_bounds(loc_arr[time_ind_0:time_ind], dist_thresh)
    
    # Set the sizes: exit once size does not change from last_size
    last_size = 0
    curr_size = 1
    
    increase_box = True
    while increase_box:

        # Extend the box forward in time; get the greatest timepoint in this region
        new_time = t_arr[time_ind]+time_thresh
        latest_time_ind = np.where(t_arr<=new_time)[0].max()
        
        # Using a sub-array, count all events within the 'box'
        subarr = loc_arr[time_ind_0:latest_time_ind]
        event_inds = np.where((subarr <= upper) & (subarr >= lower))[0]
        curr_size = event_inds.size

        # Report
        if verbose: print(last_size, '\t', curr_size,  '\t', latest_time_ind)

        # Check if the current size equals the last_size, and break
        # This means the search quits as soon as there's an outlier 
        # --> will catch more if the break criterion is more relaxed.
        if last_size == curr_size:
            break
        else:
            last_size = curr_size

        # Update the box
        _, upper, lower = get_box_bounds(loc_arr[time_ind_0:time_ind], dist_thresh)
    
        # Update the time index
        time_ind = latest_time_ind

    return time_ind


def mini_box_method(t_arr, loc_arr, dist_thresh, time_thresh, prefactor=1, verbose=False):

    latest_time_ind = 0 
    
    clusters = []

    clust_nr = 1
    while latest_time_ind+20 < len(t_arr):

        # Get the cluster indices starting from a given time point
        new_latest_time_ind = get_mini_box(latest_time_ind,t_arr, loc_arr, dist_thresh, prefactor*time_thresh)

        # Get the subarra
        ys =  loc_arr[latest_time_ind:new_latest_time_ind]
        xs = t_arr[latest_time_ind:new_latest_time_ind]

        # Get the box
        _, upper, lower = get_box_bounds(ys, dist_thresh)

        lino = f"{latest_time_ind:5d} {new_latest_time_ind:5d}"
        if new_latest_time_ind+1 >= len(t_arr): 
            new_latest_time_ind = len(t_arr)-2
        lino2 = new_latest_time_ind

        event_inds = np.where((ys <= upper) & (ys >= lower))[0]

        time_diff = abs(xs[event_inds].max()-xs[event_inds].min())

        if time_diff > time_thresh:
            clusters.append([
                event_inds.min()+latest_time_ind,
                event_inds.max()+latest_time_ind
                ])
            if verbose: print(f"Cl. Nr. {clust_nr}: {lino}, {event_inds.size:5d}, {time_diff:8.3f}")        
            clust_nr += 1

        latest_time_ind = new_latest_time_ind
        
    return clusters
