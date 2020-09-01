import numpy as np
import numpy.ma as ma


get_err = lambda x1, x2: np.sqrt((x1-x2)**2) 


get_box_bounds = lambda sub_arr, eps: (np.mean(sub_arr), np.mean(sub_arr)+eps, np.mean(sub_arr)-eps)


def get_mini_box(t_ind_0, t_arr, x_arr, d_thresh, t_thresh, verbose=False):
    """
    Find the longest box of a given width to contain the maximum number of signaling events

    :param t_ind_0: int Starting time index
    :param t_arr: np.array Trajectory array of timepoints
    :param x_arr: np.array Trajectory array of locations
    :param d_thresh: float width of box
    :param t_thresh: float buff around timepoint      

    :return: int endpoint index of a region
    """

    # Set the (initial) metrics for the 'box' -- could update along the way
    t_ind = t_ind_0 + 1
    _, upper, lower = get_box_bounds(x_arr[t_ind_0:t_ind], d_thresh)

    # Set the sizes: exit once size does not change from last_size
    last_size = 0
    curr_size = 1

    # Loop with an updating index
    # NOTE: this might crash/hang when the t_index is out-of-bounds
    while True:

        # Extend the box forward in time; get the greatest time index in this region
        curr_t_ind = np.where(t_arr <= t_arr[t_ind]+t_thresh)[0].max()

        # Count all events within the 'box'
        event_inds = np.where((x_arr[t_ind_0:curr_t_ind] <= upper) & 
                              (x_arr[t_ind_0:curr_t_ind] >= lower))[0]
        curr_size = event_inds.size

        # Report
        if verbose: print(f'last size: {last_size:4d}, '\
                          f'current size: {curr_size:4d}, '\
                          f'indices: [{t_ind_0:4d}, {curr_t_ind:4d}]')

        # Check if the current size <= the last_size: no cluster increase,
        # so exit the loop.
        # This means the search quits as soon as there's an spatial outlier 
        # --> will catch more if the break criterion is more relaxed.
        if curr_size <= last_size:
            if verbose: 
                print(f'\tlast size: {last_size:4d} > '\
                      f'current size: {curr_size:4d}: break!')            
            break
        else:
            # Update last size
            last_size = curr_size

        # Update the box
        _, upper, lower = get_box_bounds(x_arr[t_ind_0:t_ind], d_thresh)

        # Update the time index
        t_ind = curr_t_ind

    return t_ind


def quick_box_method(t_arr, x_arr, d_thresh, t_thresh, prefactor=1, verbose=False):
    """
    Decompose an event sequence into stays by identifying which clusters of events fall within
    the allowed distance and time thresholds which classify a stay.

    All boxes are of the same width (d_thresh) and are at least t_thresh long.

    Boxes are finalized when no events are included within the distance thresh when 
    extending the edge by the time threshold.

    :param t_arr: np.array Trajectory array of timepoints
    :param x_arr: np.array Trajectory array of locations
    :param d_thresh: float Width of box
    :param t_thresh: float Length of time to extend the box
    :param prefactor: float Extends the time-threshold in the cluster classifier
    :param verboss: bool Flag to produce output.

    :return: [[int,int]] endpoint indices of the identified stays
    """

    curr_t_ind = 0

    clusters = []

    clust_nr = 1
    # Skip through indices
    while curr_t_ind < len(t_arr)-2:

        # Get the cluster indices starting from a given time point
        new_t_ind = get_mini_box(curr_t_ind, t_arr, x_arr, d_thresh, prefactor*t_thresh)
        
        # Get the indices for reporting
        lino = f"[{curr_t_ind:5d} {new_t_ind:5d}]"
        
        # Get the subarrays
        ys =  x_arr[curr_t_ind:new_t_ind]
        xs = t_arr[curr_t_ind:new_t_ind]

        # Get the box around the subarray
        _, upper, lower = get_box_bounds(ys, d_thresh)
        
        # Some measurements
        event_inds = np.where((ys <= upper) & (ys >= lower))[0]
        if event_inds.size == 0:
            curr_t_ind += 1
            continue
        
        t_diff = abs(xs[event_inds].max()-xs[event_inds].min())

        # If the duration is large-enough, save the cluster & update the indices
        new_t_ind_min = event_inds.min()+curr_t_ind
        new_t_ind_max = event_inds.max()+curr_t_ind
        if t_diff > t_thresh:
            clusters.append(list(range(new_t_ind_min, new_t_ind_max+1)))
            
            new_t_ind = new_t_ind_max
            
            # Get the indices for reporting
            lino2 = f"[{new_t_ind_min:5d} {new_t_ind:5d}]"

            if verbose: 
                print(f"Cl. Nr. {clust_nr:3d}: {lino}, "\
                      f"{event_inds.size:5d}, {t_diff:8.3f}: "\
                      f"append cluster: {lino2}")     
                
            clust_nr += 1
        else:
            if verbose: 
                print(f"             {lino}, {event_inds.size:5d}, {t_diff:8.3f} ",\
                      f"--> too short; ")#,\
                      #f"{xs[event_inds].max():6.3f},{xs[event_inds].min():6.3f} {upper:7.3f},{lower:7.3f}")
            
        curr_t_ind = new_t_ind

    return clusters
