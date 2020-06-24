import numpy as np

from .checks import check_means

def get_directional_indices(t_arr, timepoint, buffer, direction):
    
    if direction == -1:
        return np.where(((t_arr < (timepoint)) & \
                         (t_arr > (timepoint - buffer))))        
    else:
        return np.where(((t_arr < (timepoint + buffer)) & \
                         (t_arr > timepoint)))    

def get_time_ind(t_arr, timepoint, time_thresh, direction):
    """
    Get the index of a region bounded by a timepoint +/- a buff

    :param t_arr: np.array Trajectory array of timepoints
    :param timepoint: float Timepoint
    :param time_thresh: float buff around timepoint  
    :param direction: int gets the min or max index of a region
    
    :return: int endpoint index of a region
    
    """
    
    indices = np.array([[]])
    n = 1
    within_limits = True
    while ((within_limits) & (indices[0].size == 0)):
        
        indices = get_directional_indices(t_arr, timepoint, n*time_thresh, direction)
        
        # Ensure the moving edge is within the time array
        within_limits = ((direction*n*time_thresh <= t_arr.max()) & \
                         (direction*n*time_thresh >= t_arr.min()))
        
        n+=1

    if indices[0].size != 0:
        if direction == 1:
            return indices[0].max() 
        else: 
            return indices[0].min() 
    else:
        #If size == 0 ==> while exited due to limits
        if direction == 1:
            return t_arr.shape[0]-1
        else: 
            return 0  
            
    
def extend_edge(t_arr, x_arr, working_index, fixed_index, means, configs, verbose=False):
    """
    Extend the edge of a potential box to include more points. 

    :param t_arr: np.array Trajectory array of timepoints
    :param x_arr: np.array Trajectory array of locations
    :param working_index: np.array Timepoint used to find the box    
    :param configs: dict containing the parameters used to define the box
    :param verbose: bool To select printing metrics to stdio
    
    :return: [float] means for the extension.
    :return: [int] new indices included in the extension
    :return: bool Indicates whether the means have converged
    
    """
    
    count_thresh = configs['count_thresh']
    eps = configs['eps']
    
    keep_running = (working_index > 1) & (working_index < len(x_arr)-1)
    
    indices = []
    
    if working_index < fixed_index: 
        # Go backwards in time
        direction = -1
    else: 
        # Go forwards in time
        direction = 1
        
    mean = means[-1]
    converged_mean = mean
    converged_mean_ind0 = working_index
    while keep_running:
        #print(mean, direction)
        # Update and store the working index
        working_index += direction*1
        indices.append(working_index)
        
        # Update and store the mean
        if direction == -1:
            mean = get_thresh_mean(x_arr[working_index:fixed_index], mean, eps)
        else:
            mean = get_thresh_mean(x_arr[fixed_index:working_index], mean, eps)
        
        means.append(mean)    
        
        if np.isnan(mean):
            #print(mean)
            break
        
        # Stopping criteria:
        # if the thresholded mean doesn't change upon getting new samples
        # * if the duration is too long and there are sufficient number of samples
        
        if mean != converged_mean:
            converged_mean = mean
            converged_mean_ind = working_index
            converged_mean_ind0 = working_index
        else:
            converged_mean_ind = working_index
        
        time_diff = abs(t_arr[fixed_index]-t_arr[working_index])
        ctime_diff = abs(t_arr[converged_mean_ind0]-t_arr[converged_mean_ind])                                                         
        if ((ctime_diff>1.0) & (mean == converged_mean)): 
            if verbose: print('cdrop', ctime_diff)
            break        
        
        
        # When the mean either converges or stops
        if ((len(indices)>count_thresh) | ((time_diff>0.5) & (len(indices)>5))):  
            #print(time_diff,len(indices))
            nr_events = min(len(indices), count_thresh)
            # see also: bug_check_means(means,nr_events,0.25)
            if check_means(means, configs, nr_events):
                if verbose: print('drop', time_diff)
                break       
                    
                    
        #print(f"{t_arr[working_index]:.3f} {time_diff:.3f} {ctime_diff:.3f}", \
        #      len(indices), fixed_index, working_index, converged_mean_ind0, converged_mean_ind, \
        #      f"\t{mean:.5f} {x_arr[working_index]:.3f} {mean+eps:.5f}",)#,[m == m0 for m in means[-count_thresh:]])            
                    
        keep_running = (working_index > 1) & (working_index < len(x_arr)-1)

    return means, indices, keep_running


def get_thresh_mean(x_sub_arr, mean, eps):
    """
    Get the mean for all events contained within the box (mean+/-eps)
    
    :param x_sub_arr: np.array Subarray of trajectory array of locations
    :param working_index: float Centreline used to define the box    
    :param working_index: float distance threshold used to define the box    

    :return: float New value for mean without the outliers
    """
    
    mask =  get_mask(x_sub_arr, mean, eps)

    return np.mean(x_sub_arr[mask])
        
    
def get_converged(converged, means, indices, configs):    
    """ 
    Create a box around a potential stay

    :param converged: Trajectory array of timepoints
    :param means: [float] List of means
    :param indices: [int Timepoint used to find the box    
    :param configs: dict containing the parameters (currently unused)
    
    :return: float Converged mean of the region
    :return: int Converge index of the region
        
    NOTE: Originally, there were additional conditions to further check for convergence, 
    but these were moved to "extend_edge"
    """
    
    '''
    # If it converged early, get the converged results;
    if converged:  & ((len(indices)>count_thresh) | ((time_diff>time_thresh) & (len(indices)>10))):
        index = min(len(indices), count_thresh)        
        last_mean = means[-index]
        last_index = indices[-index]
    '''
    
    # If it converged early, get the converged results;
    if converged:
        last_mean = means[-1]
        last_index = indices[-1]
    else:
        # else, get the boundary value
        last_mean = means[-1]
        last_index = indices[-1] 
    
    return last_mean, last_index


def make_box(t_arr, x_arr, timepoint, configs, verbose=False):
    """ 
    Create a box around a potential stay

    :param t_arr: np.array Trajectory array of timepoints
    :param x_arr: np.array Trajectory array of locations
    :param timepoint: float Timepoint used to find the box    
    :param configs: dict containing the parameters used to define the box
    :param verbox: bool To select printing metrics to stdio
    
    :return: float Mean (centroid) of the locations in the box, aka the cnetreline of the box
    :return: int Starting index (from t_arr) of the box
    :return: int Ending index of the box
    """
    
    # 0. Configs        
    time_thresh = configs['time_thresh']
    eps = configs['eps']
    count_thresh = configs['count_thresh']
              
    # 1. Initialization
    # 1.1. Init and store the start, end points from the timepoint
    if verbose: print(f"\t1.  {timepoint-time_thresh:.3f} < {timepoint:.3f} < {timepoint+time_thresh:.3f}") 

    start = get_time_ind(t_arr, timepoint, time_thresh, -1)
    end   = get_time_ind(t_arr, timepoint, time_thresh, 1)    

    starts, ends = [], []
    starts.append(start)
    ends.append(end)
    
    # 1.2. Initialize and store the mean for the region
    mean = np.mean(x_arr[start:end])
    means = [mean]
    
    
    # 2. Extend the box in the backwards direction    
    # 2.1. Extension phase
    if verbose: print(f"\t2.   {mean:.4f} {start:4d} {end:4d}")    
    means, indices, keep_running = extend_edge(t_arr, x_arr, start, end, means, configs, verbose)

    # 2.2. Check if NAN --> TODO: check why this happens! 
    if verbose: print(f"\t2.1. {mean:.4f} {start:4d} {end:4d}", keep_running)                     
    if np.isnan(means[-1]):
        if verbose: print(f"\t2.1. \t\tDrop {means[-1]:.4f} {starts[-1]:4d}", end)
        return means[-1], starts[-1], end    
    starts += indices
    
    # 2.3. If it converged early, get the converged results;    
    if verbose: print(f"\t2.2. {means[-1]:.4f} {starts[-1]:4d} {end:4d}")     
    tdiff = t_arr[end]-t_arr[starts[0]]
    mean, start = get_converged(keep_running, means, starts, configs)
    
    # 2.4. Additional check if NAN
    if np.isnan(mean):
        if verbose: print(f"\t2.3. \t\tDrop {means[-1]:.4f} {starts[-1]:4d} {end:4d}")
        return means[-1], starts[-1], end
    
    
    # 3. Extend the box in the forwards direction       
    # 3.1. Extension phase
    if verbose: print(f"\t3.   {mean:.4f} {start:4d} {end:4d}", keep_running)     
    means, indices, keep_running = extend_edge(t_arr, x_arr, end, start, means, configs, verbose)

    # 3.2. Check if NAN --> TODO: check why this happens! 
    if verbose: print(f"\t3.1. {mean:.4f} {start:4d} {end:4d}", keep_running)                     
    if np.isnan(means[-1]):
        if verbose: print(f"\t3.1. \t\tDrop {means[-1]:.4f} {start:4d} {ends[-1]:4d}")
        return means[-1], start, ends[-1]
    ends += indices   
     
    # 3.3. If it converged early, get the converged results
    if verbose: print(f"\t3.1. {means[-1]:.4f} {start:4d} {end:4d}", keep_running) 
    tdiff = t_arr[ends[-1]]-t_arr[start]
    mean, end = get_converged(keep_running, means, ends, configs)

    # 2.4. Additional check if NAN
    if np.isnan(mean):
        if verbose: print(f"\n\t3.3. \t\tDrop {means[-1]:.4f} {start:4d} {ends[-1]:4d}")
        return means[-1], start, ends[-1]
    
    
    # 4
    if verbose: print(f"\t4.   {mean:.4f} {start:4d} {end:4d}") 
        
    return mean, start, end


def get_mask(sub_arr, center, buff, offset=0):
    """
    Get the mask for an array which is contained within a region
    
    :param x_sub_arr: np.array Subarray of trajectory array of locations
    :param center: float Centreline used to define the box  
    :param buff: float Distance from the centerline, _ie_ the boundaries
    :param offset: int Index used to adjust the mask from local (offset=0) to global
    
    :return: np.array mask (global when offset != 0) index of the region 
    """
    
    upper = center + buff
    lower = center - buff
    
    return np.where((sub_arr < upper) & (sub_arr >= lower))[0] + offset
