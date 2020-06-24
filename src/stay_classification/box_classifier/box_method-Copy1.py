import numpy as np
import numpy.ma as ma

tdiff = lambda x : np.concatenate([np.array([0.0]), x[1:]-x[:-1]])

time_thresh = 5/12
eps=0.25
min_samples = 50
    
from stay_classification.checks import check_means


def get_thresh_duration(eps, mean):
    
    upper = mean+eps
    lower = mean-eps
    
    def meth(sub_arr, start):
        
        mask =  np.where((sub_arr < upper) & (sub_arr > lower))[0] + start
    
    
        return mask.min(), mask.max(0)
        
    return meth

def get_thing():
    return 'this'

def asymm_box_method(t_arr,time_thresh,x_loc,eps,timepoint, verbose=False):
    
    count_thresh = 50
        
    box_start = np.where((t_arr < (timepoint)) & (t_arr > (timepoint-time_thresh)))[0].min()
    box_end = np.where((t_arr < (timepoint+time_thresh)) & (t_arr > (timepoint)))[0].max()    

    new_start, new_end = box_start, box_end

    mean = np.mean(x_loc[new_start:new_end])

    tdiffs2 = [t_arr[new_end]-t_arr[new_start]]
    counts2 = [get_counts(eps,mean)(x_loc[new_start:new_end])]

    means = [mean]

    if verbose: print("\t1.", mean, new_start, new_end)            
        
    new_starts = []
    new_ends = []

    new_starts.append(new_start)
    new_ends.append(new_end)

    keep_running = True
    
    while keep_running:

        # Get the current count of events --> OBSOLETE
        count = get_counts(eps,mean)(x_loc[new_start:new_end])

        if (new_start > 1):
            new_start -= 1

        mean = get_thresh_mean(eps,mean)(x_loc[new_start:new_end])
        means.append(mean)    

        new_starts.append(new_start)
        
        #if verbose: print(mean, new_start, new_end)
        # When the mean either converges or stops
        if len(new_starts)>count_thresh:
            if check_means(means,count_thresh):
                break       
                
        keep_running = new_start > 1
        
    # If it converged early, get the converged results;
    if keep_running:
        last_mean = means[-count_thresh]
        last_start = new_starts[-count_thresh]
    else:
        # else, get the boundary value
        last_mean = mean
        last_start = new_start    
    
    new_start = last_start      
    mean = last_mean

    if verbose: print("\t2.", mean, new_start, new_end)            
            
    keep_running = True
    while keep_running:

        # Get the current count of events --> OBSOLETE
        count = get_counts(eps,mean)(x_loc[new_start:new_end])

        # check if the index is within bounds
        if (new_end < len(t_arr)-1):
            new_end += 1

        mean = get_thresh_mean(eps,mean)(x_loc[new_start:new_end])
        means.append(mean)    

        #new_starts.append(new_start)
        new_ends.append(new_end)
        
        #if verbose: print(mean, new_start, new_end)
        # When the mean either converges or stops
        if len(new_ends)>count_thresh:
            if check_means(means,count_thresh):
                break      
                
        keep_running = new_end < len(t_arr)-1
    
    # If it converged early, get the converged results;
    if keep_running:
        last_mean = means[-count_thresh]
        last_end = new_ends[-count_thresh]
    else:
        # else, get the boundary value
        last_mean = mean
        last_end = new_end      
            
    if verbose: print("\t3.", last_mean,last_start,last_end)                
    return last_mean,last_start,last_end

def extend_box(x_loc, working_index, fixed_index, means, count_thresh = 50):

    keep_running = True
    
    indices = []
    
    if working_index < fixed_index: 
        # Go backwards in time
        direction = -1
    else: 
        # Go forwards in time
        direction = 1
        
    mean = means[-1]
    while keep_running:

        # Update and store the working index
        working_index += direction*1
        indices.append(working_index)
        
        # Update and store the mean
        if direction == -1:
            mean = get_thresh_mean(eps,mean)(x_loc[working_index:fixed_index])
        else:
            mean = get_thresh_mean(eps,mean)(x_loc[fixed_index:working_index])
        means.append(mean)    
        
        # When the mean either converges or stops
        if len(indices)>count_thresh:
            if check_means(means,count_thresh):
                break       
                    
        keep_running = (working_index > 1) & (working_index < len(x_loc)-1)
        
    return means, indices, keep_running


def get_counts(eps, mean):
    
    upper = mean+eps
    lower = mean-eps
    
    def meth(sub_arr):
        
        return np.where((sub_arr < upper) & (sub_arr > lower))[0].size
    
    return meth


def get_thresh_mean(eps, mean):
    
    upper = mean+eps
    lower = mean-eps
    
    def meth(sub_arr):
        
        mask =  np.where((sub_arr < upper) & (sub_arr > lower))
    
        return np.mean(sub_arr[mask])
        
    return meth
    

def get_converged(converged, means, indices, count_thresh):
    
    # If it converged early, get the converged results;
    if converged & (len(indices)>count_thresh):
        last_mean = means[-count_thresh]
        last_index = indices[-count_thresh]
    else:
        # else, get the boundary value
        last_mean = means[-1]
        last_index = indices[-1] 
    
    return last_mean, last_index

def asymm_box_method_modular(t_arr, time_thresh, x_loc, eps, timepoint, count_thresh = 50, verbose=False):
        
    # 1.
    # Initialize and store the start, end points from the timepoint
    start = np.where((t_arr < (timepoint)) & \
                     (t_arr > (timepoint-time_thresh)))[0].min()
    end = np.where((t_arr < (timepoint+time_thresh)) & \
                   (t_arr > (timepoint)))[0].max()    

    starts = []
    ends = []

    starts.append(start)
    ends.append(end)
    
    # Initialize and store the mean for the region
    mean = np.mean(x_loc[start:end])
    means = [mean]
    
    
    # 2.
    if verbose: print("\t1.", mean, start, end)
    
    # Extend the box in the backwards direction
    means, indices, keep_running = extend_box(x_loc, start, end, means)
    starts += indices
    
    if verbose: print("\n\t2.", means[-1], starts[-1], end) 
    
    # If it converged early, get the converged results;
    mean, start = get_converged(keep_running, means, starts, count_thresh)

    #Debuging
    if verbose: print("\t2.1.", mean, start, end, keep_running) 


                
    # 3. 
    if verbose: print("\n\t3.", mean, start, end)
    
    # Extend the box in the forwards direction    
    means, indices, keep_running = extend_box(x_loc, end, start, means)
    ends += indices
            
    # If it converged early, get the converged results;
    mean, end = get_converged(keep_running, means, ends, count_thresh)
     
    #Debugging
    if verbose: print("\t3.1.", means[-1], start, end, keep_running) 
    
    # 4
    if verbose: print("\n\t4.", mean, start, end) 
        
    return mean, start, end

def get_slope(t_subarr, x_subarr):
    
    ub_xdata = t_subarr - t_subarr.mean()
    ub_ydata = x_subarr - x_subarr.mean()
    
    return (ub_xdata.T.dot(ub_ydata))/(ub_xdata.T.dot(ub_xdata))