import numpy as np

import warnings

'''
TODO 

**also** chekc the TODOs in the script
0. break this into smaller subscripts
    * trajectory creation
    * masking
    * noise 

0. rename 'journey' with 'trajectory'
    * journey implies travel
    
0. since slopes are determined by stays, maybe remove this field
    * specify the slope of the travel
        * the location of the travel (and stays) will be sorted to minimize the slope, cutting the adjacent stays as needed

0. add Assert so that always [stay, trav, ..., trav, stay]     

1. add some documentation
2. keep the segment indices
    * use later for seg. dept. noise, also training
3. Update the noise enrichment
    * segment-dependent noise
    * configurable noise distributions for each segment
4. include asserts to ensure no overlaps of stay regions
5. Put all into class
    * x, y, noisy y
    * segment idices, features
    * various returns: np.arrays, pd.DataFrames
    
6. segments' endpoints to coincide with the sparse stays
    * some stays are shorter after masking
7. improve the duplication for the data
    * include some specific $x$-locations which are duplicated in certain segments
        * this is like a tower which is pinged multiple times but only gives it's location
    * include some specific $\Delta x$'s which are duplicated showing an effecitve radius when triangulation fails
    * segement-/location-specific noise
        * try also with the array of weights passed in `np.random.choice`
            * changes the probab. of picking up specific events in the full array
    
'''

"""
Examples of stays
# Go to work, no lunch
stays = [
        get_stay(  0,  20,  2),
        get_stay( 30,  70, -1),
        get_stay( 80, 100,  2)
    ]
     
# Go to work with a lunch break
stays = [
        get_stay(  0,  20,  2),
        get_stay( 30,  55, -1),
        get_stay( 60,  65,  0.5),
        get_stay( 70,  75,  -1),
        get_stay( 80, 100,  2)
    ]


# Work, gym, shop: stay1.T > stay2.T > stay3.T
stays = [
        get_stay(  0,  20,  2),
        get_stay( 30,  55, -1),
        get_stay( 60,  65,  0.5),
        get_stay( 70,  75,  2.5),
        get_stay( 80, 100,  2)
    ]
"""

# Stay and travels included as lists
#### TODO: all but get_stay should be lambdas
def get_stay(start,stop,loc,slope=0):     
    """ 
    Assigns values to attribute as key/value-pairs in a dict.

    :param start: [float] time as beginning of stay
    :param stop:  [float] time at end of stay
    :param loc:   [float] location of stay
    :param slope: [float] unused in a stay; retained here for generality
    
    :return: [dict] A dict of attribute-value pairs for the stays
    """
    
    return {"type": "stay", "loc":  loc, "start": start, "end":  stop, "slope":  slope}

get_trav = lambda start,stop,loc,slope:   {"type": "trav", "loc":  loc, "start": start, "end":  stop, "slope":  slope}

get_seg  = lambda start,stop,loc,slope=0: get_stay(start,stop,loc,slope) if slope == 0 else get_trav(start,stop,loc,slope)
get_seg_info = lambda seg: (seg['type'], seg['loc'], seg['start'], seg['end'], seg['slope'])



def check_stay(stay):
    if stay['type'] == 'stay':
        return True
    else:
        raise ValueError("\'type\' is not \'stay\'")

get_stay_info = lambda stay: get_seg_info(stay)[1:] if check_stay(stay) else None
#get_stay_info = lambda stay: (stay['loc'], stay['start'], stay['end'], stay['slope'])


rand_range = lambda low, high, size: (high-low)*np.random.rand(size) + low

#### TODO: move to another module
def list_interleave(a, b):
    c = (len(a+b))*[None]
    c[0::2] = a
    c[1::2] = b
    return c


#### TODO: move to another module
def arr_interleave(a, b):
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c


def get_travels(x, stay_list, threshold=0.5):
    """
    Generate travels from stays (in between each stay)
    """
    travels = []
    
    get_slope = lambda x1,y1,x2,y2: (y1-y2)/(x1-x2)    
    
    for stay1, stay2 in list(zip(stay_list[:-1],stay_list[1:])):

        # Get the quantities
        _, loc1, start1, stop1, _ = get_seg_info(stay1) 
        _, loc2, start2, stop2, _ = get_seg_info(stay2) 
        
        # Adjust the stoppoints as needed
        start1 = max(start1,x[0])
        start2 = max(start2,x[0])
        stop1 = min(stop1,x[-1])
        stop2 = min(stop2,x[-1])
        
        
        # Get the indices; check if overlapping
        #### TODO: put in some asserts (do this earlier!)
        if start1 < stop1:
            stop1_ind = np.where((x <stop1))[0][-1]
        else: 
            stop1_ind = np.where((x==stop1))[0][ 0]        
        
        # Get the indices
        if start2 < stop2:
            start2_ind = np.where((x>=start2))[0][0]
        else: 
            start2_ind = np.where((x==stop2))[0][ 0]
        
        if np.abs(loc1-loc2) < threshold:
            warnings.warn(f"the distance between the consecutive locations is within the threshold {threshold}")
        
        # Get the slope
        #### TODO: include assert to ensure no adjacent (or overlapping) stays with div0 error
        slope = get_slope(stop1,loc1,start2,loc2)
    
        # Add to the slopes
        travels.append(get_trav(x[stop1_ind], x[start2_ind], loc1, slope))
        
    return travels #fff


def get_seg_mask(x, start, stop):
    
    x_buff = np.mean(np.abs(x[:-1]-x[1:]))/2.0
    
    # Find the associated indices
    if start < stop:    
        start_ind = np.where((x>=start))[0][0]
        stop_ind = np.where((x<stop))[0][0]
        mask = np.where((x>=start) & (x<stop))        

    elif start == stop: 
        # For single point stays (in beginning/stop)

        # In case these are beyond the upper limit
        if start <= x[0]:

            start = max(start,x[0])
            stop = max(stop,x[0])

        elif start >= x[-1]:
            start = min(start,x[-1])
            stop = min(stop,x[-1])
        else:
            pass

        # Include a buffer for single points
        mask = np.where((x>=start-x_buff) & (x<stop+x_buff))

    else:
        # otherwise, the start and stop overlap
        raise AssertionError('start is greater than stop point')
      
    return mask
  

def get_journey_path(x, seg_list):
    
    fff = np.zeros(x.size)
    
    x_buff = np.mean(np.abs(x[:-1]-x[1:]))/2.0
    
    for seg in seg_list:

        #### NOTE
        # keep all here since mask indices are scope-dependent
        
        # Get the segment details
        _, loc, start, stop, slope = get_seg_info(seg)   
            
        # Find the associated indices
        if start < stop:    
            start_ind = np.where((x>=start))[0][0]
            stop_ind = np.where((x<stop))[0][0]
            mask = np.where((x>=start) & (x<stop))        
        
        elif start == stop: 
            # For single point stays (in beginning/stop)
            
            # In case these are beyond the upper limit
            if start <= x[0]:
                
                start = max(start,x[0])
                stop = max(stop,x[0])
                
            elif start >= x[-1]:
                start = min(start,x[-1])
                stop = min(stop,x[-1])
            else:
                pass
            
            # Include a buffer for single points
            mask = np.where((x>=start-x_buff) & (x<stop+x_buff))
            
        else:
            # otherwise, the start and stop overlap
            raise AssertionError('start is greater than stop point')
        
        # Create the line semgnet
        if slope == 0:
            fff[mask] = loc

        else:
            fff[mask] = slope*(x[mask]-start) + loc
    return fff


def get_segments(x, stays, threshold=0.5):
    
    segments = list_interleave(stays, get_travels(x, stays, threshold))
    
    return segments


def get_stay_paths(x, seg_list):   
    '''
    #### NOTE: this may be irrelevant
    '''

    fff = np.zeros(x.size)
    
    for seg in seg_list:

        
        _, loc, start, stop, _ = get_seg_info(seg)  
        
        if start < stop:
            start_ind = np.where((x>=start))[0][0]
            stop_ind = np.where((x<stop))[0][0]
        
            mask = np.where((x>=start) & (x<stop))
        else:
            mask = np.where((x==start))
        
        fff[mask] = loc        
        
    return fff


def get_stay_segs(stay_list):
    
    stay_segs_t, stay_segs_x = [], []
    
    for stay in stay_list:
        
        _, loc, start, stop, _ = get_seg_info(stay) 
        
        stay_segs_x += [loc, loc, None]
        stay_segs_t += [start, stop,None]

    return np.array(stay_segs_t), np.array(stay_segs_x) 

def get_travel_paths(x, y, seg_list, threshold=0.5):
    #### NOTE: this may be irrelevant
    fff = y.copy()
    
    for seg in seg_list:

        
        _, loc, start, stop, slope = get_seg_info(seg)   
        
        print(seg)
        
        if start < stop:
            start_ind = np.where((x>=start))[0][0]
            stop_ind = np.where((x<stop))[0][0]
        
            mask = np.where((x>=start) & (x<stop))
        else:
            mask = np.where((x==start))
            
        
        fff[mask] = slope*(x[mask]-stop) + loc
    
    return fff


def get_adjusted_stays(segs, time_suba):
    """
    Adjust the stay boundaries after the masking, as there is a reduction in the number of events

    :param segs: [list(dict)] segment dictionary
    :param time_suba:  [np.array] reduced time-array (after masking)
    
    :return: [list(dict)] List of new stays
    """
    
    new_stays = []
    
    for seg in segs:

        type_, loc_, start_, stop_ ,_ = get_seg_info(seg) 

        ####TODO: generalize to any seg, since the travels are also affected.
        if  type_ == 'stay':
            subarr = time_suba[np.where((time_suba >= start_) & \
                                        (time_suba <= stop_))]
            
            new_t0, new_t1 = np.min(subarr),np.max(subarr)        
            
            new_stays.append(get_stay(new_t0,new_t1,loc_))
        
    return new_stays


def get_stay_indices(segs, time_subarr):
    """
    Get the indices based on the segments

    :param segs: [list(dict)] segment dictionary
    :param time_subarr:  [np.array] reduced time-array (after masking)
    
    :return: [list(dict)] List of new stays
    """
    
    stay_indices = []
    
    for seg in segs:

        type_, loc_, start_, stop_ ,_ = get_seg_info(seg) 

        ####TODO: generalize to any seg, since the travels are also affected.
        if  type_ == 'stay':
            subarr = np.where((time_subarr >= start_) & \
                              (time_subarr <= stop_))[0]

            new_t0, new_t1 = np.min(subarr),np.max(subarr)        

            stay_indices.append((new_t0,new_t1))
        
    return stay_indices

#---
#### NOTE: Non functoinal
# Meant to smoothen out the slopes
def slope_func(xmean):
    def inner (x,a,b,c): 
        return a*np.tanh(b*(x-xmean)) + c
    return inner


def smooth_slope(x,y):

    from scipy.optimize import curve_fit

    xmean = x.mean()
    
    popt, pcov = curve_fit(slope_func(xmean), x, y)
    
    return slope_func(xmean)(x, *popt)
    
 
