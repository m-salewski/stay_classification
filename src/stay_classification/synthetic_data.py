import numpy as np

'''
TODO 
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
get_stay = lambda start,stop,loc,slope=0: {"type": "stay", "loc":  loc, "start": start, "end":  stop, "slope":  slope}

get_trav = lambda start,stop,loc,slope:   {"type": "trav", "loc":  loc, "start": start, "end":  stop, "slope":  slope}

get_seg  = lambda start,stop,loc,slope=0: get_stay(start,stop,loc,slope) if slope == 0 else get_trav(start,stop,loc,slope)

get_stay_info =  lambda stay: (stay['loc'], stay['start'], stay['end'], stay['slope'])


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
        loc1, start1, stop1, _ = get_stay_info(stay1) 
        loc2, start2, stop2, _ = get_stay_info(stay2) 
        
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


def get_journey_path(x, seg_list):
    
    fff = np.zeros(x.size)
    
    x_buff = np.mean(np.abs(x[:-1]-x[1:]))/2.0
    
    for seg in seg_list:

        #### NOTE
        # keep all here since mask indices are scope-depstopent
        
        # Get the segment details
        loc, start, stop, slope = get_stay_info(seg)   
            
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
    
    segments = linterleave(stays, get_travels(x, stays, threshold))
    
    return segments


def get_stay_paths(x, seg_list):   
    '''
    #### NOTE: this may be irrelevant
    '''

    fff = np.zeros(x.size)
    
    for seg in seg_list:

        
        loc, start, stop, _ = get_stay_info(seg)  
        
        if start < stop:
            start_ind = np.where((x>=start))[0][0]
            stop_ind = np.where((x<stop))[0][0]
        
            mask = np.where((x>=start) & (x<stop))
        else:
            mask = np.where((x==start))
        
        fff[mask] = loc        
        
    return fff


def get_travel_paths(x, y, seg_list, threshold=0.5):
    #### NOTE: this may be irrelevant
    fff = y.copy()
    
    for seg in seg_list:

        
        loc, start, stop, slope = get_stay_info(seg)   
        
        print(seg)
        
        if start < stop:
            start_ind = np.where((x>=start))[0][0]
            stop_ind = np.where((x<stop))[0][0]
        
            mask = np.where((x>=start) & (x<stop))
        else:
            mask = np.where((x==start))
            
        
        fff[mask] = slope*(x[mask]-stop) + loc
    
    return fff


def get_noisy_bumps(x, **kwargs):
    
    remask = lambda x: int(not bool(x))
    
    #print(len(kwargs)%5)
    assert len(kwargs)%5 == 0, "Number of kwargs is wrong!"
    
    kwargs_len = int(len(kwargs)/5)

    fff = np.zeros(x.size)
    
    for nn in range(kwargs_len):
        n=nn+1
        #print(n)
        amp   = kwargs[f'bump{n}_amp']
        slope = kwargs[f'bump{n}_slope']
        start = kwargs[f'bump{n}_start']
        end   = kwargs[f'bump{n}_end']    
        eta   = kwargs[f'bump{n}_eta']  
    
        # The tanh-bump
        ggg = amp*np.tanh( 1*slope*(x-start)) \
            + amp*np.tanh(-1*slope*(x-end))   
        
        # Create the mask to target only the bumps
        mask = ggg>0.001
        mask = np.array(list(map(remask, mask)))        
        
        noise = np.random.normal(loc=0.0, scale=eta, size=ggg.size)        
        if nn != 0:
            noise = noise*mask
        
        # Add the noisy bumps
        fff += ggg + noise
        
    return fff


def get_noise_event(y):
    
    act0=0.050
    act1=0.025
    act2=0.085
    trp0=0.035
    trp1=0.030
    trp2=0.080

    if abs(y) < 0.1:
        eta = np.random.normal(loc=0.0, scale=act0, size=1)
        
    elif (y >= 0.91) & (y < 1.01):
        eta = np.random.normal(loc=0.0, scale=act1, size=1)   
    
    elif (y >= -2.01) & (y < -1.81):
        eta = np.random.normal(loc=0.0, scale=act2, size=1)
        
    else:
        eta = np.random.normal(loc=0.0, scale=trp2, size=1)
    
    return y + eta


def get_noise(yyy):
    yyy = yyy.copy()
    for n,yy in enumerate(yyy):
        #print(n)
        yyy[n] = get_noise_event(yy)

    return yyy    


# Masking to sparsify the signal
#### TODO: make the sparsing location/segment dependent

def get_frac_mask(size, frac, verbose):
    
    int_frac = int(frac*size) 
    
    # Get the fraction of "on"s
    out_arr_1s = np.ones(int_frac)

    # Get the remaining fraction of "off"s
    out_arr_0s = np.zeros(size-int_frac)

    # Concat and shuffle
    out_arr = np.concatenate([out_arr_0s,out_arr_1s])
    np.random.shuffle(out_arr)    
    
    if verbose: print(np.sum(out_arr)/size)
    
    return out_arr

def get_mask_indices(mask):
    
    mask_indices = (mask == 1)
    
    return mask_indices

def get_mask(size, frac, verbose=False):
    
    return get_mask_indices(get_frac_mask(size, frac, verbose))

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
    
 
