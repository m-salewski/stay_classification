import numpy as np

import warnings

from synthetic_data.trajectory import get_stay_paths, get_seg_mask

'''
    
def get_stay_paths(x, seg_list):   
    

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
'''

"""
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
    
"""
def get_seg_noise(seg,noise):
    seg['noise'] = noise
    return seg

get_noisy_seg_info = lambda seg: (seg['noise'], seg['loc'], seg['start'], seg['end'])

get_noisy_segs = lambda segments, noises: [get_seg_noise(seg,noises[n]) for n, seg in enumerate(segments)]

get_noise_arr = lambda mn, mx, size: (mx - mn)*np.random.random_sample(size) + mn

get_add_noise = lambda eta: lambda y1: y1 + np.random.normal(loc=0.0, scale=eta, size=1)

def get_noisy_path(x, y, seg_list):
        
    y = y.copy()
    
    for seg in seg_list:

        #### NOTE
        # keep all here since mask indices are scope-depstopent
        
        # Get the segment details
        noise, loc, start, stop = get_noisy_seg_info(seg)   
            
        # Find the associated indices
        mask = get_seg_mask(x,start,stop)
        
        y[mask] = map_array(get_add_noise(noise))(y[mask])
                
    return y



def get_noisy_bumps(x, **kwargs):

    
    for nn in range(kwargs_len):
        n=nn+1

    
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
    """Under construction!!!"""    
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


# np.array function
map_array = lambda f: lambda x: np.fromiter((f(xi) for xi in x), x.dtype, count=len(x))

def get_noise_(yyy):
    yyy = yyy.copy()
    for n,yy in enumerate(yyy):
        #print(n)
        yyy[n] = get_noise_event(yy)

    return yyy    


def get_noise(yyy):
    
    yyy = yyy.copy()
    
    yyy = map_array(get_noise_event)(yyy)

    return yyy   


def get_radial_noise(x, loc, rad, sig):

    x = x.copy()
    
    rand_range = lambda h, sig, l : ((sig+h)-h)*np.random.random() + l

    rand_radius = lambda loc, rad, sig: ((-1)**np.random.randint(0,2,1)[0] )*rand_range(rad, sig, loc)

    x = map_array(lambda y: rand_radius(y, 0.5, 0.01), x)
    
    return x
