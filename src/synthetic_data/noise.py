import numpy as np

import warnings

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

def get_noisy_bumps(x, **kwargs):
    """Obsolete"""
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


def get_noise(yyy):
    yyy = yyy.copy()
    for n,yy in enumerate(yyy):
        #print(n)
        yyy[n] = get_noise_event(yy)

    return yyy    

