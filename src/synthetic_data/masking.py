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

# Masking to sparsify the signal
#### TODO: make the sparsing location/segment dependent

def get_frac_mask(size, frac, verbose=False):
    
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


def get_mask_with_duplicates(time_arr, target_frac=1.0, target_dupl_frac=0.0, verbose=False):
    
    """ Return a (sub)array with/out duplicates
    
    Get a fraction of time of the time array, where a fraction of it contains duplicates. 
    The duplicate fraction refers to the fraction of duplicates in the final array.
    
    Args:    
        time_arr      (np.array): time points in hours
        target_frac      (float): the fraction of input array to be output as a mask
        target_dupl_frac (float): the fraction of the output events to be duplicated
    
    Returns: 
        np.array: mask to be applied to the time_arr (includes duplicates

    Raises:
        ValueError: If `target_dupl_frac` is too large compared to `target_frac`.

    Examples:
        >>> t_arr.size
        1000
        >>> mask = get_mask_with_duplicates(t_arr, 0.9, 0.1)
        >>> mask
        array([   32,    32,    89, ..., 960, 971, 998])
        >>> mask.size
        900
        >>> np.unique(mask).size
        810
    """

    get_frac_outer = lambda size: lambda frac: int(frac*size)

    get_duplicates_counts = lambda arr: (arr.size, len(set(arr)), arr.size-len(set(arr)))

    from collections import Counter    
    
    # Compute the adjusted final fraction when duplicates are present
    adjusted_frac = (1.0-target_dupl_frac)*target_frac
    dupli_frac = (target_dupl_frac)*target_frac

    base_frac_int = get_frac_outer(time_arr.size)(adjusted_frac)
    dupl_frac_int = get_frac_outer(time_arr.size)(dupli_frac)
    dupl_frac_int = min(dupl_frac_int,base_frac_int)
    
    # Get the unique subset of time points
    time_arr_sub0 = np.random.choice(time_arr, base_frac_int, replace=False)

    # Get the indices of the time points
    mask_ = np.where(np.in1d(time_arr, time_arr_sub0))[0]

    if dupl_frac_int > 0:
        
        # The set of unique duplicates: will always drw from this set
        mask_dups = np.random.choice(mask_, dupl_frac_int, replace=False)

        iterations = 8
        for n in range(iterations):

            # Get a subsample from the duplicates
            # 1. The fraction controls the mulitplicity of duplicates
            base_subsamp_frac = 0.05
            mask_dups_sub = np.random.choice(mask_dups, get_frac_outer(dupl_frac_int)(base_subsamp_frac), \
                                             replace=True, ) 
            # 2. add back to the duplicates --> keep all events; just increase their frequencies
            mask_dups = np.concatenate((mask_dups, mask_dups_sub))

        # Get the final set of the duplicates
        mask_dups = np.random.choice(mask_dups, dupl_frac_int, replace=True, )
        
        if verbose:
            # Check the frequencies of the duplicates 
            # 1. for the duplicates, count the frequency for each duplicate, ie 1 appears 3x, 2, appears 1x, etc. 
            freqs = Counter(mask_dups.tolist())
            print(sum(freqs.values()))

            # 2. for the frequencies, count the frequency of a given frequency, ie. how many 1's, 2's, etc.
            freqs = Counter(list(freqs.values()))
            print(freqs)    
            print('freq',sum(freqs.values()), dupl_frac_int, np.unique(mask_dups).size)
            print()

        # Add the duplicate mask back to the original mask
        mask_ = np.concatenate((mask_, mask_dups))
        
        if verbose:
            totals, uniques, duplicates = get_duplicates_counts(mask_)
            print(totals, uniques, duplicates, round(100.*duplicates/totals,2))    
    
    mask_.sort()
    mask_ = mask_.astype(int)
    
    return mask_