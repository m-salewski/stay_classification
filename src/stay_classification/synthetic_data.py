import numpy as np

def get_smooth_bumps(x, **kwargs):
    fff = x.copy()
    return fff

def get_bump_dict(dc):
    
    keys = []
    
    for k,v in dc.items():
        
        keys.append((k.split('_')[0].replace('bump','')))
    
    keys = list(set(keys))
    
    return keys

def get_bumps(x, **kwargs):
    
    '''
    TODO: check if the dict is complete, also with some defaults
    if (len(kwargs)%4 != 0) & :
        assert len(kwargs)%4 == 0, "Number of kwargs is wrong!"
    '''
    
    kwargs_len = int(len(kwargs)/4)

    fff = np.zeros(x.size)
    
    for n in range(kwargs_len):

        amp   = kwargs[f'bump{n}_amp']
        slope = kwargs[f'bump{n}_slope']        
        start = kwargs[f'bump{n}_start']
        end   = kwargs[f'bump{n}_end']    
        
        slope_out_key = f'bump{n}_slope_out'
        if slope_out_key in kwargs.keys():
            slope_out = kwargs[slope_out_key]
        else:
            slope_out = -kwargs[f'bump{n}_slope']
        
        start_ind = np.where((x>=start))[0][0]

        end_ind = np.where((x<end))[0][0]

        mask = np.where((x>=start) & (x<end))
        
        fff[mask] = amp        
        
    return fff

def get_sloped_bumps(x, yyy, **kwargs):
    
    yyy = yyy.copy()
    
    get_x0 = lambda m,x,y,y0: x-((y-y0)/m)
    
    kwargs_len = int(len(kwargs)/4)
    print(kwargs_len)
    for n in range(kwargs_len):

        amp   = kwargs[f'bump{n}_amp']
        slope = kwargs[f'bump{n}_slope']        
        start = kwargs[f'bump{n}_start']
        end   = kwargs[f'bump{n}_end']    
        
        slope_out_key = f'bump{n}_slope_out'
        if slope_out_key in kwargs.keys():
            slope_out = kwargs[slope_out_key]
        else:
            slope_out = -kwargs[f'bump{n}_slope']
        '''
        start_ind = np.where((x>=start))[0][0]
        end_ind = np.where((x<end))[0][0]
        mask = np.where((x>=start) & (x<end))
        '''
        
        start_ind = np.where((x>=start))[0][0]
        end_ind = np.where((x<end))[0][-1]
        #mask = np.where((x>=start) & (x<end))       
        '''
        # Get the pre- & post-step indices
        start_ind = mask[0][0]
        end_ind = mask[0][-1]
        '''
        print(n,start, end,start_ind, end_ind)
        # Get the midpoint values within the step
        ymid_start = np.mean(yyy[start_ind-1:start_ind+1])
        xmid_start = np.mean(x[start_ind-1:start_ind+1])

        # Get the pre- & post-step amplitudes
        y0_start_end = yyy[start_ind:min(start_ind+1,x.size)][-1]
        
        ind_0 = max(start_ind-1,0)
        if ind_0 != 0: 
            ind_1 = start_ind
            if ind_0 == ind_1: ind_1 += 1
            y0_start_start = yyy[ind_0:ind_1][0]

            # Get the x-locations for the start and end of the sloped region
            x0_start_end = get_x0(slope, xmid_start, ymid_start, y0_start_end)
            x0_start_start = get_x0(slope, xmid_start, ymid_start, y0_start_start)
            #print(n,start, end, slope, x0_start_end, x0_start_end)
            slope_mask = np.where((x>=x0_start_start) & (x<x0_start_end))

            # Compute the sloped region
            yyy[slope_mask] = slope*(x[slope_mask]-x0_start_start) + y0_start_start
            
        # Get the pre- & post-step amplitudes
        y0_end_end = yyy[end_ind:end_ind+2][-1]
        y0_end_start = yyy[end_ind-1:end_ind][-1]
        
        # Get the midpoint values within the step
        ymid_end = np.mean(yyy[end_ind:end_ind+2])
        xmid_end = np.mean(x[end_ind-1:end_ind+1])
        
        # Get the x-locations for the start and end of the sloped region
        return_slope = -slope
        x0_end_start = get_x0(return_slope, xmid_end, ymid_end, y0_end_start)
        x0_end_end   = get_x0(return_slope, xmid_end, ymid_end, y0_end_end)
        slope_mask = np.where((x>=x0_end_start) & (x<x0_end_end))

        # Compute the sloped region
        yyy[slope_mask] = return_slope*(x[slope_mask]-x0_end_start) + y0_end_start 
        
    return yyy
"""
# Go to work, no lunch
dc = {
        "bump1_amp":  0.5, "bump1_slope": 0.5, "bump1_start": 20 , "bump1_end": 80
     }
     
# Go to work with a lunch break
dc = {
        "bump1_amp":  0.5, "bump1_slope": 0.5, "bump1_start": 20 , "bump1_end": 80,\
        "bump2_amp":  0.25, "bump2_slope": 0.95, "bump2_start": 40 , "bump2_end": 50
     }

# Work, gym, shop: A1.T > A2.T > A3.T
dc = {
        "bump1_amp":  0.5, "bump1_slope": 0.5, "bump1_start": 20 , "bump1_end": 60,\
        "bump2_amp": -1.0, "bump2_slope": 0.75, "bump2_start": 60 , "bump2_end": 80,\
        "bump3_amp": -1.5, "bump3_slope": 0.95, "bump3_start": 80 , "bump3_end": 90  
     }
"""

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

"""
# Go to work, no lunch
dc = {
        "bump1_amp":  0.5, "bump1_slope": 0.5, "bump1_start": 20 , "bump1_end": 80, "bump1_eta": 0.015  
     }
     
# Go to work with a lunch break
dc = {
        "bump1_amp":  0.5, "bump1_slope": 0.5, "bump1_start": 20 , "bump1_end": 80, "bump1_eta": 0.015,
        "bump2_amp": -1.5, "bump2_slope": 0.95, "bump2_start": 40 , "bump2_end": 50, "bump2_eta": 0.25    
     }


# WOrk, gym, shop: A1.T > A2.T > A3.T
dc = {
        "bump1_amp":  0.5, "bump1_slope": 0.5, "bump1_start": 20 , "bump1_end": 60,\
        "bump2_amp": -1.0, "bump2_slope": 0.75, "bump2_start": 60 , "bump2_end": 80,\
        "bump3_amp": -1.5, "bump3_slope": 0.95, "bump3_start": 80 , "bump3_end": 90  
     }
"""

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

# Masking

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
