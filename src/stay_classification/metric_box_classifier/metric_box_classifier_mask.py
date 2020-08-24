import numpy as np


def _get_iqr(data):
    
    q25 = np.quantile(data, 0.25, interpolation='lower')
    q75 = np.quantile(data, 0.75, interpolation='higher')
    return q25, q75


def get_iqr(data):
    
    q25, q75 = _get_iqr(data)
    
    return abs(q75 - q25)


def iqr_metrics(data, iqr_fact=1.5):
    
    q25, q75 = _get_iqr(data)
    
    iqr = abs(q75 - q25)

    iqr_boost = iqr*iqr_fact
    
    full_range = q75 - q25 + 2*iqr_boost
    min_range = ys[np.where(ys < q75+iqr_boost )].max() - ys[np.where(ys > q25-iqr_boost )].min()
    
    return full_range, min_range


def get_iqr_mask(sub_arr, index, iqr_fact = 1.5, within=True):
    """
    """
    
    # Mask to include only events within the IQR
    q25, q75 = _get_iqr(sub_arr)
    
    iqr = abs(q75 - q25)
        
    if within:
        mask = np.where((sub_arr > (q25 - iqr_fact * iqr)) & (sub_arr < (q75 + iqr_fact * iqr)))
        
    else:
        mask =  np.where((sub_arr <= (q25 - iqr_fact * iqr)) | (sub_arr >= (q75 + iqr_fact * iqr)))
    
    mask[0][:] += index
    
    return mask


def get_iqr_filtered_clusters(x_arr, clusts, iqr_fact=1.5, verb=False):
    """
    """
    
    new_clusters = []
    
    for c in clusts:
        cc = get_iqr_mask(x_arr[c], c[0], iqr_fact, True)[0]
        
        ccc = list(range(cc[0], cc[-1]+1))
        
        if verb: print(f"[{c[0]:5d},{c[-1]:5d}] vs. [{cc[0]:5d},{cc[-1]:5d}]")
        #f" vs. [{ccc[0]:5d},{ccc[-1]:5d}]" )

        new_clusters.append(ccc)
        
    return new_clusters
