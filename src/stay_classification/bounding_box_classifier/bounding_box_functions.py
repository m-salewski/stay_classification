import numpy as np

import matplotlib.pyplot as plt

def bbox(t_arr, x_arr):
    return np.array([t_arr.min(), t_arr.max(), x_arr.min(), x_arr.max()])


def plot_box(bbox, ax=None, plot_dict=None):
    
    if plot_dict == None:
        plot_dict = {"linestyle":'--',
                     "dashes":[4,2,],
                    "color":"grey", 
                    "linewidth":2}
    
    ms = [0,0,1,1,0]
    ns = [2,3,3,2,2]
    
    ts,xs = [],[]
    
    for i in range(5):        
        ts.append(bbox[ms[i]])
        xs.append(bbox[ns[i]])
    
    if ax == None:
        ax = plt.plot(ts,xs, **plot_dict)
        return ax
    else:
        ax.plot(ts,xs, **plot_dict)
        return None