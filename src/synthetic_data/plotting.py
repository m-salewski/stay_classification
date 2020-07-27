import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

segs_plot_kwargs = {
    'linestyle':'--', 
    'marker':'o', 
    'color':'k', 
    'linewidth':4.0, 
    'markerfacecolor':'w', 
    'markersize':6.0, 
    'markeredgewidth':2.0
    }

'''
def plot_trajectory(t_arr, x_arr, nx_arr, t_segs, x_segs, eps):

    plt.figure(figsize=(20,5))

    plt.plot(t_segs, x_segs, **segs_plot_kwargs, label='adjusted raw stays')
    plt.plot(t_arr, x_arr, ':', label='raw journey')
    plt.plot(t_arr, nx_arr, '.-', label='noisy journey', alpha=0.25)

    plt.legend();

    plt.xlabel(r'time, $t$ [arb.]')
    plt.ylabel(r'position, $x$ [arb.]')

    ymin = nx_arr.min()-1*eps
    ymax = nx_arr.max()+1*eps

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))

    ax.set_xlim(-0.05, 24.05)

    ax.set_title('Trajectory', fontsize=24)
    ax.grid(visible=True); 

    return ax
'''

def plot_trajectory(t_arr, x_arr, nx_arr, t_segs, x_segs, eps):
    

    plt.figure(figsize=(20,5))

    if (t_segs is not None) & (x_segs is not None):
        plt.plot(t_segs, x_segs, **segs_plot_kwargs, label='adjusted raw stays')
        ymin = min([x for x in x_segs.tolist() if x != None])-1*eps
        ymax = max([x for x in x_segs.tolist() if x != None])+1*eps         
    
    if (t_arr is not None) & (x_arr is not None):
        plt.plot(t_arr, x_arr, ':', color='C0', label='raw journey')
        ymin = x_arr.min()-1*eps
        ymax = x_arr.max()+1*eps
    
    if (t_arr is not None) & (nx_arr is not None):
        plt.plot(t_arr, nx_arr, '.-', color='C1', label='noisy journey', alpha=0.25)
        ymin = nx_arr.min()-1*eps
        ymax = nx_arr.max()+1*eps        

    plt.legend();

    plt.xlabel(r'time, $t$ [arb.]')
    plt.ylabel(r'position, $x$ [arb.]')



    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))

    ax.set_xlim(-0.05, 24.05)

    ax.set_title('Trajectory', fontsize=24)
    ax.grid(visible=True); 

    return ax


def add_plot_seg_boxes(t_segs, x_segs, dist_thresh, ax):
    
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    for n in range(0,len(t_segs),3):

        begin = t_segs[n]
        end = t_segs[n+1]

        loc = x_segs[n]

        rect = Rectangle((begin, loc-dist_thresh), end-begin, 2*dist_thresh)

        # Create patch collection with specified colour/alpha
        pc = PatchCollection([rect], \
                             facecolor='gray', alpha=0.2, edgecolor='k',linewidth=0)

        # Add collection to axes
        ax.add_collection(pc)
        
    return None