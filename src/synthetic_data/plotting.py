import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot_trajectory(t_arr, x_arr, nx_arr, t_segs, x_segs, eps):
    
    segs_plot_kwargs = {
        'linestyle':'--', 
        'marker':'o', 
        'color':'k', 
        'linewidth':4.0, 
        'markerfacecolor':'w', 
        'markersize':6.0, 
        'markeredgewidth':2.0
        }

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