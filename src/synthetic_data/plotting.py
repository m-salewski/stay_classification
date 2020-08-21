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
def plot_trajectory(t_arr, x_arr, nx_arr, t_segs, x_segs, dist_thresh):

    plt.figure(figsize=(20,5))

    plt.plot(t_segs, x_segs, **segs_plot_kwargs, label='adjusted raw stays')
    plt.plot(t_arr, x_arr, ':', label='raw journey')
    plt.plot(t_arr, nx_arr, '.-', label='noisy journey', alpha=0.25)

    plt.legend();

    plt.xlabel(r'time, $t$ [arb.]')
    plt.ylabel(r'position, $x$ [arb.]')

    ymin = nx_arr.min()-1*dist_thresh
    ymax = nx_arr.max()+1*dist_thresh

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))

    ax.set_xlim(-0.05, 24.05)

    ax.set_title('Trajectory', fontsize=24)
    ax.grid(visible=True); 

    return ax
'''

def plot_trajectory(t_arr, x_arr, nx_arr, t_segs, x_segs, dist_thresh):
    
    plt.figure(figsize=(20,5))
    ax = plt.gca()
    
    add_plot_trajectory(t_arr, x_arr, nx_arr, t_segs, x_segs, dist_thresh, ax)

    return plt.gca()

def add_plot_trajectory(t_arr, x_arr, nx_arr, t_segs, x_segs, dist_thresh, ax):
    
    if (t_segs is not None) & (x_segs is not None):
        ax.plot(t_segs, x_segs, **segs_plot_kwargs, label='adjusted raw stays')
        ymin = min([x for x in x_segs.tolist() if x != None])-1*dist_thresh
        ymax = max([x for x in x_segs.tolist() if x != None])+1*dist_thresh         
    
    if (t_arr is not None) & (x_arr is not None):
        ax.plot(t_arr, x_arr, ':', color='C0', label='raw journey')
        ymin = x_arr.min()-1*dist_thresh
        ymax = x_arr.max()+1*dist_thresh
    
    if (t_arr is not None) & (nx_arr is not None):
        ax.plot(t_arr, nx_arr, '.-', color='C1', label='noisy journey', alpha=0.25)
        ymin = nx_arr.min()-1*dist_thresh
        ymax = nx_arr.max()+1*dist_thresh        

    ax.legend();

    ax.set_xlabel(r'time, $t$ [arb.]')
    ax.set_ylabel(r'position, $x$ [arb.]')

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))

    ax.set_xlim(-0.05, 24.05)

    ax.set_title('Trajectory', fontsize=24)
    ax.grid(visible=True); 

    return None


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

#from synthetic_data.plotting import plot_trajectory, add_plot_trajectory, add_plot_seg_boxes
#from helper__metric_box__explore import iqr_metrics, get_boxplot_quants, get_clusters_rev 

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


def get_boxplot_quants(t_arr, x_arr, clusters):
    
    data = []
    labels = []
    positions = []
    widths = []

    for cl_nr, clust in enumerate(clusters):

        # Get the subseqs
        xs = t_arr[clust]
        ys = x_arr[clust]
        data.append(ys)
        widths.append(xs[-1]-xs[0])
        
        pos = (xs[-1]+xs[0])/2
        positions.append(pos)
        labels.append(f"{pos:.2f}")
    
    return data, labels, positions, widths


def get_boxplot_centers(t_arr, x_arr, clusters):
    
    data = []
    labels = []
    positions = []
    widths = []

    for cl_nr, clust in enumerate(clusters):

        # Get the subseqs
        xs = t_arr[clust]
        ys = x_arr[clust]

        # Mask to include only events within the IQR
        q5l = np.quantile(ys,0.5, interpolation='lower')
        q5h = np.quantile(ys,0.5, interpolation='higher')

        data.append((q5l+q5h)*0.5)
        positions.append((xs[-1]+xs[0])/2)
        
    return data, positions


def get_boxplot_lines(t_arr, x_arr, clusters):
    
    data = []
    labels = []
    positions = []
    widths = []

    for cl_nr, clust in enumerate(clusters):

        # Get the subseqs
        xs = t_arr[clust]
        ys = x_arr[clust]

        # Mask to include only events within the IQR
        q5l = np.quantile(ys,0.5, interpolation='lower')
        q5h = np.quantile(ys,0.5, interpolation='higher')

        data.append((q5l+q5h)*0.5)
        data.append((q5l+q5h)*0.5)
        
        positions.append(xs[0])
        positions.append(xs[-1])        
        
    return data, positions


def get_boxplot_iqr_midpoints(t_arr, x_arr, clusters):
    
    data = []
    labels = []
    positions = []
    widths = []

    for cl_nr, clust in enumerate(clusters):

        # Get the subseqs
        xs = t_arr[clust]
        ys = x_arr[clust]

        # Mask to include only events within the IQR
        q25 = np.quantile(ys,0.25, interpolation='lower')
        q75 = np.quantile(ys,0.75, interpolation='higher')

        data.append((q25+q75)*0.5)
        positions.append((xs[-1]+xs[0])/2)
        
    return data, positions


def add_plot_cluster_boxplots(time_arr, noise_arr, clusters, dist_thresh, ax):
    """
    """

    ax.set_xlim(-0.05, 24.05)

    bp_data, labels, positions, widths = get_boxplot_quants(time_arr, noise_arr, clusters)

    axt = ax.twiny()
    _ = axt.boxplot(bp_data, labels=labels, positions=positions, boxprops=dict(color='red'), widths=widths)   

    for label in axt.get_xticklabels():
        label.set_rotation(90)
    axt.set_xticklabels(labels, visible=True, color='red')

    axt.set_xlim(ax.get_xlim())
    axt.legend(['Clusters'], bbox_to_anchor=(1.15, 0.6), loc='center right', ncol=1);
    
    return None


def plot_cluster_boxplots(time_arr, noise_arr, clusters, dist_thresh):
    """
    """
    
    plt.figure(figsize=(20,5))
    ax = plt.gca()
    
    add_plot_cluster_boxplots(time_arr, noise_arr, clusters, dist_thresh, ax)

    return plt.gca()

