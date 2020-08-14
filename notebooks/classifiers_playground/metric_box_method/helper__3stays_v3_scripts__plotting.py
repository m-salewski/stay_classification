import numpy as np
#from numpy import mean as np.mean

from synthetic_data.plotting import plot_trajectory, add_plot_trajectory, add_plot_seg_boxes
from helper__metric_box__explore import iqr_metrics, get_boxplot_quants, get_clusters_rev 

def plot_cluster_boxplots(time_arr, raw_arr, noise_arr, t_segs, x_segs, clusters, clusters_rev, configs):
    """
    """
    dist_thresh = configs['dist_thresh']
    
    ax = plot_trajectory(time_arr, raw_arr, noise_arr, t_segs, x_segs, dist_thresh);
    add_plot_seg_boxes(t_segs, x_segs, dist_thresh, ax)

    ax.set_xlim([5.75,18.25])

    bp_data, labels, positions, widths = get_boxplot_quants(time_arr, noise_arr, clusters)

    axt = ax.twiny()
    _ = axt.boxplot(bp_data, labels=labels, positions=positions, boxprops=dict(color='red'), widths=widths)   

    for label in axt.get_xticklabels():
        label.set_rotation(90)
    axt.set_xticklabels(labels, visible=True, color='red')


    axt.set_xlim(ax.get_xlim())
    axt.legend(['forward clusters'], bbox_to_anchor=(1.15, 0.6), loc='center right', ncol=1);


    bp_data, labels, positions, widths = get_boxplot_quants(time_arr[::-1], noise_arr[::-1], clusters_rev)

    axt = ax.twiny()
    labels = list(map(lambda x: f"{x:.2f}", positions))
    _ = axt.boxplot(bp_data, labels=labels, positions=positions, boxprops=dict(color='blue'), widths=widths)
    axt.legend(['reverse clusters'], bbox_to_anchor=(1.15, 0.4), loc='center right', ncol=1);

    for label in axt.get_xticklabels():
        label.set_rotation(90)
        
    axt.set_xticklabels(labels, visible=False)

    axt.set_xlim(ax.get_xlim())
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper right', ncol=1)
    
    return ax


def add_plot_cluster_boxplots(time_arr, raw_arr, noise_arr, t_segs, x_segs, clusters, clusters_rev, configs, ax):
    """
    """
    dist_thresh = configs['dist_thresh']
    
    add_plot_trajectory(time_arr, raw_arr, noise_arr, t_segs, x_segs, dist_thresh, ax);
    add_plot_seg_boxes(t_segs, x_segs, dist_thresh, ax)

    ax.set_xlim([5.75,18.25])

    bp_data, labels, positions, widths = get_boxplot_quants(time_arr, noise_arr, clusters)

    axt = ax.twiny()
    _ = axt.boxplot(bp_data, labels=labels, positions=positions, boxprops=dict(color='red'), widths=widths)   

    for label in axt.get_xticklabels():
        label.set_rotation(90)
    axt.set_xticklabels(labels, visible=True, color='red')


    axt.set_xlim(ax.get_xlim())
    axt.legend(['forward clusters'], bbox_to_anchor=(1.15, 0.6), loc='center right', ncol=1);


    bp_data, labels, positions, widths = get_boxplot_quants(time_arr[::-1], noise_arr[::-1], clusters_rev)

    axt = ax.twiny()
    labels = list(map(lambda x: f"{x:.2f}", positions))
    _ = axt.boxplot(bp_data, labels=labels, positions=positions, boxprops=dict(color='blue'), widths=widths)
    axt.legend(['reverse clusters'], bbox_to_anchor=(1.15, 0.4), loc='center right', ncol=1);

    for label in axt.get_xticklabels():
        label.set_rotation(90)
        
    axt.set_xticklabels(labels, visible=False)

    axt.set_xlim(ax.get_xlim())
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper right', ncol=1)
    
    return ax
