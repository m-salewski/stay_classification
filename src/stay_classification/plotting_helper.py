import matplotlib.pyplot as plt
import numpy as np

import random 

def plot_regular(labels, core_samples_mask, x_in, y_in, ax):

    # Black removed and is used for noise instead.

    unique_labels = set(labels)

    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    random.shuffle(colors)

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        x = x_in[class_member_mask & core_samples_mask]
        y = y_in[class_member_mask & core_samples_mask]    

        #print("for",k,"size is",xy.shape)

        ax.plot(x,y, 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14, label='{}, big'.format(k))

        x = x_in[class_member_mask & ~core_samples_mask]
        y = y_in[class_member_mask & ~core_samples_mask]    
        #print("for",k,"size is",xy.shape)    

        ax.plot(x,y, 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6, label='{}, small'.format(k))
        if k!=-1:
            y_mean = y_in[class_member_mask].mean()
            ax.plot([x_in[0],x_in[-1]],[y_mean, y_mean], '--', color=tuple(col), label='mean')    

    return ax


def plot_refined(fine_label, fine_core_samples_mask, xx_coarse, yy_coarse, coarse_label, col, ax):

    if fine_label==-1:

        col = [0, 0, 0, 1]                
        fine_cluster_mask = (fine_label == -1)
        x = xx_coarse[fine_cluster_mask]
        y = yy_coarse[fine_cluster_mask]  

        ax.plot(x,y, 'o', markerfacecolor=tuple(col),
                        markeredgecolor='k', markersize=6, label=f'{fine_label}')

        fine_std = np.std(y)

    else:

        fine_cluster_mask = (fine_label == fine_label)

        x = xx_coarse[fine_cluster_mask & fine_core_samples_mask]
        y = yy_coarse[fine_cluster_mask & fine_core_samples_mask]    

        ax.plot(x,y, 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14, label='{}{}, big'.format(coarse_label, fine_label))

        x = xx_coarse[fine_cluster_mask & ~fine_core_samples_mask]
        y = yy_coarse[fine_cluster_mask & ~fine_core_samples_mask]  

        ax.plot(x,y, 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6, label='{}{}, small'.format(coarse_label, fine_label))

        y_mean = yy_coarse[fine_cluster_mask].mean()
        ax.plot([xx_coarse[0],xx_coarse[-1]],[y_mean, y_mean], '--', color=tuple(col), label='{}{}, mean'.format(coarse_label, fine_label))    

        fine_std = np.std(yy_coarse[fine_cluster_mask])
        
    return ax, fine_std