import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator

lims=[-0.01,1.01]

def add_plot_pr_scatter(p_correct, r_correct, ax, title=None):
    """
    """
    
    #ax.plot(r_correct, p_correct, 'o', alpha=0.5, markersize=8, markeredgecolor="None")
    binw=0.02
    bins=np.arange(0.0,1.0+binw,binw)

    cmap = cm.inferno
    norm = Normalize()

    h = ax.hist2d(np.array(r_correct), np.array(p_correct), \
                  bins=bins, norm=norm, cmin=binw, cmap=cmap, alpha=0.5, edgecolor=None)
    
    axins1 = inset_axes(ax, width="2%", height="90%", loc='center left')
    cb = plt.colorbar(h[3], cax=axins1)
    

    ax.set_xlim(lims)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))    
    ax.xaxis.set_minor_locator(MultipleLocator(binw))     

    ax.set_ylim(np.array(lims))
    ax.yaxis.set_major_locator(MultipleLocator(0.1)) 
    ax.yaxis.set_minor_locator(MultipleLocator(binw))  

    ax.set_xlabel("precision")
    ax.set_ylabel("recall")
    if title:
        ax.set_title(title)
    ax.grid()
    
    return None


def add_plot_pr_hist(bins, hist_p, hist_r, ax, title=None):
    """
    """
    binw=abs(bins[1]-bins[0])
    
    _ = ax.bar(bins[:-1], hist_p, alpha=0.5, width=binw, align='center', label="precision")
    _ = ax.bar(bins[:-1], hist_r, alpha=0.5, width=binw, align='center', label="recall")
    
    ax.set_xlabel("precision, recall")
    ax.set_ylabel("frac.")
        
    ax.set_xlim(lims)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))    
    ax.xaxis.set_minor_locator(MultipleLocator(binw))
    
    ax.set_ylim(np.array(lims))
    ax.yaxis.set_major_locator(MultipleLocator(0.1)) 
    
    ax.grid()
    
    if title: ax.set_title(title)
    
    ax.legend()
    
    return None


def add_plot_pr_cumsum(bins, hist_p, hist_r, ax, title=None):
    """
    """
    binw=abs(bins[1]-bins[0])
    
    cumsum_p = np.cumsum(hist_p)
    cumsum_r = np.cumsum(hist_r)
    
    _ = ax.plot(bins[:-1], cumsum_p, '-.', lw=2, alpha=0.95, label="precision")
    _ = ax.plot(bins[:-1], cumsum_r, ':',  lw=2, alpha=0.95, label="recall")
    
    ax.plot([0.9,0.9],[0,1], '--', color='gray', lw=1.2, alpha=0.5)
    ax.plot([0,1],[0.9,0.9], '--', color='gray', lw=1.2, alpha=0.5)
    
    ax.set_xlabel("precision, recall")
    ax.set_ylabel("rel. frac.")
    
    ax.set_xlim(lims)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))    
    ax.xaxis.set_minor_locator(MultipleLocator(binw))    
    
    ax.set_ylim(np.array(lims))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))  
    
    if title:
        ax.set_title(title)
    ax.grid()
    ax.legend()
    
    return None


def add_plot_err_cumsum(bins, hist_c, hist_i, ax, title=None):
    """
    """
    binw=abs(bins[1]-bins[0])

    cumsum_c = np.cumsum(hist_c)
    cumsum_i = np.cumsum(hist_i)
    cumsum_0 = np.cumsum(hist_c + hist_i)
    
    _ = ax.plot(bins[:-1], cumsum_c, '--', alpha=0.95,label="Correct")
    _ = ax.plot(bins[:-1], cumsum_i, '-.', alpha=0.95,label="Incorrect")
    _ = ax.plot(bins[:-1], cumsum_0, '-' , alpha=0.95,label="Total" )   
    
    ax.plot([0.9,0.9],[0,1], '--', color='gray', lw=1.2, alpha=0.5)
    ax.plot([0,1],[0.9,0.9], '--', color='gray', lw=1.2, alpha=0.5)
    
    ax.set_xlabel("error")
    ax.set_ylabel("frac.")
    
    ax.set_xlim(lims)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))    
    ax.xaxis.set_minor_locator(MultipleLocator(binw)) 
    
    ax.set_ylim(np.array(lims))
    ax.yaxis.set_major_locator(MultipleLocator(0.1)) 
    
    if title:
        ax.set_title(title)
    ax.grid()
    ax.legend()
    
    return None


def plot_scores_stats(p_correct, r_correct, p_incorrect, r_incorrect, title):
    """
    """
    fig, axs = plt.subplots(2,3, figsize=[20,12])
    axs = axs.flatten()

    lims=[-0.01,1.01]
    
    # Scatter plot: recall vs precision (Correct)
    add_plot_pr_scatter(p_correct, r_correct, axs[0])
    
    # Histograms
    binw=0.02
    bins=np.arange(0.0,1.0+binw,binw)
    total = len(p_correct) + len(p_incorrect)
    
    hp, _ = np.histogram(np.array(p_correct), bins=bins, density=False)
    hr, _ = np.histogram(np.array( r_correct), bins=bins, density=False)

    ## Histogram plot for absolute fractions
    add_plot_pr_hist(bins, hp/total, hr/total, axs[1], f"Correct stays")
    axs[1].set_ylabel('abs. frac.')

    ## Cum. Sum for relative fractions
    add_plot_pr_cumsum(bins, hp/len(p_correct), hr/len(r_correct), axs[2])
    axs[2].set_ylabel('rel. frac.')
    
    
    # Scatter plot: recall vs precision (incorrect)    
    add_plot_pr_scatter(p_incorrect, r_incorrect, axs[3])

    # Histograms
    hp, _ = np.histogram(np.array(p_incorrect), bins=bins, density=False)
    hr, _ = np.histogram(np.array( r_incorrect), bins=bins, density=False)

    ## Histogram plot for absolute fractions
    add_plot_pr_hist(bins, hp/total, hr/total, axs[4], f"Incorrect stays")
    axs[4].set_ylabel('abs. frac.')
    
    ## Cum. Sum for relative fractions
    add_plot_pr_cumsum(bins, hp/len(p_incorrect), hr/len(r_incorrect), axs[5])
    axs[5].set_ylabel('rel. frac.')
    
    fig.suptitle(title, fontsize=16)

    return fig, axs


def plot_errs_stats(errs_correct, errs_incorrect, title):
    """
    """
    fig, axs = plt.subplots(1,3, figsize=[20,6])
    axs = axs.flatten()

    lims=[-.01,1.01]

    total = len(errs_correct)+len(errs_incorrect)
    correct_frac = (len(errs_correct)/total)
    incorrect_frac = (len(errs_incorrect)/total)

    binw=0.01
    bins=np.arange(0.0,1.0+binw,binw)
    
    hpc, _ = np.histogram(np.array(errs_correct), bins=bins, density=False)
    hpi, _ = np.histogram(np.array(errs_incorrect), bins=bins, density=False)

    ## Histogram plot for absolute fractions
    add_plot_pr_hist(bins, hpc/total, hpi/total, axs[0], f"Hist. (abs. frac.)")
    axs[0].set_xlabel("error")
    axs[0].set_xlabel(f"error, bin width: {binw:6.2f}") 
    axs[0].legend(['Correct', 'Incorrect'])
    
    add_plot_pr_hist(bins, hpc/len(errs_correct), hpi/len(errs_incorrect), axs[1], f"Hist. (rel. frac.)")
    axs[1].set_xlabel("error")    
    axs[1].set_xlabel(f"error, bin width: {binw:6.2f}")
    axs[1].legend(['Correct', 'Incorrect'])

    add_plot_err_cumsum(bins, hpc/total, hpi/total, axs[2], f"C.-sum. (abs. frac.)")
    axs[2].set_xlim([9e-4, 1.1e0])
    axs[2].set_xscale('log')
    fig.suptitle(title, fontsize=16)
    
    return fig, axs


def plot_scores_stats_cominbed(p_correct, r_correct, p_incorrect, r_incorrect, title):
    """
    """
    fig, axs = plt.subplots(2,2, figsize=[20,12])
    axs = axs.flatten()

    lims=np.array([-.01,1.01])

    total = len(p_correct)+len(p_incorrect)
    correct_frac = (len(p_correct)/total)
    incorrect_frac = (len(p_incorrect)/total)

    # histograms    
    binw=0.02
    bins=np.arange(0.0,1.0+binw,binw)

    hpc = np.histogram(np.array(  p_correct), bins=bins, density=False)[0]/total
    hrc = np.histogram(np.array(  r_correct), bins=bins, density=False)[0]/total 
    hpi = np.histogram(np.array(p_incorrect), bins=bins, density=False)[0]/total
    hri = np.histogram(np.array(r_incorrect), bins=bins, density=False)[0]/total


    add_plot_pr_hist(bins, hpc, hpi, axs[0], f"Prec. hist. (abs. frac.)")
    axs[0].legend(['Correct', 'Incorrect'])
    axs[0].set_xlabel("Precision")

    
    add_plot_err_cumsum(bins, hpc, hpi, axs[1], f"Prec. C.-sum. (abs. frac.)")
    axs[1].set_xlabel("Precision")
    
    add_plot_pr_hist(bins, hrc, hri, axs[2], f"Rec. hist. (abs. frac.)")
    axs[2].legend(['Correct', 'Incorrect'])
    axs[2].set_xlabel("Recall")
    
    add_plot_err_cumsum(bins, hrc, hri, axs[3], f"Rec. C.-sum. (abs. frac.)")
    axs[3].set_xlabel("Recall")

    
    fig.suptitle(title, fontsize=16)
    

    return fig, axs
