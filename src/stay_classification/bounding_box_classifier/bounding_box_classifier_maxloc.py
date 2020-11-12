import numpy as np

import matplotlib.pyplot as plt

def get_max_loc_(d_thresh):
    
    """
    Get the approximate location of the largest cluster, ie that with the most events
    """
    
    def meth(arr):
        
        loc = None
        
        # Try the bins using the steps in the dist. thresh.
        # TODO: test if this matters? - a least in the early stages
        bins = np.arange(arr.min(), arr.max(), d_thresh)
        
        if bins.size > 1:
            hist_data, hist_bins = np.histogram(arr, bins=bins)
        else: 
            hist_data, hist_bins = np.histogram(arr)
        
        if bins.size <= 1: 
            # When the distance is too small
            hist_data, hist_bins = np.histogram(arr)
            max_bin = np.where(hist_data == hist_data.max())[0]       
            loc = 0.5*(hist_bins[max_bin][0] + hist_bins[max_bin+1][0])
        
        else:
            # Here the bins are shifted to better approximate the location
            #NOTE: this might be overkill
            counts = 0
            best_loc = 0.0
            
            shift_frac = 1/2 # Could also use 1/3
            shift_intervals = 1
            shifts = range(2*shift_intervals+1)
            
            for n in shifts:
                
                # Shift the bins to maximize counts in a bin
                bins_ = bins+(n-shift_intervals)*d_thresh*shift_frac               
                hist_data, hist_bins = np.histogram(arr, bins=bins_)
                
                # Save the location with the most counts
                max_counts = hist_data.max()                    
                max_bin = np.where(hist_data == max_counts)[0]        
                #Since the chosen bin is the left edge, 
                # to get the location, the midpoint is used.
                loc = 0.5*(hist_bins[max_bin][0] + hist_bins[max_bin+1][0])
                #print(max_counts, loc, hist_data, hist_bins)
                if max_counts > counts:
                    best_loc = loc
                    counts = max_counts
                    
            loc = best_loc

        return loc

    
    return meth

def plot_max_loc_(d_thresh):
    
    plt.figure(figsize=[10,6])
    
    def meth(arr, ax=None):
        
        if ax == None:
            ax = plt.subplot(111)
            
        # Get the bins using the steps in the dist. thresh.        
        bins = np.arange(arr.min(), arr.max(), d_thresh)
        
        if bins.size <= 1: 
            bins = None

            hist_data = ax.hist(arr, bins=bins)
            hist_data, hist_bins = np.histogram(arr, bins=hist_data[1])

            #print(hist_data,hist_bins, hist_data.max())
            max_bin = np.where(hist_data == hist_data.max())        
            loc = hist_bins[max_bin][0]  
        
        else:
            align = ['edge', 'center', 'edge']
            counts = 0
            best_loc = 0.0
            
            shift_frac = 1/2
            shift_intervals = 1
            shifts = range(2*shift_intervals+1)
            
            for n in shifts:
                
                # Shift the bins to maximize counts in a bin
                bins_ = bins+(n-shift_intervals)*d_thresh*shift_frac        
                hist_data, hist_bins =np.histogram(arr, bins=bins_)
                
                width=d_thresh
                if n > 0: width=-d_thresh
                
                # Save the location with the most counts
                max_counts = hist_data.max()                    
                max_bin = np.where(hist_data == max_counts)        
                loc = hist_bins[max_bin][0]                       
                if max_counts > counts:
                    best_loc = loc
                    counts = max_counts
                
                ax.bar(hist_bins[:-1], hist_data, width=d_thresh, 
                        align='center', alpha=0.5, 
                        label=f'{(n-shift_intervals)/3:5.2f}: max. loc. {loc:6.3f}')
            
            loc = best_loc        
        
        ax.set_title(f"Current location with max. events: {loc:6.3}")
        ax.set_xlabel(rf"$x$-bins ($\Delta x$ ={d_thresh:6.4})[km]")
        ax.set_ylabel("Counts")
        ax.legend()
        
        return loc, counts, ax
    
    return meth
