# Stay classification: clustering with box-refinement

As of **26.08.2020**

## Algorithm

1. `get_clusters_3`
    1. initial clustering and merging
        1. initial pass
            1. confine the errors and speeds up the compute.
        2. identifies clusters based on spatio-temporal nearness
            1. events must first be close in time $\to$ identifies many small clusters
            2. adding a new event must be close to the statistical center of a cluster
        3. adjacent clusters are merged when they not sufficiently separated         
2. `get_extended_clusters`
    1. extend the cluster "boxes" bwd/fwd to obtain limits
3. `separate_clusters_hier`
    1. identify and separate overlapping clusters
4. `merge_combo`
    1. since the above splits clusters, need to re-merge
5. `shift_box`
    1. in order to identify small clusters as part of travels, shift the box fwd/bwd and look for changes
        * **TODO** apply _only_ to small enough clusters
6. `get_iqr_trimmed_clusters`
    1. refine the cluster limits by keeping only events which fall within the IQR of the cluster.


## Current evaluation

### Results 

Mostly, the results are good: more than 80% of the trajectories have prec and/or recall above 0.8

The main issue
* loss of precision due to non-stays being identified
    * this (loss) is increased with higher event density but compensated by an increase in recall
        * since many clusters are identified and the stay events have a higher probability of being classified as stay events.
<br/>

**TODO** since stay events are classified and not the accuracy of the stays, it would be useful to have a measure of the stay accuracy: _ie_ once a stay is identified, how much of that stay is correctly classified.

* 40% of chains result in some problem:
    * not correct stays; p/rec are below 0.8
    * leading 25% with 4 stays, 15% with 2 or fewer stays
    * **Note** these have not all been checked to see where the main problems are
        * missing large stays
        
* the rest has not been evaluated

### Pros
* can separate sequence and identify rudimentary clusters

### Cons
* can't distinguish all stays from travels 
    * nothing created to do so.


# Clustering

## Split the clusters which have a temporal gap

### IQR-plotting

For each sub-cluster, plot the quantile boxes with whistkers.

**Notes**
* the boxes usually capture the baseline of the underlying stay
* the forward and backward clusters
    * usually the same clusters in the stays with similar IQRs
    * usually different in the 

### From here

At this point, it seems that the basic clusters are formed. 

Next, use the IQRs of these clusters as the new bounds for extending the cluster: essentially using the extensible box method.

Note that the IQR can be larger than the allow distance threshold; the box would therefore need to be the smaller of the two but with the same mean and/or median


## ToDos

* check that the newly split clusters are 
    * contain enough samples
    * $\checkmark$ ~~have a total duration longer than the threshold~~ see `get_extended_clusters`
    * $\checkmark$ ~~are not embedded within another cluster~~ possible, but unused; see ~~`separate_embedded_clusters`~~ 
* $\checkmark$ ~~check that there are no embedded clusters~~ see ~~`separate_embedded_clusters`~~
    * ~~times don't overlap~~ $\checkmark\to$ have function
    * ~~if refined clusters are embedded, check if one cluster is noise~~ ignored
* check the stddev of the refined clusters are smaller
* check on adjacent clusters
    * $\checkmark$ ~~check on the limits of time between adjacent clusters, esp. when they have the same mean and/or median~~ see `merge_clusters_combo`
        * if at $x_a$ for $t_1$ then again at $x_a$ for $t_2$, duration between going from and back to $x_a$ should reflect some mimimal amount of time ($2\times$ travel and a stay), 
            * e.g. 
                * from $x_a$ to an unknown $x_b$ with a stay at $x_b$ of the miminum duration and then back to $x_a$ 
                * should satisfy some criterion or _there is no inbetween travel and $t_1$ and $t_2$ are part of the same cluster_
    * $\checkmark$ ~~check of the metrics for inter-subcluster gaps~~ see `merge_clusters_combo`
        * these are also clusters but unlabeled after the first round of clustering
    * **!!!** check that the identified stays during the substages are not immediately adjacent
        * _ie_ no stays should have indices like `[10, ..., 20]` followed by `[21, ..., 30]`
            * such stays are possible $\to$ but there should be a minimum travel time if the locations are distinct
* check the gaps between (short) clusters
    * look for limits for the length of a travel
* include a measure of the placement of the stays and measure the deviation from the location of the true stay
    * can use a segment-based approach as in the evaluation script.
* include a measure which splits the scoring (prec/rec/err) into the overlapping and the non-overlapping stays of a trajectory   
    * then score them individually,  
    * then have a measure of how much the non-overlapping parts can be ignored
        * _ie_ does the classification get all the stays distinctly and mostly correctly classified?
* for the extend-box portion, 
    * start with the biggest boxes and extend these first
    * then, see if they absorb smaller boxes, and exclude these from further extensions.
 
## Notes: 

* **Gaps** 
    * since there are subclusters for a given cluster which are separated by gaps, these gaps ...

### 1-stay
* gap-merging works well when good density; poorly when otherwise

### 3 clusters
* overlapping clusters
    * embedded and also identitcal clusters
    * **Notes**
        * usually have the same median and mean ("M&M"), but not always
    * overlap on edge between two spatially close clusters
        * maybe not sharing M&M,
    * megacluster when two clusters share an $x$ (rare)
        * should be avoidable with gap durations
* missing short clusters
    * these usually occur on the edges
    * if using a IQR-postfilter, many of these will get dropped
    
### 4 clusters
* mis-identified cluster, aka "floater"
    * part of a travel
        * in the canonical 3-stays, these are always between larger stays
        * **Todo** 
            * _check if these have insufficient events with the IQR-mask_
            * _check if these have insufficient duration with the IQR-mask_            
    * seems to be short in duration
* overlapping clusters
    * embedded and also identitcal clusters
    * **Notes**
        * usually have the same median and mean ("M&M"), but not always
    * overlap on edge between two spatially close clusters
        * maybe not sharing M&M,
    * megacluster when two clusters share an $x$ (rare)
        * should be avoidable with gap durations
    * missing short clusters and overlaps/duplicates

## Future directions

It may be best to use this as an initial pass for the box, or _any_, method, thereby confining the errors and speeding up the compute.