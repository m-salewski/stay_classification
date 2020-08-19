# Stay classification: clustering with box-refinement

## Algorithm

### Summary

1. `get_clusters_x` finds the clusters based on nearness of events in space and time
    * events must first be close in time $\to$ identifies many small clusters
    * adding a new event must be close to the statistical center of a cluster      
2. `merge_clusters` merges neraby clusters based on ...
    * adjacent clusters are merged when they not sufficiently separated   
3. `merge_clusters_2`  merges neraby clusters based on ...
4. `extend_clusters` extend the clusters
5. `separate_clusters` break the overlapping clusters and then re-merge
6. `merge_clusters_2` merge the separated clusters
7. `intersect` the forward and backward clusters


### Notes

Mostly, the results are good: more than 80% of the trajectories have prec and/or recall above 0.8

The main issue
* loss of precision due to non-stays being identified
    * this (loss) is increased with higher event density but compensated by an increase in recall
        * since many clusters are identified and the stay events have a higher probability of being classified as stay events.
<br/>

**TODO** since stay events are classified and not the accuracy of the stays, it would be useful to have a measure of the stay accuracy: _ie_ once a stay is identified, how much of that stay is correctly classified.



## Current evaluation

### Pros
* can separate sequence and identify rudimentary clusters, then these are refined with extending a box of uncertainty around these in order to correclty assess their bounds.

### Cons
* can't distinguish stays from travels 
    * nothing created to do so.

## ToDos


* check that the newly split clusters are 
    * contain enough samples
    * $\checkmark$ ~~have a total duration longer than the threshold~~ see `get_extended_clusters`
    * $\checkmark$ ~~are not embedded within another cluster~~ possible, but unused; see `separate_embedded_clusters`
* $\checkmark$ ~~check that there are no embedded clusters~~ see `separate_embedded_clusters`
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
## Notes: 

* **Gaps** 
    * since there are subclusters for a given cluster which are separated by gaps, these gaps ...

### 1-stay
* gap-merging works well when good density; poorly when otherwise

## Future directions

It may be best to use this as an initial pass for the box, or _any_, method, thereby confining the errors and speeding up the compute.