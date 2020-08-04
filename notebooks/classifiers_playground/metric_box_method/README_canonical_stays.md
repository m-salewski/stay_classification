# Stay classification: cluster-splitting with box-refinement & shifting boxes


## Algorithmic Development Notes

### Gaps
* **Gaps** 
    * since there are subclusters for a given cluster which are separated by gaps, these gaps should also be smaller than the duration threshold; they most likely will already be, by definition, but there is a bit of randomness due to the noise.


### IQR-plotting

For each sub-cluster, plot the quantile boxes with whistkers.

**Notes**
* the boxes usually capture the baseline of the underlying stay
* the forward and backward clusters
    * usually the same clusters in the stays with similar IQRs
    * usually different in the 
    
### Travel logic

Let stat. metric be $\bar{x}_i$ for $\mathrm{cluster}_i$'s location for $i=1,2$,
<br/>
and $d(\bar{x}_1,\bar{x}_2) < \delta$
<br/>
and $t'_1$ is the last timepoint of $\mathrm{cluster}_1$ and $t_2$ is the first for $\mathrm{cluster}_2$,

Then, for all $t\in[t'_1,t_2]$, $\exists$ an $\bar{x}$ $\$$ 
1. the duration of $\Delta t := t_2-t'_1 \geq \tau$ + travel time with minimum velocity $v = 3.5$ km/hr
    * e.g. let $\bar{x}_1 = \bar{x}_2$ and $d(\bar{x}_1,\bar{x}) = 2\delta$,
    then, the overall duration $\Delta t = 2\cdot 2\delta/v + \tau$
    * specifically: $\bar{x}_i = 0$, $d(\bar{x}_i,\bar{x}) = 2$km, then $\Delta t \geq 1.31$hr when $\delta = 0.5$km and $\tau = 0.167$hr
2. $d(\bar{x}_1,\bar{x}) < \delta$ and $d(\bar{x},\bar{x}_2) < \delta$


## Current evaluation

### Pros
* can separate sequence and identify rudimentary clusters

### Cons
* can't distinguish stays from travels 
    * nothing created to do so.

## ToDos

* check that the newly split clusters are 
    * contain enough samples
    * have a total duration longer than the threshold
    * are not embedded within another cluster
* check that there are no embedded clusters
    * times don't overlap $\checkmark\to$ have function
    * if refined clusters are embedded, check if one cluster is noise
* check the stddev of the refined clusters are smaller
* check on the limits of time between adjacent clusters, esp. when they have the same mean and/or median
    * if at $x_a$ for $t_1$ then again at $x_a$ for $t_2$, duration between going from and back to $x_a$ should reflect some mimimal amount of time ($2\times$ travel and a stay), 
        * e.g. 
            * from $x_a$ to an unknown $x_b$ with a stay at $x_b$ of the miminum duration and then back to $x_a$ 
            * should satisfy some criterion or _there is no inbetween travel and $t_1$ and $t_2$ are part of the same cluster_
* check of the metrics for inter-subcluster gaps
    * these are also clusters but unlabeled after the first round of clustering

## Notes: 

### 1-stay
* gap-merging works well when good density; poorly when otherwise

## Future directions

It may be best to use this as an initial pass for the box, or _any_, method, thereby confining the errors and speeding up the compute.