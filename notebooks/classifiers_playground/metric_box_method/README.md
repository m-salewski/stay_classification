# Stay classification: cluster-splitting

## Current evaluation

### Pros
* can separate sequence and identify rudimentary clusters

### Cons
* can't distinguish stays from travels 
    * nothing created to do so.

## Notes
* need to ensure the time-backward trajectory works in roughly the same way as the time-forward one


### from `classifier_1D__metric_box__explore_IQR_v1`
* fwd/bwd clustering tend to catch the same clusters
* IQR of a metric-box cluster consistently captures the stays
    * this means: if there's a cluster for (part of) a stay, the segment line is usually within the IQR
    * some neighboring clusters have nearly overlappign IQRs or medians
* Plotting the cluster-IQRs (midpoints and cluster lengths) shows outlines of the clusters

### from `classifier_1D__metric_box__explore_combine_with_box_v1`
Applying box-method doesn't really help
* coalesces some neighboring clusters
    * good when the medians/IQRs are similar, but
    * bad when otherwise
* problem of overlapping clusters is increased
    * **TODO** need to merge/intersect/crop which is unhelpful

### from `classifier_1D__metric_box__explore_combine_with_box_v2`
Shifting the boxes seems useful, but more work is needed
* helps when (part of) a travel is identified in a cluster, but
* doesn't help when the cluster is part-stay/part-travel
* thresholds are unclear 
* **TODO** apply shifting when creating the box?


## ToDos

* **RENAME this module**
* check that the newly split clusters are 
    * contain enough samples
    * have a total duration longer than the threshold
    * are not embedded within another cluster
        * likely okay
* check that there are no embedded clusters
    * times don't overlap $\checkmark\to$ have function
    * if refined clusters are embedded, check if one cluster is noise
* check the stddev of the refined clusters are smalle
* $\checkmark$ need to ensure the time-backward trajectory works in roughly the same way as the time-forward one
* merge neighboring clusters if they are close enough in time and space
* create some cluster characterisations/metrics which help to distinguish stays/travels

## Future directions

It may be best to use this as an initial pass for the box, or _any_, method, thereby confining the errors and speeding up the compute.
