# Quick-box classification

**21.07.2020**

## Current Meth

**Idea**: cluster events are collected in a bounding box (cylinder in 2D1T): properly centered, the B-box should stop collecting events if extended beyond a cluster.

1. Specify time-duration and location threshold
2. Get initial set of points
    * use index=0, and the highest index from the time increment
3. Get bouding box (circle) for set of points
4. Count the events in the box
5. Extend the box by the time increment and count again
6. if count increases, repeat; else, break and restart with new starting index

_Consider_: better to stop early and then stitch (if cluster centroids are within each other's boxes? $\to$ drifting!) rather than stop later. 

## Current issues

* as with all methods, there is a risk of false positives when 
    * the travel is "slow" (with a small slope $\leq 2\varepsilon/\tau$), with is effectively the limit of the algorithm
    * the density of events is high
* the event sequence is cut too frequently
    * this is okay, possibly desired $\to$ need to then have a refinement where small clusters are aggregated into larger clusters.

## ToDos

* $\checkmark$ check that the newly split clusters are 
    * $\checkmark$ ~~contain enough samples~~
    * $\checkmark$  ~~have a total duration longer than the threshold~~
    * $\checkmark$  ~~are not embedded within another cluster~~ $\to$ by definition
* $\checkmark$ check that there are no embedded clusters
    * $\checkmark$  ~~times don't overlap~~ $\to$ by definition
    * $\boldsymbol{\times}$ ~~if refined clusters are embedded, check if one cluster is noise~~ $\to$ not needed
* check the stddev of the refined clusters are smaller

