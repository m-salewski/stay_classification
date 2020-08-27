# Quick-box classification

**21.07.2020**

## Current Meth

**Idea**: stay events are collected in a bounding box (cylinder in 2D1T): properly centered on a statistical location, the B-box stops collecting events when at most one falls out of the current box

1. Specify time-duration and distanct threshold
2. Get initial set of points
    * use index=0, and the highest index from the time increment
3. Get bonding box (circle) for set of points
4. Count the events in the box
5. Extend the forward edge of the box by the time increment, include any events falling within and count again
6. if count increases, repeat; else, break and restart with current index as new starting index

**Note**: this stops early, as soon as an event falls outside of the current box; ignoring events as outliers would be useful.

## Current issues

* as with all methods, there is a risk of false positives when 
    * the travel is "slow" (with a small slope $\leq 2\varepsilon/\tau$), with is effectively the limit of the algorithm
    * the density of events is high
* the event sequence is cut too frequently
    * this is okay, possibly desired $\to$ need to then have a refinement where small clusters are aggregated into larger clusters.
* outliers are not detected and ignored.
* no implicit buffers around events are considered
    * _e.g._ each event has a null-uncertainty and is included in a box _only_ when it is within the box
    * generally, each event has a nonzero uncertainty which could intersect a box with higher probability
        * this would call for more advanced methods.

## ToDos

* $\checkmark$ check that the newly split clusters are 
    * $\checkmark$ ~~contain enough samples~~
    * $\checkmark$  ~~have a total duration longer than the threshold~~
    * $\checkmark$  ~~are not embedded within another cluster~~ $\to$ by definition
* $\checkmark$ check that there are no embedded clusters
    * $\checkmark$  ~~times don't overlap~~ $\to$ by definition
    * $\boldsymbol{\times}$ ~~if refined clusters are embedded, check if one cluster is noise~~ $\to$ not needed
* check the stddev of the refined clusters are smaller

