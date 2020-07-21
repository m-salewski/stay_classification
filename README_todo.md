# TODO

## Sythetic data

1. See it's own `README.md`
2. Specifically, create trajectory class to store all the data
    * Use the functions from the scripts and have a few get/set methods
3. Write method to output CSVs and work from pandas DFs

## Illustration

1. $\checkmark$ ~~Find the correct NBs~~
2. $\checkmark$ ~~Grab the figures~~
    * ~~with the overall problem~~
    * ~~with the box method solution~~
3. $\checkmark$ Include realworld example with map and 2D1T plot
    * $\checkmark$ ~~make the connection to the 1D1T plots~~
4. Revise and correct the text
5. $\checkmark$ ~~Create a final html NB~~ $\to$ moved to the root `README.md`

## Evaluation methods

1. Turn evalutation metrics into module
**Note** see the quick_box NBs

## Box methods

### 1-way box methods

#### basic box method (currently as `quick_box`; initially as `bounding_box`)
1. ~~Update and clean up~~
    * there may be some redudancy in the scripts $\to$ check this!
2. apply batch evaluation 
    * **Note** this method doesn't distinguish travels so well as it looks for mini-clusters
        * it has a generically high false-pos. rate.

#### metric box method (currently as `cluster_splitting`)
1. Revisit and clean up
2. check if it is more useful than the basic box
    * ideally if it gives a better starting point than the basic box method
3. Apply the evaluation
    * is it better than the above method
4. Document it's limitations

### 2-way box method (currently as `box_method`) 
1. Refine algorithm and make it robust
2. Ensure that it produces what it should
    * Test it with the evaluations
3. Include previous box methods as initial passes to refine upon
    * this will give a speed-up but also improve accuracy.
    * will require merging and checking 
    * Some ideas:
        1. first pass with `quick_box`
        2. apply 2-way box to longest cluster
            * check if the box increases
            * check slope
        3. iterate over all remaining clusters 
            * start with next longest
            * skip if they overlap with a newly processed cluster

## Other methods

1. Piecewise linear
    * can it be made to converge (and faster)?
    * can it be adopted into the box method(s) to improve speed?
        * should it?
    * can it hold up in 2D?
   
2. DB-Scan
    * rename as "DB_split"
        * since the DBSCAN clusters will be split by 
            * the time requirement
            * inbetween cluster(s)
    * try again with the random data in batch
    * apply the evaluation
