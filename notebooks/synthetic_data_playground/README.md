# Synthetic data 

## Summary



## TODOs

1. include the params for generating the number of stays in the configs
    1. default to the log-normal distribution
2. Inlcude functionality for stay location bounds in configs
3. Include module for saving the data as pickle or CSV
4. Include conversions to & from datetime
5. include plotting from CSV/DataFrame
6. Include functionality to change from DataFrame to arrays
    * _ie_ have the classifiers work with the DF
    * **TODO** create additional modules for handling general CSV/DataFrames
7. Augment the scripts to include 2D
    * start with the same time segs and apply the same noise to both arrays
8. Include canonical subtrajectories: minimal and irreducible parts of full trajectories
    * these are fow faster testing and targeted improving
9. Include constraints in the general trajectories so that the stays and travels obey the same rules as the canonicals
    * _ie_ so that travels aren't slower than some `min_veloctiy` ($\approx 3.6$ km/hr)
