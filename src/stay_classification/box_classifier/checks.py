import numpy as np
import numpy.ma as ma


def check_means(means, configs, nr_events):
    """
    Check if means are converged given a number of means is sufficient.
    
    :param means: [float] List of means
    :param configs: dict containing the parameters (currently unused)
    
    :return: bool Mean of the region are converged        
    """
    
    #NOTE: Originally, there were additional conditions to further check for convergence, 
    # but these were moved to "extend_edge"
    #TODO: Check some alternatives:
    # 1. means could be less than 10% of eps:
    #    `return all([abs(m - m0)<=eps/10 for m in means[-nr_samples:]])`
    # 2. time-duration between first and last mean is sufficiently long
    # 3. both
    #TODO: Remove the `nr_events` when possible
    
    #count_thresh = configs['count_thresh']
     
    m0 = means[-nr_events]
    
    return all([m == m0 for m in means[-nr_events:]])


def check_times(clust1_t0,clust1_t1,clust2_t0,clust2_t1):
    
    assert clust1_t0<clust1_t1, "Cluster1 times are wrong"
    assert clust2_t0<clust2_t1, "Cluster2 times are wrong"
    
    embedded =  check_embed(clust1_t0,clust1_t1,clust2_t0,clust2_t1)
            
    overlap = check_overlap(clust1_t0,clust1_t1,clust2_t0,clust2_t1)
    
    if overlap: 
        print('overlap')
        return True
    elif embedded: 
        print('embedded')
        return True
    else:
        print('safe')
        return False
            
def check_embed(clust1_t0,clust1_t1,clust2_t0,clust2_t1):
    
    embedded = ( \
        ((clust1_t0>=clust2_t0) & (clust1_t1<clust2_t1)) | \
        ((clust2_t0>=clust1_t0) & (clust2_t1<clust1_t1))
               )  
    return embedded

def check_overlap(clust1_t0,clust1_t1,clust2_t0,clust2_t1):
    
    overlap = ( \
        #Check is cluster1 is within cluster 2               
        (((clust1_t0>=clust2_t0) & (clust1_t0<clust2_t1)) & \
         ((clust2_t1>=clust1_t0) & (clust2_t1<clust1_t1))) | \
        #Check is cluster2 is within cluster 1
        (((clust2_t0>=clust1_t0) & (clust2_t0<clust1_t1)) & \
         ((clust1_t1>=clust2_t0) & (clust1_t1<clust2_t1)))                
               )  
    return overlap
