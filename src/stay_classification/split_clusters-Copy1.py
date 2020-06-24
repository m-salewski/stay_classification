import numpy as np
import numpy.ma as ma
tdiff = lambda x : np.concatenate([np.array([0.0]), x[1:]-x[:-1]])

time_thresh = 5.0
eps=0.05
min_samples = 5

def get_split_clusters(labels, time_thresh, x_in, y_in):

    labels = labels.copy()
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    print('Old labels:', set(labels))

    # Create a updated label for each new cluster
    new_label = max(list(labels))+1

    for k in unique_labels:
        
        if (k!=-1):     
        
            cluster_member_mask = (labels == k)
            x = x_in[cluster_member_mask]

            labels = get_split_cluster(labels, new_label, k, cluster_member_mask, time_thresh, x, y_in)
        else:
            print('\tbut do nothing for k =',k)
        
    print('New labels:',set(labels))
    return labels

get_plural = lambda n: f"there is {n} break" if n==1 else f"there are {n} breaks"

def get_split_cluster(labels, new_label, cluster_label, cluster_member_mask, time_thresh, x_in, y_in):

    # Check if there are time breaks; 
    # if so, relabel the event with new class labels
    td = tdiff(x_in.reshape(x_in.size))
    td_breaks = (td > time_thresh)

    if np.any(td_breaks) & (cluster_label != -1):       

        # Get a subset of labels 
        sublabels = labels[cluster_member_mask]         

        # Measure the durations between all events; look for the large gaps
        new_td_breaks = ma.nonzero(td_breaks)[0]
        print(f"\ttime break(s) found: for k={cluster_label}, {get_plural(new_td_breaks.size)}")

        # for each break, get the ranges for the new labels
        lowerbound = 0
        for m in new_td_breaks:

            #print(f"for {m,lowerbound}:", x[m-1],x[lowerbound],x[m-1]-x[lowerbound])            

            if x_in[m-1]-x_in[lowerbound] <= 2.0:
                sublabels[lowerbound:m]=-1
            else:
                sublabels[lowerbound:m]=new_label

            lowerbound = m
            new_label += 1

        # Update the labels    
        labels[cluster_member_mask] = sublabels
    else:
        print(f"\tpass on k={cluster_label}")

    return labels

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