import numpy as np
import numpy.ma as ma

print_clusts = lambda clusts : [print(f"[{c[0]:4d},{c[-1]:4d}]") for c in clusts]

print_ctimes = lambda clusts : [print(f"[{time_arr[c[0]]:6.3f},{time_arr[c[-1]]:6.3f}]") for c in clusts]

print_ctdiff = lambda clusts : [print(f"{time_arr[c[-1]] - time_arr[c[0]]:6.3f}") for c in clusts]

print_times = lambda l: list(map(lambda x: f"{x:6.3f}",l))

def intersecting_bounds(a1,a2,b1,b2):
    """
    Check whether two ranges intersect
    """
    return (((a1 >= b1) & (a1 <= b2)) | 
            ((a2 >= b1) & (a2 <= b2)) | 
            ((b1 >= a1) & (b1 <= a2)) | 
            ((b2 >= a1) & (b2 <= a2)))    


def contains(a1,a2,b1,b2):
    """
    Check whether one range contains another
    """    
    return (((a1 >= b1) & (a2 <= b2)) | # a in b
            ((b1 >= a1) & (b2 <= a2)))  # b in a  

inter_bounds = lambda p1, p2: intersecting_bounds(p1[0],p1[-1],p2[0],p2[-1])
conta_bounds = lambda p1, p2: contains(p1[0],p1[-1],p2[0],p2[-1])
