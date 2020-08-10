import numpy as np

from .trajectory import get_stay

mind_speed = 3.5


get_t_to  = lambda t1, x1, x2 ,m : np.sign(x2-x1)*(x2 - x1)/m + t1
get_t_fro = lambda t2, x1, x2 ,m : np.sign(x2-x1)*(x1 - x2)/m + t2

def gen_stays(t_bounds, x_locs):

    stays = []
    for n in range(int(len(t_bounds)/2)):
        nn = 2*n
        stay = get_stay(t_bounds[nn], t_bounds[nn+1], x_locs[::-1][n])
        stays.append(stay)    

    return stays


def get1():

    # 1 stay
    t_bounds = [6.0, 18.0]
    x_locs = [0.0]
    
    return gen_stays(t_bounds, x_locs)


def get2(x_shift=0, x_dist=1.0):

    
    # 2 stays
    # x_shift
    # midpoint: 0
    # shifted right: -5.4, 
    # shifted left: 5.4
    
    if x_shift > 0:
        x_shift = min(x_shift, 5.7)
    if x_shift < 0:
        x_shift = max(x_shift, -5.7)
        
    x_midpt_1 = -x_dist/2
    x_midpt_2 = +x_dist/2    
    
    x_locs = [x_midpt_1, x_midpt_2]
    
    t_dist = abs(12 - get_t(12, x_locs[0], 0, mind_speed))
    t_bounds = [ 6, 
                12-x_shift-t_dist, 
                12-x_shift+t_dist,
                18.0] 

    return gen_stays(t_bounds, x_locs)


def get3_core(x_locs, middle_len, shift=12.0):
    
    time_thresh = 1/6
    
    t_midpt_1 = 12-middle_len/2.0
    t_midpt_2 = 12+middle_len/2.0  
    
    t_fro = get_t_fro(t_midpt_1, x_locs[0], x_locs[1], mind_speed)
    t_to  = get_t_to(t_midpt_2, x_locs[1], x_locs[2], mind_speed)
    
    if (t_fro + shift) - 6.0 < time_thresh:        
        shift = 6.1 + time_thresh - t_fro 
        
    if (18.0 - (t_to + shift)) < time_thresh:        
        shift = 17.90 - t_to - time_thresh
        
    t_bounds = [6.0, 
                t_fro     + shift, 
                t_midpt_1 + shift, 
                t_midpt_2 + shift, 
                t_to      + shift, 
                18.0]

    return t_bounds, x_locs


def get3e(x_dist=1.0, middle_len=1.0, middle_shift=12.0):

    # 3 stays, 2 equal
    #get3e(0.50, (1/6 -- 5.66))
    
    
    x_midpt_1 = -x_dist/2
    x_midpt_2 = +x_dist/2    
    
    x_locs = [x_midpt_1, x_midpt_2, x_midpt_1]

    t_bounds, x_locs = get3_core(x_locs, middle_len, middle_shift)
    
    return gen_stays(t_bounds, x_locs)


def get3(x_dist=1.0, middle_len=1.0, middle_shift=12.0):

    # 3 stays, 0 equal
    
    x_midpt_1 = -x_dist/2
    x_midpt_2 = +x_dist/2    
    
    x_locs = [x_midpt_1, 0, x_midpt_2]

    t_bounds, x_locs = get3_core(x_locs, middle_len, middle_shift)
    
    return gen_stays(t_bounds, x_locs)
