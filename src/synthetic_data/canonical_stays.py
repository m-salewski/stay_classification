import numpy as np

from .trajectory import get_stay

mind_speed = 3.5


get_t = lambda time_point, x1, x2 ,m : (x1 - x2)/m + time_point

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


def get2(x_dist=1.0):

    # 2 stays
    
    x_midpt_1 = -x_dist/2
    x_midpt_2 = +x_dist/2    
    
    x_locs = [x_midpt_1, x_midpt_2]
    
    t_dist = abs(12 - get_t(12, x_locs[0], 0, mind_speed))
    t_bounds = [6.0, 
                12-t_dist, 
                12+t_dist,
                18.0] 

    return gen_stays(t_bounds, x_locs)


def get3_core(x_locs, middle_len):
    
    t_midpt_1 = 12-middle_len
    t_midpt_2 = 12+middle_len   
    
    t_bounds = [6.0, 
                get_t(t_midpt_1, x_locs[0], x_locs[1], mind_speed), 
                t_midpt_1, 
                t_midpt_2, 
                get_t(t_midpt_2, x_locs[1], x_locs[2], mind_speed), 
                18.0]

    return t_bounds, x_locs


def get3e(x_dist=1.0, middle_len=1.0):

    # 3 stays, 2 equal
    
    x_midpt_1 = -x_dist/2
    x_midpt_2 = +x_dist/2    
    
    x_locs = [x_midpt_1, x_midpt_2, x_midpt_1]

    t_bounds, x_locs = get3_core(x_locs, middle_len)
    
    return gen_stays(t_bounds, x_locs)


def get3(x_dist=1.0, middle_len=1.0):

    # 3 stays, 0 equal
    
    x_midpt_1 = -x_dist/2
    x_midpt_2 = +x_dist/2    
    
    x_locs = [x_midpt_1, 0, x_midpt_1]

    t_bounds, x_locs = get3_core(x_locs, middle_len)

    return gen_stays(t_bounds, x_locs)