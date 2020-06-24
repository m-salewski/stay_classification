#

from ..box_method import *

def test__get_time_ind():
    
    t_arr = np.arange(2,10,2)
    time_thresh = 1
    
    tests = \
        [
            get_time_ind(t_arr, 1, time_thresh, -1)==0,
            get_time_ind(t_arr, 10, time_thresh, 1)==(t_arr.size - 1),
            get_time_ind(t_arr, 5, time_thresh, -1)==0,
            get_time_ind(t_arr, 5, time_thresh,  1)==(t_arr.size - 1)
        ]
    
    return all(tests)
