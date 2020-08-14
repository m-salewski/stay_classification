import numpy as np

def list_intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

def contains(a1,a2,b1,b2):
    """
    Check whether one range contains another
    """    
    return ((a1 >= b1) & (a2 <= b2))

inter_bounds = lambda p1, p2: intersecting_bounds(p1[0],p1[-1],p2[0],p2[-1])
conta_bounds = lambda p1, p2: contains(p1[0],p1[-1],p2[0],p2[-1])


def intersecting_bounds(a1,a2,b1,b2):
    """
    Check whether two ranges intersect
    
    Specifically: b1 < a1 < b2 and a2 > b2
    """
    '''print(a1,a2,b1,b2)
    
    print((a1 >  b1), (a1 <= b2), (a2 >= b2))
    print((a1 >= b1), (a1 <  b2), (a2 >  b2))'''
    
    cond1 = (((a1 >  b1) & (a1 < b2)) & (a2 > b2)) | \
             (((a1 > b1) & (a1 < b2)) & (a2 > b2))
    return cond1


def split_lists1( list1, list2): 
    
    """
    Usage:
    split_list(range(10), list(range(4,7)))
    ([0, 1, 2, 3], [4, 5, 6], [7, 8, 9])
    """
    uneql = lambda l1, l2: ([e for e in l2 if e < min(l1)], l1, [e for e in l2 if e > max(l1)])
    equal = lambda l1, l2: ([e for e in l1 if e < min(l2)], \
                            [e for e in l2 if ((e >= min(l2)) & (e <= max(l1)))], \
                            [e for e in l2 if e > max(l1)])
    '''
    if len(list1) < len(list2):
        return uneql(list1, list2)
    
    elif len(list1) < len(list2):
        return uneql(list2, list1)
    '''
    # This is version 2
    #else:
    if min(list1) < min(list2):
        return equal(list1, list2)
    else:
        return equal(list2, list1)


def split_lists2( list1, list2): 
    
    if list1 == list2: 
        return list1
    
    print(f"l1: {list1},\nl2: {list2}")
    print()
    """
    Usage:
    split_list(range(10), list(range(4,7)))
    ([0, 1, 2, 3], [4, 5, 6], [7, 8, 9])
    """
    emend = lambda l1, l2: ([e for e in l1  if e < min(l2)], [e for e in l2])    
    embeg = lambda l1, l2: ([e for e in l1], [e for e in l2 if e > max(l1)])
    embed = lambda l1, l2: ([e for e in l1 if e < min(l2)], \
                            [e for e in l2 if ((e >= min(l2)) & (e <= max(l1)))], \
                            [e for e in l1 if e > max(l2)])
        
    equal = lambda l1, l2: ([e for e in l1 if e < min(l2)], \
                            [e for e in l2 if ((e >= min(l2)) & (e <= max(l1)))], \
                            [e for e in l2 if e > max(l1)])        
    
    if inter_bounds(list1,list2):
        # l1 \cap l2 \neq \emptyset, l2.min < l1.min
        print('l1 intersects l2')
        return equal(list2,list1)
        
    elif inter_bounds(list2,list1):
        # l1 \cap l2 \neq \emptyset, l1.min < l2.min
        print('l2 intersects l1')
        return equal(list1,list2)
        
    elif conta_bounds(list1,list2):
        # l1 \subset l2
        if list1[-1] == list2[-1]:
            print('l2 embeds l1, shares end')
            return emend(list2,list1)
        
        elif list1[0] == list2[0]:
            print('l2 embeds l1, shares begin')
            return embeg(list1,list2)
        
        else:
            print('l2 embeds l1')             
            return embed(list2,list1)
    
    else:
        # l1 \subset l1
        if list1[-1] == list2[-1]:
            print('l1 embeds l2, shares end')
            return emend(list1,list2)
        
        elif list1[0] == list2[0]:
            print('l1 embeds l2, shares begin')
            return embeg(list2,list1)
        
        else:
            print('l1 embeds l2')             
            return embed(list1,list2)
        
        
def contiguous_sublists(lst1, lst2): 
    """
    Split intersecting and interleaved lists 
    into maximal contiguous sublists
    """
    #print(lst1, lst2)
    # Gather all unique elements together and order
    lst3 = sorted(list(set(lst1+lst2)))

    # Classifying function according to origin and intersections
    def get_flag(e):
        if (e in lst1) & (e in lst2):
            return 0
        elif (e in lst1):
            return 1
        else:
            return 2
    
    # output list
    all_lists = []
    
    # Initializations
    e = lst3[0]    
    flag = get_flag(e)
    lst = [e]
    
    # Walk through all elements
    for e in lst3[1:]:
        flag_ = get_flag(e)
        
        if flag == flag_:
            lst.append(e)
        else:
            all_lists.append(lst)
            lst = [e]
            flag = flag_
    
    # close the last list
    all_lists.append(lst)        
    
    # Sort the lists (maybe unnecessary)
    #all_lists = sorted([l for l in all_lists if l != []])
    
    return all_lists     


def separate_clusters_hier(clusters, verbose=False):
    """
    
    Test:
    clusts = [list(range(2,4)), list(range(8)), list(range(4,8)), list(range(6,10))]
    separate_clusters_hier(clusts)
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    """    
    # Check if any clusters intersect
    new_clusts = clusters.copy()
    
    result, m, n = check_intersecting_clusters_inds(new_clusts)
    
    if result:
        if verbose: print(m, n)
    else:
        return clusters

    while result:
        
        if verbose: print_clusts(new_clusts)
        if verbose: print(len(new_clusts))
        c1 = new_clusts.pop(m)
        if m < n: 
            n-=1
        c2 = new_clusts.pop(n)

        if verbose: print(len(new_clusts))
        if verbose: print(f"{m:4d}: [{c1[0]:4d},{c1[-1]:4d}] and [{c2[0]:4d},{c2[-1]:4d}]")

        '''sc1 = set(c1)
        sc2 = set(c2)

        intersexion = sc1.intersection(sc2)

        sc1_diff = sc1.difference(intersexion)
        sc2_diff = sc2.difference(intersexion)                
        '''
        
        sc1_diff, intersec, sc2_diff = split_list(c1,c2)        
        
        #sc1_diff = list(sc1_diff)
        if len(sc1_diff) > 0:
            new_clusts.append(sc1_diff)
        if verbose: print("len(sc1_diff)",len(sc1_diff))
        
        #intersexion = list(intersexion)
        if len(intersec) > 0:
            new_clusts.append(intersec)
        if verbose: print("len(intersexion)",len(intersec))
        
        #sc2_diff = list(sc2_diff)
        if len(sc2_diff) > 0:
            new_clusts.append(sc2_diff)
        if verbose: print("len(sc2_diff)",len(sc2_diff))
        
        if verbose: print(len(new_clusts))
        if verbose: print_clusts(new_clusts) 
        
        result, m, n = check_intersecting_clusters_inds(new_clusts)
        if verbose: print(result,m,n)
        if verbose: print()
        
    return new_clusts

