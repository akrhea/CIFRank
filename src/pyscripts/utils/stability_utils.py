#!/usr/bin/env python3

import numpy as np

def get_shared_weight(corr):
    '''
    Taken in correlation between error terms
    Return weight of shared error term

    Based on quadratic formula
    '''

    assert (corr>=0) & (corr<=1), 'err_corr must be between 0 and 1 (inclusive)'
    
    if corr==0.5:
        w = 0.5
    else:
        w = (2*corr-2*np.sqrt(corr-corr**2))/(4*corr-2)
        
    assert (w>=0) & (w<=1), 'Error: weight not between 0 and 1: {}'.format(w)
    
    return abs(w)

def rescale(arr, new_min=0, new_max=1):
    old_min = np.min(arr)
    old_max = np.max(arr)
    return ((arr-old_min)/(old_max-old_min))*(new_max-new_min)+new_min

def calc_rank(seed, y):

    '''
    Function adapted from https://www.geeksforgeeks.org/rank-elements-array/
    Randomly breaks ties using np random seed
    '''

    # Set random seed
    np.random.seed(seed)

    # Initialize rank vector 
    R = [0 for i in range(len(y))] 

    # Create an auxiliary array of tuples 
    # Each tuple stores the data as well as its index in y 
    # T[][0] is the data and T[][1] is the index of data in y
    T = [(y[i], i) for i in range(len(y))] 
    
    # Sort T according to first element 
    T.sort(key=lambda x: x[0], reverse=True)

    # Loop through items in T
    i=0
    while i < len(y): 

        # Get number of elements with equal rank 
        j = i 
        while j < len(y) - 1 and T[j][0] == T[j + 1][0]: 
            j += 1
        n = j - i + 1

        # If there is no tie
        if n==1:
            
            # Get ID of this element
            idx = T[i][1] 
            
            # Set rank
            rank = i+1
            
            # Assign rank
            R[idx] = rank 
            
        # If there is a tie
        if n>1: 
            
            # Create array of ranks to be assigned
            ranks = list(np.arange(i+1, i+1+n)) 
            
            # Randomly shuffle the ranks
            np.random.shuffle(ranks) 
            
            # Create list of element IDs
            ids = [T[i+x][1] for x in range(n)] 
            
            # Assign rank to each element
            for ind, idx in enumerate(ids):
                R[idx] = ranks[ind] 

        # Increment i 
        i += n 
    
    # return rank vector
    return R

def subgroup_rank(rank, groups, which_group):
    group_inds = np.argwhere(groups==which_group).flatten()
    group_ranks = np.array(rank[group_inds])
    return calc_rank(seed=0, y=-1*group_ranks) # will not use seed because no ties