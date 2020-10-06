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