#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal, binomial
from itertools import combinations, product
from functools import partial
import os
import pathlib

'''
Generate data
Adapted from https://github.com/akrhea/CIFRank/blob/master/src/Expected_Kendall_Tau.ipynb
Data-generating process adapted to match mv_m2

Needs to be debugged
'''

def rescale(arr, new_min=0, new_max=1):
    old_min = arr.min()
    old_max = arr.max()
    return ((arr-old_min)/(old_max-old_min))*(new_max-new_min)+new_min

def gen_A(seed, M, prob_priv=0.6):
    '''
    Function to generate race (A)
    A has no parent nodes
    '''
    
    # Set random seed for race generation
    np.random.seed(seed)
    
    # Generate race with binomial (only 2 races)
    # White (privileged) encoded as 1, Black encoded as 0
    return binomial(n=1, p=prob_priv, size=M)

def gen_Err(seed, M, mu=0, sd=1):
    '''
    Function to generate noise node (Epsilon-X or Epsilon-Y)
    Error nodes have no parent nodes
    
    Note: "epsilon," "error," and "noise" are used interchangeably
    '''
    
    # Set random seed for noise generation
    np.random.seed(seed)
    
    # Noise is Gaussian
    return normal(loc=mu, scale=sd, size=M)

def gen_X(seed, a, err_input, err=None,
          base_mu_0=-1, base_sd_0=1, 
          base_mu_1=0, base_sd_1=0.5):
    '''
    Function to generate LSAT scores (X)
    A is parent of X
    
    If err_input==False, then A is the only parent of X
    If err_input==True, then Epsilon-X is also a parent of X

    base_mu_0 and base_sd_0 control parameters for white (race=1)
    base_mu_1 and base_sd_1 control parameters for Black (race=0)
    '''
    
    # Set random seed for LSAT score generation
    np.random.seed(seed)

    # bundle parameters into lists
    base_mus = [base_mu_0, base_mu_1]
    base_sds = [base_sd_0, base_sd_1]
    
    # draw each base score from normal distribution designated by race
    lsat = [normal(loc=base_mus[x], scale=base_sds[x], size=1)[0] for x in a]
    
    # Add noise if DAG specifies X is child of error node
    if err_input:
        lsat = lsat + err

    # Shift and rescale LSAT score to [120, 180]
    lsat = rescale(lsat, new_min=120, new_max=180)

    return lsat
    
def gen_Y(a, x, err_input, err=None,
          a_weight=0.4, x_weight=0.8):
    '''
    Function to generate 1L GPAs (Y)
    A and X are parents of Y
    
    If err_input==False, then A and X are the only parents of Y
    If err_input==True, then Epsilon-Y is also a parent of Y

    Does not require random seed, 
    because Y is linear function of A and X

    a_weight determines linear coefficient on a
    x_weight determines linear coefficient on x
    '''

    # Calculate GPA from race (A) and LSAT (X)
    gpa = a_weight*a + x_weight*x
    
    # Add noise if DAG specifies Y is child of error node
    if err_input:
        gpa = gpa + err
        
    # Rescale GPA to [0, 4]
    gpa = rescale(gpa, new_min=0, new_max=4)
    return gpa

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

def gen_data(a_seed, y_err_seed, x_err_seed, x_seed, rank_seed,# random seeds can be set seperately
             x_err_input, y_err_input, # which nodes receive noise as input (X and/or Y)
             M=10, # number of rows in dataset
             x_err_mu=0, x_err_sd=1, # X-noise settings
             y_err_mu=0, y_err_sd=1, # Y-noise settings
             prob_priv=0.6, # race setting
             x_base_mu_0=-1, x_base_sd_0=1, # lsat settings
             x_base_mu_1=0, x_base_sd_1=0.5, # more lsat settings
             y_a_weight=0.4, y_x_weight=0.8, # gpa settings
             normalize=True, # whether to rescale X and Y to [0,1]
             observed=True, # whether to drop unobserved noise columns in output
             save=False, # whether to save data to CSV
             output_filepath=None): # filepath for saving to CSV
    '''
    Function to generate dataset: A, Epsilon-X, Epsilon-Y, X, and Y
    Returns dataframe including rank of Y (allows ties)
    '''
    
    # Generate race node (A)
    a = gen_A(seed=a_seed, M=M, prob_priv=prob_priv)
    
    if y_err_input:
        # Generate noise node to be parent of Y (Epsilon-Y)
        y_err = gen_Err(seed=y_err_seed, mu=y_err_mu, sd=y_err_sd, M=M)
    else:
        y_err=None
        
    if x_err_input:
        # Generate noise node to be parent of X (Epsilon-X)
        x_err = gen_Err(seed=x_err_seed, mu=x_err_mu, sd=x_err_sd, M=M)
    else:
        x_err=None
    
    # Generate LSAT score node (X)
    x = gen_X(seed=x_seed, a=a, err_input = x_err_input, err=x_err,
              base_mu_0=x_base_mu_0, base_sd_0=x_base_sd_0,
              base_mu_1=x_base_mu_1, base_sd_1=x_base_sd_1)

    # Rescale X to [0,1] if normalize is set to True
    if normalize:
        x = rescale(x)
    
    # Generate first-year GPA node (Y)
    y = gen_Y(a=a, x=x, err_input = y_err_input, err=y_err,
              a_weight=y_a_weight, x_weight=y_x_weight)
    
    # Rescale Y to [0,1] if normalize is set to True
    if normalize:
        y = rescale(y)

    # Compile columns into dataframe
    data = pd.DataFrame({'a':a, 'x_err': x_err, 'y_err': y_err, 'x':x, 'y':y})
    
    # Calculate rank
    data['rank'] = calc_rank(rank_seed, y)

    # Drop unobserved noise nodes if observed set to True
    if observed:
        data.drop(columns=['x_err', 'y_err'], inplace=True)
    
    # Save to CSV if save set to True
    if save:
        data.to_csv(output_filepath)

    return(data)


def gen_data_and_sample_noise(n_runs, # number of re-rankings to sample
                                x_err_input, y_err_input, # whether X and Y have noise parents
                                M=10, # number of rows in dataset
                                a_seed=0, # seed for race (all other seeds based on this)
                                x_err_mu=0, x_err_sd=1, # X-noise settings
                                y_err_mu=0, y_err_sd=1, # Y-noise settings
                                prob_priv=0.6, # race setting
                                x_base_mu_0=-1, x_base_sd_0=1, # lsat settings
                                x_base_mu_1=0, x_base_sd_1=0.5, # more lsat settings
                                y_a_weight=0.4, y_x_weight=0.8, # gpa settings
                                normalize=True, # whether to rescale X and Y to [0,1])
                                output_dir='default', # folder within out/synthetic_data/stability/
                                data_filename='unobserved_samp_1.csv', # name of original dataset file
                                save_rankings=False, # whether to save rankings to CSV
                                rankings_filename='unobserved_samp_1.csv'): # name of rankings file
                                
    '''
    Generate original dataset and save to CSV
    Generate n_runs additional datasets by re-sampling the noise
    '''

    # Set initial seeds
    x_seed=a_seed+1
    rank_seed=a_seed+2
    x_err_seed=n_runs+1+rank_seed # will not be used unless x_err_input==True
    y_err_seed=(n_runs+1)*2+rank_seed # will not be used unless y_err_input==True

    # Set output filepaths
    base_output_dir = pathlib.Path(__file__).parents[2] / 'out' / 'synthetic_data' / 'stability' / output_dir
    data_output_filepath = base_output_dir / 'data' / data_filename
    rankings_output_filepath = base_output_dir / 'rankings' / rankings_filename

    # Create partial data generation function
    # Include all params that will remain constant during noise sampling
    gen_data_partial = partial(gen_data, M=M, 
                                a_seed=a_seed, x_seed=x_seed, 
                                x_err_input=x_err_input, y_err_input=y_err_input,
                                x_err_mu=x_err_mu, x_err_sd=x_err_sd, 
                                y_err_mu=y_err_mu, y_err_sd=y_err_sd,
                                prob_priv=prob_priv, 
                                normalize=normalize, 
                                observed=False, # Do not drop unobserved error columns
                                x_base_mu_0=x_base_mu_0, x_base_sd_0=x_base_sd_0,
                                x_base_mu_1=x_base_mu_1, x_base_sd_1=x_base_sd_1,
                                y_a_weight=y_a_weight, y_x_weight=y_x_weight)

    # Generate baseline dataset with initial seeds and save to CSV
    data = gen_data_partial(y_err_seed=y_err_seed,  x_err_seed=x_err_seed, rank_seed=rank_seed,
                            save=True, output_filepath=data_output_filepath)
    
    # Generate additional datasets
    for i in range(n_runs):

        # Increment noise seeds
        rank_seed +=1
        x_err_seed +=1
        y_err_seed +=1
        
        # Generate dataset with non-constant seeds incremented and do not save to CSV
        sim = gen_data_partial(y_err_seed=y_err_seed,  x_err_seed=x_err_seed, rank_seed=rank_seed,
                                save=False)
        
        # Add this ranking to the original dataframe
        data['rank_'+str(i+1)] = sim['rank']

    # Save rankings to CSV if save_rankings set to True
    if save_rankings:
        data.to_csv(rankings_output_filepath)
    
    return data


def resample_noise_from_data(n_runs, # number of re-rankings to sample
                            orig_data, # original dataset
                            x_err_input, y_err_input, # whether X and Y have noise parents
                            x_err_mu=0, x_err_sd=1, # X-Noise settings
                            y_err_mu=0, y_err_sd=1, # Y-Noise settings
                            y_a_weight=0.4, y_x_weight=0.8, # GPA settings
                            rank_seed=2, # initial random seed for rank (other noise seeds based on this)
                            save=False, # whether to save rankings to CSV
                            output_filepath=None): # filepath for saving to CSV

    '''
    Takes in observed dataset with columns A, X, Y, and rank
    Generate dataset and sample n_runs re-rankings derived from noise distribution
    Uses default settings from gen_data
    '''

    # Set initial seeds
    x_err_seed=n_runs+1+rank_seed # will not be used unless x_err_input==True
    y_err_seed=(n_runs+1)*2+rank_seed # will not be used unless y_err_input==True
    
    # get number of rows from original dataset
    M = len(orig_data)

    # Generate additional rankings by re-sampling noise distribution
    for i in range(0, n_runs):

        # Increment noise seeds
        rank_seed +=1
        x_err_seed +=1
        y_err_seed +=1

        if x_err_input:
            # Generate new noise node to be parent of X (Epsilon-X)
            x_err = gen_Err(seed=x_err_seed, mu=x_err_mu, sd=x_err_sd, M=M)
            new_x = orig_data.x.values + x_err
        else:
            new_x = orig_data.x.values

        if y_err_input:
            # Generate new noise node to be parent of Y (Epsilon-Y)
            y_err = gen_Err(seed=y_err_seed, mu=y_err_mu, sd=y_err_sd, M=M)
        else:
            y_err = None

        new_y = gen_Y(orig_data.a.values, new_x, 
                      err_input=y_err_input, err=y_err,
                      a_weight=y_a_weight, x_weight=y_x_weight)

        new_rank = calc_rank(rank_seed, new_y)
        
        # Add this ranking to the original dataframe
        orig_data['rank_'+str(i+1)] = new_rank

    # Save rankings to CSV if save set to True
    if save:
        orig_data.to_csv(output_filepath)

    return orig_data

def gen_data_and_resample_noise(n_runs, # number of re-rankings to sample
                                x_err_input, y_err_input, # which nodes receive noise as input (X and/or Y)
                                M=10, # number of rows in dataset
                                a_seed=0, # seed for race (all other seeds based on this)    
                                x_err_mu=0, x_err_sd=1, # X-noise settings
                                y_err_mu=0, y_err_sd=1, # Y-noise settings
                                prob_priv=0.6, # race setting
                                x_base_mu_0=-1, x_base_sd_0=1, # lsat settings
                                x_base_mu_1=0, x_base_sd_1=0.5, # more lsat settings
                                y_a_weight=0.4, y_x_weight=0.8, # gpa settings
                                normalize=True, # whether to rescale X and Y to [0,1]):
                                output_dir='default', # folder within out/synthetic_data/stability/ 
                                data_filename='observed_samp_1.csv', # name of original dataset file
                                save_rankings=True, # whether to save rankings to CSV
                                rankings_filename='observed_samp_1.csv'): # name of rankings file

    '''
    Generate original observed dataset: A, X, Y, and rank
    Save dataset to CSV
    Resample rankings
    '''

    # Set initial seeds
    x_seed=a_seed+1
    rank_seed=a_seed+3
    x_err_seed=n_runs+1+rank_seed # will not be used unless x_err_input==True
    y_err_seed=(n_runs+1)*2+rank_seed # will not be used unless y_err_input==True

    # Set output filepaths
    base_output_dir = pathlib.Path(__file__).parents[2] / 'out' / 'synthetic_data' / 'stability' / output_dir
    data_output_filepath = base_output_dir / 'data' / data_filename
    rankings_output_filepath = base_output_dir / 'rankings' / rankings_filename

    # Generate original dataset with initial seeds
    orig_data = gen_data(y_err_seed=y_err_seed,  x_err_seed=x_err_seed,
                            x_seed=x_seed, a_seed=a_seed, rank_seed=rank_seed,
                            x_err_input=x_err_input, y_err_input=y_err_input, M=M,
                            x_err_mu=x_err_mu, x_err_sd=x_err_sd, 
                            y_err_mu=y_err_mu, y_err_sd=y_err_sd,
                            prob_priv=prob_priv, 
                            normalize=normalize, 
                            observed=True, # Drop unobserved error columns
                            x_base_mu_0=x_base_mu_0, x_base_sd_0=x_base_sd_0,
                            x_base_mu_1=x_base_mu_1, x_base_sd_1=x_base_sd_1,
                            y_a_weight=y_a_weight, y_x_weight=y_x_weight,
                            save=True, output_filepath=data_output_filepath) # Save dataset to CSV

    # Use orig_data to resample rankings from noise distribution
    rankings_data = resample_noise_from_data(n_runs=n_runs,
                                            orig_data=orig_data, 
                                            x_err_input=x_err_input, y_err_input=y_err_input, 
                                            x_err_mu=x_err_mu, x_err_sd=x_err_sd, 
                                            y_err_mu=y_err_mu, y_err_sd=y_err_sd, 
                                            y_a_weight=y_a_weight, y_x_weight=y_x_weight, 
                                            rank_seed=rank_seed,
                                            save=save_rankings, 
                                            output_filepath=rankings_output_filepath)

    return rankings_data

def sampling_distribution(s_samples, # number of original dataset samples
                            n_runs, # number of re-rankings to sample from noise distribution of each sample
                            x_err_input, y_err_input, # which nodes receive noise as input (X and/or Y)
                            M=10, # number of rows in dataset
                            a_seed=0, # initial seed for race (all other seeds based on this)
                            x_err_mu=0, x_err_sd=1, # X-noise settings
                            y_err_mu=0, y_err_sd=1, # Y-noise settings
                            prob_priv=0.6, # race setting
                            x_base_mu_0=-1, x_base_sd_0=1, # lsat settings
                            x_base_mu_1=0, x_base_sd_1=0.5, # more lsat settings
                            y_a_weight=0.4, y_x_weight=0.8, # gpa settings
                            normalize=True, # whether to rescale X and Y to [0,1])
                            output_dir='default'): # folder within out/synthetic_data/stability/

    '''
    Generate s_samples of original dataset, save each to CSV
    For each dataset, sample rankings from noise distribution and save to CSV
    '''

    # Create partial function for sampling noise distribution
    # Include all parameters which will remain constant
    gen_data_and_resample_noise_partial = partial(gen_data_and_resample_noise,
                                                    n_runs=n_runs, M=M,
                                                    x_err_input=x_err_input, y_err_input=y_err_input,
                                                    x_err_mu=x_err_mu, x_err_sd=x_err_sd,
                                                    y_err_mu=y_err_mu, y_err_sd=y_err_sd,
                                                    prob_priv=prob_priv,
                                                    x_base_mu_0=x_base_mu_0, x_base_sd_0=x_base_sd_0,
                                                    x_base_mu_1=x_base_mu_1, x_base_sd_1=x_base_sd_1,
                                                    y_a_weight=y_a_weight, y_x_weight=y_x_weight,
                                                    normalize=normalize,
                                                    output_dir=output_dir,
                                                    save_rankings=True)
    
    # Generate s_samples of original dataset and sample rankings from noise distribution of each
    for i in range(s_samples):
        gen_data_and_resample_noise_partial(a_seed=a_seed, 
                                            data_filename='observed_samp_{}.csv'.format(i+1),
                                            rankings_filename='observed_samp_{}.csv'.format(i+1))

        # Increment race seed (which will control all other seeds)
        a_seed += (n_runs+1)*3+2

    return