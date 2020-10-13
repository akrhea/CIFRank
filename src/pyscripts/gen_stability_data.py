#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal, binomial
from itertools import combinations, product
from functools import partial
import os, pathlib, argparse, sys
from utils.stability_utils import calc_rank, rescale

'''
Alene Rhea, October 2020

This script generates s_samples of a synthetic dataset containing A, X, and Y.
For each sample, n_runs of the noise distribution will be sampled.

Adapted from https://github.com/akrhea/CIFRank/blob/master/src/Expected_Kendall_Tau.ipynb
Data-generating process adapted to approximately match Ke Yang's mv_m2
'''



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
          mu_0=-1, sd_0=1, 
          mu_1=0, sd_1=0.5):
    '''
    Function to generate LSAT scores (X)
    A is parent of X
    
    If err_input==False, then A is the only parent of X
    If err_input==True, then Epsilon-X is also a parent of X

    mu_0 and sd_0 control parameters for white (race=1)
    mu_1 and sd_1 control parameters for Black (race=0)
    '''
    
    # Set random seed for LSAT score generation
    np.random.seed(seed)

    # bundle parameters into lists
    mus = [mu_0, mu_1]
    sds = [sd_0, sd_1]
    
    # draw each base score from normal distribution designated by race
    lsat = [normal(loc=mus[x], scale=sds[x], size=1)[0] for x in a]
    
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

def gen_data(a_seed, y_err_seed, x_err_seed, x_seed, # random seeds can be set seperately
             x_err_input, y_err_input, # which nodes receive noise as input (X and/or Y)
             unmeasured_confounding, # whether X and Y share a noise parent
             M, # number of rows in dataset
             x_err_mu=0, x_err_sd=1, # X-noise settings
             y_err_mu=0, y_err_sd=1, # Y-noise settings
             prob_priv=0.6, # race setting
             x_mu_0=-1, x_sd_0=1, # lsat settings
             x_mu_1=0, x_sd_1=0.5, # more lsat settings
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

    if unmeasured_confounding:
        # Ensure X and Y both have error parent nodes
        x_err_input=True
        y_err_input=True

        # Use same seed for both errors
        y_err_seed=x_err_seed
    
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
    prescale_x = gen_X(seed=x_seed, a=a, err_input = x_err_input, err=x_err,
                        mu_0=x_mu_0, sd_0=x_sd_0,
                        mu_1=x_mu_1, sd_1=x_sd_1)

    # Rescale X to [0,1] if normalize is set to True
    if normalize:
        x = rescale(prescale_x)
    else:
        x = prescale_x
    
    # Generate first-year GPA node (Y)
    prescale_y = gen_Y(a=a, x=x, err_input = y_err_input, err=y_err,
                        a_weight=y_a_weight, x_weight=y_x_weight)
    
    # Rescale Y to [0,1] if normalize is set to True
    if normalize:
        y = rescale(prescale_y)
    else:
        y = prescale_y

    # Compile columns into dataframe
    data = pd.DataFrame({'a':a, 'x_err': x_err, 'prescale_x': prescale_x, 'x':x, 
                                'y_err': y_err, 'prescale_y': prescale_y, 'y':y})

    # Drop unobserved noise nodes if observed set to True
    if observed:
        data.drop(columns=['x_err', 'y_err', 'prescale_x', 'prescale_y'], inplace=True)
    
    # Save to CSV if save set to True
    if save:
       data.to_csv(output_filepath, index=False)

    return data


def gen_data_and_sample_noise(n_runs, # number of re-rankings to sample
                                x_err_input, y_err_input, # whether X and Y have noise parents
                                unmeasured_confounding, # whether X and Y share a noise parent
                                M, # number of rows in dataset
                                a_seed=0, # seed for race (all other seeds based on this)
                                x_err_mu=0, x_err_sd=1, # X-noise settings
                                y_err_mu=0, y_err_sd=1, # Y-noise settings
                                prob_priv=0.6, # race setting
                                x_mu_0=-1, x_sd_0=1, # lsat settings
                                x_mu_1=0, x_sd_1=0.5, # more lsat settings
                                y_a_weight=0.4, y_x_weight=0.8, # gpa settings
                                normalize=True, # whether to rescale X and Y to [0,1])
                                observed=False, # whether to drop unobserved columns
                                output_dir='default', # folder within out/synthetic_data/
                                data_filename='unobserved_samp_1.csv', # name of original dataset file
                                save_rankings=False, # whether to save rankings to CSV
                                rankings_filename='unobserved_samp_1.pkl'): # name of rankings file
                                
    '''
    Generate original dataset and save to CSV
    Generate n_runs additional datasets using all same seeds except noise seed
    Save original rank and additional n_runs ranks to pickle

    This implementation is favored for synthetic data experiments
    See gen_data_and_resample_noise for favored implementation for real data experiments
    '''

    # Set initial seeds
    x_seed=a_seed+1
    rank_seed=a_seed+2
    x_err_seed=n_runs+1+rank_seed # will not be used unless x_err_input==True
    y_err_seed=(n_runs+1)*2+rank_seed # will not be used unless y_err_input==True

    # Set output filepaths
    base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    base_output_dir = base_repo_dir / 'out' / 'synthetic_data' / output_dir
    data_output_filepath = base_output_dir / 'data' / data_filename
    rankings_output_filepath = base_output_dir / 'noise_rankings' / rankings_filename

    # Create partial data generation function
    # Include all params that will remain constant during noise sampling
    gen_data_partial = partial(gen_data, M=M, 
                                a_seed=a_seed, x_seed=x_seed, 
                                x_err_input=x_err_input, y_err_input=y_err_input,
                                unmeasured_confounding=unmeasured_confounding,
                                x_err_mu=x_err_mu, x_err_sd=x_err_sd, 
                                y_err_mu=y_err_mu, y_err_sd=y_err_sd,
                                prob_priv=prob_priv, 
                                normalize=normalize, 
                                observed=observed,
                                x_mu_0=x_mu_0, x_sd_0=x_sd_0,
                                x_mu_1=x_mu_1, x_sd_1=x_sd_1,
                                y_a_weight=y_a_weight, y_x_weight=y_x_weight)

    # Generate baseline dataset with initial seeds and save to CSV
    data = gen_data_partial(y_err_seed=y_err_seed, x_err_seed=x_err_seed,
                            save=True, output_filepath=data_output_filepath)
    
    # Create dataframe to hold rank permutations
    rankings = pd.DataFrame()

    # Get original rank from original data
    rankings['rank'] = calc_rank(rank_seed, data['y'])

    # Generate additional datasets
    for i in range(n_runs):

        # Increment noise seeds
        rank_seed +=1
        x_err_seed +=1
        y_err_seed +=1
        
        # Generate dataset with non-constant seeds incremented and do not save to CSV
        sim = gen_data_partial(y_err_seed=y_err_seed, x_err_seed=x_err_seed, save=False)
        
        # Add this ranking to the original dataframe
        rankings['rank_'+str(i+1)] = calc_rank(rank_seed, sim['y'])

    # Save rankings to CSV if save_rankings set to True
    if save_rankings:
       rankings.to_pickle(rankings_output_filepath, compression='gzip')
    
    return (data, rankings)


def resample_noise_from_data(n_runs, # number of re-rankings to sample
                            orig_data, # original dataset
                            x_err_input, y_err_input, # whether X and Y have noise parents
                            unmeasured_confounding, # whether X and Y share a noise parent
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

    # Set initial seed
    x_err_seed=n_runs+1+rank_seed # will not be used unless x_err_input==True

    if unmeasured_confounding:
        # Ensure X and Y both have error parent nodes
        x_err_input=True
        y_err_input=True

        # Use same seed for both errors
        y_err_seed=x_err_seed

    else:
        y_err_seed=(n_runs+1)*2+rank_seed # will not be used unless y_err_input==True
    
    # get number of rows from original dataset
    M = len(orig_data)

    # Create dataframe to hold rank permutations
    rankings = pd.DataFrame()

    # Get original rank from original data
    rankings['rank'] = calc_rank(rank_seed, orig_data['y'])

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
        rankings['rank_'+str(i+1)] = new_rank

    # Save rankings to CSV if save set to True
    if save:
       rankings.to_pickle(output_filepath, compression='gzip')

    return rankings

def gen_data_and_resample_noise(n_runs, # number of re-rankings to sample
                                x_err_input, y_err_input, # which nodes receive noise as input (X and/or Y)
                                unmeasured_confounding, # whether X and Y share a noise parent
                                M, # number of rows in dataset
                                a_seed=0, # seed for race (all other seeds based on this)    
                                x_err_mu=0, x_err_sd=1, # X-noise settings
                                y_err_mu=0, y_err_sd=1, # Y-noise settings
                                prob_priv=0.6, # race setting
                                x_mu_0=-1, x_sd_0=1, # lsat settings
                                x_mu_1=0, x_sd_1=0.5, # more lsat settings
                                y_a_weight=0.4, y_x_weight=0.8, # gpa settings
                                normalize=True, # whether to rescale X and Y to [0,1])
                                observed=True, # whether to drop unobserved columns
                                output_dir='default', # folder within out/synthetic_data/
                                data_filename='observed_samp_1.csv', # name of original dataset file
                                save_rankings=True, # whether to save rankings to CSV
                                rankings_filename='observed_samp_1.pkl'): # name of rankings file

    '''
    Generate original observed dataset: A, X, Y, and rank
    Save dataset to CSV
    Resample rankings by sampling from noise distribution

    There is an outstanding question of how scaling is handled in this function

    This implementation is favored for real data experiments
    See gen_data_and_sample_noise for favored implementation for synthetic data experiments
    '''

    # Set initial seeds
    x_seed=a_seed+1
    rank_seed=a_seed+2
    x_err_seed=n_runs+1+rank_seed # will not be used unless x_err_input==True
    y_err_seed=(n_runs+1)*2+rank_seed # will not be used unless y_err_input==True

    # Set output filepaths
    base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    base_output_dir = base_repo_dir / 'out' / 'synthetic_data'/ output_dir
    data_output_filepath = base_output_dir / 'data' / data_filename
    rankings_output_filepath = base_output_dir / 'noise_rankings' / rankings_filename

    # Generate original dataset with initial seeds
    orig_data = gen_data(y_err_seed=y_err_seed,  x_err_seed=x_err_seed,
                            x_seed=x_seed, a_seed=a_seed, rank_seed=rank_seed,
                            x_err_input=x_err_input, y_err_input=y_err_input, 
                            unmeasured_confounding=unmeasured_confounding, M=M,
                            x_err_mu=x_err_mu, x_err_sd=x_err_sd, 
                            y_err_mu=y_err_mu, y_err_sd=y_err_sd,
                            prob_priv=prob_priv, 
                            normalize=normalize, 
                            observed=observed,
                            x_mu_0=x_mu_0, x_sd_0=x_sd_0,
                            x_mu_1=x_mu_1, x_sd_1=x_sd_1,
                            y_a_weight=y_a_weight, y_x_weight=y_x_weight,
                            save=True, output_filepath=data_output_filepath) # Save dataset to CSV

    # Use orig_data to resample rankings from noise distribution
    rankings_data = resample_noise_from_data(n_runs=n_runs,
                                            orig_data=orig_data, 
                                            x_err_input=x_err_input, y_err_input=y_err_input,
                                            unmeasured_confounding=unmeasured_confounding, 
                                            x_err_mu=x_err_mu, x_err_sd=x_err_sd, 
                                            y_err_mu=y_err_mu, y_err_sd=y_err_sd, 
                                            y_a_weight=y_a_weight, y_x_weight=y_x_weight, 
                                            rank_seed=rank_seed,
                                            save=save_rankings, 
                                            output_filepath=rankings_output_filepath)

    return (orig_data, rankings_data)

def sampling_distribution(args):
                            
    '''
    Generate s_samples of original dataset, save each to CSV
    For each dataset, sample rankings from noise distribution and save to CSV

    Required Arguments
        s_samples:                  number of original dataset samples
        n_runs:                     number of re-rankings to sample from noise distribution of each sample
        m_rows:                     number of rows in dataset
        x_err_input:                whether X node receives noise as input
        y_err_input:                whether X node receives noise as input
        unmeasured_confounding:     whether X and Y share a noise parent


    Optional Arguments
        output_dir: folder within out/synthetic_data/ to store output 
                        default = 'default'
        seed:       initial seed for race (all other seeds based on this) 
                        default = 0
        x_err_mu:   xpected value of X's noise parent
                        default = 0
        x_err_sd:   std dev of X's noise parent
                        default = 1
        y_err_mu:   expected value of Y's noise parent
                        default = 0
        y_err_sd:   std dev of Y's noise parent
                        default = 1
        prov_priv:  probability of privileged race (A=1)
                        default = 0.6
        x_mu_0:     expected value of X for A=0
                        default = -1
        x_sd_0:     std dev of X for A=0
                        default = 1 
        x_mu_1:     expected value of X for A=1
                        default = 0
        x_sd_1:     std dev of X for A=1
                        default = 0.5
        y_a_weight: coefficient of A value on Y      
                        default = 0.4
        y_x_weight: coefficient of X value on Y      
                        default = 0.8
        normalize:  whether to rescale X and Y nodes to [0,1]
                        default = True
        observed:  whether to drop unobserved columns from data
                        default = True
    '''

    # Create partial function for sampling noise distribution
    # Include all parameters which will remain constant
    gen_data_and_sample_noise_partial = partial(gen_data_and_sample_noise,
                                                    n_runs=args.n_runs, 
                                                    M=args.m_rows,
                                                    x_err_input=args.x_err_input, 
                                                    y_err_input=args.y_err_input,
                                                    unmeasured_confounding=args.unmeasured_confounding,
                                                    x_err_mu=args.x_err_mu, 
                                                    x_err_sd=args.x_err_sd,
                                                    y_err_mu=args.y_err_mu, 
                                                    y_err_sd=args.y_err_sd,
                                                    prob_priv=args.prob_priv,
                                                    x_mu_0=args.x_mu_0, 
                                                    x_sd_0=args.x_sd_0,
                                                    x_mu_1=args.x_mu_1, 
                                                    x_sd_1=args.x_sd_1,
                                                    y_a_weight=args.y_a_weight, 
                                                    y_x_weight=args.y_x_weight,
                                                    normalize=args.normalize,
                                                    observed=args.observed,
                                                    output_dir=args.output_dir,
                                                    save_rankings=True)
    seed = args.seed
    # Generate s_samples of original dataset and sample rankings from noise distribution of each
    for i in range(args.s_samples):
        gen_data_and_sample_noise_partial(a_seed=seed, 
                                            data_filename='samp_{}.csv'.format(i+1),
                                            rankings_filename='samp_{}.pkl'.format(i+1))

        # Increment race seed (which will control all other seeds)
        seed += (args.n_runs+1)*3+2

        print('Finished generating sample {} of {}'.format(i+1, args.s_samples))

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Synthetic Data Generation for Stability Analysis")

    # Required arguments
    parser.add_argument("--s_samples", type=int, help='number of samples to draw for sampling distribution')
    parser.add_argument("--n_runs", type=int, help='number of runs for error distribution')
    parser.add_argument("--m_rows", type=int, help='number of rows in dataset')

    # Boolean arguments (add flag to set to true)
    parser.add_argument("--x_err_input", action='store_true', 
                        help='boolean flag indicating X has an error parent')

    parser.add_argument("--y_err_input", action='store_true', 
                        help='boolean flag indicating Y has an error parent')

    parser.add_argument("--unmeasured_confounding", action='store_true', 
                        help='boolean flag indicating X and Y share a noise parent')

    parser.add_argument("--do_not_normalize", action='store_true', 
                        help='boolean flag indicating X and Y will not be scaled to [0,1]')

    parser.add_argument("--unobserved", action='store_true', 
                        help='boolean flag indicating unobserved columns will not be dropped from data')

    # Optional arguments
    parser.add_argument("--seed", type=int, default=0, help='Initial seed for race sample; basis of all other seeds')
    parser.add_argument("--output_dir", type=str, default='default',  help='folder within out/synthetic_data/ to store output')
    parser.add_argument("--x_err_mu", type=float, default=0)
    parser.add_argument("--x_err_sd", type=float, default=1)
    parser.add_argument("--y_err_mu", type=float, default=0)
    parser.add_argument("--y_err_sd", type=float, default=1)
    parser.add_argument("--prob_priv", type=float, default=0.6)
    parser.add_argument("--x_mu_0", type=float, default=-1)
    parser.add_argument("--x_sd_0", type=float, default=1)
    parser.add_argument("--x_mu_1", type=float, default=0)
    parser.add_argument("--x_sd_1", type=float, default=0.5)
    parser.add_argument("--y_a_weight", type=float, default=0.4)
    parser.add_argument("--y_x_weight", type=float, default=0.8)
    
    args = parser.parse_args()

    # Set normalize argument based on do_not_normalize
    if args.do_not_normalize==True:
        args.normalize=False
    else:
        args.normalize=True

    # Set observed argument based on unobserved
    if args.unobserved==True:
        args.observed=False
    else:
        args.observed=True

    # Ensure X and Y both have error node parents if unmeasured confounding flag is passed
    if args.unmeasured_confounding:
        args.x_err_input=True
        args.y_err_input=True

    sampling_distribution(args)