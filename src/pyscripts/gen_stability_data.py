#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal, binomial
from itertools import combinations, product
from functools import partial
import os, pathlib, argparse, sys, copy
from utils.stability_utils import calc_rank, rescale, get_shared_weight
from utils.basic_utils import print_attributes

'''
Alene Rhea, November 2020

This script generates s_samples of a synthetic dataset containing A, X, and Y.
For each sample, n_runs of the noise distribution will be sampled.

Adapted from https://github.com/akrhea/CIFRank/blob/master/src/Expected_Kendall_Tau.ipynb
Data-generating process adapted to approximately match Ke Yang's mv_m2
'''


def gen_R(seed, m_rows, race_prob=[0.5, 0.2, 0.3]):
    '''
    Function to generate race (R)
    R has no parent nodes

    race_prob is a vector of probabilities 
    race_prob[0] : proportion of population that is white
    race_prob[1] : proportion of population that is Black
    race_prob[2] : proportion of population that is Asian
    '''

    # Set random seed for race generation
    np.random.seed(seed)

    race_vec = ['w', 'b', 'a']

    assert sum(race_prob)==1, \
        'race_prob vector must sum to 1'

    assert len(race_prob)==len(race_vec), \
        'race_prob vector must be same length as number of races to generate ({})'.format(len(race_vec))
    
    # White (privileged) encoded as 'w' 
    # Black encoded as 'b'
    # Asian encoded as 'a'
    return np.random.choice(a=race_vec, size=m_rows, p=race_prob)

def gen_S(seed, m_rows, sex_prob=[0.6, 0.4]):
    '''
    Function to generate sex (S)
    S has no parent nodes

    sex_prob is a vector of probabilities 
    sex_prob[0] : proportion of population that is male
    sex_prob[1] : proportion of population that is female
    '''

    # Set random seed for sex generation
    np.random.seed(seed)

    sex_vec = ['m', 'f']

    assert sum(sex_prob)==1, \
        'sex_prob vector must sum to 1'

    assert len(sex_prob)==len(sex_vec), \
        'sex_prob vector must be same length as number of sexes to generate ({})'.format(len(sex_vec))
    
    # Male (privileged) encoded as 'm' 
    # Female encoded as 'f'
    return np.random.choice(a=sex_vec, size=m_rows, p=sex_prob)

def gen_A(r, s):
    '''
    Generate intersectional identity column from race and sex assignments
    e.g. 'am' for Asian male
    '''
    return [''.join(i) for i in zip(*[r,s])]

def gen_Err(seed, m_rows, mu=0, sd=1):
    '''
    Function to generate noise node
    Error nodes have no parent nodes
    
    Note: "epsilon," "error," and "noise" are used interchangeably
    '''
    
    # Set random seed for noise generation
    np.random.seed(seed)
    
    # Noise is Gaussian
    return normal(loc=mu, scale=sd, size=m_rows)

def gen_X(seed, a, err_input, err=None,
          mu_dict={'wm':2, 'bm':-1, 'am':0, 'wf':0, 'bf':-2, 'af':1}, 
          sd_dict={'wm':2, 'bm':0.5, 'am':1, 'wf':1.5, 'bf':1, 'af':1.5}):
    '''
    Function to generate X
    A is parent of X
    
    If err_input==False, then A is the only parent of X
    If err_input==True, then Epsilon-X is also a parent of X
    '''
    
    # Set random seed for X generation
    np.random.seed(seed)

    # draw each base score from normal distribution designated by race
    x = [normal(loc=mu_dict[x], scale=sd_dict[x], size=1)[0] for x in a]
    
    # Add noise if DAG specifies X is child of error node
    if err_input:
        x = x + err

    return x
    
def gen_Y(s, r, x, err_input, err=None,
          s_boost_dict={'m':1, 'f':0}, 
          r_boost_dict={'w':1, 'b':-1, 'a':0},
          x_weight=0.8):
    '''
    Function to generate Y
    S, R, and X are parents of Y
    
    If err_input==False, then S, R, and X are the only parents of Y
    If err_input==True, then Epsilon-Y is also a parent of Y

    Does not require random seed, 
    because Y is linear function of S, R, and X

    x_weight determines linear coefficient on X
    '''

    # Calculate Y from X, S, and R
    y = x_weight*x + np.vectorize(s_boost_dict.__getitem__)(s) \
                   + np.vectorize(r_boost_dict.__getitem__)(r)
    
    # Add noise if DAG specifies Y is child of error node
    if err_input:
        y = y + err

    return y

def gen_data(args):
    
            # sex_seed, race_seed, y_err_seed, x_err_seed, shared_err_seed, x_seed, # random seeds can be set seperately
            #  x_err_input, y_err_input, # which nodes receive noise as input (X and/or Y)
            #  m_rows, # number of rows in dataset
            #  shared_err_mu=0, shared_err_sd=1, # Shared error node settings
            #  x_err_mu=0, x_err_sd=1, # X-noise settings
            #  y_err_mu=0, y_err_sd=1, # Y-noise settings
            #  x_shared_err_weight=0.5, # How much of X noise input comes from shared error node
            #  y_shared_err_weight=0.5, # How much of Y noise input comes from shared error node
             
            #  race_prob=[0.5, 0.2, 0.3], # race setting
            #  x_mu_0=-1, x_sd_0=1, # lsat settings
            #  x_mu_1=0, x_sd_1=0.5, # more lsat settings
            #  y_a_weight=0.4, y_x_weight=0.8, # gpa settings
            #  normalize=True, # whether to rescale X and Y to [0,1]
            #  observed=True, # whether to drop unobserved noise columns in output
            #  save=False, # whether to save data to CSV
            #  output_filepath=None): # filepath for saving to CSV
    '''
    Function to generate dataset: A (race and sex), Epsilon-X, Epsilon-Y, Shared Epsilon, X, and Y
    Returns dataframe including rank of Y (allows ties)
    '''

    if args.check_seeds:
        args.print_seeds()

    assert (0<=args.x_shared_err_weight) & (args.x_shared_err_weight<=1), \
            'x_shared_err_weight must be between 0 and 1 (inclusive)'

    assert (0<=args.y_shared_err_weight) & (args.y_shared_err_weight<=1), \
            'y_shared_err_weight must be between 0 and 1 (inclusive)'
    
    # Generate race node (R)
    r = gen_R(seed=args.race_seed, m_rows=args.m_rows, race_prob=args.race_prob)
    s = gen_S(seed=args.sex_seed, m_rows=args.m_rows, sex_prob=args.sex_prob)
    a = gen_A(r=r, s=s)

    if args.x_err_input or args.y_err_input:
        # Generate noise node to be parent of both X and Y
        shared_err = gen_Err(seed=args.shared_err_seed, mu=args.shared_err_mu, sd=args.shared_err_sd, m_rows=args.m_rows)
    
    if args.y_err_input:
        # Generate noise node to be parent of Y (Epsilon-Y)
        y_err = gen_Err(seed=args.y_err_seed, mu=args.y_err_mu, sd=args.y_err_sd, m_rows=args.m_rows)

        # Create linear combination of shared_err and y_err
        # To pass to Y generation function
        y_err_comb = args.y_shared_err_weight*shared_err + (1-args.y_shared_err_weight)*y_err

    else:
        y_err=None
        y_err_comb=None
        
    if args.x_err_input:
        # Generate noise node to be parent of X (Epsilon-X)
        x_err = gen_Err(seed=args.x_err_seed, mu=args.x_err_mu, sd=args.x_err_sd, m_rows=args.m_rows)

        # Create linear combination of shared_err and x_err
        # To pass to Y generation function
        x_err_comb = args.x_shared_err_weight*shared_err + (1-args.x_shared_err_weight)*x_err
    else:
        x_err=None
        x_err_comb=None
    
    # Generate X
    prescale_x = gen_X(seed=args.x_seed, a=a, err_input = args.x_err_input, err=x_err_comb,
                        mu_dict=args.x_mu_dict, sd_dict=args.x_sd_dict)

    # Rescale X to [0,1] if normalize is set to True
    if args.normalize:
        x = rescale(prescale_x)
    else:
        x = prescale_x
    
    # Generate Y
    prescale_y = gen_Y(s=s, r=r, x=x, err_input = args.y_err_input, err=y_err_comb,
                        s_boost_dict=args.y_s_boost_dict, r_boost_dict=args.y_r_boost_dict, 
                        x_weight=args.y_x_weight)
    
    # Rescale Y to [0,1] if normalize is set to True
    if args.normalize:
        y = rescale(prescale_y)
    else:
        y = prescale_y

    # Compile columns into dataframe
    data = pd.DataFrame({'a':a, 'r':r, 's':s, 
                         'shared_err': shared_err,
                         'x_err': x_err, 'prescale_x': prescale_x, 'x':x, 
                         'y_err': y_err, 'prescale_y': prescale_y, 'y':y})

    # Drop unobserved noise nodes if observed set to True
    if args.observed: 
        data.drop(columns=['shared_err', 'x_err', 'y_err', 'prescale_x', 'prescale_y'], inplace=True)
    
    # Save to CSV if save set to True
    if args.save:
       data.to_csv(args.output_filepath, index=False)

    return data


def gen_data_and_sample_noise(args):
                                
    '''
    Generate original dataset and save to CSV
    Generate n_runs additional datasets using all same seeds except noise seeds
    Save original rank and additional n_runs ranks to pickle

    This implementation is favored for synthetic data experiments
    See gen_data_and_resample_noise for favored implementation for real data experiments
    '''

    # Set base output directory
    base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    base_output_dir = base_repo_dir / 'out' / 'synthetic_data' / args.output_dir

    # Generate baseline dataset with initial seeds and save to CSV
    args.save = True 
    args.output_filepath = base_output_dir / 'data' / args.data_filename
    data = gen_data(args)
    
    # Create dataframe to hold rank permutations
    rankings = pd.DataFrame()

    # Get original rank from original data
    rankings['rank'] = calc_rank(args.rank_seed, data['y'])

    # Update save argument in order to not save additional noise datasets
    args.save = False

    # Generate additional datasets
    for i in range(args.n_runs):

        # Increment noise seeds
        args.rank_seed +=1
        args.shared_err_seed +=1
        args.x_err_seed +=1
        args.y_err_seed +=1
        
        # Generate dataset with non-constant seeds incremented
        sim = gen_data(args)
        
        # Add this ranking to the original dataframe
        rankings['rank_'+str(i+1)] = calc_rank(args.rank_seed, sim['y'])

    # Save rankings to CSV if save_rankings set to True
    if args.save_rankings:
       rankings_output_filepath = base_output_dir / 'noise_rankings' / args.rankings_filename
       rankings.to_pickle(rankings_output_filepath, compression='gzip')
    
    return (data, rankings)
    
def resample_noise_from_data(orig_data, args):

    '''
    Takes in observed dataset with columns A, R, S, X, Y, and rank
    Generate dataset and sample n_runs re-rankings derived from noise distribution
    Uses default settings from gen_data

    Needs to be tested if used.
    '''

    # Create dataframe to hold rank permutations
    rankings = pd.DataFrame()

    # Get original rank from original data
    rankings['rank'] = calc_rank(args.rank_seed, orig_data['y'])

    # Save original seeds into new variables
    rank_seed = args.rank_seed
    shared_err_seed = args.shared_err_seed
    x_err_seed = args.x_err_seed
    y_err_seed = args.y_err_seed

    # Generate additional rankings by re-sampling noise distribution
    for i in range(0, args.n_runs):

        # Increment noise seeds
        rank_seed +=1
        shared_err_seed +=1
        x_err_seed +=1
        y_err_seed +=1

        if args.x_err_input or args.y_err_input:
            # Generate noise node to be parent of both X and Y
            shared_err = gen_Err(seed=shared_err_seed, mu=args.shared_err_mu, sd=args.shared_err_sd, m_rows=args.m_rows)
    
        if args.x_err_input:
            # Generate new noise node to be parent of X (Epsilon-X)
            x_err = gen_Err(seed=x_err_seed, mu=args.x_err_mu, sd=args.x_err_sd, m_rows=args.m_rows)

            # Create linear combination of shared_err and x_err
            x_err_comb = args.x_shared_err_weight*shared_err + (1-args.x_shared_err_weight)*x_err

            # Add new noise node to original x values
            new_x = orig_data.x.values + x_err_comb
        else:
            # Keep original x values if DAG specifies X has no noise input
            new_x = orig_data.x.values

        if args.y_err_input:
            # Generate noise node to be parent of Y (Epsilon-Y)
            y_err = gen_Err(seed=y_err_seed, mu=args.y_err_mu, sd=args.y_err_sd, m_rows=args.m_rows)

            # Create linear combination of shared_err and y_err
            # To pass to Y generation function
            y_err_comb = args.y_shared_err_weight*shared_err + (1-args.y_shared_err_weight)*y_err

        else:
            y_err_comb = None

        new_y = gen_Y(s=orig_data.s.values, r=orig_data.r.values, x = new_x, 
                      err_input=args.y_err_input, err=y_err_comb,
                      s_boost_dict=args.y_s_boost_dict, r_boost_dict=args.y_r_boost_dict,
                      x_weight=args.y_x_weight)

        new_rank = calc_rank(rank_seed, new_y)
        
        # Add this ranking to the original dataframe
        rankings['rank_'+str(i+1)] = new_rank

    return rankings

def gen_data_and_resample_noise(args):

    '''
    Generate original observed dataset: S, R, X, Y, and rank
    Save dataset to CSV
    Resample rankings by sampling from noise distribution

    There is an outstanding question of how scaling is handled in this function

    This implementation may be favored for real data experiments in the future?
    See gen_data_and_sample_noise for favored implementation for synthetic data experiments

    Needs to be tested if used.
    '''

    # Set output filepaths
    args.save = True 
    base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    base_output_dir = base_repo_dir / 'out' / 'synthetic_data'/ args.output_dir
    args.output_filepath = base_output_dir / 'data' / args.data_filename
    
    # Generate original dataset with initial seeds
    orig_data = gen_data(args)

    # Use orig_data to resample rankings from noise distribution
    rankings_data = resample_noise_from_data(orig_data, args)

    # Save rankings to CSV if save_rankings set to True
    if args.save_rankings:
       rankings_output_filepath = base_output_dir / 'noise_rankings' / args.rankings_filename
       rankings_data.to_pickle(rankings_output_filepath, compression='gzip')

    return (orig_data, rankings_data)

def sampling_distribution(args):
                            
    '''
    Generate s_samples of original dataset, save each to CSV
    For each dataset, sample rankings from noise distribution and save to CSV
    '''

    r = args.race_seed
    # Generate s_samples of original dataset and sample rankings from noise distribution of each
    for i in range(args.s_samples):

        # Re-instate parameter object
        args = params(output_dir=args.output_dir, 
                      race_seed=r, 
                      save=True, 
                      save_rankings=True, 
                      observed=True,
                      data_filename = 'samp_{}.csv'.format(i+1),
                      rankings_filename='samp_{}.pkl'.format(i+1))

        # Generate data and noisy rankings
        gen_data_and_sample_noise(args)

        # Increment race seed (which will control all other seeds)
        r += (args.n_runs+1)*4+3

        print('\nFinished generating sample {} of {}\n\n'.format(i+1, args.s_samples))

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Synthetic Data Generation for Stability Analysis")

    # Required arguments
    parser.add_argument("--s_samples", type=int, help='number of samples to draw for sampling distribution')
    parser.add_argument("--n_runs", type=int, help='number of runs for error distribution')
    parser.add_argument("--m_rows", type=int, help='number of rows in dataset')

    # Boolean arguments (add flag to set to true)
    parser.add_argument("--check_seeds", action='store_true', 
                        help='boolean flag, will print all random seeds if set to True')

    parser.add_argument("--no_x_err", action='store_true', 
                        help='boolean flag indicating X has no error parents')

    parser.add_argument("--no_y_err", action='store_true', 
                        help='boolean flag indicating Y has no error parents')

    parser.add_argument("--do_not_normalize", action='store_true', 
                        help='boolean flag indicating X and Y will not be scaled to [0,1]')

    parser.add_argument("--unobserved", action='store_true', 
                        help='boolean flag indicating unobserved columns will not be dropped from data')

    # Optional arguments
    parser.add_argument("--seed", type=int, default=0, help='Initial seed for race sample; basis of all other seeds')
    parser.add_argument("--output_dir", type=str, default='default',  help='folder within out/synthetic_data/ to store output')

    parser.add_argument("--x_err_mu", type=float, default=0)
    parser.add_argument("--x_err_sd", type=float, default=1)
    parser.add_argument("--shared_err_mu", type=float, default=0)
    parser.add_argument("--shared_err_sd", type=float, default=1)
    parser.add_argument("--y_err_mu", type=float, default=0)
    parser.add_argument("--y_err_sd", type=float, default=1)

    parser.add_argument("--err_corr", type=float, default=0.5)

    parser.add_argument("--prob_white", type=float, default=0.5)
    parser.add_argument("--prob_black", type=float, default=0.2)
    parser.add_argument("--prob_asian", type=float, default=0.3)

    parser.add_argument("--prob_male", type=float, default=0.6)
    parser.add_argument("--prob_female", type=float, default=0.4)

    parser.add_argument("--x_mu_wm", type=float, default=2)
    parser.add_argument("--x_mu_bm", type=float, default=-1)
    parser.add_argument("--x_mu_am", type=float, default=0)
    parser.add_argument("--x_mu_wf", type=float, default=0) 
    parser.add_argument("--x_mu_bf", type=float, default=-2)
    parser.add_argument("--x_mu_af", type=float, default=1)

    parser.add_argument("--x_sd_wm", type=float, default=2)
    parser.add_argument("--x_sd_bm", type=float, default=0.5)
    parser.add_argument("--x_sd_am", type=float, default=1)
    parser.add_argument("--x_sd_wf", type=float, default=1.5) 
    parser.add_argument("--x_sd_bf", type=float, default=1)
    parser.add_argument("--x_sd_af", type=float, default=1.5)

    parser.add_argument("--y_boost_male", type=float, default=1)
    parser.add_argument("--y_boost_female", type=float, default=0)
    parser.add_argument("--y_boost_white", type=float, default=1)
    parser.add_argument("--y_boost_black", type=float, default=-1)
    parser.add_argument("--y_boost_asian", type=float, default=0)

    parser.add_argument("--y_x_weight", type=float, default=0.8)

    cli_args = parser.parse_args()

    # Set normalize argument based on do_not_normalize
    if cli_args.do_not_normalize==True:
        cli_args.normalize=False
    else:
        cli_args.normalize=True

    # Set x_err_input argument based on no_x_err
    if cli_args.no_x_err==True:
        cli_args.x_err_input=False
    else:
        cli_args.x_err_input=True

    # Set y_err_input argument based on no_y_err
    if cli_args.no_y_err==True:
        cli_args.y_err_input=False
    else:
        cli_args.y_err_input=True

    # Set observed argument based on unobserved
    if cli_args.unobserved==True:
        cli_args.observed=False
    else:
        cli_args.observed=True

    # Define parameter class to pass between simulation functions
    class params:
        # Class variables shared by all params instances
        s_samples = cli_args.s_samples 
        n_runs = cli_args.n_runs
        m_rows = cli_args.m_rows
        x_err_input = cli_args.x_err_input
        y_err_input = cli_args.y_err_input
        x_err_mu = cli_args.x_err_mu
        x_err_sd = cli_args.x_err_sd
        y_err_mu = cli_args.y_err_mu
        y_err_sd = cli_args.y_err_sd
        shared_err_mu = cli_args.shared_err_mu
        shared_err_sd = cli_args.shared_err_sd
        x_shared_err_weight = get_shared_weight(cli_args.err_corr)
        y_shared_err_weight = get_shared_weight(cli_args.err_corr)
        race_prob = [cli_args.prob_white, cli_args.prob_black, cli_args.prob_asian]
        sex_prob = [cli_args.prob_male, cli_args.prob_female]
        normalize = cli_args.normalize
        x_mu_dict = {'wm': cli_args.x_mu_wm, 'bm': cli_args.x_mu_bm, 'am': cli_args.x_mu_am, 
                     'wf': cli_args.x_mu_wf, 'bf': cli_args.x_mu_bf, 'af': cli_args.x_mu_af} 
        x_sd_dict = {'wm': cli_args.x_sd_wm, 'bm': cli_args.x_sd_bm, 'am': cli_args.x_sd_am, 
                     'wf': cli_args.x_sd_wf, 'bf': cli_args.x_sd_bf, 'af': cli_args.x_sd_af} 
        y_s_boost_dict = {'m': cli_args.y_boost_male, 'f': cli_args.y_boost_female}
        y_r_boost_dict = {'w': cli_args.y_boost_white, 'b': cli_args.y_boost_black, 'a': cli_args.y_boost_asian}
        y_x_weight = cli_args.y_x_weight
        check_seeds = cli_args.check_seeds
        
        def __init__(self, output_dir, race_seed, sex_seed=None, x_seed=None, 
                    x_err_seed=None,  y_err_seed=None, shared_err_seed=None, rank_seed=None, 
                    save=None, save_rankings=None, output_filepath=None, 
                    data_filename=None, rankings_filename=None,
                    observed=None):
            # instance variable unique to each params instance

            self.output_dir = output_dir
            
            self.race_seed = race_seed
            self.sex_seed = sex_seed if sex_seed is not None else race_seed+1
            self.x_seed = x_seed if x_seed is not None else race_seed+2
            self.rank_seed = rank_seed if rank_seed is not None else race_seed+3
            
            self.shared_err_seed = shared_err_seed if shared_err_seed is not None else self.n_runs+1+self.rank_seed
            self.x_err_seed = x_err_seed if x_err_seed is not None else (self.n_runs+1)*2+self.rank_seed
            self.y_err_seed = y_err_seed if y_err_seed is not None else (self.n_runs+1)*3+self.rank_seed
            
            self.save = save
            self.save_rankings=save_rankings
            self.output_filepath = output_filepath
            self.data_filename=data_filename
            self.rankings_filename=rankings_filename
            self.observed = observed
            
        def print_seeds(self):
            #Function to print random seeds

            seeds = ['race_seed', 'sex_seed', 'x_seed', 'rank_seed', 
                    'shared_err_seed', 'x_err_seed', 'y_err_seed']
            for s in seeds:
                print('{}:'.format(s), getattr(self, s))
            print('\n')

    # Intialize parameter object to pass to data simulation
    args = params(output_dir=cli_args.output_dir, 
                  race_seed=cli_args.seed)

    if cli_args.check_seeds:
        print('INITIAL ARGUMENTS:')
        print_attributes(args)

    sampling_distribution(args)