#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse, os, pathlib
from scipy.stats import kendalltau
from utils.stability_utils import calc_rank


def calc_expected_kendalls_taus(args):

    '''
    Based on src/notebooks/Experiment_Skeleton.ipynb
    '''

    # Highest seed not yet used in data gen
    seed = args.s_samples*((args.n_runs+1)*3+2)

    # Initialize arrays to hold Kendall's Tau distances
    noise_kts = np.zeros([args.s_samples, args.n_runs])
    counter_kts_xres = np.zeros([args.s_samples, 2])
    counter_kts_nonres = np.zeros([args.s_samples, 2])
    kts_xres_a0_a1 = np.zeros(args.s_samples)
    kts_nonres_a0_a1 = np.zeros(args.s_samples)

    # Loop through s_samples
    for s in range(1, args.s_samples+1):
        
        # Read in counterfactual data
        counter = pd.read_pickle(args.base_repo_dir /'out'/\
                                'counterfactual_data'/'stability'/args.output_dir/
                                'counter_samp_{}.pkl'.format(s), compression='gzip')
        
        # Get list of counterfactual Y columns
        counter_y_cols = [x for x in counter.columns if 'cf_y_' in x]
        
        # Calculate ranks for each counterfactual Y
        for y in counter_y_cols:
            counter['rank_'+y[5:]] = calc_rank(seed=seed, y=counter[y])
            seed += 1
        
        # Read in noise distribution of ranks
        noise = pd.read_pickle(args.base_repo_dir /'out'/\
                                'synthetic_data'/'stability'/args.output_dir/\
                                'noise_rankings'/'samp_{}.pkl'.format(s), 
                                compression='gzip')
        
        # Get original rank
        orig_rank = noise['rank']
        
        # Get KT distances between original rank and each rank from noise distribution
        # Average will give expected noise KT for this sample
        for n in range(1, args.n_runs+1):
            noise_kt, noise_p = kendalltau(orig_rank, noise['rank_{}'.format(n)])
            noise_kts[s-1][n-1] = noise_kt
        
        # For each intervention, A<-0 and A<-1
        for a in range(2):
            try:
                # Get KT distance between original rank and counterfactual Y with resolving X
                counter_kts_nonres[s-1][a] = kendalltau(orig_rank, counter['rank_nonres_a{}'.format(a)])[0]
                
                # Get KT distance between original rank and counterfactual Y with non-resolving X
                counter_kts_xres[s-1][a] = kendalltau(orig_rank, counter['rank_xres_a{}'.format(a)])[0]
                
            # Catch exception for A=a not present in original dataset
            # A<-a intervention will not have been performed
            except:
                counter_kts_nonres[s-1][a] = np.nan
                counter_kts_xres[s-1][a] = np.nan
        
        # Get KT distance between counterfactual ranks for intervention A<-0 and for intervention A<-1
        try:
            # Distance between ranks resulting from each intervention for non-resolving X
            kts_nonres_a0_a1[s-1] = kendalltau(counter['rank_nonres_a0'], counter['rank_nonres_a1'])[0]   
            
            # Distance between ranks resulting from each intervention for resolving X
            kts_xres_a0_a1[s-1] = kendalltau(counter['rank_xres_a0'], counter['rank_xres_a1'])[0]
            
        # Catch exception for only one value of A present in original dataset
        # Only one intervention will have been performed
        except:
            kts_nonres_a0_a1[s-1] = np.nan
            kts_xres_a0_a1[s-1] = np.nan

    # Get expected KT distance between original rank and rank from re-sampled noise
    # Expectation taken over n_runs of noise distribution
    # E[ KT(original rank, rank from re-sampled noise) ]
    # 1 value for each sample
    exp_kt_noise = np.mean(noise_kts, axis=1)

    # Get expected value of expected KT distance between original rank and rank from re-sampled noise
    # Iterated expectation: expectation first taken over n_runs of noise distribution, then over s_samples
    # E[ E[ KT(original rank, rank from re-sampled noise) ] ]
    # 1 value for entire experiment trial
    exp_exp_kt_noise = np.mean(exp_kt_noise)

    # Get expected KT distance between original rank and counterfactual Y with non-resolving X
    # For intervention A<-0
    # Expectation taken over s_samples
    # E[ KT(original rank, counterfactual rank with non-resolving X for A<-0) ]
    # 1 value for entire experiment trial
    exp_kt_counter_nonres_a0 = np.nanmean(counter_kts_nonres[:,0])

    # Get expected KT distance between original rank and counterfactual Y with non-resolving X
    # For intervention A<-1
    # Expectation taken over s_samples
    # E[ KT(original rank, counterfactual rank with non-resolving X for A<-1) ]
    # 1 value for entire experiment trial
    exp_kt_counter_nonres_a1 = np.nanmean(counter_kts_nonres[:,1])

    # Get expected KT distance between original rank and counterfactual Y with resolving X
    # For intervention A<-0
    # Expectation taken over s_samples
    # E[ KT(original rank, counterfactual rank with resolving X for A<-0) ]
    # 1 value for entire experiment trial
    exp_kt_counter_xres_a0 = np.nanmean(counter_kts_xres[:,0])

    # Get expected KT distance between original rank and counterfactual Y with resolving X
    # For intervention A<-1
    # Expectation taken over s_samples
    # E[ KT(original rank, counterfactual rank with resolving X for A<-1) ]
    # 1 value for entire experiment trial
    exp_kt_counter_xres_a1 = np.nanmean(counter_kts_xres[:,1])
    
    # Bundle expected KTs into dict
    exp_dict = {'exp_exp_noise': exp_exp_kt_noise, 
                'exp_nonres0': exp_kt_counter_nonres_a0,
                'exp_nonres1': exp_kt_counter_nonres_a1,
                'exp_xres0': exp_kt_counter_xres_a0,
                'exp_xres1': exp_kt_counter_xres_a1}

    # Bundle noise KTs into df
    noise_kt_df = pd.DataFrame(noise_kts, columns=['orig_noise'+str(i+1) for i in range(args.n_runs)])
    
    # Bundle counterfactual KTs into df
    cf_kt_df = pd.concat([ pd.DataFrame(exp_kt_noise, columns=['exp_orig_noise']),
                            pd.DataFrame(counter_kts_nonres, columns=['orig_nonres0', 'orig_nonres1']), 
                            pd.DataFrame(counter_kts_xres, columns=['orig_xres0', 'orig_xres1']), 
                            pd.DataFrame(kts_nonres_a0_a1, columns=['nonres0_nonres1']),
                            pd.DataFrame(kts_xres_a0_a1, columns=['xres0_xres1']),
                        ], axis=1)

    return (noise_kt_df, cf_kt_df, exp_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate Kendall's Tau Distances")

    # Required arguments
    parser.add_argument("--s_samples", type=int, help='number of samples to draw for sampling distribution')
    parser.add_argument("--n_runs", type=int, help='number of runs for error distribution')

    # Optional argument
    parser.add_argument("--output_dir", type=str, default='default')

    # Parse arguments
    args = parser.parse_args()

    # Save repo root directory
    args.base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]

    # Save high-level output directory
    main_output_dir = args.output_dir

    # Define path to traverse
    output_path = args.base_repo_dir /'out'/'counterfactual_data'/'stability'/ main_output_dir   

    # Initialize list of dicts of expected KTs
    exp_dicts = []

    # Initialize list of lists of experiment subdirectories
    sub_dirs = []

    # Walk through experiment directory and subdirectories
    for root, dirs, files in os.walk(output_path):

        # Exclude hidden files
        files = [f for f in files if not f[0] == '.']

        # If directory contains files
        if len(files)>0:

            # Isolate experiment subdirectory
            args.output_dir = main_output_dir+ root.split(main_output_dir)[1]

            # Save subdirectory to list of lists
            sub_dir = args.output_dir.split('/')[1:]
            sub_dirs.append(sub_dir)

            # Calculate KTs from files in this subdir
            noise_kt_df, cf_kt_df, exp_dict = calc_expected_kendalls_taus(args)

            
            # Save cf and noise distance dfs to CSV
            filename = '_'.join(sub_dir)+'.pkl'

            noise_kt_df.to_pickle(args.base_repo_dir/'out'/'kendalls_tau_distances'/\
                                    main_output_dir/'noise'/filename, 
                                    compression='gzip')

            cf_kt_df.to_pickle(args.base_repo_dir/'out'/'kendalls_tau_distances'/\
                                main_output_dir/'counterfactuals'/filename, 
                                compression='gzip')
            
            # Add dict of expected KTs to saved list of dicts
            exp_dicts.append(exp_dict)

            print('Distance calculation for {} complete'.format(args.output_dir))
            
    # Get list of indices which lex-sort list of lists of subdirectories
    dirs_sort = np.lexsort(tuple([np.array(sub_dirs).T[i] \
                                 for i in np.flip(np.arange(np.array(sub_dirs).shape[1]))]))

    # Create expected KT df with multi-index of lex-sorted subdirectories
    exp_kt_df = pd.DataFrame(list(np.array(exp_dicts)[dirs_sort]), 
                            index=pd.MultiIndex.from_tuples(sorted(sub_dirs)))

    # Save expected KT df to CSV
    exp_kt_df.to_pickle(args.base_repo_dir/'out'/'kendalls_tau_distances'/\
                                main_output_dir/'expected.pkl', 
                        compression='gzip')
