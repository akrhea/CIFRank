#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse, os, pathlib
#from scipy.stats import kendalltau
from utils.stability_utils import calc_rank
from utils.eval_utils import num_retained_at_top_k, kendalls_tau_scipy


def calc_expected_distances(args):

    '''
    Based on src/notebooks/Experiment_Skeleton.ipynb
    '''

    # Highest seed not yet used in data gen
    seed = args.s_samples*((args.n_runs+1)*3+2)

    # Initialize arrays to hold distances
    noise_dists = np.zeros([len(args.metrics), args.s_samples, args.n_runs])
    counter_dists_xres0 = np.zeros([args.s_samples, len(args.metrics)])
    counter_dists_nonres0 = np.zeros([args.s_samples, len(args.metrics)])

    if args.all_interventions:
        counter_dists_xres1 = np.zeros([args.s_samples, len(args.metrics)])
        counter_dists_nonres1 = np.zeros([args.s_samples, len(args.metrics)])
        dists_xres_a0_a1 = np.zeros([args.s_samples, len(args.metrics)])
        dists_nonres_a0_a1 = np.zeros([args.s_samples, len(args.metrics])

    # Loop through s_samples
    for s in range(1, args.s_samples+1):
        
        # Read in counterfactual data
        counter = pd.read_pickle(args.base_repo_dir /'out'/\
                                'counterfactual_data'/args.output_dir/
                                'counter_samp_{}.pkl'.format(s), compression='gzip')
        
        # Get list of counterfactual Y columns
        if args.all_interventions:
            counter_y_cols = [x for x in counter.columns if 'cf_y_' in x]
        else:
            counter_y_cols = [x for x in counter.columns if (('cf_y_' in x)&&(x[-1]))]
        
        # Calculate ranks for each counterfactual Y
        for y in counter_y_cols:
            counter['rank_'+y[5:]] = calc_rank(seed=seed, y=counter[y])
            seed += 1
        
        # Read in noise distribution of ranks
        noise = pd.read_pickle(args.base_repo_dir /'out'/\
                                'synthetic_data'/args.output_dir/\
                                'noise_rankings'/'samp_{}.pkl'.format(s), 
                                compression='gzip')
        
        # Get original rank
        orig_rank = noise['rank']
        
        # Get distances between original rank and each rank from noise distribution
        # Averages will give expected noise distances for this sample
        for n in range(1, args.n_runs+1):
            for m, metric in enumerate(args.metrics):
                noise_dists[m][s-1][n-1] = metric(orig_rank, noise['rank_{}'.format(n)])

        for m, metric in enumerate(args.metrics):
            try:
                # Get distance between original rank and counterfactual Y with resolving X
                counter_dists_nonres0[s-1][m] = metric(orig_rank, counter['rank_nonres_a0'])
                
                # Get distance between original rank and counterfactual Y with non-resolving X
                counter_dists_xres0[s-1][m] = metric(orig_rank, counter['rank_xres_a0'])
                
            # Catch exception for A=0 not present in original dataset
            # A<-0 intervention will not have been performed
            except:
                counter_dists_nonres0[s-1][m] = np.nan
                counter_dists_xres0[s-1][m] = np.nan

            if args.all_interventions:
                try:
                    # Get distance between original rank and counterfactual Y with resolving X
                    counter_dists_nonres1[s-1][m] = metric(orig_rank, counter['rank_nonres_a1'])
                    
                    # Get distance between original rank and counterfactual Y with non-resolving X
                    counter_dists_xres1[s-1][m] = metric(orig_rank, counter['rank_xres_a1'])
                
                # Catch exception for A=1 not present in original dataset
                # A<-1 intervention will not have been performed
                except:
                    counter_dists_nonres1[s-1][m] = np.nan
                    counter_dists_xres1[s-1][m] = np.nan

        if args.all_interventions:
            for m, metric in enumerate(args.metrics):
                # Get distances between counterfactual ranks for intervention A<-0 and for intervention A<-1
                try:
                    # Distance between ranks resulting from each intervention for non-resolving X
                    dists_nonres_a0_a1[s-1][m] = metric(counter['rank_nonres_a0'], counter['rank_nonres_a1'])
                    
                    # Distance between ranks resulting from each intervention for resolving X
                    dists_xres_a0_a1[s-1][m] = metric(counter['rank_xres_a0'], counter['rank_xres_a1'])
                    
                # Catch exception for only one value of A present in original dataset
                # Only one intervention will have been performed
                except:
                    dists_nonres_a0_a1[s-1][m] = np.nan
                    dists_xres_a0_a1[s-1][m] = np.nan

    # Get expected distances between original rank and rank from re-sampled noise
    # Expectations taken over n_runs of noise distribution
    # E[ distance (original rank, rank from re-sampled noise) ]
    # 1 value per metric for each sample
    exp_noise_dist = np.mean(noise_dists, axis=2)

    # Get expected value of expected distances between original rank and rank from re-sampled noise
    # Iterated expectation: expectation first taken over n_runs of noise distribution, then over s_samples
    # E[ E[ distance(original rank, rank from re-sampled noise) ] ]
    # 1 value per metric for entire experiment trial
    exp_exp_noise_dist = np.mean(exp_noise_dist)

    # Get expected distance between original rank and counterfactual Y with non-resolving X
    # For intervention A<-0
    # Expectation taken over s_samples
    # E[ distance(original rank, counterfactual rank with non-resolving X for A<-0) ]
    # 1 value per metric for entire experiment trial
    exp_cf_dist_nonres_a0 = np.nanmean(counter_dists_nonres0(axis=0))

    if args.all_interventions:
        # Get expected distance between original rank and counterfactual Y with non-resolving X
        # For intervention A<-1
        # Expectation taken over s_samples
        # E[ distance(original rank, counterfactual rank with non-resolving X for A<-1) ]
        # 1 value per metric for entire experiment trial
        exp_cf_dist_nonres_a1 = np.nanmean(counter_dists_nonres1(axis=0))

    # Get expected distance between original rank and counterfactual Y with resolving X
    # For intervention A<-0
    # Expectation taken over s_samples
    # E[ distance(original rank, counterfactual rank with resolving X for A<-0) ]
    # 1 value per metric for entire experiment trial
    exp_cf_dist_xres_a0 = np.nanmean(counter_dists_xres0(axis=0))

    if args.all_interventions:
        # Get expected distance between original rank and counterfactual Y with resolving X
        # For intervention A<-1
        # Expectation taken over s_samples
        # E[ distance(original rank, counterfactual rank with resolving X for A<-1) ]
        # 1 value per metric for entire experiment trial
        exp_cf_dist_xres_a1 = np.nanmean(counter_dists_xres1(axis=0))
    
    # Bundle expected distances into dataframe
    if args.all_interventions:
        exp_dists_df = pd.concat([pd.DataFrame(exp_exp_noise_dist, columns=['exp_exp_noise']),
                                  pd.DataFrame(exp_cf_dist_nonres_a0, columns=['exp_nonres0']),
                                  pd.DataFrame(exp_cf_dist_nonres_a1, columns=['exp_nonres1']),
                                  pd.DataFrame(exp_cf_dist_xres_a0, columns=['exp_xres0']),
                                  pd.DataFrame(exp_cf_dist_xres_a1, columns=['exp_xres1'])],
                                axis=1)
    else:
        exp_dists_df = pd.concat([pd.DataFrame(exp_exp_noise_dist, columns=['exp_exp_noise']),
                                  pd.DataFrame(exp_cf_dist_nonres_a0, columns=['exp_nonres0']),
                                  pd.DataFrame(exp_cf_dist_xres_a0, columns=['exp_xres0'])],
                                axis=1)

    # Bundle expected noise distances into 1d list of dataframes
    exp_noise_dfs = []
    for m, metric in enumerate(args.metrics):
        noise_df = pd.DataFrame(noise_dists[m, :, :], columns=['orig_noise'+str(i+1) for i in range(args.n_runs)])
        exp_noise_dfs.append(noise_df)
    
    # Bundle counterfactual distances into 1d list of dataframes
    cf_dist_dfs = []
    for m, metric in enumerate(args.metrics):
        if args.all_interventions:
            cf_dist_df = pd.concat([ pd.DataFrame(exp_noise_dist, columns=['exp_orig_noise']),
                                    pd.DataFrame(counter_dists_nonres0, columns=['orig_nonres0']), 
                                    pd.DataFrame(counter_dists_nonres1, columns=['orig_nonres1']), 
                                    pd.DataFrame(counter_dists_xres0, columns=['orig_xres0']), 
                                    pd.DataFrame(counter_dists_xres1, columns=['orig_xres1']), 
                                    pd.DataFrame(dists_nonres_a0_a1, columns=['nonres0_nonres1']),
                                    pd.DataFrame(dists_xres_a0_a1, columns=['xres0_xres1']),
                                ], axis=1)
        else:
            cf_dist_df = pd.concat([ pd.DataFrame(exp_noise_dist, columns=['exp_orig_noise']),
                                    pd.DataFrame(counter_dists_nonres0, columns=['orig_nonres0']), 
                                    pd.DataFrame(counter_dists_xres0, columns=['orig_xres0']),
                                ], axis=1)
        cf_dist_dfs.append(cf_dist_df)

    return (exp_noise_dfs, cf_dist_dfs, exp_dists_df)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate Distances")

    # Required arguments
    parser.add_argument("--s_samples", type=int, help='Number of samples to draw for sampling distribution')
    parser.add_argument("--n_runs", type=int, help='Number of runs for error distribution')

    # Optional arguments
    parser.add_argument("--output_dir", type=str, default='default')

    parser.add_argument("--all_interventions", action='store_true', 
                        help='Boolean flag indicating whether all intervention counterfactuals will be evaluated.\
                              If not set, only intervention to minority group will be evaluated.')

    parser.add_argument("--retained_at_top_k", action='store_true', 
                        help='Boolean flag indicating whether to additionally evaluate num_retained_at_top_k.\
                            If not set, the number retained at top k will not be evaluated.')')

    parser.add_argument("--no_kendalls_tau", action='store_true', 
                        help='Boolean flag indicating whether to exclude Kendall\'s Tau from evaluation.\
                            If not set, Kendall\'s Tau will be evaluated.')

    parser.add_argument("--k", type=int, default=5,
                        help='Integer to use for "top k" evaluations.')
            

    # Parse arguments
    args = parser.parse_args()

    # Create list of distance functions
    args.metrics = []
    if not args.no_kendalls_tau:
        args.metrics.append(partial(kendalls_tau_scipy))
    if args.retained_at_top_k:
        args.metrics.append(partial(num_retained_at_top_k, args.k))

    # Save repo root directory
    args.base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]

    # Save high-level output directory
    main_output_dir = args.output_dir

    # Define path to traverse
    output_path = args.base_repo_dir /'out'/'counterfactual_data'/ main_output_dir   

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
            exp_noise_dfs, cf_dist_dfs, exp_dists_df = calc_expected_distances(args)

            print('len(exp_noise_dfs): ', len(exp_noise_dfs))
            print('exp_noise_dfs[0].shape: ', exp_noise_dfs[0].shape)
            print('exp_noise_dfs[0].columns: ', exp_noise_dfs[0].columns)
            print('\n')
            print('len(cf_dist_dfs): ', len(cf_dist_dfs))
            print('cf_dist_dfs[0].shape: ', cf_dist_dfs[0].shape)
            print('cf_dist_dfs[0].columns: ', cf_dist_dfs[0].columns)
            print('\n')
            print('exp_dists_df.shape: ', exp_dists_df.shape)
            print('exp_dists_df.columns: ', exp_dists_df.columns)
            print('\n\n\n')

            '''
            How to save results?
            '''
            # Save cf and noise distance dfs to CSV
            # filename = '_'.join(sub_dir)+'.pkl'
            # noise_kt_df.to_pickle(args.base_repo_dir/'out'/'distances'/\
            #                         main_output_dir/'noise'/filename, 
            #                         compression='gzip')

            # cf_kt_df.to_pickle(args.base_repo_dir/'out'/'distances'/\
            #                     main_output_dir/'counterfactuals'/filename, 
            #                     compression='gzip')
            
            # # Add dict of expected KTs to saved list of dicts
            # exp_dicts.append(exp_dict)

            print('Distance calculation for {} complete'.format(args.output_dir))
            
    # Get list of indices which lex-sort list of lists of subdirectories
    dirs_sort = np.lexsort(tuple([np.array(sub_dirs).T[i] \
                                 for i in np.flip(np.arange(np.array(sub_dirs).shape[1]))]))

    # Create expected KT df with multi-index of lex-sorted subdirectories
    # exp_kt_df = pd.DataFrame(list(np.array(exp_dicts)[dirs_sort]), 
     #                       index=pd.MultiIndex.from_tuples(sorted(sub_dirs)))

    # Save expected KT df to pickle
    #exp_kt_df.to_pickle(args.base_repo_dir/'out'/'distances'/\
    #                            main_output_dir/'expected.pkl', 
    #                     compression='gzip')
