#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse, os, pathlib
from functools import partial
from utils.stability_utils import calc_rank
from utils.eval_utils import calculate_kendall_tau_distance_quick,\
                             num_retained_at_top_k,\
                             change_in_percent_at_top_k,\
                             percent_change_in_percent_at_top_k,\
                             ratio_of_percent_at_top_k,\
                             prob_lower,\
                             prob_lower_group,\
                             prob_lower_group_ratio
                                

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
        dists_nonres_a0_a1 = np.zeros([args.s_samples, len(args.metrics)])

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
            counter_y_cols = [x for x in counter.columns if (('cf_y_' in x)&(x[-1]=='0'))]
        
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

        # Read original dataset
        data = pd.read_csv(args.base_repo_dir /'out'/\
                                'synthetic_data'/args.output_dir/\
                                'data'/'samp_{}.csv'.format(s))

        # Create list of partial functions for metrics
        metrics_dict = {'kendalls_tau': partial(calculate_kendall_tau_distance_quick), 
                        'num_retained_at_top_k': partial(num_retained_at_top_k, k=args.k), 
                        'change_in_percent_at_top_k': \
                            partial(change_in_percent_at_top_k, 
                                    k=args.k, groups=data['a'], which_group=0),
                        'percent_change_in_percent_at_top_k': \
                            partial(percent_change_in_percent_at_top_k, 
                                    k=args.k, groups=data['a'], which_group=0),
                        'ratio_of_percent_at_top_k': \
                            partial(ratio_of_percent_at_top_k, 
                                    k=args.k, groups=data['a'], which_group=0),
                        'prob_lower': partial(prob_lower),
                        'prob_lower_group': partial(prob_lower_group, 
                                            groups=data['a'], which_group=0),
                        'prob_lower_group_ratio': partial(prob_lower_group_ratio, 
                                                      groups=data['a'], groupa=0, groupb=1) }
        metrics = [metrics_dict[m] for m in args.metrics]
        
        # Get distances between original rank and each rank from noise distribution
        # Averages will give expected noise distances for this sample
        for n in range(1, args.n_runs+1):
            for m, metric in enumerate(metrics):
                noise_dists[m][s-1][n-1] = metric(rank1=orig_rank, rank2=noise['rank_{}'.format(n)])

        for m, metric in enumerate(metrics):
            try:
                # Get distance between original rank and counterfactual Y with resolving X
                counter_dists_nonres0[s-1][m] = metric(rank1=orig_rank, rank2=counter['rank_nonres_a0'])
                
                # Get distance between original rank and counterfactual Y with non-resolving X
                counter_dists_xres0[s-1][m] = metric(rank1=orig_rank, rank2=counter['rank_xres_a0'])
                
            # Catch exception for A=0 not present in original dataset
            # A<-0 intervention will not have been performed
            except:
                counter_dists_nonres0[s-1][m] = np.nan
                counter_dists_xres0[s-1][m] = np.nan

            if args.all_interventions:
                try:
                    # Get distance between original rank and counterfactual Y with resolving X
                    counter_dists_nonres1[s-1][m] = metric(rank1=orig_rank, rank2=counter['rank_nonres_a1'])
                    
                    # Get distance between original rank and counterfactual Y with non-resolving X
                    counter_dists_xres1[s-1][m] = metric(rank1=orig_rank, rank2=counter['rank_xres_a1'])
                
                # Catch exception for A=1 not present in original dataset
                # A<-1 intervention will not have been performed
                except:

                    counter_dists_nonres1[s-1][m] = np.nan
                    counter_dists_xres1[s-1][m] = np.nan

        if args.all_interventions:
            for m, metric in enumerate(metrics):
                # Get distances between counterfactual ranks for intervention A<-0 and for intervention A<-1
                try:
                    # Distance between ranks resulting from each intervention for non-resolving X
                    dists_nonres_a0_a1[s-1][m] = metric(rank1=counter['rank_nonres_a0'], 
                                                        rank2=counter['rank_nonres_a1'])
                    
                    # Distance between ranks resulting from each intervention for resolving X
                    dists_xres_a0_a1[s-1][m] = metric(rank1=counter['rank_xres_a0'], 
                                                      rank2=counter['rank_xres_a1'])
                    
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
    exp_exp_noise_dist = np.mean(exp_noise_dist, axis=1)

    # Get expected distance between original rank and counterfactual Y with non-resolving X
    # For intervention A<-0
    # Expectation taken over s_samples
    # E[ distance(original rank, counterfactual rank with non-resolving X for A<-0) ]
    # 1 value per metric for entire experiment trial
    exp_cf_dist_nonres_a0 = np.nanmean(counter_dists_nonres0, axis=0)

    if args.all_interventions:
        # Get expected distance between original rank and counterfactual Y with non-resolving X
        # For intervention A<-1
        # Expectation taken over s_samples
        # E[ distance(original rank, counterfactual rank with non-resolving X for A<-1) ]
        # 1 value per metric for entire experiment trial
        exp_cf_dist_nonres_a1 = np.nanmean(counter_dists_nonres1, axis=0)

    # Get expected distance between original rank and counterfactual Y with resolving X
    # For intervention A<-0
    # Expectation taken over s_samples
    # E[ distance(original rank, counterfactual rank with resolving X for A<-0) ]
    # 1 value per metric for entire experiment trial
    exp_cf_dist_xres_a0 = np.nanmean(counter_dists_xres0, axis=0)

    if args.all_interventions:
        # Get expected distance between original rank and counterfactual Y with resolving X
        # For intervention A<-1
        # Expectation taken over s_samples
        # E[ distance(original rank, counterfactual rank with resolving X for A<-1) ]
        # 1 value per metric for entire experiment trial
        exp_cf_dist_xres_a1 = np.nanmean(counter_dists_xres1, axis=0)

    # Bundle expected distances into list of dictionaries
    exp_dicts = []
    for m, metric in enumerate(metrics):
        if args.all_interventions:
            exp_dict = {'exp_exp_noise':exp_exp_noise_dist[m],
                        'exp_nonres0': exp_cf_dist_nonres_a0[m],
                        'exp_nonres1': exp_cf_dist_nonres_a1[m],
                        'exp_xres0': exp_cf_dist_xres_a0[m],
                        'exp_xres1': exp_cf_dist_xres_a1[m]}
        else:
            exp_dict = {'exp_exp_noise':exp_exp_noise_dist[m],
                        'exp_nonres0': exp_cf_dist_nonres_a0[m],
                        'exp_xres0': exp_cf_dist_xres_a0[m]}
        exp_dicts.append(exp_dict)

    # Bundle expected noise distances into 1d list of dataframes
    exp_noise_dfs = []
    for m, metric in enumerate(metrics):
        noise_df = pd.DataFrame(noise_dists[m, :, :], columns=['orig_noise'+str(i+1) for i in range(args.n_runs)])
        exp_noise_dfs.append(noise_df)
    
    # Bundle counterfactual distances into 1d list of dataframes
    cf_dist_dfs = []
    for m, metric in enumerate(metrics):
        if args.all_interventions:
            cf_dist_df = pd.concat([ pd.DataFrame(exp_noise_dist[m], columns=['exp_orig_noise']),
                                    pd.DataFrame(counter_dists_nonres0[:,m], columns=['orig_nonres0']), 
                                    pd.DataFrame(counter_dists_nonres1[:,m], columns=['orig_nonres1']), 
                                    pd.DataFrame(counter_dists_xres0[:,m], columns=['orig_xres0']), 
                                    pd.DataFrame(counter_dists_xres1[:,m], columns=['orig_xres1']), 
                                    pd.DataFrame(dists_nonres_a0_a1[:,m], columns=['nonres0_nonres1']),
                                    pd.DataFrame(dists_xres_a0_a1[:,m], columns=['xres0_xres1']),
                                ], axis=1)
        else:
            cf_dist_df = pd.concat([ pd.DataFrame(exp_noise_dist[m], columns=['exp_orig_noise']),
                                    pd.DataFrame(counter_dists_nonres0[:,m], columns=['orig_nonres0']), 
                                    pd.DataFrame(counter_dists_xres0[:,m], columns=['orig_xres0']),
                                ], axis=1)
        cf_dist_dfs.append(cf_dist_df)

    return (exp_noise_dfs, cf_dist_dfs, exp_dicts)


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

    parser.add_argument("--all_metrics", action='store_true', 
                            help='Boolean flag indicating whether to evaluate\
                                  all distance metrics.\
                                  If not set, only those Kendall\'s Tau and \
                                  any others specifically set will be evaluated.')

    parser.add_argument("--num_retained_at_top_k", action='store_true', 
                        help='Boolean flag indicating whether to additionally evaluate\
                              the distance metric num_retained_at_top_k.\
                              If not set, num_retained_at_top_k will not be evaluated.')
    
    parser.add_argument("--change_in_percent_at_top_k", action='store_true', 
                        help='Boolean flag indicating whether to additionally evaluate\
                              the distance metric percent_change_in_percent_at_top_k.\
                              If not set, change_in_percent_at_top_k will not be evaluated.')

    parser.add_argument("--percent_change_in_percent_at_top_k", action='store_true', 
                        help='Boolean flag indicating whether to additionally evaluate\
                              the distance metric change_in_percent_at_top_k.\
                              If not set, percent_change_in_percent_at_top_k will not be evaluated.')

    parser.add_argument("--ratio_of_percent_at_top_k", action='store_true', 
                        help='Boolean flag indicating whether to additionally evaluate\
                              the distance metric ratio_of_percent_at_top_k.\
                              If not set, ratio_of_percent_at_top_k will not be evaluated.')

    parser.add_argument("--prob_lower", action='store_true', 
                        help='Boolean flag indicating whether to additionally evaluate\
                              the distance metric prob_lower.\
                              If not set, prob_lower will not be evaluated.')

    parser.add_argument("--prob_lower_group", action='store_true', 
                        help='Boolean flag indicating whether to additionally evaluate\
                              the distance metric prob_lower_group.\
                              If not set, prob_lower_group will not be evaluated.')

    parser.add_argument("--prob_lower_group_ratio", action='store_true', 
                        help='Boolean flag indicating whether to additionally evaluate\
                              the distance metric prob_lower_group_ratio.\
                              If not set, prob_lower_group_ratio will not be evaluated.')

    parser.add_argument("--no_kendalls_tau", action='store_true', 
                        help='Boolean flag indicating whether to exclude Kendall\'s Tau from evaluation.\
                            If not set, Kendall\'s Tau will be evaluated.')

    parser.add_argument("--k", type=int, default=0,
                        help='Integer to use for "top k" evaluations.')
            

    # Parse arguments
    args = parser.parse_args()

    if args.k==0:
        args.k=None

    # Create list of names of distance functions
    args.metrics = []
    if not args.no_kendalls_tau:
        args.metrics.append('kendalls_tau')
    if args.num_retained_at_top_k | args.all_metrics:
        args.metrics.append('num_retained_at_top_k')
    if args.change_in_percent_at_top_k | args.all_metrics:
        args.metrics.append('change_in_percent_at_top_k')
    if args.percent_change_in_percent_at_top_k | args.all_metrics:
        args.metrics.append('percent_change_in_percent_at_top_k')
    if args.ratio_of_percent_at_top_k | args.all_metrics:
        args.metrics.append('ratio_of_percent_at_top_k')
    if args.prob_lower | args.all_metrics:
        args.metrics.append('prob_lower')
    if args.prob_lower_group | args.all_metrics:
        args.metrics.append('prob_lower_group')
    if args.prob_lower_group_ratio | args.all_metrics:
        args.metrics.append('prob_lower_group_ratio')

    # Save repo root directory
    args.base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]

    # Save high-level output directory
    main_output_dir = args.output_dir

    # Define path to traverse
    output_path = args.base_repo_dir /'out'/'counterfactual_data'/ main_output_dir   

    # Initialize list of lists to hold expected distances
    exp_dicts_by_metric = [[] for m in range(len(args.metrics))]

    # Initialize list of lists of experiment subdirectories
    sub_dirs = []

    print('output_path: ', output_path)
    print(os.walk(output_path))

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

            # Calculate distances from files in this subdir
            exp_noise_dfs, cf_dist_dfs, exp_dicts = calc_expected_distances(args)

            filename = '_'.join(sub_dir)+'.pkl'
            for m, exp_noise_df in enumerate(exp_noise_dfs):
                exp_noise_df.to_pickle(args.base_repo_dir/'out'/'distance_metrics'/\
                                        main_output_dir/args.metrics[m]/'noise'/\
                                        filename, 
                                        compression='gzip')

            for m, cf_dist_df in enumerate(cf_dist_dfs):
                cf_dist_df.to_pickle(args.base_repo_dir/'out'/'distance_metrics'/\
                                        main_output_dir/args.metrics[m]/'counterfactuals'/\
                                        filename, 
                                        compression='gzip')

            for m, exp_dict in enumerate(exp_dicts):
                exp_dicts_by_metric[m].append(exp_dict)

            print('Distance calculation for {} complete'.format(args.output_dir))
            
    # Get list of indices which lex-sort list of lists of subdirectories
    dirs_sort = np.lexsort(tuple([np.array(sub_dirs).T[i] \
                                 for i in np.flip(np.arange(np.array(sub_dirs).shape[1]))]))

    for m, exp_dict in enumerate(exp_dicts):
        # Create expected distance df with multi-index of lex-sorted subdirectories
        exp_df = pd.DataFrame(list(np.array(exp_dicts_by_metric[m])[dirs_sort]), 
                              index=pd.MultiIndex.from_tuples(sorted(sub_dirs)))

        # Save expected distance df to pickle
        exp_df.to_pickle(args.base_repo_dir/'out'/'distance_metrics'/\
                                   main_output_dir/args.metrics[m]/'expected.pkl', 
                            compression='gzip')
