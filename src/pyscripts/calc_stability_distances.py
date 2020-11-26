#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse, os, pathlib
from functools import partial
from itertools import product
from utils.stability_utils import calc_rank
from utils.eval_utils import calculate_kendall_tau_distance_quick,\
                             subgroup_kt,\
                             change_in_cond_exp_rank,\
                             cond_exp_rank,\
                             compute_rKL,\
                             change_in_rKL,\
                             compute_igf_ratio,\
                             num_retained_at_top_k,\
                             change_in_percent_at_top_k,\
                             percent_change_in_percent_at_top_k,\
                             ratio_of_percent_at_top_k,\
                             prob_lower,\
                             prob_lower_group,\
                             prob_lower_group_ratio
                             


def calc_expected_distances(args):

    '''
    Can only handle 1 intervention (e.g. A<-bf)
    IGF-Ratio is working on rank not Y -- needs to be fixed
    '''

    # Highest seed not yet used in data gen
    seed = args.s_samples*((args.n_runs+1)*4+3)+args.initial_seed

    # Initialize arrays to hold distances
    noise_dists = np.zeros([len(args.metrics), args.s_samples, args.n_runs])
    counter_dists_xres = np.zeros([args.s_samples, len(args.metrics)])
    counter_dists_nonres = np.zeros([args.s_samples, len(args.metrics)])

    # List of all intersectional groups
    all_groups = ['wm', 'bm', 'am', 'wf', 'bf', 'af']
                
    # Loop through s_samples
    for s in range(1, args.s_samples+1):

        # Read in counterfactual data
        counter = pd.read_pickle(args.base_repo_dir /'out'/\
                                'counterfactual_data'/args.output_dir/
                                'counter_samp_{}.pkl'.format(s), compression='gzip')
        
        # Get list of counterfactual Y columns
        # depends on 2-letter group (e.g. "am" or "bf")
        counter_y_cols = [x for x in counter.columns if (('cf_y_' in x)&(x[-2:]==args.intervention_group))]

        assert len(counter_y_cols)>0, \
            'Found no counterfactual data for intervention group {}'.format(args.intervention_group)
        
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

        # Set group column based on which_group
        if args.which_group in all_groups:
            # Intersectional
            group_col = counter['original_a']
        elif (args.which_group=='f') or (args.which_group=='m'):
            # Sex column
            group_col = counter['original_a'].map(lambda x: x[1])
        elif ((args.which_group=='w') or (args.which_group=='b')) or (args.which_group=='a'):
            # Race column
            group_col = counter['original_a'].map(lambda x: x[0])
        else:
            # Throw error
            assert 1==2, 'Invalid which_group value.'

        # For "change" metrics, calculate original to measure against
        if ('change_in_cond_exp_rank' in args.metrics) and (args.which_group in group_col.unique()):
            orig_cond_exp_rank = cond_exp_rank(rank=orig_rank, groups=group_col, 
                                                    which_group=args.which_group, normalize=True)
        else:
            orig_cond_exp_rank = np.nan

                            
        if 'change_in_rkl' in args.metrics:
            orig_rkl = compute_rKL(rank=orig_rank, groups=group_col)
        else:
            orig_rkl = np.nan

        # Create list of partial functions for metrics
        metrics_dict = {'kendalls_tau': partial(calculate_kendall_tau_distance_quick,
                                                rank1=orig_rank), 
                        'subgroup_kt': partial(subgroup_kt, 
                                                rank1=orig_rank, 
                                                groups=group_col, 
                                                which_group=args.which_group),
                        'change_in_cond_exp_rank': partial(change_in_cond_exp_rank, 
                                                            exp1 = orig_cond_exp_rank,
                                                            groups=group_col, 
                                                            which_group=args.which_group,
                                                            normalize=True),
                        'change_in_rkl': partial(change_in_rKL, 
                                                    rKL1=orig_rkl, 
                                                    groups=group_col),
                        'igf_ratio': partial(compute_igf_ratio,
                                                groups=group_col,
                                                which_group=args.which_group,
                                                k=args.k),
                        'igf_ratio_full': partial(compute_igf_ratio,
                                                    k=args.k),
                        'num_retained_at_top_k': partial(num_retained_at_top_k, 
                                                            rank1=orig_rank, 
                                                            k=args.k), 
                        'change_in_percent_at_top_k': \
                            partial(change_in_percent_at_top_k, 
                                        rank1=orig_rank, 
                                        k=args.k, 
                                        groups=group_col, 
                                        which_group=args.which_group),
                        'percent_change_in_percent_at_top_k': \
                            partial(percent_change_in_percent_at_top_k, 
                                        rank1=orig_rank, 
                                        k=args.k, 
                                        groups=group_col, 
                                        which_group=args.which_group),
                        'ratio_of_percent_at_top_k': \
                            partial(ratio_of_percent_at_top_k, 
                                        rank1=orig_rank, 
                                        k=args.k, 
                                        groups=group_col, 
                                        which_group=args.which_group),
                        'prob_lower': partial(prob_lower, 
                                                rank1=orig_rank),
                        'prob_lower_group': partial(prob_lower_group, 
                                                        rank1=orig_rank, 
                                                        groups=group_col,
                                                        which_group=args.which_group),
                        'prob_lower_group_ratio': partial(prob_lower_group_ratio,  
                                                            rank1=orig_rank, 
                                                            groups=group_col, 
                                                            groupa=args.which_group, 
                                                            groupb=args.other_group)
                        }
        
        # List of partial functions of metrics
        metrics = [metrics_dict[m] for m in args.metrics]

        # List of metric names to automatically fill with NA
        na_metrics = []

        # Confirm which_group exists in sample
        if args.which_group not in group_col.unique():

            # List of metrics which depend on "which_group"
            which_group_metrics = ['subgroup_kt', 'change_in_cond_exp_rank', 'igf_ratio', 
                               'change_in_percent_at_top_k', 'percent_change_in_percent_at_top_k', 
                               'ratio_of_percent_at_top_k', 'prob_lower', 'prob_lower_group', 
                               'prob_lower_group_ratio']
                               
            # List of metrics to automatically fill with NA
            na_metrics = [metrics_dict[m] for m in args.metrics if m in which_group_metrics]

        # Confirm other_group exists in sample
        if args.other_group not in group_col.unique():

            # List of metrics which depend on "other_group"
            other_group_metrics = ['prob_lower_group_ratio']

            # List of metrics to automatically fill with NA
            na_metrics = na_metrics + [metrics_dict[m] for m in args.metrics if m in other_group_metrics]

        # Get distances between original rank and each rank from noise distribution
        # Averages will give expected noise distances for this sample
        for n in range(1, args.n_runs+1):
            for m, metric in enumerate(metrics):
                if metric in na_metrics:
                    # Auto-fill with NaN if necessary
                    noise_dists[m][s-1][n-1] = np.nan
                else:
                    noise_dists[m][s-1][n-1] = metric(rank2=noise['rank_{}'.format(n)])

        # Get distances between original rank and counterfactual ranks
        for m, metric in enumerate(metrics):
            if metric in na_metrics:
                # Auto-fill with NaN if necessary
                counter_dists_nonres[s-1][m] = np.nan
                counter_dists_xres[s-1][m] = np.nan
            else:
                # Get distance between original rank and counterfactual Y with resolving X
                counter_dists_nonres[s-1][m] = metric(rank2=counter['rank_nonres_'+args.intervention_group])
                
                # Get distance between original rank and counterfactual Y with non-resolving X
                counter_dists_xres[s-1][m] = metric(rank2=counter['rank_xres_'+args.intervention_group])

        print('Calculating {} distances for group {}: {} of {} samples complete'\
                .format(args.output_dir, args.which_group, s, args.s_samples))

    '''
    Get expected distances between original rank and rank from re-sampled noise
    Expectations taken over n_runs of noise distribution
    E[ distance (original rank, rank from re-sampled noise) ]
    1 value per metric for each sample
    '''
    exp_noise_dist = np.mean(noise_dists, axis=2)
    
    '''
    Get expected value of expected distances between original rank and rank from re-sampled noise
    Iterated expectation: expectation first taken over n_runs of noise distribution, then over s_samples
    E[ E[ distance(original rank, rank from re-sampled noise) ] ]
    1 value per metric for entire experiment trial
    '''
    exp_exp_noise_dist = np.mean(exp_noise_dist, axis=1)

    '''
    Get expected distance between original rank and counterfactual Y with non-resolving X
    For intervention A<-intervention_group
    Expectation taken over s_samples
    E[ distance(original rank, counterfactual rank with non-resolving X for A<-intervention_group) ]
    1 value per metric for entire experiment trial
    '''
    exp_cf_dist_nonres = np.nanmean(counter_dists_nonres, axis=0)


    '''
    Get expected distance between original rank and counterfactual Y with resolving X
    For intervention For intervention A<-intervention_group
    Expectation taken over s_samples
    E[ distance(original rank, counterfactual rank with resolving X for A<-intervention_group) ]
    1 value per metric for entire experiment trial
    '''
    exp_cf_dist_xres = np.nanmean(counter_dists_xres, axis=0)

    # Bundle expected distances into list of dictionaries
    exp_dicts = []
    for m, metric in enumerate(metrics):
        exp_dict = {'exp_exp_noise':exp_exp_noise_dist[m],
                    'exp_nonres': exp_cf_dist_nonres[m],
                    'exp_xres': exp_cf_dist_xres[m]}
        exp_dicts.append(exp_dict)

    # Bundle expected noise distances into 1d list of dataframes
    exp_noise_dfs = []
    for m, metric in enumerate(metrics):
        noise_df = pd.DataFrame(noise_dists[m, :, :], columns=['orig_noise'+str(i+1) for i in range(args.n_runs)])
        exp_noise_dfs.append(noise_df)
    
    # Bundle counterfactual distances into 1d list of dataframes
    cf_dist_dfs = []
    for m, metric in enumerate(metrics):
        cf_dist_df = pd.concat([ pd.DataFrame(exp_noise_dist[m], columns=['exp_orig_noise']),
                                pd.DataFrame(counter_dists_nonres[:,m], columns=['orig_nonres_'+args.intervention_group]), 
                                pd.DataFrame(counter_dists_xres[:,m], columns=['orig_xres_'+args.intervention_group]),
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
    parser.add_argument("--output_analysis_subdir", type=str, default=None)
    parser.add_argument("--sd_list", nargs='+', type=float, default=None)
    parser.add_argument("--corr_list", nargs='+', type=float, default=None)
    parser.add_argument("--initial_seed", type=int, default=0, help='Initial seed for race sample passed to gen_stability_data.py')

    parser.add_argument("--all_interventions", action='store_true', 
                        help='Boolean flag indicating whether all intervention counterfactuals will be evaluated.\
                              If not set, only intervention to minority group will be evaluated.')

    parser.add_argument("--all_metrics", action='store_true', 
                            help='Boolean flag indicating whether to evaluate\
                                  all distance metrics.\
                                  If not set, only those Kendall\'s Tau and \
                                  any others specifically set will be evaluated.')

    parser.add_argument("--subgroup_kt", action='store_true', 
                        help='Boolean flag indicating whether to additionally evaluate\
                              the distance metric subgroup_kt.\
                              If not set, subgroup_kt will not be evaluated.')

    parser.add_argument("--change_in_cond_exp_rank", action='store_true', 
                        help='Boolean flag indicating whether to additionally evaluate\
                              the distance metric change_in_cond_exp_rank.\
                              If not set, change_in_cond_exp_rank will not be evaluated.')

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

    parser.add_argument("--change_in_rkl", action='store_true', 
                        help='Boolean flag indicating whether to additionally evaluate\
                              the distance metric change in rKL.\
                              If not set, change in rKL will not be evaluated.')

    parser.add_argument("--igf_ratio", action='store_true', 
                        help='Boolean flag indicating whether to additionally evaluate\
                              the distance metric IGF-Ratio.\
                              If not set, IGF-Ratio will not be evaluated.')

    parser.add_argument("--igf_ratio_full", action='store_true', 
                        help='Boolean flag indicating whether to additionally evaluate\
                              the distance metric IGF-Ratio, where the group is the full dataset.\
                              If not set, full IGF-Ratio will not be evaluated.')

    parser.add_argument("--no_kendalls_tau", action='store_true', 
                        help='Boolean flag indicating whether to exclude Kendall\'s Tau from evaluation.\
                            If not set, Kendall\'s Tau will be evaluated.')
    
    parser.add_argument("--intervention_group", type=str, default='bf',
                        help='which group was used for counterfactual intervention A<-a')

    parser.add_argument("--which_group", type=str, default='bf',
                        help='which group to use for group-specific metrics')

    parser.add_argument("--other_group", type=str, default='wm',
                        help='which group to use as comparison group in group-specific metrics')
    
    parser.add_argument("--k", type=int, default=0,
                        help='Integer to use for "top k" evaluations.')            

    # Parse arguments
    args = parser.parse_args()

    if args.prob_lower_group_ratio:
        assert args.which_group!=args.other_group, \
            'Must provide 2 different groups for comparison in prob_lower_group_ratio.'

    if args.k==0:
        args.k=None

    # Create list of names of distance functions
    args.metrics = []
    if not args.no_kendalls_tau:
        args.metrics.append('kendalls_tau')
    if args.subgroup_kt| args.all_metrics:
        args.metrics.append('subgroup_kt')
    if args.change_in_cond_exp_rank| args.all_metrics:
        args.metrics.append('change_in_cond_exp_rank')
    if args.change_in_rkl | args.all_metrics:
        args.metrics.append('change_in_rkl')
    if args.igf_ratio | args.all_metrics:
        args.metrics.append('igf_ratio')
    if args.igf_ratio_full | args.all_metrics:
        args.metrics.append('igf_ratio_full')
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

    # Directory to save metrics to
    save_dir = main_output_dir
    if args.output_analysis_subdir:
        save_dir = save_dir+'/'+args.output_analysis_subdir

    # Initialize list of lists to hold expected distances
    exp_dicts_by_metric = [[] for m in range(len(args.metrics))]

    # If specific corr_list and sd_list provided
    # Loop through combinations in nested loop
    if args.corr_list and args.sd_list:
        for corr in args.corr_list:
            for sd in args.sd_list:
                args.output_dir = main_output_dir +'/corr_{}/sd_{}'.format(corr,sd)
                exp_noise_dfs, cf_dist_dfs, exp_dicts = calc_expected_distances(args)

                filename = 'corr_{}_sd_{}.pkl'.format(corr, sd)
                for m, exp_noise_df in enumerate(exp_noise_dfs):
                    exp_noise_df.to_pickle(args.base_repo_dir/'out'/'distance_metrics'/\
                                            save_dir/args.metrics[m]/'noise'/\
                                            filename, 
                                            compression='gzip')

                for m, cf_dist_df in enumerate(cf_dist_dfs):
                    cf_dist_df.to_pickle(args.base_repo_dir/'out'/'distance_metrics'/\
                                            save_dir/args.metrics[m]/'counterfactuals'/\
                                            filename, 
                                            compression='gzip')

                for m, exp_dict in enumerate(exp_dicts):
                    exp_dicts_by_metric[m].append(exp_dict)

                print('Distance calculation for {} complete\n'.format(args.output_dir))

        for m, exp_dict in enumerate(exp_dicts):
            # Create expected distance df with multi-index of subdirectories
            exp_df = pd.DataFrame(list(np.array(exp_dicts_by_metric[m])), 
                                index=pd.MultiIndex.from_tuples(\
                                        list(product(['corr_{}'.format(x) for x in args.corr_list], 
                                                     ['sd_{}'.format(x) for x in args.sd_list]))))

            # Save expected distance df to pickle
            exp_df.to_pickle(args.base_repo_dir/'out'/'distance_metrics'/\
                                    save_dir/args.metrics[m]/'expected.pkl', 
                                compression='gzip')

    else:
        # Initialize list of lists of experiment subdirectories
        sub_dirs = []

        # Define path to traverse
        output_path = args.base_repo_dir /'out'/'counterfactual_data'/ main_output_dir  

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
                                            save_dir/args.metrics[m]/'noise'/\
                                            filename, 
                                            compression='gzip')

                for m, cf_dist_df in enumerate(cf_dist_dfs):
                    cf_dist_df.to_pickle(args.base_repo_dir/'out'/'distance_metrics'/\
                                            save_dir/args.metrics[m]/'counterfactuals'/\
                                            filename, 
                                            compression='gzip')

                for m, exp_dict in enumerate(exp_dicts):
                    exp_dicts_by_metric[m].append(exp_dict)

                print('Distance calculation for {} complete\n'.format(args.output_dir))
                
        # Get list of indices which lex-sort list of lists of subdirectories
        dirs_sort = np.lexsort(tuple([np.array(sub_dirs).T[i] \
                                    for i in np.flip(np.arange(np.array(sub_dirs).shape[1]))]))

        for m, exp_dict in enumerate(exp_dicts):
            # Create expected distance df with multi-index of lex-sorted subdirectories
            exp_df = pd.DataFrame(list(np.array(exp_dicts_by_metric[m])[dirs_sort]), 
                                index=pd.MultiIndex.from_tuples(sorted(sub_dirs)))

            # Save expected distance df to pickle
            exp_df.to_pickle(args.base_repo_dir/'out'/'distance_metrics'/\
                                    save_dir/args.metrics[m]/'expected.pkl', 
                                compression='gzip')
