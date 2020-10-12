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
        counter = pd.read_csv(args.base_repo_dir /'out'/\
                                'counterfactual_data'/'stability'/args.output_dir/
                                'counter_samp_{}.csv'.format(s))
        
        # Get list of counterfactual Y columns
        counter_y_cols = [x for x in counter.columns if 'cf_y_' in x]
        
        # Calculate ranks for each counterfactual Y
        for y in counter_y_cols:
            counter['rank_'+y[5:]] = calc_rank(seed=seed, y=counter[y])
            seed += 1
        
        # Read in noise distribution data
        noise = pd.read_csv(args.base_repo_dir /'out'/\
                            'synthetic_data'/'stability'/args.output_dir/\
                            'rankings'/'observed_samp_{}.csv'.format(s))
        
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
    
    exp_dict = {'exp_exp_noise': exp_exp_kt_noise, 
                'exp_nonres0': exp_kt_counter_nonres_a0,
                'exp_nonres1': exp_kt_counter_nonres_a1,
                'exp_xres0': exp_kt_counter_xres_a0,
                'exp_xres1': exp_kt_counter_xres_a1}

    noise_kt_df = pd.DataFrame(noise_kts, columns=['orig_noise'+str(i+1) for i in range(args.n_runs)])
    
    cf_kt_df = pd.concat([ pd.DataFrame(exp_kt_noise, columns=['exp_orig_noise']),
                            pd.DataFrame(counter_kts_nonres, columns=['orig_nonres0', 'orig_nonres1']), 
                            pd.DataFrame(counter_kts_xres, columns=['orig_xres0', 'orig_xres1']), 
                            pd.DataFrame(kts_nonres_a0_a1, columns=['nonres0_nonres1']),
                            pd.DataFrame(kts_xres_a0_a1, columns=['xres0_xres1']),
                        ], axis=1)

    return (noise_kt_df, cf_kt_df, exp_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get Counterfactual Data")

    # Optional argument
    parser.add_argument("--output_dir", type=str, default='default')

    args = parser.parse_args()

    args.base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]

    exp_dicts = []
    noise_kt_dfs = []
    cf_kt_dfs = []

    # Remove this hard coding
    ERR_INPUTS=["x", "y", "xy_ind", "xy_conf"]
    ERR_SDS=[0.0, 0.1, 0.2]

    # Convert below into oswalk
    for err_input in ERR_INPUTS:
        for err_sd in ERR_SDS:
            args.output_dir = "err_sd/{}/{}".format(err_input, err_sd)

            noise_kt_df, cf_kt_df, exp_dict = calc_expected_kendalls_taus(args)
            
            exp_dicts.append(exp_dict)
            noise_kt_dfs.append(noise_kt_df)
            cf_kt_dfs.append(cf_kt_df)
