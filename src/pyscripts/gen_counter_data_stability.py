import pandas as pd
import os, argparse, pathlib, itertools
from utils.basic_utils import writeToCSV


def get_counterfactual_data(args):

    # Create list of groups to produce counterfactual data for
    # BF included as default, removed only if no_cf_bf flag is set
    cf_groups = ['bf']
    if args.no_cf_bf:
        cf_groups.remove('bf')
    if args.cf_wm:
        cf_groups.append('wm')
    if args.cf_bm:
        cf_groups.append('bm')
    if args.cf_am:
        cf_groups.append('am')
    if args.cf_wf:
        cf_groups.append('wf')
    if args.cf_af:
        cf_groups.append('af')

    assert len(cf_groups)>0, \
            'Must select at least 1 group to calculate counterfactual intervention for'
    
    # Define base repo directory
    try:
        # When executed directly from script
        base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    except:
        # When executed from jupyter notebook
        base_repo_dir  = pathlib.Path(os.getcwd()).parents[1]
        
    # Read in estimated parameters
    param_dir = base_repo_dir / 'out/parameter_data/{}'.format(args.output_dir)    
    y_params_df = pd.read_csv(param_dir / 'params_y.csv', index_col='Unnamed: 0')
    x_params_df = pd.read_csv(param_dir / 'params_x.csv', index_col='Unnamed: 0')

    # Get number of samples drawn from parameter files
    s_samples = len(y_params_df)
    
    # Assert parameter files are the same length
    assert s_samples == len(x_params_df), \
           'Parameter files indicate different number of samples.'
    
    # Loop through s_samples
    for s in range(1, s_samples + 1):
        
        # Read in this sample of the data
        this_df = pd.read_csv(base_repo_dir / 'out/synthetic_data/{}/data/samp_{}.csv'\
                                                                            .format(args.output_dir, s))
        
        # Intialize dataframe to hold counterfactual data
        counter_df = pd.DataFrame({'original_a':this_df['a'], 'original_x':this_df['x'], 'original_y':this_df['y']})
        
        # Isolate parameters estimated from this sample
        x_params = x_params_df.loc[s]
        y_params = y_params_df.loc[s]
        
        # Calcuate X-residuals
        x_residuals = this_df['x'] - this_df['a'].map(x_params)

        # Calcuate Y-residuals
        y_residuals = this_df['y'] - (this_df['a'].map(y_params) + \
                                      y_params['x']*this_df['x'])
        
        # Loop through groups to produce CF data for
        for group in cf_groups:

            if group not in this_df['a'].unique():
                if len(cf_groups)==1:
                    new_group = this_df['a'].unique()[0]
                    print('WARNING! {} group not present in sample. Cannot perform requested counterfactual intervention. \
                            Will performing A<-{} intevention instead.'.format(group, new_group))
                    group=new_group
                else:
                    print('WARNING! {} group not present in sample. Cannot perform requested counterfactual intervention. \
                            Moving on to next intervention group.'.format(group))
            
            # Get baseline X prediction for A <- group
            counter_base_x = x_params[group]

            # Estimate counterfactual X
            counter_x = counter_base_x + x_residuals

            # Get baseline Y prediction for A <- group for non-resolving X
            counter_base_y_nonres = counter_x*y_params['x'] + \
                                    y_params[group]
                                    
            
            # Estimate counterfactual Y for non-resolving X
            counter_y_nonres = counter_base_y_nonres + y_residuals

            # Get baseline Y prediction for A <- group for resolving X
            counter_base_y_res = y_params['x']*this_df['x'] + \
                                 y_params[group]
            
            # Estimate counterfactual Y for resolving X
            counter_y_res = counter_base_y_res + y_residuals
            
            # Save counterfactual Y for non-resolving X
            counter_df['cf_y_nonres_{}'.format(group)] = counter_y_nonres
            
            # Save counterfactual Y for resolving X
            counter_df['cf_y_xres_{}'.format(group)] = counter_y_res

        # Save counterfactual data for this sample to CSV
        counter_df.to_pickle(base_repo_dir / 'out/counterfactual_data/{}/counter_samp_{}.pkl'\
                                                        .format(args.output_dir, s), compression='gzip')

        print('Finished generating counterfactual data for sample {} of {}'.format(s, s_samples))

    return 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get Counterfactual Data")

    # Optional argument
    parser.add_argument("--output_dir", type=str, default='default')
    parser.add_argument("--no_cf_bf", action='store_true', 
                        help='boolean flag, will NOT calculate counterfactual data for A<-black female if set to True')
    parser.add_argument("--cf_bm", action='store_true', 
                        help='boolean flag, will calculate counterfactual data for A<-black male if set to True')
    parser.add_argument("--cf_wm", action='store_true', 
                        help='boolean flag, will calculate counterfactual data for A<-white male if set to True')
    parser.add_argument("--cf_am", action='store_true', 
                        help='boolean flag, will calculate counterfactual data for A<-asian male if set to True')
    parser.add_argument("--cf_wf", action='store_true', 
                        help='boolean flag, will calculate counterfactual data for A<-white female if set to True')
    parser.add_argument("--cf_af", action='store_true', 
                        help='boolean flag, will calculate counterfactual data for A<-asian female if set to True')
    args = parser.parse_args()
    get_counterfactual_data(args)
