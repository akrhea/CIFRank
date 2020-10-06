import pandas as pd
import os, argparse, pathlib, itertools
from utils.basic_utils import writeToCSV


def get_counterfactual_data(args):
    
    # Define base repo directory
    try:
        # When executed directly from script
        base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    except:
        # When executed from jupyter notebook
        base_repo_dir  = pathlib.Path(os.getcwd()).parents[1]
        
    # Read in estimated parameters
    param_dir = base_repo_dir / 'out/parameter_data/stability/{}'.format(args.output_dir)    
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
        this_df = pd.read_csv(base_repo_dir / 'out/synthetic_data/stability/{}/data/observed_samp_{}.csv'\
                                                                            .format(args.output_dir, s))
        
        # Intialize dataframe to hold counterfactual data
        counter_df = pd.DataFrame({'original_a':this_df['a'], 'original_x':this_df['x'], 'original_y':this_df['y']})

        # Get list of groups in A (usually [0,1])
        group_list = [x for x in this_df['a'].unique()]
        
        # Isolate parameters estimated from this sample
        x_params = x_params_df.loc[s]
        y_params = y_params_df.loc[s]
        
        # Calcuate X-residuals
        x_residuals = this_df['x'] - ( x_params['a']*this_df['a'] + \
                                      x_params['intercept'] ).values

        # Calcuate Y-residuals
        y_residuals = this_df['y'] - ( y_params['a']*this_df['a'] + \
                                      y_params['x']*this_df['x'] + \
                                      y_params['intercept'] ).values
        
        # Loop through groups in A present in this sample
        for group in group_list:
            
            # Get baseline X prediction for A <- group
            counter_base_x = group*x_params['a'] + \
                             x_params['intercept']

            # Estimate counterfactual X
            counter_x = counter_base_x + x_residuals

            # Get baseline Y prediction for A <- group for non-resolving X
            counter_base_y_nonres = group*y_params['a'] + \
                                    counter_x*y_params['x'] + \
                                    y_params['intercept']
            
            # Estimate counterfactual Y for non-resolving X
            counter_y_nonres = counter_base_y_nonres + y_residuals

            # Get baseline Y prediction for A <- group for resolving X
            counter_base_y_res = group*y_params['a'] + \
                                 this_df['x']*y_params['x'] + \
                                 y_params['intercept']
            
            # Estimate counterfactual Y for resolving X
            counter_y_res = counter_base_y_res + y_residuals
            
            # Save counterfactual Y for non-resolving X
            counter_df['cf_y_nonres_a{}'.format(group)] = counter_y_nonres
            
            # Save counterfactual Y for resolving X
            counter_df['cf_y_xres_a{}'.format(group)] = counter_y_res

        # Save counterfactual data for this sample to CSV
        counter_df.to_csv(base_repo_dir / 'out/counterfactual_data/stability/{}/counter_samp_{}.csv'\
                                                        .format(args.output_dir, s), index=False)

        print('Finished generating counterfactual data for sample {} of {}'.format(s, s_samples))

        
        
    return 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get Counterfactual Data")

    # Optional argument
    parser.add_argument("--output_dir", type=str, default='default')

    args = parser.parse_args()
    get_counterfactual_data(args)
