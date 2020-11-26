#!/bin/bash

# This script generates a medium-large amount of data
# across 11 error correlation levels and 18 error SDs.

# Data generation only (including counterfactual data).
# Does not calculate distances or fairness metrics.

# 4 values of (A) and the expected demographic breakdown: 
################# white male (wm): 0.7*0.6 = 42% of population
################# Black male (bm): 0.3*0.6 = 18% of population
################# white female (wf): 0.7*0.4 = 28% of population
################# Black female (bf): 0.3*0.4 = 12% of population

EXPER_NAME="inter_corr_4a"

S_SAMPLES=500
N_RUNS=200
M_ROWS=100

# Array of error correlations to test
ERR_CORRS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Array of error standard deviations to test
ERR_SDS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.5 3.0)
                             
for err_corr in ${ERR_CORRS[@]}
do
    echo "BEGIN ERR_CORR=$err_corr"
    for err_sd in ${ERR_SDS[@]}
    do
        echo "BEGIN ERR_CORR=$err_corr, ERR_SD=$err_sd"

        # Set output directory 
        output_dir="$EXPER_NAME/corr_$err_corr/sd_$err_sd"

        # Create necessary directories
        mkdir -p "out/synthetic_data/$output_dir/data/"
        mkdir -p "out/synthetic_data/$output_dir/noise_rankings/"
        mkdir -p "out/parameter_data/$output_dir/"
        mkdir -p "out/counterfactual_data/$output_dir/"

        # Generate data
        python src/pyscripts/gen_stability_data.py \
        --s_samples $S_SAMPLES --n_runs $N_RUNS \
        --m_rows $M_ROWS --output_dir $output_dir \
        --shared_err_sd $err_sd --x_err_sd $err_sd --y_err_sd $err_sd \
        --err_corr $err_corr --prob_white 0.7 --prob_black 0.3 --prob_asian 0.0

        # Estimate causal model on the data
        Rscript --vanilla src/rscripts/stability.R $S_SAMPLES $output_dir

        # Get counterfactual data from estimated causal model
        python src/pyscripts/gen_counter_data_stability.py --output_dir $output_dir

    done
done
