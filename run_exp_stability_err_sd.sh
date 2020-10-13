#!/bin/bash

# This script generates data for an experiment
# on the effect of the standard deviations of
# the X/Y error nodes on ranking stability

# Input settings
# S_SAMPLES=1000
# N_RUNS=500
# M_ROWS=50
S_SAMPLES=20
N_RUNS=5
M_ROWS=10

# Array of error standard deviations to test
#ERR_SDS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
ERR_SDS=(0.0 0.5 1.0)

# Array of error inputs to test
ERR_INPUTS=("x" "y" "xy_ind" "xy_conf")

for err_sd in ${ERR_SDS[@]}
do
    for err_input in ${ERR_INPUTS[@]}
    do
        echo "BEGIN ERR_INPUT=$err_input, ERR_SD=$err_sd"

        # Set output directory 
        output_dir="err_sd/$err_input/$err_sd"

        # Create necessary directories
        mkdir -p "out/synthetic_data/stability/$output_dir/data/"
        mkdir -p "out/synthetic_data/stability/$output_dir/noise_rankings/"
        mkdir -p "out/parameter_data/stability/$output_dir/"
        mkdir -p "out/counterfactual_data/stability/$output_dir/"
        mkdir -p "out/kendalls_tau_distances/err_sd/noise/"
        mkdir -p "out/kendalls_tau_distances/err_sd/counterfactuals/"

        if [ "$err_input" = "x" ]
        then
            # Generate data with error node as parent of X
            python src/pyscripts/gen_stability_data.py \
            --s_samples $S_SAMPLES --n_runs $N_RUNS \
            --m_rows $M_ROWS --output_dir $output_dir \
            --x_err_input --x_err_sd $err_sd
        elif [ "$err_input" = "y" ]
        then
            # Generate data with error node as parent of Y
            python src/pyscripts/gen_stability_data.py \
            --s_samples $S_SAMPLES --n_runs $N_RUNS \
            --m_rows $M_ROWS --output_dir $output_dir \
            --y_err_input --y_err_sd $err_sd
        elif [ "$err_input" = "xy_ind" ]
        then
            # Generate data with X and Y each having seperate noise parents
            python src/pyscripts/gen_stability_data.py \
            --s_samples $S_SAMPLES --n_runs $N_RUNS \
            --m_rows $M_ROWS --output_dir $output_dir \
            --x_err_input --x_err_sd $err_sd \
            --y_err_input --y_err_sd $err_sd 
        elif [ "$err_input" = "xy_conf" ]
        then
            # Generate data with X and Y sharing a single noise parent
            python src/pyscripts/gen_stability_data.py \
            --s_samples $S_SAMPLES --n_runs $N_RUNS \
            --m_rows $M_ROWS --output_dir $output_dir \
            --unmeasured_confounding \
            --x_err_sd $err_sd --y_err_sd $err_sd 
        fi

        # Estimate causal model on the data
        Rscript --vanilla src/rscripts/stability.R $S_SAMPLES $output_dir

        # Get counterfactual data from estimated causal model
        python src/pyscripts/gen_counter_data_stability.py --output_dir $output_dir

    done
done

# Calculate Kendall's Tau distances
python src/pyscripts/calc_stability_distances.py --output_dir err_sd --s_samples $S_SAMPLES --n_runs $N_RUNS