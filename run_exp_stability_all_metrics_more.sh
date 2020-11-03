#!/bin/bash

# This script generates a medium amount of data
# to test out the use of eight new metrics
# across 4 DAG configurations and 11 error SDs.

# Differs from all_metrics because uses 2x all params
# 25 rows --> 50 rows
# 50 n_runs --> 100 n_runs
# 100 s_samples --> 200 s_samples

EXPER_NAME="all_metrics_more"
S_SAMPLES=200
N_RUNS=100
M_ROWS=50

# Array of error standard deviations to test
ERR_SDS=(0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)

# Array of error inputs to test
ERR_INPUTS=("x" "y" "xy_ind" "xy_conf")

# Array of metrics names to test
METRICS=("kendalls_tau" \
        "num_retained_at_top_k" \
        "change_in_percent_at_top_k" \
        "percent_change_in_percent_at_top_k" \
        "ratio_of_percent_at_top_k" \
        "prob_lower" \
        "prob_lower_group" \
        "prob_lower_group_ratio")
                             

for err_input in ${ERR_INPUTS[@]}
do
    echo "BEGIN ERR_INPUT=$err_input"
    for err_sd in ${ERR_SDS[@]}
    do
        echo "BEGIN ERR_INPUT=$err_input, ERR_SD=$err_sd"

        # Set output directory 
        output_dir="$EXPER_NAME/$err_input/$err_sd"

        # Create necessary directories
        mkdir -p "out/synthetic_data/$output_dir/data/"
        mkdir -p "out/synthetic_data/$output_dir/noise_rankings/"
        mkdir -p "out/parameter_data/$output_dir/"
        mkdir -p "out/counterfactual_data/$output_dir/"
        for metric in ${METRICS[@]}
        do
            mkdir -p "out/distance_metrics/$EXPER_NAME/$metric//noise/"
            mkdir -p "out/distance_metrics/$EXPER_NAME/$metric/counterfactuals/"
        done

        if [ "$err_input" = "x" ]
        then
            # Generate data with error node as parent of X
            python src/pyscripts/gen_stability_data.py \
            --s_samples $S_SAMPLES --n_runs $N_RUNS \
            --m_rows $M_ROWS --output_dir $output_dir \
            --x_err_input --x_err_sd $err_sd \
            --x_shared_err_weight 0
        elif [ "$err_input" = "y" ]
        then
            # Generate data with error node as parent of Y
            python src/pyscripts/gen_stability_data.py \
            --s_samples $S_SAMPLES --n_runs $N_RUNS \
            --m_rows $M_ROWS --output_dir $output_dir \
            --y_err_input --y_err_sd $err_sd \
            --y_shared_err_weight 0
        elif [ "$err_input" = "xy_ind" ]
        then
            # Generate data with X and Y each having seperate noise parents
            python src/pyscripts/gen_stability_data.py \
            --s_samples $S_SAMPLES --n_runs $N_RUNS \
            --m_rows $M_ROWS --output_dir $output_dir \
            --x_err_input --x_err_sd $err_sd \
            --y_err_input --y_err_sd $err_sd \
            --x_shared_err_weight 0 --y_shared_err_weight 0
        elif [ "$err_input" = "xy_conf" ]
        then
            # Generate data with X and Y sharing a single noise parent
            python src/pyscripts/gen_stability_data.py \
            --s_samples $S_SAMPLES --n_runs $N_RUNS \
            --m_rows $M_ROWS --output_dir $output_dir \
            --x_err_input --y_err_input
            --shared_err_sd $err_sd --x_err_sd $err_sd --y_err_sd $err_sd \
            --x_shared_err_weight 1 --y_shared_err_weight 1
        fi

        # Estimate causal model on the data
        Rscript --vanilla src/rscripts/stability.R $S_SAMPLES $output_dir

        # Get counterfactual data from estimated causal model
        python src/pyscripts/gen_counter_data_stability.py --output_dir $output_dir

    done
done

# Calculate Kendall's Tau distances
python src/pyscripts/calc_stability_distances.py \
--s_samples $S_SAMPLES --n_runs $N_RUNS \
--output_dir $EXPER_NAME --all_metrics