#!/bin/bash

# This script generates a medium amount of data
# to test out how the degree of correlation
# between X-epsilon and Y-epsilon
# effects top-four metrics


EXPER_NAME="corr_errors"
S_SAMPLES=200
N_RUNS=100
# S_SAMPLES=20
# N_RUNS=10

M_ROWS=50

ERR_SDS=(0.0 0.5 1.0 1.5 2.0)

SHARED_WGTS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
#SHARED_WGTS=(0.0 0.5 1.0)

# Array of metrics names to test
METRICS=("kendalls_tau" \
        "num_retained_at_top_k" \
        "change_in_percent_at_top_k" \
        "prob_lower" )

for shared_wgt in ${SHARED_WGTS[@]}  
do  
    echo "SHARED_WGT=$shared_wgt"   

    for err_sd in ${ERR_SDS[@]} 
    do    
        echo "BEGIN SHARED_WGT=$shared_wgt, ERR_SD=$err_sd"

        # Set output directory 
        output_dir="$EXPER_NAME/wgt_$shared_wgt/sd_$err_sd/"

        # Create necessary directories
        mkdir -p "out/synthetic_data/$output_dir/data/"
        mkdir -p "out/synthetic_data/$output_dir/noise_rankings/"
        mkdir -p "out/parameter_data/$output_dir/"
        mkdir -p "out/counterfactual_data/$output_dir/"
        for metric in ${METRICS[@]}
        do
            mkdir -p "out/distance_metrics/$EXPER_NAME/$metric/noise/"
            mkdir -p "out/distance_metrics/$EXPER_NAME/$metric/counterfactuals/"
        done

        # Generate data
        python src/pyscripts/gen_stability_data.py \
        --s_samples $S_SAMPLES --n_runs $N_RUNS \
        --m_rows $M_ROWS --output_dir $output_dir \
        --x_err_input --y_err_input \
        --shared_err_sd $err_sd --x_err_sd $err_sd --y_err_sd $err_sd \
        --x_shared_err_weight $shared_wgt --y_shared_err_weight $shared_wgt

        # Estimate causal model on the data
        Rscript --vanilla src/rscripts/stability.R $S_SAMPLES $output_dir

        # Get counterfactual data from estimated causal model
        python src/pyscripts/gen_counter_data_stability.py --output_dir $output_dir

    done
done

# Calculate Kendall's Tau distances
python src/pyscripts/calc_stability_distances.py \
--s_samples $S_SAMPLES --n_runs $N_RUNS \
--output_dir $EXPER_NAME \
--num_retained_at_top_k \
--change_in_percent_at_top_k \
--prob_lower