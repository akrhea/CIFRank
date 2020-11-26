#!/bin/bash

# This script calculates group-specific metrics for each group
# for 3 key levels of error standard deviation (0, 1, and 2)
# and for 2 levels of error correlation (0 and 1)

EXPER_NAME="inter_corr"

S_SAMPLES=100
N_RUNS=100
M_ROWS=100

# Array of error correlations to test
ERR_CORRS=(0.0 1.0)

# Array of error standard deviations to test
ERR_SDS=(0.0 1.0 2.0)

# Array of metrics names to test
METRICS=("subgroup_kt" \
         "change_in_cond_exp_rank" \
         "change_in_rkl")

IDENTITY_GROUPS=("wm" "bm" "wf" "bf" "m" "f" "w" "b" "am" "af" "a")

for group in ${IDENTITY_GROUPS[@]}
do
    for metric in ${METRICS[@]}
    do
        # Make output directories
        mkdir -p "out/distance_metrics/$EXPER_NAME/groups/$group/$metric/noise/"
        mkdir -p "out/distance_metrics/$EXPER_NAME/groups/$group/$metric/counterfactuals/"
    done

    # Calculate distances
    python -W ignore src/pyscripts/calc_stability_distances.py \
    --s_samples $S_SAMPLES --n_runs $N_RUNS \
     --sd_list "${ERR_SDS[@]}" --corr_list "${ERR_CORRS[@]}" \
    --output_dir $EXPER_NAME --output_analysis_subdir "groups/$group" \
    --which_group $group --no_kendalls_tau\
    --change_in_rkl --subgroup_kt --change_in_cond_exp_rank
done

