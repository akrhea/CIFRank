#!/bin/bash


###########################################################
#						                                  #
# INPUT SETTING for experiment, change based on dataset   #
#						                                  #
###########################################################
OUTPUT_DIR='default'
S_SAMPLES=200
N_RUNS=500
M_ROWS=10


#################################################
#						                        #
# Functions for complete routine of experiments	#
#						                        #
#################################################

#################################################
#						                        #
# Functions to generate data	                #
#						                        #
#################################################
python src/pyscripts/gen_stability_data.py --s_samples $S_SAMPLES --n_runs $N_RUNS --m_rows $M_ROWS --x_err_input True --y_err_input True --output_dir $OUTPUT_DIR

#########################################################
#						                                #
# Function to estimate causal model on the data,        #
# causal model is specified in 'rscripts'               #
#                                                       #
# Not working						                    #
#                                                       #
#########################################################
# Rscript --vanilla src/rscripts/stability.R --s_samples $S_SAMPLES --output_dir $OUTPUT_DIR


############################################################################
#						                                                   #
# Functions to get counterfactual data from estimated causal model         #
#						                                                   #
############################################################################

python src/pyscripts/gen_counter_data_stability.py --output_dir $OUTPUT_DIR

