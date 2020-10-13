#!/usr/bin/env Rscript

# Script to be run from bash script located in CIFR_Stability repo root dir

# Take in arguments from command line
args = commandArgs(trailingOnly=TRUE)
s_samples <- args[1]
output_dir <- args[2]

# Use "here" library to get path to repo root directory
library(here)

# Initialize dataframe to hold model parameters for node X
params.x <- data.frame(a=numeric(s_samples),
                          intercept=numeric(s_samples))

# Initialize dataframe to hold model parameters for node Y
params.y <- data.frame(a=numeric(s_samples),
                          x=numeric(s_samples),
                          intercept=numeric(s_samples))

for (i in 1:s_samples) {

  # need to update filepath formats if script run from bash script
  data_i <- read.csv(here('out','synthetic_data',output_dir,'data', 
                          paste('samp_', i, '.csv', sep='')))

  # estimate model for x with lin reg
  model.x <- lm(x ~ a, data = data_i)
  coefs.x <- coef(summary(model.x))

  params.x$a[i]  <- tryCatch({coefs.x['a', 'Estimate']},
                              error=function(e) { # catch error
                                message(paste('Attribute A may take only one ',
                                              'value in sample ',
                                              i, '.', sep=''))
                                message(paste('Original error message:', e))
                                return(0)})

  params.x$intercept[i] <- coefs.x['(Intercept)', 'Estimate']

  # estimate model for y with lin reg
  model.y <- lm(y ~ x + a, data = data_i)
  coefs.y <- coef(summary(model.y))

  params.y$a[i]  <- tryCatch({coefs.y['a', 'Estimate']},
                             error=function(e) { # catch error
                               message(paste('Attribute A may take only one ',
                                             'value in sample ',
                                             i, '.', sep=''))
                               message(paste('Original error message:', e))
                               return(0)})

  params.y$x[i]  <- tryCatch({coefs.y['x', 'Estimate']},
                             error=function(e) { # catch error
                               message(paste('Attribute X may take only one ',
                                             'value in sample ',
                                             i, '.', sep=''))
                               message(paste('Original error message:', e))
                               return(0)})

  params.y$intercept[i] <- coefs.y['(Intercept)', 'Estimate']

  print(paste("Done causal model estimation of synthetic data trial ", i))
}

# save dataframe of x parameters to csv
write.csv(params.x, file = here('out','parameter_data',
                                output_dir,'params_x.csv'))


# save dataframe of y parameters to csv
write.csv(params.y, file = here('out','parameter_data',
                                output_dir,'params_y.csv'))

