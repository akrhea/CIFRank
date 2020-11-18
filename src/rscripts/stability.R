#!/usr/bin/env Rscript

# Script to be run from bash script located in CIFR_Stability repo root dir

# Take in arguments from command line
#args = commandArgs(trailingOnly=TRUE)
#s_samples <- args[1]
#output_dir <- args[2]

s_samples <- 10
output_dir <- 'default'

# Use "here" library to get path to repo root directory
library(here)

# Vector of group names stored in A
groups <- c('wm', 'bm', 'am', 'wf', 'bf', 'af')

# Initialize dataframe to hold model parameters for node X
params.x <- data.frame(wm=numeric(s_samples),
                       bm=numeric(s_samples),
                       am=numeric(s_samples),
                       wf=numeric(s_samples),
                       bf=numeric(s_samples),
                       af=numeric(s_samples))

# Initialize dataframe to hold model parameters for node Y
params.y <- data.frame(x=numeric(s_samples),
                       wm=numeric(s_samples),
                       bm=numeric(s_samples),
                       am=numeric(s_samples),
                       wf=numeric(s_samples),
                       bf=numeric(s_samples),
                       af=numeric(s_samples))

for (i in 1:s_samples) {

  data_i <- read.csv(here('out','synthetic_data',output_dir,'data',
                          paste('samp_', i, '.csv', sep='')))

   # estimate model for x with lin reg
   model.x <- lm(x ~ a, data = data_i)
   coefs.x <- data.frame(coef(summary(model.x)))
  
   # estimate model for x with lin reg
   model.y <- lm(y ~ x+a, data = data_i)
   coefs.y <- data.frame(coef(summary(model.y)))
   
   # Get estimated coefficients for each group
   for (group in groups){
     params.x[i,group]<- ifelse(
                           !is.na(coefs.x[paste('a',group,sep=''), 'Estimate']),
                           coefs.x[paste('a',group,sep=''), 'Estimate'],
                           coefs.x['(Intercept)', 'Estimate'])
     
     params.y[i,group]<- ifelse(
                           !is.na(coefs.y[paste('a',group,sep=''), 'Estimate']),
                           coefs.y[paste('a',group,sep=''), 'Estimate'],
                           coefs.y['(Intercept)', 'Estimate'])
     
   }
   
   # Get estimated coefficient for X
   params.x[i,'x'] <- coefs.y['x', 'Estimate']

  print(paste("Done causal model estimation of synthetic data trial ", i))
}

# save dataframe of x parameters to csv
write.csv(params.x, file = here('out','parameter_data',
                               output_dir,'params_x.csv'))


# save dataframe of y parameters to csv
write.csv(params.y, file = here('out','parameter_data',
                               output_dir,'params_y.csv'))
