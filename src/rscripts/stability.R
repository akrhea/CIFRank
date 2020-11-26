#!/usr/bin/env Rscript

# Script to be run from bash script located in CIFR_Stability repo root dir

# Take in arguments from command line
args = commandArgs(trailingOnly=TRUE)
s_samples <- args[1]
output_dir <- args[2]

# Uncomment below for testing in RStudio
# s_samples <- 100
# output_dir <- 'inter_corr/corr_0.5/sd_0.6'

# Use "here" library to get path to repo root directory
library(here)

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

# Uncomment below for testing in RStudio
# for (i in 1:1) {

for (i in 1:s_samples) {

   data_i <- read.csv(here('out','synthetic_data',output_dir,'data',
                          paste('samp_', i, '.csv', sep='')))
   
   # Uncomment below for testing in RStudio
   #data_i <- read.csv('/Users/AKR/CIFR_Ind_Stud/CIFRank_Stability/out/synthetic_data/inter_corr/corr_0.5/sd_1.0/data/samp_1.csv')
   
   # estimate model for x with lin reg
   model.x <- lm(x ~ a-1, data = data_i)
   coefs.x <- data.frame(coef(summary(model.x)))

   # estimate model for x with lin reg
   model.y <- lm(y ~ x+a-1, data = data_i)
   coefs.y <- data.frame(coef(summary(model.y)))
   
   # interaction term models (alternative to above)
   # model.x.rs <- lm(x ~ r*s-1, data = data_i)
   # coefs.x.rs <- data.frame(coef(summary(model.x.rs)))
   # model.y.rs <- lm(y ~ x+r*s-1, data = data_i)
   # coefs.y.rs <- data.frame(coef(summary(model.y.rs)))
   
   # Get estimated coefficients for each group present in sample
   groups <- unique(data_i$a)
   for (group in groups){
     params.x[i,group]<- ifelse(
                           !is.na(coefs.x[paste('a',group,sep=''), 'Estimate']),
                           coefs.x[paste('a',group,sep=''), 'Estimate'], 0)

     params.y[i,group]<- ifelse(
                           !is.na(coefs.y[paste('a',group,sep=''), 'Estimate']),
                           coefs.y[paste('a',group,sep=''), 'Estimate'], 0)
   }
   
   # Get estimated coefficient for X
   params.y[i,'x'] <- coefs.y['x', 'Estimate']

  print(paste("Done causal model estimation of synthetic data trial ", i))
}

# save dataframe of x parameters to csv
write.csv(params.x, file = here('out','parameter_data',
                               output_dir,'params_x.csv'))

# save dataframe of y parameters to csv
write.csv(params.y, file = here('out','parameter_data',
                               output_dir,'params_y.csv'))