# # Uncomment to take in arguments
# args = commandArgs(trailingOnly=TRUE)
# s_samples <- args[1]
# output_dir <- args[2]

s_samples <- 200
output_dir <- 'default'

# Initialize dataframe to hold model parameters for node X
params.x <- data.frame(a=numeric(s_samples),
                          intercept=numeric(s_samples))

# Initialize dataframe to hold model parameters for node Y
params.y <- data.frame(a=numeric(s_samples),
                          x=numeric(s_samples),
                          intercept=numeric(s_samples))

for (i in 1:s_samples) {

  # need to update filepath formats if script run from bash script
  data_i <- read.csv(paste(dirname(dirname(getwd())),
                   '/out/synthetic_data/stability/', output_dir,
                   '/data/observed_samp_', i,'.csv', sep=''))

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
write.csv(params.x, file=paste(dirname(dirname(getwd())),
                               '/out/parameter_data/stability/', output_dir,
                               '/params_x.csv', sep=''))

# save dataframe of y parameters to csv
write.csv(params.y, file=paste(dirname(dirname(getwd())),
                               '/out/parameter_data/stability/', output_dir,
                               '/params_y.csv', sep=''))

