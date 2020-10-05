# args = commandArgs(trailingOnly=TRUE)
#output_p <- args[1]
# run_n <- args[2]
run_n <- 100

for (i in 1:run_n) {
  
  # need to update filepath formats if script run from bash script
  data_i <- read.csv(paste(dirname(dirname(getwd())),
                   '/out/synthetic_data/stability/default/data/observed_samp_',
                   i,'.csv',
                   sep=''))
  
  # estimate x parameters with lin reg
  model.x <- lm(x ~ a - 1, data = data_i)
  
  # save x parameters to csv
  # need to save anything except estimate? [ Std. error, t value, Pr(>|t|) ]
  write.csv(data.frame(summary(model.x)$coefficients),
           file=paste(dirname(dirname(getwd())),
                      '/out/parameter_data/stability/default/',
                      "R", i, "_x.csv", sep=''))
  
  # estimate y parameters with lin reg
  model.y <- lm(y ~ x + a - 1 , data = data_i)
  
  # save t parameters to csv
  write.csv(data.frame(summary(model.y)$coefficients),
            file=paste(dirname(dirname(getwd())),
                       '/out/parameter_data/stability/default/',
                       "R", i, "_y.csv", sep=''))

  print(paste("Done causal model estimation of synthetic data trial ", i))
}

