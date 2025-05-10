# Library packages
library(dplyr)
library(tidyverse)

# Set the working directory to the current script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load functions for DFR model and required Python scripts
function_path = "../../code"
function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                              ignore.case=TRUE)
sapply(function_sources, source, .GlobalEnv)
reticulate::source_python('../../code/DNN.py')

# Set up parallel processing
library(doRNG)
library(doParallel)
cl = makeCluster(60) 
length(cl)
registerDoParallel(cl)

# Define parameter grid
param_grid = expand.grid(n = c(100,200,500,1000), nu1 = c(0,0.5,1,1.5,2), rrr = 1:500)

# Use a single foreach loop
result = foreach(params = t(param_grid), .packages = c('tidyverse', 'dplyr', 'frechet', 'foreach', 'reshape2', 'vegan', 'reticulate')) %dorng% {
  
  n = params[1]  # Sample size
  nu1 = params[2] # mean parameter for normal distribution
  rrr = params[3]  # Replication index
  
  # Reload necessary functions for each iteration
  function_path = "../../code"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  reticulate::source_python('../../code/DNN.py')
  
  # Define simulation parameters
  N = 100  # Number of observations per sample
  nOut = 100  # Number of test samples
  sigma0 = 3  # Base standard deviation
  kappa = 1  # Shape parameter for gamma distribution
  
  set.seed(rrr)  # Set seed for reproducibility
  
  x = data.frame(X1 = runif(n, -1, 0), X2 = runif(n, 0, 1), X3 = runif(n, 1, 2),
                 X4 = rnorm(n, 1, 1), X5 = rnorm(n, -1, 1),
                 X6 = rnorm(n, 0, 3),
                 X7 = sample(c(0,1),n,TRUE,c(0.3,0.7)), 
                 X8 = sample(c(0,1),n,TRUE,c(0.4,0.6)), 
                 X9 = sample(c(0,1),n,TRUE,c(0.5,0.5)),
                 X10 = rbeta(n, 2, 2),
                 X11 = rbeta(n, 3, 2),
                 X12 = rbeta(n, 2, 3)
  )
  xout = data.frame(X1 = runif(nOut, -1, 0), X2 = runif(nOut, 0, 1), X3 = runif(nOut, 1, 2),
                    X4 = rnorm(nOut, 1, 1), 
                    X5 = rnorm(nOut, -1, 1),
                    X6 = rnorm(nOut, 0, 3),
                    X7 = sample(c(0,1),nOut,TRUE,c(0.3,0.7)), 
                    X8 = sample(c(0,1),nOut,TRUE,c(0.4,0.6)), 
                    X9 = sample(c(0,1),nOut,TRUE,c(0.5,0.5)),
                    X10 = rbeta(nOut, 2, 2),
                    X11 = rbeta(nOut, 3, 2),
                    X12 = rbeta(nOut, 2, 3)
  )
  
  # Generate response data
  y = list()
  yMean <- matrix(nrow = n, ncol = N-1)
  mu.sd = NULL
  for(i in 1:n) {
    # Generate expected mean and standard deviation
    expect_eta_Z = 3 * x[i,10] + 
      3 * (cos(pi * x[i,1]) + sin(pi * x[i,2])) * x[i,8] +
      (x[i,4] + x[i,5]) * x[i,7]
    
    expect_sigma_Z = 3 * x[i,11] + 
      2*(sin(pi * x[i,1]) + cos(pi * x[i,2])) * x[i,8] +
      (x[i,4]^2 + x[i,5]^2) * x[i,7]
    
    # Sample mean and standard deviation
    mu = rnorm(1, mean = expect_eta_Z, sd = nu1)
    sigma = expect_sigma_Z
    
    # Generate response and true quantile values
    y[[i]] = sort(mu + sigma * rnorm(N)) # rnorm(N, mean = mu, sd = sigma)
    yMean[i, ] = expect_eta_Z + expect_sigma_Z * qnorm(c(1:(N-1))/N)
    
    mu.sd = rbind(mu.sd, data.frame(mu, sigma))
  }
  
  # Generate true quantile values for test data
  yMeanOut = t(apply(xout, 1, function(xOuti) {
    3 * xOuti[10] + 3 * (cos(pi * xOuti[1])+sin(pi * xOuti[2])) * xOuti[8] +
      (xOuti[4] + xOuti[5]) * xOuti[7] +
      (3 * xOuti[11] + 2*(sin(pi * xOuti[1])+cos(pi * xOuti[2]))*xOuti[8] +
      (xOuti[4]^2 + xOuti[5]^2) * xOuti[7]) * qnorm(1:(N - 1) / N)
  }
    ))
  
  # DFR
  res_dfr = DFR(y = y, x = x, xout = xout, 
                   optns = list(type = "measure", manifold = list(method = "isomap", k = ifelse(n == 100, 20, 0.1*n)), r = 2, 
                                layer = 4, hidden = ifelse(n == 100, 32, 64), dropout = 0.3, lr = 0.0005, 
                                num_epochs = 2000, seed = rrr))
  
  # Calculate prediction errors
  err_q = data.frame(
    n = n, r = rrr, nu1 = nu1,
    DFR = sum((res_dfr$yPred[, -N] - yMeanOut)^2) / ((N - 1) * nOut)
  )
  
  return(err_q)
}

# Stop the parallel cluster
stopCluster(cl)

# Combine results and summarize
err = do.call(rbind, result)

# Summarize and display mean squared prediction error (MSPE)
err %>%
  select(n, nu1, DFR) %>%
  gather(key = "method", value = Error, -c("n", "nu1")) %>%
  group_by(n, nu1, method) %>%
  dplyr::summarise(MSPE = mean(Error)) %>%
  mutate(method = factor(method, levels = c("DFR"))) %>%
  dplyr::select(n, nu1, method, MSPE) %>%
  spread(key = method, value = MSPE) %>%
  knitr::kable(digits = 3)
