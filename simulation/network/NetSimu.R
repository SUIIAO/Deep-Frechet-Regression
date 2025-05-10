# Library packages
library(dplyr)
library(ggplot2)
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
cl = parallel::detectCores() - 1 # Leave one core free
length(cl)
registerDoParallel(cl)

# Define parameter grid for simulations
param_grid = expand.grid(n = c(100,200,500,1000), rrr = 1:500)

# Perform simulations using foreach loop
result = foreach(params = t(param_grid), .packages = c('tidyverse', 'dplyr', 'frechet', 'foreach', 'reshape2', 'vegan', 'reticulate')) %dorng% {
  
  n = params[1]  # Sample size
  rrr = params[2]  # Replication index
  
  # Reload necessary functions for each iteration
  function_path = "../../code"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  reticulate::source_python('../../code/DNN.py')
  
  nOut = 100 # Number of test samples
  m = 10 # Number of nodes
  d = m*(m-1)/2 # Number of edges
  
  set.seed(rrr) # Set seed for reproducibility
  
  # Generate predictor data for training and test sets
  X = data.frame(
    X1 = runif(n,0,1), X2 = runif(n,-1/2,1/2), X3 = runif(n,1,2),
    X4 = rnorm(n,0,1), X5 = rnorm(n,0,1), X6 = rnorm(n,5,5),
    X7 = sample(c(0,1),n,TRUE,c(0.4,0.6)),
    X8 = sample(c(0,1),n,TRUE,c(0.3,0.7)),
    X9 = sample(c(0,1),n,TRUE,c(0.6,0.3))
  )
  xout = data.frame(
    X1 = runif(nOut,0,1), X2 = runif(nOut,-1/2,1/2), X3 = runif(nOut,1,2),
    X4 = rnorm(n,0,1), X5 = rnorm(n,0,1), X6 = rnorm(n,5,5),
    X7 = sample(c(0,1),nOut,TRUE,c(0.4,0.6)),
    X8 = sample(c(0,1),nOut,TRUE,c(0.3,0.7)),
    X9 = sample(c(0,1),nOut,TRUE,c(0.6,0.3))
  )
  
  # Generate response data
  L = list()
  LMean = list()
  true_ab = NULL
  for(i in 1:n){
    a = sin(pi*X[i,1])*X[i,8] + cos(pi*X[i,2])*(1-X[i,8])
    b = X[i,4]^2*X[i,7]+X[i,5]^2*(1-X[i,7])
    true_ab = rbind(true_ab, data.frame(a = a, b = b))
    
    bkVec = -rbeta(d, shape1 = a, shape2 = b)
    temp = matrix(0, nrow = m, ncol = m)
    temp[lower.tri(temp)] = bkVec
    temp = temp + t(temp)
    diag(temp) = -colSums(temp)
    L[[i]] = temp
    
    temp = matrix(0, nrow = m, ncol = m)
    temp[lower.tri(temp)] = -rep(a/(a+b), d)
    temp = temp + t(temp)
    diag(temp) = -colSums(temp)
    LMean[[i]] = temp
  }
  
  # Generate true quantile values for test data
  LMeanOut = list()
  for(i in 1:nOut){
    a = sin(pi*xout[i,1])*xout[i,8]+cos(pi*xout[i,2])*(1-xout[i,8])
    b = xout[i,4]^2*xout[i,7]+xout[i,5]^2*(1-xout[i,7])
    
    temp = matrix(0, nrow = m, ncol = m)
    temp[lower.tri(temp)] <- -rep(a/(a+b), d)
    temp = temp + t(temp)
    diag(temp) = -colSums(temp)
    LMeanOut[[i]] = temp
  }
  
  # DFR
  res_drf = DFR(y = L, x = X, xout = xout, 
                   optns = list(type = "network", manifold = list(method = "isomap", k = n*0.2), r = 2, 
                                layer = 4, hidden = 32, dropout = 0.3, lr = 0.0005, 
                                num_epochs = 2000, seed = rrr))
  err_dfr = sapply(1:length(LMeanOut), function(j) sum((res_drf$yPred[[j]] - LMeanOut[[j]])^2))
  
  # Prediction errors
  err_q = data.frame(n = n, r = rrr, DFR = mean(err_dfr))
  
  return(err_q)
}
# Stop the parallel cluster
stopCluster(cl)

# Combine results and summarize
err = do.call(rbind, result)

# Summarize and display mean squared prediction error (MSPE)
err  %>%
  select(n, DFR) %>%
  gather(key = "method", value = Error, -c("n")) %>%
  group_by(n, method) %>%
  dplyr::summarise(MSPE = mean(Error)) %>%
  mutate(method = factor(method, levels = c("DFR"))) %>%
  dplyr::select(n, method, MSPE) %>%
  spread(key = method, value = MSPE) %>%
  knitr::kable(digits = 3)