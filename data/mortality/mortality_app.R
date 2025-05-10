# Set working directory to the current script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load necessary libraries
library(fdadensity)
library(dplyr)
library(tidyverse)
library(foreach)

# Load DFR functions and Python scripts
function_path = "../../code"
function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, ignore.case=TRUE)
sapply(function_sources, source, .GlobalEnv)

# Load Python scripts using reticulate
reticulate::source_python('../../code/DNN.py')

# Load the mortality data
mortality = readRDS("mortality.RData")

# Extract predictor matrix and density data
x_pred = mortality$pred

# Convert density data to quantile functions
quan = foreach(i = (1:nrow(x_pred)), .combine = "rbind") %do% {
  x = mortality$density[[i]]$x
  y = mortality$density[[i]]$y
  y.quantile = dens2quantile(dens = y, dSup = x)
}

# Set up parallel processing
library(doRNG)
library(doParallel)
cl = makeCluster(60)  # Create a cluster with 3 cores
length(cl)
registerDoParallel(cl)

# Perform leave-one-out cross-validation
result = foreach(i = 1:nrow(x_pred), .packages = c('tidyverse', 'dplyr', 'frechet', 'foreach', 'reshape2', 'vegan', 'reticulate', 'fdadensity')) %dorng% {
  
  n = nrow(x_pred)
  ind_test = i
  ind_remain = (1:n)[-i]
  
  # Split data into training and test sets
  quan_train = quan[ind_remain, ]
  quan_test = matrix(quan[ind_test, ], nrow = 1)
  x_pred_train = x_pred[ind_remain, ]
  x_pred_test = matrix(x_pred[ind_test, ], nrow = 1)
  
  # Reload necessary functions for each iteration
  function_path = "../../code"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  reticulate::source_python('../../code/DNN.py')
  
  # Fit deep Frechet regression (DFR) model
  y = lapply(1:nrow(quan_train), function(j) quan_train[j, ])
  res_dfr = DFR(y = y, x = x_pred_train, xout = x_pred_test,
                optns = list(type = "measure", manifold = list(method = "isomap", k = 30), r = 2,
                             layer = 4, hidden = 32, dropout = 0.3, lr = 0.0005,
                             num_epochs = 2000, seed = i))
  err_dfr = sapply(1:nrow(quan_test), function(j) mean((res_dfr$yPred[j, ] - quan_test[j, ])^2))
  
  # Compile errors for this iteration
  err_q = data.frame( i = i, test_DFR = err_dfr)
  
  return(err_q)
}

# Stop the parallel cluster
stopCluster(cl)

# Combine results from all iterations
err = do.call(rbind, result)

# Summarize mean squared prediction errors (MSPE)
err %>%
  dplyr::select(DFR = test_DFR) %>%
  gather(key = "method", value = Error) %>%
  group_by(method) %>%
  dplyr::summarise(MSPE = mean(Error)) %>%
  mutate(method = factor(method, levels = c("DFR"))) %>%
  dplyr::select(method, MSPE) %>%
  spread(key = method, value = MSPE)
