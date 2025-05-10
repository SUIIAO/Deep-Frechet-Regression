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

# Load Taxi Data
taxi = readRDS("taxi_data.RData")
x_pred = taxi$pred
y_nw = taxi$taxigl

meanLapM_y = sum((Reduce("+", y_nw) / length(y_nw))^2)

# Set up parallel processing
library(doRNG)
library(doParallel)
cl = makeCluster(60) 
length(cl)
registerDoParallel(cl)

# Define parameter grid for simulations
param_grid = expand.grid(q = 1:100, i = 1:10)

# Perform simulations using foreach loop
result = foreach(params = t(param_grid), .packages = c('tidyverse', 'dplyr', 'frechet', 'foreach', 'reshape2', 'vegan', 'reticulate')) %dorng% {
  
  q = params[1] # Replication index
  i = params[2] # i-th fold
  n = nrow(x_pred) # Sample size
  
  set.seed(q) # Set seed for reproducibility
  
  # Split data into training and test sets
  cv_fold = data.frame(rd_ind = sample(1:n,n,replace = FALSE),
                       fold = rep(1:10,length.out=n))
  
  ind_test = cv_fold[cv_fold$fold==i,1]
  ind_remain = cv_fold[cv_fold$fold!=i,1]
  
  y_nw_train = y_nw[ind_remain]
  y_nw_test = y_nw[ind_test]
  x_pred_train = x_pred[ind_remain,]
  x_pred_test = x_pred[ind_test,]
  
  x_pred_q = x_pred
  x_pred_train_mean = as.vector(colMeans(x_pred_train))
  x_pred_train_sd = as.vector(apply(x_pred_train, 2, sd))
  
  x_pred_train_mean[which(names(x_pred_train) %in% c("MTWT", "FS"))] = 0
  x_pred_train_sd[which(names(x_pred_train) %in% c("MTWT", "FS"))] = 1
  
  x_pred_train = t((t(x_pred_train) - x_pred_train_mean)/x_pred_train_sd) ## standardize individually
  x_pred_test = t((t(x_pred_test) - x_pred_train_mean)/x_pred_train_sd)  ## standardize individually
  
  # Reload necessary functions for each iteration
  function_path = "../../code"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  reticulate::source_python('../../code/DNN.py')
  
  # DFR
  res_dfr = DFR(y = y_nw_train, x = x_pred_train, xout = x_pred_test, 
                   optns = list(type = "network", manifold = list(method = "isomap", k = 20),
                                r = 2, layer = 4, hidden = 64, dropout = 0.3, 
                                lr = 0.0005, num_epochs = 2000, seed = q))
  err_dfr = sapply(1:length(y_nw_test), function(j) sum((res_dfr$yPred[[j]] - y_nw_test[[j]])^2))/meanLapM_y
  
  # Compile errors for this iteration
  err_q = data.frame(n = n, q = q, k = i, test_DFR = mean(err_dfr))
  
  return(err_q)
}
stopCluster(cl)


# Stop the parallel cluster
stopCluster(cl)

# Combine results and summarize
err = do.call(rbind, result)

# Section 6: DFR achieves better prediction performance, resulting in a 45\% and 55\% improvement in prediction accuracy compared to GFR and SDR
err %>%
  select(q, k, DFR = test_DFR) %>%
  gather(key = "method", value = Error, -c("q", "k")) %>%
  group_by(q, method) %>%
  dplyr::summarise(MSPE = mean(Error)) %>%
  group_by(method) %>%
  dplyr::summarise(MSPE = mean(MSPE)) %>%
  mutate(method = factor(method, levels = c("DFR"))) %>%
  dplyr::select(method, MSPE) %>%
  spread(key = method, value = MSPE) %>%
  select(DFR, GFR, SDR) %>%
  knitr::kable(digits = 5)
