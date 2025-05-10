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
  
  # Define simulation parameters
  N = 100  # Number of observations per sample
  nOut = 100  # Number of test samples
  sigma0 = 3  # Base standard deviation
  kappa = 1  # Shape parameter for gamma distribution
  nu1 = 1 # mean parameter for normal distribution
  
  set.seed(rrr)  # Set seed for reproducibility
  
  x = data.frame(X1 = runif(n, -1, 0), X2 = runif(n, 0, 1), X3 = runif(n, 1, 2),
                 X4 = rnorm(n, 0, 1), X5 = rnorm(n, -10, 3), X6 = rnorm(n, 10, 3),
                 X7 = sample(c(0,1),n,TRUE,c(0.4,0.6)), 
                 X8 = sample(c(0,1),n,TRUE,c(0.3,0.7)), 
                 X9 = sample(c(0,1),n,TRUE,c(0.6,0.3))
  )
  xout = data.frame(X1 = runif(nOut, -1, 0), X2 = runif(nOut, 0, 1), X3 = runif(nOut, 1, 2),
                    X4 = rnorm(nOut, 0, 1), X5 = rnorm(nOut, -10, 3), X6 = rnorm(nOut, 10, 3),
                    X7 = sample(c(0,1),nOut,TRUE,c(0.4,0.6)), 
                    X8 = sample(c(0,1),nOut,TRUE,c(0.3,0.7)), 
                    X9 = sample(c(0,1),nOut,TRUE,c(0.6,0.3))
  )
  
  # Generate response data
  y = list()
  yMean = matrix(nrow = n, ncol = N-1)
  mu.sd = NULL
  for(i in 1:n) {
    # Generate expected mean and standard deviation
    expect_eta_Z = 3 * (sin(pi * x[i,1])+cos(pi * x[i,2])) * x[i,8] +
      (5*x[i,4]^2 + x[i,5]) * x[i,7]
    
    expect_sigma_Z = sigma0 + 0.5*(sin(pi * x[i,1])+cos(pi * x[i,2])) * x[i,8] +
      abs(5*x[i,4]^2 + x[i,5]) * x[i,7]
    
    # Sample mean and standard deviation
    mu = rnorm(1, mean = expect_eta_Z, sd = nu1)
    sigma = rgamma(1, shape = expect_sigma_Z^2/kappa, scale = kappa/expect_sigma_Z)
    
    # Generate response and true quantile values
    y[[i]] = sort(mu + sigma * rnorm(N)) # rnorm(N, mean = mu, sd = sigma)
    yMean[i, ] = expect_eta_Z + expect_sigma_Z * qnorm(c(1:(N-1))/N)
    
    mu.sd = rbind(mu.sd, data.frame(mu, sigma))
  }
  
  # Generate true quantile values for test data
  yMeanOut = t(apply(xout, 1, function(xOuti) 3 * (sin(pi * xOuti[1])+cos(pi * xOuti[2])) * xOuti[8] +
                       (5*xOuti[4]^2 + xOuti[5]) * xOuti[7] +
                       (sigma0 + 0.5*(sin(pi * xOuti[1])+cos(pi * xOuti[2]))*xOuti[8] +
                          abs(5*xOuti[4]^2 + xOuti[5]) * xOuti[7]) * qnorm(1:(N - 1) / N)))
  
  # DFR
  res_isomap = DFR(y = y, x = x, xout = xout, 
                   optns = list(type = "measure", manifold = list(method = "isomap", k = ifelse(n == 100, 20, 0.1*n)), r = 2, 
                                layer = 4, hidden = ifelse(n == 100, 32, 64), dropout = 0.3, lr = 0.0005, 
                                num_epochs = 2000, seed = rrr))
  
  res_tsne = DFR(y = y, x = x, xout = xout, 
                 optns = list(type = "measure", manifold = list(method = "tsne"), r = 2, 
                              layer = 4, hidden = ifelse(n == 100, 32, 64), dropout = 0.3, lr = 0.0005, 
                              num_epochs = 2000, seed = rrr))
  
  res_umap = DFR(y = y, x = x, xout = xout, 
                 optns = list(type = "measure", manifold = list(method = "umap"), r = 2, 
                              layer = 4, hidden = ifelse(n == 100, 32, 64), dropout = 0.3, lr = 0.0005, 
                              num_epochs = 2000, seed = rrr))
  
  res_le = DFR(y = y, x = x, xout = xout, 
               optns = list(type = "measure", manifold = list(method = "le"), r = 2, 
                            layer = 4, hidden = ifelse(n == 100, 32, 64), dropout = 0.3, lr = 0.0005, 
                            num_epochs = 2000, seed = rrr))
  
  res_diffuse = DFR(y = y, x = x, xout = xout, 
                    optns = list(type = "measure", manifold = list(method = "diffuse"), r = 2, 
                                 layer = 4, hidden = ifelse(n == 100, 32, 64), dropout = 0.3, lr = 0.0005, 
                                 num_epochs = 2000, seed = rrr))
  
  err_q = data.frame(
    n = n, r = rrr,
    test_DFR_isomap = sum((res_isomap$yPred[, -N] - yMeanOut)^2) / ((N - 1) * nOut),
    test_DFR_tsne = sum((res_tsne$yPred[, -N] - yMeanOut)^2) / ((N - 1) * nOut),
    test_DFR_umap = sum((res_umap$yPred[, -N] - yMeanOut)^2) / ((N - 1) * nOut),
    test_DFR_le = sum((res_le$yPred[, -N] - yMeanOut)^2) / ((N - 1) * nOut),
    test_DFR_diffuse = sum((res_diffuse$yPred[, -N] - yMeanOut)^2) / ((N - 1) * nOut)
  )
  return(err_q)
}

# Stop the parallel cluster
stopCluster(cl)

# Combine results and summarize
err = do.call(rbind, result)

# Summarize and display mean squared prediction error
err %>%
  select(n, 
         ISOMAP = test_DFR_isomap, 
         tSNE = test_DFR_tsne,
         UMAP = test_DFR_umap,
         LE = test_DFR_le,
         Diffusion = test_DFR_diffuse) %>%
  gather(key = "method", value = Error, -c("n")) %>%
  group_by(n, method) %>%
  dplyr::summarise(MSPE = mean(Error)) %>%
  mutate(method = factor(method, levels = c("ISOMAP", "tSNE", 
                                            "UMAP", "LE", "Diffusion"))) %>%
  dplyr::select(n, method, MSPE) %>%
  spread(key = method, value = MSPE) %>%
  knitr::kable(digits = 3)

# Summarize and display median squared prediction error
err %>%
  select(n, 
         ISOMAP = test_DFR_isomap, 
         tSNE = test_DFR_tsne,
         UMAP = test_DFR_umap,
         LE = test_DFR_le,
         Diffusion = test_DFR_diffuse) %>%
  gather(key = "method", value = Error, -c("n")) %>%
  group_by(n, method) %>%
  dplyr::summarise(MSPE = median(Error)) %>%
  mutate(method = factor(method, levels = c("ISOMAP", "tSNE", 
                                            "UMAP", "LE", "Diffusion"))) %>%
  dplyr::select(n, method, MSPE) %>%
  spread(key = method, value = MSPE) %>%
  knitr::kable(digits = 3)
