#' @title Deep Frechet Regression (DFR)
#' @description Implements Deep Frechet Regression (DFR) for metric-space-valued responses with Euclidean predictors using various manifold learning techniques.
#' @param x An n by p matrix or data frame of predictors. Can also be a vector of length n if p = 1.
#' @param y A list of n observations, where each element represents the metric space-valued response.
#' @param xout Optional. An nOut by p matrix or data frame of output predictor levels for predictions. Default is \code{NULL}.
#' @param optns A list of options specified as \code{list(name = value)}. See `Details` for available control options.
#' @details The control options are:
#' \describe{
#'   \item{type}{The type of data ('measure' for probability measures or 'network' for network data). Required.}
#'   \item{manifold}{A list specifying the manifold learning method ('isomap', 'tsne', 'umap', 'le', 'diffuse') and its parameters. Default is ISOMAP with k = 10.}
#'   \item{r}{The dimension of the low-dimensional representation. Default is 2. Must be \code{<= 2} for local Frechet regression.}
#'   \item{layer}{Number of hidden layers for the neural network. Default is 4.}
#'   \item{hidden}{Number of neurons per layer. Default is 32.}
#'   \item{num_epochs}{Number of training epochs. Default is 2000.}
#'   \item{lr}{Learning rate for the neural network training. Default is 0.0005.}
#'   \item{dropout}{Dropout rate for regularization. Default is 0.3.}
#'   \item{bw}{Bandwidth for local Frechet regression. If not specified, it is set to 10% of the range of the intermediate estimates.}
#'   \item{seed}{Random seed for reproducibility. If not specified, a random seed is generated.}
#' }
#' @return A list containing:
#' \item{yFit}{Fitted values for the training data.}
#' \item{yPred}{Predicted values for the test data if \code{xout} is provided.}
#' \item{type}{Type of data ('measure' or 'network').}
#' \item{manifold}{Details of the manifold learning method used.}
#' \item{r}{Dimension of the low-dimensional representation.}
#' \item{hidden}{Number of neurons per layer.}
#' \item{num_epochs}{Number of training epochs.}
#' \item{lr}{Learning rate used.}
#' \item{dropout}{Dropout rate used.}
#' \item{bw}{Bandwidth used for local Frechet regression.}
#' @export
#' 
DFR = function(x, y, xout = NULL, optns = list()){
  
  # Check if inputs x and y are provided
  if (is.null(y) | is.null(x)) {
    stop("requires the input of both y and x")
  }
  
  # Check if the data type is specified
  if (is.null(optns$type)) {
    stop("requires the input of data type")
  }
  
  # Ensure x is a matrix
  if (!is.matrix(x)) {
    if (is.data.frame(x) | is.vector(x)) {
      x <- as.matrix(x)
    } else {
      stop("x must be a matrix or a data frame or a vector")
    }
  }
  
  n = nrow(x)  # Number of observations
  d = ncol(x)  # Number of predictor dimensions
  
  # Set random seed for reproducibility
  if (is.null(optns$seed)) {
    seed = rbinom(1, 1e3, 0.5)
  } else {
    seed = optns$seed
  }
  
  # Specify manifold learning method, default to ISOMAP
  if(is.null(optns$manifold)){
    warning("Manifold learning technique is not specified, using ISOMAP.")
    manifold = list()
    manifold$method = "isomap"
    manifold$k = 10
  }else{
    manifold = optns$manifold
    if(manifold$method == "isomap"){
      if(is.null(manifold$k)){
        manifold$k = 10
      }
    }
  }
  
  # Set default dimension of the low-dimensional representation
  if(is.null(optns$r)){
    r = 2
  }else{
    r = optns$r
    if(r > 2){
      stop("r must be less than or equal to 2 for local frechet regression")
    }
  }
  
  # Set neural network parameters with default values
  if(is.null(optns$layer)){
    warning("Hidden layer is not specified, using 4 hidden layers.")
    layer = 4
  }else{
    layer = optns$layer
  }
  if(is.null(optns$hidden)){
    warning("Number of neurons of each layer is not specified, using 32 neurons.")
    hidden = 32
  }else{
    hidden = optns$hidden
  }
  if(is.null(optns$num_epochs)){
    warning("Number of epochs is not specified, using 2000 epochs.")
    num_epochs = 2000
  }else{
    num_epochs = optns$num_epochs
  }
  if(is.null(optns$lr)){
    warning("Learning rate is not specified, using 0.0005.")
    lr = 0.0005
  }else{
    lr = optns$lr
  }
  if(is.null(optns$dropout)){
    warning("Dropout rate is not specified, using 0.3.")
    dropout = 0.3
  }else{
    dropout = optns$dropout
  }
  
  # Compute the pairwise distance matrix based on the data type
  if(manifold$method %in% c("umap")){
    if(optns$type == "measure"){
      y_mat = do.call(rbind, y)
      N = ncol(y_mat)
      dist.den = dist(y_mat)/sqrt(N)
    }else if(optns$type == "network"){
      y_mat = do.call(rbind, lapply(y, function(i) as.numeric(i)))
      dist.den = dist(y_mat)
    }
  }else{
    if(optns$type == "measure"){
      y_mat = do.call(rbind, y)
      N = ncol(y_mat)
      dist.den = as.matrix(dist(y_mat, upper = T, diag = T))/sqrt(N)
    }else if(optns$type == "network"){
      y_mat = do.call(rbind, lapply(y, function(i) as.numeric(i)))
      dist.den = as.matrix(dist(y_mat, upper = T, diag = T))
    }
  }
  
  # Perform manifold learning based on the specified method
  if(manifold$method == "isomap"){
    x_manifold = as.data.frame(vegan::isomap(dist.den, k = manifold$k, ndim = r)$points)
  }else if(manifold$method == "tsne"){
    x_manifold = as.data.frame(Rtsne::Rtsne(X = dist.den, dims = r, is_distance = TRUE)$Y)
  }else if(manifold$method == "umap"){
    x_manifold = uwot::umap(X = dist.den, n_components = r)
  }else if(manifold$method == "le"){
    x_manifold = as.data.frame(laplacian_eigenmaps(D = dist.den, d = r))
  }else if(manifold$method == "diffuse"){
    x_manifold = diffusionmap(D = dist.den)$X[,1:r]
    while(any(x_manifold %in% c(Inf,-Inf))){
      x_manifold = diffusionmap(D = dist.den)$X[,1:r]
    }
  }
  
  # Scale the manifold representation
  Z = matrix(0, nrow = n, ncol = r)
  
  for(i in 1:r){
    Z[,i] = scale(x_manifold[,i])[,1]
  }
  
  # Train neural network models for each dimension of the manifold representation
  x_manifold_out = list()
  if(is.null(xout)){
    for (i in 1:r) {
      x_manifold_out[[i]] = DNN_regression(x = x, y = matrix(Z[,i], ncol = 1), 
                                           xout = x, layer = layer,
                                           hidden = hidden, dropout = dropout, 
                                           lr = lr, num_epochs = num_epochs, seed = seed)
    }
  }else{
    for (i in 1:r) {
      x_manifold_out[[i]] = DNN_regression(x = x, y = matrix(Z[,i], ncol = 1), 
                                           xout = rbind(x,xout), 
                                           layer = layer,
                                           hidden = hidden, dropout = dropout, 
                                           lr = lr, num_epochs = num_epochs, seed = seed)
    }
  }
  
  # Combine predictions for all dimensions
  x_manifold_out = do.call(cbind, x_manifold_out)
  
  # Perform local Frechet regression using the manifold representation
  if(is.null(optns$bw)){
    rg = apply(matrix(x_manifold_out[1:n,],nrow = n, r), 2, function(xxx) range(xxx)[2]-range(xxx)[1])
    bw = rg*0.1
    if(any(bw == 0)){
      bw[bw == 0] = 0.1
    }
  }else{
    bw = optns$bw
  }
  if(optns$type == "measure"){
    if(is.null(xout)){
      res = lrem(y = y, x = x_manifold_out, 
                optns = list(bwReg = bw))
    }else{
      res = lrem(y = y, x = x_manifold_out[1:n,], 
                xOut = x_manifold_out[(n+1):nrow(x_manifold_out),], 
                optns = list(bwReg = bw))
    }
    
  }else if(optns$type == "network"){
    if(is.null(xout)){
      res = lnr(gl = y, x = x_manifold_out, 
                optns = list(bwReg = bw))
    }else{
      res = lnr(gl = y, x = x_manifold_out[1:n,], 
                xOut = x_manifold_out[(n+1):nrow(x_manifold_out),], 
                optns = list(bwReg = bw))
    }
  }
  
  # Return the results
  if(is.null(xout)){
    if(optns$type == "measure"){
      return(list(yFit = res$qf, yFitSup = res$qfSupp,
                  type = "measure", manifold = manifold, r = r, hidden = hidden,
                  num_epochs = num_epochs, lr = lr, dropout = dropout, bw = bw))
    }else if(optns$type == "network"){
      return(list(yFit = res$fit,
                  type = "network", manifold = manifold, r = r, hidden = hidden,
                  num_epochs = num_epochs, lr = lr, dropout = dropout, bw = bw))
    }
  }else{
    if(optns$type == "measure"){
      return(list(yFit = res$qf, yFitSup = res$qfSupp, 
                  yPred = res$qp, yPredSup = res$qpSupp,
                  type = "measure", manifold = manifold, r = r, hidden = hidden,
                  num_epochs = num_epochs, lr = lr, dropout = dropout, bw = bw))
    }else if(optns$type == "network"){
      return(list(yFit = res$fit, yPred = res$predict,
                  type = "network", manifold = manifold, r = r, hidden = hidden,
                  num_epochs = num_epochs, lr = lr, dropout = dropout, bw = bw))
    }
  }
}
