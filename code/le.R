# modified from do.llle in R package Rdimtools
laplacian_eigenmaps <- function(D, d, k = round(nrow(D)/2)) {
  n <- nrow(D)
  
  # Step 1: Construct adjacency graph (k-nearest neighbors)
  neighbors <- t(apply(D, 1, function(x) order(x)[2:(k+1)]))
  
  # Step 2: Compute weight matrix
  W <- matrix(0, n, n)
  for (i in 1:n) {
    W[i, neighbors[i,]] <- 1 # exp(-D[i, neighbors[i,]]^2 / 2/ t )
  }
  W <- (W + t(W)) / 2  # Make symmetric
  
  # Step 3: Compute graph Laplacian
  D <- diag(rowSums(W))
  L <- D - W
  
  epsilon <- 1e-8  # Small regularization constant
  D_reg <- D + diag(epsilon, nrow(D))
  # Step 4: Solve generalized eigenproblem
  eig <- geigen::geigen(L, D_reg)

  # eig <- geigen::geigen(L, D)
  Y <- eig$vectors[, order(eig$values)[2:(d+1)]]
  
  return(Y)
}
