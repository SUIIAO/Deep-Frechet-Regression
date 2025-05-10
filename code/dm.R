# Modified from "diffuse" function in "diffusionMap" package
diffusionmap = function (D, eps.val = diffusionMap::epsilonCompute(D)*10^3, 
                         neigen = NULL, t = 0, maxdim = 50, delta = 10^-5) {
  start = proc.time()[3]
  D = as.matrix(D)
  n = dim(D)[1]
  K = exp(-D^2/(eps.val))
  v = sqrt(apply(K, 1, sum))
  A = K/(v %*% t(v))
  ind = which(A > delta, arr.ind = TRUE)
  Asp = Matrix::sparseMatrix(i = ind[, 1], j = ind[, 2], x = A[ind], 
                     dims = c(n, n))
  f = function(x, A = NULL) {
    as.matrix(A %*% x)
  }

  if (is.null(neigen)) {
    neff = min(maxdim + 1, n)
  } else {
    neff = min(neigen + 1, n)
  }
  decomp = igraph::arpack(f, extra = Asp, sym = TRUE, options = list(which = "LA", 
                                                             nev = neff, n = n, ncv = max(min(c(n, 4 * neff)))))
  psi = decomp$vectors/(decomp$vectors[, 1] %*% matrix(1, 1, 
                                                       neff))
  phi = decomp$vectors * (decomp$vectors[, 1] %*% matrix(1, 
                                                         1, neff))
  eigenvals = decomp$values

  if (t <= 0) {
    lambda = eigenvals[-1]/(1 - eigenvals[-1])
    lambda = rep(1, n) %*% t(lambda)
    if (is.null(neigen)) {
      lam = lambda[1, ]/lambda[1, 1]
      neigen = min(which(lam < 0.05))
      neigen = min(neigen, maxdim)
      eigenvals = eigenvals[1:(neigen + 1)]
      
    }
    X = psi[, 2:(neigen + 1)] * lambda[, 1:neigen]
  } else {
    lambda = eigenvals[-1]^t
    lambda = rep(1, n) %*% t(lambda)
    if (is.null(neigen)) {
      lam = lambda[1, ]/lambda[1, 1]
      neigen = min(which(lam < 0.05))
      neigen = min(neigen, maxdim)
      eigenvals = eigenvals[1:(neigen + 1)]
      
    }
    X = psi[, 2:(neigen + 1)] * lambda[, 1:neigen]
  }

  y = list(X = X, phi0 = phi[, 1], eigenvals = eigenvals[-1], 
           eigenmult = lambda[1, 1:neigen], psi = psi, phi = phi, 
           neigen = neigen, epsilon = eps.val)
  class(y) = "diffuse"
  return(y)
}
