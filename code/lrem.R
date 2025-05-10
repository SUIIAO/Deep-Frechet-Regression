#' @title Local Regression with Empirical Measures (REM)
#' @description Local regression for empirical measures with Euclidean predictors.
#' @param y A list of \eqn{n} empirical measures, represented as vectors of
#' observed values. The \eqn{i}th vector contains \eqn{N_i} observations for
#' the \eqn{i}th individual.
#' @param x An n by p matrix or data frame of predictors. It can be a vector of 
#' length n if p = 1.
#' @param xOut An nOut by p matrix or data frame of output predictor levels.
#' It can be a vector of length p if nOut = 1 or a vector of length nOut if p = 1
#' or a scalar if both p and nOut are equal to 1. Default is \code{NULL}.
#' @param optns A list of options control parameters specified by \code{list(name = value)}. 
#' See `Details'.
#' @details Available control options are
#' \describe{
#' \item{bw}{A vector of length p used as the bandwidth for local REM. If not 
#' specified, the bandwidth will be selected using cross-validation.}
#' \item{bwRange}{A 2 by p matrix whose columns contain the bandwidth selection 
#' range for each corresponding dimension of the predictor \code{x} for the case 
#' when \code{bw} is \code{NULL}. Default is \code{NULL} and is automatically 
#' chosen by a data-adaptive method.}
#' \item{kernel}{A character holding the type of kernel functions for local REM; 
#' \code{"rect"}, \code{"gauss"}, \code{"epan"}, \code{"gausvar"}, \code{"quar"} 
#' - default: \code{"gauss"}.}
#' \item{lower}{The lower bound of the support of the measure. Default is \code{NULL}.}
#' \item{upper}{The upper bound of the support of the measure. Default is \code{NULL}.}
#' }
#' @return A \code{rem} object, which is a list containing the following components:
#' \item{qf}{A matrix holding the quantile functions corresponding to \code{x}. 
#' Each row gives a quantile and the domain grid is given in \code{qfSupp}.}
#' \item{qfSupp}{A vector giving the domain grid of \code{qf}.}
#' \item{qp}{A matrix holding the quantile functions corresponding to 
#' \code{xOut}. Each row gives a quantile and the domain grid is given in 
#' \code{qpSupp}. Included if \code{xOut} is not \code{NULL}.}
#' \item{qpSupp}{A vector giving the domain grid of \code{qp}.
#' Included if \code{xOut} is not \code{NULL}.}
#' \item{residuals}{Wasserstein distances between the empirical and fitted measures.}
#' \item{y}{The empirical measures used.}
#' \item{x}{The predictors used.}
#' \item{xOut}{The output predictor levels used.}
#' \item{optns}{The control options used.}
#' @references
#' \itemize{
#' \item \cite{Zhou, Y. and Müller, H.G., 2023. Wasserstein Regression with Empirical Measures and Density Estimation for Sparse Data. arXiv preprint arXiv:2308.12540.}
#' \item \cite{Petersen, A. and Müller, H.-G. (2019). Fréchet regression for random objects with Euclidean predictors. The Annals of Statistics, 47(2), 691--719.}
#' }
#' @export

lrem <- function(y = NULL,
                x = NULL,
                xOut = NULL,
                optns = list()) {
  if (is.null(y) | is.null(x)) {
    stop("requires the input of both y and x")
  }
  if (!is.matrix(x)) {
    if (is.data.frame(x) | is.vector(x)) {
      x <- as.matrix(x)
    } else {
      stop("x must be a matrix or a data frame or a vector")
    }
  }
  n <- nrow(x) # number of observations
  p <- ncol(x) # number of covariates
  if (!is.list(y)) {
    stop("y must be a list")
  }
  if (length(y) != n) {
    stop("the number of rows in x must be the same as the number of empirical measures in y")
  }
  if (!is.null(xOut)) {
    if (!is.matrix(xOut)) {
      if (is.data.frame(xOut)) {
        xOut <- as.matrix(xOut)
      } else if (is.vector(xOut)) {
        if (p == 1) {
          xOut <- as.matrix(xOut)
        } else {
          xOut <- t(xOut)
        }
      } else {
        stop("xOut must be a matrix or a data frame or a vector")
      }
    }
    if (ncol(xOut) != p) {
      stop("x and xOut must have the same number of columns")
    }
    nOut <- nrow(xOut) # number of predictions
  } else {
    nOut <- 0
  }
  
  if (!is.null(optns$bw)) {
    if (sum(optns$bw <= 0) > 0) {
      stop("bandwidth must be positive")
    }
    if (length(optns$bw) != p) {
      stop("dimension of bandwidth does not agree with x")
    }
  }
  if (!is.null(optns$bwRange)) {
    if (!is.matrix(optns$bwRange) & !is.vector(optns$bwRange)) {
      stop("bwRange must be a matrix or vector")
    }
    if (is.vector(optns$bwRange)) {
      optns$bwRange <- matrix(optns$bwRange, length(optns$bwRange))
      if (ncol(x) > 1) {
        stop("bwRange must be a matrix")
      } else {
        if (nrow(optns$bwRange) != 2) {
          stop("bwRange must have the lower and upper bound for the bandwidth range")
        }
      }
    } else {
      if (ncol(optns$bwRange) != ncol(x)) {
        stop("bwRange must have the same number of columns as x")
      }
      if (nrow(optns$bwRange) != 2) {
        stop("bwRange must have two rows")
      }
    }
  }
  
  if (is.null(optns$kernel)) {
    optns$kernel <- "gauss"
  }

  N <- sapply(y, length)
  y <- lapply(1:n, function(i) {
    sort(y[[i]])
  }) # sort observed values
  
  M <- min(plcm(N), n * max(N), 5000) # least common multiple of N_i
  yM <- t(sapply(1:n, function(i) {
    residual <- M %% N[i]
    if(residual) {
      sort(c(rep(y[[i]], each = M %/% N[i]), sample(y[[i]], residual)))
    } else {
      rep(y[[i]], each = M %/% N[i])
    }
  })) # n by M

  # initialization of OSQP solver
  A <- cbind(diag(M), rep(0, M)) + cbind(rep(0, M), -diag(M))
  if (!is.null(optns$upper) &
    !is.null(optns$lower)) {
    # if lower & upper are neither NULL
    l <- c(optns$lower, rep(0, M - 1), -optns$upper)
  } else if (!is.null(optns$upper)) {
    # if lower is NULL
    A <- A[, -1]
    l <- c(rep(0, M - 1), -optns$upper)
  } else if (!is.null(optns$lower)) {
    # if upper is NULL
    A <- A[, -ncol(A)]
    l <- c(optns$lower, rep(0, M - 1))
  } else {
    # if both lower and upper are NULL
    A <- A[, -c(1, ncol(A))]
    l <- rep(0, M - 1)
  }
  # P <- as(diag(M), "sparseMatrix")
  # A <- as(t(A), "sparseMatrix")
  P <- diag(M)
  A <- t(A)
  q <- rep(0, M)
  u <- rep(Inf, length(l))
  model <-
    osqp::osqp(
      P = P,
      q = q,
      A = A,
      l = l,
      u = u,
      osqp::osqpSettings(max_iter = 1e05, eps_abs = 1e-05, eps_rel = 1e-05, verbose = FALSE)
    )

  # select kernel
  Kern <- kerFctn(optns$kernel)
  K <- function(x, h) {
    k <- 1
    for (i in 1:p) {
      k <- k * Kern(x[i] / h[i])
    }
    return(as.numeric(k))
  }

  if (is.null(optns$bw)) {
    optns$bw <- bwCV(
      xin = x,
      qin = yM,
      xout = xOut,
      optns = optns
    )
  } else {
    if (ncol(x) == 1) {
      if (optns$bw[1] < max(diff(sort(x[, 1]))) &
          !is.null(optns$kernel)) {
        if (optns$kernel %in% c("rect", "quar", "epan")) {
          warning("optns$bw was set too small and is reset to be chosen by CV.")
          optns$bw <-
            bwCV(
              xin = x,
              qin = yM,
              xout = xOut,
              optns = optns
            )
        }
      }
    } else {
      if (optns$bw[1] < max(diff(sort(x[, 1]))) &
        optns$bw[2] < max(diff(sort(x[, 2]))) & !is.null(optns$kernel)) {
        if (optns$kernel %in% c("rect", "quar", "epan")) {
          warning("optns$bw was set too small and is reset to be chosen by CV.")
          optns$bw <-
            bwCV(
              xin = x,
              qin = yM,
              xout = xOut,
              optns = optns
            )
        }
      }
    }
  }

  qf <- matrix(nrow = n, ncol = M)
  residuals <- rep.int(0, n)
  for (i in 1:n) {
    a <- x[i, ]
    if (p > 1) {
      mu1 <-
        rowMeans(apply(x, 1, function(xi) {
          K(xi - a, optns$bw) * (xi - a)
        }))
      mu2 <-
        matrix(rowMeans(apply(x, 1, function(xi) {
          K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a))
        })), ncol = p)
    } else {
      mu1 <-
        mean(apply(x, 1, function(xi) {
          K(xi - a, optns$bw) * (xi - a)
        }))
      mu2 <-
        mean(apply(x, 1, function(xi) {
          K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a))
        }))
    }
    wc <- t(mu1) %*% solve(mu2) # 1 by p
    w <- apply(x, 1, function(xi) {
      K(xi - a, optns$bw) * (1 - wc %*% (xi - a))
    }) # weight
    qNew <- apply(yM, 2, weighted.mean, w) # M
    if (any(w < 0)) {
      # if negative weights exist, project
      model$Update(q = -qNew)
      qNew <- sort(model$Solve()$x)
    }
    if (!is.null(optns$upper)) {
      qNew <- pmin(qNew, optns$upper)
    }
    if (!is.null(optns$lower)) {
      qNew <- pmax(qNew, optns$lower)
    }
    qf[i, ] <- qNew
    residuals[i] <- sqrt(sum((yM[i, ] - qf[i, ])^2) / M)
  }
  qfSupp <- 1:M / M

  if (nOut > 0) {
    qp <- matrix(nrow = nOut, ncol = M)
    for (i in 1:nOut) {
      a <- xOut[i, ]
      if (p > 1) {
        mu1 <-
          rowMeans(apply(x, 1, function(xi) {
            K(xi - a, optns$bw) * (xi - a)
          }))
        mu2 <-
          matrix(rowMeans(apply(x, 1, function(xi) {
            K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a))
          })), ncol = p)
      } else {
        mu1 <-
          mean(apply(x, 1, function(xi) {
            K(xi - a, optns$bw) * (xi - a)
          }))
        mu2 <-
          mean(apply(x, 1, function(xi) {
            K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a))
          }))
      }
      wc <- t(mu1) %*% solve(mu2) # 1 by p
      w <- apply(x, 1, function(xi) {
        K(xi - a, optns$bw) * (1 - wc %*% (xi - a))
      }) # weight
      qNew <- apply(yM, 2, weighted.mean, w) # M
      if (any(w < 0)) {
        # if negative weights exist
        model$Update(q = -qNew)
        qNew <- sort(model$Solve()$x)
      }
      if (!is.null(optns$upper)) {
        qNew <- pmin(qNew, optns$upper)
      }
      if (!is.null(optns$lower)) {
        qNew <- pmax(qNew, optns$lower)
      }
      qp[i, ] <- qNew
    }
    qpSupp <- 1:M / M
    
    res <-
      list(
        qf = qf,
        qfSupp = qfSupp,
        qp = qp,
        qpSupp = qpSupp,
        residuals = residuals,
        y = y,
        x = x,
        xOut = xOut,
        optns = optns
      )
  } else {
    res <- list(
      qf = qf,
      qfSupp = qfSupp,
      residuals = residuals,
      y = y,
      x = x,
      optns = optns
    )
  }

  class(res) <- "rem"
  res
}
