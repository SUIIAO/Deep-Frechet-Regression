#' @title Local Network Regression
#' @description Local regression for network data quantified by graph Laplacians with Euclidean predictors.
#' @param gl a list (length n) of m by m matrices (resp. an m by m by n array) where
#'   \code{M[[i]]} (resp. \code{M[, , i]}) holds the i-th graph Laplacian
#'   of dimension m by m.
#' @param x an n by p matrix or data frame of predictors.
#'   It can be a vector of length n if p = 1.
#' @param xOut an nOut by p matrix or data frame of output predictor levels.
#'   It can be a vector of length p if m = 1 or a vector of length nOut if p = 1
#'   or a scalar if both p and nOut are equal to 0. Default is \code{NULL}.
#' @param optns a list of options control parameters specified by
#'   \code{list(name = value)}. See `Details'.
#' @details Available control options are
#' \describe{
#' \item{metric}{choice of metric. 'frobenius' and 'power' are supported,
#'   which corresponds to Frobenius metric and Euclidean power metric,
#'   respectively. One can use 'f' and 'p' for short.
#'   Default is Frobenius metric.}
#' \item{alpha}{the power for Euclidean power metric.
#'   Default is 1 which corresponds to Frobenius metric.}
#' \item{kernel}{name of the kernel function to be chosen from.
#'   Available options are 'gauss', 'rect', 'epan', 'gausvar' and 'quar'.
#'   Default is 'gauss'.}
#' \item{bw}{bandwidth for local network regression, if not entered
#'   it would be chosen from cross validation.}
#' \item{digits}{the integer indicating the number of decimal places (round)
#'   to be kept in the output. Default is \code{NULL}, which means no round operation.}
#' }
#' @return A \code{nr} object --- a list containing the following fields:
#' \item{fit}{a list of estimated graph Laplacians at \code{x}.}
#' \item{predict}{a list of predicted graph Laplacians at \code{xOut}.
#'   Included if \code{xOut} is not \code{NULL}.}
#' \item{residuals}{Frobenius distance between the true and
#'   fitted graph Laplacians.}
#' \item{gl}{the graph Laplacians used.}
#' \item{x}{the predictor used.}
#' \item{xOut}{the output predictor level used.}
#' \item{optns}{the control options used.}
#' @examples
#' set.seed(1)
#' n <- 100
#' m <- 10
#' d <- m * (m - 1) / 2
#' xOut <- seq(0.1, 0.9, length.out = 9)
#' x <- runif(n, min = 0, max = 1)
#' gl <- list()
#' for (i in 1:n) {
#'   glVec <- -rbeta(d, shape1 = x[i], shape2 = 1 - x[i])
#'   temp <- matrix(0, nrow = m, ncol = m)
#'   temp[lower.tri(temp)] <- glVec
#'   temp <- temp + t(temp)
#'   diag(temp) <- -colSums(temp)
#'   gl[[i]] <- temp
#' }
#' res <- lnr(gl, x, xOut)
#' res$predict
#' @references
#' \itemize{
#' \item \cite{Zhou, Y. and Müller, H.G., 2022. Network Regression with Graph Laplacians. arXiv preprint arXiv:2109.02981.}
#' \item \cite{Petersen, A. and Müller, H.-G. (2019). Fréchet regression for random objects with Euclidean predictors. The Annals of Statistics, 47(2), 691--719.}
#' }
#' @export

lnr <- function(gl = NULL, x = NULL, xOut = NULL, optns = list()) {
  if (is.null(gl) | is.null(x)) {
    stop("requires the input of both gl and x")
  }
  if (is.null(optns$metric)) {
    optns$metric <- "frobenius"
  }
  if (!(optns$metric %in% c("frobenius", "power"))) {
    stop("metric choice not supported")
  }
  if (is.null(optns$alpha)) {
    optns$alpha <- 1
  }
  if (optns$alpha < 0) {
    stop("alpha must be non-negative")
  }
  if (is.null(optns$kernel)) {
    optns$kernel <- "gauss"
  }
  if (is.null(optns$bw)) {
    optns$bw <- NA
  }
  if (is.null(optns$digits)) {
    optns$digits <- NA
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
  if (p > 2) {
    stop("local method is designed to work in low dimensional case (p is either 1 or 2)")
  }
  if (!is.na(sum(optns$bw))) {
    if (sum(optns$bw <= 0) > 0) {
      stop("bandwidth must be positive")
    }
    if (length(optns$bw) != p) {
      stop("dimension of bandwidth does not agree with x")
    }
  }
  if (!is.list(gl)) {
    if (is.array(gl)) {
      gl <- lapply(seq(dim(gl)[3]), function(i) gl[, , i])
    } else {
      stop("gl must be a list or an array")
    }
  }
  if (length(gl) != n) {
    stop("the number of rows in x must be the same as the number of graph Laplacians in gl")
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
  nodes <- colnames(gl[[1]])
  m <- ncol(gl[[1]]) # number of nodes
  if (is.null(nodes)) nodes <- 1:m
  glVec <- matrix(unlist(gl), ncol = m^2, byrow = TRUE) # n by m^2
  if (substr(optns$metric, 1, 1) == "p") {
    glAlpha <- lapply(gl, function(gli) {
      eigenDecom <- eigen(gli)
      Lambda <- pmax(Re(eigenDecom$values), 0) # exclude 0i
      U <- eigenDecom$vectors
      U %*% diag(Lambda^optns$alpha) %*% t(U)
    })
    glAlphaVec <- matrix(unlist(glAlpha), ncol = m^2, byrow = TRUE) # n by m^2
  }

  # initialization of OSQP solver
  W <- 2^32 # bound on the weights of edges in the graph
  nConsts <- m^2 # number of constraints
  l <- c(rep.int(0, m * (m + 1) / 2), rep.int(-W, m * (m - 1) / 2))
  u <- rep.int(0, nConsts)
  q <- rep.int(0, m^2)
  P <- diag(m^2)
  consts <- matrix(0, nrow = nConsts, ncol = m^2)
  k <- 0
  for (i in 1:(m - 1)) {
    for (j in (i + 1):m) {
      k <- k + 1
      consts[k, (j - 1) * m + i] <- 1
      consts[k, (i - 1) * m + j] <- -1
    }
  }
  for (i in 1:m) {
    consts[k + i, ((i - 1) * m + 1):(i * m)] <- rep(1, m)
  }
  k <- k + m
  for (i in 1:(m - 1)) {
    for (j in (i + 1):m) {
      k <- k + 1
      consts[k, (j - 1) * m + i] <- 1
    }
  }
  model <- osqp::osqp(P, q, consts, l, u, osqp::osqpSettings(verbose = FALSE))

  # select kernel
  Kern <- kerFctn(optns$kernel)
  K <- function(x, h) {
    k <- 1
    for (i in 1:p) {
      k <- k * Kern(x[i] / h[i])
    }
    return(as.numeric(k))
  }

  # choose bandwidth by cross-validation
  if (is.na(sum(optns$bw))) {
    hs <- matrix(0, p, 20)
    for (l in 1:p) {
      hs[l, ] <- exp(seq(
        from = log(n^(-1 / (1 + p)) * (max(x[, l]) - min(x[, l])) / 10),
        to = log(5 * n^(-1 / (1 + p)) * (max(x[, l]) - min(x[, l]))),
        length.out = 20
      ))
    }
    cv <- array(0, 20^p)
    for (k in 0:(20^p - 1)) {
      h <- array(0, p)
      for (l in 1:p) {
        kl <- floor((k %% (20^l)) / (20^(l - 1))) + 1
        h[l] <- hs[l, kl]
      }
      for (j in 1:n) {
        a <- x[j, ]
        if (p > 1) {
          mu1 <- rowMeans(apply(as.matrix(x[-j, ]), 1, function(xi) K(xi - a, h) * (xi - a)))
          mu2 <- matrix(rowMeans(apply(as.matrix(x[-j, ]), 1, function(xi) K(xi - a, h) * ((xi - a) %*% t(xi - a)))), ncol = p)
        } else {
          mu1 <- mean(apply(as.matrix(x[-j, ]), 1, function(xi) K(xi - a, h) * (xi - a)))
          mu2 <- mean(apply(as.matrix(x[-j, ]), 1, function(xi) K(xi - a, h) * ((xi - a) %*% t(xi - a))))
        }
        skip <- FALSE
        tryCatch(solve(mu2), error = function(e) skip <<- TRUE)
        if (skip) {
          cv[k + 1] <- Inf
          break
        }
        mu2_inv = tryCatch(solve(mu2), 
                           error = function(e){
                             MASS::ginv(mu2)
                           } )
        wc <- t(mu1) %*% mu2_inv # 1 by p
        w <- apply(as.matrix(x[-j, ]), 1, function(xi) {
          K(xi - a, h) * (1 - wc %*% (xi - a))
        }) # weight
        if (substr(optns$metric, 1, 1) == "f") {
          qNew <- apply(glVec[-j, ], 2, weighted.mean, w) # m^2
          model$Update(q = -qNew)
          fitj <- matrix(model$Solve()$x, ncol = m, dimnames = list(nodes, nodes))
          fitj <- (fitj + t(fitj)) / 2 # symmetrize
          if (!is.na(optns$digits)) fitj <- round(fitj, digits = optns$digits) # round
          fitj[fitj > 0] <- 0 # off diagonal should be negative
          diag(fitj) <- 0
          diag(fitj) <- -colSums(fitj)
          cv[k + 1] <- cv[k + 1] + sum((gl[[j]] - fitj)^2) / n
        } else if (substr(optns$metric, 1, 1) == "p") {
          bAlpha <- matrix(apply(glAlphaVec[-j, ], 2, weighted.mean, w), ncol = m) # m by m
          eigenDecom <- eigen(bAlpha)
          Lambda <- pmax(Re(eigenDecom$values), 0) # projection to M_m
          U <- eigenDecom$vectors
          qNew <- as.vector(U %*% diag(Lambda^(1 / optns$alpha)) %*% t(U)) # inverse power
          model$Update(q = -qNew)
          fitj <- matrix(model$Solve()$x, ncol = m, dimnames = list(nodes, nodes))
          fitj <- (fitj + t(fitj)) / 2 # symmetrize
          if (!is.na(optns$digits)) fitj <- round(fitj, digits = optns$digits) # round
          fitj[fitj > 0] <- 0 # off diagonal should be negative
          diag(fitj) <- 0
          diag(fitj) <- -colSums(fitj)
          cv[k + 1] <- cv[k + 1] + sum((gl[[j]] - fitj)^2) / n
          # eigenDecom <- eigen(fitj)
          # Lambda <- pmax(Re(eigenDecom$values), 0)
          # U <- eigenDecom$vectors
          # fitjAlpha <- U%*%diag(Lambda^optns$alpha)%*%t(U)
          # cv[k+1] <- cv[k+1] + sum((glAlpha[[j]]-fitjAlpha)^2)/n# using Euclidean power metric
        }
      }
    }
    bwi <- which.min(cv)
    optns$bw <- array(0, p)
    for (l in 1:p) {
      kl <- floor((bwi %% (20^l)) / (20^(l - 1))) + 1
      optns$bw[l] <- hs[l, kl]
    }
  }

  fit <- vector(mode = "list", length = n)
  residuals <- rep.int(0, n)
  if (substr(optns$metric, 1, 1) == "f") {
    for (i in 1:n) {
      a <- x[i, ]
      if (p > 1) {
        mu1 <- rowMeans(apply(x, 1, function(xi) K(xi - a, optns$bw) * (xi - a)))
        mu2 <- matrix(rowMeans(apply(x, 1, function(xi) K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a)))), ncol = p)
      } else {
        mu1 <- mean(apply(x, 1, function(xi) K(xi - a, optns$bw) * (xi - a)))
        mu2 <- mean(apply(x, 1, function(xi) K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a))))
      }
      mu2_inv = tryCatch(solve(mu2), 
                         error = function(e){
                           MASS::ginv(mu2)
                         } )
      
      wc <- t(mu1) %*% mu2_inv # 1 by p
      w <- apply(x, 1, function(xi) {
        K(xi - a, optns$bw) * (1 - wc %*% (xi - a))
      }) # weight
      qNew <- apply(glVec, 2, weighted.mean, w) # m^2
      model$Update(q = -qNew)
      temp <- matrix(model$Solve()$x, ncol = m, dimnames = list(nodes, nodes))
      temp <- (temp + t(temp)) / 2 # symmetrize
      if (!is.na(optns$digits)) temp <- round(temp, digits = optns$digits) # round
      temp[temp > 0] <- 0 # off diagonal should be negative
      diag(temp) <- 0
      diag(temp) <- -colSums(temp)
      fit[[i]] <- temp
      residuals[i] <- sqrt(sum((gl[[i]] - temp)^2))
    }
    if (nOut > 0) {
      predict <- vector(mode = "list", length = nOut)
      for (i in 1:nOut) {
        a <- xOut[i, ]
        if (p > 1) {
          mu1 <- rowMeans(apply(x, 1, function(xi) K(xi - a, optns$bw) * (xi - a)))
          mu2 <- matrix(rowMeans(apply(x, 1, function(xi) K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a)))), ncol = p)
        } else {
          mu1 <- mean(apply(x, 1, function(xi) K(xi - a, optns$bw) * (xi - a)))
          mu2 <- mean(apply(x, 1, function(xi) K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a))))
        }
        mu2_inv = tryCatch(solve(mu2), 
                           error = function(e){
                             MASS::ginv(mu2)
                           } )
        wc <- t(mu1) %*% mu2_inv # 1 by p
        w <- apply(x, 1, function(xi) {
          K(xi - a, optns$bw) * (1 - wc %*% (xi - a))
        }) # weight
        qNew <- apply(glVec, 2, weighted.mean, w) # m^2
        model$Update(q = -qNew)
        temp <- matrix(model$Solve()$x, ncol = m, dimnames = list(nodes, nodes))
        temp <- (temp + t(temp)) / 2 # symmetrize
        if (!is.na(optns$digits)) temp <- round(temp, digits = optns$digits) # round
        temp[temp > 0] <- 0 # off diagonal should be negative
        diag(temp) <- 0
        diag(temp) <- -colSums(temp)
        predict[[i]] <- temp
      }
      res <- list(fit = fit, predict = predict, residuals = residuals, gl = gl, x = x, xOut = xOut, optns = optns)
    } else {
      res <- list(fit = fit, residuals = residuals, gl = gl, x = x, optns = optns)
    }
  } else if (substr(optns$metric, 1, 1) == "p") {
    for (i in 1:n) {
      a <- x[i, ]
      if (p > 1) {
        mu1 <- rowMeans(apply(x, 1, function(xi) K(xi - a, optns$bw) * (xi - a)))
        mu2 <- matrix(rowMeans(apply(x, 1, function(xi) K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a)))), ncol = p)
      } else {
        mu1 <- mean(apply(x, 1, function(xi) K(xi - a, optns$bw) * (xi - a)))
        mu2 <- mean(apply(x, 1, function(xi) K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a))))
      }
      mu2_inv = tryCatch(solve(mu2), 
                         error = function(e){
                           MASS::ginv(mu2)
                         } )
      wc <- t(mu1) %*% mu2_inv # 1 by p
      w <- apply(x, 1, function(xi) {
        K(xi - a, optns$bw) * (1 - wc %*% (xi - a))
      }) # weight
      bAlpha <- matrix(apply(glAlphaVec, 2, weighted.mean, w), ncol = m) # m by m
      eigenDecom <- eigen(bAlpha)
      Lambda <- pmax(Re(eigenDecom$values), 0) # projection to M_m
      U <- eigenDecom$vectors
      qNew <- as.vector(U %*% diag(Lambda^(1 / optns$alpha)) %*% t(U)) # inverse power
      model$Update(q = -qNew)
      temp <- matrix(model$Solve()$x, ncol = m, dimnames = list(nodes, nodes))
      temp <- (temp + t(temp)) / 2 # symmetrize
      if (!is.na(optns$digits)) temp <- round(temp, digits = optns$digits) # round
      temp[temp > 0] <- 0 # off diagonal should be negative
      diag(temp) <- 0
      diag(temp) <- -colSums(temp)
      fit[[i]] <- temp
      residuals[i] <- sqrt(sum((gl[[i]] - temp)^2))
      # eigenDecom <- eigen(fit[[i]])
      # Lambda <- pmax(Re(eigenDecom$values), 0)
      # U <- eigenDecom$vectors
      # fitiAlpha <- U%*%diag(Lambda^optns$alpha)%*%t(U)
      # residuals[i] <- sqrt(sum((glAlpha[[i]]-fitiAlpha)^2))# using Euclidean power metric
    }
    if (nOut > 0) {
      predict <- vector(mode = "list", length = nOut)
      for (i in 1:nOut) {
        a <- xOut[i, ]
        if (p > 1) {
          mu1 <- rowMeans(apply(x, 1, function(xi) K(xi - a, optns$bw) * (xi - a)))
          mu2 <- matrix(rowMeans(apply(x, 1, function(xi) K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a)))), ncol = p)
        } else {
          mu1 <- mean(apply(x, 1, function(xi) K(xi - a, optns$bw) * (xi - a)))
          mu2 <- mean(apply(x, 1, function(xi) K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a))))
        }
        mu2_inv = tryCatch(solve(mu2), 
                           error = function(e){
                             MASS::ginv(mu2)
                           } )
        wc <- t(mu1) %*% mu2_inv # 1 by p
        w <- apply(x, 1, function(xi) {
          K(xi - a, optns$bw) * (1 - wc %*% (xi - a))
        }) # weight
        bAlpha <- matrix(apply(glAlphaVec, 2, weighted.mean, w), ncol = m) # m^2
        eigenDecom <- eigen(bAlpha)
        Lambda <- pmax(Re(eigenDecom$values), 0) # projection to M_m
        U <- eigenDecom$vectors
        qNew <- as.vector(U %*% diag(Lambda^(1 / optns$alpha)) %*% t(U)) # inverse power
        model$Update(q = -qNew)
        temp <- matrix(model$Solve()$x, ncol = m, dimnames = list(nodes, nodes))
        temp <- (temp + t(temp)) / 2 # symmetrize
        if (!is.na(optns$digits)) temp <- round(temp, digits = optns$digits) # round
        temp[temp > 0] <- 0 # off diagonal should be negative
        diag(temp) <- 0
        diag(temp) <- -colSums(temp)
        predict[[i]] <- temp
      }
      res <- list(fit = fit, predict = predict, residuals = residuals, gl = gl, x = x, xOut = xOut, optns = optns)
    } else {
      res <- list(fit = fit, residuals = residuals, gl = gl, x = x, optns = optns)
    }
  }
  class(res) <- "nr"
  res
}
