# set up bandwidth range
SetBwRange <- function(xin, xout, kernel_type) {
  xinSt <- unique(sort(xin))
  bw.min <- max(diff(xinSt), xinSt[2] - min(xout), max(xout) -
                  xinSt[length(xinSt) - 1]) * 1.1 / (ifelse(kernel_type == "gauss", 3, 1) *
                                                     ifelse(kernel_type == "gausvar", 2.5, 1))
  bw.max <- diff(range(xin)) / 3
  if (bw.max < bw.min) {
    if (bw.min > bw.max * 3 / 2) {
      # warning("Data is too sparse.")
      bw.max <- bw.min * 1.01
    } else {
      bw.max <- bw.max * 3 / 2
    }
  }
  return(list(min = bw.min, max = bw.max))
}

# bandwidth selection via cross validation
bwCV <- function(xin, qin, xout, optns) {
  if(is.null(xout)) {
    xout <- xin
  }
  n <- nrow(xin)
  p <- ncol(xin)
  # initialization of OSQP solver
  M <- ncol(qin)
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
      osqp::osqpSettings(verbose = FALSE)
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
  
  # k-fold
  objFctn <- function(h) {
    numFolds <- ifelse(n > 30, 10, n)# leave-one-out or 10-fold cross-validation
    folds <- sample(c(rep.int(1:numFolds, n%/%numFolds), seq_len(n%%numFolds)))
    
    cv <- 0
    for (foldidx in seq_len(numFolds)) {
      # nn by M
      testidx <- which(folds == foldidx)
      for (j in testidx) {
        a <- xin[j, ]
        if (p > 1) {
          mu1 <-
            rowMeans(apply(xin[-testidx, ], 1, function(xi)
              K(xi - a, h) * (xi - a)))
          mu2 <-
            matrix(rowMeans(apply(xin[-testidx, ], 1, function(xi)
              K(xi - a, h) * ((xi - a) %*% t(xi - a)))), ncol = p)
        } else {
          mu1 <-
            mean(sapply(xin[-testidx, ], function(xi)
              K(xi - a, h) * (xi - a)))
          mu2 <-
            mean(sapply(xin[-testidx, ], function(xi)
              K(xi - a, h) * (xi - a)^2))
        }
        wc <- t(mu1) %*% solve(mu2) # 1 by p
        w <- apply(as.matrix(xin[-testidx, ]), 1, function(xi) {
          K(xi - a, h) * (1 - wc %*% (xi - a))
        }) # weight
        qNew <- apply(qin[-testidx,], 2, weighted.mean, w) # N
        if (any(w < 0)) {
          # if negative weights exist
          model$Update(q = -qNew)
          cv <- cv + sum((qin[j, ] - sort(model$Solve()$x))^2) / (n * M)
        } else {
          cv <- cv + sum((qin[j, ] - qNew)^2) / (n * M)
        }
      }
    }
    cv
  }
  
  if (p == 1) {
    aux <-
      SetBwRange(xin = xin[, 1],
                 xout = xout[, 1],
                 kernel_type = optns$kernel)
    bwRange <- matrix(c(aux$min, aux$max), nrow = 2, ncol = 1)
  } else {
    aux <-
      SetBwRange(xin = xin[, 1],
                 xout = xout[, 1],
                 kernel_type = optns$kernel)
    aux2 <-
      SetBwRange(xin = xin[, 2],
                 xout = xout[, 2],
                 kernel_type = optns$kernel)
    bwRange <-
      as.matrix(cbind(c(aux$min, aux$max), c(aux2$min, aux2$max)))
  }
  if (!is.null(optns$bw)) {
    if (p == 1) {
      if (min(optns$bw) < min(bwRange)) {
        message("Minimum bandwidth is too small and has been reset.")
      } else {
        bwRange[1, 1] <- min(optns$bw)
      }
      if (max(optns$bw) > min(bwRange)) {
        bwRange[2, 1] <- max(optns$bw)
      } else {
        message("Maximum bandwidth is too small and has been reset.")
      }
    } else {
      # Check for first dimension of the predictor
      if (min(optns$bw[, 1]) < min(bwRange[, 1])) {
        message("Minimum bandwidth of first predictor dimension is too small and has been reset.")
      } else {
        bwRange[1, 1] <- min(optns$bw[, 1])
      }
      if (max(optns$bw[, 1]) > min(bwRange[, 1])) {
        bwRange[2, 1] <- max(optns$bw[, 1])
      } else {
        message("Maximum bandwidth of first predictor dimension is too small and has been reset.")
      }
      # Check for second dimension of the predictor
      if (min(optns$bw[, 2]) < min(bwRange[, 2])) {
        message("Minimum bandwidth of second predictor dimension is too small and has been reset.")
      } else {
        bwRange[1, 2] <- min(optns$bw[, 2])
      }
      if (max(optns$bw[, 2]) > min(bwRange[, 2])) {
        bwRange[2, 2] <- max(optns$bw[, 2])
      } else {
        message("Maximum bandwidth of second predictor dimension is too small and has been reset.")
      }
    }
  }
  if (p == 1) {
    res <- optimize(f = objFctn, interval = bwRange[, 1])$minimum
  } else {
    res <-
      optim(
        par = colMeans(bwRange),
        fn = objFctn,
        lower = bwRange[1, ],
        upper = bwRange[2, ],
        method = "L-BFGS-B"
      )$par
  }
  res
}
