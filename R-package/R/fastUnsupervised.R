
#' An useful wrapper of \code{\link[stats]{prcomp}} performing a principal components analysis on the given trainset / testset (\code{Xtrain} / \code{Xtest}).
#' 
#' @param Xtrain the encoded \code{data.frame} of train data. Must be a \code{data.frame} of \code{numeric}
#' @param Xtest the encoded \code{data.frame} of train data. Must be a \code{data.frame} of \code{numeric}
#' @param center a logical value indicating whether the variables should be shifted to be zero centered. 
#' Alternately, a vector of length equal the number of columns of data can be supplied. The value is passed to \code{scale}.
#' @param scale. a logical value indicating whether the variables should be scaled to have unit variance before the analysis takes place. 
#' The default is \code{FALSE} for consistency with S, but in general scaling is advisable. 
#' Alternatively, a vector of length equal the number of columns of data can be supplied. The value is passed to \code{scale}.
#' @param removeZeroVarPredictors a logical value indicating whether removing zero variance predictors before calling \code{\link[stats]{prcomp}} 
#' preventing errors due to the fact that the latter cannot rescale a constant/zero column to unit variance.    
#' @param varThreshold a threshold indicating the proportion of variance that should be explained. Must be a numeric between 0 and 1. 
#' @param doPlot a logical value indicating whether plotting the proportion of variance explained vs. principal components. 
#' @param verbose a logical value indicating whether verbose mode should be enabled.  
#' @examples
#' ## data 
#' Xtrain <- data.frame(a = rep(1:10 , each = 2), b = 20:1, 
#'                      c = rep(as.Date(c("2007-06-22", "2004-02-13")),10) , d = 20:1)
#' Xtest <- data.frame(a = rep(2:11 , each = 2), b = 1:20, 
#'                     c = rep(as.Date(c("2007-03-01", "2004-05-23")),10) , d = 1:20)
#' 
#' ## encode data sets 
#' l = ff.makeFeatureSet(Xtrain,Xtest,c("C","N","D","N"))
#' Xtrain = l$traindata
#' Xtest = l$testdata
#'
#' ffPCA = ff.pca(Xtrain = Xtrain , Xtest = Xtest , center = TRUE , scale. = TRUE , 
#'                removeZeroVarPredictors = TRUE , 
#'                varThreshold = 0.95 , doPlot = FALSE , verbose = TRUE)
#'                                
#' numComp <- ffPCA$numComp 
#' numComp.elbow <- ffPCA$numComp.elbow 
#' numComp.threshold <- ffPCA$numComp.threshold 
#' 
#' PC_Xtrain_95Var = ffPCA$PC.train[1:numComp.threshold,,drop=FALSE]
#' PC_Xtest_95Var = ffPCA$PC.test[1:numComp.threshold,,drop=FALSE]
#' 
#' @export
#' @return a list whose components are the number of principal components (\code{numComp}), the number of principal components to hold 
#' so that the proportion of variance 
#' explained by each subsequent principal component drops off as an elbow in the screen plot (\code{numComp.elbow}), the number of principal 
#' components explaining a given (specified by the \code{varThreshold} input parameter) proportion of variance (\code{numComp.threshold}), 
#' the threshold indicating the proportion of variance that should be explained (\code{varThreshold}), 
#' the cumulative sum of proportion of variance explained by each principal component (\code{cumVar}), the proportion of variance 
#' explained by each principal component (\code{var}), the principal components for train and test set (\code{PC.train} and \code{PC.test})      
#' 
ff.pca <- function(Xtrain,Xtest,
                   center=TRUE,scale.=FALSE,removeZeroVarPredictors=TRUE,
                   varThreshold = 0.95 , 
                   doPlot = TRUE, verbose=FALSE) {
  
  if (!is.null(Xtrain) & !is.null(Xtest)) stopifnot(  sum(colnames(Xtrain) != colnames(Xtest)) == 0 )
  stopifnot( ! is.null(Xtrain) || ! is.null(Xtest) )
  
  is_Xtrain_NULL <- is.null(Xtrain)
  is_Xtest_NULL <- is.null(Xtest)
  n.train <- if (is_Xtrain_NULL) NA else nrow(Xtrain)
  
  data = rbind(Xtrain,Xtest)
  rm(list=c("Xtrain","Xtest"))
  
  if(removeZeroVarPredictors) {
    x <- as.matrix(data)
    x <- scale(x, center = center, scale = scale.)
    cen <- attr(x, "scaled:center")
    sc <- attr(x, "scaled:scale")
    
    predToRemove = which(sc == 0) 
    if (length(predToRemove)>0) {
      if(verbose) cat(">>> removing zero variance predictors:",colnames(data)[predToRemove],"...\n")
      data = data[,-predToRemove,drop=F]
    } 
    rm(list=c("x"))
  }
  
  pr.out = prcomp(data, scale=scale. , retx = TRUE)
  rm(list=c("data"))
  
  if (verbose) cat(">>> found ",ncol(pr.out$x),"PCs ... \n")
  pr.var = pr.out$sdev^2
  pve = pr.var/sum(pr.var)
  
  PC.num = which(-diff(pve) == max(-diff(pve))) + 1 
  
  if (verbose) cat(">>> Number of PCs to hold according the elbow rule: first ",PC.num," PCs ... \n")
  
  cumVar <- cumsum(pve) 
  numCompTh <- which.max(cumVar > varThreshold)
  
  if (verbose) cat(">>> Number of PCs to hold explaining ",varThreshold*100,"% of variance: first ",numCompTh," PCs ... \n")
  
  if (doPlot) {
    if (length(pve)>500) {
      plot(pve[1:numCompTh] , xlab="Principal Component" , 
           ylab = "Proportion of Variance Explained" , ylim=c(0,1) , type='b' , 
           main = paste("Considered PCs explaining ",varThreshold*100,"% of Variance") )  
    } else {
      plot(pve , xlab="Principal Component" , 
           ylab = "Proportion of Variance Explained" , ylim=c(0,1) , type='b' , 
           main = paste("Considered PCs explaining 100% of Variance") )
    }
    
  } 
  
  ## PC.train / PC.test 
  if (!is_Xtrain_NULL && !is_Xtest_NULL) {
    PC.train = pr.out$x[1:n.train,,drop=F]
    PC.test = pr.out$x[((n.train+1):nrow(pr.out$x)),,drop=F]  
  } else if (!is_Xtrain_NULL ) {
    PC.train = pr.out$x
    PC.test = NULL  
  } else {
    PC.train = NULL
    PC.test = pr.out$x
  }
  
  return(list(numComp = ncol(pr.out$x) , numComp.elbow = PC.num , 
              numComp.threshold = numCompTh , varThreshold = varThreshold , 
              cumVar = cumVar , var = pve , 
              PC.train = PC.train ,  PC.test = PC.test ))
}

#' An useful wrapper of \code{\link[stats]{kmeans}} performing k-means clustering on the given trainset / testset (\code{Xtrain} / \code{Xtest}) and 
#' assuming a number of cluster from 1:\code{max_centers}. The best number of cluster is computed so that the variation in 
#' the within group sum of squares between two subsequent number of clusters is maximized in absolute value.  
#' 
#' @param Xtrain the encoded \code{data.frame} of train data. Must be a \code{data.frame} of \code{numeric}
#' @param Xtest the encoded \code{data.frame} of train data. Must be a \code{data.frame} of \code{numeric}
#' @param max_centers the max number of clusters to be evaluated 
#' @param nstart how many random sets should be chosen? Such a parameter is passed to \code{\link[stats]{kmeans}}. 
#' @param iter.max the maximum number of iterations allowed. Such a parameter is passed to \code{\link[stats]{kmeans}}. 
#' @param varThreshold a threshold indicating the proportion of variance that should be explained. Must be a numeric between 0 and 1. 
#' @param doPlot a logical value indicating whether plotting.  
#' @param verbose a logical value indicating whether verbose mode should be enabled.  
#' @examples
#' ## data 
#' Xtrain <- data.frame(a = rep(1:10 , each = 2), b = 20:1, 
#'                      c = rep(as.Date(c("2007-06-22", "2004-02-13")),10) , d = 20:1)
#' Xtest <- data.frame(a = rep(2:11 , each = 2), b = 1:20, 
#'                     c = rep(as.Date(c("2007-03-01", "2004-05-23")),10) , d = 1:20)
#' 
#' ## encode data sets 
#' l = ff.makeFeatureSet(Xtrain,Xtest,c("C","N","D","N"))
#' Xtrain = l$traindata
#' Xtest = l$testdata
#' 
#' ffKmeans = ff.kmeans(Xtrain = Xtrain , Xtest = Xtest , 
#'                      max_centers = 10 , verbose = TRUE)
#' 
#' best_n_cluters = ffKmeans$best_n_cluters 
#' 
#' K.train = ffKmeans$K.train
#' K.test = ffKmeans$K.test
#' 
#' @export
#' @return a list whose components are the max number of clusters evaluated (\code{max_centers}), the best number of clusters
#'  (\code{best_n_cluters}), the sequence of within groups sum of squares across number of clusters (\code{wss}), 
#'  assuming \code{best_n_cluters} as number of clusters a vector of integers (from 1:\code{best_n_cluters}) 
#'  indicating the cluster to which each point is allocated in the train set (\code{K.train}) and test set (\code{K.test}).       
#' 
ff.kmeans <- function(Xtrain,Xtest,
                     max_centers = 10, nstart = 5, iter.max = 10, 
                     doPlot=FALSE,verbose=FALSE) {
  if (!is.null(Xtrain) & !is.null(Xtest)) stopifnot(  sum(colnames(Xtrain) != colnames(Xtest)) == 0 )
  stopifnot( ! is.null(Xtrain) || ! is.null(Xtest) )
  
  is_Xtrain_NULL <- is.null(Xtrain)
  is_Xtest_NULL <- is.null(Xtest)
  n.train <- if (is_Xtrain_NULL) NA else nrow(Xtrain)
  
  data = rbind(Xtrain,Xtest)
  rm(list=c("Xtrain","Xtest"))
  
  wss = rep(NA,max_centers)
  for (i in 1:max_centers) wss[i] <- sum(kmeans(x = data , iter.max = iter.max , centers = i , nstart = nstart)$withinss)
  
  ## plot
  if (doPlot) {
    plot(1:max_centers, wss, type="b", 
         xlab="Number of Clusters",
         ylab="Within groups sum of squares", 
         main="Within groups sum of squares vs. Number of Clusters")  
  }
  
  ## find best number of clusters 
  wss_delta = rep(0,length(wss)-1)
  for (i in 1:length(wss)-1) wss_delta[i] = (wss[i+1] - wss[i])/wss[i]
  best_n_cluters = which(wss_delta == min(wss_delta))+1
  if (verbose) cat(">>> best number of clusters:",best_n_cluters," ...\n")
  
  ## kopt 
  kopt = kmeans(x = data , centers = best_n_cluters , nstart = nstart)
  
  ## PC.train / PC.test 
  if (!is_Xtrain_NULL && !is_Xtest_NULL) {
    K.train = kopt$cluster[1:n.train]
    K.test = kopt$cluster[((n.train+1):length(kopt$cluster))]  
  } else if (!is_Xtrain_NULL ) {
    K.train = kopt$cluster
    K.test = NULL  
  } else {
    K.train = NULL
    K.test = kopt$cluster
  }
  
  return(list(max_centers = max_centers,
              best_n_cluters = best_n_cluters,
              wss = wss , 
              K.train = K.train , 
              K.test = K.test ))
}

