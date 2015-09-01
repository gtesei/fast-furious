
#' Filter predictors according to specified criteria. 
#' 
#' @param traindata the train set 
#' @param testdata the test set 
#' @param y the response variable. Must be not \code{NULL} if \code{correlationThreshold} is not \code{NULL}. 
#' @param removeOnlyZeroVariacePredictors \code{TRUE} to remove only zero variance predictors  
#' @param performVarianceAnalysisOnTrainSetOnly \code{TRUE} to perform the variance analysis on the train set only  
#' @param correlationThreshold a correlation threshold above which keeping predictors 
#'        (considered only if \code{removeOnlyZeroVariacePredictors} is \code{FALSE}).  
#' @param removePredictorsMakingIllConditionedSquareMatrix \code{TRUE} to predictors making ill conditioned square matrices 
#' @param removeHighCorrelatedPredictors \code{TRUE} to remove high correlared predictors 
#' @param featureScaling \code{TRUE} to perform feature scaling
#' @param verbose \code{TRUE} to set verbose mode 
#' 
#' @examples
#' Xtrain <- data.frame( a = rep(1:3 , each = 2), b = c(4:1,6,6), c = rep(1,6))
#' Xtest <-  Xtrain + runif(nrow(Xtrain))
#' l = ff.featureFilter (traindata = Xtrain,
#'                        testdata = Xtest,
#'                        removeOnlyZeroVariacePredictors=TRUE)
#' Xtrain = l$traindata
#' Xtest = l$testdata 
#' @importFrom caret preProcess
#' @importFrom caret nearZeroVar
#' @importFrom subselect trim.matrix
#' @export
#' @return the list of trainset and testset after applying the specified filters 
#' 

ff.featureFilter <- function(traindata,
                          testdata,
                          y = NULL,
                          removeOnlyZeroVariacePredictors=FALSE,
                          performVarianceAnalysisOnTrainSetOnly = TRUE , 
                          correlationThreshold = NULL, 
                          removePredictorsMakingIllConditionedSquareMatrix = TRUE, 
                          removeHighCorrelatedPredictors = TRUE, 
                          featureScaling = TRUE, 
                          verbose = TRUE) {
  
  stopifnot(  ! (is.null(testdata) && is.null(traindata)) )
  stopifnot(  ! (removeOnlyZeroVariacePredictors && (! is.null(correlationThreshold))) )
  stopifnot(  ! (is.null(y) && (! is.null(correlationThreshold))) )
  
  data = rbind(testdata,traindata)
  
  ### removing near zero var predictors 
  if (! removeOnlyZeroVariacePredictors ) { 
    PredToDel = NULL 
    if (performVarianceAnalysisOnTrainSetOnly) { 
      if (verbose) cat(">>> applying caret nearZeroVar performing caret nearZeroVar function on train set only ... \n")
      PredToDel = caret::nearZeroVar(traindata)
    } else {
      if (verbose) cat(">>> applying caret nearZeroVar performing caret nearZeroVar function on both train set and test set ... \n")
      PredToDel = caret::nearZeroVar(data)
    }
    
    if (! is.null(correlationThreshold) ) {
      if (verbose) cat(">>> computing correlation ... \n")
      corrValues <- apply(traindata,
                               MARGIN = 2,
                               FUN = function(x, y) cor(x, y),
                               y = y)
      PredToReinsert = as.numeric(which(! is.na(corrValues) & corrValues > correlationThreshold))
      
      if (verbose) cat(">> There are high correlated predictors with response variable. N. ",length(PredToReinsert)," - predictors: ", 
          paste(colnames(data) [PredToReinsert] , collapse=" " ) , " ... \n ")
      
      PredToDel = PredToDel[! PredToDel %in% PredToReinsert]
    } 
    
    if (length(PredToDel) > 0) {
      if (verbose) cat("removing ",length(PredToDel)," nearZeroVar predictors: ", 
          paste(colnames(data) [PredToDel] , collapse=" " ) , " ... \n ")
      data  =  data  [,-PredToDel]
    } 
  } else {
    if (verbose) cat(">>> removing zero variance predictors only  ... \n")
    card = NULL
    
    if (performVarianceAnalysisOnTrainSetOnly) { 
      if (verbose) cat(">>> removing zero variance predictors only performing variance analysis on train set only ... \n")
      card = apply(traindata,2,function(x)  length(unique(x))  )
    } else {
      if (verbose) cat(">>> removing zero variance predictors only performing variance analysis on both train set and test set ... \n")
      card = apply(data,2,function(x)  length(unique(x))  )
    }
    
    PredToDel = as.numeric(which(card < 2))
    if (length(PredToDel) > 0) {
      if (verbose) cat("removing ",length(PredToDel)," ZeroVariacePredictors predictors: ", 
          paste(colnames(data) [PredToDel] , collapse=" " ) , " ... \n ")
      data  =  data  [,-PredToDel]
    } 
  }
  
  ### removing predictors that make ill-conditioned square matrix
  if (removePredictorsMakingIllConditionedSquareMatrix) {
    if (verbose) cat(">>> finding for predictors that make ill-conditioned square matrix ... \n")
    PredToDel = subselect::trim.matrix( cov( data ) )
    if (length(PredToDel$numbers.discarded) > 0) {
      if (verbose) cat("removing ",length(PredToDel$numbers.discarded)," predictors that make ill-conditioned square matrix: ", 
          paste(colnames(data) [PredToDel$numbers.discarded] , collapse=" " ) , " ... \n ")
      data  =  data  [,-PredToDel$numbers.discarded]
    }
  }
  
  # removing high correlated predictors 
  if (removeHighCorrelatedPredictors) {
    if (verbose) cat(">>> finding for high correlated predictors ... \n")
    PredToDel = caret::findCorrelation(cor( data )) 
    if (length(PredToDel) > 0) {
      if (verbose) cat("removing ",length(PredToDel), " removing high correlated predictors: ",
          paste(colnames(data) [PredToDel] , collapse=" " ) , " ... \n ")
      data =  data  [,-PredToDel]
    }
  }

  ## feature scaling 
  if (featureScaling) {
    if (verbose) cat(">>> feature scaling ... \n")
    scaler = caret::preProcess(data,method = c("center","scale"))
    data = predict(scaler,data)
  }
  
  ## reassembling 
  if ( ! is.null(testdata) && ! is.null(traindata) ) {
    testdata = data[1:(dim(testdata)[1]),]
    traindata = data[((dim(testdata)[1])+1):(dim(data)[1]),]  
  } else if (is.null(testdata)) {
    traindata = data 
  } else if (is.null(traindata)) {
    testdata = data 
  }
  
  return(list(traindata = traindata,testdata = testdata))
}

#' Make polynomial terms of a \code{data.frame} 
#' 
#' @param x a \code{data.frame} of \code{numeric}
#' @param n the polynomial degree 
#' @param direction if set to \code{0} returns the terms \code{x^(1/n),x^(1/(n-1)),...,x,x^2,...,x^n}. 
#' If set to \code{-1} returns the terms \code{x^(1/n),x^(1/(n-1)),...,x}.
#' If set to \code{1} returns the terms\code{x,x^2,...,x^n}. 
#' 
#' @examples
#' Xtrain <- data.frame( a = rep(1:3 , each = 2), b = c(4:1,6,6), c = rep(1,6))
#' Xtest <-  Xtrain + runif(nrow(Xtrain))
#' data = rbind(Xtrain,Xtest)
#' data.poly = ff.poly(x=data,n=3)
#' Xtrain.poly = data.poly[1:nrow(Xtrain),]
#' Xtest.poly = data.poly[(nrow(Xtrain)+1):nrow(data),]
#' @export
#' @return the \code{data.frame} with the specified polynomial terms
#'

ff.poly = function (x,n,direction=0) {
  stopifnot(identical(class(x),'data.frame') , identical(class(n),'numeric') )
  stopifnot(  sum(unlist(lapply(x,function(x) {
    return(! (is.atomic(x)  && (! is.character(x)) && ! is.factor(x))  )
  }))) == 0 )
  
  if (n == 1) {
    return (x)
  } 
  
  x.poly = NULL
  x.poly.2 = NULL
  
  ##
  if (direction>=0) {
    x.poly = as.data.frame(matrix(rep(0 , nrow(x)*ncol(x)*(n-1)) , nrow = nrow(x)))
    lapply(2:n,function(i){
      d = x 
      d[] <- lapply(X = x , FUN = function(x){
        return(x^i)
      })  
      colnames(d) = paste(colnames(x),'^',i,sep='')
      x.poly[,((i-2)*ncol(x)+1):((i-1)*ncol(x))] <<- d 
      colnames(x.poly)[((i-2)*ncol(x)+1):((i-1)*ncol(x))] <<- colnames(d)
    })   
  }
 
  
  ##
  if (direction<=0) {
    x.poly.2 = as.data.frame(matrix(rep(0 , nrow(x)*ncol(x)*(n-1)) , nrow = nrow(x)))
    lapply(2:n,function(i){
      d = x 
      d[] <- lapply(X = x , FUN = function(x){
        return(x^(1/i))
      })  
      colnames(d) = paste(colnames(x),'^1/',i,sep='')
      x.poly.2[,((i-2)*ncol(x)+1):((i-1)*ncol(x))] <<- d 
      colnames(x.poly.2)[((i-2)*ncol(x)+1):((i-1)*ncol(x))] <<- colnames(d)
    })
  }
 
  ##
  if (direction>0) {
    return (cbind(x,x.poly))
  } else if (direction==0) {
    return (cbind(x,x.poly,x.poly.2))
  } else {
    return (cbind(x,x.poly.2)) 
  }
}