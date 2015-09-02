
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
#' @param removeIdenticalPredictors \code{TRUE} to remove identical predictors (using \code{base::identical} function) 
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
                          removeIdenticalPredictors = FALSE, 
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
  
  ## removing identical predictors 
  if (removeIdenticalPredictors) {
    colToRemove = rep(F,ncol(data))
    
    lapply( 1:(ncol(data)-1) , function(i) {
      lapply( (i+1):ncol(data) ,function(j) {
        if (identical(data[,i],data[,j])) {
          colToRemove[j] <<- T
        } 
      })
    })
    
    if (sum(colToRemove) > 0) {
      if (verbose) cat("removing ",sum(colToRemove)," identical predictors: ", 
                       paste(colnames(data) [colToRemove] , collapse=" " ) , " ... \n ")
    }
    data = data[,-which(colToRemove)]
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
#' If set to \code{1} returns the terms \code{x,x^2,...,x^n}. 
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

#' Filter a \code{data.frame} of numeric according to a given threshold of correlation 
#' 
#' @param Xtrain a train set \code{data.frame} of \code{numeric}
#' @param Xtest a test set \code{data.frame} of \code{numeric}
#' @param y the output variable (as numeric vector)
#' @param method a character string indicating which correlation method is to be used for the test. One of "pearson", "kendall", or "spearman".
#' @param abs_th an absolute threshold (= number of data frame columns)
#' @param rel_th a relative threshold (= percentage of data frame columns)
#' @param verbose  \code{TRUE} to enable verbose mode 
#' 
#' @examples
#' Xtrain <- data.frame( a = rep(1:3 , each = 2), b = c(4:1,6,6), c = rep(1,6))
#' Xtest <-  Xtrain + runif(nrow(Xtrain))
#' y = 1:6
#' l = ff.corrFilter(Xtrain=Xtrain,Xtest=Xtest,y=y,rel_th=0.5)
#' Xtrain.filtered = l$Xtrain
#' Xtest.filtered =l$Xtest 
#' @export
#' @return a \code{list} of filtered train set and test set with correlation test results 
#'

ff.corrFilter = function(Xtrain,Xtest,y,abs_th=NULL,rel_th=1,method = 'pearson',verbose=F) {
  warn_def = getOption('warn')
  options(warn=-1)
  
  ####
  stopifnot(is.null(rel_th) || is.null(abs_th))
  if (! is.null(rel_th) ) stopifnot(  rel_th >0 && rel_th <=1 )
  if (! is.null(abs_th) ) stopifnot(  abs_th >0 && abs_th <=ncol(Xtrain) )
  
  stopifnot(  ! (is.null(Xtrain) || is.null(Xtest)) )
  stopifnot(  ncol(Xtrain) == ncol(Xtest) )
  stopifnot(  ncol(Xtrain) > 0  )
  stopifnot(  nrow(Xtrain) > 0  )
  stopifnot(  nrow(Xtest) > 0  )
  
  stopifnot(  sum(unlist(lapply(Xtrain,function(x) {
    return(! (is.atomic(x)  && ! is.character(x) && ! is.factor(x)))
  }))) == 0 )
  
  stopifnot(  sum(unlist(lapply(Xtest,function(x) {
    return(! (is.atomic(x)  && ! is.character(x) && ! is.factor(x)))
  }))) == 0 )
  
  stopifnot(identical(method,'pearson') || identical(method,'kendall') || identical(method,'spearman')) 
  
  ### TypeIError test 
  getPvalueTypeIError = function(x,y) {
    test = NA
    pvalue = NA
    estimate = NA
    interpretation = NA 
    
    ## type casting and understanding stat test 
    if (class(x) == "integer") x = as.numeric(x)
    if (class(y) == "integer") y = as.numeric(y)
    
    if ( class(x) == "factor" & class(y) == "numeric" ) {
      # C -> Q
      test = "anova"
    } else if (class(x) == "factor" & class(y) == "factor" ) {
      # C -> C
      test = "chi-square"
    } else if (class(x) == "numeric" & class(y) == "numeric" ) {
      test = method
    }  else {
      # Q -> C 
      # it performs anova test x ~ y 
      test = "ANOVA"
      tmp = x 
      x = y 
      y = tmp 
    }
    
    ## performing stat test and computing p-value
    if (test == "anova") {                
      test.anova = aov(y~x)
      pvalue = summary(test.anova)[[1]][["Pr(>F)"]][1]
      estimate = NULL
      if (pvalue < 0.5) {
        interpretation = 'means differ'
      } else {
        interpretation = 'data do not give you any reason to conclude that means differ'
      }
    } else if (test == "chi-square") {    
      test.chisq = chisq.test(x = x , y = y)
      pvalue = test.chisq$p.value
      estimate = NULL
    } else {                             
      ###  pearson /  kendall / spearman
      test.corr = cor.test(x =  x , y =  y , method = method)
      pvalue = test.corr$p.value
      estimate = test.corr$estimate
      if (pvalue < 0.5) {
        interpretation = 'there is correlation'
      } else {
        interpretation = 'data do not give you any reason to conclude that the correlation is real'
      }
    }
    
    return(list(test=test,pvalue=pvalue,estimate=estimate,interpretation=interpretation))
  }
  
  ## 
  int_rel_th = abs_th
  if (! is.null(rel_th) ) {
    int_rel_th = floor(ncol(Xtrain) * rel_th)
  } 
  
  ## corr analysis 
  aa = lapply(Xtrain , function(x) {
    dummy = list(
      test = method,
      pvalue=1, 
      estimate = 0, 
      interpretation = "error")
    setNames(object = dummy , nm = names(x))
    
    ret = plyr::failwith( dummy, getPvalueTypeIError , quiet = !verbose)(x=x,y=y)
    
    return (ret)
  })
  
  ## make data frame test results  
  aadf = data.frame(predictor = rep(NA,length(aa)) , 
                    test = rep(NA,length(aa)) , 
                    pvalue = rep(NA,length(aa)) , 
                    estimate = rep(NA,length(aa)) , 
                    interpretation = rep(NA,length(aa)))
  lapply(seq_along(aa) , function(i) {
    aadf[i,]$predictor <<- names(aa[i])
    aadf[i,]$test <<- aa[[i]]$test
    aadf[i,]$pvalue <<- aa[[i]]$pvalue
    aadf[i,]$estimate <<- aa[[i]]$estimate
    aadf[i,]$interpretation <<- aa[[i]]$interpretation
  })
  aadf = aadf[order(abs(aadf$estimate) , decreasing = T), ]
  
  ## cut to the given threshold 
  aadf_cut = aadf[1:int_rel_th,,drop=F]
  
  options(warn=warn_def)
  
  return(list(
    Xtrain = Xtrain[,aadf_cut$predictor,drop=F],  
    Xtest = Xtest[,aadf_cut$predictor,drop=F], 
    test.results = aadf
  ))
  
}