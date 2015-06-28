library(subselect)
library(caret)

featureSelect <- function(traindata,
                          testdata,
                          y = NULL,
                          removeOnlyZeroVariacePredictors=F,
                          performVarianceAnalysisOnTrainSetOnly = T , 
                          correlationRhreshold = NA, 
                          removePredictorsMakingIllConditionedSquareMatrix = T, 
                          removeHighCorrelatedPredictors = T, 
                          featureScaling = T) {
  
  data = rbind(testdata,traindata)
  
  ### removing near zero var predictors 
  if (! removeOnlyZeroVariacePredictors ) { 
    PredToDel = NULL 
    if (performVarianceAnalysisOnTrainSetOnly) { 
      cat(">>> applying caret nearZeroVar performing caret nearZeroVar function on train set only ... \n")
      PredToDel = nearZeroVar(traindata)
    } else {
      cat(">>> applying caret nearZeroVar performing caret nearZeroVar function on both train set and test set ... \n")
      PredToDel = nearZeroVar(data)
    }
    
    if (! is.na(correlationRhreshold) ) {
      cat(">>> computing correlation ... \n")
      corrValues <- apply(traindata,
                               MARGIN = 2,
                               FUN = function(x, y) cor(x, y),
                               y = y)
      PredToReinsert = as.numeric(which(! is.na(corrValues) & corrValues > correlationRhreshold))
      
      cat(">> There are high correlated predictors with response variable. N. ",length(PredToReinsert)," - predictors: ", 
          paste(colnames(data) [PredToReinsert] , collapse=" " ) , " ... \n ")
      
      PredToDel = PredToDel[! PredToDel %in% PredToReinsert]
    } 
    
    if (length(PredToDel) > 0) {
      cat("removing ",length(PredToDel)," nearZeroVar predictors: ", 
          paste(colnames(data) [PredToDel] , collapse=" " ) , " ... \n ")
      data  =  data  [,-PredToDel]
    } 
  } else {
    cat(">>> removing zero variance predictors only  ... \n")
    card = NULL
    
    if (performVarianceAnalysisOnTrainSetOnly) { 
      cat(">>> removing zero variance predictors only performing variance analysis on train set only ... \n")
      card = apply(traindata,2,function(x)  length(unique(x))  )
    } else {
      cat(">>> removing zero variance predictors only performing variance analysis on both train set and test set ... \n")
      card = apply(data,2,function(x)  length(unique(x))  )
    }
    
    PredToDel = as.numeric(which(card < 2))
    if (length(PredToDel) > 0) {
      cat("removing ",length(PredToDel)," ZeroVariacePredictors predictors: ", 
          paste(colnames(data) [PredToDel] , collapse=" " ) , " ... \n ")
      data  =  data  [,-PredToDel]
    } 
  }
  
  ### removing predictors that make ill-conditioned square matrix
  if (removePredictorsMakingIllConditionedSquareMatrix) {
    cat(">>> finding for predictors that make ill-conditioned square matrix ... \n")
    PredToDel = trim.matrix( cov( data ) )
    if (length(PredToDel$numbers.discarded) > 0) {
      cat("removing ",length(PredToDel$numbers.discarded)," predictors that make ill-conditioned square matrix: ", 
          paste(colnames(data) [PredToDel$numbers.discarded] , collapse=" " ) , " ... \n ")
      data  =  data  [,-PredToDel$numbers.discarded]
    }
  }
  
  # removing high correlated predictors 
  if (removeHighCorrelatedPredictors) {
    cat(">>> finding for high correlated predictors ... \n")
    PredToDel = findCorrelation(cor( data )) 
    if (length(PredToDel) > 0) {
      cat("removing ",length(PredToDel), " removing high correlated predictors: ",
          paste(colnames(data) [PredToDel] , collapse=" " ) , " ... \n ")
      data =  data  [,-PredToDel]
    }
  }

  ## feature scaling 
  if (featureScaling) {
    cat(">>> feature scaling ... \n")
    scaler = preProcess(data,method = c("center","scale"))
    data = predict(scaler,data)
  }
  
  ## reassembling 
  testdata = data[1:(dim(testdata)[1]),]
  traindata = data[((dim(testdata)[1])+1):(dim(data)[1]),]
  
  return(list(traindata = traindata,testdata = testdata))
}