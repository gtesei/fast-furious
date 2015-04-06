library(subselect)
library(caret)

featureSelect <- function(traindata,testdata,featureScaling = T) {
  data = rbind(testdata,traindata)
  
  ### removing predictors that make ill-conditioned square matrix
  PredToDel = trim.matrix( cov( data ) )
  if (length(PredToDel$numbers.discarded) > 0) {
    cat("removing ",length(PredToDel$numbers.discarded)," predictors that make ill-conditioned square matrix: ", 
        paste(colnames(data) [PredToDel$numbers.discarded] , collapse=" " ) , " ... \n ")
    data  =  data  [,-PredToDel$numbers.discarded]
  }
  
  ### removing near zero var predictors 
  PredToDel = nearZeroVar(data)
  if (length(PredToDel) > 0) {
    cat("removing ",length(PredToDel)," nearZeroVar predictors: ", 
        paste(colnames(data) [PredToDel] , collapse=" " ) , " ... \n ")
    data  =  data  [,-PredToDel]
  }
  
  # rmoving high correlated predictors on Xtrain_quant
  PredToDel = findCorrelation(cor( data )) 
  if (length(PredToDel) > 0) {
    cat("removing ",length(PredToDel), " predictors: ",
        paste(colnames(data) [PredToDel] , collapse=" " ) , " ... \n ")
    data =  data  [,-PredToDel]
  }
  
  ## feature scaling 
  if (featureScaling) {
    scaler = preProcess(data,method = c("center","scale"))
    data = predict(scaler,data)
  }
  
  ## reassembling 
  testdata = data[1:(dim(testdata)[1]),]
  traindata = data[((dim(testdata)[1])+1):(dim(data)[1]),]
  
  list(traindata,testdata)
}