library(data.table)

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/liberty-mutual-fire-peril/"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/liberty-mutual-fire-peril/"
  } else {
    stop("unrecognized type.")
  }
  
  if (file.exists(base.path1))  {
    ret = base.path1
  } else {
    ret = base.path2
  }
  
  ret
} 

buildDecisionMatrix = function (data) {
  predictors.name = colnames(data)
  DecisionMatrix = data.frame(predictor = predictors.name)
  
  tmp = matrix(rep(NA,length(predictors.name)) , nrow = length(predictors.name) , ncol = length(predictors.name) )
  colnames(tmp) = paste0("pvalue_",predictors.name)
  tmp = as.data.frame(tmp)
  DecisionMatrix = cbind(DecisionMatrix , tmp)
  
  DecisionMatrix$NAs = as.numeric( apply(data,2,FUN = function(x) sum(is.na(x))) )
  
  tmp = matrix(rep(NA,length(predictors.name)) , nrow = length(predictors.name) , ncol = length(predictors.name) )
  colnames(tmp) = paste0("NA_",predictors.name)
  tmp = as.data.frame(tmp)
  DecisionMatrix = cbind(DecisionMatrix , tmp)
  
  pnum = length(predictors.name)
  #pvalue
  for (i in 1:pnum) {
    for (j in 1:pnum) {
      if (i == j ) {
        DecisionMatrix[i,(j+1)] = 0 
      } else {
        tmp = na.omit(data[,c(i,j)])
        DecisionMatrix[i,(j+1)] <- getPvalueTypeIError(x = tmp[,2], y = tmp[,1])
      }
    }
  }
  #NA
  for (i in 1:pnum) {
    for (j in 1:pnum) {
      if (i == j ) {
        DecisionMatrix[i,(j+pnum+2)] = -1 
      } else {
        toImpute = sum(is.na(data[,i]))
        if (toImpute == 0){
          DecisionMatrix[i,(j+pnum+2)] = -1 
        } else {
          DecisionMatrix[i,(j+pnum+2)] <- sum(is.na(data[,i]) & is.na(data[,j]) )  / sum(is.na(data[,i]))
        }
      }
    }
  }
  
  DecisionMatrix
}

findImputePredictors = function(DecisionMatrix) {
  predictors.name = DecisionMatrix$predictor
  pnum = length(predictors.name)
  ImputePredictors = data.frame(predictor = DecisionMatrix$predictor, NAs = DecisionMatrix$NAs , 
                                need.impute = ifelse(DecisionMatrix$NAs > 0 , T , F)  , 
                                predictors = rep(NA,pnum) , predictorIndex = rep(NA,pnum))
  for (i in 1:pnum) {
    if(ImputePredictors[i,]$need.impute) {
      candidates = which(DecisionMatrix[i,((pnum+3):(2*pnum+2)),] == 0)
      stat.sign = which( DecisionMatrix[i,1+candidates,] < 0.05)
      predIdx = candidates[stat.sign] 
      ImputePredictors[i,]$predictors = paste(predictors.name[predIdx] , collapse = ",")
      ImputePredictors[i,]$predictorIndex = paste(predIdx , collapse = ",")
    } 
  }
  
  ImputePredictors
}



##############  Loading data sets (train, test, sample) ... 
source(paste0(getBasePath("code") , "__BestFinalPredictorSelector_Lib.R"))

Xtrain = as.data.frame(fread(paste0(getBasePath(), "Xtrain_reg.csv" ) , header = TRUE , sep=","  ))
Xtest = as.data.frame(fread(paste0(getBasePath(), "Xtest_reg.csv" ) , header = TRUE , sep="," ) )
ytrain = as.data.frame(fread(paste0(getBasePath(), "ytrain_reg.csv" ) , header = TRUE   ))


Xtrain$var4 = as.factor(Xtrain$var4)
Xtrain$dummy = as.factor(Xtrain$dummy)

Xtest$var4 = as.factor(Xtest$var4)
Xtest$dummy = as.factor(Xtest$dummy)

## building decision matrix on Xtest 
DecisionMatrix = buildDecisionMatrix(Xtest)
DecisionMatrix

ImputePredictors = findImputePredictors(DecisionMatrix)
ImputePredictors



