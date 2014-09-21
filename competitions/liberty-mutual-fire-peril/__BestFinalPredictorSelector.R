
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

##############  Loading data sets (train, test, sample) ... 
source(paste0(getBasePath("code") , "__BestFinalPredictorSelector_Lib.R"))

train.fn = "train.csv"
test.fn = "test.csv"
sampleSub.fn = "sampleSubmission.csv"

train.tab = fread(paste(getBasePath(),train.fn,sep="") , header = TRUE , sep=","  )
test.tab = fread(paste(getBasePath(),test.fn,sep="") , header = TRUE , sep=","  )

## data formatting 
train = getData4Analysis(train.tab)
test = getData4Analysis(test.tab)

train.bkp = train 
test.bkp = test 

dim(train)
dim(test)

cat("==========================> Finding best predictors for regression problem ... ")

l = getBestPredictors(train[ , -c(1,2,303)] , response = train$target , test , 
                      polynomialDegree = 1 , NAthreshold = 0.05, 
                      pValueAdjust = F, verbose = T )
BestPredictorTrainIndex = l[[1]]
BestPredictorTestIndex = l[[2]]

cat("on train set, found ", length(BestPredictorTrainIndex)  , "predictors:" , 
    paste(colnames(train[ , -c(1,2,303)]) [BestPredictorTrainIndex] , collapse=" " ) , " \n") 
cat("on test set, found ", length(BestPredictorTestIndex)  , "predictors:" , 
    paste(colnames(test) [BestPredictorTestIndex] , collapse=" " ) , " \n") 

ytrain = train$target
Xtrain = train[ , -c(1,2,303)] [,BestPredictorTrainIndex]
Xtest = test[,BestPredictorTestIndex]

## describe 
library(Hmisc)
describe(x = ytrain)
describe(x = Xtrain)
describe(x = Xtest)

write.csv(ytrain,quote=F,row.names=F,file=paste0(getBasePath(),"ytrain_reg.csv"))
write.csv(Xtrain,quote=F,row.names=F,file=paste0(getBasePath(),"Xtrain_reg.csv"))
write.csv(Xtest,quote=F,row.names=F,file=paste0(getBasePath(),"Xtest_reg.csv"))

cat("==========================> Finding best predictors for classification problem ... ")

l = getBestPredictors(train[ , -c(1,2,303)] , response = train$target_0 , test , 
                      polynomialDegree = 1 , NAthreshold = 0.05, 
                      pValueAdjust = T, pValueAdjustMethod = "bonferroni", verbose = T )
BestPredictorTrainIndex = l[[1]]
BestPredictorTestIndex = l[[2]]

cat("on train set, found ", length(BestPredictorTrainIndex)  , "predictors:" , 
    paste(colnames(train[ , -c(1,2,303)]) [BestPredictorTrainIndex] , collapse=" " ) , " \n") 
cat("on test set, found ", length(BestPredictorTestIndex)  , "predictors:" , 
    paste(colnames(test) [BestPredictorTestIndex] , collapse=" " ) , " \n") 

ytrain = train$target_0
Xtrain = train[ , -c(1,2,303)] [,BestPredictorTrainIndex]
Xtest = test[,BestPredictorTestIndex]

## describe 
library(Hmisc)
describe(x = ytrain)
describe(x = Xtrain)
describe(x = Xtest)

write.csv(ytrain,quote=F,row.names=F,file=paste0(getBasePath(),"ytrain_class.csv"))
write.csv(Xtrain,quote=F,row.names=F,file=paste0(getBasePath(),"Xtrain_class.csv"))
write.csv(Xtest,quote=F,row.names=F,file=paste0(getBasePath(),"Xtest_class.csv"))


