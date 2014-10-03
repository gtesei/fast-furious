
getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/liberty-mutual-fire-peril"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/data_pre_process"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/data_pre_process/"
  } else {
    stop("unrecognized type.")
  }
  
  if (file.exists(base.path1))  {
    ret = paste0(base.path1,"/")
  } else {
    ret = base.path2
  }
  
  ret
} 

getData4Analysis = function(myData.tab) {
  myData = as.data.frame.matrix(myData.tab) 
  ## set NAs
  myData$var1 = ifelse(myData$var1 == "Z" , NA , myData$var1)
  myData$var2 = ifelse(myData$var2 == "Z" , NA , myData$var2)
  myData$var3 = ifelse(myData$var3 == "Z" , NA , myData$var3)
  myData$var4 = ifelse(myData$var4 == "Z" , NA , myData$var4)
  myData$var5 = ifelse(myData$var5 == "Z" , NA , myData$var5)
  myData$var6 = ifelse(myData$var6 == "Z" , NA , myData$var6)
  myData$var7 = ifelse(myData$var7 == "Z" , NA , myData$var7)
  myData$var8 = ifelse(myData$var8 == "Z" , NA , myData$var8)
  myData$var9 = ifelse(myData$var9 == "Z" , NA , myData$var9)
  
  ## set correct classes for regression 
  myData$var1 = as.numeric(myData$var1)
  myData$var2 = as.factor(myData$var2)
  myData$var3 = as.factor(myData$var3)
  
  ## TODO BETTER: perdi l'informazione sul secondo livello  
  #myData$var4_4 = factor(myData$var4 , ordered = T)
  myData$var4 = factor(myData$var4 )
  #myData$var4 = factor( ifelse(is.na(myData$var4), NA , substring(myData$var4 , 1 ,1) ) , ordered = T)
  #myData$var4 = factor( ifelse(is.na(myData$var4), NA , substring(myData$var4 , 1 ,1) ) )
  
  myData$var5 = as.factor(myData$var5)
  myData$var6 = as.factor(myData$var6)
  myData$var7 = as.numeric(myData$var7)
  myData$var8 = as.numeric(myData$var8)
  myData$var9 = as.factor(myData$var9)
  myData$dummy = as.factor(myData$dummy)
  
  if (! is.null(myData$target) ) {
    myData$target_0 = factor(ifelse(myData$target == 0,0,1))
  }
  
  ## garbage collector 
  gc() 
  
  ## return myData
  myData
}

##############  Loading data sets (train, test, sample) ... 
source(paste0(getBasePath("code") , "SelectBestPredictors_Lib.R"))

FIND_BEST_PREDICTORS_ONLY_FOR_REGRESSION = T 

train.fn = "train.csv"
test.fn = "test.csv"
sampleSub.fn = "sampleSubmission.csv"

cat ("loading ",paste(getBasePath(),train.fn,sep="")," and " , paste(getBasePath(),test.fn,sep="") ,"... \n")
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

if (! FIND_BEST_PREDICTORS_ONLY_FOR_REGRESSION) {
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
}
