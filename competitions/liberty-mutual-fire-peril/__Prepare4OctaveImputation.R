
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

############# LIBs
source(paste0(getBasePath("code") , "SelectBestPredictors_Lib.R"))
source(paste0( getBasePath("code") , "Impute_Lib.R"))

############## Loading data sets and finding best predictor set... 
Xtrain = Xtest = ytrain = NULL

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

############## computing ImputePredictors matrix ... 
ImputePredictors = prepare4Octave (data = Xtest , verbose = T , debug = F)

############## formatting data frame for Octave and serializing ... 
lIdx = which(as.numeric(lapply(ImputePredictors,is.logical)) == 1)
for (i in lIdx) {
  ImputePredictors[,i] = as.numeric(ifelse(ImputePredictors[,i], 1 , 0))
}

ImputePredictors4Octave = data.frame(predictor = 1:(dim(ImputePredictors)[1]) )
ImputePredictors4Octave$Nas = ImputePredictors$NAs
ImputePredictors4Octave$need.impute = ImputePredictors$need.impute
ImputePredictors4Octave$is.factor = ImputePredictors$is.factor

imax = 0 
for (i in 1:(dim(ImputePredictors)[1]) ) {
  v = as.numeric(unlist( strsplit( ImputePredictors[i,]$predictorIndex , "-" ) ) )
  if (length(v) > imax) {
    imax = length(v)
  } 
}

for (i in 1:imax ) {
  tmp = data.frame(x = rep(-1,(dim(ImputePredictors)[1])) )
  colnames(tmp) = c(paste0("predIdx_",i) )
  ImputePredictors4Octave = cbind(ImputePredictors4Octave , tmp)
}

for (i in 1:(dim(ImputePredictors)[1]) ) {
  v = as.numeric(unlist( strsplit( ImputePredictors[i,]$predictorIndex , "-" ) ) )
  for (j in 1:length(v)) {
    if (! is.na(v[j])) ImputePredictors4Octave[i,(4+j)] = v[j]
  }
}

write.csv(ImputePredictors4Octave,quote=T,row.names=F,file=paste0(getBasePath(),"ImputePredictors4octave.zat"))

############## prepare Xtrain, Xtest for each predictor to impute  TODO

  
  