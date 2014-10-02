

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/liberty-mutual-fire-peril"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/liberty-mutual-fire-peril"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/liberty-mutual-fire-peril/"
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

##############  Loading data sets (train, test, sample) ... 
source(paste0( getBasePath("code") , "/__BestFinalPredictorSelector_Lib.R"))
source(paste0( getBasePath("code") , "/__Impute_Lib.R"))

Xtrain = as.data.frame(fread(paste0(getBasePath(), "Xtrain_reg.csv" ) , header = TRUE , sep=","  ))
Xtest = as.data.frame(fread(paste0(getBasePath(), "Xtest_reg.csv" ) , header = TRUE , sep="," ) )
ytrain = as.data.frame(fread(paste0(getBasePath(), "ytrain_reg.csv" ) , header = TRUE   ))


Xtrain$var4 = as.factor(Xtrain$var4)
Xtrain$dummy = as.factor(Xtrain$dummy)

Xtest$var4 = as.factor(Xtest$var4)
Xtest$dummy = as.factor(Xtest$dummy)

## imputing Xtest 
l = imputeFastFurious (data = Xtest , verbose = T , debug = F)
Xtest.imputed = l[[1]]
ImputePredictors = l[[2]]
DecisionMatrix = l[[3]]

write.csv(Xtest.imputed,quote=F,row.names=F,file=paste0(getBasePath(),"/Xtest_reg_imputed.csv"))
write.csv(ImputePredictors,quote=F,row.names=F,file=paste0(getBasePath(),"/Xtest_reg_imputeModel___ImputePredictors.csv"))


