
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

############ MODE
GEN_FROM_SCRATCH = T 

############## Loading data sets (train, test, sample) ... 
Xtrain = Xtest = ytrain = NULL

if (! GEN_FROM_SCRATCH) {
  Xtrain = as.data.frame(fread(paste0(getBasePath(), "Xtrain_reg.csv" ) , header = TRUE , sep=","  ))
  Xtest = as.data.frame(fread(paste0(getBasePath(), "Xtest_reg.csv" ) , header = TRUE , sep="," ) )
  ytrain = as.data.frame(fread(paste0(getBasePath(), "ytrain_reg.csv" ) , header = TRUE   ))
  
  Xtrain$var4 = as.factor(Xtrain$var4)
  Xtrain$dummy = as.factor(Xtrain$dummy)
  
  Xtest$var4 = as.factor(Xtest$var4)
  Xtest$dummy = as.factor(Xtest$dummy)
} else {
  source(paste0( getBasePath("code") , "../competitions/liberty-mutual-fire-peril/__BestFinalPredictorSelector.R"))
}

ImputePredictors = prepare4Octave (data = Xtest , verbose = T , debug = F)

write.csv(ImputePredictors,quote=F,row.names=F,file=paste0(getBasePath(),"__ImputePredictors4octave.csv"))

  
  