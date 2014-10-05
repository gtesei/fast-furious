## notes_seizure_R
library(caret)
library(Hmisc)
library(data.table)

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/seizure-prediction/Dog_1_digest"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/seizure-prediction/Dog_1_digest/"
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

#########
Xtrain_mean_sd = fread(paste(getBasePath(),"Xtrain_mean_sd.zat",sep="") , header = TRUE , sep=","  )
Xtrain_quant = fread(paste(getBasePath(),"Xtrain_quant.zat",sep="") , header = TRUE , sep=","  )

Xtest_mean_sd = fread(paste(getBasePath(),"Xtest_mean_sd.zat",sep="") , header = TRUE , sep=","  )
Xtrain_mean_sd = fread(paste(getBasePath(),"Xtrain_mean_sd.zat",sep="") , header = TRUE , sep=","  )

ytrain = fread(paste(getBasePath(),"ytrain.zat",sep="") , header = TRUE , sep=","  )
#########
ls()
describe(Xtrain_mean_sd)
describe(Xtest_mean_sd)

describe(Xtrain_quant)
describe(Xtest_quant)

describe(ytrain)

ps = sample((dim(Xtrain_mean_sd)[2]),16)

featurePlot(x = Xtrain_mean_sd[,ps], y = ytrain,
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth")) 

featurePlot(x = Xtrain_quant[,ps], y = ytrain,
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth")) 


controlObject <- trainControl(method = "repeatedcv", repeats = 5, number = 10)


predictAndMeasure = function(model,model.label,trainingData,ytrain,testData,ytest,tm , grid = NULL,verbose=F) {
  pred = predict(model , trainingData) 
  RMSE.train = RMSE(obs = ytrain , pred = pred)
  
  pred = predict(model , testData) 
  RMSE.test = RMSE(obs = ytest , pred = pred)
  
  if (verbose) cat("****** RMSE(train) =",RMSE.train," -  RMSE(test) =",RMSE.test,"  --  Time elapsed(sec.):",tm[[3]], "...  \n")
  
  perf.grid = NULL
  if (is.null(grid)) { 
    perf.grid = data.frame(predictor = c(model.label) , RMSE.train = c(RMSE.train) , RMSE.test = c(RMSE.test) , time = c(tm[[3]]))
  } else {
    .grid = data.frame(predictor = c(model.label) , RMSE.train = c(RMSE.train) , RMSE.test = c(RMSE.test) , time = c(tm[[3]]))
    perf.grid = rbind(grid, .grid)
  }
  
  perf.grid
}
