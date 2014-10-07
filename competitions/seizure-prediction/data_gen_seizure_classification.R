## notes_seizure_R
library(caret)
library(Hmisc)
library(data.table)
library(verification)

getBasePath = function (type = "data" , ds="") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/seizure-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/seizure-prediction/"
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
  
  if (ds != "" ) {
    ret = paste0(paste0(ret,ds),"_digest/")
  }
  ret
} 

ds = "Dog_2"

######### load data sets 
Xtrain_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"Xtrain_mean_sd.zat",sep="") , header = F , sep=","  ))
Xtrain_quant = as.data.frame( fread(paste(getBasePath(type = "data" , ds=ds),"Xtrain_quant.zat",sep="") , header = F , sep=","  ))

Xtest_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"Xtest_mean_sd.zat",sep="") , header = F , sep=","  ))
Xtest_quant = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"Xtest_quant.zat",sep="") , header = F , sep=","  ))

ytrain = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"ytrain.zat",sep="") , header = F , sep=","  ))

######### making train / xval set ... 
#Xtrain_mean_sd$preict = as.factor(ytrain[,2])
#Xtrain_quant$preict = as.factor(ytrain[,2])

set.seed(975)
forTraining <- createDataPartition(ytrain[,2], p = 3/4)[[1]]
Xtrain_mean_sd.train <- Xtrain_mean_sd[ forTraining,]
Xtrain_mean_sd.xval <- Xtrain_mean_sd[-forTraining,]

Xtrain_quant.train <- Xtrain_quant[ forTraining,]
Xtrain_quant.xval <- Xtrain_quant[-forTraining,]

ytrain.train = ytrain[forTraining,]
ytrain.test = ytrain[-forTraining,]

######### models in action 
predictAndMeasure = function(model,model.label,trainingData,ytrain,testData,ytest,tm , 
                             ytrain.cat , ytest.cat ,   
                             grid = NULL,verbose=F) {
  pred.train = predict(model , trainingData) 
  RMSE.train = RMSE(obs = ytrain , pred = pred.train)
  
  pred = predict(model , testData) 
  RMSE.test = RMSE(obs = ytest , pred = pred)
  
  if (verbose) cat("****** RMSE(train) =",RMSE.train," -  RMSE(test) =",RMSE.test,"  --  Time elapsed(sec.):",tm[[3]], "...  \n")
  
  ## fitter.cat 
  train.cat = data.frame( cat = ytrain.cat , pr =  pred.train )
  test.cat = data.frame(cat = ytest.cat , pr =  pred )
  ctrl <- trainControl(summaryFunction = twoClassSummary, classProbs = TRUE)
  fitter.cat <- train( cat ~ pr ,  data=train.cat , method = "glm", metric = "ROC", trControl = ctrl)
  pred.train.cat = predict(fitter.cat, train.cat ) 
  pred.cat = predict(fitter.cat, test.cat ) 
  
  acc.train = sum(ytrain.cat == pred.train.cat) / length(ytrain.cat)
  acc.test = sum(ytest.cat == pred.cat) / length(ytest.cat)
  
  roc.train = roc.area(as.numeric(ytrain.cat == 1) , as.numeric(pred.train.cat == 1) )$A
  roc.test = roc.area(as.numeric(ytest.cat == 1) , as.numeric(pred.cat == 1) )$A
  
  ### perf.grid 
  perf.grid = NULL
  if (is.null(grid)) { 
    perf.grid = data.frame(predictor = c(model.label) , RMSE.train = c(RMSE.train) , RMSE.test = c(RMSE.test) , 
                           acc.train = c(acc.train) , acc.test = c(acc.test) , 
                           roc.train = c(roc.train) , roc.test =c(roc.test),
                           time = c(tm[[3]]))
  } else {
    .grid = data.frame(predictor = c(model.label) , RMSE.train = c(RMSE.train) , 
                       RMSE.test = c(RMSE.test) , 
                       acc.train = c(acc.train) , acc.test = c(acc.test) , 
                       roc.train = c(roc.train) , roc.test =c(roc.test),
                       time = c(tm[[3]]) )
    perf.grid = rbind(grid, .grid)
  }
  
  perf.grid
}

verbose = T
ctrl <- trainControl(method = "LGOCV", summaryFunction = twoClassSummary, classProbs = TRUE , savePredictions = T)

######################################################## logistic regression 
if (verbose) cat("****** [Xtrain_mean_sd] logistic regression ...  \n")
set.seed(669); ptm <- proc.time()
logisticReg <- train(Xtrain_mean_sd.train , y = as.factor(ytrain.train[,2]), method = "glm", metric = "ROC", trControl = ctrl) 
if (verbose) logisticReg
tm = proc.time() - ptm
perf.grid = predictAndMeasure (linearReg,"Logistic Reg (mean sd)",Xtrain_mean_sd.train,
                               Xtrain_mean_sd.train$time_before_seizure,
                               Xtrain_mean_sd.xval,Xtrain_mean_sd.xval$time_before_seizure,
                               tm ,  ytrain.cat , ytest.cat ,    
                               grid = NULL , verbose)

if (verbose) cat("****** [Xtrain_quant] linear regression ...  \n")
set.seed(669); ptm <- proc.time()
linearReg.quant <- train(time_before_seizure ~  . , data = Xtrain_quant.train, method = "lm", trControl = controlObject) 
if (verbose) linearReg.quant
tm = proc.time() - ptm
perf.grid = predictAndMeasure (linearReg.quant,"Linear Reg (quantiles)",Xtrain_quant.train,
                               Xtrain_quant.train$time_before_seizure,
                               Xtrain_quant.xval,Xtrain_quant.xval$time_before_seizure, 
                               tm,  ytrain.cat , ytest.cat ,    
                               grid = perf.grid , verbose )


##### saving on disk 
write.csv(perf.grid,quote=FALSE,file=paste0(getBasePath(),"Dog_1_perf_grid_regress.csv"), row.names=FALSE)
