library(caret)
library(Hmisc)
library(data.table)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)
library(fBasics)

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/restaurant-revenue-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/restaurant-revenue-prediction/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/restaurant-revenue-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/restaurant-revenue-prediction/"
  } else if (type == "process") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/data_process"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/data_process/"
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

buildData.basic = function(train.raw , test.raw) {
  ## remove id 
  train = train.raw[ , -1] 
  test = test.raw[ , -1] 
  
  ## 2014 should be the target year ... so use open date to misure the number of years between open date and the target year 
  train$years.to.target = 2014 - year(as.Date( train.raw[,2] , "%m/%d/%Y"))
  test$years.to.target = 2014 - year(as.Date( test.raw[,2] , "%m/%d/%Y"))
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## City
  l = encodeCategoricalFeature (train[,1] , test[,1] , colname.prefix = "city" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## City Group 
  l = encodeCategoricalFeature (train[,1] , test[,1] , colname.prefix = "city.group" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## Type
  l = encodeCategoricalFeature (train[,1] , test[,1] , colname.prefix = "type" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## extracting y 
  y = train[,38]
  train = train[,-38]
  
#   ## P29
#   l = encodeCategoricalFeature (train[,29] , test[,29] , colname.prefix = "P29" , asNumeric=F)
#   tr = l[[1]]
#   ts = l[[2]]
#   
#   train = cbind(train,tr)
#   test = cbind(test,ts)
#   
#   train = train[ , -29]
#   test = test[ , -29]
  
  ## P35
  l = encodeCategoricalFeature (train[,35] , test[,35] , colname.prefix = "P35" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -35]
  test = test[ , -35]
  
  ## P25
  l = encodeCategoricalFeature (train[,25] , test[,25] , colname.prefix = "P25" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -25]
  test = test[ , -25]
  
  ## P36
  l = encodeCategoricalFeature (train[,36] , test[,36] , colname.prefix = "P36" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -36]
  test = test[ , -36]
  
    ## P31
#     l = encodeCategoricalFeature (train[,31] , test[,31] , colname.prefix = "P31" , asNumeric=F)
#     tr = l[[1]]
#     ts = l[[2]]
#     
#     train = cbind(train,tr)
#     test = cbind(test,ts)
#     
#     train = train[ , -31]
#     test = test[ , -31]
  
  #   # P27
  #   l = encodeCategoricalFeature (train[,27] , test[,27] , colname.prefix = "P27" , asNumeric=F)
  #   tr = l[[1]]
  #   ts = l[[2]]
  #   
  #   train = cbind(train,tr)
  #   test = cbind(test,ts)
  #   
  #   train = train[ , -27]
  #   test = test[ , -27]
  
  #   ## P17
  #   l = encodeCategoricalFeature (train[,17] , test[,17] , colname.prefix = "P17" , asNumeric=F)
  #   tr = l[[1]]
  #   ts = l[[2]]
  #   
  #   train = cbind(train,tr)
  #   test = cbind(test,ts)
  #   
  #   train = train[ , -17]
  #   test = test[ , -17]
  #   
#     ## P5
#     l = encodeCategoricalFeature (train[,5] , test[,5] , colname.prefix = "P5" , asNumeric=F)
#     tr = l[[1]]
#     ts = l[[2]]
#     
#     train = cbind(train,tr)
#     test = cbind(test,ts)
#     
#     train = train[ , -5]
#     test = test[ , -5]
  
  ## 
  
  ## high revenue combinations 
  train$hrc = ifelse(train$P1 == 5 & train$P8 == 3 , 1 , 0) 
  test$hrc = ifelse(test$P1 == 5 & test$P8 == 3 , 1 , 0) 

  list(train,y,test)
}

predict.train.k.folds = function (traindata , 
                                  y , 
                                  model.label = "RandomForest_Reg", 
                                  controlObject , 
                                  k = 6 ) {
  ### train set 
  folds = kfolds(k,dim(traindata)[1])
  
  pred.1.train = rep(NA,dim(traindata)[1])
  
  for(j in 1:k) {  
    if (verbose) cat("--k-fold:: ",j, "/",k , "\n")
    traindata.train <- traindata[ folds != j,]
    traindata.y.train = y[folds != j]
    
    traindata.xval <- traindata[folds == j,]
    traindata.y.xval = y[folds == j]
    
    ###
    pred.1.train.fold = reg.trainAndPredict( traindata.y.train , 
                                             traindata.train , 
                                             traindata.xval , 
                                             model.label , 
                                             controlObject, 
                                             best.tuning = T) 
    
    pred.1.train[folds == j] = pred.1.train.fold
  } ### end of k-fold 
  
  ## check 
  if (sum(is.na(pred.1.train)) > 0)
    stop("something wrong (NAs) in tip.train")
  
  pred.1.train
}

build.pred.vector = function (Ids, init.lev, end.lev) {
  mat = matrix(NA,length(Ids),(end.lev-init.lev+2))
  mat[,1] = Ids
  pred.vector = as.data.frame(mat)
  colnames(pred.vector)[1] = "Id"
  colnames(pred.vector)[2:(end.lev-init.lev+2)] = paste0("Pred_",(init.lev:end.lev))
  
#   pred_final = rep(NA,length(Ids))
#   pred.vector = cbind(pred.vector,pred_final)
#   colnames(pred.vector)[(end.lev-init.lev+2)+1] = "Pred_final"
  
  pred.vector
}

build.res.vector = function (Ids, init.lev, end.lev) {
  mat = matrix(NA,length(Ids),(end.lev-init.lev+2))
  mat[,1] = Ids
  res.vector = as.data.frame(mat)
  colnames(res.vector)[1] = "Id"
  colnames(res.vector)[2:(end.lev-init.lev+2)] = paste0("res.",(init.lev:(end.lev)))
  
  res.vector
}

build.blank.model.boosting.grid = function (init.lev, end.lev) {
  model.boosting = expand.grid(
    lev = init.lev:end.lev ,  
    model = NA , 
    need.finding = T , 
    removeOnlyZeroVariacePredictors = c(T,F) , 
    performVarianceAnalysisOnTrainSetOnly = T , 
    correlationRhreshold = c(NA,0.1) , 
    res.train.mean = NA , 
    res.train.sd = NA , 
    RMSE.train.kfold = NA , 
    RMSE.xval.winner = NA 
  )
  
  toRemove = NULL
  for (ll in  (1:(dim(model.boosting)[1])) ) 
    if (model.boosting[ll,]$removeOnlyZeroVariacePredictors & 
          (! is.na(model.boosting[ll,]$correlationRhreshold)) ) 
      toRemove = c(toRemove,ll) 
      
  model.boosting[-toRemove,]
}

####### 
verbose = T

source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))
source(paste0( getBasePath("process") , "/Resample_Lib.R"))
source(paste0( getBasePath("process") , "/Regression_Lib.R"))
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))

controlObject <- trainControl(method = "repeatedcv", repeats = 10, number = 30)

cat(">>> resampling:: repeatedcv 10 30 \n")

RegModels = c("Average" , "Mode",  
              "LinearReg", "RobustLinearReg", 
              "PLS_Reg" , "Ridge_Reg" , "Enet_Reg" , 
              "KNN_Reg", 
              "SVM_Reg", 
              "BaggedTree_Reg"
              , "RandomForest_Reg"
              , "Cubist_Reg" 
              #, "NNet"
) 

#RegModels = c("Average","Mode")

cat("****** Available regression models ******\n") 
print(RegModels)

#######
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

train.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train.csv" , sep=''))) 

test.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                      "test.csv" , sep='')))

####### basic feature processing 
l = buildData.basic(train.raw , test.raw)
train = l[[1]]
y = l[[2]]
test = l[[3]]


##### BOOSTING MODELS 

##############################################  CONFIGURATION 
INIT_LEV = 2 
END_LEV = 4

.model.boosting = as.data.frame( fread(paste(getBasePath("data") , 
                                             "model_boosting0_grid.csv" , sep='')))

.pred.vector = as.data.frame( fread(paste(getBasePath("data") , 
                                          "model_boosting0_pred.test.csv" , sep='')))

.res.vector = as.data.frame( fread(paste(getBasePath("data") , 
                                         "model_boosting0_res.csv" , sep='')))

##############################################  END OF CONFIGURATION 

## working levels 
model.boosting = build.blank.model.boosting.grid(INIT_LEV,END_LEV)

## merge
model.boosting = rbind(.model.boosting , model.boosting)
model.boosting = model.boosting[order(model.boosting$lev , decreasing = F),]

LEVELS = sort(unique(model.boosting$lev))

cat ("********************",LEVELS,"*************************\n")
print(model.boosting)
cat ("*********************************************\n")

##### TEST PREDICTIONS  
pred.vector = build.pred.vector(Ids=test.raw$Id, init.lev=INIT_LEV ,end.lev=END_LEV)
pred.vector = merge(x = .pred.vector , y = pred.vector , by = c("Id") )

##### TRAIN RESIDUALS ESTIMATION  
res.vector = build.res.vector (Ids=train.raw$Id, init.lev = INIT_LEV, end.lev  = END_LEV)
res.vector = merge(x = .res.vector , y = res.vector , by = c("Id") )

##### PROCESSING   
for (level in LEVELS) {
  cat("******** PROCESSING LEVEL ",level,"********\n")
  
  ## unroll parameters  
  models = as.character(model.boosting[model.boosting$lev==level,]$model)
  need.finding = as.logical(model.boosting[model.boosting$lev==level,]$need.finding)
  removeOnlyZeroVariacePredictors = as.logical(model.boosting[model.boosting$lev==level,]$removeOnlyZeroVariacePredictors) 
  performVarianceAnalysisOnTrainSetOnly = as.logical(model.boosting[model.boosting$lev==level,]$performVarianceAnalysisOnTrainSetOnly) 
  correlationRhreshold = as.numeric(model.boosting[model.boosting$lev==level,]$correlationRhreshold) 
  res.train.mean = as.numeric(model.boosting[model.boosting$lev==level,]$res.train.mean) 
  res.train.sd = as.numeric(model.boosting[model.boosting$lev==level,]$res.train.sd) 
  RMSE.train.kfold = as.numeric(model.boosting[model.boosting$lev==level,]$RMSE.train.kfold) 
  RMSE.xval.winner = as.numeric(model.boosting[model.boosting$lev==level,]$RMSE.xval.winner) 
  
  variants = length(models)
  
  ## residuals 
  res = res.vector[,level]
  
  ## candidates
  model.candidates = matrix(NA,1,variants)
  res.candidates = matrix(NA,length(res),variants)
  pred.candidates = matrix(NA,length(test.raw$Id),variants)
  RMSE.xval.candidates = matrix(NA,1,variants)
  
  ## finding best models fitting level
  if (sum(need.finding) > 0) {
    cat(">> FINDING best model on ",variants,"variants of level ",level,"... \n")
    for (var in 1:variants) {
      
      ## unrolling paramenters ... 
      removeOnlyZeroVariacePredictors.var =  removeOnlyZeroVariacePredictors[var]
      performVarianceAnalysisOnTrainSetOnly.var = performVarianceAnalysisOnTrainSetOnly[var] 
      correlationRhreshold.var = correlationRhreshold[var]
      
      cat("> processing variant N.",var,"...\n") 
      cat("> removeOnlyZeroVariacePredictors:",removeOnlyZeroVariacePredictors.var,"\n")
      cat("> performVarianceAnalysisOnTrainSetOnly:",performVarianceAnalysisOnTrainSetOnly.var,"\n")
      cat("> correlationRhreshold:",correlationRhreshold.var,"\n")
       
      ## processing data 
      l = featureSelect (train,test,y=y,
                         removeOnlyZeroVariacePredictors = removeOnlyZeroVariacePredictors.var,
                         performVarianceAnalysisOnTrainSetOnly = performVarianceAnalysisOnTrainSetOnly.var,
                         correlationRhreshold = correlationRhreshold.var
      )
      traindata = l[[1]]
      testdata = l[[2]]
      
      ## finding the best model for fitting residuals  
      l = trainAndPredict.kfold.reg (k = 6,traindata, res ,RegModels,controlObject)
      model.winner = l[[1]]
      .grid = l[[2]]
      perf.kfold = l[[3]]
      
      best.RMSE = .grid$best.perf
      
      ### results 
      if (verbose) {
        cat("****** RMSE - each model/fold ****** \n")
        print(perf.kfold)
        cat("\n****** RMSE - mean ****** \n")
        print(.grid)
        cat("\n>>>>>>>>>>>> The winner is ... ",model.winner,"\n")
      }
      
      ### making prediction on test set 
      pred.res.test = 
        reg.trainAndPredict( res , 
                             traindata , 
                             testdata , 
                             model.winner , 
                             controlObject, 
                             best.tuning = T)
      
      ### estimating residuals 
      cat(">>> estimating residuals on training set .. \n")
      pred.res.train = predict.train.k.folds (traindata , 
                                              res , 
                                              model.label = model.winner ,
                                              controlObject , 
                                              k = 6 )
       
      cat(">>> prediction of residuals in train set - mean =",mean(pred.res.train)," sd =",sd(pred.res.train)," ... \n")
      cat(">>>               residuals in train set - mean =",mean(res)," sd =",sd(res)," ... \n")
      
      
      ## updating 
      model.candidates[var] = model.winner
      res.candidates[,var] = res - pred.res.train
      pred.candidates[,var] = pred.res.test
      RMSE.xval.candidates[var] = best.RMSE
      
      ##
      model.boosting[model.boosting$lev==level &
                       model.boosting$removeOnlyZeroVariacePredictors == removeOnlyZeroVariacePredictors.var &
                       model.boosting$performVarianceAnalysisOnTrainSetOnly == performVarianceAnalysisOnTrainSetOnly.var & 
                        (is.na(model.boosting$correlationRhreshold) == is.na(correlationRhreshold.var))    ,]$model = model.winner
      
      model.boosting[model.boosting$lev==level &
                       model.boosting$removeOnlyZeroVariacePredictors == removeOnlyZeroVariacePredictors.var &
                       model.boosting$performVarianceAnalysisOnTrainSetOnly == performVarianceAnalysisOnTrainSetOnly.var & 
                       (is.na(model.boosting$correlationRhreshold) == is.na(correlationRhreshold.var))    ,]$res.train.mean = mean(res - pred.res.train)
      
      model.boosting[model.boosting$lev==level &
                       model.boosting$removeOnlyZeroVariacePredictors == removeOnlyZeroVariacePredictors.var &
                       model.boosting$performVarianceAnalysisOnTrainSetOnly == performVarianceAnalysisOnTrainSetOnly.var & 
                       (is.na(model.boosting$correlationRhreshold) == is.na(correlationRhreshold.var))    ,]$res.train.sd = sd(res - pred.res.train)
      
      model.boosting[model.boosting$lev==level &
                       model.boosting$removeOnlyZeroVariacePredictors == removeOnlyZeroVariacePredictors.var &
                       model.boosting$performVarianceAnalysisOnTrainSetOnly == performVarianceAnalysisOnTrainSetOnly.var & 
                       (is.na(model.boosting$correlationRhreshold) == is.na(correlationRhreshold.var))    ,]$RMSE.train.kfold = RMSE(pred = pred.res.train, obs = res)
      
      model.boosting[model.boosting$lev==level &
                       model.boosting$removeOnlyZeroVariacePredictors == removeOnlyZeroVariacePredictors.var &
                       model.boosting$performVarianceAnalysisOnTrainSetOnly == performVarianceAnalysisOnTrainSetOnly.var & 
                       (is.na(model.boosting$correlationRhreshold) == is.na(correlationRhreshold.var))    ,]$RMSE.xval.winner = best.RMSE
      
      model.boosting[model.boosting$lev==level &
                       model.boosting$removeOnlyZeroVariacePredictors == removeOnlyZeroVariacePredictors.var &
                       model.boosting$performVarianceAnalysisOnTrainSetOnly == performVarianceAnalysisOnTrainSetOnly.var & 
                       (is.na(model.boosting$correlationRhreshold) == is.na(correlationRhreshold.var))    ,]$need.finding = F
    }
    
    ### comparing performances of variants to choose the best performant one
    cat(">> Comparing performances of variants to choose the best performant one ... \n")
    print(model.candidates)
    print(RMSE.xval.candidates)
    
    idx.best = which(RMSE.xval.candidates == min(RMSE.xval.candidates) ) 
    model.best = model.candidates[idx.best]
    res.best = res.candidates[,idx.best]
    pred.best = pred.candidates[,idx.best]
    RMSE.xval.best = RMSE.xval.candidates[idx.best]
    
    cat(">> Winner:",model.best,"[",idx.best,"] \n")
    
    ###
    pred.vector[,level+1] = pred.best
    res.vector[,level+1] = res.best
    
  } else {
    cat(">> NO NEED to finding best models for ",variants,"variants of level ",level,"... \n" )
  }
  
  cat("******** END OF LEVEL ",level,"********\n")
}

##### PREDICTING 
.pred.vector = pred.vector 
.pred.vector$pred.final = apply(.pred.vector , 1 , function(x) {
  res = sum(x[2:(length(x))])
})

pred = .pred.vector$pred.final

## some comparisons ... 
cat("min  y train = ",min(y) , " vs. min pred = ", min(pred)  , " \n")
cat("max  y train = ",max(y) , " vs. max pred = ", max(pred)  , " \n")
cat("mean y train = ",mean(y), " vs. mean pred = ",mean(pred) ,  "\n")

##### STORING 
cat(">>> Storing on disk ... \n")

write.csv(model.boosting,
          quote=FALSE, 
          file=paste(getBasePath("data"),"model_boosting_",INIT_LEV,"_",END_LEV,"_grid.csv",sep='') ,
          row.names=FALSE)

write.csv(pred.vector,
          quote=FALSE, 
          file=paste(getBasePath("data"),"model_boosting_",INIT_LEV,"_",END_LEV,"_pred.test.csv",sep='') ,
          row.names=FALSE)

write.csv(res.vector,
          quote=FALSE, 
          file=paste(getBasePath("data"),"model_boosting_",INIT_LEV,"_",END_LEV,"_res.csv",sep='') ,
          row.names=FALSE)

write.csv(data.frame(Id = test.raw$Id , Prediction = pred),
          quote=FALSE, 
          file=paste(getBasePath("data"),"model_boosting_",INIT_LEV,"_",END_LEV,"_______pred_final.csv",sep='') ,
          row.names=FALSE)
