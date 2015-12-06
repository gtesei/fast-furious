library(data.table)
library(xgboost)
library(methods)
library(fastfurious)

### CONFIG 
DEBUG = FALSE
CUTOFF = 9000

#### PROCS
RMSLE.xgb = function (preds, dtrain,th_err=1.5) {
  obs <- xgboost::getinfo(dtrain, "label")
  if ( sum(preds<0) >0 ) {
    preds = ifelse(preds >=0 , preds , th_err)
  }
  rmsle = sqrt(    sum( (log(preds+1) - log(obs+1))^2 )   /length(preds))
  return(list(metric = "RMSLE", value = rmsle))
}

getData = function() {
  ## Xtrain / Xtest / Ytrain 
  Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_NAs6.csv" , sep='') , stringsAsFactors = F))
  Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_NAs6.csv" , sep='') , stringsAsFactors = F))
  Ytrain = as.data.frame( fread(paste(ff.getPath("elab") , "Ytrain_NAs6.csv" , sep='') , stringsAsFactors = F))
  Ytrain = Ytrain$Ytrain
  
  test_id = Xtest$REN_ID
  
  ## MEMO #1: remove REN_ID in train / test set before fitting models 
  predToDel = c("REN_ID")
  for (pp in predToDel) {
    cat(">>> removing ",pp,"...\n")
    Xtrain[,pp] <- NULL
    Xtest[,pp] <- NULL
  }
  
  cat(">>> loaded Ytrain:",length(Ytrain),"\n")
  cat(">>> loaded Xtrain:",dim(Xtrain),"\n")
  cat(">>> loaded Xtest:",dim(Xtest),"\n")
  cat(">>> loaded test_id:",length(test_id),"\n")
  
  stopifnot(sum(Xtrain==Inf)==0)
  stopifnot(sum(Xtest==Inf)==0)
  stopifnot(sum(is.na(Xtrain))==0)
  stopifnot(sum(is.na(Xtest))==0)
  
  return(list(
    Ytrain = Ytrain, 
    Xtrain = Xtrain, 
    Xtest = Xtest, 
    test_id = test_id
  ))
}

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/deloitte-western-australia-rental-prices/data')
ff.bindPath(type = 'code' , sub_path = 'competitions/deloitte-western-australia-rental-prices')
ff.bindPath(type = 'elab' , sub_path = 'dataset/deloitte-western-australia-rental-prices/elab' ,  createDir = T) 

ff.bindPath(type = 'ensemble_1' , sub_path = 'dataset/deloitte-western-australia-rental-prices/ensembles/ensemble_1',createDir = T) ## out 
ff.bindPath(type = 'best_tune_1' , sub_path = 'dataset/deloitte-western-australia-rental-prices/ensembles/best_tune_1',createDir = T) ## out 
ff.bindPath(type = 'submission_1' , sub_path = 'dataset/deloitte-western-australia-rental-prices/ensembles/pred_ensemble_1',createDir = T) ## out 

ff.bindPath(type = 'ensemble_2' , sub_path = 'dataset/deloitte-western-australia-rental-prices/ensembles/ensemble_2',createDir = T) ## out 
ff.bindPath(type = 'best_tune_2' , sub_path = 'dataset/deloitte-western-australia-rental-prices/ensembles/best_tune_2',createDir = T) ## out 
ff.bindPath(type = 'submission_2' , sub_path = 'dataset/deloitte-western-australia-rental-prices/ensembles/pred_ensemble_2',createDir = T) ## out 

######

## data 
data = getData()
Ytrain = data$Ytrain
Xtrain = data$Xtrain
Xtest = data$Xtest 
test_id = data$test_id 
rm(data)
gc()
if (DEBUG) {
  cat("> debug .. \n")
  Xtrain = Xtrain[1:100,]
  #Xtest = Xtrain[,1:10]
  Ytrain = Ytrain[1:100]
  gc()
}

#### cutting outliers 
cat("There ",sum(Ytrain>CUTOFF),"observations with rent > ",CUTOFF," --> removing them ... \n")
Ytrain = Ytrain[Ytrain<=CUTOFF]
Xtrain = Xtrain[which(Ytrain<=CUTOFF),]

#### removing bad predictors 
l = ff.featureFilter (Xtrain,
                      Xtest,
                      removeOnlyZeroVariacePredictors=TRUE,
                      performVarianceAnalysisOnTrainSetOnly = TRUE , 
                      removePredictorsMakingIllConditionedSquareMatrix = FALSE, 
                      removeHighCorrelatedPredictors = FALSE, 
                      featureScaling = FALSE)
Xtrain = l$traindata
Xtest = l$testdata  

cat(">>> applying log transf. to Y ... \n")
Ytrain <- log(Ytrain)

## nrounds same of model without outliers removal 
nrounds = 20998

###### XBG 
cat(">>> XBG training [nrounds=",nrounds,"]...\n")
data = rbind(Xtrain,Xtest)
x = as.matrix(data)
x = matrix(as.numeric(x),nrow(x),ncol(x))

trind = 1:nrow(Xtrain)
teind = (nrow(Xtrain)+1):nrow(x)

rm(Xtrain)
rm(Xtest)
rm(data)
### 
param <- list("objective" = "reg:linear" ,
              "min_child_weight" = 6 , 
              "subsample" = 0.7 , 
              "colsample_bytree" = 0.6 , 
              "scale_pos_weight" = 0.8 , 
              "silent" = 1 , 
              "max_depth" = 8 , 
              "max_delta_step" = 2 )

param['eta'] = 0.02
param['max_depth'] = 9

dtrain <- xgboost::xgb.DMatrix(x[trind,], label = Ytrain)
bst = xgboost::xgb.train(param = param,  
                         dtrain , 
                         nrounds = nrounds, 
                         feval = RMSLE.xgb , maximize = FALSE , verbose = FALSE)
cat(">>> XGB predicting ...\n")
pred = xgboost::predict(bst,x[teind,])

cat(">>> applying log-reverse transformation to predictions ... \n")
pred <- exp(pred)

## pred 
stopifnot(sum(is.na(pred))==0)
stopifnot(sum(pred==Inf)==0)
submission <- data.frame(REN_ID=test_id)
submission$REN_BASE_RENT <- pred
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "paul_436_removeOutliersITS_",CUTOFF,".csv" , sep='') ,
          row.names=FALSE)

