library(data.table)
library(xgboost)
library(fastfurious)

### FUNCS

RMSPE.xgb <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  
  ##
  labels<-exp(as.numeric(labels))-1
  preds<-exp(as.numeric(preds))-1
  
  ##
  ignIdx = which(labels==0)
  if (length(ignIdx)>0) {
    labels = labels[-ignIdx]
    preds = preds[-ignIdx]
  }
  stopifnot(sum(labels==0)==0)
  err <- sqrt(  mean(  ((rep(1,length(preds))-preds/labels)^2)   ) )
  return(list(metric = "RMSPE", value = err))
}

RMSPE <- function(preds, labels) {
  ignIdx = which(labels==0)
  if (length(ignIdx)>0) {
    labels = labels[-ignIdx]
    preds = preds[-ignIdx]
  }
  stopifnot(sum(labels==0)==0)
  err <- sqrt(  mean(  ((1-preds/labels)^2)   ) )
  return(err)
}

buildIDModelList = function(list) {
  stopifnot(length(list)>0)
  for (i in 1:length(list)) {
    fn = ""
    if (DEBUG) fn = "DEBUG_"
    for (j in 1:length(list[[i]])) {
      fn = paste(fn,names(list[[i]][j]),list[[i]][j],sep="")
      if (j < length(list[[i]])) {
        fn = paste(fn,"_",sep="")
      }
    }
    fn = paste0(fn,".csv")
    list[[i]]$id = fn 
  }
  return(list)
}

getData = function(mod) {
  dataProc = mod$dataProc
  if (identical("debug",dataProc)) {
    ## Xtrain / Xtest / Ytrain 
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain.csv" , sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest.csv" , sep='') , stringsAsFactors = F))
    Ytrain = Xtrain$Sales
    
    Xval = Xtrain[1:100,]
    Yval = Ytrain[1:100]
    Xtrain = Xtrain[101:300,]
    Ytrain = Ytrain[101:300]
    
    train_id = 101:300
    xval_id = 1:100
    test_id = Xtest$Id
    
    Xtrain$Sales <- NULL
    Xval$Sales <- NULL
    
    Xtrain$Id <- NULL
    Xval$Id <- NULL
    Xtest$Id <- NULL
    
  } else if (identical("data1",dataProc)) {
    ## Xtrain / Xtest / Ytrain 
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain3_tr.csv" , sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest3.csv" , sep='') , stringsAsFactors = F))
    Xval = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain3_xval.csv" , sep='') , stringsAsFactors = F))
    
    Ytrain = Xtrain$Sales
    Yval <- Xval$Sales
    
    train_id = Xtrain$Id
    xval_id = Xval$Id
    test_id = Xtest$Id
    
    Xtrain$Sales <- NULL
    Xval$Sales <- NULL
    
    Xtrain$Id <- NULL
    Xval$Id <- NULL
    Xtest$Id <- NULL
    
  }
  cat(">>> loaded Ytrain:",length(Ytrain),"\n")
  cat(">>> loaded Yval:",length(Yval),"\n")
  cat(">>> loaded Xtrain:",dim(Xtrain),"\n")
  cat(">>> loaded Xval:",dim(Xval),"\n")
  cat(">>> loaded Xtest:",dim(Xtest),"\n")
  cat(">>> loaded train_id:",length(train_id),"\n")
  cat(">>> loaded xval_id:",length(xval_id),"\n")
  cat(">>> loaded test_id:",length(test_id),"\n")
  
  stopifnot(sum(Xtrain==Inf)==0)
  stopifnot(sum(Xval==Inf)==0)
  stopifnot(ncol(Xtrain)==ncol(Xval))
  stopifnot(sum(Xtest==Inf)==0)
  stopifnot(sum(is.na(Xtrain))==0)
  stopifnot(sum(is.na(Xtest))==0)
  
  return(list(
    Ytrain = Ytrain, 
    Yval = Yval, 
    Xtrain = Xtrain, 
    Xval = Xval, 
    Xtest = Xtest, 
    train_id = train_id, 
    xval_id = xval_id, 
    test_id = test_id
  ))
}

### CONFIG 
DEBUG = F
nFolds = 4
nrepeats =  1

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/rossmann-store-sales')
ff.bindPath(type = 'code' , sub_path = 'competitions/rossmann-store-sales')
ff.bindPath(type = 'elab' , sub_path = 'dataset/rossmann-store-sales/elab') 

ff.bindPath(type = 'ensemble_1' , sub_path = 'dataset/rossmann-store-sales/ensembles/ensemble_1',createDir = T) ## out 
ff.bindPath(type = 'best_tune_1' , sub_path = 'dataset/rossmann-store-sales/ensembles/best_tune_1',createDir = T) ## out 
ff.bindPath(type = 'submission_1' , sub_path = 'dataset/rossmann-store-sales/ensembles/pred_ensemble_1',createDir = T) ## out 

ff.bindPath(type = 'ensemble_2' , sub_path = 'dataset/rossmann-store-sales/ensembles/ensemble_2',createDir = T) ## out 
ff.bindPath(type = 'best_tune_2' , sub_path = 'dataset/rossmann-store-sales/ensembles/best_tune_2',createDir = T) ## out 
ff.bindPath(type = 'submission_2' , sub_path = 'dataset/rossmann-store-sales/ensembles/pred_ensemble_2',createDir = T) ## out 
##


################# MODELS 

modelList = list(
  ##############################################################################
  #                                    1 LAYER                                 #
  ##############################################################################
  
  #list(layer = 1  , dataProc = "debug", ytranf="log", mod = 'xgbTreeGTJ'  , eta=0.01 , max_depth = 5, tune=T , MAX_NROUNDS = 100000)
  #list(layer = 1  , dataProc = "data1", ytranf="log", mod = 'xgbTreeGTJ'  , eta=0.01 , max_depth = 5, tune=T , MAX_NROUNDS = 100000)
  list(layer = 1  , dataProc = "data1", ytranf="log", mod = 'xgbTreeGTJ'  , eta=0.001 , max_depth = 10, tune=T , MAX_NROUNDS = 100000)
)

modelList = buildIDModelList(modelList)

##############
## MAIN LOOP 
##############
ptm <- proc.time()
for (m in  seq_along(modelList) ) { 
  cat(">>> now processing:\n")
  print(modelList[[m]])
  
  ## data 
  data = getData(modelList[[m]])
  Ytrain = data$Ytrain
  Yval = data$Yval
  Xtrain = data$Xtrain
  Xval = data$Xval
  Xtest = data$Xtest 
  train_id = data$train_id
  xval_id = data$xval_id
  test_id = data$test_id 
  rm(data)
  gc()
  
  ## check for transformations 
  if (!is.null(modelList[[m]]$ytranf)) {
    cat(">>> applying ",modelList[[m]]$ytranf," transf. to Y ... \n")
    Ytrain <- log(Ytrain+1)
    Yval <- log(Yval+1)
  }
  
  ###### >>>> XGB 
  if (modelList[[m]]$mod !="xgbTreeGTJ") stop("currently supported only XBG at first layer")
  
  #### data precessing 
  n_train = nrow(Xtrain)
  n_val = nrow(Xval)
  n_test = nrow(Xtest)
  
  ttrain = rbind(Xtrain,Xval)
  ### removing bad predictors 
  l = ff.featureFilter (ttrain,
                        Xtest,
                        removeOnlyZeroVariacePredictors=TRUE,
                        performVarianceAnalysisOnTrainSetOnly = TRUE , 
                        removePredictorsMakingIllConditionedSquareMatrix = FALSE, 
                        removeHighCorrelatedPredictors = FALSE, 
                        featureScaling = FALSE)
  ttrain <- l$traindata
  Xtrain <- l$traindata[1:n_train,]
  Xval <- l$traindata[(n_train+1):(n_train+n_val),]
  Xtest <- l$testdata
  
  ###### XBG 
  cat(">>> XBG training [MAX_NROUNDS=",modelList[[m]]$MAX_NROUNDS,"]...\n")
  data = rbind(Xtrain,Xval,Xtest)
  x = as.matrix(data)
  x = matrix(as.numeric(x),nrow(x),ncol(x))
  
  trind = 1:n_train
  tval = (n_train+1):(n_train+n_val)
  teind = (n_train+n_val+1):nrow(x)
  
  rm(Xtrain)
  rm(Xval)
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
  
  stopifnot(!is.null(modelList[[m]]$eta))
  stopifnot(!is.null(modelList[[m]]$max_depth))
  param['eta'] = modelList[[m]]$eta
  param['max_depth'] = modelList[[m]]$max_depth
  
  ##
  dtrain <- xgboost::xgb.DMatrix(x[trind,], label = Ytrain)
  dval <- xgboost::xgb.DMatrix(x[tval,], label = Yval)
  watchlist<-list(val=dval,train=dtrain)
  
  ##
  bst = xgboost::xgb.train(param = param,  
                           data  = dtrain , 
                           watchlist = watchlist,
                           early.stop.round = 100,
                           nrounds = modelList[[m]]$MAX_NROUNDS, 
                           feval = RMSPE.xgb , 
                           maximize = FALSE , 
                           verbose = FALSE)
  
  nrounds <- bst$bestInd
  ## write best tune on disk
  if (modelList[[m]]$tune) {
    cat(">>> saving best tune [RMSPE:",bst$bestScore,"]... \n")
    tuneGrid = data.frame(model=modelList[[m]]$mod,RMSPE=bst$bestScore) 
    tuneGrid = cbind(tuneGrid,data.frame(early.stop=nrounds))
    write.csv(tuneGrid,
              quote=FALSE, 
              file=paste0(ff.getPath(paste0("best_tune_",modelList[[m]]$layer)),modelList[[m]]$id) ,
              row.names=FALSE)
  }
  
  ## predicting without re-feeding whole train set
  cat("\n>>> XGB predicting without re-feeding whole train set / ensembling ...\n")      
  pred_norfd = xgboost::predict(bst,x[teind,])
  enseb = xgboost::predict(bst,x[tval,])
  if (!is.null(modelList[[m]]$ytranf)) {
    cat(">>> applying reverse ",modelList[[m]]$ytranf," transf. to predictions ... \n")
    pred_norfd <- exp(pred_norfd)-1
    enseb <- exp(enseb)-1
  }
  
  xval_score <- RMSPE(preds = enseb , labels = exp(Yval)-1)
  cat(">>> xval_score",xval_score,"\n")
  
  ## predicting re-feeding whole train set
  cat('>>> XGB predicting re-feeding whole train set [nrounds:',nrounds,'] ...\n')      
  dtrain <- xgboost::xgb.DMatrix(x[1:(n_train+n_val),], label = c(Ytrain,Yval))
  bst = xgboost::xgb.train(param = param,  
                           dtrain , 
                           nrounds = nrounds, 
                           feval = RMSPE.xgb , maximize = FALSE , verbose = FALSE)
  pred = xgboost::predict(bst,x[teind,])
  if (!is.null(modelList[[m]]$ytranf)) {
    cat(">>> applying reverse ",modelList[[m]]$ytranf," transf. to predictions ... \n")
    pred <- exp(pred)-1
  }
  
  ## write predictions on disk 
  cat(">>> saving predictions refeed / no-refeed ... \n")
  stopifnot(sum(is.na(pred))==0,sum(pred==Inf)==0,length(pred)==n_test)
  stopifnot(sum(is.na(pred_norfd))==0,sum(pred_norfd==Inf)==0,length(pred_norfd)==n_test)
  submission <- data.frame(Id=test_id)
  
  ## with refeed  
  submission$Sales <- pred
  print(head(submission))
  write.csv(submission,
            quote=FALSE, 
            file=paste0(ff.getPath(paste0("submission_",modelList[[m]]$layer)),modelList[[m]]$id) ,
            row.names=FALSE)
  ## with refeed  
  submission$Sales <- pred_norfd
  print(head(submission))
  write.csv(submission,
            quote=FALSE, 
            file=paste0(ff.getPath(paste0("submission_",modelList[[m]]$layer)),"partial_",modelList[[m]]$id) ,
            row.names=FALSE)
  
  ## write ensemble on disk 
  cat(">>> saving ensemble ... \n")
  stopifnot(sum(is.na(enseb))==0)
  stopifnot(sum(enseb==Inf)==0)
  write.csv(data.frame(id = xval_id , enseb=enseb),
            quote=FALSE, 
            file=paste0(ff.getPath(paste0("ensemble_",modelList[[m]]$layer)),modelList[[m]]$id) ,
            row.names=FALSE)
}