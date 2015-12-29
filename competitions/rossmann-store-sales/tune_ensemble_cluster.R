library(data.table)
library(xgboost)
library(fastfurious)

### FUNCS
RMSPE = function(pred, obs) {
  ignIdx = which(obs==0)
  if (length(ignIdx)>0) {
    obs = obs[-ignIdx]
    pred = pred[-ignIdx]
  }
  stopifnot(sum(obs==0)==0)
  rmspe = sqrt(mean((1-pred/obs)^2))
  return (rmspe)
}
RMSPE.exp = function(pred, obs) {
  ignIdx = which(obs==0)
  if (length(ignIdx)>0) {
    obs = obs[-ignIdx]
    pred = pred[-ignIdx]
  }
  stopifnot(sum(obs==0)==0)
  if (modelList[[m]]$ytranf=="log") {
    obs <- exp(as.numeric(obs))-1
    pred <- exp(as.numeric(pred))-1
  } else if ((substr(x = modelList[[m]]$ytranf, start = 1 , stop = nchar('power_')) == 'power_')) {
    obs <- obs^(1/pow)
    pred <- pred^(1/pow)
  } else {
    stop("unrecognized transf") 
  }
  rmspe = sqrt(mean((1-pred/obs)^2))
  return (rmspe)
}

RMSPECostSummary <- function (data, lev = NULL, model = NULL) {
  c(postResample(data[, "pred"], data[, "obs"]),
    RMSPE = RMSPE(pred = data[, "pred"], obs = data[, "obs"]))
}
RMSPECostSummary.exp <- function (data, lev = NULL, model = NULL) {
  c(postResample(data[, "pred"], data[, "obs"]),
    RMSPE = RMSPE.exp(pred = data[, "pred"], obs = data[, "obs"]))
}

RMSPE.xgb <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- sqrt(mean((1-preds/labels)^2))
  return(list(metric = "RMSPE", value = err))
}
RMSPE.xgb.exp <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  
  ignIdx = which(labels==0)
  if (length(ignIdx)>0) {
    labels = obs[-ignIdx]
    preds = preds[-ignIdx]
  }
  stopifnot(sum(labels==0)==0)
  if (modelList[[m]]$ytranf=="log") {
    labels <- exp(as.numeric(labels))-1
    preds <- exp(as.numeric(preds))-1
  } else if ((substr(x = modelList[[m]]$ytranf, start = 1 , stop = nchar('power_')) == 'power_')) {
    labels <- labels^(1/pow)
    preds <- preds^(1/pow)
  } else {
    stop("unrecognized transf") 
  }
  
  err <- sqrt(mean((1-preds/labels)^2))
  return(list(metric = "RMSPE", value = err))
}

buildIDModelList = function(list) {
  stopifnot(length(list)>0)
  for (i in 1:length(list)) {
    fn = ""
    if (DEBUG) {
      fn = "DEBUG_cluster_" 
    } else {
      fn = "cluster_" 
    }
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
  Ytrain = NULL
  Xtrain = NULL
  Xtest = NULL
  test_id = NULL
  
  dataProc = mod$dataProc
  if (identical("base",dataProc)) {
    ## Xtrain / Xtest / Ytrain 
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain.csv" , sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest.csv" , sep='') , stringsAsFactors = F))
    Ytrain = Xtrain$Sales
    test_id = Xtest$Id
    
    Xtrain$Sales <- NULL
    Xtest$Id <- NULL
  } else if (identical("bech",dataProc)) {
    ## Xtrain / Xtest / Ytrain 
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_bench.csv" , sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_bench.csv" , sep='') , stringsAsFactors = F))
    Ytrain = Xtrain$Sales
    test_id = Xtest$Id
    
    Xtrain$Sales <- NULL
    Xtest$Id <- NULL
  } else {
    stop(paste0("unrecognized type of dataProc:",dataProc))
  }
  
  cat(">>> loaded Ytrain:",length(Ytrain),"\n")
  cat(">>> loaded Xtrain:",dim(Xtrain),"\n")
  cat(">>> loaded Xtest:",dim(Xtest),"\n")
  cat(">>> loaded test_id:",length(test_id),"\n")
  
  return(list(
    Ytrain = Ytrain, 
    Xtrain = Xtrain, 
    Xtest = Xtest, 
    test_id = test_id
  ))
}

### CONFIG 
DEBUG = F
nFolds = 5
nrepeats =  1

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/rossmann-store-sales')
ff.bindPath(type = 'code' , sub_path = 'competitions/rossmann-store-sales')
ff.bindPath(type = 'elab' , sub_path = 'dataset/rossmann-store-sales/elab') 

ff.bindPath(type = 'ensemble_1' , sub_path = 'dataset/rossmann-store-sales/ensembles/ensemble_1',createDir = T) ## out 
ff.bindPath(type = 'best_tune_1' , sub_path = 'dataset/rossmann-store-sales/ensembles/best_tune_1',createDir = T) ## out 
ff.bindPath(type = 'submission_1' , sub_path = 'dataset/rossmann-store-sales/ensembles/pred_ensemble_1',createDir = T) ## out 
##

### DATA 
Xtest_Open_0 = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_Open_0.csv" , sep='') , stringsAsFactors = F))


################# MODELS 

modelList = list(
  ##############################################################################
  #                                    1 LAYER                                 #
  ##############################################################################
  
  #list(layer = 1  , dataProc = "base", mod = 'xgbTreeGTJ'  , eta=0.02 , max_depth = 6, tune=T)
  #list(layer = 1  , dataProc = "base", mod = 'glm'  , tune=F)
  #list(layer = 1  , dataProc = "base", mod = 'cubist'  , tune=T)
  #list(layer = 1  , dataProc = "base", ytranf = "power_0.01", mod = 'glm'  , tune=F)
  
  
  
  #list(layer = 1  , dataProc = "bech", ytranf = "log", mod = 'glm'  , tune=F)
  list(layer = 1  , dataProc = "bech", ytranf = "log", mod = 'cubist'  , tune=T)
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
  
  ## 
  stores = sort(unique(c(Xtrain$Store,Xtest$Store)))
  pred = rep(NA,nrow(Xtest))
  if (modelList[[m]]$tune) tuneGrid = data.frame(model=rep(modelList[[m]]$mod,length(stores)),Store=stores,secs=NA,RMSPE=NA)
  
  ## CLUSTERING & TUNING 
  for (st in seq_along(stores)) {
    cat(">>> [TUNE]: processing Store [",stores[st],"] [",st,"/",length(stores),"] .. \n" )
    
    no_test_obs = F 
    if ( sum(Xtest$Store == st)==0) {
      if (!modelList[[m]]$tune) {
        cat(">>> no observations on test set about such store & this is a non-parametric model --> skipping ... \n")
        next
      } else {
        no_test_obs = T
      }
    }
    
    trIdx = which(Xtrain$Store==st)
    teIdx = which(Xtest$Store==st)
    
    Xtrain_i = Xtrain[trIdx,]
    if (no_test_obs) {
      Xtest_i = Xtest[1:10,]
    } else {
      Xtest_i = Xtest[teIdx,]
    }
    Ytrain_i = Ytrain[trIdx]
    
    ## check for transformations 
    if (!is.null(modelList[[m]]$ytranf)) {
      cat(">>> applying ",modelList[[m]]$ytranf," transf. to Ytrain_i ... \n")
      if (modelList[[m]]$ytranf=="log") {
        Ytrain_i <- log(Ytrain_i+1)  
      } else if ((substr(x = modelList[[m]]$ytranf, start = 1 , stop = nchar('power_')) == 'power_')) {
        pow = as.numeric(substr(x = modelList[[m]]$ytranf, start = (nchar('power_')+1) , stop = nchar(modelList[[m]]$ytranf)))
        Ytrain_i <- Ytrain_i^pow
      } else {
        stop("unrecognized transf") 
      }
    }
    
    ### encoding datasets for not tree based models 
    if (modelList[[m]]$mod != "xgbTreeGTJ" && modelList[[m]]$dataProc != 'bech')  {
      cat(">>> encoding datasets for not tree based models ... \n")
      l <- ff.makeFeatureSet(data.train = Xtrain_i , data.test = Xtest_i , 
                             meta = c("C","C","N","N","C","C","C","C","N","N","N"), 
                             scaleNumericFeatures = F , 
                             remove1DummyVarInCatPreds = F , 
                             parallelize = T)
      Xtrain_i <- l$traindata
      Xtest_i <- l$testdata
      rm(l)
      
    }
    
    ##############
    ## TUNE 
    ##############
    ctr = trainControl(method = "repeatedcv", repeats = nrepeats, number = nFolds) 
    if (is.null(modelList[[m]]$ytranf)) {
      ctr$summaryFunction <- RMSPECostSummary
    } else {
      ctr$summaryFunction <- RMSPECostSummary.exp
    }
    
    if (modelList[[m]]$mod == "xgbTreeGTJ") {
      l = ff.trainAndPredict.reg ( Ytrain=Ytrain_i ,
                                   Xtrain=Xtrain_i , 
                                   Xtest=Xtest_i , 
                                   model.label=modelList[[m]]$mod , 
                                   controlObject=ctr, 
                                   best.tuning = TRUE, 
                                   verbose = TRUE, 
                                   removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = TRUE, 
                                   xgb.metric.fun = RMSPE.xgb, 
                                   xgb.maximize = FALSE, 
                                   xgb.metric.label = 'RMSPE', 
                                   xgb.foldList = NULL,
                                   xgb.eta = modelList[[m]]$eta, 
                                   xgb.max_depth = modelList[[m]]$max_depth, 
                                   xgb.cv.default = F
                                   #,nrounds = 3000
      )
    } else {
      l = ff.trainAndPredict.reg ( Ytrain=Ytrain_i ,
                                   Xtrain=Xtrain_i , 
                                   Xtest=Xtest_i , 
                                   model.label=modelList[[m]]$mod , 
                                   controlObject=ctr, 
                                   best.tuning = TRUE, 
                                   verbose = TRUE, 
                                   removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = TRUE, 
                                   xgb.metric.fun = RMSPE.xgb, 
                                   xgb.maximize = FALSE, 
                                   xgb.metric.label = 'RMSPE', 
                                   xgb.foldList = NULL,
                                   xgb.eta = modelList[[m]]$eta, 
                                   xgb.max_depth = modelList[[m]]$max_depth, 
                                   xgb.cv.default = F
                                   , metric = "RMSPE", maximize=F
      )
    }
    
    
    if ( !is.null(l$model) ) {
      pred_i = l$pred
      if (!is.null(modelList[[m]]$ytranf)) {
        cat(">>> applying inverse tranf. of ",modelList[[m]]$ytranf," to predictions ... \n")
        if (modelList[[m]]$ytranf=="log") {
          pred_i <- exp(pred_i)-1  
        } else if (substr(x = modelList[[m]]$ytranf, start = 1 , stop = nchar('power_')) == 'power_') {
          pow = as.numeric(substr(x = modelList[[m]]$ytranf, start = (nchar('power_')+1) , stop = nchar(modelList[[m]]$ytranf)))
          pred_i <- pred_i^(1/pow)
        } else {
          stop("unrecognized transf") 
        }
      }
      stopifnot(sum(is.na(pred_i))==0,sum(pred_i==Inf)==0)
      perf_mod_i = min(l$model$results$RMSPE)
      if (modelList[[m]]$tune) bestTune_i = l$model$bestTune
      secs_i = l$secs 
      rm(l)
    } else {
      stop(paste('model',modelList[[m]]$mod,':error!'))
    }
    
    ## update tune grid 
    if (modelList[[m]]$tune) {
      tuneGrid[tuneGrid$Store==st,]$secs <- secs_i
      tuneGrid[tuneGrid$Store==st,]$RMSPE <- perf_mod_i
      if (ncol(tuneGrid)==4) {
        tt = as.data.frame(matrix(rep(NA,(ncol(bestTune_i)*nrow(tuneGrid))),ncol=ncol(bestTune_i),nrow=nrow(tuneGrid)))
        colnames(tt) = colnames(bestTune_i)
        tuneGrid = cbind(tuneGrid,tt)
      }
      tuneGrid[tuneGrid$Store==st,5:ncol(tuneGrid)] <- bestTune_i
    }
    
    ## update pred 
    if (!no_test_obs) {
      pred[teIdx] <- pred_i  
    }
  }
  
  ## checks 
  stopifnot(sum(is.na(pred))==0)
  stopifnot(sum(pred==Inf)==0)
  
  ## write best tune on disk 
  if (modelList[[m]]$tune) {
    stopifnot(sum(is.na(tuneGrid))==0)
    write.csv(tuneGrid,
              quote=FALSE, 
              file=paste0(ff.getPath(paste0("best_tune_",modelList[[m]]$layer)),modelList[[m]]$id) ,
              row.names=FALSE)
  }
  
  ## write prediction on disk 
  submission <- data.frame(Id=test_id)
  submission$Sales <- pred
  if (modelList[[m]]$dataProc != 'bech') submission = rbind(submission,Xtest_Open_0) ## attach Sales-0 predictions 
  stopifnot(nrow(submission)==41088)
  print(head(submission))
  write.csv(submission,
            quote=FALSE, 
            file=paste0(ff.getPath(paste0("submission_",modelList[[m]]$layer)),modelList[[m]]$id) ,
            row.names=FALSE)
  
  ##############
  ## ENSEMB 
  ##############
  cat(">>> Ensembling ... \n") 
  bestTune = NULL
  if (modelList[[m]]$tune) {
    bestTune = as.data.frame( fread(paste0(ff.getPath(paste0("best_tune_",modelList[[m]]$layer)),modelList[[m]]$id)))
    stopifnot( nrow(bestTune)>0 , !is.null(bestTune) )
  }
  submission = as.data.frame( fread(paste0(ff.getPath(paste0("submission_",modelList[[m]]$layer)),modelList[[m]]$id)))
  stopifnot( nrow(submission)>0 , !is.null(submission) )
  
  ##
  ff.setMaxCuncurrentThreads(4)
  predTrain = rep(NA,nrow(Xtrain))
  predTest = pred 
  
  ## CLUSTERING & ENSEMBLING 
  for (st in seq_along(stores)) {
    cat(">>> [ENSEMB]: processing Store [",stores[st],"] [",st,"/",length(stores),"] .. \n" )
    
    no_test_obs = F 
    if ( sum(Xtest$Store == st)==0) {
        no_test_obs = T
    }
    
    trIdx = which(Xtrain$Store==st)
    teIdx = which(Xtest$Store==st)
    
    Xtrain_i = Xtrain[trIdx,]
    if (no_test_obs) {
      Xtest_i = Xtest[1:10,]  
    } else {
      Xtest_i = Xtest[teIdx,]  
    }
    Ytrain_i = Ytrain[trIdx]
    Ytest_i = if (!no_test_obs) submission$Sales[teIdx] else submission$Sales[1:10]
    ## check for transformations 
    if (!is.null(modelList[[m]]$ytranf)) {
      cat(">>> applying ",modelList[[m]]$ytranf," transf. to Ytrain_i ... \n")
      Ytrain_i <- log(Ytrain_i+1)
    }
    
    ## resampling 
    index = caret::createMultiFolds(y=Ytrain_i, nFolds, 1)
    indexOut <- lapply(index, function(training, allSamples) allSamples[-unique(training)], allSamples = seq(along = Ytrain_i))
    controlObject = trainControl(method = "cv", 
                                 ## The method doesn't really matter
                                 ## since we defined the resamples
                                 index = index, 
                                 indexOut = indexOut)
    if (is.null(modelList[[m]]$ytranf)) {  
      controlObject$summaryFunction <- RMSPECostSummary
    } else {
      controlObject$summaryFunction <- RMSPECostSummary.exp
    }
    rm(list = c("index","indexOut"))
    
    ## createEnsemble
    ens = NULL 
    if (modelList[[m]]$mod == "xgbTreeGTJ") {
      ens = ff.createEnsemble (Xtrain = Xtrain_i,
                               Xtest = Xtest_i,
                               y =Ytrain_i,
                               caretModelName = 'xgbTree', 
                               predTest <- Ytest_i,
                               bestTune = expand.grid(
                                 nrounds = bestTune[bestTune$Store==st,]$early.stop ,
                                 max_depth = if (!is.null(modelList[[m]]$max_depth)) modelList[[m]]$max_depth else 8 ,  
                                 eta = modelList[[m]]$eta ),
                               removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = TRUE, 
                               controlObject = controlObject, 
                               parallelize = T,
                               verbose = T , 
                               regression = TRUE, 
                               
                               ### ... 
                               objective = "reg:linear",
                               #eval_metric = "auc", 
                               subsample = 0.7 , 
                               colsample_bytree = 0.6 , 
                               scale_pos_weight = 0.8 , 
                               max_delta_step = 2)
      
    } else if (modelList[[m]]$tune){
      ens = ff.createEnsemble (Xtrain = Xtrain_i,
                               Xtest = Xtest_i,
                               y = Ytrain_i,
                               caretModelName = modelList[[m]]$mod, 
                               predTest = Ytest_i,
                               bestTune = bestTune[bestTune$Store==st, 5:ncol(bestTune) , drop = F], 
                               removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = T, 
                               controlObject = controlObject, 
                               parallelize = T,
                               verbose = T , 
                               regression = T,
                               metric = "RMSPE" , maximize=F)
    } else {
      ens = ff.createEnsemble (Xtrain = Xtrain_i,
                               Xtest = Xtest_i,
                               y = Ytrain_i,
                               caretModelName = modelList[[m]]$mod, 
                               predTest = Ytest_i,
                               bestTune = NULL, 
                               removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = T, 
                               controlObject = controlObject, 
                               parallelize = T,
                               verbose = T , 
                               regression = T,
                               metric = "RMSPE", maximize=F)
    }
    
    ## update ensembles  
    predTrain[trIdx] = ens$predTrain
    if (!no_test_obs) {
      predTest[teIdx] = ens$predTest  
    }
  }
  
  ## checks 
  stopifnot(sum(is.na(predTrain))==0)
  rmse_pred_test = sqrt(mean((pred-predTest)^2))
  if (rmse_pred_test>0.1) stop("rmse_pred_test>0.1")
  
  ## assemble 
  if (!is.null(modelList[[m]]$ytranf)) {
    cat(">>> applying inverse tranf. of ",modelList[[m]]$ytranf," to ensembling predictions ... \n")
    if (modelList[[m]]$ytranf=="log") {
      predTrain <- exp(predTrain)-1
      #predTest <- exp(predTest)-1
    } else if ((substr(x = modelList[[m]]$ytranf, start = 1 , stop = nchar('power_')) == 'power_')) {
      predTrain <- predTrain^(1/pow)
      #predTest <- predTest^(1/pow)
    } else {
      stop("unrecognized transf") 
    }
  }
  assemble = c(predTrain,submission$Sales)
  stopifnot(sum(is.na(assemble))==0)
  stopifnot(sum(assemble==Inf)==0)
  write.csv(data.frame(id = seq_along(assemble) , assemble=assemble),
            quote=FALSE, 
            file=paste0(ff.getPath(paste0("ensemble_",modelList[[m]]$layer)),modelList[[m]]$id) ,
            row.names=FALSE)
}
####### end of loop 
tm = proc.time() - ptm
secs = as.numeric(tm[3])
cat(">>> MAIN LOOP >>> Time elapsed:",secs," secs. [",secs/60,"min.] [",secs/(60*60),"hours] \n")
############## end of MAIN LOOP     

    
    