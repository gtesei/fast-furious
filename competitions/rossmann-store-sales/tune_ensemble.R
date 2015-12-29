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
  
  if (!is.null(modelList[[m]]$ytranf)) {
    obs <- exp(as.numeric(obs))-1
    pred <- exp(as.numeric(pred))-1
  }
  rmspe = sqrt(mean( ((1-pred/obs)^2) ) )
  return (rmspe)
}

RMSPE.ens = function(pred, obs) {
  ignIdx = which(obs==0)
  if (length(ignIdx)>0) {
    obs = obs[-ignIdx]
    pred = pred[-ignIdx]
  }
  
  stopifnot(sum(obs==0)==0)
  
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)
  
  rmspe = sqrt(mean( ((1-pred/obs)^2) ) )
  return (rmspe)
}

RMSPECostSummary <- function (data, lev = NULL, model = NULL) {
  c(postResample(data[, "pred"], data[, "obs"]),
    RMSPE = RMSPE(pred = data[, "pred"], obs = data[, "obs"]))
}

RMSPE.xgb.exp <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  ignIdx = which(labels==0)
  if (length(ignIdx)>0) {
    labels = labels[-ignIdx]
    preds = preds[-ignIdx]
  }
  
  labels <- exp(as.numeric(labels))-1
  preds <- exp(as.numeric(preds))-1
  
  stopifnot(sum(labels==0)==0)
  err <- sqrt(  mean(  ((rep(1,length(preds))-preds/labels)^2)   ) )
  return(list(metric = "RMSPE", value = err))
}

RMSPE.xgb <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  ignIdx = which(labels==0)
  if (length(ignIdx)>0) {
    labels = labels[-ignIdx]
    preds = preds[-ignIdx]
  }
  stopifnot(sum(labels==0)==0)
  err <- sqrt(  mean(  ((rep(1,length(preds))-preds/labels)^2)   ) )
  return(list(metric = "RMSPE", value = err))
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
  } else if (identical("base2",dataProc)) {
    ## Xtrain / Xtest / Ytrain 
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain2.csv" , sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest2.csv" , sep='') , stringsAsFactors = F))
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
  } else if (identical("default",dataProc) && mod$layer > 1) {
    prev_layer = mod$layer -1 
    
    ## raw data 
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_bench.csv" , sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_bench.csv" , sep='') , stringsAsFactors = F))
    Ytrain = Xtrain$Sales
    test_id = Xtest$Id
    
    rm(list=c("Xtrain","Xtest"))
    gc()
    
    ##
    ff.bindPath(type = 'ensembles' , sub_path = 'dataset/rossmann-store-sales/ensembles/') 
    ens_dirs = list.files( ff.getPath('ensembles') )
    ens_dirs = ens_dirs[which(substr(x = ens_dirs, start = 1 , stop = nchar('ensemble_')) == "ensemble_")]
    cat(">>> Found ",length(ens_dirs),"ensemble directory:",ens_dirs,"\n")
    
    ## ens_dir
    ensembles_scores = NULL
    Xtrain = NULL
    Xtest = NULL
    
    for (i in 1:length(ens_dirs)) {
      if (i>prev_layer) next 
      ens_dir = paste0('ensemble_',i)
      stopifnot(ens_dir %in% ens_dirs) 
      ensembles_i = list.files( paste0(ff.getPath('ensembles') , ens_dir) )
      cat(">>> processing ",ens_dir," --> found ",length(ensembles_i),"ensembles...\n")
      if (length(ensembles_i) == 0) next 
      
      ensembles_scores_i = data.frame(ID = ensembles_i , layer = rep(i,length(ensembles_i)) , RMSPE=NA)
      Xtrain_i = data.frame(matrix(rep(NA,length(Ytrain)*length(ensembles_i)),ncol=length(ensembles_i),nrow=length(Ytrain)))
      Xtest_i = data.frame(matrix(rep(NA,length(test_id)*length(ensembles_i)),ncol=length(ensembles_i),nrow=length(test_id)))
      colnames(Xtrain_i) = ensembles_i
      colnames(Xtest_i) = ensembles_i
      
      for (j in ensembles_i) {
        cat("processing",j,"...\n")
        sub_j = as.data.frame( fread( paste(ff.getPath('ensembles') , ens_dir, .Platform$file.sep,j, sep='') , stringsAsFactors = F))
        predTrain = sub_j[1:length(Ytrain),'assemble']
        predTest = sub_j[(length(Ytrain)+1):nrow(sub_j),'assemble']
        ensembles_scores_i[ensembles_scores_i$ID==j,'RMSPE'] <- RMSPE.ens(pred=predTrain, obs=Ytrain)
        trIdx = which(colnames(Xtrain_i) == j)
        Xtrain_i[,trIdx] = predTrain
        Xtest_i[,trIdx] = predTest
      }
      
      if (is.null(ensembles_scores)) {
        ensembles_scores = ensembles_scores_i
        Xtrain = Xtrain_i
        Xtest = Xtest_i 
      } else {
        ensembles_scores = rbind(ensembles_scores,ensembles_scores_i)
        Xtrain = rbind(Xtrain,Xtrain_i)
        Xtest = rbind(Xtest,Xtest_i)
      }
      rm(list=c("ensembles_scores_i","Xtrain_i","Xtest_i"))
    }
    
    ## post 
    ensembles_scores <- ensembles_scores[order(ensembles_scores$RMSPE,decreasing = F),]
    
    ## apply here threshold
    if (!is.null(mod$threshold)) {
      cat(">>> cutting at ",mod$threshold,"...\n")
      takeIdx = which(ensembles_scores$RMSPE <= mod$threshold)
      Xtrain = Xtrain[,takeIdx,drop=F]
      Xtest = Xtest[,takeIdx,drop=F]
    }
    
  } else {
    stop(paste0("unrecognized type of dataProc:",dataProc))
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

### DATA 
Xtest_Open_0 = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_Open_0.csv" , sep='') , stringsAsFactors = F))

################# MODELS 

modelList = list(
  ##############################################################################
  #                                    1 LAYER                                 #
  ##############################################################################
  
  #list(layer = 1  , dataProc = "base", mod = 'xgbTreeGTJ'  , eta=0.02 , max_depth = 6, tune=T)
  #list(layer = 1  , dataProc = "base2", mod = 'xgbTreeGTJ'  , eta=0.01 , max_depth = 6, tune=T)
  #list(layer = 1  , dataProc = "base", mod = 'cubist'  , tune=T)
  #list(layer = 1  , dataProc = "base", mod = 'rlm'  , tune=F)
  #list(layer = 1  , dataProc = "base2", ytranf="log", mod = 'glm'  , tune=F)
  
  #list(layer = 1  , dataProc = "bech", ytranf="log", mod = 'glm'  , tune=F)
  #list(layer = 1  , dataProc = "bech", mod = 'glm'  , tune=F)
  #list(layer = 1  , dataProc = "bech", ytranf="log", mod = 'rlm'  , tune=F)
  #list(layer = 1  , dataProc = "bech", ytranf="log", mod = 'pls'  , tune=T)
  #list(layer = 1  , dataProc = "bech", ytranf="log", mod = 'xgbTreeGTJ'  , eta=0.01 , max_depth = 10, tune=T), 
  #list(layer = 1  , dataProc = "bech", ytranf="log", mod = 'xgbTreeGTJ'  , eta=0.005 , max_depth = 10, tune=T)
  
  #list(layer = 1  , dataProc = "bech", ytranf="log", mod = 'gbm'  , tune=T) 
  #list(layer = 1  , dataProc = "bech", ytranf="log", mod = 'treebag'  , tune=T) 
  
  list(layer = 1  , dataProc = "bech", ytranf="log", mod = 'xgbTreeGTJ'  , eta=0.01 , max_depth = 5, tune=T)
  
  ##############################################################################
  #                                    2 LAYER                                 #
  ##############################################################################
  
  #list(layer = 2  , dataProc = "default", mod = 'glm'  ,tune=F)
  #list(layer = 2  , dataProc = "default", mod = 'enet'  ,tune=T)
  #list(layer = 2  , dataProc = "default", mod = 'xgbTreeGTJ'  , eta=0.01 , tune=T)
  
  #list(layer = 2  , dataProc = "default", mod = 'ridge'  ,tune=T), 
  #list(layer = 2  , dataProc = "default", mod = 'knn'  ,tune=T)
  
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
  
  ### encoding datasets for not tree based models 
  if (modelList[[m]]$mod != "xgbTreeGTJ" & modelList[[m]]$mod == "base")  {
    cat(">>> encoding datasets for not tree based models ... \n")
    l <- ff.makeFeatureSet(data.train = Xtrain , data.test = Xtest , 
                           meta = c("N","C","N","N","C","C","C","C","N","N","N"), 
                           scaleNumericFeatures = F , 
                           remove1DummyVarInCatPreds = F , 
                           parallelize = T)
    Xtrain <- l$traindata
    Xtest <- l$testdata
    rm(l)
  }
  
  ## check for transformations 
  if (!is.null(modelList[[m]]$ytranf)) {
    cat(">>> applying ",modelList[[m]]$ytranf," transf. to Y ... \n")
    Ytrain <- log(Ytrain+1)
  }
  ## train_size
  if (!is.null(modelList[[m]]$train_size)) {
    cat(">>> cutting train / ytrain to ",modelList[[m]]$train_size,"  ... \n")
    h<-sample(nrow(Xtrain),modelList[[m]]$train_size)
    Xtrain <- Xtrain[h,]
    Ytrain <- Ytrain[h]
  }
  
  ##############
  ## TUNE 
  ##############
  controlObject = trainControl(method = "repeatedcv", repeats = nrepeats, number = nFolds) 
  controlObject$summaryFunction <- RMSPECostSummary
  if (modelList[[m]]$mod == "xgbTreeGTJ") { 
  param <- list(  objective           = "reg:linear", 
                  booster             =  "gbtree",
                  eta                 = 0.01, # 0.06, #0.01,
                  max_depth           = 10, #changed from default of 8
                  subsample           = 0.9, # 0.7
                  colsample_bytree    = 0.7 # 0.7
                  #num_parallel_tree   = 2
                  # alpha = 0.0001, 
                  # lambda = 1
  )
  param['eta'] = modelList[[m]]$eta
  param['max_depth'] = modelList[[m]]$max_depth
 
  l = ff.trainAndPredict.reg ( Ytrain=Ytrain ,
                               Xtrain=Xtrain , 
                               Xtest=Xtest , 
                               model.label=modelList[[m]]$mod , 
                               controlObject=controlObject, 
                               best.tuning = TRUE, 
                               verbose = TRUE, 
                               removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = TRUE, 
                               xgb.metric.fun = if (is.null(modelList[[m]]$ytranf)) RMSPE.xgb else RMSPE.xgb.exp,
                               xgb.maximize = FALSE, 
                               xgb.metric.label = 'RMSPE', 
                               xgb.foldList = NULL,
                               xgb.eta = NULL, 
                               xgb.max_depth = NULL,
                               xgb.param = param, 
                               xgb.cv.default = F
                               #,nrounds = 42
                               , metric = "RMSPE"
  )
  } else {
    l = ff.trainAndPredict.reg ( Ytrain=Ytrain ,
                                 Xtrain=Xtrain , 
                                 Xtest=Xtest , 
                                 model.label=modelList[[m]]$mod , 
                                 controlObject=controlObject, 
                                 best.tuning = TRUE, 
                                 verbose = TRUE, 
                                 removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = TRUE, 
                                 xgb.metric.fun = if (is.null(modelList[[m]]$ytranf)) RMSPE.xgb else RMSPE.xgb.exp, 
                                 xgb.maximize = FALSE, 
                                 xgb.metric.label = 'RMSPE', 
                                 xgb.foldList = NULL,
                                 xgb.eta = NULL, 
                                 xgb.max_depth = NULL, 
                                 xgb.cv.default = F
                                 , metric = "RMSPE", maximize=F
    )
  }
  
  if ( !is.null(l$model) ) {
    perf_mod = min(l$model$results$RMSPE)
    stopifnot(! is.null(perf_mod) , perf_mod >=0 )
    bestTune = l$model$bestTune
    pred = l$pred
    if (!is.null(modelList[[m]]$ytranf)) {
      cat(">>> applying inverse tranf. of ",modelList[[m]]$ytranf," to predictions ... \n")
      pred <- exp(pred)-1
    }
    secs = l$secs 
    rm(l)
  } else {
    stop(paste('model',modelList[[m]]$mod,':error!'))
  }
  
  ## write best tune on disk 
  if (modelList[[m]]$tune) {
    tuneGrid = data.frame(model=modelList[[m]]$mod,secs=secs,RMSPE=perf_mod) 
    tuneGrid = cbind(tuneGrid,bestTune)
    write.csv(tuneGrid,
              quote=FALSE, 
              file=paste0(ff.getPath(paste0("best_tune_",modelList[[m]]$layer)),modelList[[m]]$id) ,
              row.names=FALSE)
  }
  
  ## write prediction on disk 
  stopifnot(sum(is.na(pred))==0)
  stopifnot(sum(pred==Inf)==0)
  submission <- data.frame(Id=test_id)
  submission$Sales <- pred
  if (modelList[[m]]$dataProc == "base" || modelList[[m]]$dataProc == "base2") submission = rbind(submission,Xtest_Open_0) ## attach Sales-0 predictions 
  if (is.null(modelList[[m]]$train_size)) stopifnot(nrow(submission)==41088)
  print(head(submission))
  write.csv(submission,
            quote=FALSE, 
            file=paste0(ff.getPath(paste0("submission_",modelList[[m]]$layer)),modelList[[m]]$id) ,
            row.names=FALSE)
  
  
  
  ##############
  ## ENSEMB 
  ##############
  if (!is.null(modelList[[m]]$train_size)) {
    cat(">>> no ensembling for this model .... \n")
    next 
  }
  cat(">>> Ensembling ... \n") 
  bestTune = NULL
  if (modelList[[m]]$tune) {
    bestTune = as.data.frame( fread(paste0(ff.getPath(paste0("best_tune_",modelList[[m]]$layer)),modelList[[m]]$id)))
    stopifnot( nrow(bestTune)>0 , !is.null(bestTune) )
  }
  submission = as.data.frame( fread(paste0(ff.getPath(paste0("submission_",modelList[[m]]$layer)),modelList[[m]]$id)))
  stopifnot( nrow(submission)>0 , !is.null(submission) )
  
  nFolds = controlObject$number
  nrepeats =  controlObject$repeats
  index = caret::createMultiFolds(y=Ytrain, nFolds, nrepeats)
  indexOut <- lapply(index, function(training, allSamples) allSamples[-unique(training)], allSamples = seq(along = Ytrain))
  controlObject = trainControl(method = "cv", 
                               ## The method doesn't really matter
                               ## since we defined the resamples
                               index = index, 
                               indexOut = indexOut)
  controlObject$summaryFunction <- RMSPECostSummary
  rm(list = c("index","indexOut"))
  
  ## createEnsemble
  ff.setMaxCuncurrentThreads(4)
  ens = NULL 
  if (modelList[[m]]$mod == "xgbTreeGTJ") {
    ens = ff.createEnsemble (Xtrain = Xtrain,
                             Xtest = Xtest,
                             y =Ytrain,
                             caretModelName = 'xgbTree', 
                             predTest <- submission$Sales,
                             bestTune = expand.grid(
                               nrounds = bestTune$early.stop ,
                               max_depth = if (!is.null(modelList[[m]]$max_depth)) modelList[[m]]$max_depth else 8 ,  
                               eta = modelList[[m]]$eta ),
                             removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = TRUE, 
                             controlObject = controlObject, 
                             parallelize = T,
                             verbose = T , 
                             regression = TRUE, 
                             
                             ### ... 
                             objective = "reg:linear",
                             booster             =  "gbtree",
                             subsample           = 0.9, # 0.7
                             colsample_bytree    = 0.7, # 0.7
                             metric = "RMSPE" , maximize=F
                             )
    
  } else if (modelList[[m]]$tune){
    ens = ff.createEnsemble (Xtrain = Xtrain,
                             Xtest = Xtest,
                             y = Ytrain,
                             caretModelName = modelList[[m]]$mod, 
                             predTest = submission$Sales,
                             bestTune = bestTune[, 4:ncol(bestTune) , drop = F], 
                             removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = T, 
                             controlObject = controlObject, 
                             parallelize = T,
                             verbose = T , 
                             regression = T, 
                             metric = "RMSPE" , maximize=F)
  } else {
    ens = ff.createEnsemble (Xtrain = Xtrain,
                             Xtest = Xtest,
                             y = Ytrain,
                             caretModelName = modelList[[m]]$mod, 
                             predTest = submission$Sales,
                             bestTune = NULL, 
                             removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = T, 
                             controlObject = controlObject, 
                             parallelize = T,
                             verbose = T , 
                             regression = T, 
                             metric = "RMSPE" , maximize=F)
  }
  
  predTrain <- ens$predTrain
  predTest <- ens$predTest 
  if (!is.null(modelList[[m]]$ytranf)) {
    cat(">>> applying inverse tranf. of ",modelList[[m]]$ytranf," to ensembling predictions ... \n")
    predTrain <- exp(predTrain)-1
    ##predTest <- exp(predTest)-1
  }
  
  ## assemble 
  assemble = c(predTrain,predTest)
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


