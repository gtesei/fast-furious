library(data.table)
library(xgboost)
library(fastfurious)

### FUNCS
RMSLE = function(pred, obs) {
  if (sum(pred<0)>0) {
    pred = ifelse(pred >=0 , pred , 1.5)
  }
  rmsle = sqrt(    sum( (log(pred+1) - log(obs+1))^2 )   / length(pred))
  return (rmsle)
}

RMSLECostSummary <- function (data, lev = NULL, model = NULL) {
  c(postResample(data[, "pred"], data[, "obs"]),
    RMSLE = RMSLE(pred = data[, "pred"], obs = data[, "obs"]))
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
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_base.csv" , sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_base.csv" , sep='') , stringsAsFactors = F))
    Ytrain = as.data.frame( fread(paste(ff.getPath("elab") , "Ytrain_base.csv" , sep='') , stringsAsFactors = F))
    Ytrain = Ytrain$Ytrain
    
    test_id = Xtest$REN_ID
    
    ## MEMO #1: remove REN_ID in train / test set before fitting models 
    predToDel = c("REN_ID")
    for (pp in predToDel) {
      cat(">>> removing ",pp,"...\n")
      Xtrain[,pp] <- NULL
      Xtest[,pp] <- NULL
    }
  } else if (identical("base2",dataProc)) {
    ## Xtrain / Xtest / Ytrain 
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_base2.csv" , sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_base2.csv" , sep='') , stringsAsFactors = F))
    Ytrain = as.data.frame( fread(paste(ff.getPath("elab") , "Ytrain_base2.csv" , sep='') , stringsAsFactors = F))
    Ytrain = Ytrain$Ytrain
    
    test_id = Xtest$REN_ID
    
    ## MEMO #1: remove REN_ID in train / test set before fitting models 
    predToDel = c("REN_ID")
    for (pp in predToDel) {
      cat(">>> removing ",pp,"...\n")
      Xtrain[,pp] <- NULL
      Xtest[,pp] <- NULL
    }
  } else if (identical("NAs_2_3",dataProc)) {
    ## Xtrain / Xtest / Ytrain 
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_NAs_2_3.csv" , sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_NAs_2_3.csv" , sep='') , stringsAsFactors = F))
    Ytrain = as.data.frame( fread(paste(ff.getPath("elab") , "Ytrain_NAs_2_3.csv" , sep='') , stringsAsFactors = F))
    Ytrain = Ytrain$Ytrain
    
    test_id = Xtest$REN_ID
    
    ## MEMO #1: remove REN_ID in train / test set before fitting models 
    predToDel = c("REN_ID")
    for (pp in predToDel) {
      cat(">>> removing ",pp,"...\n")
      Xtrain[,pp] <- NULL
      Xtest[,pp] <- NULL
    }
  } else if (identical("NAs4",dataProc)) {
    ## Xtrain / Xtest / Ytrain 
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_NAs4.csv" , sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_NAs4.csv" , sep='') , stringsAsFactors = F))
    Ytrain = as.data.frame( fread(paste(ff.getPath("elab") , "Ytrain_NAs4.csv" , sep='') , stringsAsFactors = F))
    Ytrain = Ytrain$Ytrain
    
    test_id = Xtest$REN_ID
    
    ## MEMO #1: remove REN_ID in train / test set before fitting models 
    predToDel = c("REN_ID")
    for (pp in predToDel) {
      cat(">>> removing ",pp,"...\n")
      Xtrain[,pp] <- NULL
      Xtest[,pp] <- NULL
    }
  } else if (identical("NAs5",dataProc)) {
    ## Xtrain / Xtest / Ytrain 
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_NAs5.csv" , sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_NAs5.csv" , sep='') , stringsAsFactors = F))
    Ytrain = as.data.frame( fread(paste(ff.getPath("elab") , "Ytrain_NAs5.csv" , sep='') , stringsAsFactors = F))
    Ytrain = Ytrain$Ytrain
    
    test_id = Xtest$REN_ID
    
    ## MEMO #1: remove REN_ID in train / test set before fitting models 
    predToDel = c("REN_ID")
    for (pp in predToDel) {
      cat(">>> removing ",pp,"...\n")
      Xtrain[,pp] <- NULL
      Xtest[,pp] <- NULL
    }
  } else if (identical("default",dataProc) && mod$layer > 1) {
    prev_layer = mod$layer -1 
    
    ## raw data 
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_base.csv" , sep='') , stringsAsFactors = F))
    Ytrain = as.data.frame( fread(paste(ff.getPath("elab") , "Ytrain_base.csv" , sep='') , stringsAsFactors = F))
    Ytrain = Ytrain$Ytrain
    test_id = Xtest$REN_ID
  
    gc()
    
    ##
    ff.bindPath(type = 'ensembles' , sub_path = 'dataset/deloitte-western-australia-rental-prices/ensembles/') 
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
      
      ensembles_scores_i = data.frame(ID = ensembles_i , layer = rep(i,length(ensembles_i)) , RMSLE=NA)
      Xtrain_i = data.frame(matrix(rep(NA,length(Ytrain)*length(ensembles_i)),ncol=length(ensembles_i),nrow=length(Ytrain)))
      Xtest_i = data.frame(matrix(rep(NA,length(test_id)*length(ensembles_i)),ncol=length(ensembles_i),nrow=length(test_id)))
      colnames(Xtrain_i) = ensembles_i
      colnames(Xtest_i) = ensembles_i
      
      for (j in ensembles_i) {
        cat("processing",j,"...\n")
        sub_j = as.data.frame( fread( paste(ff.getPath('ensembles') , ens_dir, .Platform$file.sep,j, sep='') , stringsAsFactors = F))
        predTrain = sub_j[1:length(Ytrain),'assemble']
        predTest = sub_j[(length(Ytrain)+1):nrow(sub_j),'assemble']
        ensembles_scores_i[ensembles_scores_i$ID==j,'RMSLE'] <- RMSLE(pred=predTrain, obs=Ytrain)
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
    ensembles_scores <- ensembles_scores[order(ensembles_scores$RMSLE,decreasing = F),]
    
    ## apply here threshold
    if (!is.null(mod$threshold)) {
      cat(">>> cutting at ",mod$threshold,"...\n")
      takeIdx = which(ensembles_scores$RMSLE > mod$threshold) ### !!!! 
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


##
source(paste0(ff.getPath("code"),"fastImpute.R"))

################# MODELS 

modelList = list(
  ##############################################################################
  #                                    1 LAYER                                 #
  ##############################################################################
  
  #list(layer = 1  , dataProc = "base", mod = 'xgbTreeGTJ'  , eta=0.02 , max_depth = 6, tune=T)
  #list(layer = 1  , dataProc = "base2", mod = 'xgbTreeGTJ'  , eta=0.02 , max_depth = 6, tune=T), 
  #list(layer = 1  , dataProc = "NAs_2_3", mod = 'xgbTreeGTJ'  , eta=0.02 , max_depth = 6, tune=T)
  
  #list(layer = 1  , dataProc = "base", mod = 'cubist'  , tune=T)
  #list(layer = 1  , dataProc = "base", mod = 'rlm'  , tune=F)
  
  #list(layer = 1  , dataProc = "base", ytranf="log", mod = 'xgbTreeGTJ'  , eta=0.02 , max_depth = 6, tune=T)
  
  #list(layer = 1  , dataProc = "NAs4", mod = 'xgbTreeGTJ'  , eta=0.02 , max_depth = 9, tune=T) 
  
  #list(layer = 1  , dataProc = "NAs4", ytranf="log", mod = 'xgbTreeGTJ'  , eta=0.02 , max_depth = 9, tune=T)
  
  list(layer = 1  , dataProc = "NAs5", ytranf="log", mod = 'xgbTreeGTJ'  , eta=0.02 , max_depth = 9, tune=T)
  
  ##############################################################################
  #                                    2 LAYER                                 #
  ##############################################################################
  
  #list(layer = 2  , dataProc = "default", mod = 'glm'  ,tune=F , threshold = 0.32),
  #list(layer = 2  , dataProc = "default", mod = 'enet'  ,tune=T , threshold = 0.32),
  
  #list(layer = 2  , dataProc = "default", mod = 'xgbTreeGTJ'  , eta=0.01 , tune=T) ## TODO 
  
  #list(layer = 2  , dataProc = "default", mod = 'ridge'  ,tune=T , threshold = 0.32) 
  
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
  
  ## check for transformations 
  if (!is.null(modelList[[m]]$ytranf)) {
    cat(">>> applying ",modelList[[m]]$ytranf," transf. to Y ... \n")
    Ytrain <- log(Ytrain)
  }
  
  ##############
  ## TUNE 
  ##############
  controlObject = trainControl(method = "repeatedcv", repeats = 1, number = 4 , summaryFunction = RMSLECostSummary)
 
  if (modelList[[m]]$mod == "xgbTreeGTJ") {
  l = ff.trainAndPredict.reg ( Ytrain=Ytrain ,
                                 Xtrain=Xtrain , 
                                 Xtest=Xtest , 
                                 model.label=modelList[[m]]$mod , 
                                 controlObject=controlObject, 
                                 best.tuning = TRUE, 
                                 verbose = TRUE, 
                                 removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = TRUE, 
                                 xgb.metric.fun = RMSLE.xgb, 
                                 xgb.maximize = FALSE, 
                                 xgb.metric.label = 'RMSLE', 
                                 xgb.foldList = NULL,
                                 xgb.eta = modelList[[m]]$eta, 
                                 xgb.max_depth = modelList[[m]]$max_depth, 
                                 xgb.cv.default = F)
  } else {
    l = ff.trainAndPredict.reg ( Ytrain=Ytrain ,
                                 Xtrain=Xtrain , 
                                 Xtest=Xtest , 
                                 model.label=modelList[[m]]$mod , 
                                 controlObject=controlObject, 
                                 best.tuning = TRUE, 
                                 verbose = TRUE, 
                                 removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = TRUE, 
                                 xgb.metric.fun = RMSLE.xgb, 
                                 xgb.maximize = FALSE, 
                                 xgb.metric.label = 'RMSLE', 
                                 xgb.foldList = NULL,
                                 xgb.eta = NULL, 
                                 xgb.max_depth = NULL, 
                                 xgb.cv.default = F
                                 , metric = "RMSLE", maximize=F
    )
  }
  
  if ( !is.null(l$model) ) {
    perf_mod = min(l$model$results$RMSLE)
    stopifnot(! is.null(perf_mod) , perf_mod >=0 )
    bestTune = l$model$bestTune
    pred = l$pred
    if (!is.null(modelList[[m]]$ytranf)) {
      cat(">>> applying inverse tranf. of ",modelList[[m]]$ytranf," to predictions ... \n")
      pred <- exp(pred)
    }
    secs = l$secs 
    rm(l)
  } else {
    stop(paste('model',modelList[[m]]$mod,':error!'))
  }
  
  ## write prediction on disk 
  stopifnot(sum(is.na(pred))==0)
  stopifnot(sum(pred==Inf)==0)
  submission <- data.frame(REN_ID=test_id)
  submission$REN_BASE_RENT <- pred
  print(head(submission))
  write.csv(submission,
            quote=FALSE, 
            file=paste0(ff.getPath(paste0("submission_",modelList[[m]]$layer)),modelList[[m]]$id) ,
            row.names=FALSE)
  
  ## write best tune on disk 
  if (modelList[[m]]$tune) {
    tuneGrid = data.frame(model=modelList[[m]]$mod,secs=secs,RMSLE=perf_mod) 
    tuneGrid = cbind(tuneGrid,bestTune)
    write.csv(tuneGrid,
              quote=FALSE, 
              file=paste0(ff.getPath(paste0("best_tune_",modelList[[m]]$layer)),modelList[[m]]$id) ,
              row.names=FALSE)
  }
  
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
  
  nFolds = controlObject$number
  nrepeats =  controlObject$repeats
  index = caret::createMultiFolds(y=Ytrain, nFolds, nrepeats)
  indexOut <- lapply(index, function(training, allSamples) allSamples[-unique(training)], allSamples = seq(along = Ytrain))
  controlObject = trainControl(method = "cv", 
                               ## The method doesn't really matter
                               ## since we defined the resamples
                               index = index, 
                               indexOut = indexOut , 
                               summaryFunction = RMSLECostSummary )
  rm(list = c("index","indexOut"))
  
  
  ## converting fields to numeric for caret 
#   a = lapply(1:ncol(Xtrain), function(i) {
#     Xtrain[,i] <<- as.numeric(Xtrain[,i]) 
#     Xtest[,i] <<- as.numeric(Xtest[,i]) 
#   })
  
  ## createEnsemble
  ff.setMaxCuncurrentThreads(4)
  ens = NULL 
  if (modelList[[m]]$mod == "xgbTreeGTJ") {
    ens = ff.createEnsemble (Xtrain = Xtrain,
                             Xtest = Xtest,
                             y =Ytrain,
                             caretModelName = 'xgbTree', 
                             predTest <- submission$REN_BASE_RENT,
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
                             #eval_metric = "auc", 
                             subsample = 0.7 , 
                             colsample_bytree = 0.6 , 
                             scale_pos_weight = 0.8 , 
                             max_delta_step = 2)
    
  } else if (modelList[[m]]$tune){
    ens = ff.createEnsemble (Xtrain = Xtrain,
                             Xtest = Xtest,
                             y = Ytrain,
                             caretModelName = modelList[[m]]$mod, 
                             predTest = submission$REN_BASE_RENT,
                             bestTune = bestTune[, 4:ncol(bestTune) , drop = F], 
                             removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = T, 
                             controlObject = controlObject, 
                             parallelize = T,
                             verbose = T , 
                             regression = T, 
                             metric = "RMSLE" , maximize=F)
  } else {
    ens = ff.createEnsemble (Xtrain = Xtrain,
                             Xtest = Xtest,
                             y = Ytrain,
                             caretModelName = modelList[[m]]$mod, 
                             predTest = submission$REN_BASE_RENT,
                             bestTune = NULL, 
                             removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = T, 
                             controlObject = controlObject, 
                             parallelize = T,
                             verbose = T , 
                             regression = T, 
                             metric = "RMSLE" , maximize=F)
  }
  
  predTrain <- ens$predTrain
  predTest <- ens$predTest 
  if (!is.null(modelList[[m]]$ytranf)) {
    cat(">>> applying inverse tranf. of ",modelList[[m]]$ytranf," to ensembling predictions ... \n")
    predTrain <- exp(predTrain)
    #predTest <- exp(predTest)
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