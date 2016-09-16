library(Hmisc)
library(data.table)
library(FeatureHashing)
library(xgboost)
library(plyr)
library(Matrix)
library(fastfurious)

### FUNCS   
makeResampleIndexSameTubeAssemblyId = function (foldList) {
  
  stopifnot(! is.null(foldList) && length(foldList)>0)
  
  nFolds = unique(sort(foldList[[1]]))
  
  ## par out 
  ## e.g. Fold1.Rep1 - ResFold1.Rep1 / .. /  Fold5.Rep3 - ResFold5.Rep3
  index = list()
  indexOut = list()
  
  ##
  for (i in seq_along(foldList)) {
    for (j in seq_along(nFolds) ) {
      iF = which(foldList[[i]] != j) 
      iR = which(foldList[[i]] == j) 
      
      ## partition checks 
      stopifnot(length(intersect(iF,iR)) == 0)
      stopifnot(identical(seq_along(foldList[[i]]),sort(union(iF,iR))))
      
      index[[paste('Fold',j,'.Rep',i,sep='')]] = iF
      indexOut[[paste('ResFold',j,'.Rep',i,sep='')]]  = iR
    }
  }
  
  ## 
  return(list(
    index = index , 
    indexOut = indexOut
  )) 
}

createFoldsSameTubeAssemblyId = function (data,
                                          nFolds = 8, 
                                          repeats = 3, 
                                          seeds) {  
  ## par out 
  foldList = list()
  for (i in 1:repeats) {
    folds_i_name = paste0('folds.',i)
    foldList[[folds_i_name]] = rep(NA_integer_,nrow(data)) 
  }
  
  ## 
  clusters = ddply(data , .(people_id) , function(x) c(num = nrow(x)))
  clusters = clusters[order(clusters$num , decreasing = T),]
  stopifnot(sum(clusters$num)==nrow(data)) 
  
  for (j in 1:repeats) {
    folds = list()
    for (i in 1:nFolds) {
      folds_i_name = paste0('Fold',i)
      folds[[folds_i_name]] = rep(NA_integer_,nrow(data)) 
    }
    
    idx = 1 
    
    ##
    if (! is.null(seeds)) set.seed(seeds[j])
    seq = sample(1:nFolds)
    
    while (idx<=nrow(clusters)) {
      ## fw
      for (k in 1:length(seq)) {
        folds_k_name = paste0('Fold',seq[k])
        idx_k = min(which(is.na(folds[[folds_k_name]])))
        folds[[folds_k_name]][idx_k] = clusters[idx,'people_id'] 
        idx = idx + 1 
        if (idx > nrow(clusters)) break 
      }
      
      if (idx > nrow(clusters)) break 
      
      ## bw 
      for (k in length(seq):1) {
        folds_k_name = paste0('Fold',seq[k])
        idx_k = min(which(is.na(folds[[folds_k_name]])))
        folds[[folds_k_name]][idx_k] = clusters[idx,'people_id'] 
        idx = idx + 1 
        if (idx > nrow(clusters)) break 
      }
    }
    
    ## remove NAs and convert to chars 
    for (k in seq_along(seq)) folds[[k]] = as.integer(na.omit(folds[[k]]))
    
    ## union check 
    stopifnot(identical(intersect(clusters$people_id , Reduce(union , folds) ) , clusters$people_id)) 
    
    ## intersect check 
    samp = sample(1:nFolds,2,replace = F)
    stopifnot(length(intersect( folds[[samp[1]]] , folds[[samp[2]]]))==0) 
    
    ## refill nFolds 
    folds_j_name = paste0('folds.',j)
    for (k in 1:nFolds) {
      idx_k = which( data$people_id %in% folds[[k]] )
      foldList[[folds_j_name]] [idx_k] = k
    }
    
    ## checks
    stopifnot(sum( is.na(foldList[[folds_j_name]]) ) == 0) 
    stopifnot( length(foldList[[folds_j_name]]) == nrow(data) ) 
    stopifnot(identical(intersect(unique(sort(foldList[[folds_j_name]])) , 1:nFolds), 1:nFolds))
  }
  
  return(foldList)
}

get_data_base = function (train,test,people) {
  
  ##
  train[,activity_id:=NULL]
  test_id = test$activity_id
  test[,activity_id:=NULL]
  
  ## remove date of people 
  people[,date:=NULL]
  
  train_full = merge(x = train,y = people , by = "people_id", all.x = T)
  test_full = merge(x = test,y = people , by = "people_id", all.x = T)
  
  stopifnot(nrow(train)==nrow(train_full))
  stopifnot(nrow(test)==nrow(test_full))
  
  ## handle date 
  train_full$as_date = as.Date(train_full$date)
  train_full$as_date_year = as.numeric(format(train_full$as_date, "%Y"))
  train_full$as_date_month = as.numeric(format(train_full$as_date, "%m"))
  train_full$as_date_day = as.numeric(format(train_full$as_date, "%d"))
  train_full[,date:=NULL]
  train_full[,as_date:=NULL]
  
  test_full$as_date = as.Date(test_full$date)
  test_full$as_date_year = as.numeric(format(test_full$as_date, "%Y"))
  test_full$as_date_month = as.numeric(format(test_full$as_date, "%m"))
  test_full$as_date_day = as.numeric(format(test_full$as_date, "%d"))
  test_full[,date:=NULL]
  test_full[,as_date:=NULL]
  
  ## encode 
  feature.names <- names(train_full)
  cat("assuming text variables are categorical & replacing them with numeric ids\n")
  for (f in feature.names) {
    if (class(train_full[[f]])=="character") {
      cat(">>> ",f," is character -> encoding \n")
      levels <- unique(c(train_full[[f]], test_full[[f]]))
      train_full[[f]] <- as.integer(factor(train_full[[f]], levels=levels))
      test_full[[f]]  <- as.integer(factor(test_full[[f]],  levels=levels))
    } else if (class(train_full[[f]])=="logical") {
      cat(">>> ",f," is logical -> encoding \n")
      train_full[[f]] <- as.integer(train_full[[f]])
      test_full[[f]]  <- as.integer(test_full[[f]])
    }
  }
  
  stopifnot(sum(is.na(train_full))==0)
  stopifnot(sum(is.na(test_full))==0)
  
  ## outcome 
  Ytrain = train_full$outcome
  train_full[,outcome:=NULL]
  
  return(list(
    Ytrain = Ytrain ,
    Xtrain = train_full , 
    Xtest = test_full, 
    test_id = test_id 
  ))
}

getData = function(mod) {
  Ytrain = NULL
  Xtrain = NULL
  Xtest = NULL
  
  dataProc = mod$dataProc
  if (identical("base",dataProc)) {
    cat(">>> dataProc base ...\n")
    l = get_data_base(train,test,people)
    Ytrain = l$Ytrain
    Xtrain = l$Xtrain
    Xtest = l$Xtest 
    test_id = l$test_id
    rm(l)
    gc()
  } else {
    stop(paste0("unrecognized dataProc:",dataProc))
  }
  
  cat(">>> loaded Ytrain:",length(Ytrain),"\n")
  cat(">>> loaded Xtrain:",dim(Xtrain),"\n")
  cat(">>> loaded Xtest:",dim(Xtest),"\n")
  
  return(list(
    Ytrain = as.numeric(Ytrain), 
    #Xtrain = as.data.frame(Xtrain), 
    #Xtest = as.data.frame(Xtest), 
    Xtrain = Xtrain, 
    Xtest = Xtest, 
    test_id = test_id 
  ))
  
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

### CONFIG 
DEBUG = F

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/predicting-red-hat-business-value')
ff.bindPath(type = 'code' , sub_path = 'competitions/predicting-red-hat-business-value')

ff.bindPath(type = 'ensemble_1' , sub_path = 'dataset/predicting-red-hat-business-value/ensembles/ensemble_1',createDir = T) ## out 
ff.bindPath(type = 'best_tune_1' , sub_path = 'dataset/predicting-red-hat-business-value/best_tune_1',createDir = T) ## out 
ff.bindPath(type = 'submission_1' , sub_path = 'dataset/predicting-red-hat-business-value/pred_ensemble_1',createDir = T) ## out 

## DATA 
cat(Sys.time())
cat("Reading data\n")
### data 
prefix = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/predicting-red-hat-business-value/'
train <- fread(paste0(prefix,"act_train.csv"), header=TRUE)
test <- fread(paste0(prefix,"act_test.csv"), header=TRUE)
people <- fread(paste0(prefix,"people.csv"), header=TRUE)
sample_submission <- fread(paste0(prefix,"sample_submission.csv"), header=TRUE)

################# MODELS 
modelList = list(
  
  ##############################################################################
  #                                    1 LAYER                                 #
  ##############################################################################
  
  #list(layer = 1  , dataProc = "base", mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T)
  
  list(layer = 1  , dataProc = "base", mod = 'xgbTreeGTJ'  , eta=0.01 , tune=T)
  
  #list(layer = 1  , dataProc = "base", mod = 'glmnet_alpha_1'   , tune=T),
  #list(layer = 1  , dataProc = "base", mod = 'glmnet_alpha_0'   , tune=T),
  
  
  ##############################################################################
  #                                    2 LAYER                                 #
  ##############################################################################
  
  ############## 
  #list(layer = 2  , dataProc = "default", mod = 'glmnet_alpha_1'  , tune=T), 
  #list(layer = 2  , dataProc = "default", mod = 'glmnet_alpha_0.4'  , tune=T), 
  #list(layer = 2  , dataProc = "default", mod = 'glmnet_alpha_0.5'  , tune=T), 
  #list(layer = 2  , dataProc = "default", mod = 'glmnet_alpha_0.6'  , tune=T), 
  #list(layer = 2  , dataProc = "default", mod = 'glmnet_alpha_0'  , tune=T),
  
  #list(layer = 2  , dataProc = "default", mod = 'glm'  ,tune=F),
  #list(layer = 2  , dataProc = "default", mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T), 
  #list(layer = 2  , dataProc = "default", mod = 'xgbTreeGTJ'  , eta=0.01 , tune=T), 
  #list(layer = 2  , dataProc = "default", mod = 'xgbTreeGTJ'  , eta=0.005 , tune=T)
  
  
  ### threshold 0.7 
  #list(layer = 2  , dataProc = "default", threshold = 0.7 , mod = 'glmnet_alpha_1'  , tune=T), 
  #list(layer = 2  , dataProc = "default", threshold = 0.7 , mod = 'glmnet_alpha_0.5'  , tune=T), 
  #list(layer = 2  , dataProc = "default", threshold = 0.7 , mod = 'glmnet_alpha_0'  , tune=T),
  
  #list(layer = 2  , dataProc = "default", threshold = 0.7 , mod = 'glm'  ,tune=F),
  #list(layer = 2  , dataProc = "default", threshold = 0.7 , mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T), 
  #list(layer = 2  , dataProc = "default", threshold = 0.7 , mod = 'libsvm'  ,tune=T),
  
  ### threshold 0.75
  #list(layer = 2  , dataProc = "default", threshold = 0.75 , mod = 'glmnet_alpha_1'  , tune=T), 
  #list(layer = 2  , dataProc = "default", threshold = 0.75 , mod = 'glmnet_alpha_0.5'  , tune=T), 
  #list(layer = 2  , dataProc = "default", threshold = 0.75 , mod = 'glmnet_alpha_0'  , tune=T), 
  
  #list(layer = 2  , dataProc = "default", threshold = 0.75 , mod = 'glm'  ,tune=F),
  #list(layer = 2  , dataProc = "default", threshold = 0.75 , mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T), 
  #list(layer = 2  , dataProc = "default", threshold = 0.75 , mod = 'libsvm'  ,tune=T), 
  
  ############## kmeans = 1 
  ### 
  #list(layer = 2  , dataProc = "default", kmeans = 1 , mod = 'glmnet_alpha_1'  , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 1 , mod = 'glmnet_alpha_0.5'  , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 1 , mod = 'glmnet_alpha_0'  , tune=T),
  #list(layer = 2  , dataProc = "default", kmeans = 1 , mod = 'glm'  ,tune=F),
  #list(layer = 2  , dataProc = "default", kmeans = 1 , mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 1 , mod = 'libsvm'  ,tune=T),
  
  ############## kmeans = 3 
  ### 
  #list(layer = 2  , dataProc = "default", kmeans = 3 , mod = 'glmnet_alpha_1'  , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 3 , mod = 'glmnet_alpha_0.5'  , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 3 , mod = 'glmnet_alpha_0'  , tune=T),
  #list(layer = 2  , dataProc = "default", kmeans = 3 , mod = 'glm'  ,tune=F),
  #list(layer = 2  , dataProc = "default", kmeans = 3 , mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 3 , mod = 'libsvm'  ,tune=T), 
  
  ############## err_an 
  ### 
  #list(layer = 2  , dataProc = "default", kmeans = 3 , err_an =T , mod = 'glmnet_alpha_1'  , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 3 , err_an =T , mod = 'glmnet_alpha_0.5'  , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 3 , err_an =T , mod = 'glmnet_alpha_0'  , tune=T),
  #list(layer = 2  , dataProc = "default", kmeans = 3 , err_an =T , mod = 'glm'  ,tune=F),
  #list(layer = 2  , dataProc = "default", kmeans = 3 , err_an =T , mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T)
  #list(layer = 2  , dataProc = "default", kmeans = 3 , err_an =T , mod = 'libsvm'  ,tune=T)
  
  ##############################################################################
  #                                    3 LAYER                                 #
  ##############################################################################
  
  #   list(layer = 3  , dataProc = "default", mod = 'glmnet_alpha_1'  , tune=T), 
  #   list(layer = 3  , dataProc = "default", mod = 'glmnet_alpha_0.4'  , tune=T), 
  #   list(layer = 3  , dataProc = "default", mod = 'glmnet_alpha_0.5'  , tune=T), 
  #   list(layer = 3  , dataProc = "default", mod = 'glmnet_alpha_0.6'  , tune=T), 
  #   list(layer = 3  , dataProc = "default", mod = 'glmnet_alpha_0'  , tune=T),
  #   
  #   list(layer = 3  , dataProc = "default", mod = 'glm'  ,tune=F),
  #   list(layer = 3  , dataProc = "default", mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T), 
  #   list(layer = 3  , dataProc = "default", mod = 'xgbTreeGTJ'  , eta=0.01 , tune=T), 
  #   list(layer = 3  , dataProc = "default", mod = 'xgbTreeGTJ'  , eta=0.005 , tune=T)
  
  
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
    Xtrain = Xtrain[1:1000,]
    #Xtest = Xtrain[,1:10]
    Ytrain = Ytrain[1:1000]
    gc()
  }
  
  ##############
  ## TUNE 
  ##############
  foldList = createFoldsSameTubeAssemblyId(data = Xtrain, nFolds = 4, repeats = 1, seeds=c(123))
  resamples = makeResampleIndexSameTubeAssemblyId(foldList)
  
  xbg.foldList = list()
  for (idx in 1:4) {
    xbg.foldList[[idx]] = which(foldList[[1]]==idx)
  } 
  
  
  #Xtrain$people_id <- NULL
  #Xtest$people_id <- NULL
  Xtrain[,people_id:=NULL]
  Xtest[,people_id:=NULL]
  
  
  controlObject = trainControl(method = "repeatedcv", 
                               ##repeats = 1, number = 4 , 
                               summaryFunction = twoClassSummary , 
                               index = resamples$index, 
                               indexOut = resamples$indexOut, 
                               classProbs = TRUE)
  
  l = ff.trainAndPredict.class ( Ytrain=Ytrain ,
                                 Xtrain=Xtrain , 
                                 Xtest=Xtest , 
                                 model.label=modelList[[m]]$mod , 
                                 controlObject=controlObject, 
                                 best.tuning = TRUE, 
                                 verbose = TRUE, 
                                 removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = F, 
                                 xgb.metric.fun = NULL, 
                                 xgb.maximize = TRUE, 
                                 metric.label = 'auc', 
                                 xgb.foldList = xbg.foldList,
                                 xgb.eta = modelList[[m]]$eta, 
                                 xgb.max_depth = modelList[[m]]$max_depth, 
                                 xgb.cv.default = FALSE)
  
  if ( !is.null(l$model) ) {
    roc_mod = max(l$model$results$ROC)
    bestTune = l$model$bestTune
    pred = l$pred
    pred.prob = l$pred.prob
    secs = l$secs 
    rm(l)
  } else {
    stop(paste('model',modelList[[m]]$mod,':error!'))
  }
  
  ## write prediction on disk 
  submission <- data.frame(activity_id=test_id)
  submission$outcome <- pred.prob
  print(head(submission))
  write.csv(submission,
            quote=FALSE, 
            file=paste0(ff.getPath(paste0("submission_",modelList[[m]]$layer)),modelList[[m]]$id) ,
            row.names=FALSE)
  
  ## write best tune on disk 
  tuneGrid = data.frame(model=modelList[[m]]$mod,secs=secs,ROC=roc_mod) 
  tuneGrid = cbind(tuneGrid,bestTune)
  write.csv(tuneGrid,
            quote=FALSE, 
            file=paste0(ff.getPath(paste0("best_tune_",modelList[[m]]$layer)),modelList[[m]]$id) ,
            row.names=FALSE)
  
  #############
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
  
  ## controlObject 
  # nFolds = controlObject$number
  # nrepeats =  controlObject$repeats
  # index = caret::createMultiFolds(y=Ytrain, nFolds, nrepeats)
  # indexOut <- lapply(index, function(training, allSamples) allSamples[-unique(training)], allSamples = seq(along = Ytrain))
  #foldList = createFoldsSameTubeAssemblyId(data = train_set_cl, nFolds = 4, repeats = 1, seeds=c(123))
  #resamples = makeResampleIndexSameTubeAssemblyId(foldList)
  controlObject = trainControl(method = "repeatedcv", 
                               ## The method doesn't really matter
                               ## since we defined the resamples
                               index = resamples$index, 
                               indexOut = resamples$indexOut, 
                               summaryFunction = twoClassSummary , classProbs = TRUE)
  rm(list = c("index","indexOut"))
  
  ## createEnsemble
  ff.setMaxCuncurrentThreads(4)
  ens = NULL 
  if (modelList[[m]]$mod == "xgbTreeGTJ") {
    ens = ff.createEnsemble (Xtrain = Xtrain,
                             Xtest = Xtest,
                             y = Ytrain,
                             caretModelName = 'xgbTree', 
                             predTest <- submission$target,
                             bestTune = expand.grid(
                               nrounds = bestTune$early.stop ,
                               max_depth = if (!is.null(modelList[[m]]$max_depth)) modelList[[m]]$max_depth else 8 ,  
                               eta = modelList[[m]]$eta , 
                               gamma = 0, 
                               colsample_bytree = 0.6 ,
                               min_child_weight = 1),
                             removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = F, 
                             controlObject = controlObject, 
                             parallelize = T,
                             verbose = T , 
                             regression = FALSE, 
                             
                             ### ... 
                             objective = "binary:logistic",
                             eval_metric = "auc", 
                             subsample = 0.7 , 
                             scale_pos_weight = 0.8, 
                             #silent = 1 , 
                             max_delta_step = 2)
    
  } else if (modelList[[m]]$tune){
    ens = ff.createEnsemble (Xtrain = Xtrain,
                             Xtest = Xtest,
                             y = Ytrain,
                             caretModelName = modelList[[m]]$mod, 
                             predTest = submission$target,
                             bestTune = bestTune[, 4:ncol(bestTune) , drop = F], 
                             removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = F, 
                             controlObject = controlObject, 
                             parallelize = T,
                             verbose = T , 
                             regression = FALSE)
  } else {
    ens = ff.createEnsemble (Xtrain = Xtrain,
                             Xtest = Xtest,
                             y = Ytrain,
                             caretModelName = modelList[[m]]$mod, 
                             predTest = submission$target,
                             bestTune = NULL, 
                             removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = F, 
                             controlObject = controlObject, 
                             parallelize = T,
                             verbose = T , 
                             regression = FALSE)
  }
  
  ## assemble 
  assemble = c(ens$predTrain,ens$predTest)
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

