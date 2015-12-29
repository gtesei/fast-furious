library(data.table)
library(xgboost)
library(fastfurious)

### FUNCS
RMSPE = function(pred, obs) {
  rmspe = sqrt(mean((1-pred/obs)^2))
  return (rmspe)
}

RMSPECostSummary <- function (data, lev = NULL, model = NULL) {
  c(postResample(data[, "pred"], data[, "obs"]),
    RMSPE = RMSPE(pred = data[, "pred"], obs = data[, "obs"]))
}

RMSPE.xgb <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
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
  list(layer = 1  , dataProc = "base", mod = 'glm'  , tune=F)
  #list(layer = 1  , dataProc = "base", mod = 'cubist'  , tune=T)
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
    
    trIdx = which(Xtrain$Store==st)
    teIdx = which(Xtest$Store==st)
    
    Xtrain_i = Xtrain[trIdx,]
    Xtest_i = Xtest[teIdx,]
    Ytrain_i = Ytrain[trIdx]
    
    ### encoding datasets for not tree based models 
    if (modelList[[m]]$mod != "xgbTreeGTJ")  {
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
    controlObject = trainControl(method = "repeatedcv", repeats = nrepeats, number = nFolds , summaryFunction = RMSPECostSummary)
    l = ff.trainAndPredict.reg ( Ytrain=Ytrain_i ,
                                 Xtrain=Xtrain_i , 
                                 Xtest=Xtest_i , 
                                 model.label=modelList[[m]]$mod , 
                                 controlObject=controlObject, 
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
    
    if ( !is.null(l$model) ) {
      perf_mod_i = min(l$model$results$RMSPE)
      stopifnot(! is.null(perf_mod_i) , perf_mod_i >=0 )
      bestTune_i = l$model$bestTune
      pred_i = l$pred
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
    pred[teIdx] <- pred_i  
    
    
    #####
    cat(">>> [ENSEMB]: processing Store [",stores[st],"] [",st,"/",length(stores),"] .. \n" )
    Ytest_i = pred_i
    
    ## resampling 
    index = caret::createMultiFolds(y=Ytrain_i, nFolds, 1)
    indexOut <- lapply(index, function(training, allSamples) allSamples[-unique(training)], allSamples = seq(along = Ytrain_i))
    controlObject = trainControl(method = "cv", 
                                 ## The method doesn't really matter
                                 ## since we defined the resamples
                                 index = index, 
                                 indexOut = indexOut , 
                                 summaryFunction = RMSPECostSummary )
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
                                 nrounds = tuneGrid[tuneGrid$Store==st,]$early.stop ,
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
                               bestTune = tuneGrid[tuneGrid$Store==st, 5:ncol(tuneGrid) , drop = F], 
                               removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = T, 
                               controlObject = controlObject, 
                               parallelize = T,
                               verbose = T , 
                               regression = T)
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
                               regression = T)
    }
    
    
    #####
    data = data.frame(time = Xtrain_i$Date , Sales = Ytrain_i , Pred = ens$predTrain)
    data = data[order(data$time,decreasing = F),]
    plot(x = data$time , y = data$Sales , type="l")
    lines(x=data$time,y=data$Pred,col=2)
    stop("here!!")
  }
  
  ## checks 
  stopifnot(sum(is.na(pred))==0)
  if (modelList[[m]]$tune) stopifnot(sum(is.na(tuneGrid))==0)
  
 
}
####### end of loop 
tm = proc.time() - ptm
secs = as.numeric(tm[3])
cat(">>> MAIN LOOP >>> Time elapsed:",secs," secs. [",secs/60,"min.] [",secs/(60*60),"hours] \n")
############## end of MAIN LOOP     

    
    