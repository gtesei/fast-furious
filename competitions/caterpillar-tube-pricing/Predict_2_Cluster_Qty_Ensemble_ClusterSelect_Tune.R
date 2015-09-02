require(xgboost)
require(methods)
library(data.table)
library(plyr)
library(Hmisc)

library(lattice)
require(gridExtra) 
library(fastfurious)
library(parallel)

### FUNCS 
reg_train_predict = function( YtrainingSet ,
                              XtrainingSet , 
                              testSet , 
                              model.label , 
                              controlObject, 
                              best.tuning = F, ... ) {
  model = NULL 
  pred = NULL 
  secs = NULL
  
  ptm <- proc.time()
  l = tryCatch({
    if (model.label == "LinearReg") {   ### LinearReg
      model <- train(y = YtrainingSet, x = XtrainingSet , method = "lm", trControl = controlObject)
      
      pred = as.numeric( predict(model , testSet )  ) 
    } else if (model.label == "RobustLinearReg") {   ### RobustLinearReg
      #######
      l = featureSelect (XtrainingSet,
                         testSet,
                         removeOnlyZeroVariacePredictors=T,
                         performVarianceAnalysisOnTrainSetOnly = T , 
                         removePredictorsMakingIllConditionedSquareMatrix = T, 
                         removeHighCorrelatedPredictors = F, 
                         featureScaling = F)
      XtrainingSet = l$traindata
      testSet = l$testdata
      #######
      model <- train(y = YtrainingSet, x = XtrainingSet , method = "rlm", preProcess="pca", trControl = controlObject)
      
      pred = as.numeric( predict(model , testSet )  ) 
    } else if (model.label == "KNN_Reg") {  ### KNN_Reg
      model <- train(y = YtrainingSet, x = XtrainingSet , method = "knn", preProc = c("center", "scale"), 
                     tuneGrid = data.frame(.k = 1:10),
                     trControl = controlObject)
      
      pred = as.numeric( predict(model , testSet )  )
    } else if (model.label == "PLS_Reg") {  ### PLS_Reg
      #######
      l = featureSelect (XtrainingSet,
                         testSet,
                         removeOnlyZeroVariacePredictors=T,
                         performVarianceAnalysisOnTrainSetOnly = T , 
                         removePredictorsMakingIllConditionedSquareMatrix = T, 
                         removeHighCorrelatedPredictors = F, 
                         featureScaling = F)
      XtrainingSet = l$traindata
      testSet = l$testdata
      #######
      .tuneGrid = expand.grid(.ncomp = 1:10)
      
      model <- train(y = YtrainingSet, x = XtrainingSet ,
                     method = "pls",
                     tuneGrid = .tuneGrid , 
                     trControl = controlObject)
      
      pred = as.numeric( predict(model , testSet )  )
    } else if (model.label == "Ridge_Reg") {  ### Ridge_Reg
      #######
      l = featureSelect (XtrainingSet,
                         testSet,
                         removeOnlyZeroVariacePredictors=T,
                         performVarianceAnalysisOnTrainSetOnly = T , 
                         removePredictorsMakingIllConditionedSquareMatrix = T, 
                         removeHighCorrelatedPredictors = F, 
                         featureScaling = F)
      XtrainingSet = l$traindata
      testSet = l$testdata
      #######
      ridgeGrid <- data.frame(.lambda = seq(0, .1, length = 15))
      if (best.tuning) data.frame(.lambda = seq(0, .1, length = 25))
      
      model <- train(y = YtrainingSet, x = XtrainingSet ,
                     method = "ridge",
                     tuneGrid = ridgeGrid,
                     trControl = controlObject)
      
      pred = as.numeric( predict(model , testSet )  )
    } else if (model.label == "Enet_Reg") {  ### Enet_Reg
      #######
      l = featureSelect (XtrainingSet,
                         testSet,
                         removeOnlyZeroVariacePredictors=T,
                         performVarianceAnalysisOnTrainSetOnly = T , 
                         removePredictorsMakingIllConditionedSquareMatrix = T, 
                         removeHighCorrelatedPredictors = F, 
                         featureScaling = F)
      XtrainingSet = l$traindata
      testSet = l$testdata
      #######
      
      enetGrid <- expand.grid(.lambda = c(0, 0.01, .1), .fraction = seq(.05, 1, length = 20))
      if (best.tuning) enetGrid <- expand.grid(.lambda = c(0, 0.01,.1,.5,.8), .fraction = seq(.05, 1, length = 30))
      
      model <- train(y = YtrainingSet, x = XtrainingSet ,
                     method = "enet",
                     tuneGrid = enetGrid, 
                     trControl = controlObject)
      
      pred = as.numeric( predict(model , testSet )  )
    } else if (model.label == "SVM_Reg") {  ### SVM_Reg
      model <- train(y = YtrainingSet, x = XtrainingSet ,
                     method = "svmRadial",
                     tuneLength = 15,
                     trControl = controlObject)
      
      pred = as.numeric( predict(model , testSet )  )
    } else if (model.label == "BaggedTree_Reg") {  ### BaggedTree_Reg
      model <- train(y = YtrainingSet, x = XtrainingSet ,
                     method = "treebag",
                     trControl = controlObject)
      
      pred = as.numeric( predict(model , testSet )  )
    } else if (model.label == "GBM_Reg") {  ### GBM
      gbmGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                             n.trees = seq(100, 1000, by = 50),
                             shrinkage = c(0.01, 0.1), 
                             n.minobsinnode = 10)
      
      model <- train(y = YtrainingSet, x = XtrainingSet ,
                     method = "gbm",
                     tuneGrid = gbmGrid,
                     bag.fraction = 0.5 , 
                     trControl = controlObject, 
                     verbose = F)
      
      pred = as.numeric( predict(model , testSet )  )
    } else if (model.label == "RandomForest_Reg") {  ### RandomForest_Reg
      model <- train(y = YtrainingSet, x = XtrainingSet ,
                     method = "rf",
                     tuneLength = 10,
                     ntrees = 1000,
                     importance = TRUE,
                     trControl = controlObject)
    
      pred = as.numeric( predict(model , testSet )  )
    } else if (model.label == "Cubist_Reg") {  ### Cubist_Reg
      cubistGrid <- expand.grid(.committees = c(1, 5, 10, 50, 75, 100),
                                .neighbors = c(0, 1, 3, 5, 7, 9))
      
      model <- train(y = YtrainingSet, x = XtrainingSet ,
                     method = "cubist",
                     tuneGrid = cubistGrid,
                     trControl = controlObject)
      
      pred = as.numeric( predict(model , testSet )  )
    } else if (model.label == "Average") {  ### Average 
      
      ltset = ifelse( ! is.null(dim(testSet)) , dim(testSet) , length(testSet) )
      pred = rep(mean(YtrainingSet),ltset)
      
    } else if (model.label == "NNet") {  ### Neural Networks 
      nnetGrid <- expand.grid(.decay = c(0.001, .01, .1),
                              .size = seq(1, 27, by = 2),
                              .bag = FALSE)
      
      model <- train(y = YtrainingSet, x = XtrainingSet ,
                     method = "avNNet",
                     tuneGrid = nnetGrid,
                     linout = TRUE,
                     trace = FALSE,
                     maxit = 1000,
                     trControl = controlObject)
      
      pred = as.numeric( predict(model , testSet )  )  
    } else if (model.label == "XGBoost") {  ### XGBoost 
      model <- train(y = YtrainingSet, x = XtrainingSet ,
                     method = "xgbTree",
                     trControl = controlObject, ... )
      
      pred = as.numeric( predict(model , testSet )  )  
    } else {
      stop("unrecognized model.label.")
    }
    list(model=model,pred=pred)
  } , error = function(err) { 
    print(paste("ERROR:  ",err))
    NULL
  })
  
  tm = proc.time() - ptm
  secs = as.numeric(tm[3])
  cat(">> ",model.label,": time elapsed:",secs," secs. [min:",secs/60,"] [hours:",secs/(60*60),"]\n")
  
  if (! is.null(l)) {
    model = l$model 
    pred = l$pred
  }
  
  return(list(pred = pred, model = model, secs = secs))
}

xgb_cross_val = function( data , y , cv.nround = 3000 , param , nfold = 5 , verbose=T) {
  
  inCV = T
  early.stop = cv.nround 
  perf.xg = NULL 
  
  while (inCV) {
    #cat(">> cv.nround: ",cv.nround,"\n") 
    bst.cv = xgb.cv(param=param, data = data , label = y, 
                    nfold = nfold, nrounds=cv.nround , verbose=verbose)
    print(bst.cv)
    early.stop = which(bst.cv$test.rmse.mean == min(bst.cv$test.rmse.mean) )
    if (length(early.stop)>1) early.stop = early.stop[length(early.stop)-1]
    cat(">> early.stop: ",early.stop," [test.rmse.mean:",bst.cv[early.stop,]$test.rmse.mean,"]\n") 
    if (early.stop < cv.nround) {
      inCV = F
      perf.xg = min(bst.cv$test.rmse.mean)
      cat(">> stopping [early.stop < cv.nround=",cv.nround,"] [perf.xg=",perf.xg,"] ... \n") 
    } else {
      cat(">> redo-cv [early.stop == cv.nround=",cv.nround,"] with 2*cv.nround ... \n") 
      cv.nround = cv.nround * 2 
    }
    gc()
  }
  
  return(list(early.stop=early.stop,perf.cv=perf.xg))
}

xgb_train_and_predict = function(train_set,
                                 y, 
                                 test_set, 
                                 param,
                                 cv.nround = 3000 , 
                                 nfold = 5 , 
                                 verbose=T) {
  
  data = rbind(train_set,test_set)
  
  ##
  x = as.matrix(data)
  x = matrix(as.numeric(x),nrow(x),ncol(x))
  
  trind = 1:nrow(train_set)
  teind = (nrow(train_set)+1):nrow(x)
  
  #### Run Cross Valication
  cat(">> Cross validation ... \n")
  
  xgb_xval = xgb_cross_val (data = x[trind,], 
                            y = y,  
                            cv.nround = cv.nround , 
                            param = param ,
                            nfold = nfold , 
                            verbose=verbose)
  
  # Prediction
  cat(">> Prediction ... \n")
  bst = xgboost(param=param, 
                data = x[trind,], 
                label = y, 
                nrounds=xgb_xval$early.stop,
                verbose=verbose)
  
  # Make prediction
  if (length(teind)>1){
    pred = predict(bst,x[teind,])
  } else {
    yy = rbind(x[teind,],x[teind,])
    .pred = predict(bst,yy)
    pred = .pred[1]
  }
  
  #pred = pred^-(1/0.3) 
  
  return(list(pred=pred,
              perf.cv=xgb_xval$perf.cv,
              early.stop=xgb_xval$early.stop))
}

cluster_by = function(predictor.train,predictor.test,num_bids = 8,verbose=T) {
  require(Hmisc)
  
  ## clustering by quantity 
  if (verbose) {
    print(describe(predictor.train))
    print(describe(predictor.test))
  }
  
  data = as.vector(c(predictor.train,predictor.test))
  q = as.numeric(quantile(data, probs = ((1:num_bids)/num_bids)))
  
  ## counting cluster card 
  num=rep(0,num_bids)
  for (i in 1:num_bids)
    if (i == 1) {
      num[i] = sum(data<=q[i])
    } else {
      num[i] = sum(data<=q[i] & data>q[i-1])
    }
  if (verbose) print(describe(num))
  
  ## mapping quantity to cluster qty 
  qty2lev = data.frame(qty = sort(unique(data)) , lev = NA)
  for (i in 1:nrow(qty2lev)) {
    for (k in 1:length(q)) {
      if (k == 1) {
        if (qty2lev[i,]$qty <= q[1])  {
          qty2lev[i,]$lev = 1
          break
        } 
      } else {
        if (qty2lev[i,]$qty <= q[k] & qty2lev[i,]$qty > q[k-1] )  {
          qty2lev[i,]$lev = k
          break
        } 
      }
    }
  }
  
  ## mapping qty_lev on data 
  if (verbose) cat(">> mapping qty_lev to data ... \n")
  tr_qty_lev = rep(NA,length(predictor.train))
  for (i in 1:length(predictor.train))
    tr_qty_lev[i] = qty2lev[qty2lev$qty==predictor.train[i],]$lev
  
  ts_qty_lev = rep(NA,length(predictor.test))
  for (i in 1:length(predictor.test))
    ts_qty_lev[i] = qty2lev[qty2lev$qty==predictor.test[i],]$lev
  
  return( list(levels.train = tr_qty_lev , levels.test = ts_qty_lev , theresolds = q) )
}

################# FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/caterpillar-tube-pricing/competition_data')
ff.bindPath(type = 'process' , sub_path = 'data_process')
ff.bindPath(type = 'elab' , sub_path = 'dataset/caterpillar-tube-pricing/elab')
ff.bindPath(type = 'docs' , sub_path = 'dataset/caterpillar-tube-pricing/docs')
ff.bindPath(type = 'submission' , sub_path = 'dataset/caterpillar-tube-pricing')
ff.bindPath(type = 'submission_old' , sub_path = 'dataset/caterpillar-tube-pricing/submission')

source(paste0( ff.getPath("process") , "/FeatureSelection_Lib.R"))

################# SETTINGS
DEBUG_MODE = F

################# MODELS 
reg_models = c("XGBoost" , "Cubist_Reg" )


cluster_levs = 1:8 ##<<<<<<<---- :::::: <<<<<<<<< 
################# DATA IN 
sample_submission = as.data.frame( fread(paste(ff.getPath("data") , 
                                               "sample_submission.csv" , sep=''))) 

best_prediction = as.data.frame( fread(paste(ff.getPath("submission_old") , 
                                             "sub_cluster_by_qty_ensemble.csv" , sep='')))

## elab 
train_enc = as.data.frame( fread(paste(ff.getPath("elab") , 
                                       "train_enc.csv" , sep=''))) 

test_enc = as.data.frame( fread(paste(ff.getPath("elab") , 
                                      "test_enc.csv" , sep=''))) 

train_enc_date = as.data.frame( fread(paste(ff.getPath("elab") , 
                                            "train_enc_date.csv" , sep=''))) 

test_enc_date = as.data.frame( fread(paste(ff.getPath("elab") , 
                                           "test_enc_date.csv" , sep=''))) 

## tech props 
tube_base = as.data.frame( fread(paste(ff.getPath("elab") , 
                                  "tube_base.csv" , sep='')))

bom_base = as.data.frame( fread(paste(ff.getPath("elab") , 
                                       "bom_base.csv" , sep='')))

spec_enc = as.data.frame( fread(paste(ff.getPath("elab") , 
                                       "spec_enc.csv" , sep='')))


####>>>>>>>>>> PROCESSING 
## build technical feature set 
tube = cbind(tube_base,bom_base)
tube = cbind(tube,spec_enc)
dim(tube) ## 180 (encoded) technical features  
# [1] 21198   180

## putting quote_date in data set  
head_train_set = train_enc_date
head_test_set = test_enc_date

## build train_set and test_set 
train_set = merge(x = head_train_set , y = tube , by = 'tube_assembly_id' , all = F)
test_set = merge(x = head_test_set , y = tube , by = 'tube_assembly_id' , all = F)

######### feature scaling 
cat(">>> Feature scaling ... \n")
feature2scal = c(
  "quote_date"     ,    "annual_usage"   ,     "min_order_quantity"    ,       
  "diameter"       ,      "wall"         ,      "length"               , "num_bends"      ,     "bend_radius"     ,    
  "num_boss"       ,      "num_bracket"  ,      
  "CP_001_weight"  ,     "CP_002_weight"  ,     "CP_003_weight"  ,     "CP_004_weight"  ,    "CP_005_weight"   ,    "CP_006_weight"    ,  
  "CP_007_weight"  ,     "CP_008_weight"  ,     "CP_009_weight"  ,    "CP_010_weight"   ,    "CP_011_weight"   ,    "CP_012_weight"    ,  
  "CP_014_weight"  ,     "CP_015_weight"  ,     "CP_016_weight"  ,     "CP_017_weight"  ,     "CP_018_weight"  ,     "CP_019_weight"   ,   
  "CP_020_weight"  ,     "CP_021_weight"  ,     "CP_022_weight"  ,     "CP_023_weight"  ,     "CP_024_weight"  ,     "CP_025_weight"   ,   
  "CP_026_weight"  ,     "CP_027_weight"  ,      "CP_028_weight" ,      "CP_029_weight" ,      "OTHER_weight"  
)

trans.scal <- preProcess(rbind(train_set[,feature2scal],test_set[,feature2scal]),
                         method = c("center", "scale") )

print(trans.scal)

train_set[,feature2scal] = predict(trans.scal,train_set[,feature2scal])
test_set[,feature2scal] = predict(trans.scal,test_set[,feature2scal])

######### 

## clustering 
cls = cluster_by(predictor.train=train_set$quantity,
           predictor.test=test_set$quantity,
           num_bids = length(cluster_levs),
           verbose=T)

train_set$qty_lev = cls$levels.train
test_set$qty_lev = cls$levels.test

## grid 
grid = expand.grid(cluster = cluster_levs , model= reg_models)
grid$model = as.character(grid$model)

##############
## MAIN LOOP 
##############
ptm <- proc.time()
res_list = mclapply( 1:nrow(grid) , function(i) { 
  cls = grid[i,]$cluster
  model.label = grid[i,]$model
  pred = best_prediction$cost
  
  ###
  pid = paste('[cluster:',cls,'][model:',model.label,']',sep='')
  cat('>>> processing ',pid,'... \n')
  
  ## define train / test set 
  train_set_cl = train_set[train_set$qty_lev == cls,]
  test_set_cl = test_set[test_set$qty_lev== cls,]
  cat(pid,'>>> train observations:',nrow(train_set_cl), '- test observations:',nrow(test_set_cl), ' \n')
  if (nrow(test_set_cl) == 0) stop('something wrong') 
  
  pred_cl_idx = which(test_set$qty_lev==cls)
  stopifnot ( length(pred_cl_idx) == nrow(test_set_cl) ) 
  
  ##############
  ## DATA PROC 
  ##############
  
  ## tube_assembly_id , id 
  train_set_cl[, 'tube_assembly_id'] = NULL 
  test_set_cl [, 'tube_assembly_id'] = NULL 
  test_set_cl [, 'id'] = NULL 
  
  ## material_id 
  #   cat(">>> encoding material_id [",unique(c(train_set_cl$material_id , 
  #                                             test_set_cl$material_id)),"] [",length(unique(c(train_set_cl$material_id , 
  #                                                                                             test_set_cl$material_id))),"] ... \n")
  #   l = encodeCategoricalFeature (train_set_cl$material_id , test_set_cl$material_id , colname.prefix = "material_id" , asNumeric=F)
  #   cat(">>> train_set before encoding:",ncol(train_set_cl)," - test_set before encoding:",ncol(test_set_cl)," ... \n")
  #   train_set_cl = cbind(train_set_cl , l$traindata)
  #   test_set_cl = cbind(test_set_cl , l$testdata)
  
  train_set_cl[, 'material_id'] = NULL 
  test_set_cl [, 'material_id'] = NULL 
  cat(pid,">>> train_set after encoding:",ncol(train_set_cl)," - test_set after encoding:",ncol(test_set_cl)," ... \n")
  
  ## y, data 
  y = train_set_cl$cost   
  train_set_cl[, 'cost'] = NULL 
  
  ####### remove zero variance predictors   
  l = featureSelect (train_set_cl,
                     test_set_cl,
                     removeOnlyZeroVariacePredictors=T, ### <<< :::: <<< ---------
                     performVarianceAnalysisOnTrainSetOnly = T , 
                     removePredictorsMakingIllConditionedSquareMatrix = F, 
                     removeHighCorrelatedPredictors = F, 
                     featureScaling = F)
  train_set_cl = l$traindata
  test_set_cl = l$testdata
  
  #######  DEBUG
  if (DEBUG_MODE) {
    cat(pid,">>> Debug mode ... \n")
    train_set_cl = train_set_cl[1:20,]
    y = y[1:20]
  } else {
    cat(pid,">>> Production mode ... \n")
  }
  #######  end of DEBUG 
  
  ##############
  ## MODELING 
  ##############
  ## 
  rmse_xval_mod = NULL 
  pred_mod = NULL
  secs_mod = NULL
  early.stop = NULL
  committees = NULL
  neighbors = NULL 
  ### 
  if (model.label == "XGBoost") {
    ptm_xgb <- proc.time()
    
    param <- list("objective" = "reg:linear",
                  "eval_metric" = "rmse", 
                  "eta" = 0.05,  
                  "gamma" = 0.7,  
                  "max_depth" = 20, 
                  "subsample" = 0.5 , ## suggested in ESLII
                  "nthread" = 10, 
                  "min_child_weight" = 1 , 
                  "colsample_bytree" = 0.5, 
                  "max_delta_step" = 1)
    
    cat(pid,">> XGBoost Params:\n")
    print(param)
    xgb = xgb_train_and_predict (train_set = train_set_cl,
                                 y = y,  
                                 test_set = test_set_cl, 
                                 param = param,
                                 cv.nround = 3000 , 
                                 nfold = min(5,nrow(train_set_cl)) , 
                                 verbose=F)
    pred_mod = xgb$pred
    rmse_xval_mod = xgb$perf.cv
    early.stop = xgb$early.stop
    
    tm_xgb = proc.time() - ptm_xgb
    secs_mod = as.numeric(tm_xgb[3])
    
  } else {
    # 8-fold repteated 3 times 
    controlObject <- trainControl(method = "repeatedcv", repeats = 3, number = 8)
    l = reg_train_predict ( YtrainingSet = y , 
                            XtrainingSet = train_set_cl, 
                            testSet = test_set_cl, 
                            model.label = model.label, 
                            controlObject = controlObject, 
                            best.tuning = T)
    
    if ( !is.null(l$model) ) {
      rmse_xval_mod = min(l$model$results$RMSE)
      committees = l$model$bestTune$committees
      neighbors = l$model$bestTune$neighbors
    } else {
      rmse_xval_mod = 1000000
    }
    secs_mod = l$secs 
    pred_mod = l$pred
  }
  
  stopifnot(sum(is.na(pred_mod)) == 0) 
  stopifnot(length(pred_mod) == length(pred_cl_idx)) 
  
  cat(pid,'>> number of prediction < 0:',sum(pred_mod<0),' ... repleaced with 1.5 \n')
  pred_mod = ifelse(pred_mod<0,1.5,pred_mod)
  
  pred[pred_cl_idx] = pred_mod
    
  # output 
  return(list(cluster = cls,
              model.label = model.label,
              rmse_xval_mod = rmse_xval_mod,
              early.stop = early.stop,
              committees = committees, 
              neighbors = neighbors, 
              pred = pred,
              secs_mod = secs_mod)) 
  },
mc.cores = nrow(grid))
####### end of parallel loop 
tm = proc.time() - ptm
secs = as.numeric(tm[3])
cat(">>> MAIN LOOP >>> Time elapsed:",secs," secs. [",secs/60,"min.] [",secs/(60*60),"hours] \n")
############## end of MAIN LOOP 

## write on disk 
cat(">>> writing prediction / cluster_perf on disk ... \n")
grid$secs = NA
grid$rmse = NA
grid$early.stop = NA
grid$committees = NA
grid$neighbors = NA
alist = lapply( res_list , function(ll) { 
  cls = ll$cluster
  model.label = ll$model.label
  
  grid[grid$cluster == cls & grid$model == model.label,]$secs <<- ll$secs_mod
  grid[grid$cluster == cls & grid$model == model.label,]$rmse <<- ll$rmse_xval_mod
  if (model.label == "XGBoost") {
    grid[grid$cluster == cls & grid$model == model.label,]$early.stop <<- ll$early.stop
  } else {
    grid[grid$cluster == cls & grid$model == model.label,]$committees <<- ll$committees
    grid[grid$cluster == cls & grid$model == model.label,]$neighbors <<- ll$neighbors
  }
    
  pid = paste('[cluster:',cls,'][model:',model.label,']',sep='')
  fn =  paste('sub_cluster_by_qty_lev_',cls,'_mod_',model.label,'.csv',sep='')
  cat(pid,": writing prediction on disk ... \n")
  
  stopifnot( sum(ll$pred<0) == 0 )
  sample_submission$cost = ll$pred
#   write.csv(sample_submission,quote=FALSE, 
#             file=paste(ff.getPath("submission"),fn,sep='') ,
#             row.names=FALSE)
  })

write.csv(grid,quote=FALSE, 
          file=paste(ff.getPath("submission"),'grid_qty_lev_mod.csv',sep='') ,
          row.names=FALSE)
