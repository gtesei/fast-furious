require(xgboost)
require(methods)
library(data.table)
library(plyr)
library(Hmisc)

library(lattice)
require(gridExtra) 

library(parallel)

### FUNCS 
getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/caterpillar-tube-pricing/competition_data"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/caterpillar-tube-pricing/competition_data/"
  } else if(type == "submission") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/caterpillar-tube-pricing"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/caterpillar-tube-pricing/"
  } else if(type == "submission_old") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/caterpillar-tube-pricing/submission"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/caterpillar-tube-pricing/submission/"
  } else if(type == "docs") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/caterpillar-tube-pricing/docs"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/caterpillar-tube-pricing/docs/"
  } else if(type == "elab") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/caterpillar-tube-pricing/elab"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/caterpillar-tube-pricing/elab/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/caterpillar-tube-pricing"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/caterpillar-tube-pricing/"
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

################# FAST-FURIOUS SOURCES
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))
source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))

################# SETTINGS
DEBUG_MODE = F
DO_PLOT = F
CLUSTER = c(8)

################# MODELS 
reg_models = c("XGBoost" , "LinearReg" , "RobustLinearReg" , "KNN_Reg" , "PLS_Reg" , "Ridge_Reg" , "Enet_Reg" , 
               "SVM_Reg" , "BaggedTree_Reg" , "RandomForest_Reg" , "Cubist_Reg" , "GBM_Reg")


cluster_levs = 1:8 
if (DO_PLOT) {
  model_list_by_cluster = vector(mode = 'list',length = length(cluster_levs))
}

################# DATA IN 
sample_submission = as.data.frame( fread(paste(getBasePath("data") , 
                                               "sample_submission.csv" , sep=''))) 

best_prediction = as.data.frame( fread(paste(getBasePath("submission_old") , 
                                             "sub_cluster_by_qty_ensemble.csv" , sep='')))

## docs 
xgb_grid = as.data.frame( fread(paste(getBasePath("docs") , 
                                              "cluster_lev_1_cluster_by_qty.csv" , sep=''))) 

## elab 
train_enc = as.data.frame( fread(paste(getBasePath("elab") , 
                                       "train_enc.csv" , sep=''))) 

test_enc = as.data.frame( fread(paste(getBasePath("elab") , 
                                      "test_enc.csv" , sep=''))) 

train_enc_date = as.data.frame( fread(paste(getBasePath("elab") , 
                                            "train_enc_date.csv" , sep=''))) 

test_enc_date = as.data.frame( fread(paste(getBasePath("elab") , 
                                           "test_enc_date.csv" , sep=''))) 

## tech props 
tube_base = as.data.frame( fread(paste(getBasePath("elab") , 
                                  "tube_base.csv" , sep='')))

bom_base = as.data.frame( fread(paste(getBasePath("elab") , 
                                       "bom_base.csv" , sep='')))

spec_enc = as.data.frame( fread(paste(getBasePath("elab") , 
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
# head_train_set = train_enc
# head_test_set = test_enc 

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

## pred 
pred = rep(NA,nrow(test_set))

## cluster_perf
cluster_perf = data.frame(cluster_lev = cluster_levs , 
                          cost_mean = NA, 
                          cost_sd = NA, 
                          zeta = NA, 
                          num_train = NA, 
                          num_test = NA , 
                          rmse_xval = NA , 
                          winner = NA)
## model performances 
mdf = as.data.frame(matrix(rep(NA,length(reg_models)*nrow(cluster_perf)),
                     nrow = nrow(cluster_perf) , 
                     ncol = length(reg_models)))
colnames(mdf) = reg_models
cluster_perf = cbind(cluster_perf , mdf)

## model excecution time  
mdf = as.data.frame(matrix(rep(NA,length(reg_models)*nrow(cluster_perf)),
                           nrow = nrow(cluster_perf) , 
                           ncol = length(reg_models)))
colnames(mdf) = paste0(reg_models,"_secs")
cluster_perf = cbind(cluster_perf , mdf)
  
##############
## MAIN LOOP 
##############
ptm <- proc.time()
cluster_levs = intersect(cluster_levs , CLUSTER)
for (cl in seq_along(cluster_levs)) {
  cat('>>> processing cluster_lev ',cluster_levs[cl], '[',cl,'/',length(cluster_levs),'] ... \n')
  
  ## define train / test set 
  train_set_cl = train_set[train_set$qty_lev == cluster_levs[cl],]
  test_set_cl = test_set[test_set$qty_lev== cluster_levs[cl],]
  cat('>>> train observations:',nrow(train_set_cl), '- test observations:',nrow(test_set_cl), ' \n')
  if (nrow(test_set_cl) == 0) next 
  
  pred_cl_idx = which(test_set$qty_lev==cluster_levs[cl])
  stopifnot ( length(pred_cl_idx) == nrow(test_set_cl) ) 
  
  ## mean / sd / zeta / num 
  cost_mean = mean(train_set_cl$cost)
  cost_sd = sd(train_set_cl$cost)
  zeta = ifelse( is.na(cost_sd) , Inf , cost_mean / cost_sd)
  
  cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$cost_mean = cost_mean
  cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$cost_sd = cost_sd
  cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$zeta = zeta
  cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$num_train = nrow(train_set_cl)
  cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$num_test = nrow(test_set_cl)
  
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
  cat(">>> train_set after encoding:",ncol(train_set_cl)," - test_set after encoding:",ncol(test_set_cl)," ... \n")
  
  ## y, data 
  y = train_set_cl$cost   
  train_set_cl[, 'cost'] = NULL 
  
  ####### remove zero variance predictors   
  l = featureSelect (train_set_cl,
                     test_set_cl,
                     removeOnlyZeroVariacePredictors=T,
                     performVarianceAnalysisOnTrainSetOnly = T , 
                     removePredictorsMakingIllConditionedSquareMatrix = F, 
                     removeHighCorrelatedPredictors = F, 
                     featureScaling = F)
  train_set_cl = l$traindata
  test_set_cl = l$testdata
  
  #######  DEBUG
  if (DEBUG_MODE) {
    cat(">>> Debug mode ... \n")
    train_set_cl = train_set_cl[1:20,]
    y = y[1:20]
  } else {
    cat(">>> Production mode ... \n")
  }
  #######  end of DEBUG 

  ##############
  ## MODELING 
  ##############

  # 8-fold repteated 3 times 
  controlObject <- trainControl(method = "repeatedcv", repeats = 3, number = 8)
 
  # model list 
  if (DO_PLOT) { 
    model_list = list()
    model_list[ reg_models ] = list(NULL)
  }

  # best model params 
  pred_best = rep(NA,length(pred_cl_idx))
  mod_best = NULL
  rmse_best = NULL 
  
  #######  parallel loop over models  
  ll = mclapply( seq_along(reg_models) , function(m) {
    model.label = reg_models[m]
    #cat(">>> Training ",model.label,"...\n")
    
    # model perf / pred 
    rmse_xval_mod = NULL
    secs_mod = NULL 
    pred_mod = NULL 
    
    l = NULL 
    if (model.label == "XGBoost") {
      early.stop = xgb_grid[xgb_grid$cluster_lev==cluster_levs[cl],]$early.stop
      #cat(">>> XGBoost: using nrounds = ",early.stop,"...\n")
      l = reg_train_predict ( YtrainingSet = y , 
                              XtrainingSet = train_set_cl, 
                              testSet = test_set_cl, 
                              model.label = model.label, 
                              controlObject = controlObject, 
                              best.tuning = T, 
                              
                              tuneGrid = expand.grid(
                                nrounds = early.stop, 
                                max_depth = 20,  
                                eta = 0.05 ),
                      
                              objective = "reg:linear",
                              eval_metric = "rmse", 
                              gamma = 0.7,  
                              subsample = 0.5 , ## suggested in ESLII
                              nthread = 10, 
                              min_child_weight = 1 , 
                              colsample_bytree = 0.5, 
                              max_delta_step = 1
                              )
    } else {
      l = reg_train_predict ( YtrainingSet = y , 
                              XtrainingSet = train_set_cl, 
                              testSet = test_set_cl, 
                              model.label = model.label, 
                              controlObject = controlObject, 
                              best.tuning = T)
    }
    
    if ( !is.null(l$model) && length(l$model$results$RMSE) > 1 ) {
      rmse_xval_mod = min(l$model$results$RMSE)
    } else if (! is.null(l$model) ) {
      rmse_xval_mod = l$model$results$RMSE
    } else {
      rmse_xval_mod = 1000000
    }
    secs_mod = l$secs 
    pred_mod = l$pred
    
    if (DO_PLOT) {
      return(list(pred_mod = pred_mod , 
                  model.label = model.label , 
                  model=l$model , 
                  rmse_xval_mod = rmse_xval_mod , 
                  secs_mod = secs_mod))
    } else {
      return(list(pred_mod = pred_mod , 
                  model.label = model.label , 
                  rmse_xval_mod = rmse_xval_mod , 
                  secs_mod = secs_mod))
    }
  },
  mc.cores = length(reg_models))
  ####### end of parallel loop 

  ## updates 
  for (jj in seq_along(ll)) {
    
    # update grid 
    cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] , ll[[jj]]$model.label ] = ll[[jj]]$rmse_xval_mod
    cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] , paste0(ll[[jj]]$model.label,'_secs') ] = ll[[jj]]$secs_mod
    
    # update best 
    if (is.null(rmse_best) || rmse_best > ll[[jj]]$rmse_xval_mod) {
      pred_best = ll[[jj]]$pred_mod
      mod_best = ll[[jj]]$model.label
      rmse_best = ll[[jj]]$rmse_xval_mod 
    }
    
    # update model_list 
    if (DO_PLOT && is.null(ll[[jj]]$model))  {
      model_list[ ll[[jj]]$model.label ] = NULL
    } else if (DO_PLOT) {
      model_list[ ll[[jj]]$model.label ] = list(ll[[jj]]$model)
    }
  }

  ## update pred / grid 
  pred[pred_cl_idx] = pred_best
  cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$rmse_xval = rmse_best
  cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$winner = mod_best

  ## save model for plotting  
  if (DO_PLOT) {
    model_list_by_cluster[cl] = list(model_list)
  }
} 
tm = proc.time() - ptm
secs = as.numeric(tm[3])
cat(">>> MAIN LOOP >>> Time elapsed:",secs," secs. [",secs/60,"min.] [",secs/(60*60),"hours] \n")
############## end of MAIN LOOP 

## use predictions from best_prediction for imputing NAs 
cat(">>> using predictions from best_prediction for imputing NAs  ... \n")
pred[which(is.na(pred))] = best_prediction[which(is.na(pred)),]$cost

## some basic check 
stopifnot(sum(is.na(pred)) == 0) 
pred_real = pred
pred = ifelse(pred<0,1.5,pred)

cat('>> number of prediction < 0:',sum(pred_real<0),' ... repleaced with 1.5 \n')

## write on disk 
cat(">> writing prediction / cluster_perf on disk ... \n")

sample_submission$cost = pred 
write.csv(sample_submission,quote=FALSE, 
          file=paste(getBasePath("submission"),'sub_cluster_by_qty_ensemble.csv',sep='') ,
          row.names=FALSE)

write.csv(cluster_perf,quote=FALSE, 
          file=paste(getBasePath("submission"),'cluster_lev_1_cluster_by_qty_ensemble.csv',sep='') ,
          row.names=FALSE)

## now, let's plotting ...
if (DO_PLOT) {
  cat(">> Now, let's plotting ... \n")
  Main_Did = c( 'qty [1,2]'    , 'qty (2,5]'   , 'qty (5,9]'   , 
                'qty (9,14]'   , 'qty (14,24]' , 'qty (24,48]' , 
                'qty (48,100]' , 'qty (100,2500]' )
  
  parallelplot_RMSE = list()
  parallelplot_Rsquared = list()
  dotplot_Default = list() 
  
  for (cl in seq_along(cluster_levs)) {
    model_list = model_list_by_cluster[[cl]] 
    
    if (cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$winner != "LinearReg") {
      model_list[ "LinearReg" ] = NULL ## probably out of bounds  
    }
    
    allResamples <- resamples(model_list)
    
    # parallelplot
    parallelplot_RMSE[[cl]] = parallelplot(allResamples , 
                                           #sub="with log scales" , 
                                           main = Main_Did[cl] )
    trellis.device(device="jpeg", filename=paste(getBasePath("docs"),"/images/parallelplot_RMSE___dump__",cl,".jpg",
                                                 sep=''))
    print(parallelplot_RMSE[[cl]])
    dev.off()
    
    # parallelplot_Rsquared
    parallelplot_Rsquared[[cl]] = parallelplot(allResamples , metric = "Rsquared" , 
                                               #sub="with log scales" , 
                                               main = Main_Did[cl] )
    trellis.device(device="jpeg", filename=paste(getBasePath("docs"),"/images/parallelplot_Rsquared___dump__",cl,".jpg",
                                                 sep=''))
    print(parallelplot_Rsquared[[cl]])
    dev.off()
    
    # dotplot
    dotplot_Default[[cl]] = dotplot(allResamples , metric = "RMSE" , main = Main_Did[cl])
    trellis.device(device="jpeg", filename=paste(getBasePath("docs"),"/images/dotplot___dump__",cl,".jpg",
                                                 sep=''))
    print(dotplot_Default[[cl]])
    dev.off()
  }
  
  # parallelplot_RMSE_1
  trellis.device(device="jpeg", filename=paste0(getBasePath("docs"),"/images/parallelplot_RMSE_1.jpg"))
  final_plot = grid.arrange(parallelplot_RMSE[[1]],parallelplot_RMSE[[2]], 
                            parallelplot_RMSE[[3]],parallelplot_RMSE[[4]], 
                            ncol=2, 
                            nrow=2)
  dev.off()
  
  # parallelplot_RMSE_2
  trellis.device(device="jpeg", filename=paste0(getBasePath("docs"),"/images/parallelplot_RMSE_2.jpg"))
  final_plot = grid.arrange(parallelplot_RMSE[[5]],parallelplot_RMSE[[6]],
                            parallelplot_RMSE[[7]],parallelplot_RMSE[[8]],
                            ncol=2, 
                            nrow=2)
  dev.off()
  
  # parallelplot_Rsquared_1
  trellis.device(device="jpeg", filename=paste0(getBasePath("docs"),"/images/parallelplot_Rsquared_1.jpg"))
  final_plot = grid.arrange(parallelplot_Rsquared[[1]],parallelplot_Rsquared[[2]], 
                            parallelplot_Rsquared[[3]],parallelplot_Rsquared[[4]], 
                            ncol=2, 
                            nrow=2)
  dev.off()
  
  # parallelplot_Rsquared_2
  trellis.device(device="jpeg", filename=paste0(getBasePath("docs"),"/images/parallelplot_Rsquared_2.jpg"))
  final_plot = grid.arrange(parallelplot_Rsquared[[5]],parallelplot_Rsquared[[6]], 
                            parallelplot_Rsquared[[7]],parallelplot_Rsquared[[8]], 
                            ncol=2, 
                            nrow=2)
  dev.off()
  
  # dotplot_Default_1
  trellis.device(device="jpeg", filename=paste0(getBasePath("docs"),"/images/dotplot_Default_1.jpg"))
  final_plot = grid.arrange(dotplot_Default[[1]],dotplot_Default[[2]], 
                            dotplot_Default[[3]],dotplot_Default[[4]], 
                            ncol=2, 
                            nrow=2)
  dev.off()
  
  # dotplot_Default_2
  trellis.device(device="jpeg", filename=paste0(getBasePath("docs"),"/images/dotplot_Default_2.jpg"))
  final_plot = grid.arrange(dotplot_Default[[5]],dotplot_Default[[6]], 
                            dotplot_Default[[7]],dotplot_Default[[8]], 
                            ncol=2, 
                            nrow=2)
  dev.off()
}

