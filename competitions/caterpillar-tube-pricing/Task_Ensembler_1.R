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
ff.bindPath(type = 'best_tune' , sub_path = 'dataset/caterpillar-tube-pricing/best_tune_1')
ff.bindPath(type = 'ensemble' , sub_path = 'dataset/caterpillar-tube-pricing/ensemble_1')

ff.setMaxCuncurrentThreads(8)

source(paste0( ff.getPath("process") , "/FeatureSelection_Lib.R"))
source(paste0( ff.getPath("process") , "/Regression_Lib.R"))

################# SETTINGS
DEBUG_MODE = F

################# MODELS 
modelList = list(
  #  list(cluster_levs = 1 , model.label = 'xgbTree'	, tune=T, tuneFile = '1_xgbTree.csv' , outFile = '1_xgbTree.csv'), 
  
  #   list(cluster_levs = 2 , model.label = 'xgbTree'  , tune=T, tuneFile = '2_xgbTree.csv' , outFile = '2_xgbTree.csv'), 
  #   list(cluster_levs = 2 , model.label = 'cubist'  , tune=T, tuneFile = '2_cubist.csv' , outFile = '2_cubist.csv'), 
  #   list(cluster_levs = 4 , model.label = 'xgbTree'  , tune=T, tuneFile = '4_xgbTree.csv' , outFile = '4_xgbTree.csv'), 
  #   list(cluster_levs = 4 , model.label = 'cubist'  , tune=T, tuneFile = '4_cubist.csv' , outFile = '4_cubist.csv'),
  
  #   list(cluster_levs = 8 , model.label = 'xgbTree'  , tune=T, tuneFile = '8_xgbTree.csv' , outFile = '8_xgbTree.csv'),
  #   list(cluster_levs = 8 , model.label = 'cubist'  , tune=T, tuneFile = '8_cubist.csv' , outFile = '8_cubist.csv'),
  #   list(cluster_levs = 8 , model.label = 'pls'  , tune=T, tuneFile = '8_pls.csv' , outFile = '8_pls.csv'),
  #   list(cluster_levs = 8 , model.label = 'ridge'  , tune=T, tuneFile = '8_ridge.csv' , outFile = '8_ridge.csv'),
  #   list(cluster_levs = 8 , model.label = 'enet'  , tune=T, tuneFile = '8_enet.csv' , outFile = '8_enet.csv'),
  #   list(cluster_levs = 8 , model.label = 'knn'  , tune=T, tuneFile = '8_knn.csv' , outFile = '8_knn.csv'),
  #   list(cluster_levs = 8 , model.label = 'svmRadial'  , tune=T, tuneFile = '8_svmRadial.csv' , outFile = '8_svmRadial.csv'),
  #   list(cluster_levs = 8 , model.label = 'gbm'  , tune=T, tuneFile = '8_gbm.csv' , outFile = '8_gbm.csv'),
  #   list(cluster_levs = 8 , model.label = 'treebag'  , tune=F, tuneFile = NULL , outFile = '8_treebag.csv'),
  
  #list(cluster_levs = 8 , model.label = 'treebag'  , tune=F, tuneFile = NULL , matid = TRUE , outFile = '8_treebag_matid.csv' )
  
  #list(cluster_levs = 4 , model.label = 'treebag'  , tune=F, tuneFile = NULL , matid = TRUE , outFile = '4_treebag_matid.csv' )
  #list(cluster_levs = 4 , model.label = 'treebag'  , tune=F, tuneFile = NULL , outFile = '4_treebag.csv' ),
  #list(cluster_levs = 2 , model.label = 'treebag'  , tune=F, tuneFile = NULL , outFile = '2_treebag.csv' )
  #list(cluster_levs = 1 , model.label = 'treebag'  , tune=F, tuneFile = NULL , outFile = '1_treebag.csv' )
  
  list(cluster_levs = 2 , model.label = 'treebag'  , tune=F, tuneFile = NULL , matid = TRUE , outFile = '2_treebag_matid.csv' ),
  list(cluster_levs = 1 , model.label = 'treebag'  , tune=F, tuneFile = NULL , matid = TRUE , outFile = '1_treebag_matid.csv' )
)


################# DATA IN 

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
ptm <- proc.time()
##############
## MAIN LOOP 
##############
res_lists = lapply( seq_along(modelList) , function(m) { 
  
  cat(">>> now processing:\n")
  print(modelList[[m]])
  
  bestTune = NULL
  if (modelList[[m]]$tune) {
    bestTune = as.data.frame( fread(paste(ff.getPath("best_tune") , modelList[[m]]$tuneFile , sep='')))
  }
  
  ## clustering 
  if (modelList[[m]]$cluster_levs > 1)  {
    cls = cluster_by(predictor.train=train_set$quantity,
                     predictor.test=test_set$quantity,
                     num_bids = modelList[[m]]$cluster_levs,
                     verbose=T)
    
    train_set$qty_lev = cls$levels.train
    test_set$qty_lev = cls$levels.test
  } else {
    cat(">> only 1 cluster ...\n")
    train_set$qty_lev = 1
    test_set$qty_lev = 1
  }
  
  res_list = lapply( 1:modelList[[m]]$cluster_levs , function(i) { 
    cl = i
    model.label = modelList[[m]]$model.label
    
    ###
    pid = paste('[cluster:',cl,'/',modelList[[m]]$cluster_levs,'][model:',model.label,']',sep='')
    cat('>>> processing ',pid,'... \n')
    
    ## define train / test set 
    train_set_cl = train_set[train_set$qty_lev == cl,]
    test_set_cl = test_set[test_set$qty_lev== cl,]
    cat(pid,'>>> train observations:',nrow(train_set_cl), '- test observations:',nrow(test_set_cl), ' \n')
    if (nrow(test_set_cl) == 0) stop('something wrong') 
    
    predTest_cl_idx = which(test_set$qty_lev==cl)
    stopifnot ( length(predTest_cl_idx) == nrow(test_set_cl) ) 
    
    predTrain_cl_idx = which(train_set$qty_lev==cl)
    stopifnot ( length(predTrain_cl_idx) == nrow(train_set_cl) ) 
    
    ##############
    ## DATA PROC 
    ##############
    
    ## tube_assembly_id , id 
    train_set_cl[, 'tube_assembly_id'] = NULL 
    test_set_cl [, 'tube_assembly_id'] = NULL 
    test_set_cl [, 'id'] = NULL 
    
    ## material_id 
    if ( length(modelList[[m]]$matid)>0 && modelList[[m]]$matid ) {
        cat(">>> encoding material_id [",unique(c(train_set_cl$material_id , 
                                                  test_set_cl$material_id)),"] [",length(unique(c(train_set_cl$material_id , 
                                                                                                  test_set_cl$material_id))),"] ... \n")
        l = ff.encodeCategoricalFeature (train_set_cl$material_id , test_set_cl$material_id , colname.prefix = "material_id" , asNumeric=F)
        train_set_cl = cbind(train_set_cl , l$traindata)
        test_set_cl = cbind(test_set_cl , l$testdata)
    }
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
      train_set_cl = train_set_cl[,1:5]
      #y = y[1:100]
    } else {
      cat(pid,">>> Production mode ... \n")
    }
    #######  end of DEBUG 
    
    ##############
    ## ENSEMB 
    ##############
    
    if (modelList[[m]]$tune && sum(is.na(bestTune[bestTune$cluster == cl, 5:ncol(bestTune) , drop = F]))>0 ) {
      cat(pid,">>> no tune parameters ==> return ... ")
      return(list(ensemble = list(
        predTrain = rep(-1,length(predTrain_cl_idx)), 
        predTest = rep(-1,length(predTest_cl_idx))),
        predTest_cl_idx = predTest_cl_idx, 
        predTrain_cl_idx = predTrain_cl_idx))
      
    } else {
      
      nFolds = 4 
      #if (modelList[[m]]$cluster_levs<=2) nFolds = 4
      
      ens = NULL
      if (identical(model.label,'xgbTree')) {
        ens = ff.createEnsemble (Xtrain = train_set_cl,
                                 Xtest = test_set_cl,
                                 y = y,
                                 bestTune = expand.grid(
                                   nrounds = bestTune[bestTune$cluster == cl, ]$early.stop ,
                                   max_depth = 20,  
                                   eta = 0.05 ),
                                 caretModelName = model.label, 
                                 nfold=nFolds,
                                 parallelize = T,
                                 verbose = T , 
                                 
                                 objective = "reg:linear",
                                 eval_metric = "rmse", 
                                 gamma = 0.7,  
                                 subsample = 0.5 , ## suggested in ESLII
                                 nthread = 10, 
                                 min_child_weight = 1 , 
                                 colsample_bytree = 0.5, 
                                 max_delta_step = 1)
        
      } else {
        
        tgrid = NULL
        if (modelList[[m]]$tune) {
          tgrid = bestTune[bestTune$cluster == cl, 5:ncol(bestTune) , drop = F]
        }
        
        ens = ff.createEnsemble (Xtrain = train_set_cl,
                                 Xtest = test_set_cl,
                                 y = y,
                                 bestTune = tgrid,
                                 caretModelName = model.label, 
                                 nfold=nFolds,
                                 parallelize = T,
                                 verbose = T)
      }
      
      stopifnot ( length(predTrain_cl_idx) == length(ens$predTrain) )
      stopifnot ( length(predTest_cl_idx) == length(ens$predTest) ) 
      
      # output 
      return(list(ensemble = ens,
                  predTest_cl_idx = predTest_cl_idx, 
                  predTrain_cl_idx = predTrain_cl_idx)) 
    }
  })
  
  predTrain = rep(NA,nrow(train_set)) 
  predTest = rep(NA,nrow(test_set))
  
  cat(">>> re-assembling and writing on disk ... \n")
  for (i in seq_along(res_list)) {
    if ( (! is.numeric(res_list[[i]]$ensemble$predTrain)) || (! is.numeric(res_list[[i]]$ensemble$predTest)) ) {
      stop('predictions are not numeric')
    }
    predTrain[res_list[[i]]$predTrain_cl_idx] = res_list[[i]]$ensemble$predTrain
    predTest[res_list[[i]]$predTest_cl_idx] = res_list[[i]]$ensemble$predTest
  }
  stopifnot( sum(is.na(predTrain)) == 0 )
  stopifnot( sum(is.na(predTest))  == 0 )
  
  predTrain = ifelse(predTrain == -1 , NA , predTrain)
  predTest = ifelse(predTest == -1 , NA , predTest)
  
  
  ## assemble 
  assemble = c(predTrain,predTest)
  write.csv(data.frame(id = seq_along(assemble) , assemble=assemble),
            quote=FALSE, 
            file=paste(ff.getPath("ensemble"), modelList[[m]]$outFile ,sep='') ,
            row.names=FALSE)
})
####### end of parallel loop 
tm = proc.time() - ptm
secs = as.numeric(tm[3])
cat(">>> MAIN LOOP >>> Time elapsed:",secs," secs. [",secs/60,"min.] [",secs/(60*60),"hours] \n")
############## end of MAIN LOOP 


