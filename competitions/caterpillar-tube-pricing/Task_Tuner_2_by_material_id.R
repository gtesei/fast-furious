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
getData = function() {
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
  
  return(structure(list(
    train_set = train_set , 
    test_set = test_set 
  )))
}

getMetaData = function() {
  cat(">>> Loading meta-features from ",ff.getPath('ensemble')," ... \n")
  mf = list.files( ff.getPath('ensemble') )
  # if (useOnly4Plus) {
  #   cat(">>> using option useOnly4Plus ... \n")
  #   mf = list.files( ff.getPath('ensemble') , pattern = '4_*|8_*')
  # }
  print(mf)
  meta_list = lapply(mf,function(x) {
    data = as.data.frame( fread(paste0(ff.getPath("ensemble") , x)))
    train_meta = data[1:train_obs,'assemble']
    test_meta = data[(train_obs+1):(train_obs+test_obs),'assemble']
    meta_name = strsplit(x = x , split = '\\.')[[1]][1]
    return( list(train_meta = train_meta , test_meta = test_meta, meta_name=meta_name) )
  })
  
  train_meta = as.data.frame(matrix(rep(-1,train_obs*length(mf)),ncol = length(mf)))
  test_meta = as.data.frame(matrix(rep(-1,test_obs*length(mf)),ncol = length(mf)))
  
  for (i in seq_along(meta_list)) {
    train_meta[,i] = meta_list[[i]]$train_meta
    test_meta[,i] = meta_list[[i]]$test_meta
    
    colnames(train_meta)[i] = meta_list[[i]]$meta_name
    colnames(test_meta)[i] = meta_list[[i]]$meta_name
  }
  
  stopifnot( sum(train_meta == -1 , na.rm = T) == 0 )
  stopifnot( sum(test_meta == -1 , na.rm = T) == 0  )
  
  return(structure(list(
    train_meta = train_meta, 
    test_meta = test_meta 
  )))
}
################# FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/caterpillar-tube-pricing/competition_data')
ff.bindPath(type = 'elab' , sub_path = 'dataset/caterpillar-tube-pricing/elab')
ff.bindPath(type = 'process' , sub_path = 'data_process')

ff.bindPath(type = 'ensemble' , sub_path = 'dataset/caterpillar-tube-pricing/ensemble_1') ## in 

ff.bindPath(type = 'best_tune' , sub_path = 'dataset/caterpillar-tube-pricing/best_tune_2',createDir = TRUE) ## out 
ff.bindPath(type = 'submission' , sub_path = 'dataset/caterpillar-tube-pricing/pred_ensemble_1',createDir = TRUE) ## out 

ff.setMaxCuncurrentThreads(16)

source(paste0( ff.getPath("process") , "/Regression_Lib.R"))

################# SETTINGS
DEBUG_MODE = F
useOnly4Plus = F

################# MODELS 
modelList = list(
  
  #list(model.label = 'cubist'  , tune=T, useOnlyFeat=T, predFile = 'material_id_cubist_useOnlyFeat.csv'), 
  #list(model.label = 'xgbTree'  , tune=T, useOnlyFeat=T, predFile = 'material_id_xgbTree_useOnlyFeat.csv'),
  list(model.label = 'knn'  , tune=T, useOnlyFeat=T, predFile = 'material_id_knn_useOnlyFeat.csv'), 
  list(model.label = 'gbm'  , tune=T, useOnlyFeat=T, predFile = 'material_id_gbm_useOnlyFeat.csv')

)


################# build features set  
sample_submission = as.data.frame( fread(paste(ff.getPath("data") , 
                                               "sample_submission.csv" , sep=''))) 

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
dl = getData()
train_set = dl$train_set
test_set = dl$test_set

train_obs = nrow(train_set)
test_obs = nrow(test_set)

cat(">>> Train observations:",train_obs," - test observations:",test_obs,"\n")

## meta-features 
ml = getMetaData()
train_meta = ml$train_meta
test_meta = ml$test_meta

## clusters 
clusters = unique(sort(ddply(test_set , .(material_id) , function(x) c(num=dim(x)[1]))$material_id))

##############
## MAIN LOOP 
##############
ptm <- proc.time()
res_lists = lapply( seq_along(modelList) , function(m) { 
  cat(">>> now processing:\n")
  print(modelList[[m]])
  
  ## make train set / test set 
  if(length(modelList[[m]]$useOnlyFeat)>0 && modelList[[m]]$useOnlyFeat) {
    train_meta = train_set
    test_meta = test_set
  } else if (modelList[[m]]$useALLFeat) {
    train_meta = cbind(train_meta,train_set)
    test_meta = cbind(test_meta,test_set)
  } else if (modelList[[m]]$useQty) {
    train_meta = cbind(train_meta,quantity = train_set$quantity)
    test_meta = cbind(test_meta,quantity = test_set$quantity)
    
    ##
    train_meta = cbind(train_meta, train_set[,'material_id' , drop=F] )
    test_meta = cbind(test_meta, test_set[,'material_id' , drop=F])
    
    train_meta = cbind(train_meta,cost = train_set$cost)
  } else {
    ##
    train_meta = cbind(train_meta, train_set[,'material_id' , drop=F] )
    test_meta = cbind(test_meta, test_set[,'material_id' , drop=F])
    
    train_meta = cbind(train_meta,cost = train_set$cost)
  }
  
  mc_cores = min( length(clusters), ff.getMaxCuncurrentThreads())
  cat(">>> setting mc.cores <--",mc_cores,"...\n")
  res_list = parallel::mclapply( seq_along(clusters) , function(i) {
    cl = clusters[i]
    model.label = modelList[[m]]$model.label
    
    ###
    pid = paste('[cluster:',cl,'/',length(clusters),'][model:',model.label,']',sep='')
    cat('>>> processing ',pid,'... \n')
    
    ## define train / test set 
    train_set_cl = train_meta[train_meta$material_id == cl,]
    test_set_cl = test_meta[test_meta$material_id== cl,]
    cat(pid,'>>> train observations:',nrow(train_set_cl), '- test observations:',nrow(test_set_cl), ' \n')
    if (nrow(train_set_cl) == 0) stop('something wrong in train_set_cl') 
    if (nrow(test_set_cl) == 0) stop('something wrong in test_set_cl') 
    
    predTest_cl_idx = which(test_meta$material_id==cl)
    stopifnot ( length(predTest_cl_idx) == nrow(test_set_cl) ) 
    
    predTrain_cl_idx = which(train_meta$material_id==cl)
    stopifnot ( length(predTrain_cl_idx) == nrow(train_set_cl) ) 
    
    ## remove NAs columns 
    getNAColumns = function(data) {
      return(which(unlist(lapply(train_set_cl,function(x) {
        sum(is.na(x)>0)
      })) > 0))
    }
    naIdx = unique(c(getNAColumns(train_set_cl),getNAColumns(test_set_cl)))
    cat(">>> removing NAs columns:",naIdx,"--",colnames(train_set_cl)[naIdx],"\n")
    if (length(naIdx)>0) {
      train_set_cl = train_set_cl[,-naIdx]
      test_set_cl = test_set_cl[,-naIdx]
    }
    
    ##############
    ## DATA PROC 
    ##############
    
    ## tube_assembly_id , id 
    train_set_cl[, 'tube_assembly_id'] = NULL 
    test_set_cl [, 'tube_assembly_id'] = NULL 
    test_set_cl [, 'id'] = NULL 
    
    ## material_id 
    train_set_cl[, 'material_id'] = NULL 
    test_set_cl [, 'material_id'] = NULL 
    
    cat(pid,">>> train_set after encoding:",ncol(train_set_cl)," - test_set after encoding:",ncol(test_set_cl)," ... \n")
    
    ## y, data 
    y = train_set_cl$cost   
    train_set_cl[, 'cost'] = NULL 
    
    ####### remove zero variance predictors   
    l = ff.featureFilter (train_set_cl,
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
      train_set_cl = train_set_cl[,1:2]
      #y = y[1:100]
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
    bestTune = NULL
    ### 
    
    # 8-fold repteated 3 times 
    controlObject <- trainControl(method = "repeatedcv", repeats = 3, number = 8)
    l = ff.regTrainAndPredict ( Ytrain = y , 
                                Xtrain = train_set_cl, 
                                Xtest = test_set_cl, 
                                model.label = model.label, 
                                controlObject = controlObject, 
                                best.tuning = T, 
                                removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = T)
    
    if ( !is.null(l$model) ) {
      rmse_xval_mod = min(l$model$results$RMSE)
      bestTune = l$model$bestTune
      pred_mod = l$pred
    } else {
      #rmse_xval_mod = 1000000
      stop(paste('model',model.label,'got errors!!'))
    }
    secs_mod = l$secs 
    
    # output 
    return(list(cluster = cl,
                model.label = model.label,
                rmse_xval_mod = rmse_xval_mod,
                pred_mod = pred_mod, 
                predTest_cl_idx = predTest_cl_idx, 
                predTrain_cl_idx = predTrain_cl_idx, 
                bestTune = bestTune, 
                secs_mod = secs_mod))
    
  } , mc.cores = mc_cores )
  
  ## re-assemble prediction 
  predTest = rep(NA,test_obs)
  cat(">>> re-assembling prediction and writing on disk ... \n")
  for (i in seq_along(res_list)) {
    if ( ! is.numeric(res_list[[i]]$pred_mod) ) {
      stop('predictions are not numeric')
    }
    predTest[res_list[[i]]$predTest_cl_idx] = res_list[[i]]$pred_mod
  }
  stopifnot( sum(is.na(predTest))  == 0 )
  write.csv(data.frame(id = sample_submission$id , cost=predTest),
            quote=FALSE, 
            file=paste(ff.getPath("submission"), modelList[[m]]$predFile ,sep='') ,
            row.names=FALSE)
  
  ## re-assemble bestTune  
  cat(">>> re-assembling bestTune and writing on disk ... \n")  
  tuneGrid = NULL 
  for (i in seq_along(res_list)) {
    if(is.null(tuneGrid)) {
      tuneGrid = data.frame(material_id=clusters,
                            model=modelList[[m]]$model.label,
                            secs=NA,
                            rmse=NA) 
      pars = as.data.frame(
        matrix(rep(NA,ncol(res_list[[i]]$bestTune)*length(clusters)),ncol = ncol(res_list[[i]]$bestTune)))
      colnames(pars) = colnames(res_list[[i]]$bestTune)
      tuneGrid = cbind(tuneGrid,pars)
    } 
    tuneGrid[tuneGrid$material_id==res_list[[i]]$cluster,]$secs = res_list[[i]]$secs_mod
    tuneGrid[tuneGrid$material_id==res_list[[i]]$cluster,]$rmse = res_list[[i]]$rmse_xval_mod
    tuneGrid[tuneGrid$material_id==res_list[[i]]$cluster,5:ncol(tuneGrid)] = res_list[[i]]$bestTune
  }
  write.csv(tuneGrid,
            quote=FALSE, 
            file=paste(ff.getPath("best_tune"), modelList[[m]]$predFile ,sep='') ,
            row.names=FALSE)
})
####### end of parallel loop 
tm = proc.time() - ptm
secs = as.numeric(tm[3])
cat(">>> MAIN LOOP >>> Time elapsed:",secs," secs. [",secs/60,"min.] [",secs/(60*60),"hours] \n")
############## end of MAIN LOOP 


