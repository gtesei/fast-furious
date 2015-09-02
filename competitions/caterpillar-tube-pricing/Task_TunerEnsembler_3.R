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
  
  data = as.vector(c(predictor.train,predictor.test))
  
  if (num_bids>8) {
    num_bids = 20
    split_16 = as.numeric(cut2(data, g=num_bids))
    
    return( list(levels.train = split_16[1:length(predictor.train)] , levels.test = split_16[(length(predictor.train)+1):length(data)] , 
                 theresolds = NULL) ) 
  } else {
    ## clustering by quantity 
    if (verbose) {
      print(describe(predictor.train))
      print(describe(predictor.test))
    }
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
}
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

getMetaData = function(ensemble) {
  cat(">>> Loading meta-features from ",ff.getPath(ensemble)," ... \n")
  mf = list.files( ff.getPath(ensemble) )
  # if (useOnly4Plus) {
  #   cat(">>> using option useOnly4Plus ... \n")
  #   mf = list.files( ff.getPath('ensemble') , pattern = '4_*|8_*')
  # }
  print(mf)
  meta_list = lapply(mf,function(x) {
    data = as.data.frame( fread(paste0(ff.getPath(ensemble) , x)))
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

validateModelList = function(modelList) {
  lt = length(modelList)
  stopifnot(lt > 0)
  for (i in seq_along(modelList)) {
    cat (">> validating [",i,"/",lt,"] ... ")
    stopifnot(modelList[[i]]$layer == length(modelList[[i]]$useLayer))
    stopifnot( typeof(modelList[[i]]$useLayer) == 'logical')
    stopifnot( class(modelList[[i]]$cluster_levs) == 'numeric')
    stopifnot( ! is.null(modelList[[i]]$model.label) )
    stopifnot( ! is.null(modelList[[i]]$tune) )
    
    ## predFile 
    useQty = ifelse(!is.null(modelList[[i]]$useQty) && modelList[[i]]$useQty , '_useQty' , '')
    removeHCP = ifelse(!is.null(modelList[[i]]$removeHCP) && modelList[[i]]$removeHCP , '_removeHCP' , '')
    
    lr = paste(unlist(lapply(modelList[[i]]$useLayer , function(x) {
      if (x) return ('_1')
      else return ('_0')
    })) , collapse = '')
    fn = paste(modelList[[i]]$cluster_levs,'_',modelList[[i]]$model.label,useQty,removeHCP,lr,'.csv',sep='')
    if (   (! is.null(modelList[[i]]$predFile)) && (! identical(fn,modelList[[i]]$predFile)) ) {
      cat("replacing predFile [",modelList[[i]]$predFile,"] with ",fn,"..")
    }
    modelList[[i]]$predFile = fn 
    cat("... OK \n")
  }
  
  return(modelList)
}

processClusterData = function (train_set,
                               test_set,
                               cl, 
                               model.label,
                               cluster_levs, 
                               useMaterialId,
                               removeHCP,
                               debug_mode) {
  
  pid = paste('[cluster:',cl,'/',cluster_levs,'][model:',model.label,']',sep='')
  cat('>>> processing ',pid,'... \n')
  
  train_set_cl = train_set[train_set$qty_lev == cl,]
  test_set_cl = test_set[test_set$qty_lev== cl,]
  
  cat(pid,'>>> train observations:',nrow(train_set_cl), '- test observations:',nrow(test_set_cl), ' \n')
  if (nrow(train_set_cl) == 0) stop('something wrong in train_set_cl') 
  if (nrow(test_set_cl) == 0) stop('something wrong in test_set_cl') 
  
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
  if (useMaterialId) {
    cat(">>> encoding material_id [",
        unique(c(train_set_cl$material_id , test_set_cl$material_id)),"] [",length(unique(c(train_set_cl$material_id , 
                                                                                            test_set_cl$material_id))),"] ... \n")
    l = ff.encodeCategoricalFeature (train_set_cl$material_id , test_set_cl$material_id , colname.prefix = "material_id" , asNumeric=F)
    train_set_cl = cbind(train_set_cl , l$traindata)
    test_set_cl = cbind(test_set_cl , l$testdata)
  }
  train_set_cl[, 'material_id'] = NULL 
  test_set_cl [, 'material_id'] = NULL 
  
  ## y, data 
  y = train_set_cl$cost   
  train_set_cl[, 'cost'] = NULL 
  
  ## remove NAs columns 
  getNAColumns = function(data) {
    return(which(unlist(lapply(train_set_cl,function(x) {
      sum(is.na(x)>0)
    })) > 0))
  }
  naIdx = unique(c(getNAColumns(train_set_cl),getNAColumns(test_set_cl)))
  naCols = colnames(train_set_cl)[naIdx]
  cat(">>> removing NAs columns:",naIdx,"--",naCols,"\n")
  cat(pid,">>> train_set before:",ncol(train_set_cl)," - test_set before:",ncol(test_set_cl)," ... \n")
  if (length(naIdx)>0) {
    train_set_cl[,c(naCols)] = NULL
    test_set_cl[ ,c(naCols)] = NULL
  }
  
  cat(pid,">>> train_set after:",ncol(train_set_cl)," - test_set after:",ncol(test_set_cl)," ... \n")
  
  if (ncol(train_set_cl) != ncol(test_set_cl)) {
    print(colnames(train_set_cl)[unlist(lapply(colnames(train_set_cl), function(x){
      return(! x %in% colnames(test_set_cl) )
    }))])
  }
  stopifnot(ncol(train_set_cl) == ncol(test_set_cl))
  
  ####### remove zero variance predictors   
  l = ff.featureFilter (train_set_cl,
                        test_set_cl,
                        removeOnlyZeroVariacePredictors=T, ### <<< :::: <<< ---------
                        performVarianceAnalysisOnTrainSetOnly = T , 
                        removePredictorsMakingIllConditionedSquareMatrix = F, 
                        removeHighCorrelatedPredictors = removeHCP, 
                        featureScaling = F)
  train_set_cl = l$traindata
  test_set_cl = l$testdata
  
  #######  DEBUG
  if (debug_mode) {
    cat(pid,">>> Debug mode ... \n")
    train_set_cl = train_set_cl[,1:5]
    test_set_cl = test_set_cl[,1:5]
    #y = y[1:100]
  } else {
    cat(pid,">>> Production mode ... \n")
  }
  #######  end of DEBUG 
  
  return(list(
    train_set_cl = train_set_cl , 
    test_set_cl = test_set_cl , 
    predTrain_cl_idx = predTrain_cl_idx , 
    predTest_cl_idx = predTest_cl_idx, 
    ytrain = y
    ))
}

################# FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/caterpillar-tube-pricing/competition_data')
ff.bindPath(type = 'elab' , sub_path = 'dataset/caterpillar-tube-pricing/elab')
ff.bindPath(type = 'process' , sub_path = 'data_process')

ff.bindPath(type = 'ensemble_1' , sub_path = 'dataset/caterpillar-tube-pricing/ensemble_1') ## in 
ff.bindPath(type = 'ensemble_2' , sub_path = 'dataset/caterpillar-tube-pricing/ensemble_2') ## in 

ff.bindPath(type = 'best_tune_3' , sub_path = 'dataset/caterpillar-tube-pricing/best_tune_3',createDir = TRUE) ## out 
ff.bindPath(type = 'submission' , sub_path = 'dataset/caterpillar-tube-pricing/pred_ensemble_2',createDir = TRUE) ## out 
ff.bindPath(type = 'ensemble_3' , sub_path = 'dataset/caterpillar-tube-pricing/ensemble_3',createDir = TRUE) ## out 

ff.setMaxCuncurrentThreads(16)

source(paste0( ff.getPath("process") , "/Regression_Lib.R"))

################# SETTINGS
DEBUG_MODE = F

################# MODELS 

modelList = list(
  
  #list(layer = 3 , useLayer = c(F,F,T) , cluster_levs = 1 , model.label = 'enet'  , tune=T, useQty = T , predFile = '1_enet_useQty_0_0_1.csv'),
  
  #list(layer = 3 , useLayer = c(F,F,T) , cluster_levs = 1 , model.label = 'knn'  , tune=T, useQty = T , predFile = '1_knn_useQty_0_0_1.csv'),
  list(layer = 3 , useLayer = c(T,T,T) , cluster_levs = 1 , model.label = 'knn'  , tune=T, useQty = T , removeHCP = T, predFile = '1_knn_useQty_1_1_1_removeHCP.csv'),
  list(layer = 3 , useLayer = c(F,T,T) , cluster_levs = 1 , model.label = 'knn'  , tune=T, useQty = T , removeHCP = T, predFile = '1_knn_useQty_0_1_1_removeHCP.csv'),
  
  list(layer = 3 , useLayer = c(F,F,T) , cluster_levs = 8 , model.label = 'xgbTree'  , tune=T, useQty = T , predFile = '8_xgbTree_useQty_0_0_1.csv'),
  list(layer = 3 , useLayer = c(F,F,T) , cluster_levs = 8 , model.label = 'cubist'  , tune=T, useQty = T , predFile = '8_cubist_useQty_0_0_1.csv'),
  
  list(layer = 3 , useLayer = c(F,F,T) , cluster_levs = 4 , model.label = 'xgbTree'  , tune=T, useQty = T , predFile = '4_xgbTree_useQty_0_0_1.csv'),
  list(layer = 3 , useLayer = c(F,F,T) , cluster_levs = 2 , model.label = 'xgbTree'  , tune=T, useQty = T , predFile = '2_xgbTree_useQty_0_0_1.csv')
  
)

modelList = validateModelList(modelList) 
  
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

train_obs = nrow(dl$train_set)
test_obs = nrow(dl$test_set)

cat(">>> Train observations:",train_obs," - test observations:",test_obs,"\n")
cat(">>> layer 1 <<<\n")
cat(">> train:",dim(dl$train_set) , '- set:',dim(dl$test_set) , "\n")

## meta-features 
ml = getMetaData('ensemble_1')
cat(">>> layer 2 <<<\n")
cat(">> train:",dim(ml$train_meta) , '- set:',dim(ml$test_meta) , "\n")

ml2 = getMetaData('ensemble_2')
cat(">>> layer 3 <<<\n")
cat(">> train:",dim(ml2$train_meta) , '- set:',dim(ml2$test_meta) , "\n")


## data 
data = list(
  list(layer = 1 , traindata = dl$train_set , testdata = dl$test_set ), 
  list(layer = 2 , traindata = ml$train_meta , testdata = ml$test_meta), 
  list(layer = 3 , traindata = ml2$train_meta , testdata = ml2$test_meta)
  )

##############
## MAIN LOOP 
##############
ptm <- proc.time()
for (m in  seq_along(modelList) ) { 
  cat(">>> now processing:\n")
  print(modelList[[m]])
  
  ##### process model settings 
  useQty = ifelse(!is.null(modelList[[m]]$useQty) && modelList[[m]]$useQty , TRUE , FALSE)
  removeHCP = ifelse(!is.null(modelList[[m]]$removeHCP) && modelList[[m]]$removeHCP , TRUE , FALSE)
  useMaterialId = ifelse(!is.null(modelList[[m]]$useMaterialId) && modelList[[m]]$useMaterialId , TRUE , FALSE)
  
  ## make train set / test set 
  train_set = NULL
  test_set = NULL 
  for (i in seq_along(modelList[[m]]$useLayer)) {
    if (modelList[[m]]$useLayer[i]) {
      if (is.null(train_set)) {
        train_set = data[[i]]$traindata
        test_set = data[[i]]$testdata
      } else {
        train_set = cbind(train_set,data[[i]]$traindata)
        test_set = cbind(test_set,data[[i]]$testdata)
      }
    }
  }
  stopifnot(! is.null(train_set) , ! is.null(test_set))
  
  # quantity 
  if (useQty) {
    if ( ! 'quantity' %in% colnames(train_set) ) {
      train_set = cbind(train_set , data[[1]]$traindata[,'quantity' , drop = F]) 
      test_set = cbind(test_set , data[[1]]$testdata[,'quantity' , drop = F]) 
    }
  }
  
  # material_id 
  if ( ! 'material_id' %in% colnames(train_set) ) {
    train_set = cbind(train_set , data[[1]]$traindata[,'material_id' , drop = F]) 
    test_set = cbind(test_set , data[[1]]$testdata[,'material_id' , drop = F]) 
  }
  
  # output variable 
  if ( ! 'cost' %in% colnames(train_set) ) {
    train_set = cbind(train_set , data[[1]]$traindata[,'cost' , drop = F]) 
  }
  
  ##### end of model settings 
  
  ## clustering 
  if (modelList[[m]]$cluster_levs > 1)  {
    cls = cluster_by(predictor.train=data[[1]]$traindata[,'quantity'],
                     predictor.test=data[[1]]$testdata[,'quantity'],
                     num_bids = modelList[[m]]$cluster_levs,
                     verbose=T)
    
    train_set$qty_lev = cls$levels.train
    test_set$qty_lev = cls$levels.test
    if (modelList[[m]]$cluster_levs != length(unique(c(cls$levels.train,cls$levels.test)))) {
      cat(">>> changing cluster_levs from ",modelList[[m]]$cluster_levs,"to ",length(unique(c(cls$levels.train,cls$levels.test))),"...\n")
      modelList[[m]]$cluster_levs = length(unique(c(cls$levels.train,cls$levels.test)))
    }
    
  } else {
    cat(">> only 1 cluster ...\n")
    train_set$qty_lev = 1
    test_set$qty_lev = 1
  }
  
  mc_cores = min( modelList[[m]]$cluster_levs, ff.getMaxCuncurrentThreads())
  cat(">>> setting mc.cores <--",mc_cores,"...\n")
  res_list = parallel::mclapply( 1:modelList[[m]]$cluster_levs , function(i) {
    cl = i
    model.label = modelList[[m]]$model.label
    
    ## define train / test set 
    cData = processClusterData (train_set=train_set,
                                test_set=test_set,
                                cl=cl,
                                model.label=model.label,
                                cluster_levs = modelList[[m]]$cluster_levs, 
                                useMaterialId=useMaterialId,
                                removeHCP=removeHCP,
                                debug_mode=DEBUG_MODE)
    
    train_set_cl = cData$train_set_cl 
    test_set_cl = cData$test_set_cl 
    predTrain_cl_idx = cData$predTrain_cl_idx 
    predTest_cl_idx = cData$predTest_cl_idx
    y = cData$ytrain
  
    ##############
    ## MODELING 
    ##############
    rmse_xval_mod = NULL 
    pred_mod = NULL
    secs_mod = NULL
    bestTune = NULL
    
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
      tuneGrid = data.frame(cluster=sort(unique(train_set$qty_lev)),
                            model=modelList[[m]]$model.label,
                            secs=NA,
                            rmse=NA) 
      pars = as.data.frame(
        matrix(rep(NA,ncol(res_list[[i]]$bestTune)*length(sort(unique(train_set$qty_lev)))),ncol = ncol(res_list[[i]]$bestTune)))
      colnames(pars) = colnames(res_list[[i]]$bestTune)
      tuneGrid = cbind(tuneGrid,pars)
    } 
    tuneGrid[tuneGrid$cluster==res_list[[i]]$cluster,]$secs = res_list[[i]]$secs_mod
    tuneGrid[tuneGrid$cluster==res_list[[i]]$cluster,]$rmse = res_list[[i]]$rmse_xval_mod
    tuneGrid[tuneGrid$cluster==res_list[[i]]$cluster,5:ncol(tuneGrid)] = res_list[[i]]$bestTune
  }
  write.csv(tuneGrid,
            quote=FALSE, 
            file=paste(ff.getPath("best_tune_3"), modelList[[m]]$predFile ,sep='') ,
            row.names=FALSE)
  
  ##############
  ## ENSEMB 
  ##############
  cat(">>> Ensebling ... \n")  
  bestTune = NULL
  if (modelList[[m]]$tune) {
    bestTune = as.data.frame( fread(paste(ff.getPath("best_tune_3") , modelList[[m]]$predFile , sep='')))
    stopifnot(nrow(bestTune)>0)
  } 
  
  #############################
  res_list = lapply( 1:modelList[[m]]$cluster_levs , function(i) { 
    cl = i
    model.label = modelList[[m]]$model.label
    
    ## define train / test set 
    cData = processClusterData (train_set=train_set,
                                test_set=test_set,
                                cl=cl,
                                model.label=model.label,
                                cluster_levs = modelList[[m]]$cluster_levs, 
                                useMaterialId=useMaterialId,
                                removeHCP=removeHCP,
                                debug_mode=DEBUG_MODE)
    
    train_set_cl = cData$train_set_cl 
    test_set_cl = cData$test_set_cl 
    predTrain_cl_idx = cData$predTrain_cl_idx 
    predTest_cl_idx = cData$predTest_cl_idx
    y = cData$ytrain
    
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
          stopifnot(nrow(tgrid)>0)
        }
        
        cat('>>> tgrid <<< \n')
        print(tgrid)
        cat(">>> model.label <<<\n")
        print(model.label)
        
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
  }) ## end lapply ensemble 

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
            file=paste(ff.getPath("ensemble_3"), modelList[[m]]$predFile ,sep='') ,
            row.names=FALSE)
}
####### end of loop 
tm = proc.time() - ptm
secs = as.numeric(tm[3])
cat(">>> MAIN LOOP >>> Time elapsed:",secs," secs. [",secs/60,"min.] [",secs/(60*60),"hours] \n")
############## end of MAIN LOOP 


