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
getMetaData = function(ensembles_dir,report_fn=NULL,top_perc=NULL,dontTakeInReport=NULL,pickEnsembles = NULL) {
  
  ## ensembles_dir
  stopifnot(! is.null(ensembles_dir) , file.exists(ensembles_dir) )
  cat(">>> Loading meta-features from ",ensembles_dir," ... \n")
  mf = list.files( ensembles_dir )
  
  ## report_fn , top_perc
  if (! is.null(report_fn) && is.null(pickEnsembles)) {
    stopifnot(file.exists(report_fn) , ! is.null(top_perc) , is.numeric(top_perc) , top_perc > 0 , top_perc <= 1 )
    cat(">>> Loading kpi from ",report_fn," ... \n")
    kpi = as.data.frame( fread(report_fn))
    if (! is.null(dontTakeInReport) ) {
      kpi = kpi[-which(kpi$ensemble == dontTakeInReport),]
    } 
    stopifnot( all.equal(target = sort(kpi$ensemble), current = sort(mf)) )
    kpi = kpi[order(kpi$rmsle,decreasing = F),]
    kpi = kpi[1:floor(length(kpi$ensemble) * top_perc),]
    mf = kpi$ensemble
  }
  if (! is.null(pickEnsembles)) {
    cat(">>> Picking up the following ensemles ... \n")
    print(pickEnsembles)
    mf = pickEnsembles
  }
  
  cat(">>>>> Using following ensembles ... \n")
  print(mf)
  meta_list = lapply(mf,function(x) {
    data = as.data.frame( fread(paste0(ensembles_dir , x)))
    train_meta = data[1:train_obs,'assemble']
    test_meta = data[(train_obs+1):(train_obs+test_obs),'assemble']
    #meta_name = strsplit(x = x , split = '\\.')[[1]][1]
    meta_name = x
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
  clusters = ddply(data , .(tube_assembly_id) , function(x) c(num = nrow(x)))
  clusters = clusters[order(clusters$num , decreasing = T),]
  stopifnot(sum(clusters$num)==nrow(data)) 
  
  for (j in 1:repeats) {
    folds = list()
    for (i in 1:nFolds) {
      folds_i_name = paste0('Fold',i)
      folds[[folds_i_name]] = rep(NA_character_,nrow(data)) 
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
        folds[[folds_k_name]][idx_k] = clusters[idx,'tube_assembly_id'] 
        idx = idx + 1 
        if (idx > nrow(clusters)) break 
      }
      
      if (idx > nrow(clusters)) break 
      
      ## bw 
      for (k in length(seq):1) {
        folds_k_name = paste0('Fold',seq[k])
        idx_k = min(which(is.na(folds[[folds_k_name]])))
        folds[[folds_k_name]][idx_k] = clusters[idx,'tube_assembly_id'] 
        idx = idx + 1 
        if (idx > nrow(clusters)) break 
      }
    }
    
    ## remove NAs and convert to chars 
    for (k in seq_along(seq)) folds[[k]] = as.character(na.omit(folds[[k]]))
    
    ## union check 
    stopifnot(identical(intersect(clusters$tube_assembly_id , Reduce(union , folds) ) , clusters$tube_assembly_id)) 
    
    ## intersect check 
    samp = sample(1:nFolds,2,replace = F)
    stopifnot(length(intersect( folds[[samp[1]]] , folds[[samp[2]]]))==0) 
    
    ## refill nFolds 
    folds_j_name = paste0('folds.',j)
    for (k in 1:nFolds) {
      idx_k = which( data$tube_assembly_id %in% folds[[k]] )
      foldList[[folds_j_name]] [idx_k] = k
    }
    
    ## checks
    stopifnot(sum( is.na(foldList[[folds_j_name]]) ) == 0) 
    stopifnot( length(foldList[[folds_j_name]]) == nrow(data) ) 
    stopifnot(identical(intersect(unique(sort(foldList[[folds_j_name]])) , 1:nFolds), 1:nFolds))
  }
  
  return(foldList)
}

RMSLE = function(pred, obs) {
  #RMSE(pred = pred , obs = obs)
#  pen = 0 
  if (sum(pred<0)>0) {
#    pen = sum(pred<0)
    pred = ifelse(pred >=0 , pred , 1.5)
  }
  rmsle = sqrt(    sum( (log(pred+1) - log(obs+1))^2 )   / length(pred))
#  rmsle = rmsle + (pen*rmsle/10) 
  return (rmsle)
}

RMSLECostSummary <- function (data, lev = NULL, model = NULL) {
  c(postResample(data[, "pred"], data[, "obs"]),
    RMSLE = RMSLE(pred = data[, "pred"], obs = data[, "obs"]))
}

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
    useMaterialId = ifelse(!is.null(modelList[[i]]$useMaterialId) && modelList[[i]]$useMaterialId , '_useMaterialId' , '')
    logLabel = ifelse(!is.null(modelList[[i]]$logLabel) && modelList[[i]]$logLabel , '_logLabel' , '')
    nr2000 = ifelse(!is.null(modelList[[i]]$nr2000) && modelList[[i]]$nr2000 , '_nr2000' , '')
    nr3000 = ifelse(!is.null(modelList[[i]]$nr3000) && modelList[[i]]$nr3000 , '_nr3000' , '')
    nr4000 = ifelse(!is.null(modelList[[i]]$nr4000) && modelList[[i]]$nr4000 , '_nr4000' , '')
    nr5000 = ifelse(!is.null(modelList[[i]]$nr5000) && modelList[[i]]$nr5000 , '_nr5000' , '')
    power025 = ifelse(!is.null(modelList[[i]]$power025) && modelList[[i]]$power025 , '_power025' , '')
    power03 = ifelse(!is.null(modelList[[i]]$power03) && modelList[[i]]$power03 , '_power03' , '')
    power32 = ifelse(!is.null(modelList[[i]]$power32) && modelList[[i]]$power32 , '_power32' , '')
    
    useL2_10 = ifelse(!is.null(modelList[[i]]$useL2_10) && modelList[[i]]$useL2_10 , '_useL2_10' , '')
    pickEnsembles = ifelse(!is.null(modelList[[i]]$pickEnsembles) , '_pickEnsembles' , '')
    
    lr = paste(unlist(lapply(modelList[[i]]$useLayer , function(x) {
      if (x) return ('_1')
      else return ('_0')
    })) , collapse = '')
    fn = paste(modelList[[i]]$cluster_levs,'_',modelList[[i]]$model.label,useQty,removeHCP,useMaterialId,logLabel,nr2000,nr3000,nr4000,nr5000,power025,power03,power32,
               useL2_10,pickEnsembles,lr,'.csv',sep='')
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
  #train_set_cl[, 'tube_assembly_id'] = NULL 
  #test_set_cl [, 'tube_assembly_id'] = NULL 
  
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

ff.bindPath(type = 'ensemble_1' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/ensemble_1',createDir = TRUE) 
ff.bindPath(type = 'best_tune_1' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/best_tune_1',createDir = TRUE) 

## TODO load all bindings and use if else stmts in code depending on layer value 
ff.bindPath(type = 'ensemble_out' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/ensemble_2',createDir = TRUE) ## out 
ff.bindPath(type = 'best_tune_out' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/best_tune_2',createDir = TRUE) ## out 
ff.bindPath(type = 'submission' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/pred_ensemble_2',createDir = TRUE) ## out 

ff.setMaxCuncurrentThreads(16)

source(paste0( ff.getPath("process") , "/Regression_Lib.R"))

################# SETTINGS
DEBUG_MODE = F

################# MODELS 

modelList = list(
  
##############################################################################
#                                    1 LAYER                                 #
##############################################################################
  
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 8 , model.label = 'pls'  , tune=T, useQty = T , predFile = '8_pls.csv'),
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 4 , model.label = 'pls'  , tune=T, useQty = T , predFile = '4_pls.csv'),
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 2 , model.label = 'pls'  , tune=T, useQty = T , predFile = '2_pls.csv'),
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'pls'  , tune=T, useQty = T , predFile = '1_pls.csv'),
#      
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 2 , model.label = 'cubist'  , tune=T, useQty = T , predFile = '2_cubist.csv'), 
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 4 , model.label = 'cubist'  , tune=T, useQty = T , predFile = '4_cubist.csv'), 
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 8 , model.label = 'cubist'  , tune=T, useQty = T , predFile = '8_cubist.csv'), 
#   
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 8 , model.label = 'xgbTree'  , tune=T, useQty = T , predFile = '8_xgbTree.csv'),
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 2 , model.label = 'xgbTree'  , tune=T, useQty = T , predFile = '2_xgbTree.csv'), 
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 4 , model.label = 'xgbTree'  , tune=T, useQty = T , predFile = '4_xgbTree.csv'), 
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'xgbTree'  , tune=T, useQty = T , predFile = '1_xgbTree.csv'), 
#  
#   
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 8 , model.label = 'gbm'  , tune=T, useQty = T , predFile = '8_gbm.csv'),
#   
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 8 , model.label = 'knn'  , tune=T, useQty = T , predFile = '8_knn.csv'),
#   
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 8 , model.label = 'treebag'  , tune=F, useQty = T , useMaterialId = T , predFile = '8_treebag_useMaterialId.csv'),
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 8 , model.label = 'treebag'  , tune=F, useQty = T , predFile = '8_treebag.csv'),
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 4 , model.label = 'treebag'  , tune=F, useQty = T , useMaterialId = T , predFile = '4_treebag_useMaterialId.csv'),
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 4 , model.label = 'treebag'  , tune=F, useQty = T , predFile = '4_treebag.csv'),
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 2 , model.label = 'treebag'  , tune=F, useQty = T , useMaterialId = T , predFile = '2_treebag_useMaterialId.csv'),
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 2 , model.label = 'treebag'  , tune=F, useQty = T , predFile = '2_treebag.csv'),
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'treebag'  , tune=F, useQty = T , useMaterialId = T , predFile = '1_treebag_useMaterialId.csv'),

### still TODO 
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'treebag'  , tune=F, useQty = T , predFile = '1_treebag.csv'),   
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 8 , model.label = 'svmRadial'  , tune=T, useQty = T , predFile = '8_svmRadial.csv')
###### end of still TODO 

#   list(layer = 1 , useLayer = c(T) , cluster_levs = 8 , model.label = 'xgbTreeGTJ'  , tune=T, useQty = T , predFile = '8_xgbTreeGTJ.csv'),
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'xgbTreeGTJ'  , tune=T, useQty = T , predFile = '1_xgbTreeGTJ.csv'),
#   
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 8 , model.label = 'xgbTreeGTJ'  , tune=T, useQty = T , logLabel=T , predFile = '8_xgbTreeGTJ_logLabel.csv'),
  
#################  
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 8 , model.label = 'cubist'  , tune=T, logLabel=T , useQty = T , predFile = '8_cubist_useQty_logLabel.csv'), 
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 8 , model.label = 'cubist'  , tune=T, power025=T , useQty = T , predFile = '8_cubist_useQty_power025.csv'), 
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 4 , model.label = 'cubist'  , tune=T, logLabel=T , useQty = T , predFile = '4_cubist_useQty_logLabel.csv'), 
#   
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'xgbTreeGTJ'  , tune=T, nr2000=T, logLabel=T , predFile = '1_xgbTreeGTJ_logLabel_cv2000.csv'),
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'xgbTreeGTJ'  , tune=T, nr3000=T, logLabel=T , predFile = '1_xgbTreeGTJ_logLabel_cv3000.csv'),
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'xgbTreeGTJ'  , tune=T, nr4000=T, logLabel=T , predFile = '1_xgbTreeGTJ_logLabel_cv4000.csv'),
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'xgbTreeGTJ'  , tune=T, nr4000=T, power025=T , predFile = '1_xgbTreeGTJ_power025_cv4000.csv'), 
  
  
#################  
#  list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'xgbTreeGTJ'  , tune=T, useQty = T , logLabel=T , predFile = '1_xgbTreeGTJ_logLabel.csv'), 
  
  #### <<<<<<<<<<<<<<<<< :::::::: this shoube be re-do with better tune grid (more commitee and neigboors)  
#  list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'cubist'  , tune=T, useQty = T , logLabel=T , predFile = '1_cubist_logLabel.csv')
  ####
  
#################  
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'xgbTreeGTJ'  , tune=T, useQty = T , power025=T , predFile = '1_xgbTreeGTJ_power025.csv'), 
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'pls'  , tune=T, useQty = T , logLabel=T , predFile = '1_pls_logLabel.csv'), 
#   list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'knn'  , tune=T, useQty = T , logLabel=T , predFile = '1_knn_logLabel.csv')
  

# list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'ws' ,  predFile = '1_ws.csv' , tune=F , 
#      ensembles = c('1_xgbTreeGTJ_nr4000_power025_1.csv',
#                    '1_xgbTreeGTJ_logLabel_nr4000_1.csv',
#                    '1_xgbTreeGTJ_useQty_nr4000_power32_1.csv',
#                    '4_cubist_useQty_logLabel_1.csv') , 
#      weights = rep(1/4,4) , enseble_dir = 'ensemble_1' )

# list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'ws' ,  predFile = '1_ws.csv' , tune=F , 
#      ensembles = c('1_xgbTreeGTJ_useQty_nr4000_power32_1.csv',
#                    '1_xgbTreeGTJ_nr4000_power025_1.csv',
#                    '1_xgbTreeGTJ_useQty_power025_1.csv', 
#                    '1_xgbTreeGTJ_useQty_nr5000_power32_1.csv', 
#                    '1_xgbTreeGTJ_logLabel_nr4000_1.csv', 
#                    '1_xgbTreeGTJ_useQty_logLabel_cv5999_1.csv', 
#                    '1_xgbTreeGTJ_logLabel_nr3000_1.csv',
#                    '1_xgbTreeGTJ_useQty_logLabel_1.csv',
#                    '1_xgbTreeGTJ_logLabel_nr2000_1.csv',
#                    '1_xgbTreeGTJ_useQty_power03_1.csv', 
#                    '8_xgbTreeGTJ_useQty_logLabel_1.csv',
#                    '4_cubist_useQty_logLabel_1.csv',
#                    '4_cubist_useQty_logLabel_1.csv',
#                    '1_cubist_100_9_useQty_logLabel_1.csv',
#                    '8_cubist_useQty_logLabel_1.csv',
#                    '8_cubist_useQty_logLabel_1.csv',
#                    '8_cubist_useQty_power025_1.csv',
#                    '4_cubist_useQty_1.csv') , 
#      weights = rep(1/18,18) , enseble_dir = 'ensemble_1' )

#list(layer = 1 , useLayer = c(T) , cluster_levs = 1 , model.label = 'xgbTreeGTJ'  , tune=T, useQty = T , nr5000=T, power32=T , predFile = '1_xgbTreeGTJ_nr5000_power32.csv')

##############################################################################
#                                    2 LAYER                                 #
##############################################################################

#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'xgbTreeGTJ'  , tune=T,  useQty = T ,  logLabel=T , predFile = '1_xgbTreeGTJ_logLabel_0_1.csv'),
#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'xgbTreeGTJ'  , tune=T,  useQty = T ,   predFile = '1_xgbTreeGTJ_0_1.csv'),
#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 8 , model.label = 'xgbTreeGTJ'  , tune=T,  useQty = T ,   predFile = '8_xgbTreeGTJ_0_1.csv'),
#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 4 , model.label = 'xgbTreeGTJ'  , tune=T,  useQty = T ,   predFile = '4_xgbTreeGTJ_0_1.csv')

#########
#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'pls'  , tune=T, predFile = '1_pls_0_1.csv'), 
#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'cubist'  , tune=T, predFile = '1_cubist_0_1.csv'), 
#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'knn'  , tune=T, predFile = '1_knn_0_1.csv'), 
#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'xgbTreeGTJ'  , tune=T,   logLabel=T , predFile = '1_xgbTreeGTJ_logLabel_0_1.csv'),
#list(layer = 2 , useLayer = c(T,T) , cluster_levs = 1 , model.label = 'xgbTreeGTJ'  , tune=T,   logLabel=T , predFile = '1_xgbTreeGTJ_logLabel_1_1.csv')

#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , useL2_10=T, model.label = 'pls' , logLabel=T , tune=T, predFile = '1_pls_logLabel_useL2_10_0_1.csv')
#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , useL2_10=T, model.label = 'enet' , tune=T, predFile = '1_enet_useL2_10_0_1.csv')

#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , useL2_10=T, model.label = 'xgbTreeGTJ' , tune=T, predFile = '1_xgbTreeGTJ_useL2_10_0_1.csv'),
#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , useL2_10=T, model.label = 'cubist' , tune=T, predFile = '1_cubist_useL2_10_0_1.csv')

#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , useL2_10=T, model.label = 'enet' , tune=T, predFile = '1_enet_useL2_10_0_1.csv')

# list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'ws' , useQty = T ,  predFile = '1_ws_useQty_0_1.csv' , tune=F , 
#      ensembles = c('1_ws_1.csv',
#                    '1_ws_0_1.csv') , 
#      weights = rep(1/2,2) , enseble_dir = 'ensemble_out' )


# list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , pickEnsembles= c('1_xgbTreeGTJ_nr4000_power025_1.csv',
#                                                                     '1_xgbTreeGTJ_logLabel_nr4000_1.csv',
#                                                                     '1_xgbTreeGTJ_useQty_nr4000_power32_1.csv',
#                                                                     '4_cubist_useQty_logLabel_1.csv')
#        , model.label = 'enet' , tune=T, predFile = '1_enet_pickEnsembles_0_1.csv')

# list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , pickEnsembles= c('1_xgbTreeGTJ_nr4000_power025_1.csv',
#                                                                          '1_xgbTreeGTJ_logLabel_nr4000_1.csv',
#                                                                          '1_xgbTreeGTJ_useQty_nr4000_power32_1.csv',
#                                                                          '4_cubist_useQty_logLabel_1.csv')
#      , model.label = 'knn' , tune=T, predFile = '1_knn_pickEnsembles_0_1.csv')

#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'enet' , tune=T, predFile = '1_enet_0_1.csv')

# list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , pickEnsembles= c('1_xgbTreeGTJ_nr4000_power025_1.csv',
#                                                                          '1_xgbTreeGTJ_logLabel_nr4000_1.csv',
#                                                                          '1_xgbTreeGTJ_useQty_nr4000_power32_1.csv',
#                                                                          '4_cubist_useQty_logLabel_1.csv')
#      , model.label = 'cubist' , tune=T, predFile = '1_cubist_pickEnsembles_0_1.csv')

# list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'svmRadial'  , tune=T,   logLabel=T , predFile = '1_svmRadial_logLabel_0_1.csv'),
# list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'svmRadial'  , tune=T,   predFile = '1_svmRadial_0_1.csv')

# list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'rlm'  , useL2_10=T , tune=F,   predFile = '1_rlm_useL2_10_0_1.csv')


# list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'ws' , logLabel=T ,  predFile = '1_ws_logLabel_0_1.csv' , tune=F , 
#       ensembles = c('1_rlm_useL2_10_0_1.csv',
#                     '1_cubist_pickEnsembles_0_1.csv'
#                     #'1_ws_useQty_0_1.csv' , 
#                     #'1_ws_0_1.csv'
#                     ) , 
#       weights = rep(1/2,2) , enseble_dir = 'ensemble_out' )

# list(layer = 2 , useLayer = c(F,T) , cluster_levs = 4 , model.label = 'rlm'  , useL2_10=T , tune=F,   predFile = '4_rlm_useL2_10_0_1.csv')

#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'rlm'  , useL2_10=T , tune=F, predFile = '4_rlm_useL2_10_0_1.csv')


#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'bayesglm'  , useL2_10=T , tune=F,   predFile = '1_bayesglm_useL2_10_0_1.csv')
#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'bayesglm'  , tune=F,   predFile = '1_bayesglm_0_1.csv')

#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'glmStepAIC'  , useL2_10=T , tune=F,   predFile = '1_glmStepAIC_useL2_10_0_1.csv')

#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'glm'  , useL2_10=T , tune=F,   predFile = '1_glm_useL2_10_0_1.csv')
#list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'lm'  , useL2_10=T , tune=F,   predFile = '1_lm_useL2_10_0_1.csv')
list(layer = 2 , useLayer = c(F,T) , cluster_levs = 1 , model.label = 'svmLinear2'  , useL2_10=T , tune=F,   predFile = '1_svmLinear2_useL2_10_0_1.csv')


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
  logLabel = ifelse(!is.null(modelList[[m]]$logLabel) && modelList[[m]]$logLabel , TRUE , FALSE)
  nr2000 = ifelse(!is.null(modelList[[m]]$nr2000) && modelList[[m]]$nr2000 , TRUE , FALSE)
  nr3000 = ifelse(!is.null(modelList[[m]]$nr3000) && modelList[[m]]$nr3000 , TRUE , FALSE)
  nr4000 = ifelse(!is.null(modelList[[m]]$nr4000) && modelList[[m]]$nr4000 , TRUE , FALSE)
  nr5000 = ifelse(!is.null(modelList[[m]]$nr5000) && modelList[[m]]$nr5000 , TRUE , FALSE)
  power025 = ifelse(!is.null(modelList[[m]]$power025) && modelList[[m]]$power025 , TRUE , FALSE)
  power03 =ifelse(!is.null(modelList[[m]]$power03) && modelList[[m]]$power03 , TRUE , FALSE)
  power32 = ifelse(!is.null(modelList[[m]]$power32) && modelList[[m]]$power32 , TRUE , FALSE)
  useL2_10 = ifelse(!is.null(modelList[[m]]$useL2_10) && modelList[[m]]$useL2_10 , TRUE , FALSE)
  
  ##### meta-data 
  top_perc = 1 
  if(useL2_10) {
    top_perc = 0.4
  }
  ml = getMetaData(ensembles_dir = ff.getPath('ensemble_1') , 
                   ## TODO use ff.getPath
                   report_fn = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/caterpillar-tube-pricing/docs/redo_ensemble_diagnosys_1.csv', 
                   top_perc = top_perc, 
                   dontTakeInReport = c('best_greedy_8_clusters_1.csv'), 
                   pickEnsembles = modelList[[m]]$pickEnsembles)
  cat(">>> layer 2 <<<\n")
  cat(">> train:",dim(ml$train_meta) , '- set:',dim(ml$test_meta) , "\n")
  
  # ml2 = getMetaData('ensemble_2')
  # cat(">>> layer 3 <<<\n")
  # cat(">> train:",dim(ml2$train_meta) , '- set:',dim(ml2$test_meta) , "\n")
  
  
  ## data 
  data = list(
    list(layer = 1 , traindata = dl$train_set , testdata = dl$test_set )
  , list(layer = 2 , traindata = ml$train_meta , testdata = ml$test_meta)
    #   , list(layer = 3 , traindata = ml2$train_meta , testdata = ml2$test_meta)
  )
    
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
  
  # tube_assembly_id 
  if ( ! 'tube_assembly_id' %in% colnames(train_set) ) {
    train_set = cbind(train_set , data[[1]]$traindata[,'tube_assembly_id' , drop = F]) 
    test_set = cbind(test_set , data[[1]]$testdata[,'tube_assembly_id' , drop = F]) 
  }
  
  # output variable 
  if ( ! 'cost' %in% colnames(train_set) ) {
    train_set = cbind(train_set , data[[1]]$traindata[,'cost' , drop = F]) 
  }
  
  # logLabel
  if (logLabel) {
    cat(">>> using logLabel option ... \n")
    stopifnot( length(train_set$cost) == nrow(train_set) )
    train_set$cost = log(train_set$cost)
  }
  
  # power025
  if (power025) {
    stopifnot(!logLabel)
    cat(">>> using power025 option ... \n")
    stopifnot( length(train_set$cost) == nrow(train_set) )
    train_set$cost = train_set$cost^(1/16)
  }
  
  # power03
  if (power03) {
    stopifnot(!logLabel)
    stopifnot(!power025)
    cat(">>> using power03 option ... \n")
    stopifnot( length(train_set$cost) == nrow(train_set) )
    train_set$cost = (train_set$cost^(-0.3)-1)/(-0.3)
  }
  
  # power32
  if(power32) {
    stopifnot(!logLabel)
    stopifnot(!power025)
    stopifnot(!power03)
    cat(">>> using power32 option ... \n")
    stopifnot( length(train_set$cost) == nrow(train_set) )
    train_set$cost = train_set$cost^(1/32)
  }
  
  ##### end of model settings 
  
  ## weighted sum 
  if (identical('ws',modelList[[m]]$model.label))  {
    cat(">>> weigthed sum .... \n")
    stopifnot ( ! is.null(modelList[[m]]$ensembles) )
    stopifnot ( ! is.null(modelList[[m]]$weights) )
    stopifnot ( length(modelList[[m]]$ensembles) == length(modelList[[m]]$weights) )
    stopifnot ( ! is.null(modelList[[m]]$enseble_dir) )
    
    ## normalize weights
    modelList[[m]]$weights = modelList[[m]]$weights / sum(modelList[[m]]$weights) 
    
    ## preds 
    preds = as.data.frame(matrix(
      rep(NA,length(modelList[[m]]$weights)*(nrow(train_set)+nrow(test_set)) ), ncol = length(modelList[[m]]$weights) ))
    colnames(preds) = c(modelList[[m]]$ensembles) 
    for (i in seq_along(modelList[[m]]$weights)) {
      pred_i = as.data.frame( fread(paste(ff.getPath(modelList[[m]]$enseble_dir) , modelList[[m]]$ensembles[i] , sep='')))$assemble
      #pred_i = pred_i[1:nrow(train_set)]
      pred_i = ifelse(pred_i < 0 , 1.5 , pred_i)
      preds[,modelList[[m]]$ensembles[i]] = pred_i
    }
    
    ## do ws 
    pred.ws = rep(0,nrow(preds)) 
    for (i in seq_along(colnames(preds))) {
      pred.ws = pred.ws + modelList[[m]]$weights[i] * preds[,modelList[[m]]$ensembles[i]]
    }
    
    ## write on disk the prediction 
    write.csv(data.frame(id = sample_submission$id , cost=pred.ws[(nrow(train_set)+1):(nrow(train_set)+nrow(test_set))]),
              quote=FALSE, 
              file=paste(ff.getPath("submission"), modelList[[m]]$predFile ,sep='') ,
              row.names=FALSE)
      
    ## write on disk the enseble 
    write.csv(data.frame(id = seq_along(pred.ws) , assemble=pred.ws),
              quote=FALSE, 
              file=paste(ff.getPath("ensemble_out"), modelList[[m]]$predFile ,sep='') ,
              row.names=FALSE)
    
  } else {
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
      foldList = createFoldsSameTubeAssemblyId(data = train_set_cl, nFolds = 8, repeats = 3, seeds=c(123,456,789))
      resamples = makeResampleIndexSameTubeAssemblyId(foldList)
      #controlObject <- trainControl(method = "repeatedcv", repeats = 3, number = 8)
      #controlObject <- trainControl(method = "repeatedcv", repeats = 8, number = 8 , summaryFunction = RMSLECostSummary )
      controlObject <- trainControl(method = "cv", 
                                    ## The method doesn't really matter
                                    ## since we defined the resamples
                                    index = resamples$index, 
                                    indexOut = resamples$indexOut, 
                                    summaryFunction = RMSLECostSummary )
      
      ## tube_assembly_id , id 
      train_set_cl[, 'tube_assembly_id'] = NULL 
      test_set_cl [, 'tube_assembly_id'] = NULL 
      
      l = NULL
      if ( model.label == 'xgbTreeGTJ' && (nr2000 || nr3000 || nr4000 || nr5000) ) {
        nrounds = ifelse(nr2000,2000,ifelse(nr3000,3000,ifelse(nr4000,4000,ifelse(nr5000,5000,stop("nround??")))))
        
        l = ff.regTrainAndPredict ( Ytrain = y , 
                                    Xtrain = train_set_cl, 
                                    Xtest = test_set_cl, 
                                    model.label = model.label, 
                                    controlObject = controlObject, 
                                    best.tuning = T, 
                                    removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = T,
                                    nrounds=nrounds)
      } else {
        l = ff.regTrainAndPredict ( Ytrain = y , 
                                    Xtrain = train_set_cl, 
                                    Xtest = test_set_cl, 
                                    model.label = model.label, 
                                    controlObject = controlObject, 
                                    best.tuning = T, 
                                    removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = T,
                                    metric = 'RMSLE' , maximize = F)
      }
      
      if ( !is.null(l$model) ) {
        rmsle_xval_mod = min(l$model$results$RMSLE)
        bestTune = l$model$bestTune
        pred_mod = l$pred
        if (logLabel) {
          pred_mod = exp(l$pred)
        }
        if (power025) {
          pred_mod = (l$pred)^16
        }
        if (power03) {
          pred_mod = (l$pred*(-0.3)+1)^(1/-0.3)
        }
        if (power32) {
          pred_mod = (l$pred)^32
        }
      } else {
        #rmse_xval_mod = 1000000
        stop(paste('model',model.label,'got errors!!'))
      }
      secs_mod = l$secs 
      
      # output 
      return(list(cluster = cl,
                  model.label = model.label,
                  rmsle_xval_mod = rmsle_xval_mod,
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
    
    predTest = ifelse(predTest<0,1.5,predTest)   #### <<<<<<::::<<<<<<<<
    
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
                              rmsle=NA) 
        pars = as.data.frame(
          matrix(rep(NA,ncol(res_list[[i]]$bestTune)*length(sort(unique(train_set$qty_lev)))),ncol = ncol(res_list[[i]]$bestTune)))
        colnames(pars) = colnames(res_list[[i]]$bestTune)
        tuneGrid = cbind(tuneGrid,pars)
      } 
      tuneGrid[tuneGrid$cluster==res_list[[i]]$cluster,]$secs = res_list[[i]]$secs_mod
      tuneGrid[tuneGrid$cluster==res_list[[i]]$cluster,]$rmsle = res_list[[i]]$rmsle_xval_mod
      tuneGrid[tuneGrid$cluster==res_list[[i]]$cluster,5:ncol(tuneGrid)] = res_list[[i]]$bestTune
    }
    write.csv(tuneGrid,
              quote=FALSE, 
              file=paste(ff.getPath("best_tune_out"), modelList[[m]]$predFile ,sep='') ,
              row.names=FALSE)
    
    ##############
    ## ENSEMB 
    ##############
    cat(">>> Ensembling ... \n")  
    bestTune = NULL
    if (modelList[[m]]$tune) {
      bestTune = as.data.frame( fread(paste(ff.getPath("best_tune_out") , modelList[[m]]$predFile , sep='')))
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
        
        #controlObject <- trainControl(method = "none", summaryFunction = RMSLECostSummary )
        foldList = createFoldsSameTubeAssemblyId(data = train_set_cl, nFolds = nFolds, repeats = 1, seeds=c(123))
        resamples = makeResampleIndexSameTubeAssemblyId(foldList)
        controlObject <- trainControl(method = "cv", 
                                      ## The method doesn't really matter
                                      ## since we defined the resamples
                                      index = resamples$index, 
                                      indexOut = resamples$indexOut, 
                                      summaryFunction = RMSLECostSummary )
        
        ## tube_assembly_id , id 
        train_set_cl[, 'tube_assembly_id'] = NULL 
        test_set_cl [, 'tube_assembly_id'] = NULL 
        
        if (identical(model.label,'xgbTree')) {
          ens = ff.createEnsemble (Xtrain = train_set_cl,
                                   Xtest = test_set_cl,
                                   y = y,
                                   bestTune = expand.grid(
                                     nrounds = bestTune[bestTune$cluster == cl, ]$early.stop ,
                                     max_depth = 20,  
                                     eta = 0.05 ),
                                   caretModelName = model.label, 
                                   parallelize = T,
                                   verbose = T , 
                                   
                                   controlObject = controlObject, 
                                   
                                   objective = "reg:linear",
                                   gamma = 0.7,  
                                   subsample = 0.5 , ## suggested in ESLII
                                   nthread = 10, 
                                   min_child_weight = 1 , 
                                   colsample_bytree = 0.5, 
                                   max_delta_step = 1)
          
        } else if (identical(model.label,'xgbTreeGTJ')) {
          ens = ff.createEnsemble (Xtrain = train_set_cl,
                                   Xtest = test_set_cl,
                                   y = y,
                                   bestTune = expand.grid(
                                     nrounds = bestTune[bestTune$cluster == cl, ]$early.stop ,
                                     max_depth = 8,  
                                     eta = 0.02 ),
                                   caretModelName = 'xgbTree', 
                                   parallelize = T,
                                   verbose = T , 
                                   
                                   controlObject = controlObject, 
                                   
                                   objective = "reg:linear",
                                   min_child_weight = 6 , 
                                   subsample = 0.7 , 
                                   colsample_bytree = 0.6 , 
                                   scale_pos_weight = 0.8 , 
                                   #silent = 1 , 
                                   max_delta_step = 2)        
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
          
          if ( identical(model.label,"rlm") ) {
            ens = ff.createEnsemble (Xtrain = train_set_cl,
                                     Xtest = test_set_cl,
                                     y = y,
                                     bestTune = tgrid,
                                     caretModelName = model.label, 
                                     parallelize = T,
                                     verbose = T,
                                     removePredictorsMakingIllConditionedSquareMatrix = TRUE, 
                                     controlObject = controlObject, 
                                     metric = 'RMSLE' , maximize = F) 
          } else {
            ens = ff.createEnsemble (Xtrain = train_set_cl,
                                     Xtest = test_set_cl,
                                     y = y,
                                     bestTune = tgrid,
                                     caretModelName = model.label, 
                                     parallelize = T,
                                     verbose = T,
                                     controlObject = controlObject, 
                                     metric = 'RMSLE' , maximize = F)  
          }
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
      if (logLabel) {
        predTrain[res_list[[i]]$predTrain_cl_idx] = exp(res_list[[i]]$ensemble$predTrain)
        predTest[res_list[[i]]$predTest_cl_idx] = exp(res_list[[i]]$ensemble$predTest)
      }
      if (power025) {
        predTrain[res_list[[i]]$predTrain_cl_idx] = (res_list[[i]]$ensemble$predTrain)^16
        predTest[res_list[[i]]$predTest_cl_idx] = (res_list[[i]]$ensemble$predTest)^16
      }
      if (power03) {
        predTrain[res_list[[i]]$predTrain_cl_idx] = (res_list[[i]]$ensemble$predTrain*(-0.3)+1)^(1/-0.3)
        predTest[res_list[[i]]$predTest_cl_idx] = (res_list[[i]]$ensemble$predTest*(-0.3)+1)^(1/-0.3)
      }
      if (power32) {
        predTrain[res_list[[i]]$predTrain_cl_idx] = (res_list[[i]]$ensemble$predTrain)^32
        predTest[res_list[[i]]$predTest_cl_idx] = (res_list[[i]]$ensemble$predTest)^32
      }
    }
    stopifnot( sum(is.na(predTrain)) == 0 )
    stopifnot( sum(is.na(predTest))  == 0 )
    
    predTrain = ifelse(predTrain == -1 , NA , predTrain) 
    predTest = ifelse(predTest == -1 , NA , predTest)   
    
    ## assemble 
    assemble = c(predTrain,predTest)
    write.csv(data.frame(id = seq_along(assemble) , assemble=assemble),
              quote=FALSE, 
              file=paste(ff.getPath("ensemble_out"), modelList[[m]]$predFile ,sep='') ,
              row.names=FALSE)
  }
}
####### end of loop 
tm = proc.time() - ptm
secs = as.numeric(tm[3])
cat(">>> MAIN LOOP >>> Time elapsed:",secs," secs. [",secs/60,"min.] [",secs/(60*60),"hours] \n")
############## end of MAIN LOOP 


