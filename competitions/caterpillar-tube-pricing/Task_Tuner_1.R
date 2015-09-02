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

source(paste0( ff.getPath("process") , "/FeatureSelection_Lib.R"))
source(paste0( ff.getPath("process") , "/Regression_Lib.R"))

################# SETTINGS
DEBUG_MODE = F

################# MODELS 

# reg_models = c("pls" , "ridge" , "enet" , 
#                "knn")

#reg_models = c("enet" , "svmRadial",  "gbm" )

reg_models = c("gbm")



cluster_levs = 1:8 ##<<<<<<<---- :::::: <<<<<<<<< 
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
  } else {
    rmse_xval_mod = 1000000
  }
  secs_mod = l$secs 
  
  # output 
  return(list(cluster = cls,
              model.label = model.label,
              rmse_xval_mod = rmse_xval_mod,
              bestTune = bestTune, 
              secs_mod = secs_mod)) 
} , mc.cores = nrow(grid) )
####### end of parallel loop 
tm = proc.time() - ptm
secs = as.numeric(tm[3])
cat(">>> MAIN LOOP >>> Time elapsed:",secs," secs. [",secs/60,"min.] [",secs/(60*60),"hours] \n")
############## end of MAIN LOOP 

## write on disk 
cat(">>> writing tune on disk ... \n")
for (mod in reg_models) {
  tuneGrid = NULL 
  for (i in seq_along(res_list)) {
    if ( ! identical(res_list[[i]]$model.label,mod) ) next 
    if ( is.null(res_list[[i]]$bestTune) ) next 
    if(is.null(tuneGrid)) {
      tuneGrid = data.frame(cluster=cluster_levs,model=mod,secs=NA,rmse=NA) 
      pars = as.data.frame(
        matrix(rep(NA,ncol(res_list[[i]]$bestTune)*length(cluster_levs)),ncol = ncol(res_list[[i]]$bestTune)))
      colnames(pars) = colnames(res_list[[i]]$bestTune)
      tuneGrid = cbind(tuneGrid,pars)
    } 
    tuneGrid[tuneGrid$cluster==res_list[[i]]$cluster,]$secs = res_list[[i]]$secs_mod
    tuneGrid[tuneGrid$cluster==res_list[[i]]$cluster,]$rmse = res_list[[i]]$rmse_xval_mod
    tuneGrid[tuneGrid$cluster==res_list[[i]]$cluster,5:ncol(tuneGrid)] = res_list[[i]]$bestTune
  }
  if (! is.null(tuneGrid)) {
    fn = paste(length(cluster_levs),'_',mod,'.csv',sep='')
    write.csv(tuneGrid,quote=FALSE, 
              file=paste(ff.getPath("best_tune"),fn,sep='') ,
              row.names=FALSE)
  }
}
