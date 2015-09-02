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

################# FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/caterpillar-tube-pricing/competition_data')
ff.bindPath(type = 'elab' , sub_path = 'dataset/caterpillar-tube-pricing/elab')
ff.bindPath(type = 'process' , sub_path = 'data_process')

ff.bindPath(type = 'ensemble' , sub_path = 'dataset/caterpillar-tube-pricing/ensemble_1') ## in 

ff.bindPath(type = 'best_tune' , sub_path = 'dataset/caterpillar-tube-pricing/best_tune_2',createDir = TRUE) ## in  

ff.setMaxCuncurrentThreads(16)

source(paste0( ff.getPath("process") , "/Regression_Lib.R"))

################# SETTINGS
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

train_obs = nrow(train_set)
test_obs = nrow(test_set)

cat(">>> Train observations:",train_obs," - test observations:",test_obs,"\n")

## meta-features 
cat(">>> Loading meta-features from ",ff.getPath('ensemble')," ... \n")
mf = list.files( ff.getPath('ensemble') )
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

##############
train_meta = cbind(train_meta,quantity = train_set$quantity)
test_meta = cbind(test_meta,quantity = test_set$quantity)
train_meta = cbind(train_meta,cost = train_set$cost)

##
cls = cluster_by(predictor.train=train_set$quantity,
                 predictor.test=test_set$quantity,
                 num_bids = 8,
                 verbose=T)

train_meta$qty_lev = cls$levels.train
test_meta$qty_lev = cls$levels.test

## cluster n....
cl = 8 

## define train / test set 
train_set_cl = train_meta[train_meta$qty_lev == cl,]
test_set_cl = test_meta[test_meta$qty_lev== cl,]
cat('>>> train observations:',nrow(train_set_cl), '- test observations:',nrow(test_set_cl), ' \n')
if (nrow(train_set_cl) == 0) stop('something wrong in train_set_cl') 
if (nrow(test_set_cl) == 0) stop('something wrong in test_set_cl') 

predTest_cl_idx = which(test_meta$qty_lev==cl)
stopifnot ( length(predTest_cl_idx) == nrow(test_set_cl) ) 

predTrain_cl_idx = which(train_meta$qty_lev==cl)
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

## tube_assembly_id , id 
train_set_cl[, 'tube_assembly_id'] = NULL 
test_set_cl [, 'tube_assembly_id'] = NULL 
test_set_cl [, 'id'] = NULL 

## material_id 
train_set_cl[, 'material_id'] = NULL 
test_set_cl [, 'material_id'] = NULL 
cat(">>> train_set after encoding:",ncol(train_set_cl)," - test_set after encoding:",ncol(test_set_cl)," ... \n")

## cost
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

## bestTune 
bestTune = as.data.frame( fread(paste(ff.getPath("best_tune") , '8_cubist_useQty.csv' , sep='')))

# 8-fold 
kfolds = createFolds(y, k = 8, returnTrain = F)
i = 1 
train_i = train_set_cl[ -kfolds[[i]] , ]
y_i = y[-kfolds[[i]]]
test_i = train_set_cl[ kfolds[[i]] , ]

l = ff.featureFilter(traindata = train_i, 
                     testdata = test_i,
                     removeOnlyZeroVariacePredictors=TRUE,
                     performVarianceAnalysisOnTrainSetOnly = TRUE , 
                     correlationThreshold = NULL, 
                     removePredictorsMakingIllConditionedSquareMatrix = F, 
                     removeHighCorrelatedPredictors = F, 
                     featureScaling = F, 
                     verbose = T)

train_i = l$traindata
test_i = l$testdata 

controlObject <- trainControl(method = "none")
model <- train(y = y_i, x = train_i ,
               method = 'cubist',
               tuneGrid = bestTune[bestTune$cluster==1,5:ncol(bestTune) , drop=F],
               trControl = controlObject )

obs_i = y[kfolds[[i]]] 
pred_i = predict(model,test_i)

ff.plotPerformance.reg(observed = obs_i , predicted = pred_i)

mean_y = mean(y)
var_y = var(y)

####
hist(y)
hist(log(y))

##
 bc = BoxCoxTrans(y)
lambda = 0.1 ## cluster n.1 
lambda = -1.3   ## cluster n. 2 
lambda = -0.4  ## clusrer n.8 
y_bc = (y^lambda-1)/lambda
y_bc_inv = (y_bc*lambda+1)^(1/lambda)
hist(log(y))
hist(y_bc)







