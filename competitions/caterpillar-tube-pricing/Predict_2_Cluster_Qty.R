library(binhf)
library(fBasics)
library(lattice)
require(xgboost)
require(methods)
library(data.table)
library(plyr)
library(Hmisc)

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

xgb_cross_val = function( data , y , cv.nround = 3000 , nfold = 5 , verbose=T) {
  
  inCV = T
  early.stop = cv.nround 
  perf.xg = NULL 
  
  while (inCV) {
    
    #cat(">> cv.nround: ",cv.nround,"\n") 
    bst.cv = xgb.cv(param=param, data = data , label = y, 
                    nfold = nfold, nrounds=cv.nround , verbose=verbose)
    print(bst.cv)
    early.stop = which(bst.cv$test.rmse.mean == min(bst.cv$test.rmse.mean) )
    if (length(early.stop)>1) early.stop = early.stop[length(early.stop)]
    cat(">> early.stop: ",early.stop," [test.mlogloss.mean:",bst.cv[early.stop,]$test.mlogloss.mean,"]\n") 
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
                                 test_set, 
                                 param,
                                 cv.nround = 3000 , 
                                 nfold = 5 , 
                                 verbose=T) {
  
  ## tube_assembly_id , id 
  train_set[, 'tube_assembly_id'] = NULL 
  test_set [, 'tube_assembly_id'] = NULL 
  test_set [, 'id'] = NULL 
  
  ## material_id 
#   cat(">>> encoding material_id [",unique(c(train_set$material_id , test_set$material_id)),"] [",length(unique(c(train_set$material_id , test_set$material_id))),"] ... \n")
#   l = encodeCategoricalFeature (train_set$material_id , test_set$material_id , colname.prefix = "material_id" , asNumeric=F)
#   cat(">>> train_set before encoding:",ncol(train_set)," - test_set before encoding:",ncol(test_set)," ... \n")
#   train_set = cbind(train_set , l$traindata)
#   test_set = cbind(test_set , l$testdata)
  
  train_set[, 'material_id'] = NULL 
  test_set [, 'material_id'] = NULL 
  cat(">>> train_set after encoding:",ncol(train_set)," - test_set after encoding:",ncol(test_set)," ... \n")
  
  ## y, data 
  y = train_set$cost   
  train_set[, 'cost'] = NULL 
  
  ###################### data processing  
  l = featureSelect (train_set,test_set,
                     removeOnlyZeroVariacePredictors=T,
                     performVarianceAnalysisOnTrainSetOnly = T , 
                     removePredictorsMakingIllConditionedSquareMatrix = F, 
                     removeHighCorrelatedPredictors = F, 
                     featureScaling = F)
  train_set = l$traindata
  test_set = l$testdata
  ######################
  
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
  
  return(list(pred=pred,perf.cv=xgb_xval$perf.cv,early.stop=xgb_xval$early.stop))
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

################# DATA IN 
sample_submission = as.data.frame( fread(paste(getBasePath("data") , 
                                               "sample_submission.csv" , sep=''))) 

sub_base = as.data.frame( fread(paste(getBasePath("submission_old") , 
                                      "sub_base_date.csv" , sep=''))) #~0.39 LB / 11 rmse xval

## elab 
train_enc = as.data.frame( fread(paste(getBasePath("elab") , 
                                       "train_enc.csv" , sep=''))) 

test_enc = as.data.frame( fread(paste(getBasePath("elab") , 
                                      "test_enc.csv" , sep=''))) 

train_enc_date = as.data.frame( fread(paste(getBasePath("elab") , 
                                            "train_enc_date.csv" , sep=''))) 

test_enc_date = as.data.frame( fread(paste(getBasePath("elab") , 
                                           "test_enc_date.csv" , sep=''))) 

##
tube_base = as.data.frame( fread(paste(getBasePath("elab") , 
                                  "tube_base.csv" , sep='')))

bom_base = as.data.frame( fread(paste(getBasePath("elab") , 
                                       "bom_base.csv" , sep='')))

spec_enc = as.data.frame( fread(paste(getBasePath("elab") , 
                                       "spec_enc.csv" , sep='')))

################# DATA OUT 
cluster_perf = NULL 

################# PROCESSING 

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
cluster_levs = 1:8 
cls = cluster_by(predictor.train=train_set$quantity,
           predictor.test=test_set$quantity,
           num_bids = length(cluster_levs),
           verbose=T)

train_set$qty_lev = cls$levels.train
test_set$qty_lev = cls$levels.test

cluster_perf = data.frame(cluster_lev = cluster_levs , 
                          rmse_xval = NA , 
                          early.stop = NA , 
                          cost_mean = NA, 
                          cost_sd = NA, 
                          zeta = NA, 
                          num_train = NA, 
                          num_test = NA )
pred = rep(NA,nrow(test_set))

##### xgboost --> set necessary parameter
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

cat(">> XGBoost Params:\n")
print(param)
  
## main loop 
for (cl in 1:length(cluster_levs)) {
  cat('>>> processing cluster_lev ',cluster_levs[cl], '[',cl,'/',length(cluster_levs),'] ... \n')
  
  ## define train / test set 
  train_set_cl = train_set[train_set$qty_lev == cluster_levs[cl],]
  test_set_cl = test_set[test_set$qty_lev== cluster_levs[cl],]
  cat('>>> train observations:',nrow(train_set_cl), '- test observations:',nrow(test_set_cl), ' \n')
  if (nrow(test_set_cl) == 0) next 
  
  pred_cl_idx = which(test_set$qty_lev==cluster_levs[cl])
  if ( length(pred_cl_idx) != nrow(test_set_cl) ) stop('something wrong')
  
  ## mean / sd / zeta / num 
  cost_mean = mean(train_set_cl$cost)
  cost_sd = sd(train_set_cl$cost)
  zeta = ifelse( is.na(cost_sd) , Inf , cost_mean / cost_sd)
  
  cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$cost_mean = cost_mean
  cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$cost_sd = cost_sd
  cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$zeta = zeta
  cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$num_train = nrow(train_set_cl)
  cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$num_test = nrow(test_set_cl)
  
  ## handle cluster of train set vuoto 
  if (nrow(train_set_cl) == 0) {
    ## using sub_base 
    cat(">> no train observations available ==> using sub_base submission ... \n")
    pred[pred_cl_idx] = sub_base[pred_cl_idx,'cost']
    cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$rmse_xval = 11
  } else {
    ## train and predict
    xgb = xgb_train_and_predict (train_set_cl,
                                 test_set_cl, 
                                 param,
                                 cv.nround = 3000 , 
                                 nfold = min(5,nrow(train_set_cl)) , 
                                 verbose=F)
    pred[pred_cl_idx] = xgb$pred
    cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$rmse_xval = xgb$perf.cv
    cluster_perf[cluster_perf$cluster_lev==cluster_levs[cl] ,]$early.stop = xgb$early.stop
  }
}

## basic check 
if (sum(is.na(pred)) >0) stop('something wrong')
pred_real = pred
pred = ifelse(pred<0,1.5,pred)

cat('>> number of prediction < 0:',sum(pred_real<0),' ... repleaced with 1.5 \n')

# write on disk 
cat(">> writing prediction / cluster_perf on disk ... \n")

sample_submission$cost = pred 
write.csv(sample_submission,quote=FALSE, 
          file=paste(getBasePath("submission"),'sub_cluster_by_qty.csv',sep='') ,
          row.names=FALSE)

write.csv(cluster_perf,quote=FALSE, 
          file=paste(getBasePath("submission"),'cluster_lev_1_cluster_by_qty.csv',sep='') ,
          row.names=FALSE)
