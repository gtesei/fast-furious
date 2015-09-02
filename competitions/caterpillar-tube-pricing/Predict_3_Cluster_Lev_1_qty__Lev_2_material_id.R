library(binhf)
library(fBasics)
library(lattice)
require(xgboost)
require(methods)
library(data.table)
library(plyr)
library(caret)

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
                                 test_set, 
                                 param,
                                 cv.nround = 3000 , 
                                 nfold = 5 , 
                                 verbose=T) {
  
  ## tube_assembly_id , id 
  train_set[, 'tube_assembly_id'] = NULL 
  test_set [, 'tube_assembly_id'] = NULL 
  test_set [, 'id'] = NULL 
  
  ## al momento non consideriamo material_id 
  train_set[, 'material_id'] = NULL 
  test_set [, 'material_id'] = NULL 
  
  ## y, data 
  y = train_set$cost   
  #y = ((train_set$cost)^-0.3) 
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
  
  #pred = pred^-(1/0.3) 
  
  return(list(pred=pred,
              perf.cv=xgb_xval$perf.cv,
              early.stop=xgb_xval$early.stop))
}

################# FAST-FURIOUS SOURCES
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))
source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))

################# DATA IN 
sample_submission = as.data.frame( fread(paste(getBasePath("data") , 
                                               "sample_submission.csv" , sep=''))) 

sub_base = as.data.frame( fread(paste(getBasePath("submission_old") , 
                                      "sub_base_date.csv" , sep=''))) #~0.37 LB / 11 rmse xval

sub_cluster = as.data.frame( fread(paste(getBasePath("submission_old") , 
                                         "sub_cluster_by_qty.csv" , sep=''))) #~0.341 LB 

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
# sub_cluster2_base
# cluster_lev_2_grid 

################# META
feature_scale = T

################# PROCESSING 

## build technical feature set 
tube = cbind(tube_base,bom_base)
tube = cbind(tube,spec_enc)
dim(tube) ## 180 (encoded) technical features  
# [1] 21198   180

## NOT putting quote_date in data set  
head_train_set = train_enc_date
head_test_set = test_enc_date
# head_train_set = train_enc
# head_test_set = test_enc 

## build train_set and test_set 
train_set = merge(x = head_train_set , y = tube , by = 'tube_assembly_id' , all = F)
test_set = merge(x = head_test_set , y = tube , by = 'tube_assembly_id' , all = F)

######### feature scaling 
if (feature_scale) {
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
} else{
  cat(">>> No feature scaling ... \n")
}
######### 

## create cluster grid  
quantity_levels = 1:8 
material_ids = sort(unique(tube$material_id))
cls_qty = cluster_by(predictor.train=train_set$quantity,
                 predictor.test=test_set$quantity,
                 num_bids = length(quantity_levels),
                 verbose=T)

cluster_perf = expand.grid(cluster_lev_1 = quantity_levels , cluster_lev_2=material_ids) 
props_perf = c('rmse_xval','early.stop','cost_mean','cost_sd','zeta','num_train','num_test','go_deeper','lev_1_zeta','pred_used','lev_1_rmse') 
cluster_perf = cbind(cluster_perf , data.frame(matrix(rep(NA,nrow(cluster_perf)*length(props_perf)),nrow = nrow(cluster_perf))))
colnames(cluster_perf)[3:ncol(cluster_perf)] = props_perf

## update train_set /  test_set 
train_set$qty_lev = cls_qty$levels.train
test_set$qty_lev = cls_qty$levels.test

## clustering logic 
NUM_TRAIN_MIN = 300 
#NUM_TRAIN_MIN_MIN = 5 

# cluster_lev  quantity  zeta
# 7	100	1.088765244
# 8	2500	1.036895768
# 5	20	0.871664445
# 6	40	0.789187578
# 4	10	0.88339998
# 3	5	0.672756684
# 2	2	0.688849953
# 1	1	0.539287014
LEVEL_1_ZETA = c(0.539287014,0.688849953,0.672756684,0.88339998,0.871664445,0.789187578,1.088765244,1.036895768)

# cat(">>> we'll go deeper in lev 2 subclusters if train obs >",NUM_TRAIN_MIN," - or if train obs >",NUM_TRAIN_MIN_MIN,
#     " and zeta > level 1 zeta \n")

cat(">>> we'll go deeper in lev 2 subclusters if train obs >",NUM_TRAIN_MIN," ... \n")

# lev qty rmse 
# 7  100  2.040789
# 8	 2500	2.563415
# 5	 20	  4.364462
# 6	 40	  4.803652
# 4	 10	  4.985274
# 3	 5	  9.060765
# 2	 2	  16.296162
# 1	 1	  26.67484
LEVEL_1_RMSE_XVAL = c(26.67484,16.296162,9.060765,4.985274,4.364462,4.803652,2.040789,2.563415)

LEVEL_0_RMSE_XVAL = 11

cat(">>> anyway, if rmse xval < rmse xval of level 1 cluster, the latter pred will be used if it is > rmse xval of level 1 cluster (~",
    LEVEL_0_RMSE_XVAL,") ...\n")

## TODO train su tutto il data set e xval su ogni classe del cluster 1 x essere sicuri della regola sopra 

## compute stats 
cat(">>> processing stats ... \n")
compute_stats= lapply( seq_along(1:nrow(cluster_perf)) , function(i) {
  level_qty = cluster_perf[i,'cluster_lev_1']
  mat_id = cluster_perf[i,'cluster_lev_2']
  
  #cat('>> processing stats for <level_qty: ',level_qty, ', material_id:',mat_id,'>  [',i,'/',nrow(cluster_perf),'] ... \n')
  
  train_set_cl = train_set[train_set$qty_lev == level_qty & train_set$material_id == mat_id,]
  test_set_cl = test_set[test_set$qty_lev == level_qty & test_set$material_id == mat_id,]
  
  ## mean / sd / zeta / num 
  cluster_perf[i,'cost_mean'] <<- mean(train_set_cl$cost)
  cluster_perf[i,'cost_sd'] <<- sd(train_set_cl$cost)
  cluster_perf[i,'zeta'] <<- ifelse( is.na(cluster_perf[i,'cost_sd']) , Inf , cluster_perf[i,'cost_mean'] / cluster_perf[i,'cost_sd'])
  cluster_perf[i,'num_train'] <<- nrow(train_set_cl)
  cluster_perf[i,'num_test'] <<- nrow(test_set_cl)
  
  ## go_deeper? 
  if (cluster_perf[i,'num_test']==0) {
    
    cluster_perf[i,'go_deeper'] <<- F
    
  } else if ( 
    (cluster_perf[i,'num_train'] > NUM_TRAIN_MIN ) ) {
    #|| (cluster_perf[i,'num_train'] > NUM_TRAIN_MIN_MIN & cluster_perf[i,'zeta'] > LEVEL_1_ZETA[level_qty]) ) {
    
    cluster_perf[i,'go_deeper'] <<- T
    
  } else {
    cluster_perf[i,'go_deeper'] <<- F
  }
  
  ## lev 1 
  cluster_perf[i,'lev_1_zeta'] <<- LEVEL_1_ZETA[level_qty]
  cluster_perf[i,'lev_1_rmse'] <<- LEVEL_1_RMSE_XVAL[level_qty]
}) 

print(head(cluster_perf))
cat(">>> found",sum(cluster_perf$num_test == 0)," cases with 0 test set observations (out of ",nrow(cluster_perf),") ... \n")
cat(">>> of remaining cases, found",sum(cluster_perf$num_train == 0 & cluster_perf$num_test > 0)," cases with 0 train set observations (out of ",
    sum(cluster_perf$num_test > 0),") ... \n")

## pre-allocate pred 
pred = rep(NA,nrow(test_set))

##### xgboost --> set necessary parameter
param <- list("objective" = "reg:linear",
              "eval_metric" = "rmse", 
              "eta" = 0.05,  
              "gamma" = 0.7,  
              "max_depth" = 20, 
              "subsample" = 0.5 , ## suggested in ESLII
              "nthread" = 10, 
              "min_child_weight" = 1 , # 5
              "colsample_bytree" = 0.5, # 0.8 params["scale_pos_weight"] = 1.0
              "max_delta_step" = 1)

cat(">> XGBoost Params:\n")
print(param)

## main loop
cat(">>> main loop ... \n")
main_loop = lapply( seq_along(1:nrow(cluster_perf)) , function(i) {
  level_qty = cluster_perf[i,'cluster_lev_1']
  mat_id = cluster_perf[i,'cluster_lev_2']
  
  cat('>> processing [ level_qty: ',level_qty, ', material_id:',mat_id,']  [',i,'/',nrow(cluster_perf),'] ... \n')
  
  train_set_cl = train_set[train_set$qty_lev == level_qty & train_set$material_id == mat_id,]
  test_set_cl =  test_set [test_set$qty_lev ==  level_qty & test_set$material_id == mat_id,]
  pred_cl_idx =  which    (test_set$qty_lev ==  level_qty & test_set$material_id == mat_id)
  stopifnot ( length(pred_cl_idx) == nrow(test_set_cl) ) 
  
  ## check if there are observations in test set 
  if (nrow(test_set_cl)==0) {
    cat (">>> NO observation in test set ... next ... \n") 
  } else if (cluster_perf[i,'go_deeper']) {
    cat(">> going deeper ... \n")
    xgb = xgb_train_and_predict (train_set_cl,
                                 test_set_cl, 
                                 param,
                                 cv.nround = 3000 , 
                                 nfold = min(5,nrow(train_set_cl)) , 
                                 verbose=F)
    if (xgb$perf.cv < cluster_perf[i,'lev_1_rmse']) {
      cat(">> perf.cv = ",xgb$perf.cv,"< perf.cluster.lev_1 =",cluster_perf[i,'lev_1_rmse']," ===>>> using pred ... \n")
      
      pred[pred_cl_idx] <<- xgb$pred
      cluster_perf[i,'pred_used'] <<- T
      
    } else if (cluster_perf[i,'lev_1_rmse'] < LEVEL_0_RMSE_XVAL ) {
      cat(">> perf.cv = ",xgb$perf.cv,"> perf.cluster.lev_1 =",cluster_perf[i,'lev_1_rmse']," < perf.cluster.lev_0 =",LEVEL_0_RMSE_XVAL,"===>>> using sub_cluster_lev_1 ... \n")
      
      pred[pred_cl_idx] <<- sub_cluster[pred_cl_idx,'cost']
      cluster_perf[i,'pred_used'] <<- F
      
    } else {
      cat(">> perf.cv = ",xgb$perf.cv,"> perf.cluster.lev_1 =",cluster_perf[i,'lev_1_rmse']," > perf.cluster.lev_0 =",LEVEL_0_RMSE_XVAL,"===>>> using sub_base ... \n")
      
      pred[pred_cl_idx] <<- sub_base[pred_cl_idx,'cost']
      cluster_perf[i,'pred_used'] <<- F
    }
    
    cluster_perf[i,'rmse_xval'] <<- xgb$perf.cv
    cluster_perf[i,'early.stop'] <<- xgb$early.stop
  } else {
    cat(">> not going deeper ... \n")
    if (cluster_perf[i,'lev_1_rmse'] < LEVEL_0_RMSE_XVAL ) {
      cat(">> perf.cluster.lev_1 =",cluster_perf[i,'lev_1_rmse']," < perf.cluster.lev_0 =",LEVEL_0_RMSE_XVAL,"===>>> using sub_cluster_lev_1 ... \n")
      
      pred[pred_cl_idx] <<- sub_cluster[pred_cl_idx,'cost']
    } else {
      cat(">> perf.cluster.lev_1 =",cluster_perf[i,'lev_1_rmse']," > perf.cluster.lev_0 =",LEVEL_0_RMSE_XVAL,"===>>> using sub_base ... \n")
      
      pred[pred_cl_idx] <<- sub_base[pred_cl_idx,'cost']
    }
    
    cluster_perf[i,'pred_used'] <<- F
  }
})

## basic check 
stopifnot (sum(is.na(pred)) == 0)  
pred_real = pred
pred = ifelse(pred<0,1.5,pred)

cat('>> number of prediction < 0:',sum(pred_real<0),' ... repleaced with 1.5 \n')

# write on disk 
cat(">> writing prediction on disk ... \n")

# sub_cluster2_base
sample_submission$cost = pred 
write.csv(sample_submission,quote=FALSE, 
          file=paste(getBasePath("submission"),'sub_cl_2_xgb.csv',sep='') ,
          row.names=FALSE)

#cluster_lev_2_grid 
write.csv(cluster_perf,quote=FALSE, 
          file=paste(getBasePath("submission"),'cluster_lev_2_grid_xgb.csv',sep='') ,
          row.names=FALSE)


