library(binhf)
library(fBasics)
library(lattice)
require(xgboost)
require(methods)
library(data.table)
library(plyr)

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
  
  ## al momento non consideriamo material_id 
  train_set[, 'material_id'] = NULL 
  test_set [, 'material_id'] = NULL 
  
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
  
  return(list(pred=pred,perf.cv=xgb_xval$perf.cv))
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

## NOT putting quote_date in data set  
#head_train_set = train_enc_date
#head_test_set = test_enc_date
 head_train_set = train_enc
 head_test_set = test_enc 

## build train_set and test_set 
train_set = merge(x = head_train_set , y = tube , by = 'tube_assembly_id' , all = F)
test_set = merge(x = head_test_set , y = tube , by = 'tube_assembly_id' , all = F)

## clustering 
material_ids = sort(unique(tube$material_id))
cluster_perf = data.frame(material_id = material_ids , 
                          rmse_xval = NA , 
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
for (cl in 1:length(material_ids)) {
  cat('>> processing material_id ',material_ids[cl], '[',cl,'/',length(material_ids),'] ... \n')
  
  ## define train / test set 
  train_set_cl = train_set[train_set$material_id==material_ids[cl],]
  test_set_cl = test_set[test_set$material_id==material_ids[cl],]
  if (nrow(test_set_cl) == 0) next 
  
  pred_cl_idx = which(test_set$material_id==material_ids[cl])
  if ( length(pred_cl_idx) != nrow(test_set_cl) ) stop('something wrong')
  
  ## mean / sd / zeta / num 
  cost_mean = mean(train_set_cl$cost)
  cost_sd = sd(train_set_cl$cost)
  zeta = ifelse( is.na(cost_sd) , Inf , cost_mean / cost_sd)
  
  cluster_perf[cluster_perf$material_id==material_ids[cl] ,]$cost_mean = cost_mean
  cluster_perf[cluster_perf$material_id==material_ids[cl] ,]$cost_sd = cost_sd
  cluster_perf[cluster_perf$material_id==material_ids[cl] ,]$zeta = zeta
  cluster_perf[cluster_perf$material_id==material_ids[cl] ,]$num_train = nrow(train_set_cl)
  cluster_perf[cluster_perf$material_id==material_ids[cl] ,]$num_test = nrow(test_set_cl)
  
  ## handle cluster of material_id UNKNOWN o train set vuoto 
  if (material_ids[cl] == 'UNKNOWN' || nrow(train_set_cl) == 0) {
    ## using sub_base 
    cat(">> using sub_base submission ... \n")
    pred[pred_cl_idx] = sub_base[pred_cl_idx,'cost']
    cluster_perf[cluster_perf$material_id==material_ids[cl] ,]$rmse_xval = 11
  } else {
    ## train and predict
    xgb = xgb_train_and_predict (train_set_cl,
                                 test_set_cl, 
                                 param,
                                 cv.nround = 3000 , 
                                 nfold = min(5,nrow(train_set_cl)) , 
                                 verbose=F)
    pred[pred_cl_idx] = xgb$pred
    cluster_perf[cluster_perf$material_id==material_ids[cl] ,]$rmse_xval = xgb$perf.cv
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
          file=paste(getBasePath("submission"),'sub_cluster_no_0var_preds.csv',sep='') ,
          row.names=FALSE)

# sample_submission$cost = pred_real 
# write.csv(sample_submission,quote=FALSE, 
#           file=paste(getBasePath("submission"),'sub_cluster_real2.csv',sep='') ,
#           row.names=FALSE)

write.csv(cluster_perf,quote=FALSE, 
          file=paste(getBasePath("submission"),'cluster_lev_1_material_id_no_0var_preds.csv',sep='') ,
          row.names=FALSE)
