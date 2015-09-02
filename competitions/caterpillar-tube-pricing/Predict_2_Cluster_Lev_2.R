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
  train_set[, 'cost'] = NULL 
  
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

################# DATA IN 
sample_submission = as.data.frame( fread(paste(getBasePath("data") , 
                                               "sample_submission.csv" , sep=''))) 

sub_base = as.data.frame( fread(paste(getBasePath("submission") , 
                                      "sub_base.csv" , sep=''))) #~0.39 LB / 11 rmse xval

sub_cluster = as.data.frame( fread(paste(getBasePath("submission") , 
                                         "sub_cluster.csv" , sep=''))) #~0.45 LB 


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
MIN_OBS_TRAIN_FOR_CLUTERING_DEEPER = 10
MIN_ZETA_FOR_CLUTERING_DEEPER = 10
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
  
  ## clustering by quantity 
  train_set_cl_1 = train_set[train_set$material_id==material_ids[cl],]
  test_set_cl_1 = test_set[test_set$material_id==material_ids[cl],]
  if (nrow(test_set_cl_1) == 0) next 
  
  qtys = sort(unique(train_set_cl_1$quantity))
  
  cat('sub_cluster_train\n')
  sub_cluster_train = ddply(train_set_cl_1 , .(quantity) , function(x) 
    c(obs = length(x$quantity) , cost.mean=mean(x$cost) , cost.sd=sd(x$cost) , zeta=mean(x$cost)/sd(x$cost)) )
  sub_cluster_train$zeta = ifelse( is.na(sub_cluster_train$zeta) & ! is.na(sub_cluster_train$cost.mean), 
                                   100, sub_cluster_train$zeta)
  print(sub_cluster_train)
  
  cat('sub_cluster_test\n')
  sub_cluster_test = ddply(test_set_cl_1 , .(quantity) , function(x) 
    c(obs = nrow(x)) )
  print(sub_cluster_test)
  
  
  ## handle cluster of material_id UNKNOWN o train set vuoto
  if (material_ids[cl] == 'UNKNOWN' || nrow(train_set_cl_1) == 0) {
    cat('> for this case we do not go deeper in clustering ... \n')
    
    ## define train / test set 
    cat('> for this case we use the whole train set ... \n')
    train_set_cl = train_set
    
    test_set_cl = test_set[test_set$material_id==material_ids[cl],]
    pred_cl_idx = which(test_set$material_id==material_ids[cl])
    if ( length(pred_cl_idx) != nrow(test_set_cl) ) stop('something wrong')
    
    ## using sub_base 
    cat(">> using sub_base submission ... \n")
    pred[pred_cl_idx] = sub_base[pred_cl_idx,'cost']
    
  } else {
    ## process each lev 2 cluster 
    for (qty in sub_cluster_test$quantity) {
      cat('> processing qty =', qty ,' ... \n')
      
      obs_train = sub_cluster_train[sub_cluster_train$quantity == qty , 'obs']
      zeta_train = sub_cluster_train[sub_cluster_train$quantity == qty , 'zeta']
      sd_train = sub_cluster_train[sub_cluster_train$quantity == qty , 'cost.sd']
      mean_train = sub_cluster_train[sub_cluster_train$quantity == qty , 'cost.mean']
      
      go_deeper = T 
      if (     (length(obs_train) == 0) || 
               (obs_train < MIN_OBS_TRAIN_FOR_CLUTERING_DEEPER && zeta_train < MIN_ZETA_FOR_CLUTERING_DEEPER) ) {
        cat("> not going deeper ...\n")
        go_deeper = F 
      }
      
      ## define train / test set 
      train_set_cl = NULL
      test_set_cl = NULL
      pred_cl_idx = NULL
      
      # going deeper in clustering? 
      if (go_deeper) {
        ## going deeper in clustering  
        train_set_cl = train_set[train_set$material_id==material_ids[cl] & train_set$quantity == qty ,]
      } else {
        ## not going deeper in clustering  
        train_set_cl = train_set[train_set$material_id==material_ids[cl],]
      }
      
      # the test set is the same  
      test_set_cl = test_set[test_set$material_id==material_ids[cl] & test_set$quantity == qty ,]
      pred_cl_idx = which(test_set$material_id == material_ids[cl] & test_set$quantity == qty)
      if ( length(pred_cl_idx) != nrow(test_set_cl) ) stop('something wrong')
      
      ## train and predict
      if ( (! go_deeper) || (length(sd_train) > 0 && !is.na(sd_train)) ) {
        if (go_deeper) {
          xgb = xgb_train_and_predict (train_set_cl,
                                       test_set_cl, 
                                       param,
                                       cv.nround = 3000 , 
                                       nfold = min(5,nrow(train_set_cl)) , 
                                       verbose=F)
          if (xgb$perf.cv < 11) {
            pred[pred_cl_idx] = xgb$pred
          }
          else {
            cat(">> perf.cv = ",xgb$perf.cv,"> 11 ===>>> using sub_base ... \n")
            pred[pred_cl_idx] = sub_base[pred_cl_idx,'cost']
          }
          
        } else {
          ## using sub_cluster 
          cat(">> using sub_cluster submission ... \n")
          pred[pred_cl_idx] = sub_cluster[pred_cl_idx,'cost']
        }
      } else {
        cat(">> using cluster mean as prediction ...\n")
        pred[pred_cl_idx] = rep(mean_train,length(pred_cl_idx))
      }
    } ## end of for clusters
  } ## end of if UNKNOWN
}

## basic check 
if (sum(is.na(pred)) >0) stop('something wrong')
pred_real = pred
pred = ifelse(pred<0,1.5,pred)

cat('>> number of prediction < 0:',sum(pred_real<0),' ... repleaced with 1.5 \n')

# write on disk 
cat(">> writing prediction on disk ... \n")

sample_submission$cost = pred 
write.csv(sample_submission,quote=FALSE, 
          file=paste(getBasePath("submission"),'sub_cluster_lev_2_taleb.csv',sep='') ,
          row.names=FALSE)

sample_submission$cost = pred_real 
write.csv(sample_submission,quote=FALSE, 
          file=paste(getBasePath("submission"),'sub_cluster_lev_2_taleb_real.csv',sep='') ,
          row.names=FALSE)




