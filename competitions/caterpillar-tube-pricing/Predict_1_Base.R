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
    
    cat(">> cv.nround: ",cv.nround,"\n") 
    bst.cv = xgb.cv(param=param, data = data , label = y, 
                    nfold = nfold, nrounds=cv.nround , verbose=verbose)
    print(bst.cv)
    early.stop = which(bst.cv$test.rmse.mean == min(bst.cv$test.rmse.mean) )
    if (length(early.stop)>1) early.stop[length(early.stop)]
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

################# FAST-FURIOUS SOURCES
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))
source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))

################# DATA IN 
sample_submission = as.data.frame( fread(paste(getBasePath("data") , 
                                               "sample_submission.csv" , sep=''))) 


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
sub_base = NULL 

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
  "quote_date"     ,     "annual_usage"   ,     "min_order_quantity"  , # "quantity"       ,      
  "diameter"       ,     "wall"         ,       "length"              , "num_bends"      ,    "bend_radius"     ,    
  "num_boss"       ,     "num_bracket"  ,      
  "CP_001_weight"  ,     "CP_002_weight"  ,     "CP_003_weight"  ,     "CP_004_weight"  ,     "CP_005_weight"   ,    "CP_006_weight"    ,  
  "CP_007_weight"  ,     "CP_008_weight"  ,     "CP_009_weight"  ,     "CP_010_weight"   ,    "CP_011_weight"   ,    "CP_012_weight"    ,  
  "CP_014_weight"  ,     "CP_015_weight"  ,     "CP_016_weight"  ,     "CP_017_weight"  ,     "CP_018_weight"  ,     "CP_019_weight"   ,   
  "CP_020_weight"  ,     "CP_021_weight"  ,     "CP_022_weight"  ,     "CP_023_weight"  ,     "CP_024_weight"  ,     "CP_025_weight"   ,   
  "CP_026_weight"  ,     "CP_027_weight"  ,     "CP_028_weight"  ,     "CP_029_weight"  ,     "OTHER_weight"  
)

trans.scal <- preProcess(rbind(train_set[,feature2scal],test_set[,feature2scal]),
                         method = c("center", "scale") )

print(trans.scal)

train_set[,feature2scal] = predict(trans.scal,train_set[,feature2scal])
test_set[,feature2scal] = predict(trans.scal,test_set[,feature2scal])

######### 

##################### XGB 

## tube_assembly_id , id 
train_set[, 'tube_assembly_id'] = NULL 
test_set [, 'tube_assembly_id'] = NULL 
test_set [, 'id'] = NULL 

## material_id 
# cat(">>> encoding material_id [",unique(c(train_set$material_id , test_set$material_id)),"] [",length(unique(c(train_set$material_id , test_set$material_id))),"] ... \n")
# l = encodeCategoricalFeature (train_set$material_id , test_set$material_id , colname.prefix = "material_id" , asNumeric=F)
# cat(">>> train_set before encoding:",ncol(train_set)," - test_set before encoding:",ncol(test_set)," ... \n")
# train_set = cbind(train_set , l$traindata)
# test_set = cbind(test_set , l$testdata)

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

cat(">>Params:\n")
print(param)

#### Run Cross Valication
cat(">>Cross validation ... \n")

xgb_xval = xgb_cross_val (data = x[trind,], 
                          y = y,  
                          cv.nround = 56000 , 
                          nfold = 5 , 
                          verbose=F)

#### Prediction 
cat(">> Train the model for prediction ... \n")
bst = xgboost(param=param, 
              data = x[trind,], 
              label = y, 
              nrounds=xgb_xval$early.stop,
              verbose=F)

# Make prediction
pred = predict(bst,x[teind,])
pred_real = pred
pred = ifelse(pred<0,1.5,pred)

cat('>> number of prediction < 0:',sum(pred_real<0),'\n')

# write on disk 
cat(">> writing prediction on disk ... \n")

sample_submission$cost = pred 
write.csv(sample_submission,quote=FALSE, 
          file=paste(getBasePath("submission"),'sub_base_date_no_0var_preds.csv',sep='') ,
          row.names=FALSE)

# sample_submission$cost = pred_real 
# write.csv(sample_submission,quote=FALSE, 
#           file=paste(getBasePath("submission"),'sub_base_date_real.csv',sep='') ,
#           row.names=FALSE)

