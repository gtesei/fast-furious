library(data.table)
library(xgboost)
library(fastfurious)
library(caret)

### CONFIG 
DEBUG = F

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/springleaf-marketing-respons')
ff.bindPath(type = 'code' , sub_path = 'competitions/springleaf-marketing-respons')
ff.bindPath(type = 'elab' , sub_path = 'dataset/springleaf-marketing-respons/elab') 

ff.bindPath(type = 'out' , sub_path = 'dataset/springleaf-marketing-respons/elab/unbalanced_luck' , createDir = T) ## out  

####
source(paste0(ff.getPath("code"),"fastClassification.R"))

##############

cat(">>> adjusting resamples for unbalanced classes ... \n")

## Ytrain / test_id 
cat (">>> loading Ytrain / test_id ... \n")
train_raw = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
test_raw = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
Ytrain = train_raw$target
test_id = test_raw$ID 
rm(list=c("train_raw","test_raw"))
gc()

cat (">>> loading whole Xtrain_docproc2.csv / Xtest_docproc2.csv... \n")
## Xtrain / Xtest 
Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_docproc2.csv" , sep='') , stringsAsFactors = F))
Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_docproc2.csv" , sep='') , stringsAsFactors = F))

## redo resamples 
index = caret::createMultiFolds(y=Ytrain, 4, 1)
indexOut <- lapply(index, function(training, allSamples) allSamples[-unique(training)], allSamples = seq(along = Ytrain))
controlObject = trainControl(method = "repeatedcv", 
                             ## The method doesn't really matter
                             ## since we defined the resamples
                             index = index, 
                             indexOut = indexOut , 
                             summaryFunction = twoClassSummary , classProbs = TRUE)
rm(list = c("index","indexOut"))

## filter resamples
for (fI in seq_along(controlObject$index)) {
  
  idx1 = which( Ytrain[controlObject$index[fI][[1]]] == 1)
  idx0 = which( Ytrain[controlObject$index[fI][[1]]] == 0)
  
  inr = length(idx1) / length(idx0)
  
  l0 = 1.5*length(idx1) ## ratio = 0.6666667 (instead of 0.3333333)
  
  idx0_new = sample(x = idx0 , size = l0 , replace = F)
  
  inr = length(idx1) / length(idx0_new)
  
  stopifnot(length(idx0_new)==length(unique(idx0_new)))
  
  controlObject$index[fI][[1]] = c(idx1,idx0_new)
}

### train / predict 
for (i in seq_along(controlObject$index)) { 
  cat(">>> train / predict Fold:",i,"...\n") 
  
  ##
  l = fastfurious:::getCaretFactors(y=Ytrain)
  y.cat = l$y.cat
  fact.sign = l$fact.sign
  rm(l)
  
  ##
  train_i =  Xtrain[ controlObject$index[[i]] , ]
  y_i = y.cat[ controlObject$index[[i]] ]
  test_i = Xtrain[ controlObject$indexOut[[i]] , ]
  y_test_i =  Ytrain[ controlObject$indexOut[[i]] ]
  
  internalControlObject = caret::trainControl(method = "none", summaryFunction = twoClassSummary , classProbs = TRUE)
  
  model <- caret::train(y = y_i, x = train_i ,
                        method = 'xgbTree',
                        tuneGrid = expand.grid(
                          nrounds = 1173 ,
                          max_depth = 8 ,  
                          eta = 0.02 ),
                        trControl = internalControlObject , 
                        
                        ## .. 
                        objective = "binary:logistic",
                        eval_metric = "auc", 
                        subsample = 0.7 , 
                        colsample_bytree = 0.6 , 
                        scale_pos_weight = 0.8 , 
                        max_delta_step = 2)
  
  pred_test_i = predict(model,test_i,type = "prob")[,fact.sign]
  roc_i = verification::roc.area(y_test_i , pred_test_i )$A
  cat(">>> xval AUC[",i,"]:",roc_i,"\n")
  pred_test =  predict(model,Xtest,type = "prob")[,fact.sign]
  
  
  ## sub 
  submission <- data.frame(ID=test_id)
  submission$target <- pred_test
  print(head(submission))
  write.csv(submission,
            quote=FALSE, 
            file= paste( ff.getPath("out") , "pred_fold_",  i , "_auc" , roc_i , ".csv",sep='') ,
            row.names=FALSE)
}

## avg 
sub_1 = as.data.frame( fread(paste(ff.getPath("out") , "pred_fold_1_auc0.899502947150638.csv" , sep='') , stringsAsFactors = F))
sub_2 = as.data.frame( fread(paste(ff.getPath("out") , "pred_fold_2_auc0.896937271325493.csv" , sep='') , stringsAsFactors = F))

sub_3 = as.data.frame( fread(paste(ff.getPath("out") , "pred_fold_3_auc0.896093669485193.csv" , sep='') , stringsAsFactors = F))
sub_4 = as.data.frame( fread(paste(ff.getPath("out") , "pred_fold_4_auc0.898864626043946.csv" , sep='') , stringsAsFactors = F))

submission <- data.frame(ID=sub_1$ID)
submission$target <- (sub_1$target + sub_2$target + sub_3$target + sub_4$target)/4

print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file= paste( ff.getPath("out") , "pred_mean.csv",sep='') ,
          row.names=FALSE)


sub_best = as.data.frame( fread(paste(ff.getPath("out") , "layer3_dataProcdoc_default_modffOctNNet_tuneTRUE.csv" , sep='') , stringsAsFactors = F))

submission$target <- (submission$target + sub_best$target)/2
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file= paste( ff.getPath("out") , "pred_mean_final.csv",sep='') ,
          row.names=FALSE)


