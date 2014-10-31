
## notes_seizure_R
library(caret)
library(Hmisc)
library(data.table)
library(verification)
library(pROC)
library(kernlab)
library(subselect)

getBasePath = function (type = "data" , ds="") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/seizure-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/seizure-prediction/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/data_pre_process"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/data_pre_process/"
  } else {
    stop("unrecognized type.")
  }
  
  if (file.exists(base.path1))  {
    ret = paste0(base.path1,"/")
  } else {
    ret = base.path2
  }
  
  if (ds != "" ) {
    if (ds == "Patient_2") {
      cat("data from 5gen ... \n")
      ret = paste0(paste0(ret,ds),"_digest_5gen/")
    } else {
      cat("data from 4gen ... \n")
      ret = paste0(paste0(ret,ds),"_digest_4gen/")
    }
  }
  ret
} 

buildPCAFeatures = function(Xtrain_mean_sd,Xtest_mean_sd,Xtrain_quant,Xtest_quant,verbose) {
  Xtrain_pca = as.data.frame(fread(('/Users/gino/Documents/Kaggle/fast-furious/gitHub/fast-furious/dataset/seizure-prediction/Patient_2_pca_feature/Xtrain_pca.zat')))
  Xtest_pca = as.data.frame(fread(('/Users/gino/Documents/Kaggle/fast-furious/gitHub/fast-furious/dataset/seizure-prediction/Patient_2_pca_feature/Xtest_pca.zat')))
  
  colnames(Xtrain_pca) = paste("pca",rep(1:(ncol(Xtrain_pca))) , sep = "")
  colnames(Xtest_pca) = paste("pca",rep(1:(ncol(Xtest_pca))) , sep = "")
  
  Xtrain_mean_sd = cbind(Xtrain_mean_sd,Xtrain_pca)
  Xtrain_quant = cbind(Xtrain_quant,Xtrain_pca)
  
  Xtest_mean_sd = cbind(Xtest_mean_sd,Xtest_pca)
  Xtest_quant = cbind(Xtest_quant,Xtest_pca)
  
#     Xtrain_mean_sd = Xtrain_pca
#     Xtrain_quant = Xtrain_pca
#     
#     Xtest_mean_sd = Xtest_pca 
#     Xtest_quant = Xtest_pca
  
  return(list(Xtrain_mean_sd,Xtrain_quant,Xtest_mean_sd,Xtest_quant))
}

buildRelativeSpectralPowerFeatures = function(Xtrain_mean_sd,Xtest_mean_sd,Xtrain_quant,Xtest_quant,ytrain,NN=10,verbose) {
  ########
  #   1- Xtrain_mean_sd(train_index,((i-1)*16+1)) = mu;
  #   2 - Xtrain_mean_sd(train_index,((i-1)*16+2)) = sd;
  #   3 - Xtrain_mean_sd(train_index,((i-1)*16+3)) = fm;
  #   4 - Xtrain_mean_sd(train_index,((i-1)*16+4)) = P;
  
  #   5 - Xtrain_mean_sd(train_index,((i-1)*16+5)) = p0; 
  #   6 - Xtrain_mean_sd(train_index,((i-1)*16+6)) = p1;
  #   7 - Xtrain_mean_sd(train_index,((i-1)*16+7)) = p2;
  #   8 - Xtrain_mean_sd(train_index,((i-1)*16+8)) = p3;
  #   9 - Xtrain_mean_sd(train_index,((i-1)*16+9)) = p4;
  #   10 - Xtrain_mean_sd(train_index,((i-1)*16+10)) = p5;
  #   11 - Xtrain_mean_sd(train_index,((i-1)*16+11)) = p6;
  #   12 - Xtrain_mean_sd(train_index,((i-1)*16+12)) = pTail; 
  
  #   13 - Xtrain_mean_sd(train_index,((i-1)*16+13)) = f_50;
  #   14 - Xtrain_mean_sd(train_index,((i-1)*16+14)) = min_tau;
  #   15 - Xtrain_mean_sd(train_index,((i-1)*16+15)) = skw;
  #   16 - Xtrain_mean_sd(train_index,((i-1)*16+16)) = kur;
  
  P.mean = apply(Xtrain_mean_sd,2,mean)
  idx.preict = (ytrain[,2] == 1)
  P.mean.inter = apply(Xtrain_mean_sd[! idx.preict , ],2,mean)
  P.mean.preict = apply(Xtrain_mean_sd[idx.preict , ],2,mean)
  delta.mean =  ((P.mean.inter -  P.mean.preict) / P.mean)
  idx.delta.mean.asc = order(delta.mean)
  idx.delta.mean.desc = order(delta.mean, decreasing = T)
  
  delta.mean[idx.delta.mean.asc]
  delta.mean[idx.delta.mean.desc]
  
  
  ## solo le features di interesse 
  idx.filter = idx.delta.mean.asc %% 16 == 5 | idx.delta.mean.asc %% 16 == 6 | idx.delta.mean.asc %% 16 == 7 |
    idx.delta.mean.asc %% 16 == 8 | idx.delta.mean.asc %% 16 == 9 | idx.delta.mean.asc %% 16 == 10 |
    idx.delta.mean.asc %% 16 == 11 | idx.delta.mean.asc %% 16 == 12 
  
  idx.delta.mean.asc = idx.delta.mean.asc[idx.filter]
  idx.delta.mean.desc = idx.delta.mean.desc[idx.filter]
  
  ## fill train 
  feat.mat = matrix(rep(-1,nrow(Xtrain_mean_sd)*NN*NN),nrow = nrow(Xtrain_mean_sd) , ncol = NN*NN)
  for (idx.num in 1:NN) {
    for (idx.denum in 1:NN) {
      feat.mat[,idx.denum+NN*(idx.num-1)] =  (Xtrain_mean_sd[,idx.delta.mean.asc[idx.num]] / mean(Xtrain_mean_sd[,idx.delta.mean.asc[idx.num]])) / (Xtrain_mean_sd[,idx.delta.mean.desc[idx.denum]]/mean(Xtrain_mean_sd[,idx.delta.mean.desc[idx.denum]]))
    }
  }
  
  new.features.train = as.data.frame(feat.mat) 
  colnames(new.features.train) = paste("relspect",rep(1:(ncol(feat.mat))) , sep = "")
  new.features.train = new.features.train[,1:10]
  
  if (verbose) {
    for (yy in 1:10) {
      cat("----------------------------------------------------------------- \n")
      cat("-- [train] mean of relspect feature n. ", yy , "=",as.character(mean (new.features.train[,yy]))  , "\n")
      cat("-- [train] mean of relspect feature (preict) n. ", yy  ,"=",as.character(mean (new.features.train[idx.preict,yy])) , "\n")
      cat("-- [train] mean of relspect feature (interict) n. ", yy  ,"=",as.character(mean (new.features.train[!idx.preict,yy])) , "\n")
    }
  }
  
  ## fill test 
  feat.mat = matrix(rep(-1,nrow(Xtest_mean_sd)*NN*NN),nrow = nrow(x = Xtest_mean_sd) , ncol = NN*NN)
  for (idx.num in 1:NN) {
    for (idx.denum in 1:NN) {
      feat.mat[,idx.denum+NN*(idx.num-1)] =  (Xtest_mean_sd[,idx.delta.mean.asc[idx.num]]/mean(Xtest_mean_sd[,idx.delta.mean.asc[idx.num]])) / (Xtest_mean_sd[,idx.delta.mean.desc[idx.denum]]/mean(Xtest_mean_sd[,idx.delta.mean.desc[idx.denum]]))
    }
  }
  
  new.features.test = as.data.frame(feat.mat) 
  colnames(new.features.test) = paste("relspect",rep(1:(ncol(feat.mat))) , sep = "")
  new.features.test = new.features.test[,1:10]
  
  ###### merge 
  Xtrain_mean_sd = cbind(Xtrain_mean_sd,new.features.train)
  Xtest_mean_sd = cbind(Xtest_mean_sd , new.features.test)
  
  Xtrain_quant = cbind(Xtrain_quant,new.features.train)
  Xtest_quant = cbind(Xtest_quant,new.features.test)
  
  return( list(Xtrain_mean_sd,Xtest_mean_sd,Xtrain_quant,Xtest_quant) )
} 


trainAndPredict = function(ds,model.label,model.id,
                           Xtrain, ytrain.cat,
                           Xtest,
                           verbose=F) {
  ## model 
  if (model.id >= 1 && model.id <= 6) { ## logistic reg 
    model <- train( x = Xtrain , y = ytrain.cat , 
                    method = "glm", metric = "ROC", trControl = controlObject)
  } else if (model.id >= 7 && model.id <= 12) { ## lda 
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "lda", metric = "ROC" , trControl = controlObject)
  } else if (model.id >= 13 && model.id <= 18) { ## plsda 
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "pls", tuneGrid = expand.grid(.ncomp = 1:10), 
                    metric = "ROC" , trControl = controlObject)
  } else if (model.id >= 19 && model.id <= 24) { ## pm 
    glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1), .lambda = seq(.01, .2, length = 40))
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "glmnet", tuneGrid = glmnGrid, 
                    metric = "ROC", trControl = controlObject)
  } else if (model.id >= 25 && model.id <= 30) { ## nsc 
    nscGrid <- data.frame(.threshold = 0:25)
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "pam", tuneGrid = nscGrid, 
                    metric = "ROC", trControl = controlObject)
  } else if (model.id >= 31 && model.id <= 36) { # neural networks 
    nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
    maxSize <- max(nnetGrid$.size)
    numWts <- 1*(maxSize * ( (dim(Xtrain)[2]) + 1) + maxSize + 1)
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "nnet", metric = "ROC", 
                    preProc = c( "spatialSign") , 
                    tuneGrid = nnetGrid , trace = FALSE , maxit = 2000 , 
                    MaxNWts = numWts, trControl = controlObject)
  } else if (model.id >= 37 && model.id <= 42) { ## svm 
    sigmaRangeReduced <- sigest(as.matrix(Xtrain))
    svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1], .C = 2^(seq(-6, 6,by=0.2)))
    if (ds == "Patient_2") {
      cat("<<Patient_2>> setting class.weights = c(preict = 2, interict = 1) .... \n")
      model <- train( x = Xtrain , y = ytrain.cat,  
                      method = "svmRadial", tuneGrid = svmRGridReduced, 
                      class.weights = c(preict = 2, interict = 1),
                      metric = "ROC", fit = FALSE, trControl = controlObject)
    } else if (ds == "Dog_2") { 
      cat("<<Dog_2>> setting class.weights = c(preict = 13, interict = 1) .... \n")
      model <- train( x = Xtrain , y = ytrain.cat,  
                      method = "svmRadial", tuneGrid = svmRGridReduced, 
                      class.weights = c(preict = 13, interict = 1),
                      metric = "ROC", fit = FALSE, trControl = controlObject)
      } else{
      model <- train( x = Xtrain , y = ytrain.cat,  
                      method = "svmRadial", tuneGrid = svmRGridReduced, 
                      metric = "ROC", fit = FALSE, trControl = controlObject)
    }
  } else if (model.id >= 43 && model.id <= 49) { ## knn 
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "knn", 
                    tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)),
                    metric = "ROC",  trControl = controlObject)
  } else if (model.id >= 49 && model.id <= 54) { ## class trees 
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "rpart", tuneLength = 30, 
                    metric = "ROC", trControl = controlObject)
  } else if (model.id >= 55 && model.id <= 60) { ## boosted trees 
    if (model.id >= 55 && model.id <= 59) {
      ## 55 - 59 
      model <- train( x = Xtrain , y = ytrain.cat,  
                      method = "C5.0",  metric = "ROC", trControl = controlObject)
    } else {
      ## 60 - BOOSTED_TREE_QUANTILES_REDUCED 
      model <- train( x = Xtrain , y = ytrain.cat,  
                      tuneGrid = expand.grid(.trials = c(1, (1:10)*10), .model = "tree", .winnow = c(TRUE, FALSE) ),
                      method = "C5.0",  metric = "ROC", trControl = controlObject)
    }
  } else if (model.id >= 61 && model.id <= 66) { ## bagging trees 
    if (model.id >= 61 && model.id <= 65) {
      ## 61 - 65 
      model <- train( x = Xtrain , y = ytrain.cat,  
                      method = "bag",  metric = "ROC", trControl = controlObject, B = 50 ,
                      bagControl = bagControl(fit = plsBag$fit,
                                              predict = plsBag$pred,
                                              aggregate = plsBag$aggregate))
    } else {
      ## 66 - BAGGING_TREE_QUANTILES_REDUCED
      model <- train( x = Xtrain , y = ytrain.cat,  
                      method = "bag",  metric = "ROC", trControl = controlObject, 
                      tuneGrid = data.frame(vars = seq(1, 15, by = 2)), 
                      bagControl = bagControl(fit = plsBag$fit,
                                              predict = plsBag$pred,
                                              aggregate = plsBag$aggregate))
    }
  } else if (model.id == 0) { ## null prediction, i.e. pred = (0,0,0,...,0)
    model = NULL
  } else {
    stop("ma che modello ha vinto (modello) !! ")
  }
  
  ## predicting model on Xtrain and Xtest 
  pred.prob.train = pred.train = pred.prob.test = pred.test = NULL
  if (! is.null(model)) {
    pred.prob.train = predict(model , Xtrain , type = "prob")[,'preict'] 
    pred.train = predict(model , Xtrain )
    
    pred.prob.test = predict(model , Xtest , type = "prob")[,'preict'] ### <<<<<<<<<<<<----------------------------------
    pred.test = predict(model , Xtest )
  } else {
    pred.prob.train = rep(0,nrow(Xtrain))
    pred.train = rep(0,nrow(Xtrain))
    
    pred.prob.test = rep(0,nrow(Xtest)) ### <<<<<<<<<<<<----------------------------------
    pred.test = rep(0,nrow(Xtest))
  }
  
  ## accuracy 
  acc.train.all0 = sum(factor(rep('interict',length(ytrain.cat)) , levels = levels(ytrain.cat) ) == ytrain.cat) / length(ytrain.cat)
  acc.train = sum(ytrain.cat == pred.train) / length(ytrain.cat) 
  
  ## ROC 
  rocCurve <- pROC::roc(response = ytrain.cat, predictor = pred.prob.train, levels = levels(ytrain.cat) )
  roc.train = as.numeric( pROC::auc(rocCurve) )
  
  roc.train.2 = roc.area(as.numeric(ytrain.cat == 'preict') , pred.prob.train )$A
  roc.xval.min = min(roc.train,roc.train.2)
  
  ## logging 
  if (verbose) cat("******************* ", model.label, " <<" , model.id ,  ">> \n")
  if (verbose) cat("** acc.train =",acc.train, " -  acc.train.all0 =",acc.train.all0, " \n")
  if (verbose) cat("** roc.train =",roc.train," -  roc.train.2 =",roc.train.2,"  \n")
  if (verbose) cat("** roc.xval.min =",roc.xval.min, " \n")
  if (verbose) cat("** Train set:  true %preict =", as.character(sum(ytrain.cat == 'preict')/length(ytrain.cat)) ,  " - %interict =" 
                   , as.character(sum(ytrain.cat == 'interict')/length(ytrain.cat)) ,  " \n")
  if (verbose) cat("** Train set:  predicted %preict =", as.character(sum(pred.train == 'preict')/length(pred.train)) ,  " - %interict =" 
                   , as.character(sum(pred.train == 'interict')/length(pred.train)) ,  " \n")
  if (verbose) cat("** Test set:  predicted %preict =", as.character(sum(pred.test == 'preict')/length(pred.test)) ,  " - %interict =" 
                   , as.character(sum(pred.test == 'interict')/length(pred.test)) ,  " \n")
  
  list(pred.prob.train, pred.train, pred.prob.test, pred.test)
}

######################################################## CONSTANTS 
NULL_MODEL = 0

LOGISTIC_REG_MEAN_SD = 1 
LOGISTIC_REG_QUANTILES = 2
LOGISTIC_REG_MEAN_SD_SCALED = 3
LOGISTIC_REG_QUANTILES_SCALED = 4
LOGISTIC_REG_MEAN_SD_REDUCED = 5
LOGISTIC_REG_QUANTILES_REDUCED = 6

LDA_MEAN_SD = 7
LDA_QUANTILES = 8
LDA_MEAN_SD_SCALED = 9 
LDA_QUANTILES_SCALED= 10 
LDA_MEAN_SD_REDUCED = 11
LDA_REG_QUANTILES_REDUCED = 12 

PLSDA_MEAN_SD = 13
PLSDA_QUANTILES = 14
PLSDA_MEAN_SD_SCALED = 15 
PLSDA_QUANTILES_SCALED= 16 
PLSDA_MEAN_SD_REDUCED = 17
PLSDA_REG_QUANTILES_REDUCED = 18 

PM_MEAN_SD = 19
PM_QUANTILES = 20
PM_MEAN_SD_SCALED = 21 
PM_QUANTILES_SCALED= 22 
PM_MEAN_SD_REDUCED = 23
PM_REG_QUANTILES_REDUCED = 24

NSC_MEAN_SD = 25
NSC_QUANTILES = 26
NSC_MEAN_SD_SCALED = 27 
NSC_QUANTILES_SCALED= 28 
NSC_MEAN_SD_REDUCED = 29
NSC_REG_QUANTILES_REDUCED = 30

NN_MEAN_SD = 31
NN_QUANTILES = 32
NN_MEAN_SD_SCALED = 33 
NN_QUANTILES_SCALED= 34 
NN_MEAN_SD_REDUCED = 35
NN_QUANTILES_REDUCED = 36

SVM_MEAN_SD = 37
SVM_QUANTILES = 38
SVM_MEAN_SD_SCALED = 39 
SVM_QUANTILES_SCALED= 40 
SVM_MEAN_SD_REDUCED = 41
SVM_QUANTILES_REDUCED = 42

KNN_MEAN_SD = 43
KNN_QUANTILES = 44
KNN_MEAN_SD_SCALED = 45 
KNN_QUANTILES_SCALED= 46 
KNN_MEAN_SD_REDUCED = 47
KNN_QUANTILES_REDUCED = 48

CLASS_TREE_MEAN_SD = 49
CLASS_TREE_QUANTILES = 50
CLASS_TREE_MEAN_SD_SCALED = 51 
CLASS_TREE_QUANTILES_SCALED= 52 
CLASS_TREE_MEAN_SD_REDUCED = 53
CLASS_TREE_QUANTILES_REDUCED = 54

BOOSTED_TREE_MEAN_SD = 55
BOOSTED_TREE_QUANTILES = 56
BOOSTED_TREE_MEAN_SD_SCALED = 57 
BOOSTED_TREE_QUANTILES_SCALED= 58 
BOOSTED_TREE_MEAN_SD_REDUCED = 59
BOOSTED_TREE_QUANTILES_REDUCED = 60

BAGGING_TREE_MEAN_SD = 61
BAGGING_TREE_QUANTILES = 62
BAGGING_TREE_MEAN_SD_SCALED = 63 
BAGGING_TREE_QUANTILES_SCALED= 64 
BAGGING_TREE_MEAN_SD_REDUCED = 65
BAGGING_TREE_QUANTILES_REDUCED = 66

######################################################## MAIN  
sampleSubmission = as.data.frame(fread(paste(getBasePath(type = "data"),"sampleSubmission.csv",sep="") , header = T , sep=","  ))

trainClass = NULL
############# general settings ... 
verbose = T
doPlot = F 

superFeature = F

Patient_2.pca = F

SUB_DIR = "svm_weigth_only_dog_2"
if (SUB_DIR != "") {
  cat("creating directory <<",SUB_DIR,">> ... \n")
  SUB_DIR = paste0(SUB_DIR,"/")
  dir.create(paste(getBasePath(),SUB_DIR,sep=""))
}

############ models grids 
Dog_1.model = data.frame(model = c( "NN_QUANTILES_REDUCED" ) , 
                         model.id = c(NN_QUANTILES_REDUCED ) , 
                         weigth = c(0.71)) 

Dog_2.model = data.frame(model = c("SVM_QUANTILES_REDUCED" ) , 
                         model.id = c(SVM_QUANTILES_REDUCED ) , 
                         weigth = c(0.71)) 

Dog_3.model = data.frame(model = c("SVM_MEAN_SD_SCALED" ) , 
                         model.id = c(SVM_MEAN_SD_SCALED ) , 
                         weigth = c(0.71 )) 

Dog_4.model = data.frame(model = c("SVM_QUANTILES_REDUCED" ) , 
                         model.id = c(SVM_QUANTILES_REDUCED ) , 
                         weigth = c(0.71)) 

Dog_5.model = data.frame(model = c("NN_MEAN_SD_REDUCED" ) , 
                         model.id = c(NN_MEAN_SD_REDUCED ) , 
                         weigth = c(0.71 )) 

Patient_1.model = data.frame(model = c("KNN_QUANTILES_SCALED" ) , 
                             model.id = c(KNN_QUANTILES_SCALED ) , 
                             weigth = c(0.71 )) 

Patient_2.model = data.frame(model = c("SVM_MEAN_SD_REDUCED" ) , 
                             model.id = c(SVM_MEAN_SD_REDUCED ) , 
                             weigth = c(0.71 )) 

### check 
models.per.ds = nrow(Dog_1.model)
if (nrow(Dog_2.model) != models.per.ds | 
    nrow(Dog_3.model) != models.per.ds |
    nrow(Dog_4.model) != models.per.ds |
    nrow(Dog_5.model) != models.per.ds |
    nrow(Patient_1.model) != models.per.ds |
    nrow(Patient_2.model) != models.per.ds) stop("number of model per data set must be equal.")


## train models 
Dog_1.model.train = Dog_1.model
Dog_2.model.train = Dog_2.model
Dog_3.model.train = Dog_3.model
Dog_4.model.train = Dog_4.model
Dog_5.model.train = Dog_5.model
Patient_1.model.train = Patient_1.model
Patient_2.model.train = Patient_2.model

if (ncol(Dog_1.model.train) > 3) stop("Dog_1.model.train has a wrong number of columns")

############ 

### resampling method 
controlObject <- trainControl(method = "boot", number = 100 , 
                              summaryFunction = twoClassSummary , classProbs = TRUE)

# controlObject <- trainControl(method = "boot632", number = 100 , 
#                               summaryFunction = twoClassSummary , classProbs = TRUE)
# 
# controlObject <- trainControl(method = "repeatedcv", number = 10 , repeats = 20 , 
#                               summaryFunction = twoClassSummary , classProbs = TRUE)
# 
# controlObject <- trainControl(method = "LOOCV" , 
#                               summaryFunction = twoClassSummary , classProbs = TRUE)

### data sets to process 
dss = c("Dog_1","Dog_2","Dog_3","Dog_4","Dog_5","Patient_1","Patient_2")
##dss = c("Patient_2")
cat("|---------------->>> data set to process: <<",dss,">> ..\n")

source(paste0(getBasePath("code") , "SelectBestPredictors_Lib.R"))

### completing models grids
for (ds in dss) {
  
  cat("|---------------->>> processing data set <<",ds,">> ..\n")
  
  ######### loading data sets ...
  Xtrain_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"Xtrain_mean_sd.zat",sep="") , header = F , sep=","  ))
  Xtrain_quant = as.data.frame( fread(paste(getBasePath(type = "data" , ds=ds),"Xtrain_quant.zat",sep="") , header = F , sep=","  ))
  
  Xtest_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"Xtest_mean_sd.zat",sep="") , header = F , sep=","  ))
  Xtest_quant = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"Xtest_quant.zat",sep="") , header = F , sep=","  ))
  
  ytrain = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"ytrain.zat",sep="") , header = F , sep=","  ))
  
  ## names 
  colnames(Xtrain_mean_sd)  = colnames(Xtest_mean_sd) = paste("fmeansd",rep(1:((dim(Xtrain_mean_sd)[2]))) , sep = "")
  colnames(Xtrain_quant) = colnames(Xtest_quant) = paste("fquant",rep(1:((dim(Xtrain_quant)[2]))) , sep = "")
  
  ytrain.cat = factor(ytrain[,2]) 
  levels(ytrain.cat) = c("interict","preict")
  
  ###### super-features 
  if (superFeature) {
    l = buildRelativeSpectralPowerFeatures (Xtrain_mean_sd,Xtest_mean_sd,Xtrain_quant,Xtest_quant,ytrain,NN=10,verbose)
    Xtrain_mean_sd = l[[1]]
    Xtest_mean_sd = l[[2]]
    Xtrain_quant = l[[3]]
    Xtest_quant = l[[4]]
  }
  
  if (ds == "Patient_2" && Patient_2.pca) {
    cat("building PCA features ... \n") 
    l = buildPCAFeatures (Xtrain_mean_sd,Xtest_mean_sd,Xtrain_quant,Xtest_quant,verbose)
    Xtrain_mean_sd  = l[[1]]
    Xtrain_quant = l[[2]]
    Xtest_mean_sd = l[[3]]
    Xtest_quant = l[[4]]   
  }
  
  ### check 
  if ( length(grep(ds , sampleSubmission$clip)) != nrow(Xtest_mean_sd) | 
       length(grep(ds , sampleSubmission$clip)) != nrow(Xtest_quant)   )  stop ("rows in sample submission different for test set!")
  
  ###################### finding best predictors 
#   if (verbose) cat("finding min p-value predictors for meand_sd series ... \n")
#   predictors.class.linear = getPvalueFeatures( features = Xtrain_mean_sd , response = ytrain.cat , p = 1 , 
#                                                pValueAdjust = F, pValueAdjustMethod = "default", 
#                                                verbose = verbose)
#   predictors.class.linear = predictors.class.linear[order(predictors.class.linear$pValue,decreasing = F),]
#   
#   l = getPredictorsMinPvalue(predPvalues = predictors.class.linear , data = Xtrain_mean_sd , th = 0.05 , verbose = verbose)
#   var.name = l[[1]]
#   var.index = l[[2]]
#   
#   Xtrain_mean_sd  = Xtrain_mean_sd[,var.index]
#   Xtest_mean_sd = Xtest_mean_sd[,var.index]
#   
#   if (verbose) cat("finding min p-value predictors for quant series ... \n")
#   predictors.class.linear = getPvalueFeatures( features = Xtrain_quant , response = ytrain.cat , p = 1 , 
#                                                pValueAdjust = F, pValueAdjustMethod = "default", 
#                                                verbose = verbose)
#   predictors.class.linear = predictors.class.linear[order(predictors.class.linear$pValue,decreasing = F),]
#   
#   l = getPredictorsMinPvalue(predPvalues = predictors.class.linear , data = Xtrain_quant , th = 0.05 , verbose = verbose)
#   var.name = l[[1]]
#   var.index = l[[2]]
#   
#   Xtrain_quant  = Xtrain_quant[,var.index]
#   Xtest_quant = Xtest_quant[,var.index]
  ###################### end finding best predictors 
  
  ######### making train / xval set ...
  ### removing predictors that make ill-conditioned square matrix
  PredToDel = trim.matrix( cov( Xtrain_quant ) )
  if (length(PredToDel$numbers.discarded) > 0) {
    cat("removing ",length(PredToDel$numbers.discarded)," predictors that make ill-conditioned square matrix: ", paste(colnames(Xtrain_quant) [PredToDel$numbers.discarded] , collapse=" " ) , " ... \n ")
    Xtest_quant  =  Xtest_quant  [,-PredToDel$numbers.discarded]
    Xtrain_quant =  Xtrain_quant [,-PredToDel$numbers.discarded]
  }
  PredToDel = trim.matrix( cov( Xtrain_mean_sd ) )
  if (length(PredToDel$numbers.discarded) > 0) {
    cat("removing ",length(PredToDel$numbers.discarded)," predictors that make ill-conditioned square matrix: ", paste(colnames(Xtrain_mean_sd) [PredToDel$numbers.discarded] , collapse=" " ) , " ... \n ")
    Xtest_mean_sd  =  Xtest_mean_sd  [,-PredToDel$numbers.discarded]
    Xtrain_mean_sd =  Xtrain_mean_sd [,-PredToDel$numbers.discarded]
  }
  
  ### removing near zero var predictors 
  PredToDel = nearZeroVar(Xtrain_quant)
  if (length(PredToDel) > 0) {
    cat("removing ",length(PredToDel)," nearZeroVar predictors: ", paste(colnames(Xtrain_quant) [PredToDel] , collapse=" " ) , " ... \n ")
    Xtest_quant  =  Xtest_quant  [,-PredToDel]
    Xtrain_quant =  Xtrain_quant [,-PredToDel]
  }
  
  PredToDel = nearZeroVar(Xtrain_mean_sd)
  if (length(PredToDel) > 0) {
    cat("removing ",length(PredToDel)," nearZeroVar predictors: ", paste(colnames(Xtrain_mean_sd) [PredToDel] , collapse=" " ) , " ... \n ")
    Xtest_mean_sd  =  Xtest_mean_sd  [,-PredToDel]
    Xtrain_mean_sd =  Xtrain_mean_sd [,-PredToDel]
  }
  ### A. reduced+scaled  
  Xtest_quant.reduced = Xtrain_quant.reduced = NULL 
  Xtest_mean_sd.reduced = Xtrain_mean_sd.reduced = NULL 
  
  # rmoving high correlated predictors on Xtrain_quant
  PredToDel = findCorrelation(cor( Xtrain_quant )) 
  cat("PLS:: on Xtrain_quant removing ",length(PredToDel), " predictors: ",paste(colnames(Xtrain_quant) [PredToDel] , collapse=" " ) , " ... \n ")
  Xtest_quant.reduced =  Xtest_quant  [,-PredToDel]
  Xtrain_quant.reduced = Xtrain_quant [,-PredToDel]
  
  # rmoving high correlated predictors Xtrain_mean_sd
  PredToDel = findCorrelation(cor( Xtrain_mean_sd )) 
  cat("PLS:: on Xtrain_mean_sd removing ",length(PredToDel), " predictors: ",paste(colnames(Xtrain_mean_sd) [PredToDel] , collapse=" " ) , " ... \n ")
  Xtest_mean_sd.reduced =  Xtest_mean_sd  [,-PredToDel]
  Xtrain_mean_sd.reduced = Xtrain_mean_sd [,-PredToDel]
  
  # feature scaling 
  scale.reduced.mean_sd = preProcess(Xtrain_mean_sd.reduced,method = c("center","scale"))
  scale.reduced.quant = preProcess(Xtrain_quant.reduced,method = c("center","scale"))
  
  Xtest_mean_sd.reduced = predict(scale.reduced.mean_sd,Xtest_mean_sd.reduced)
  Xtrain_mean_sd.reduced = predict(scale.reduced.mean_sd,Xtrain_mean_sd.reduced)
  Xtest_quant.reduced = predict(scale.reduced.quant,Xtest_quant.reduced)
  Xtrain_quant.reduced = predict(scale.reduced.quant,Xtrain_quant.reduced)
  
  ### B. scaled only  
  scale.mean_sd = preProcess(Xtrain_mean_sd,method = c("center","scale"))
  scale.quant = preProcess(Xtrain_quant,method = c("center","scale"))
  
  Xtrain_mean_sd.scaled = predict(scale.mean_sd,Xtrain_mean_sd)
  Xtest_mean_sd.scaled = predict(scale.mean_sd,Xtest_mean_sd)
  Xtrain_quant.scaled = predict(scale.quant,Xtrain_quant)
  Xtest_quant.scaled = predict(scale.quant,Xtest_quant)
  
  ################################################################### completing grids with probabilities 
  Xtrain = Xtest = NULL
  model = NULL
  
  ########### set grid 
  grid = grid.train = NULL 
  if (ds == "Dog_1") {
    grid = Dog_1.model
    grid.train = Dog_1.model.train
  } else if (ds == "Dog_2") {
    grid = Dog_2.model
    grid.train = Dog_2.model.train
  } else if (ds == "Dog_3") {
    grid = Dog_3.model
    grid.train = Dog_3.model.train
  } else if (ds == "Dog_4") {
    grid = Dog_4.model
    grid.train = Dog_4.model.train
  } else if (ds == "Dog_5") {
    grid = Dog_5.model
    grid.train = Dog_5.model.train
  } else if (ds == "Patient_1") {
    grid = Patient_1.model
    grid.train = Patient_1.model.train
  } else if (ds == "Patient_2") {
    grid = Patient_2.model
    grid.train = Patient_2.model.train
  } else {
    stop("ma che modello ha vinto!")
  }
  
  if (verbose) cat("******************* Completing grids with probabilities .... \n")
  ## grid 
  prob.df = as.data.frame(matrix(rep(-1,(nrow(Xtest_mean_sd)*nrow(grid))), nrow=nrow(grid) , ncol = nrow(Xtest_mean_sd)))
  colnames(prob.df) = paste0("p",(1:nrow(Xtest_mean_sd)))
  if ( ncol(prob.df) != nrow(Xtest_mean_sd) ) stop("prob.df - test - has a wrong number of columns")
  grid = cbind(grid,prob.df)
  
  ## grid train
  prob.df = as.data.frame(matrix(rep(-1,(nrow(Xtrain_mean_sd)*nrow(grid.train))), nrow=nrow(grid.train) , ncol = nrow(Xtrain_mean_sd)))
  colnames(prob.df) = paste0("p",(1:nrow(Xtrain_mean_sd)))
  if ( ncol(prob.df) != nrow(ytrain) ) stop("prob.df - train - has a wrong number of columns")
  grid.train = cbind(grid.train,prob.df)
  
  for (mo in 1:nrow(grid) ) {
    model.id = grid[mo,]$model.id
    model.label = as.character(grid[mo,]$model) 
    
    ## data set 
    if (model.id %% 6 == 0) {
      Xtrain = Xtrain_quant.reduced
      Xtest  = Xtest_quant.reduced
    } else if (model.id %% 6 == 1) {
      Xtrain = Xtrain_mean_sd
      Xtest  = Xtest_mean_sd
    } else if (model.id %% 6 == 2) {
      Xtrain = Xtrain_quant
      Xtest  = Xtest_quant
    } else if (model.id %% 6 == 3) {
      Xtrain = Xtrain_mean_sd.scaled
      Xtest  = Xtest_mean_sd.scaled
    } else if (model.id %% 6 == 4) {
      Xtrain = Xtrain_quant.scaled
      Xtest  = Xtest_quant.scaled
    } else if (model.id %% 6 == 5) {
      Xtrain = Xtrain_mean_sd.reduced
      Xtest  = Xtest_mean_sd.reduced
    } else {
      stop("ma che modello ha vinto (data set) !! ")
    }
    
    ## train and predict 
    ll = trainAndPredict (ds,model.label,model.id, 
                          Xtrain, ytrain.cat, Xtest, 
                          verbose=T)
    pred.prob.train = ll[[1]] 
    pred.train = ll[[2]]
    pred.prob.test = ll[[3]]    #### <<<<<<<<<--------------------------
    pred.test = ll[[4]]
    
    if ( length(pred.prob.test) != nrow(Xtest) ) stop("pred.prob.test: wrong column number")
    grid[grid$model.id == model.id , (4:(4+length(pred.prob.test)-1))] = pred.prob.test
    
    if ( length(grep(ds , sampleSubmission$clip)) != length(pred.prob.test) )  stop ("rows in sample submission different for grid!")
    
    if ( length(pred.prob.train) != nrow(Xtrain) ) stop("pred.prob.train: wrong column number")
    grid.train[grid$model.id == model.id , (4:(4+length(pred.prob.train)-1))] = pred.prob.train
    
    ##### updating models grids 
    if (ds == "Dog_1") {
      Dog_1.model = grid 
      Dog_1.model.train = grid.train 
    } else if (ds == "Dog_2") {
      Dog_2.model = grid 
      Dog_2.model.train = grid.train 
    } else if (ds == "Dog_3") {
      Dog_3.model = grid 
      Dog_3.model.train = grid.train 
    } else if (ds == "Dog_4") {
      Dog_4.model = grid 
      Dog_4.model.train = grid.train 
    } else if (ds == "Dog_5") {
      Dog_5.model = grid 
      Dog_5.model.train = grid.train
    } else if (ds == "Patient_1") {
      Patient_1.model = grid  
      Patient_1.model.train = grid.train
    } else if (ds == "Patient_2") {
      Patient_2.model = grid  
      Patient_2.model.train = grid.train
    } else {
      stop("ma che modello ha vinto!")
    }
  }
  
  ## updating trainClass 
  ## trainPred
  if (is.null(trainClass)) {
    trainClass = ytrain[,2]
  } else {
    trainClass = c(trainClass , ytrain[,2] )
  }
}

### making avg predictions 
Dog_1.model.avg = Dog_1.model; Dog_1.model.avg[,(4:ncol(Dog_1.model.avg))] = 0 
Dog_2.model.avg = Dog_2.model; Dog_2.model.avg[,(4:ncol(Dog_2.model.avg))] = 0 
Dog_3.model.avg = Dog_3.model; Dog_3.model.avg[,(4:ncol(Dog_3.model.avg))] = 0 
Dog_4.model.avg = Dog_4.model; Dog_4.model.avg[,(4:ncol(Dog_4.model.avg))] = 0 
Dog_5.model.avg = Dog_5.model; Dog_5.model.avg[,(4:ncol(Dog_5.model.avg))] = 0 
Patient_1.model.avg = Patient_1.model; Patient_1.model.avg[,(4:ncol(Patient_1.model.avg))] = 0 
Patient_2.model.avg = Patient_2.model; Patient_2.model.avg[,(4:ncol(Patient_2.model.avg))] = 0 

Dog_1.model.avg.train = Dog_1.model.train; Dog_1.model.avg.train[,(4:ncol(Dog_1.model.avg.train))] = 0 
Dog_2.model.avg.train = Dog_2.model.train; Dog_2.model.avg.train[,(4:ncol(Dog_2.model.avg.train))] = 0 
Dog_3.model.avg.train = Dog_3.model.train; Dog_3.model.avg.train[,(4:ncol(Dog_3.model.avg.train))] = 0 
Dog_4.model.avg.train = Dog_4.model.train; Dog_4.model.avg.train[,(4:ncol(Dog_4.model.avg.train))] = 0 
Dog_5.model.avg.train = Dog_5.model.train; Dog_5.model.avg.train[,(4:ncol(Dog_5.model.avg.train))] = 0 
Patient_1.model.avg.train = Patient_1.model.train; Patient_1.model.avg.train[,(4:ncol(Patient_1.model.avg.train))] = 0 
Patient_2.model.avg.train = Patient_2.model.train; Patient_2.model.avg.train[,(4:ncol(Patient_2.model.avg.train))] = 0 

cat("|---------------->>> making avg predictions on data sets <<",dss,">> ..\n")
for (ds in dss) {
  cat("|---------------->>> processing data set <<",ds,">> ..\n")
  
  grid = grid.train = grid.avg = grig.avg.train = NULL
  if (ds == "Dog_1") {
    grid = Dog_1.model
    grid.train = Dog_1.model.train
    grid.avg = Dog_1.model.avg
    grid.avg.train = Dog_1.model.avg.train
  } else if (ds == "Dog_2") {
    grid = Dog_2.model
    grid.train = Dog_2.model.train
    grid.avg = Dog_2.model.avg
    grid.avg.train = Dog_2.model.avg.train
  } else if (ds == "Dog_3") {
    grid = Dog_3.model
    grid.train = Dog_3.model.train
    grid.avg = Dog_3.model.avg
    grid.avg.train = Dog_3.model.avg.train
  } else if (ds == "Dog_4") {
    grid = Dog_4.model
    grid.train = Dog_4.model.train
    grid.avg = Dog_4.model.avg
    grid.avg.train = Dog_4.model.avg.train
  } else if (ds == "Dog_5") {
    grid = Dog_5.model
    grid.train = Dog_5.model.train
    grid.avg = Dog_5.model.avg
    grid.avg.train = Dog_5.model.avg.train
  } else if (ds == "Patient_1") {
    grid = Patient_1.model
    grid.train = Patient_1.model.train
    grid.avg = Patient_1.model.avg
    grid.avg.train = Patient_1.model.avg.train
  } else if (ds == "Patient_2") {
    grid = Patient_2.model
    grid.train = Patient_2.model.train
    grid.avg = Patient_2.model.avg
    grid.avg.train = Patient_2.model.avg.train
  } else {
    stop("ma che data set !")
  }
  
  DENUM = rep(-1,nrow(grid))
  for ( moo in 1:nrow(grid) ) {
    DENUM[moo] = sum(grid[1:moo,3])
  }
  for ( mo in 1:nrow(grid) ) {
    for (moo in mo:nrow(grid) ) {
      w = grid[mo,3] / DENUM[moo]
      cat ("mo=",mo," - moo =",moo," w = ",w,"\n")
      
      grid.avg[moo,(4:ncol(grid.avg))] = 
        grid.avg[moo,(4:ncol(grid.avg))] + (grid[mo,(4:ncol(grid))]  * w) 
      
      grid.avg.train[moo,(4:ncol(grid.avg.train))] = 
        grid.avg.train[moo,(4:ncol(grid.avg.train))] + (grid.train[mo,(4:ncol(grid.train))]  * w) 
      
    }
  }
  
  ## updating avg models predictions 
  if (ds == "Dog_1") {
    Dog_1.model.avg = grid.avg
    Dog_1.model.avg.train = grid.avg.train
  } else if (ds == "Dog_2") {
    Dog_2.model.avg  = grid.avg
    Dog_2.model.avg.train = grid.avg.train
  } else if (ds == "Dog_3") {
    Dog_3.model.avg  = grid.avg
    Dog_3.model.avg.train = grid.avg.train
  } else if (ds == "Dog_4") {
    Dog_4.model.avg  = grid.avg
    Dog_4.model.avg.train = grid.avg.train
  } else if (ds == "Dog_5") {
    Dog_5.model.avg  = grid.avg
    Dog_5.model.avg.train = grid.avg.train
  } else if (ds == "Patient_1") {
    Patient_1.model.avg  = grid.avg
    Patient_1.model.avg.train = grid.avg.train
  } else if (ds == "Patient_2") {
    Patient_2.model.avg = grid.avg
    Patient_2.model.avg.train = grid.avg.train
  } else {
    stop("ma che data set !")
  }
}
  
###### making sub.grid
sub.grid = matrix(rep(-1,(nrow(sampleSubmission)*nrow(Dog_1.model))),  nrow=nrow(Dog_1.model) , ncol=  nrow(sampleSubmission)   )
sub.grid.idx = 1

###### making sub.grid.train 
nrow.train =  (ncol(Dog_1.model.avg.train) -4 + 1) +
              (ncol(Dog_2.model.avg.train) -4 + 1) +
              (ncol(Dog_3.model.avg.train) -4 + 1) +
              (ncol(Dog_4.model.avg.train) -4 + 1) + 
              (ncol(Dog_5.model.avg.train) -4 + 1) + 
              (ncol(Patient_1.model.avg.train) -4 + 1) + 
              (ncol(Patient_2.model.avg.train) -4 + 1)

if (nrow.train != length(trainClass)) stop("nrow.train has a number of columns different from trainClass")
  
sub.grid.train = matrix(rep(-1,(nrow.train*nrow(Dog_1.model))),  nrow=nrow(Dog_1.model) , ncol= nrow.train )
sub.grid.idx.train = 1

## filling grids ...  
for (ds in dss) {
  grid = grid.train = NULL 
  if (ds == "Dog_1") {
    grid = Dog_1.model.avg
    grid.train = Dog_1.model.avg.train
  } else if (ds == "Dog_2") {
    grid = Dog_2.model.avg
    grid.train = Dog_2.model.avg.train
  } else if (ds == "Dog_3") {
    grid = Dog_3.model.avg
    grid.train = Dog_3.model.avg.train
  } else if (ds == "Dog_4") {
    grid = Dog_4.model
    grid.train = Dog_4.model.avg.train
  } else if (ds == "Dog_5") {
    grid = Dog_5.model.avg
    grid.train = Dog_5.model.avg.train
  } else if (ds == "Patient_1") {
    grid = Patient_1.model.avg
    grid.train = Patient_1.model.avg.train
  } else if (ds == "Patient_2") {
    grid = Patient_2.model.avg
    grid.train = Patient_2.model.avg.train
  } else {
    stop("ma che data set !")
  }
  
  for ( mo in 1:nrow(grid) ) {
    
    sub.grid[mo,  sub.grid.idx:(sub.grid.idx + length(4:ncol(grid))  - 1) ]  = 
      as.matrix(grid[mo,(4:ncol(grid))])
    
    sub.grid.train[mo,  sub.grid.idx.train:(sub.grid.idx.train + length(4:ncol(grid.train))  - 1) ]  = 
      as.matrix(grid.train[mo,(4:ncol(grid.train))])
    
    if ( length(grep(ds , sampleSubmission$clip)) != length(4:ncol(grid)) )  stop ("rows in sample submission different for grid!")
  }
  
  sub.grid.idx = sub.grid.idx + length(4:ncol(grid))
  sub.grid.idx.train = sub.grid.idx.train + length(4:ncol(grid.train))
}

## submission - averaged models 
for (mo in 1:nrow(sub.grid) ) {
  label = paste0("avg_",mo)
  mySub = data.frame(clip = sampleSubmission$clip , preictal = format( sub.grid[mo,]  , scientific = F )  )
  write.csv(mySub,quote=FALSE,file=paste(getBasePath(),SUB_DIR,"mySub_class_" , label , "_fix_mod.zat" , sep=""), row.names=FALSE)
  
  ## calibrating probs ... 
  trainClass.cat = as.factor(trainClass)
  levels(trainClass.cat) =  c("interict","preict")
  train.df = data.frame(class = trainClass.cat , prob = as.numeric( format( sub.grid.train[mo,]  , scientific = F )) )
  
  ## Calibrating Probabilities - sigmoid - top model 
  sigmoidalCal <- glm(  class ~ prob  , data = train.df , family = binomial)
  #coef(summary(sigmoidalCal)) 
  sigmoidProbs <- predict(sigmoidalCal, newdata = data.frame( prob = as.numeric(format( sub.grid[mo,]  , scientific = F ) )), type = "response")
  mySub2 = data.frame(clip = sampleSubmission$clip , preictal = format(sigmoidProbs,scientific = F))  
  write.csv(mySub2,quote=FALSE,file=paste(getBasePath(),SUB_DIR,"mySub_sigmoid_calibrat_class_",label,"_fix_mod.zat",sep=""), row.names=FALSE)
   
  ## Calibrating Probabilities - Bayes - top model 
  library(klaR)
  BayesCal <- NaiveBayes( class ~ prob  , data = train.df, usekernel = TRUE)
  BayesProbs <- predict(BayesCal, newdata = data.frame(prob = as.numeric(format( sub.grid[mo,]  , scientific = F ))) )
  BayesProbs.preict <- BayesProbs$posterior[, "preict"]
  mySub3 = data.frame(clip = sampleSubmission$clip , preictal = format(BayesProbs.preict,scientific = F))
  write.csv(mySub3,quote=FALSE,file=paste(getBasePath(),SUB_DIR,"mySub_bayes_calibrat_class_",label,"_fix_mod.zat",sep=""), row.names=FALSE)
}
