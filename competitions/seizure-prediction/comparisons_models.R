## notes_seizure_R
library(caret)
library(Hmisc)
library(data.table)
library(verification)
library(pROC)
library(kernlab)
library(subselect)

getBasePath = function (type = "data" , ds = "" , gen="") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/seizure-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/seizure-prediction/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/seizure-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/seizure-prediction/"
  } else {
    stop("unrecognized type.")
  }
  
  if (file.exists(base.path1))  {
    ret = paste0(base.path1,"/")
  } else {
    ret = base.path2
  }
  
  if (ds != "" ) {
    if (gen != "") {
      cat("data from ", gen ," ... \n")
      ret = paste(paste0(ret,ds),"_digest_",gen,"/",sep="")
    } else {
      cat("data from 4gen <<default>>... \n")
      ret = paste0(paste0(ret,ds),"_digest_4gen/")  
    }
  }
  ret
} 


buildPCAFeatures = function(Xtrain_mean_sd,Xtest_mean_sd,Xtrain_quant,Xtest_quant,verbose) {  
  
  Xtrain_pca = as.data.frame(fread( paste0(getBasePath(type = "data") ,'Patient_2_pca_feature/Xtrain_pca.zat') ))
  Xtest_pca = as.data.frame(fread(  paste0(getBasePath(type = "data") , 'Patient_2_pca_feature/Xtest_pca.zat') ))
  
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

trainAndPredict = function(ds,model.label,model.id,
                           Xtrain, ytrain.cat,
                           Xtest,
                           seed=NULL, 
                           verbose=F) {
  model = NULL
  ## model 
  if (model.id >= 1 && model.id <= 6) { ## logistic reg 
    if (! is.null(seed) ) set.seed(seed)
    model <- train( x = Xtrain , y = ytrain.cat , 
                    method = "glm", metric = "ROC", trControl = controlObject)
  } else if (model.id >= 7 && model.id <= 12) { ## lda 
    if (! is.null(seed) ) set.seed(seed)
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "lda", metric = "ROC" , trControl = controlObject)
  } else if (model.id >= 13 && model.id <= 18) { ## plsda 
    if (! is.null(seed) ) set.seed(seed)
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "pls", tuneGrid = expand.grid(.ncomp = 1:10), 
                    metric = "ROC" , trControl = controlObject)
  } else if (model.id >= 19 && model.id <= 24) { ## pm 
    glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1), .lambda = seq(.01, .2, length = 40))
    if (! is.null(seed) ) set.seed(seed)
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "glmnet", tuneGrid = glmnGrid, 
                    metric = "ROC", trControl = controlObject)
  } else if (model.id >= 25 && model.id <= 30) { ## nsc 
    nscGrid <- data.frame(.threshold = 0:25)
    if (! is.null(seed) ) set.seed(seed)
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "pam", tuneGrid = nscGrid, 
                    metric = "ROC", trControl = controlObject)
  } else if (model.id >= 31 && model.id <= 36) { # neural networks 
    nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
    maxSize <- max(nnetGrid$.size)
    numWts <- 1*(maxSize * ( (dim(Xtrain)[2]) + 1) + maxSize + 1)
    if (! is.null(seed) ) set.seed(seed)
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "nnet", metric = "ROC", 
                    preProc = c( "spatialSign") , 
                    tuneGrid = nnetGrid , trace = FALSE , maxit = 2000 , 
                    MaxNWts = numWts, trControl = controlObject)
  } else if (model.id >= 37 && model.id <= 42) { ## svm 
    sigmaRangeReduced <- sigest(as.matrix(Xtrain))
    svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1], .C = 2^(seq(-6, 6,by=0.2)) )
    if (ds == "Patient_2") {
      cat("<<Patient_2>> setting class.weights = c(preict = 2, interict = 1) .... \n")
      if (! is.null(seed) ) set.seed(seed)
      model <- train( x = Xtrain , y = ytrain.cat,  
                      method = "svmRadial", 
                      tuneGrid = svmRGridReduced, 
                      class.weights = c(preict = 2, interict = 1),
                      metric = "ROC", fit = FALSE, trControl = controlObject)
    } else {
      if (! is.null(seed) ) set.seed(seed)
      model <- train( x = Xtrain , y = ytrain.cat,  
                      method = "svmRadial", tuneGrid = svmRGridReduced, 
                      metric = "ROC", fit = FALSE, trControl = controlObject)
    }
  } else if (model.id >= 43 && model.id <= 49) { ## knn 
    if (! is.null(seed) ) set.seed(seed)
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "knn", 
                    tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)),
                    metric = "ROC",  trControl = controlObject)
  } else if (model.id >= 49 && model.id <= 54) { ## class trees 
    if (! is.null(seed) ) set.seed(seed)
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "rpart", tuneLength = 30, 
                    metric = "ROC", trControl = controlObject)
  } else if (model.id >= 55 && model.id <= 60) { ## boosted trees 
    if (model.id >= 55 && model.id <= 59) {
      ## 55 - 59 
      if (! is.null(seed) ) set.seed(seed)
      model <- train( x = Xtrain , y = ytrain.cat,  
                      method = "C5.0",  metric = "ROC", trControl = controlObject)
    } else {
      ## 60 - BOOSTED_TREE_QUANTILES_REDUCED 
      if (! is.null(seed) ) set.seed(seed)
      model <- train( x = Xtrain , y = ytrain.cat,  
                      tuneGrid = expand.grid(.trials = c(1, (1:10)*10), .model = "tree", .winnow = c(TRUE, FALSE) ),
                      method = "C5.0",  metric = "ROC", trControl = controlObject)
    }
  } else if (model.id >= 61 && model.id <= 66) { ## bagging trees 
    if (model.id >= 61 && model.id <= 65) {
      ## 61 - 65 
      if (! is.null(seed) ) set.seed(seed)
      model <- train( x = Xtrain , y = ytrain.cat,  
                      method = "bag",  metric = "ROC", trControl = controlObject, B = 50 ,
                      bagControl = bagControl(fit = plsBag$fit,
                                              predict = plsBag$pred,
                                              aggregate = plsBag$aggregate))
    } else {
      ## 66 - BAGGING_TREE_QUANTILES_REDUCED
      if (! is.null(seed) ) set.seed(seed)
      model <- train( x = Xtrain , y = ytrain.cat,  
                      method = "bag",  metric = "ROC", trControl = controlObject, 
                      tuneGrid = data.frame(vars = seq(1, 15, by = 2)), 
                      bagControl = bagControl(fit = plsBag$fit,
                                              predict = plsBag$pred,
                                              aggregate = plsBag$aggregate))
    }
  } else if (model.id == 0) { ## null prediction, i.e. pred = (0,0,0,...,0)
    if (! is.null(seed) ) set.seed(seed)
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
  
  return(model)
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

#### general settings ... 
verbose = T
doPlot = F 

### resampling method 
controlObject <- trainControl(method = "boot", number = 30 , 
                              summaryFunction = twoClassSummary , classProbs = TRUE)

#### model grid 
model.grid = NULL 

# ##################################### TEST
# model.grid.test = data.frame(  
#                          model.label = c("LOG_MS_RED" , "LDA_MS_RED" , "LDA_MS_RED", "LDA_QT_RED") , 
#                          model.id = c(LOGISTIC_REG_MEAN_SD_REDUCED, LDA_MEAN_SD_REDUCED, LDA_MEAN_SD_REDUCED, LDA_REG_QUANTILES_REDUCED) , 
#                          data.source.gen = c("4gen", "7gen" , "7gen", "5gen") , 
#                          pca.feature = c(T,T,F,T) , 
#                          superFeature = c(T,T,F,T) )
# 
# #### data source 
# DS = "Patient_2"
# 
# #### out dir 
# SUB_DIR = "test"
# if (SUB_DIR != "") {
#   cat("creating directory <<",SUB_DIR,">> ... \n")
#   SUB_DIR = paste0(SUB_DIR,"/")
#   dir.create(paste(getBasePath(),SUB_DIR,sep=""))
# }
# model.grid = model.grid.test 
# ###################################################### END OF TEST

#### DATA SET TO PROCESS  
#source( paste0(getBasePath("code") , "Patient_2_grid.R") )
source( paste0(getBasePath("code") , "Dog_2_grid.R") )


cat("|---------------->>> data set to process: <<",DS,">> ..\n")

black.list = NULL
  
for (i in 1:nrow(model.grid)) {
  model.label = as.character(model.grid[i,]$model.label)
  model.id = model.grid[i,]$model.id 
  data.source.gen = as.character(model.grid[i,]$data.source.gen)
  pca.feature = model.grid[i,]$pca.feature
  superFeature = model.grid[i,]$superFeature
  
  full.model.label = paste(model.label,"_",as.character(model.id),"__ds_",data.source.gen,
                           "__pca_",as.character(pca.feature),"__super_",as.character(superFeature),sep="") 
  
  cat("-------> [", i , "/" , as.character(nrow(model.grid)) ,"] ",  full.model.label , " .. \n")

  ######### loading data sets ...
  Xtrain_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds=DS , gen=data.source.gen),"Xtrain_mean_sd.zat",sep="") , header = F , sep=","  ))
  Xtrain_quant = as.data.frame( fread(paste(getBasePath(type = "data" , ds=DS , gen=data.source.gen),"Xtrain_quant.zat",sep="") , header = F , sep=","  ))
  
  Xtest_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds=DS , gen=data.source.gen),"Xtest_mean_sd.zat",sep="") , header = F , sep=","  ))
  Xtest_quant = as.data.frame(fread(paste(getBasePath(type = "data" , ds=DS , gen=data.source.gen),"Xtest_quant.zat",sep="") , header = F , sep=","  ))
  
  ytrain = as.data.frame(fread(paste(getBasePath(type = "data" , ds=DS , gen=data.source.gen),"ytrain.zat",sep="") , header = F , sep=","  ))
  
  ## names 
  colnames(Xtrain_mean_sd)  = colnames(Xtest_mean_sd) = paste("fmeansd",rep(1:((dim(Xtrain_mean_sd)[2]))) , sep = "")
  colnames(Xtrain_quant) = colnames(Xtest_quant) = paste("fquant",rep(1:((dim(Xtrain_quant)[2]))) , sep = "")
  
  ytrain.cat = factor(ytrain[,2]) 
  levels(ytrain.cat) = c("interict","preict")
  
  ###### super-features 
  if (superFeature) {
    cat("building superFeature ... \n") 
    l = buildRelativeSpectralPowerFeatures (Xtrain_mean_sd,Xtest_mean_sd,Xtrain_quant,Xtest_quant,ytrain,NN=10,verbose)
    Xtrain_mean_sd = l[[1]]
    Xtest_mean_sd = l[[2]]
    Xtrain_quant = l[[3]]
    Xtest_quant = l[[4]]
  }
  
  if (DS == "Patient_2" && pca.feature) {
    cat("building PCA features ... \n") 
    l = buildPCAFeatures (Xtrain_mean_sd,Xtest_mean_sd,Xtrain_quant,Xtest_quant,verbose)
    Xtrain_mean_sd  = l[[1]]
    Xtrain_quant = l[[2]]
    Xtest_mean_sd = l[[3]]
    Xtest_quant = l[[4]]   
  }
  
  ### check 
  if ( length(grep(DS , sampleSubmission$clip)) != nrow(Xtest_mean_sd) | 
         length(grep(DS , sampleSubmission$clip)) != nrow(Xtest_quant)   )  stop ("rows in sample submission different for test set!")
  
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
  scale.reduced.mean_sd = preProcess(rbind(Xtrain_mean_sd.reduced,Xtest_mean_sd.reduced),method = c("center","scale"))
  scale.reduced.quant = preProcess(rbind(Xtrain_quant.reduced,Xtest_quant.reduced),method = c("center","scale"))
  
  Xtest_mean_sd.reduced = predict(scale.reduced.mean_sd,Xtest_mean_sd.reduced)
  Xtrain_mean_sd.reduced = predict(scale.reduced.mean_sd,Xtrain_mean_sd.reduced)
  Xtest_quant.reduced = predict(scale.reduced.quant,Xtest_quant.reduced)
  Xtrain_quant.reduced = predict(scale.reduced.quant,Xtrain_quant.reduced)
  
  delta = apply(Xtest_mean_sd.reduced,2,mean) - apply(Xtrain_mean_sd.reduced,2,mean)
  cat("**** differenza tra le medie dei predittori Xtest_mean_sd.reduced / Xtrain_mean_sd.reduced \n")
  print(delta)
  delta = apply(Xtest_quant.reduced,2,mean) - apply(Xtrain_quant.reduced,2,mean)
  cat("**** differenza tra le medie dei predittori Xtest_quant.reduced / Xtrain_quant.reduced \n")
  print(delta)
  
  ### B. scaled only  
  scale.mean_sd = preProcess(rbind(Xtrain_mean_sd,Xtest_mean_sd),method = c("center","scale"))
  scale.quant = preProcess(rbind(Xtrain_quant,Xtest_quant),method = c("center","scale"))
  
  Xtrain_mean_sd.scaled = predict(scale.mean_sd,Xtrain_mean_sd)
  Xtest_mean_sd.scaled = predict(scale.mean_sd,Xtest_mean_sd)
  Xtrain_quant.scaled = predict(scale.quant,Xtrain_quant)
  Xtest_quant.scaled = predict(scale.quant,Xtest_quant)
  
  delta = apply(Xtest_mean_sd.scaled,2,mean) - apply(Xtrain_mean_sd.scaled,2,mean)
  cat("**** differenza tra le medie dei predittori Xtest_mean_sd.scaled / Xtrain_mean_sd.scaled \n")
  print(delta)
  delta = apply(Xtest_quant.scaled,2,mean) - apply(Xtrain_quant.scaled,2,mean)
  cat("**** differenza tra le medie dei predittori Xtest_quant.scaled / Xtrain_quant.scaled \n")
  print(delta)
  
  ###################################################################
  
  Xtrain = Xtest = NULL
  model = NULL
  
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
  cat("|---------------->>> [", i , "/" , as.character(nrow(model.grid)) ,"] training and predicting ..\n")
  model = NULL 
  model = tryCatch({
    
    model = trainAndPredict (DS,
                             model.label,
                             model.id, 
                             Xtrain, 
                             ytrain.cat, 
                             Xtest, 
                             verbose=verbose,
                             seed=NULL)
    
  }, error = function(e) {
    message(e)
    return(NULL)
  })
  
  if(is.null(model)) {
    cat("encuntered error ... skipping ... \n")
    if (is.null(black.list)) {
      black.list = c(i)
    } else {
      black.list = c(black.list,i)
    }
    next
  } 
  
  
  ### storing model, name 
  assign(paste0("modelxx",i),model)
  
}

##model.list = list( model1,model2,model3)
model.list = list()
model.names = NULL
bl = 0 
for (i in 1:nrow(model.grid)) { 
  if (i %in% black.list) {
    bl = bl + 1
    next
  }  
  model.list[[(i-bl)]] = get(paste0("modelxx",i), environment())
  
  model.label = as.character(model.grid[i,1])
  model.id = model.grid[i,2] 
  data.source.gen = as.character(model.grid[i,3])
  pca.feature = model.grid[i,4]
  superFeature = model.grid[i,5]
  
  full.model.label = as.character(paste(model.label," ",as.character(model.id)," ds",data.source.gen,
                           "pca",ifelse(pca.feature,"T","F"),"super",ifelse(superFeature,"T","F"),sep="")) 
  if (is.null(model.names)) model.names = c(full.model.label) 
  else model.names = c(model.names,full.model.label) 
}

##names(model.list) = model.names

cvValues <- resamples( model.list )
summary(cvValues)

cvValues.ROC = cvValues$values[,grep("ROC" , colnames(cvValues$values) )]

### computing best 
best.mean = which (apply(cvValues.ROC,2,mean) == max(apply(cvValues.ROC,2,mean)) )
best.min = which (apply(cvValues.ROC,2,min) == max(apply(cvValues.ROC,2,min)) )
best.max = which (apply(cvValues.ROC,2,max) == max(apply(cvValues.ROC,2,max)) )

cat("***************** BEST MEAN  ***************** \n")
print(best.mean)
cat("********************************************** \n")

cat("***************** BEST MIN  ***************** \n")
print(best.min)
cat("********************************************** \n")

cat("***************** BEST MAX  ***************** \n")
print(best.max)
cat("********************************************** \n")

cat("***** FULL LIST - MEDIAN - (descending order) \n")
ll = (apply(cvValues.ROC,2,median))
median.list = ll[order(ll , decreasing = T)]
print(ll[order(ll , decreasing = T)])
write.csv(data.frame(mod=names(ll[order(ll , decreasing = T)]) , ROC=ll[order(ll , decreasing = T)]),
          quote=FALSE,file=paste(getBasePath(),SUB_DIR,"list_median.zat" , sep=""), row.names=FALSE)
cat("********************************************** \n")

cat("***** FULL LIST - MEAN - (descending order) \n")
ll = (apply(cvValues.ROC,2,mean))
print(ll[order(ll , decreasing = T)])
mean.list = ll[order(ll , decreasing = T)]
write.csv(data.frame(mod=names(ll[order(ll , decreasing = T)]) , ROC=ll[order(ll , decreasing = T)]),
          quote=FALSE,file=paste(getBasePath(),SUB_DIR,"list_mean.zat" , sep=""), row.names=FALSE)
cat("********************************************** \n")

cat("***** FULL LIST - MIN - (descending order) \n")
ll = (apply(cvValues.ROC,2,min))
print(ll[order(ll , decreasing = T)])
min.list = ll[order(ll , decreasing = T)]
write.csv(data.frame(mod=names(ll[order(ll , decreasing = T)]) , ROC=ll[order(ll , decreasing = T)]),
          quote=FALSE,file=paste(getBasePath(),SUB_DIR,"list_min.zat" , sep=""), row.names=FALSE)
cat("********************************************** \n")

cat("***** FULL LIST - MAX - (descending order) \n")
ll = (apply(cvValues.ROC,2,max))
print(ll[order(ll , decreasing = T)])
write.csv(data.frame(mod=names(ll[order(ll , decreasing = T)]) , ROC=ll[order(ll , decreasing = T)]),
          quote=FALSE,file=paste(getBasePath(),SUB_DIR,"list_max.zat" , sep=""), row.names=FALSE)
cat("********************************************** \n")

#splom(cvValues, metric = "ROC")
##xyplot(cvValues, metric = "ROC")  

png(filename=paste(getBasePath(),SUB_DIR,"parallelplot.png" , sep=""))
parallelplot(cvValues, metric = "ROC")
dev.off()

png(filename=paste(getBasePath(),SUB_DIR,"dotplot.png" , sep=""))
dotplot(cvValues, metric = "ROC") ### <<<<<<<<<<<<<<<<<<<<<<<<<<--------------------- il modello in alto e' il migliore 
dev.off()

dotplot(cvValues, metric = "ROC") ### <<<<<<<<<<<<<<<<<<<<<<<<<<--------------------- il modello in alto e' il migliore 

rocDiffs <- diff(cvValues, metric = "ROC")
##summary(rocDiffs)
##dotplot(rocDiffs, metric = "ROC")

if (! is.null(black.list) ) {
  cat("***** MODELS GOT ERRORS \n")
  print(model.grid[black.list,])
  write.csv(model.grid[black.list,] , quote=FALSE,file=paste(getBasePath(),SUB_DIR,"black_list.zat" , sep=""), row.names=FALSE)
} 

write.csv(model.grid , quote=FALSE,file=paste(getBasePath(),SUB_DIR,"model_grid.csv" , sep=""), row.names=T)
                         