
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

trainAndPredict = function(ds,model.label,model.id,
                           Xtrain, ytrain.cat,
                           Xtest,
                           seed=NULL, 
                           verbose=F) {
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
  
  list(pred.prob.train, pred.train, pred.prob.test, pred.test)
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

############# da modificare manualmente con il path del file di submission da cui si parte 
input.sub = "/svm_weight/IN.zat"
cat ("loading intial submission file <<",as.character(paste(getBasePath(type = "data"),input.sub,sep="")),">> \n"  )
INPUT_SUBMISSION = as.data.frame(fread(paste(getBasePath(type = "data"),input.sub,sep="") , header = T , sep=","  ))

############# general settings ... 
verbose = T
doPlot = F 
superFeature = F
Patient_2.pca = F

############# general settings ...
SUB_DIR = "svm_weight"
if (SUB_DIR != "") {
  cat("creating directory <<",SUB_DIR,">> ... \n")
  SUB_DIR = paste0(SUB_DIR,"/")
  dir.create(paste(getBasePath(),SUB_DIR,sep=""))
}

############ models grids 

## baseline mix 1 
model.grid = data.frame( data.source.to.process = c("Dog_1", "Dog_2", "Dog_3", "Dog_4", "Dog_5" , "Patient_1", "Patient_2") , 
                         model.label = c("NN_QUANTILES_REDUCED", "SVM_QUANTILES_REDUCED", "SVM_MEAN_SD_SCALED", "SVM_QUANTILES_REDUCED", 
                                         "NN_MEAN_SD_REDUCED", "KNN_QUANTILES_SCALED", "SVM_MEAN_SD_REDUCED") , 
                         model.id = c(NN_QUANTILES_REDUCED , SVM_QUANTILES_REDUCED , SVM_MEAN_SD_SCALED , SVM_QUANTILES_REDUCED, 
                                      NN_MEAN_SD_REDUCED, KNN_QUANTILES_SCALED, SVM_MEAN_SD_REDUCED) , 
                         data.source.gen = c("4gen" , "4gen" , "4gen" , "4gen" , "4gen", "4gen" , "5gen"  ), 
                         recalib.bayes = c(T,T,T,T,T,T,T), 
                         recalib.sigmoid = c(F,F,F,F,F,F,F), 
                         seed = c(-1,-1,-1,-1,-1,-1,-1) )

##### check 
if (    nrow(model.grid) > 7 ) stop("there're 7 data source!")

if (   nrow(model.grid[model.grid$data.source.to.process=="Dog_1",]) > 1 | 
       nrow(model.grid[model.grid$data.source.to.process=="Dog_2",]) > 1 | 
       nrow(model.grid[model.grid$data.source.to.process=="Dog_3",]) > 1 |
       nrow(model.grid[model.grid$data.source.to.process=="Dog_4",]) > 1 |
       nrow(model.grid[model.grid$data.source.to.process=="Dog_5",]) > 1 |
       nrow(model.grid[model.grid$data.source.to.process=="Patient_1",]) > 1 |
       nrow(model.grid[model.grid$data.source.to.process=="Patient_2",]) > 1 
       ) stop("at most 1 model per data source!") 


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

### go 
cat("|---------------->>> data set to process: <<",as.character(model.grid$data.source.to.process),">> ..\n")

### completing models grids
for (ds in model.grid$data.source.to.process) {
  
  cat("|---------------->>> processing data set <<",ds,">> ..\n")
  
  model.id = as.numeric( model.grid[model.grid$data.source.to.process==ds,]$model.id)  
  data.source.gen = as.character( model.grid[model.grid$data.source.to.process==ds,]$data.source.gen)  
  model.label = as.character( model.grid[model.grid$data.source.to.process==ds,]$model.label)  
  recalib.bayes = as.logical( model.grid[model.grid$data.source.to.process==ds,]$recalib.bayes)  
  recalib.sigmoid = as.logical( model.grid[model.grid$data.source.to.process==ds,]$recalib.sigmoid)  
  seed =  as.numeric( model.grid[model.grid$data.source.to.process==ds,]$seed) 
  
  ######### loading data sets ...
  Xtrain_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds , gen=data.source.gen),"Xtrain_mean_sd.zat",sep="") , header = F , sep=","  ))
  Xtrain_quant = as.data.frame( fread(paste(getBasePath(type = "data" , ds=ds , gen=data.source.gen),"Xtrain_quant.zat",sep="") , header = F , sep=","  ))
  
  Xtest_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds , gen=data.source.gen),"Xtest_mean_sd.zat",sep="") , header = F , sep=","  ))
  Xtest_quant = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds , gen=data.source.gen),"Xtest_quant.zat",sep="") , header = F , sep=","  ))
  
  ytrain = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds , gen=data.source.gen),"ytrain.zat",sep="") , header = F , sep=","  ))
  
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
                        verbose=verbose,seed=NULL)
  pred.prob.train = ll[[1]] 
  pred.train = ll[[2]]
  pred.prob.test = ll[[3]]    #### <<<<<<<<<--------------------------
  pred.test = ll[[4]]
  
  ## Calibrating Probabilities - Bayes / sigmoid 
  if (recalib.bayes) {
    cat("recalibrating probabilities with Bayes ... \n") 
    train.df = data.frame(class = ytrain.cat , prob = as.numeric( pred.prob.train ) )
    
    library(klaR)
    BayesCal <- NaiveBayes( class ~ prob  , data = train.df, usekernel = TRUE)
    BayesProbs <- predict(BayesCal, newdata = data.frame(prob = as.numeric(pred.prob.test )) )
    BayesProbs.preict <- BayesProbs$posterior[, "preict"]
    
    pred.prob.test = BayesProbs.preict
    
  } else if (recalib.sigmoid) {
    cat("recalibrating probabilities with sigmoid ... \n") 
    
    train.df = data.frame(class = ytrain.cat , prob = as.numeric( format( pred.prob.train  , scientific = F )) )
    sigmoidalCal <- glm(  class ~ prob  , data = train.df , family = binomial)
    sigmoidProbs <- predict(sigmoidalCal, newdata = data.frame( prob = as.numeric( pred.prob.test  )), type = "response")
    
    pred.prob.test = sigmoidProbs
  }
  
  
  ### sostituisco il blocco relativo nel data set delle predictions 
  INPUT_SUBMISSION[min(grep(ds , INPUT_SUBMISSION$clip)):max(grep(ds , INPUT_SUBMISSION$clip)) , 2] = format( pred.prob.test , scientific = F )
  
}


### scrivo su disco le predictions 
write.csv(INPUT_SUBMISSION,quote=FALSE,file=paste(getBasePath(),SUB_DIR,"mySub_selec.zat" , sep=""), row.names=FALSE)

