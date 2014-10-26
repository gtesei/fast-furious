
## notes_seizure_R
library(caret)
library(Hmisc)
library(data.table)
library(verification)
library(pROC)
library(kernlab)
library(subselect)
library(plyr)


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
    ret = paste0(paste0(ret,ds),"_digest/")
  }
  ret
} 
verify.kfolds = function(k,folds,dataset,event.level) {
  correct = T
  for(j in 1:k) {  
    ## y 
    dataset.train = dataset[folds != j]
    dataset.xval = dataset[folds == j]
    
    if (sum(dataset.train == event.level ) == 0 | 
        sum(dataset.xval == event.level ) == 0  ) {
      correct = F
      break
    }
  }
  return(correct)
}
  
kfolds = function(k,data.length) {
  k = min(k,data.length)
  folds = rep(NA,data.length)
  labels = 1:data.length
  st = floor(data.length/k)
  al_labels = NULL
  for (s in 1:k) {
    x = NULL
    if (is.null(al_labels))
      x = sample(labels,st)
    else
      x = sample(labels[-al_labels],st)
    
    folds[x] = s
    if (is.null(al_labels))
      al_labels = x
    else
      al_labels = c(al_labels,x)
  }
  ss = 1
  for (s in 1:length(folds)){
    if (is.na(folds[s])) {
      folds[s] = ss
      ss = ss + 1
    } 
  }
  folds
}

measure = function(data.source, 
                   resampling.label,resampling.id, 
                   model.label,model.id,
                   pred.prob.train , pred.prob.xval , 
                   pred.train , pred.xval,
                   ytrain, yxval,
                   tm, grid = NULL,verbose=F, doPlot=F) {
  
  ## accuracy 
  acc.xval.all0 = sum(factor(rep('interict',length(yxval)) , levels = levels(yxval) ) == yxval) / length(yxval)
  acc.train = sum(ytrain == pred.train) / length(ytrain)
  acc.xval = sum(yxval == pred.xval) / length(yxval) 
  
  ## ROC 
  rocCurve <- pROC::roc(response = ytrain, predictor = pred.prob.train, levels = levels(ytrain) )
  roc.train = as.numeric( pROC::auc(rocCurve) )
  
  rocCurve <- pROC::roc(response = yxval, predictor = pred.prob.xval , levels = levels(yxval) )
  roc.xval = as.numeric( pROC::auc(rocCurve) )
  
  roc.train.2 = roc.area(as.numeric(ytrain == 'preict') , pred.prob.train )$A
  roc.xval.2 = roc.area(as.numeric(yxval == 'preict') , pred.prob.xval )$A
  
  roc.xval.min = min(roc.xval.2,roc.xval)
  
  ## logging 
  if (verbose) cat("******************* ", model.label, " <<" , model.id ,  ">>  --  Time elapsed(sec.):",tm[[3]], " \n")
  if (verbose) cat("** acc.train =",acc.train, " -  acc.xval =",acc.xval, " - acc.xval.all0 =",acc.xval.all0, "  \n")
  if (verbose) cat("** roc.train =",roc.train," -  roc.xval =",roc.xval,"  \n")
  if (verbose) cat("** roc.train.2 =",roc.train.2," -  roc.xval.2 =",roc.xval.2,"  \n")
  if (verbose) cat("** roc.xval.min =",roc.xval.min, " \n")
  
  ## poltting 
  if (doPlot) {
    plot(rocCurve, legacy.axes = TRUE , main = paste(model.label  
                                                     , " - acc.xval=",acc.xval
                                                     , " - roc.xval=" ,roc.xval  
                                                     , " - roc.xval.2=",roc.xval.2
                                                     , collapse ="" )   )
  }
  
  ### perf.grid 
  perf.grid = NULL
  if (is.null(grid)) { 
    perf.grid = data.frame(data.source = data.source , 
                           resampling.label = resampling.label ,resampling.id = resampling.id, 
                           predictor = c(model.label) , model.id = c(model.id), 
                           acc.train = c(acc.train) , acc.xval = c(acc.xval) , acc.xval.all0 = c(acc.xval.all0), 
                           roc.train = c(roc.train) , roc.xval =c(roc.xval),
                           roc.train.2 = c(roc.train.2) , roc.xval.2 =c(roc.xval.2),
                           roc.xval.min = roc.xval.min , 
                           time = c(tm[[3]]) )
  } else {
    .grid = data.frame(data.source = data.source, 
                       resampling.label = resampling.label ,resampling.id = resampling.id, 
                       predictor = c(model.label) , model.id = c(model.id),
                       acc.train = c(acc.train) , acc.xval = c(acc.xval) , acc.xval.all0 = c(acc.xval.all0), 
                       roc.train = c(roc.train) , roc.xval =c(roc.xval),
                       roc.train.2 = c(roc.train.2) , roc.xval.2 =c(roc.xval.2),
                       roc.xval.min = roc.xval.min , 
                       time = c(tm[[3]]) )
    perf.grid = rbind(grid, .grid)
  }
  
  perf.grid
}

trainAndPredict = function(model.label,model.id,
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
    svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1], .C = 2^(seq(-4, 4)))
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "svmRadial", tuneGrid = svmRGridReduced, 
                    metric = "ROC", fit = FALSE, trControl = controlObject)
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
  roc.train.min = min(roc.train,roc.train.2)
  
  ## logging 
  if (verbose) cat("******************* ", model.label, " <<" , model.id ,  ">> \n")
  if (verbose) cat("** acc.train =",acc.train, " -  acc.train.all0 =",acc.train.all0, " \n")
  if (verbose) cat("** roc.train =",roc.train," -  roc.train.2 =",roc.train.2,"  \n")
  if (verbose) cat("** roc.train.min =",roc.train.min, " \n")
  if (verbose) cat("** Train set:  true %preict =", as.character(sum(ytrain.cat == 'preict')/length(ytrain.cat)) ,  " - %interict =" 
                   , as.character(sum(ytrain.cat == 'interict')/length(ytrain.cat)) ,  " \n")
  if (verbose) cat("** Train set:  predicted %preict =", as.character(sum(pred.train == 'preict')/length(pred.train)) ,  " - %interict =" 
                   , as.character(sum(pred.train == 'interict')/length(pred.train)) ,  " \n")
  if (verbose) cat("** Test set:  predicted %preict =", as.character(sum(pred.test == 'preict')/length(pred.test)) ,  " - %interict =" 
                   , as.character(sum(pred.test == 'interict')/length(pred.test)) ,  " \n")
  
  list(pred.prob.train, pred.train, pred.prob.test, pred.test)
}

######################################################## CONSTANTS 
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

############ models grids 
Dog_1.model = data.frame(model = c( "Boosted Trees C5.0 (Mean sd)" "NN_QUANTILES_REDUCED") , 
                         model.id = c(BOOSTED_TREE_MEAN_SD , NN_QUANTILES_REDUCED) ) 

Dog_2.model = data.frame(model = c("PM_QUANTILES_SCALED" , "SVM_MEAN_SD_REDUCED") , 
                         model.id = c(PM_QUANTILES_SCALED , SVM_MEAN_SD_REDUCED) ) 

Dog_3.model = data.frame(model = c("BOOSTED_TREE_MEAN_SD" , "PM_QUANTILES_SCALED") , 
                         model.id = c(BOOSTED_TREE_MEAN_SD , PM_QUANTILES_SCALED) ) 

Dog_4.model = data.frame(model = c("BOOSTED_TREE_QUANTILES_REDUCED" , "SVM_QUANTILES_REDUCED") , 
                         model.id = c(BOOSTED_TREE_QUANTILES_REDUCED , SVM_QUANTILES_REDUCED) ) 

Dog_5.model = data.frame(model = c("Pen. Mod. (Quant scaled)" , "BOOSTED_TREE_QUANTILES") , 
                         model.id = c(PM_QUANTILES_SCALED , BOOSTED_TREE_QUANTILES) )

Patient_1.model = data.frame(model = c("CLASS_TREE_MEAN_SD" "BAGGING_TREE_MEAN_SD") , 
                             model.id = c(CLASS_TREE_MEAN_SD , BAGGING_TREE_MEAN_SD) )

Patient_2.model = data.frame(model = c("SVM_MEAN_SD_SCALED" , "PLSDA_MEAN_SD_SCALED") , 
                             model.id = c(SVM_MEAN_SD_SCALED , PLSDA_MEAN_SD_SCALED) )

### check 
models.per.ds = nrow(Dog_1.model)
if (nrow(Dog_2.model) != models.per.ds | 
      nrow(Dog_3.model) != models.per.ds |
      nrow(Dog_4.model) != models.per.ds |
      nrow(Dog_5.model) != models.per.ds |
      nrow(Patient_1.model) != models.per.ds |
      nrow(Patient_2.model) != models.per.ds) stop("number of model per data set must be equal.")


############# model selection ... 
verbose = T
doPlot = F 

perf.grid = NULL

controlObject <- trainControl(method = "boot", number = 30 , 
                              summaryFunction = twoClassSummary , classProbs = TRUE)

# controlObject <- trainControl(method = "repeatedcv", repeats = 5, number = 10 , 
#                               summaryFunction = twoClassSummary , classProbs = TRUE)

##dss = c("Dog_1","Dog_2","Dog_3","Dog_4","Dog_5","Patient_1","Patient_2")
dss = c("Dog_1")
cat("|---------------->>> data set to process: <<",dss,">> ..\n")

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
    cat("removing ",length(PredToDel$numbers.discarded)," predictors that make ill-conditioned square matrix: ", paste(colnames(Xtrain_mean_sd) [PredToDel$numbers.discarded] , collapse=" " ) , " ... \n")
    Xtest_mean_sd  =  Xtest_mean_sd  [,-PredToDel$numbers.discarded]
    Xtrain_mean_sd =  Xtrain_mean_sd [,-PredToDel$numbers.discarded]
  }
  
  ### removing near zero var predictors 
  PredToDel = nearZeroVar(Xtrain_quant)
  if (length(PredToDel) > 0) {
    cat("removing ",length(PredToDel)," nearZeroVar predictors: ", paste(colnames(Xtrain_quant) [PredToDel] , collapse=" " ) , " ... \n")
    Xtest_quant  =  Xtest_quant  [,-PredToDel]
    Xtrain_quant =  Xtrain_quant [,-PredToDel]
  }
  
  PredToDel = nearZeroVar(Xtrain_mean_sd)
  if (length(PredToDel) > 0) {
    cat("removing ",length(PredToDel)," nearZeroVar predictors: ", paste(colnames(Xtrain_mean_sd) [PredToDel] , collapse=" " ) , " ... \n")
    Xtest_mean_sd  =  Xtest_mean_sd  [,-PredToDel]
    Xtrain_mean_sd =  Xtrain_mean_sd [,-PredToDel]
  }
  
  ### A. reduced+scaled  
  Xtest_quant.reduced = Xtrain_quant.reduced = NULL 
  Xtest_mean_sd.reduced = Xtrain_mean_sd.reduced = NULL 
  
  # rmoving high correlated predictors on Xtrain_quant
  PredToDel = findCorrelation(cor( Xtrain_quant )) 
  cat("PLS:: on Xtrain_quant removing ",length(PredToDel), " predictors: ",paste(colnames(Xtrain_quant) [PredToDel] , collapse=" " ) , " ... \n")
  Xtest_quant.reduced =  Xtest_quant  [,-PredToDel]
  Xtrain_quant.reduced = Xtrain_quant [,-PredToDel]
  
  # rmoving high correlated predictors Xtrain_mean_sd
  PredToDel = findCorrelation(cor( Xtrain_mean_sd )) 
  cat("PLS:: on Xtrain_mean_sd removing ",length(PredToDel), " predictors: ",paste(colnames(Xtrain_mean_sd) [PredToDel] , collapse=" " ) , " ... \n")
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
  
  ###################### resempling 
  k = 10
  folds = kfolds(k,dim(Xtrain_mean_sd)[1])
  while ( ! verify.kfolds(k,folds,ytrain.cat,'preict') ) {
    if (verbose) cat("--k-fold:: generated bad folds :: retrying ... \n")
    folds = kfolds(k,dim(Xtrain_mean_sd)[1])
  }
  
  for(j in 1:k) {  
    if (verbose) cat("--k-fold:: ",j, "/",k , "\n")
    ## full 
    Xtrain_mean_sd.train <- Xtrain_mean_sd[ folds != j,]
    Xtrain_mean_sd.xval <- Xtrain_mean_sd[folds == j,]
    Xtrain_quant.train <- Xtrain_quant[ folds != j ,]
    Xtrain_quant.xval <- Xtrain_quant[folds == j,]
    
    ## scaled 
    Xtrain_mean_sd.scaled.train <- Xtrain_mean_sd.scaled[ folds != j,]
    Xtrain_mean_sd.scaled.xval <- Xtrain_mean_sd.scaled[folds == j,]
    Xtrain_quant.scaled.train <- Xtrain_quant.scaled[ folds != j,]
    Xtrain_quant.scaled.xval <- Xtrain_quant.scaled[folds == j,]
    
    ## reduced 
    Xtrain_mean_sd.reduced.train <- Xtrain_mean_sd.reduced[ folds != j,]
    Xtrain_mean_sd.reduced.xval <- Xtrain_mean_sd.reduced[folds == j,]
    Xtrain_quant.reduced.train <- Xtrain_quant.reduced[ folds != j,]
    Xtrain_quant.reduced.xval <- Xtrain_quant.reduced[folds == j,]
    
    ## y 
    ytrain.cat.train = ytrain.cat[folds != j]
    ytrain.cat.xval = ytrain.cat[folds == j]
    
    ####### model grid
    grid = NULL 
    
    if (ds == "Dog_1") {
      grid = Dog_1.model
    } else if (ds == "Dog_2") {
      grid = Dog_2.model
    } else if (ds == "Dog_3") {
      grid = Dog_3.model
    } else if (ds == "Dog_4") {
      grid = Dog_4.model
    } else if (ds == "Dog_5") {
      grid = Dog_5.model
    } else if (ds == "Patient_1") {
      grid = Patient_1.model
    } else if (ds == "Patient_2") {
      grid = Patient_2.model
    } else {
      stop("ma che modello ha vinto!")
    }
    
    for ( mo in 1:nrow(grid) ) {
      model.id = grid[mo,]$model.id
      model.label = as.character(grid[mo,]$model) 
      if (verbose) cat("--k-fold:: ",j, "/",k , " model::",model.label,"<",model.id,"> \n")
      
      ## data set 
      if (model.id %% 6 == 0) {
        Xtrain = Xtrain_quant.reduced.train
        Xtest  = Xtrain_quant.reduced.xval
      } else if (model.id %% 6 == 1) {
        Xtrain = Xtrain_mean_sd.train
        Xtest  = Xtrain_mean_sd.xval
      } else if (model.id %% 6 == 2) {
        Xtrain = Xtrain_quant.train
        Xtest  = Xtrain_quant.xval
      } else if (model.id %% 6 == 3) {
        Xtrain = Xtrain_mean_sd.scaled.train
        Xtest  = Xtest_mean_sd.scaled.xval
      } else if (model.id %% 6 == 4) {
        Xtrain = Xtrain_quant.scaled.train
        Xtest  = Xtest_quant.scaled.xval
      } else if (model.id %% 6 == 5) {
        Xtrain = Xtrain_mean_sd.reduced.train
        Xtest  = Xtest_mean_sd.reduced.xval
      } else {
        stop("ma che modello ha vinto (data set) !! ")
      }
      
      ### train & predict 
      ptm <- proc.time()
      ll = trainAndPredict (model.label,model.id,
                            Xtrain, ytrain.cat.train,
                            Xtest,
                            verbose=F)
      tm = proc.time() - ptm
      
      pred.prob.train = ll[[1]] 
      pred.train = ll[[2]]
      pred.prob.test = ll[[3]]    #### <<<<<<<<<--------------------------
      pred.test = ll[[4]]
      
      perf.grid = measure (ds, 
                           "k-fold",j, 
                           model.label,model.id,
                           pred.prob.train , pred.prob.test , 
                           pred.train , pred.test,
                           ytrain.cat.train, ytrain.cat.xval,
                           tm, grid = perf.grid,verbose=F, doPlot=F)
    } 
  }
}
###### mean
perf.grid.mean = ddply(perf.grid , 
                       .(data.source,resampling.label,predictor,model.id),  
                       function(x) c(roc.xval.min=min(x$roc.xval.min),
                                     roc.xval.max=max(x$roc.xval.min),
                                     roc.xval.avg=mean(x$roc.xval.min),
                                     roc.xval.sd=sd(x$roc.xval.min) , 
                                     time_min=sum(x$time)/60))

## saving on disk perf.grids ...
if (verbose) write.csv(perf.grid,quote=FALSE,file=paste0(getBasePath(),"k_fold_perf_grid_class.csv"), row.names=FALSE)
write.csv(perf.grid.mean,quote=FALSE,file=paste0(getBasePath(),"k_fold_perf_grid_class_mean.csv"), row.names=FALSE)

perf.grid
perf.grid.mean


