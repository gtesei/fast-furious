
## notes_seizure_R
library(caret)
library(Hmisc)
library(data.table)
library(verification)
library(pROC)
library(kernlab)

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


trainAndPredict = function(model.label,model.id,
                             Xtrain, ytrain.cat,
                             Xtest,
                             verbose=F) {
  ## model 
  if (model.id.winner >= 1 && model.id.winner <= 6) { ## logistic reg 
    model <- train( x = Xtrain , y = ytrain.cat , 
                    method = "glm", metric = "ROC", trControl = controlObject)
  } else if (model.id.winner >= 7 && model.id.winner <= 12) { ## lda 
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "lda", metric = "ROC" , trControl = controlObject)
  } else if (model.id.winner >= 13 && model.id.winner <= 18) { ## plsda 
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "pls", tuneGrid = expand.grid(.ncomp = 1:10), 
                    metric = "ROC" , trControl = controlObject)
  } else if (model.id.winner >= 19 && model.id.winner <= 24) { ## pm 
    glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1), .lambda = seq(.01, .2, length = 40))
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "glmnet", tuneGrid = glmnGrid, 
                    metric = "ROC", trControl = controlObject)
  } else if (model.id.winner >= 25 && model.id.winner <= 30) { ## nsc 
    nscGrid <- data.frame(.threshold = 0:25)
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "pam", tuneGrid = nscGrid, 
                    metric = "ROC", trControl = controlObject)
  } else if (model.id.winner >= 31 && model.id.winner <= 36) { # neural networks 
    nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
    maxSize <- max(nnetGrid$.size)
    numWts <- 1*(maxSize * ( (dim(Xtrain)[2]) + 1) + maxSize + 1)
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "nnet", metric = "ROC", 
                    preProc = c( "spatialSign") , 
                    tuneGrid = nnetGrid , trace = FALSE , maxit = 2000 , 
                    MaxNWts = numWts, trControl = controlObject)
  } else if (model.id.winner >= 37 && model.id.winner <= 42) { ## svm 
    sigmaRangeReduced <- sigest(as.matrix(Xtrain))
    svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1], .C = 2^(seq(-4, 4)))
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "svmRadial", tuneGrid = svmRGridReduced, 
                    metric = "ROC", fit = FALSE, trControl = controlObject)
  } else if (model.id.winner >= 43 && model.id.winner <= 49) { ## knn 
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "knn", 
                    tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)),
                    metric = "ROC",  trControl = controlObject)
  } else if (model.id.winner >= 49 && model.id.winner <= 54) { ## class trees 
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "rpart", tuneLength = 30, 
                    metric = "ROC", trControl = controlObject)
  } else if (model.id.winner >= 55 && model.id.winner <= 60) { ## boosted trees 
    if (model.id.winner >= 55 && model.id.winner <= 59) {
      ## 55 - 59 
      model <- train( x = Xtrain , y = ytrain.cat,  
                      method = "C5.0",  metric = "ROC", trControl = controlObject)
    } else {
      ## 60 - BOOSTED_TREE_QUANTILES_REDUCED 
      model <- train( x = Xtrain , y = ytrain.cat,  
                      tuneGrid = expand.grid(.trials = c(1, (1:10)*10), .model = "tree", .winnow = c(TRUE, FALSE) ),
                      method = "C5.0",  metric = "ROC", trControl = controlObject)
    }
  } else if (model.id.winner >= 61 && model.id.winner <= 66) { ## bagging trees 
    if (model.id.winner >= 61 && model.id.winner <= 65) {
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
  } else {
    stop("ma che modello ha vinto (modello) !! ")
  }
  
  ## predicting model on Xtrain and Xtest 
  pred.prob.train = predict(model , Xtrain , type = "prob")[,'preict'] 
  pred.train = predict(model , Xtrain )
  
  pred.prob.test = predict(model , Xtest , type = "prob")[,'preict'] ### <<<<<<<<<<<<----------------------------------
  pred.test = predict(model , Xtest )
  
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

######################################################## MAIN LOOP 
sampleSubmission = as.data.frame(fread(paste(getBasePath(type = "data"),"sampleSubmission.csv",sep="") , header = T , sep=","  ))
predVect = rep(-1,dim(sampleSubmission)[1])
predVect.idx = 1

trainPred = NULL 
trainClass = NULL

 #### Model averaging 
topxx = c(0.95 , 0.9 , 0.85 , 0.80, 0.75)
labelAvg = list(lab1 = "top05", lab2 = "top10" , 
                lab3 = "top15", lab4 = "top20", 
                lab5 = "top25")
predVect_topxx = data.frame(predVect_top05 = rep(-1,dim(sampleSubmission)[1]) , predVect_top10 = rep(-1,dim(sampleSubmission)[1]), 
                            predVect_top15 = rep(-1,dim(sampleSubmission)[1]) , predVect_top20 = rep(-1,dim(sampleSubmission)[1]),
                            predVect_top25 = rep(-1,dim(sampleSubmission)[1]) )

# predVect_topxx.train = data.frame(predVect_top05 = rep(-1,dim(sampleSubmission)[1]) , predVect_top10 = rep(-1,dim(sampleSubmission)[1]), 
#                             predVect_top15 = rep(-1,dim(sampleSubmission)[1]) , predVect_top20 = rep(-1,dim(sampleSubmission)[1]),
#                             predVect_top25 = rep(-1,dim(sampleSubmission)[1]) )

############# general settings ... 
verbose = T
doPlot = F 

############ models 

Dog_1.model = data.frame(model = c("SVM (Quant reduced)" , "Boosted Trees C5.0 (Quant scaled)" , "KNN (Mean sd scaled)") , 
                         model.id = c(SVM_QUANTILES_REDUCED , BOOSTED_TREE_QUANTILES_SCALED , NSC_REG_QUANTILES_REDUCED) , 
                         weigth = c(0.6 , 0.5 , 0.4)
                         ) 

Dog_2.model = data.frame(model = c("SVM (Quant reduced)" , "Boosted Trees C5.0 (Mean sd reduced)" , "KNN (Mean sd reduced)") , 
                         model.id = c(SVM_QUANTILES_REDUCED , BOOSTED_TREE_MEAN_SD_REDUCED , KNN_MEAN_SD_REDUCED) , 
                         weigth = c(0.7 , 0.5 , 0.4)
                         ) 

############ 

controlObject <- trainControl(method = "boot", number = 30 , 
                              summaryFunction = twoClassSummary , classProbs = TRUE)

# controlObject <- trainControl(method = "repeatedcv", repeats = 5, number = 10 , 
#                               summaryFunction = twoClassSummary , classProbs = TRUE)

dss = c("Dog_1","Dog_2","Dog_3","Dog_4","Dog_5","Patient_1","Patient_2")
##dss = c("Patient_2")
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
  
  #### partitioning into train , xval ... 
#   set.seed(975)
  set.seed(429494444)
  forTraining <- createDataPartition(ytrain[,1], p = 3/4)[[1]]
  
  ## full 
  Xtrain_mean_sd.train <- Xtrain_mean_sd[ forTraining,]
  Xtrain_mean_sd.xval <- Xtrain_mean_sd[-forTraining,]
  Xtrain_quant.train <- Xtrain_quant[ forTraining,]
  Xtrain_quant.xval <- Xtrain_quant[-forTraining,]
  
  ## scaled 
  Xtrain_mean_sd.scaled.train <- Xtrain_mean_sd.scaled[ forTraining,]
  Xtrain_mean_sd.scaled.xval <- Xtrain_mean_sd.scaled[-forTraining,]
  Xtrain_quant.scaled.train <- Xtrain_quant.scaled[ forTraining,]
  Xtrain_quant.scaled.xval <- Xtrain_quant.scaled[-forTraining,]
  
  ## reduced 
  Xtrain_mean_sd.reduced.train <- Xtrain_mean_sd.reduced[ forTraining,]
  Xtrain_mean_sd.reduced.xval <- Xtrain_mean_sd.reduced[-forTraining,]
  Xtrain_quant.reduced.train <- Xtrain_quant.reduced[ forTraining,]
  Xtrain_quant.reduced.xval <- Xtrain_quant.reduced[-forTraining,]
  
  ## y 
  ytrain.cat.train = ytrain.cat[forTraining]
  ytrain.cat.xval = ytrain.cat[-forTraining] 

  #################################################### 
  
  ##### train winner models on whole train set and predict on test set 
  Xtrain = Xtest = NULL
  model = NULL
  
  ########### Model averaging  
  if (verbose) cat("******************* Model averaging .... \n")
  
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
    
    
  ################################################################### qui
    
  for (tp in 1:nrow(grid) ) {

    if (verbose) cat("*** building model average for ", tp ," models \n")
    
    weigths = rep(-1,nrow(mod.grid))
    prob.mat = matrix( data = rep(-1, nrow(mod.grid)*length(pred.prob.test)) , 
                      nrow = length(pred.prob.test) , ncol = nrow(mod.grid)  )
    
#     prob.mat.train = matrix( data = rep(-1, nrow(mod.grid)*length(pred.prob.train)) , 
#                        nrow = length(pred.prob.train) , ncol = nrow(mod.grid)  )
    
    for ( mi in 1:nrow(mod.grid) ) {
      model.id = mod.grid[mi,]$model.id
      model.label = as.character(mod.grid[mi,]$predictor) 
      weigths[mi] = roc = mod.grid[mi,]$roc.xval.2
      
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
      ll = trainAndPredict (model.label,model.id, 
                           Xtrain, ytrain.cat, Xtest, 
                           verbose=T)
      pred.prob.train.topxx = ll[[1]] 
      pred.train.topxx = ll[[2]]
      pred.prob.test.topxx = ll[[3]]    #### <<<<<<<<<--------------------------
      pred.test.topxx = ll[[4]]
      
      prob.mat[,mi] = pred.prob.test.topxx
#       prob.mat.train[,mi] = pred.prob.train.topxx
    }
    
    #### averaging 
    prob.mat.xx = prob.mat 
#     prob.mat.train.xx = prob.mat.train
    for ( mi in 1:nrow(mod.grid) ) {
      prob.mat.xx[,mi] = prob.mat[,mi] * weigths[mi]
#       prob.mat.train.xx[,mi] = prob.mat.train[,mi] * weigths[mi]
    }
    
    ### update predVect_topxx
    NUM = apply(prob.mat.xx,1,sum)
    DENUM = sum(weigths)
    predVect_topxx[predVect.idx:(predVect.idx+length(pred.prob.test)-1),tp] = NUM * (DENUM^-1)
    
#     ### update predVect_topxx.train
#     NUM = apply(prob.mat.train.xx,1,sum)
#     DENUM = sum(weigths)
#     predVect_topxx.train[predVect.idx:(predVect.idx+length(pred.prob.test)-1),tp] = NUM * (DENUM^-1)
  }
  
  ### update predVects 
  predVect[predVect.idx:(predVect.idx+length(pred.prob.test)-1)] = pred.prob.test
  predVect.idx = predVect.idx + length(pred.prob.test)
  
  ## trainPred
  if (is.null(trainPred)) {
    trainPred = pred.prob.train
    trainClass = ytrain[,2]
  } else {
    trainPred = c(trainPred , pred.prob.train)
    trainClass = c(trainClass , ytrain[,2] )
  }
  if (verbose) cat("** predVect and predVect_topxx updated \n")
}

## submission - top model 
mySub = data.frame(clip = sampleSubmission$clip , preictal = format(predVect  , scientific = F ))
write.csv(mySub,quote=FALSE,file=paste0(getBasePath(),"mySub_class.zat"), row.names=FALSE)

## Calibrating Probabilities - sigmoid - top model 
trainClass.cat = as.factor(trainClass)
levels(trainClass.cat) =  c("interict","preict")
train.df = data.frame(class = trainClass.cat , prob = trainPred )
sigmoidalCal <- glm(  class ~ prob  , data = train.df , family = binomial)
coef(summary(sigmoidalCal)) 
sigmoidProbs <- predict(sigmoidalCal, newdata = data.frame(prob = predVect), type = "response")
mySub2 = data.frame(clip = sampleSubmission$clip , preictal = format(sigmoidProbs,scientific = F))  
write.csv(mySub2,quote=FALSE,file=paste0(getBasePath(),"mySub_sigmoid_calibrat_class.zat"), row.names=FALSE)

## Calibrating Probabilities - Bayes - top model 
library(klaR)
BayesCal <- NaiveBayes( class ~ prob  , data = train.df, usekernel = TRUE)
BayesProbs <- predict(BayesCal, newdata = data.frame(prob = predVect) )
BayesProbs.preict <- BayesProbs$posterior[, "preict"]
mySub3 = data.frame(clip = sampleSubmission$clip , preictal = format(BayesProbs.preict,scientific = F))
write.csv(mySub3,quote=FALSE,file=paste0(getBasePath(),"mySub_bayes_calibrat_class.zat"), row.names=FALSE)

## submission - averaged models 
for (tp in 1:length(topxx) ) {
  label = as.character(labelAvg[tp])
  mySub = data.frame(clip = sampleSubmission$clip , preictal = format( predVect_topxx[,tp]  , scientific = F ))
  write.csv(mySub,quote=FALSE,file=paste(getBasePath(),"mySub_class_" , label , ".zat" , sep=""), row.names=FALSE)
  
#   ## Calibrating Probabilities - sigmoid - top model 
#   trainClass.cat = as.factor(predVect_topxx.train[,tp])
#   levels(trainClass.cat) =  c("interict","preict")
#   train.df = data.frame(class = trainClass.cat , prob = trainPred )
#   sigmoidalCal <- glm(  class ~ prob  , data = train.df , family = binomial)
#   coef(summary(sigmoidalCal)) 
#   sigmoidProbs <- predict(sigmoidalCal, newdata = data.frame(prob = predVect_topxx[,tp]), type = "response")
#   mySub2 = data.frame(clip = sampleSubmission$clip , preictal = format(sigmoidProbs,scientific = F))  
#   write.csv(mySub2,quote=FALSE,file=paste(getBasePath(),"mySub_sigmoid_calibrat_class_",label,".zat",sep=""), row.names=FALSE)
#   
#   ## Calibrating Probabilities - Bayes - top model 
#   library(klaR)
#   BayesCal <- NaiveBayes( class ~ prob  , data = train.df, usekernel = TRUE)
#   BayesProbs <- predict(BayesCal, newdata = data.frame(prob = predVect_topxx[,tp]) )
#   BayesProbs.preict <- BayesProbs$posterior[, "preict"]
#   mySub3 = data.frame(clip = sampleSubmission$clip , preictal = format(BayesProbs.preict,scientific = F))
#   write.csv(mySub3,quote=FALSE,file=paste(getBasePath(),"mySub_bayes_calibrat_class_",label,".zat"), row.names=FALSE)
}
