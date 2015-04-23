library(caret)
library(Hmisc)
library(verification)
library(pROC)
library(kernlab)
library(subselect)
library(plyr)

#### supported classification models 
All.ClassModels = c("Mode",
                  "LogisticReg" , 
                  "LDA" , 
                  "PLSDA" , 
                  "PMClass" , 
                  "NSC" , 
                  "NNetClass" , 
                  "SVMClass" , 
                  "KNNClass" , 
                  "ClassTrees" , 
                  "BoostedTreesClass" , 
                  "BaggingTreesClass" 
                  ) 

class.trainAndPredict = function( ytrain.cat , 
                                  Xtrain , 
                                  Xtest , 
                                  fact.sign = 'preict', 
                                  model.label , 
                                  controlObject, 
                                  best.tuning = F, 
                                  verbose = T) { 
  
  pred.prob.train = rep(1,nrow(Xtrain))
  pred.train = rep(Mode(ytrain.cat),nrow(Xtrain))
  
  pred.prob.test = rep(1,nrow(Xtest)) ### <<<<<<<<<<<<----------------------------------
  pred.test = rep(Mode(ytrain.cat),nrow(Xtest))

  ## model 
  if (model.label == "LogisticReg") { ## logistic reg 
    model <- train( x = Xtrain , y = ytrain.cat , 
                    method = "glm", metric = "ROC", trControl = controlObject)
  } else if (model.label == "LDA") { ## lda 
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "lda", metric = "ROC" , trControl = controlObject)
  } else if (model.label == "PLSDA") { ## plsda 
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "pls", tuneGrid = expand.grid(.ncomp = 1:10), 
                    metric = "ROC" , trControl = controlObject)
  } else if (model.label == "PMClass") { ## pm 
    glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1), .lambda = seq(.01, .2, length = 40))
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "glmnet", tuneGrid = glmnGrid, 
                    metric = "ROC", trControl = controlObject)
  } else if (model.label == "NSC") { ## nsc 
    nscGrid <- data.frame(.threshold = 0:25)
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "pam", tuneGrid = nscGrid, 
                    metric = "ROC", trControl = controlObject)
  } else if (model.label == "NNetClass") { # neural networks 
    nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
    maxSize <- max(nnetGrid$.size)
    numWts <- 1*(maxSize * ( (dim(Xtrain)[2]) + 1) + maxSize + 1)
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "nnet", metric = "ROC", 
                    preProc = c( "spatialSign") , 
                    tuneGrid = nnetGrid , trace = FALSE , maxit = 2000 , 
                    MaxNWts = numWts, trControl = controlObject)
  } else if (model.label == "SVMClass") { ## svm 
    sigmaRangeReduced <- sigest(as.matrix(Xtrain))
    svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1], .C = 2^(seq(-4, 4)))
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "svmRadial", tuneGrid = svmRGridReduced, 
                    metric = "ROC", fit = FALSE, trControl = controlObject)
  } else if (model.label == "KNNClass") { ## knn 
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "knn", 
                    tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)),
                    metric = "ROC",  trControl = controlObject)
  } else if (model.label == "ClassTrees") { ## class trees 
    model <- train( x = Xtrain , y = ytrain.cat,  
                    method = "rpart", tuneLength = 30, 
                    metric = "ROC", trControl = controlObject)
  } else if (model.label == "BoostedTreesClass") { ## boosted trees 
    if (! best.tuning) {
      model <- train( x = Xtrain , y = ytrain.cat,  
                      method = "C5.0",  metric = "ROC", trControl = controlObject)
    } else { 
      model <- train( x = Xtrain , y = ytrain.cat,  
                      tuneGrid = expand.grid(.trials = c(1, (1:10)*10), .model = "tree", .winnow = c(TRUE, FALSE) ),
                      method = "C5.0",  metric = "ROC", trControl = controlObject)
    }
  } else if (model.label == "BaggingTreesClass") { ## bagging trees 
    if (! best.tuning) {
      model <- train( x = Xtrain , y = ytrain.cat,  
                      method = "bag",  metric = "ROC", trControl = controlObject, B = 50 ,
                      bagControl = bagControl(fit = plsBag$fit,
                                              predict = plsBag$pred,
                                              aggregate = plsBag$aggregate))
    } else {
      model <- train( x = Xtrain , y = ytrain.cat,  
                      method = "bag",  metric = "ROC", trControl = controlObject, 
                      tuneGrid = data.frame(vars = seq(1, 15, by = 2)), 
                      bagControl = bagControl(fit = plsBag$fit,
                                              predict = plsBag$pred,
                                              aggregate = plsBag$aggregate))
    }
  } else if (model.label == "Mode") { ## Mode
    ## do nothing 
  } else {
    stop("unrecognized model")
  }
  
  if (! (model.label == "Mode") ) {
    pred.prob.train = predict(model , Xtrain , type = "prob")[,fact.sign] 
    pred.train = predict(model , Xtrain )
    
    pred.prob.test = predict(model , Xtest , type = "prob")[,fact.sign] ### <<<<<<<<<<<<----------------------------------
    pred.test = predict(model , Xtest )
  }
  
  ## accuracy 
  acc.train = sum(ytrain.cat == pred.train) / length(ytrain.cat) 
  
  ## ROC 
  rocCurve <- pROC::roc(response = ytrain.cat, predictor = pred.prob.train, levels = levels(ytrain.cat) )
  roc.train = as.numeric( pROC::auc(rocCurve) )
  
  roc.train.2 = roc.area(as.numeric(ytrain.cat == fact.sign) , pred.prob.train )$A
  roc.train.min = min(roc.train,roc.train.2)
  
  ## logging 
  if (verbose) cat("******************* ", model.label, " ******************* \n")
  if (verbose) cat("** acc.train =",acc.train, " \n")
  if (verbose) cat("** roc.train =",roc.train," -  roc.train.2 =",roc.train.2,"  \n")
  if (verbose) cat("** roc.train.min =",roc.train.min, " \n")
  
  list(pred.prob.train, pred.train, pred.prob.test, pred.test)
}

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

measure.class = function(
                   pred.prob.train , 
                   pred.prob.xval , 
                   pred.train , 
                   pred.xval,
                   ytrain, 
                   yxval,
                   fact.sign = 'preict',
                   verbose=F, 
                   doPlot=F,
                   label ="") {
  ## accuracy 
  acc.train = sum(ytrain == pred.train) / length(ytrain)
  acc.xval = sum(yxval == pred.xval) / length(yxval) 
  
  ## ROC 
  rocCurve <- pROC::roc(response = ytrain, predictor = pred.prob.train, levels = levels(ytrain) )
  roc.train = as.numeric( pROC::auc(rocCurve) )
  
  rocCurve <- pROC::roc(response = yxval, predictor = pred.prob.xval , levels = levels(yxval) )
  roc.xval = as.numeric( pROC::auc(rocCurve) )
  
  roc.train.2 = roc.area(as.numeric(ytrain == fact.sign) , pred.prob.train )$A
  roc.xval.2 = roc.area(as.numeric(yxval == fact.sign) , pred.prob.xval )$A
  
  roc.xval.min = min(roc.xval.2,roc.xval)
  
  ## logging 
  if (verbose) cat("******************* <<" , label ,  ">>  --  \n")
  if (verbose) cat("** acc.train =",acc.train, " -  acc.xval =",acc.xval, "\n")
  if (verbose) cat("** roc.train =",roc.train," -  roc.xval =",roc.xval,"  \n")
  if (verbose) cat("** roc.train.2 =",roc.train.2," -  roc.xval.2 =",roc.xval.2,"  \n")
  if (verbose) cat("** roc.xval.min =",roc.xval.min, " \n")
  
  ## poltting 
  if (doPlot) {
    plot(rocCurve, legacy.axes = TRUE , main = paste(label  
                                                     , " - acc.xval=",acc.xval
                                                     , " - roc.xval=" ,roc.xval  
                                                     , " - roc.xval.2=",roc.xval.2
                                                     , collapse ="" )   )
  }
  
  roc.xval.min
}
