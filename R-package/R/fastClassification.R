#' Trains a specified classification model on the given train set and predicts on the given test set. 
#' 
#' @param Ytrain the output variable as numeric vector
#' @param Xtrain the encoded \code{data.frame} of train data. Must be a \code{data.frame} of \code{numeric}
#' @param Xtest the encoded \code{data.frame} of test data. Must be a \code{data.frame} of \code{numeric}
#' @param model.label a string specifying which model to use. 
#' @param controlObject a list of values that define how this function acts. Must be a caret \code{trainControl} object 
#' for all models except that for \code{'xgbTreeGTJ'}.
#' @param best.tuning \code{TRUE} to use more dense tuning grid or custom routine/tuning grid if available 
#' @param verbose \code{TRUE} to enable verbose mode. 
#' @param removePredictorsMakingIllConditionedSquareMatrix_forLinearModels \code{TRUE} for removing predictors making 
#' ill-conditioned square matrices in case of fragile linear models.
#' @param xgb.metric.fun custom function to optmize/minimize for \code{'xgbTreeGTJ'}. 
#' @param xgb.maximize \code{TRUE} to maximize the specified \code{xgb.metric.fun}. 
#' @param metric.label the label of function to optmize/minimize. 
#' @param xgb.foldList custom resampling folds list for \code{'xgbTreeGTJ'}. 
#' @param xgb.eta custom \code{eta} parameter for \code{'xgbTreeGTJ'}. 
#' @param xgb.max_depth custom \code{max_depth} parameter for \code{'xgbTreeGTJ'}. 
#' @param xgb.cv.default \code{TRUE} for using \code{xgboost::xgb.cv} function (mandatory in case of fix nrounds), \code{FALSE} for using the internal 
#' \code{ff.xgb.cv} function. The main advantage of the latter is that it doesn't need to restart nrounds in case for the specified nrounds 
#' cross validation error is still decreasing.   
#' @param xgb.param custom parameters for XGBoost. 
#' @param ... arguments passed to the regression routine.  
#' 
#' @examples
#' ## suppress warnings raised because of few obs 
#' warn_def = getOption('warn')
#' options(warn=-1)
#'
#' ## data 
#' Xtrain <- data.frame( a = rep(1:10 , each = 2), b = 20:1, 
#'                       c = rep(as.Date(c("2007-06-22", "2004-02-13")),10) , d = 20:1)
#' Xtest <- data.frame( a = rep(2:11 , each = 2), b = 1:20, 
#'                      c = rep(as.Date(c("2007-03-01", "2004-05-23")),10) , d = 1:20)
#' Ytrain = c(rep(1,10),rep(0,10))
#'
#' ## encode datasets 
#' l = ff.makeFeatureSet(Xtrain,Xtest,c("C","N","D","N"))
#' Xtrain = l$traindata
#' Xtest = l$testdata
#'
#' ## make a caret control object 
#' controlObject <- trainControl(method = "repeatedcv", repeats = 2, number = 3 , 
#'                               summaryFunction = twoClassSummary , classProbs = TRUE)
#' tp = ff.trainAndPredict.class(Ytrain=Ytrain ,
#'                              Xtrain=Xtrain , 
#'                              Xtest=Xtest, 
#'                              model.label = "svmRadial" , 
#'                              controlObject=controlObject, 
#'                              verbose=TRUE , 
#'                              best.tuning=TRUE)
#'
#' pred_test = tp$pred
#' model = tp$model
#' elapsed.secs = tp$secs
#'
#' bestTune = l$model$bestTune
#' best_ROC = max(tp$model$results$ROC)
#'
#' ## restore warnings 
#' options(warn=warn_def)
#' @export
#' @return a list of test predictions, model and number of excecuting seconds.  
#' 

ff.trainAndPredict.class = function(Ytrain ,
                                    Xtrain , 
                                    Xtest , 
                                    model.label , 
                                    controlObject, 
                                    best.tuning = FALSE, 
                                    verbose = FALSE, 
                                    removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = TRUE, 
                                    metric.label = 'auc', 
                                    xgb.metric.fun = NULL, 
                                    xgb.maximize =FALSE, 
                                    xgb.foldList = NULL,
                                    xgb.eta = NULL,
                                    xgb.max_depth = NULL, 
                                    xgb.cv.default = TRUE, 
                                    xgb.param = NULL, 
                                    ... ) {
  
  model = NULL 
  pred = NULL
  pred.prob = NULL 
  secs = NULL
  
  checkModelName(model.label,regression=FALSE)
  
  ### trainAndPredictInternal
  trainAndPredictInternal = function(model.label,Ytrain,Xtrain,Xtest,controlObject,removePredictorsMakingIllConditionedSquareMatrix_forLinearModels) {
    
    ## caret metric label 
    getCaretMetric = function(metric) {
      ret = NULL
      if (metric == "auc") {
        ret = "ROC"
      } else {
        stop(paste0("unrecognized metric:",metric))
      }
      return(ret)
    }
    
    ## remove predictors making ill-conditioned square matrix for fragile linear models
    fs = removePredictorsMakingIllConditionedSquareMatrix_IFFragileLinearModel(Xtrain=Xtrain, 
                                                                                             Xtest=Xtest, 
                                                                                             model.label=model.label,
                                                                                             removePredictorsMakingIllConditionedSquareMatrix_forLinearModels=removePredictorsMakingIllConditionedSquareMatrix_forLinearModels, 
                                                                                             regression = FALSE)
    Xtrain = fs$Xtrain
    Xtest = fs$Xtest
    
    if (model.label == "glm") { ## logistic reg   
      l = getCaretFactors(y=Ytrain)
      model <- train( x = Xtrain , y = l$y.cat , method = "glm", metric = getCaretMetric(metric=metric.label) , trControl = controlObject)
      pred.prob = predict(model , Xtest , type = "prob")[,l$fact.sign]
      pred = predict(model,Xtest)
    } else if (model.label == "lda") { ## lda  
      l = getCaretFactors(y=Ytrain)
      model <- train( x = Xtrain , y = l$y.cat,  method = "lda" , metric = getCaretMetric(metric=metric.label) , trControl = controlObject)
      pred.prob = predict(model , Xtest , type = "prob")[,l$fact.sign]
      pred = predict(model,Xtest)
    } else if (model.label == "pls") { ## pls  
      l = getCaretFactors(y=Ytrain)
      model <- train( x = Xtrain , y = l$y.cat,  
                      method = "pls", tuneGrid = expand.grid(.ncomp = 1:10), 
                      metric = getCaretMetric(metric=metric.label) , trControl = controlObject)
      pred.prob = predict(model , Xtest , type = "prob")[,l$fact.sign]
      pred = predict(model,Xtest)
    } else if (model.label == "glmnet") { ## glmnet  
      l = getCaretFactors(y=Ytrain)
      glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1), .lambda = seq(.01, .2, length = 40))
      model <- train( x = Xtrain , y = l$y.cat,  
                      method = "glmnet", tuneGrid = glmnGrid, 
                      metric = getCaretMetric(metric=metric.label), trControl = controlObject)
      pred.prob = predict(model , Xtest , type = "prob")[,l$fact.sign]
      pred = predict(model,Xtest)
    } else if (model.label == "pam") { ## pam  
      l = getCaretFactors(y=Ytrain)
      nscGrid <- data.frame(.threshold = 0:25)
      model <- train( x = Xtrain , y = l$y.cat,  
                      method = "pam", tuneGrid = nscGrid, 
                      metric = getCaretMetric(metric=metric.label),  trControl = controlObject)
      pred.prob = predict(model , Xtest , type = "prob")[,l$fact.sign]
      pred = predict(model,Xtest)
    } else if (model.label == "nnet") { # neural networks 
      l = getCaretFactors(y=Ytrain)
      nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
      maxSize <- max(nnetGrid$.size)
      numWts <- 1*(maxSize * ( (dim(Xtrain)[2]) + 1) + maxSize + 1)
      model <- train( x = Xtrain , y = l$y.cat,  
                      method = "nnet", metric = getCaretMetric(metric=metric.label), 
                      preProc = c( "spatialSign") , 
                      tuneGrid = nnetGrid , trace = FALSE , maxit = 2000 , 
                      MaxNWts = numWts, trControl = controlObject)
      pred.prob = predict(model , Xtest , type = "prob")[,l$fact.sign]
      pred = predict(model,Xtest)
    } else if (model.label == "svmRadial") { ## svm 
      l = getCaretFactors(y=Ytrain)
      svmRGridReduced <- expand.grid(.sigma = kernlab::sigest(as.matrix(Xtrain)), .C = 2^(seq(-4, 4)))
      model <- train( x = Xtrain , y = l$y.cat,
                      method = "svmRadial", tuneGrid = svmRGridReduced, 
                      metric = getCaretMetric(metric=metric.label), fit = FALSE, trControl = controlObject)
      pred.prob = predict(model , Xtest , type = "prob")[,l$fact.sign]
      pred = predict(model,Xtest)
    } else if (substr(x = model.label, start = 1 , stop = nchar('glmnet_alpha_')) == 'glmnet_alpha_') { 
      alpha <- as.numeric(substr(x = model.label, start = (nchar('glmnet_alpha_')+1) , stop = nchar(model.label)))
      stopifnot(!is.na(alpha))
      
      ## metric 
      .metric.label = getCaretMetric(metric = metric.label)
      if (.metric.label != "ROC") stop(paste0("unsupported metric for libsvm: ",metric.label))
      
      ## 
      cvfit = glmnet::cv.glmnet(as.matrix(Xtrain), Ytrain, family = "binomial", type.measure = "auc" , 
                                nfolds = controlObject$number , alpha = alpha)
      
      ##
      pred.prob = as.numeric(predict(cvfit, newx = as.matrix(Xtest), s = "lambda.min", type = "response"))
      pred = ifelse(pred.prob>0.5,1,0) 
      
      ##
      alist = list()
      alist[getCaretMetric(metric=metric.label)] = max(cvfit$cvm)
      model = list(
        results = alist,
        bestTune = data.frame(lambda.min = cvfit$lambda.min))
      
    } else if (model.label == "libsvm") { ## e1071 
      
      ## metric 
      .metric.label = getCaretMetric(metric = metric.label)
      if (.metric.label != "ROC") stop(paste0("unsupported metric for libsvm: ",metric.label))
      
      ## 
      l = getCaretFactors(y=Ytrain)
      tuneGrid <- expand.grid(gamma = kernlab::sigest(as.matrix(Xtrain)) , C = 2^(seq(-4, 4)) , metric.mean = NA , metric.sd = NA)
      colnames(tuneGrid)[3] = .metric.label
      colnames(tuneGrid)[4] = paste0(.metric.label,"SD")
      
      ## 
      if (controlObject$method != "repeatedcv") stop(paste0("unsupported resampling method for libsvm: ",controlObject$method))
      index = caret::createMultiFolds(y=Ytrain, controlObject$number, controlObject$repeats)
      indexOut <- lapply(index, function(training, allSamples) allSamples[-unique(training)], allSamples = seq(along = Ytrain))
      
      ##
      aL = lapply( 1:nrow(tuneGrid) , function(j) {
        rocs = rep(NA,length(index))
        lapply (seq_along(index) , function(i) {
          fit = e1071::svm(x = Xtrain[ index[[i]] , ] , y = Ytrain[index[[i]]] , kernel = "radial" , gamma = tuneGrid[j,]$gamma , cost = tuneGrid[j,]$C) 
          pred = predict(fit , Xtrain[ indexOut[[i]] , ])
          roc_1 = verification::roc.area(Ytrain[indexOut[[i]]] , pred )$A
          #roc_2 = as.numeric( pROC::auc(pROC::roc(response = l$y.cat[indexOut[[i]]], predictor = pred, levels = levels(l$y.cat) )))
          #rocs[i] <<- min(roc_1,roc_2)
          rocs[i] <<- roc_1
        })
        tuneGrid[j,3] <<- mean(rocs)
        tuneGrid[j,4] <<- sd(rocs)
        if (verbose) cat(">>> [",j,"/",nrow(tuneGrid),"] gamma:",tuneGrid[j,]$gamma," - cost:",tuneGrid[j,]$C,"--> AUC:",tuneGrid[j,3],"  ... \n")
      })
      tuneGrid = tuneGrid[order(tuneGrid[,3] , decreasing = T) , ]
      
      ##
      fit = e1071::svm(x = Xtrain , y = Ytrain , kernel = "radial" , gamma = tuneGrid[1,]$gamma , cost = tuneGrid[1,]$C)
      pred.prob = predict(fit , Xtest)
      pred = ifelse(pred.prob>0.5,1,0) 

      model = list(
        results = tuneGrid,
        bestTune = tuneGrid[1,1:2,drop=F])
      
    } else if (model.label == "knn") { ## knn 
      l = getCaretFactors(y=Ytrain)
      model <- train( x = Xtrain , y =  l$y.cat,
                      method = "knn", 
                      tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)),
                      metric = getCaretMetric(metric=metric.label),  trControl = controlObject)
      pred.prob = predict(model , Xtest , type = "prob")[,l$fact.sign]
      pred = predict(model,Xtest)
    } else if (model.label == "rpart") { ## class trees 
      l = getCaretFactors(y=Ytrain)
      model <- train( x = Xtrain , y = l$y.cat,  
                      method = "rpart", tuneLength = 30, 
                      metric = getCaretMetric(metric=metric.label), trControl = controlObject)
      pred.prob = predict(model , Xtest , type = "prob")[,l$fact.sign]
      pred = predict(model,Xtest)
    } else if (model.label == "C5.0") { ## boosted trees 
      l = getCaretFactors(y=Ytrain)
      if (! best.tuning) {
        model <- train( x = Xtrain , y = l$y.cat,  
                        method = "C5.0",  metric = getCaretMetric(metric=metric.label), trControl = controlObject)
      } else { 
        model <- train( x = Xtrain , y = l$y.cat,  
                        tuneGrid = expand.grid(.trials = c(1, (1:10)*10), .model = "tree", .winnow = c(TRUE, FALSE) ),
                        method = "C5.0",  metric = getCaretMetric(metric=metric.label), trControl = controlObject)
      }
      pred.prob = predict(model , Xtest , type = "prob")[,l$fact.sign]
      pred = predict(model,Xtest)
    } else if (model.label == "bag") { ## bagging trees 
      l = getCaretFactors(y=Ytrain)
      if (! best.tuning) {
        model <- train( x = Xtrain , y = l$y.cat,  
                        method = "bag",  metric = getCaretMetric(metric=metric.label), trControl = controlObject, B = 50 ,
                        bagControl = bagControl(fit = plsBag$fit,
                                                predict = plsBag$pred,
                                                aggregate = plsBag$aggregate))
      } else {
        model <- train( x = Xtrain , y = l$y.cat,  
                        method = "bag",  metric = getCaretMetric(metric=metric.label) , trControl = controlObject, 
                        tuneGrid = data.frame(vars = seq(1, 15, by = 2)), 
                        bagControl = bagControl(fit = plsBag$fit,
                                                predict = plsBag$pred,
                                                aggregate = plsBag$aggregate))
      }
      pred.prob = predict(model , Xtest , type = "prob")[,l$fact.sign]
      pred = predict(model,Xtest)
    } else if (model.label == "xgbTreeGTJ") {  ### xgbTreeGTJ 
      
      ## param 
      param = NULL
      if (! is.null(xgb.param)) {
        param = xgb.param
        if (! is.null(xgb.eta)) stop("xgb.eta must be NULL if xgb.param is not NULL")
        if (! is.null(xgb.max_depth)) stop("xgb.max_depth must be NULL if xgb.param is not NULL")
      } else {
        param <- list("objective" = "binary:logistic" ,
                      "eval_metric" = "auc" , 
                      "min_child_weight" = 6 , 
                      "subsample" = 0.7 , 
                      "colsample_bytree" = 0.6 , 
                      "scale_pos_weight" = 0.8 , 
                      "silent" = 1 , 
                      "max_depth" = 8 , ### <<<< 1?
                      "max_delta_step" = 2)
        
        param['eta'] = 0.02
        if (! is.null(xgb.eta)) param['eta'] = xgb.eta
        if (! is.null(xgb.max_depth)) param['max_depth'] = xgb.max_depth
      }
      
      
      ## fix nrounds? 
      fix.nround = FALSE
      nrounds = 400
      fPar = list(...)
      if (length(fPar)>0) {
        if ('nrounds' %in% names(fPar)) {
          fix.nround = TRUE 
          nrounds = as.integer(fPar['nrounds'])
        }
      }
      
      if (verbose) cat('>> xgbTreeGTJ fix.nround:',fix.nround,' - nrounds:',nrounds,'\n')
      if (verbose) cat(">> xgbTreeGTJ params:\n")
      if (verbose) print(param)
      
      xgb = xgb_train_and_predict (train_set = Xtrain,
                                   y = Ytrain,  
                                   test_set = Xtest, 
                                   foldList = xgb.foldList, 
                                   xgb.metric.fun = xgb.metric.fun,
                                   xgb.maximize = xgb.maximize, 
                                   xgb.metric.label = metric.label, 
                                   param = param,
                                   cv.nround = nrounds , 
                                   fix.nround = fix.nround, 
                                   nfold = min(controlObject$number,nrow(Xtrain)) , 
                                   xgb.cv.default = xgb.cv.default,
                                   verbose=verbose)
      pred.prob = xgb$pred
      pred = ifelse(pred.prob>0.5,1,0) 
      early.stop = xgb$early.stop
      
      alist = list()
      alist[getCaretMetric(metric=metric.label)] = xgb$perf.cv
      model = list(
        results = alist,
        bestTune = data.frame(early.stop = xgb$early.stop)
      )
    } else {
      stop("unrecognized model.label!")
    }
    return(list(model=model, pred=pred , pred.prob=pred.prob ))
  }
  
  ptm <- proc.time()
  l = plyr::failwith( NULL, trainAndPredictInternal , quiet = !verbose)(model.label=model.label,
                                                                        Ytrain=Ytrain,
                                                                        Xtrain=Xtrain,
                                                                        Xtest=Xtest,
                                                                        controlObject=controlObject, 
                                                                        removePredictorsMakingIllConditionedSquareMatrix_forLinearModels=removePredictorsMakingIllConditionedSquareMatrix_forLinearModels)
  tm = proc.time() - ptm
  secs = as.numeric(tm[3])
  
  ##
  if (! is.null(l)) {
    model = l$model 
    pred = l$pred
    pred.prob = l$pred.prob
  } 
  
  ##
  if (verbose) cat(">> ",model.label,": time elapsed:",secs," secs. [min:",secs/60,"] [hours:",secs/(60*60),"]\n")
  if (verbose) {
    print(model)
  }
  
  ##
  return(list(pred = pred, pred.prob = pred.prob , model = model, secs = secs))
}