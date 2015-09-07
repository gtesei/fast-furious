

RMSLE.xgb = function (preds, dtrain,th_err=1.5) {
  obs <- xgboost::getinfo(dtrain, "label")
  if ( sum(preds<0) >0 ) {
    preds = ifelse(preds >=0 , preds , th_err)
  }
  rmsle = sqrt(    sum( (log(preds+1) - log(obs+1))^2 )   /length(preds))
  return(list(metric = "rmsle", value = rmsle))
}

RMSE.xgb = function (preds, dtrain) {
  obs <- xgboost::getinfo(dtrain, "label")
  rmse = caret::RMSE(pred = preds , obs = obs)
  return(list(metric = "rmse", value = rmse))
}

xgb_cross_val = function( data , 
                          y , 
                          foldList, 
                          xgb.metric.fun, 
                          xgb.maximize, 
                          xgb.metric.label,
                          cv.nround = 3000 , 
                          fix.nround = FALSE, 
                          param , 
                          nfold = 5 , 
                          verbose=TRUE) {
  
  stopifnot(identical(xgb.maximize,FALSE))
  
  inCV = TRUE
  early.stop = cv.nround 
  perf.xg = NULL 
  perf.last = NULL
  
  lab = paste('test.',xgb.metric.label,'.mean',sep='')
  
  if (! is.null(foldList)) {
    if (verbose) cat(">>> using resamples in foldList ... \n")
  }
  
  while (inCV) { 
    bst.cv = xgboost::xgb.cv(param=param, data = data, label = y, 
                    nfold = nfold, nrounds=cv.nround , folds = foldList, 
                    feval = xgb.metric.fun , maximize = xgb.maximize, verbose=FALSE)
    
    if (verbose) print(bst.cv)
    early.stop = which(bst.cv[[lab]] == min(bst.cv[[lab]]))
    if (length(early.stop)>1) early.stop = early.stop[length(early.stop)]
    if (fix.nround) {
      inCV = FALSE
      perf.xg = bst.cv[[lab]][cv.nround] 
      early.stop = cv.nround
    } else if ( early.stop < cv.nround || (! is.null(perf.last) && min(bst.cv[[lab]]) > perf.last) )  {
      inCV = FALSE
      perf.xg = min(bst.cv[[lab]])
      if (verbose) cat('>> stopping [',early.stop,'=early.stop < cv.nround=',cv.nround,'] [perf.xg=',perf.xg,'] ... \n') 
    } else {
      if(verbose) cat(">> redo-cv [early.stop == cv.nround=",cv.nround,"] with 2*cv.nround ... \n") 
      cv.nround = cv.nround * 2 
      perf.last = min(bst.cv[[lab]])
    }
    gc()
  }
  
  return(list(early.stop=early.stop,perf.cv=perf.xg))
}

xgb_train_and_predict = function(train_set,
                                 y, 
                                 test_set, 
                                 foldList, 
                                 xgb.metric.fun, 
                                 xgb.maximize, 
                                 xgb.metric.label, 
                                 param,
                                 cv.nround = 3000 , 
                                 fix.nround = FALSE, 
                                 nfold = 5 , 
                                 verbose=TRUE) {
  
  data = rbind(train_set,test_set)
  
  ##
  x = as.matrix(data)
  x = matrix(as.numeric(x),nrow(x),ncol(x))
  
  trind = 1:nrow(train_set)
  teind = (nrow(train_set)+1):nrow(x)
  
  ## cross-valication
  if (verbose) cat(">> xgb: cross-validation ... \n")
  
  xgb_xval = xgb_cross_val (data = x[trind,], 
                            y = y,  
                            foldList = foldList, 
                            xgb.metric.fun = xgb.metric.fun, 
                            xgb.maximize = xgb.maximize, 
                            xgb.metric.label = xgb.metric.label, 
                            cv.nround = cv.nround , 
                            fix.nround = fix.nround, 
                            param = param ,
                            nfold = nfold , 
                            verbose=verbose)
  
  ## prediction
  if (fix.nround) stopifnot(xgb_xval$early.stop == cv.nround)
  if (verbose) cat('>> xgb: prediction [early.stop:',xgb_xval$early.stop,'] ... \n')
  dtrain <- xgboost::xgb.DMatrix(x[trind,], label = y)
  bst = xgboost::xgb.train(param = param,  
                  dtrain , 
                  nrounds = xgb_xval$early.stop, 
                  feval = xgb.metric.fun , maximize = xgb.maximize , verbose = FALSE)
  
  # workaround for length 1 preds 
  if (length(teind)>1){
    pred = xgboost::predict(bst,x[teind,])
  } else {
    yy = rbind(x[teind,],x[teind,])
    .pred = xgboost::predict(bst,yy)
    pred = .pred[1]
  }
  
  return(list(pred=pred,
              perf.cv=xgb_xval$perf.cv,
              early.stop=xgb_xval$early.stop))
}

checkModelName = function(model.label) {
 models = c('bayesglm','glm','glmStepAIC','rlm','knn','pls','ridge','enet','svmRadial','treebag', 
            'gbm','rf','cubist','avNNet','xgbTreeGTJ','xgbTree')
 if (! model.label %in% models) stop(paste0('unrecognized model name: ',model.label)) 
}

removePredictorsMakingIllConditionedSquareMatrix_IFFragileLinearModel = function (Xtrain, 
                                                                                  Xtest, 
                                                                                  model.label,
                                                                                  removePredictorsMakingIllConditionedSquareMatrix_forLinearModels) {
  
  fragile_LinearModels = c('rlm','pls','ridge','enet')
  if (! model.label %in% fragile_LinearModels || ! removePredictorsMakingIllConditionedSquareMatrix_forLinearModels) {
    return (list(
      Xtrain = Xtrain, 
      Xtest = Xtest
    )) 
  }
  
  l = ff.featureFilter (Xtrain,
                        Xtest,
                        removeOnlyZeroVariacePredictors=TRUE,
                        performVarianceAnalysisOnTrainSetOnly = TRUE , 
                        removePredictorsMakingIllConditionedSquareMatrix = TRUE, 
                        removeHighCorrelatedPredictors = FALSE, 
                        featureScaling = FALSE)
  
  return (list(
    Xtrain = l$traindata, 
    Xtest = l$testdata
  ))  
}

#' Trains a specified model on the given train set and predicts on the given test set. 
#' 
#' @param Ytrain the output variable as numeric vector
#' @param Xtrain the encoded \code{data.frame} of train data. Must be a \code{data.frame} of \code{numeric}
#' @param Xtest the encoded \code{data.frame} of test data. Must be a \code{data.frame} of \code{numeric}
#' @param model.label a string specifying which model to use. Possible values are \code{'lm'}, \code{'bayesglm'}, 
#' \code{'glm'}, \code{'glmStepAIC'}, \code{'rlm'}, \code{'knn'}, \code{'pls'}, \code{'ridge'}, \code{'enet'}, 
#' \code{'svmRadial'}, \code{'treebag'}, \code{'gbm'}, \code{'rf'}, \code{'cubist'}, \code{'avNNet'}, 
#' \code{'xgbTreeGTJ'}, \code{'xgbTree'}
#' @param controlObject a list of values that define how this function acts. Must be a caret \code{trainControl} object 
#' for all models except that for \code{'xgbTreeGTJ'} and \code{'xgbTree'}. In the latter case only if \code{best.tuning} is \code{TRUE}.
#' @param best.tuning \code{TRUE} to use more dense tuning grid or custom routine if available 
#' @param verbose \code{TRUE} to enable verbose mode. 
#' @param removePredictorsMakingIllConditionedSquareMatrix_forLinearModels \code{TRUE} for removing predictors making 
#' ill-conditioned square matrices in case of fragile linear models, i.e. \code{c('rlm','pls','ridge','enet')}.
#' @param xgb.metric.fun custom function to optmize/minimize for \code{'xgbTreeGTJ'} and \code{'xgbTree'}. 
#' In the latter case only if \code{best.tuning} is \code{TRUE}.
#' @param xgb.maximize \code{TRUE} to maximize the specified \code{xgb.metric.fun}. Only for \code{'xgbTreeGTJ'} and \code{'xgbTree'}. 
#' In the latter case only if \code{best.tuning} is \code{TRUE}.
#' @param xgb.metric.label custom label of function to optmize/minimize for \code{'xgbTreeGTJ'} and \code{'xgbTree'}. 
#' In the latter case only if \code{best.tuning} is \code{TRUE}.
#' @param xgb.foldList custom resampling folds list for \code{'xgbTreeGTJ'} and \code{'xgbTree'}. 
#' In the latter case only if \code{best.tuning} is \code{TRUE}.
#' @param xgb.eta custom \code{eta} parameter for \code{'xgbTreeGTJ'} and \code{'xgbTree'}. 
#' In the latter case only if \code{best.tuning} is \code{TRUE}.
#' @param ... arguments passed to the regression routine.  
#' 
#' @examples
#' ## suppress warnings raised because of few obs 
#' warn_def = getOption('warn')
#' options(warn=-1)
#'
#' ## data 
#' Xtrain <- data.frame( a = rep(1:10 , each = 2), b = 20:1, 
#' c = rep(as.Date(c("2007-06-22", "2004-02-13")),10) )
#' Xtest <- data.frame( a = rep(2:11 , each = 2), b = 1:20, 
#' c = rep(as.Date(c("2007-03-01", "2004-05-23")),10) )
#' Ytrain = 1:20 + runif(nrow(Xtrain))
#' 
#' ## encode datasets 
#' l = ff.makeFeatureSet(Xtrain,Xtest,c("C","N","D"))
#' Xtrain = l$traindata
#' Xtest = l$testdata
#'
#' ## make a caret control object 
#' controlObject <- trainControl(method = "repeatedcv", repeats = 1, number = 2)
#'
#' tp = ff.trainAndPredict.reg(Ytrain=Ytrain ,
#'                           Xtrain=Xtrain , 
#'                           Xtest=Xtest , 
#'                           model.label = "cubist" , 
#'                           controlObject=controlObject)
#'
#' pred_test = tp$pred
#' model = tp$model
#' elapsed.secs = tp$secs
#'
#' ## restore warnings 
#' options(warn=warn_def)
#' @export
#' @return a list of test predictions, model and number of excecuting seconds.  
#' 
ff.trainAndPredict.reg = function(Ytrain ,
                                 Xtrain , 
                                 Xtest , 
                                 model.label , 
                                 controlObject, 
                                 best.tuning = FALSE, 
                                 verbose = FALSE, 
                                 removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = TRUE, 
                                 xgb.metric.fun = RMSLE.xgb, 
                                 xgb.maximize =FALSE, 
                                 xgb.metric.label = 'rmsle', 
                                 xgb.foldList = NULL,
                                 xgb.eta = NULL,
                                 ... ) {
  
  model = NULL 
  pred = NULL 
  secs = NULL
  
  checkModelName(model.label)
  
  ### trainAndPredictInternal
  trainAndPredictInternal = function(model.label,Ytrain,Xtrain,Xtest,controlObject,removePredictorsMakingIllConditionedSquareMatrix_forLinearModels) {
    fs = removePredictorsMakingIllConditionedSquareMatrix_IFFragileLinearModel(Xtrain=Xtrain, 
                                                                               Xtest=Xtest, 
                                                                               model.label=model.label,
                                                                               removePredictorsMakingIllConditionedSquareMatrix_forLinearModels=removePredictorsMakingIllConditionedSquareMatrix_forLinearModels)
    Xtrain = fs$Xtrain
    Xtest = fs$Xtest
    
    if (model.label == "lm") {   ### LinearReg
      model <- caret::train(y = Ytrain, x = Xtrain , method = "lm", trControl = controlObject)
      pred = as.numeric( predict(model , Xtest )  ) 
    } else if (model.label == "bayesglm") {   ### bayesglm
      model <- caret::train(y = Ytrain, x = Xtrain , method = "bayesglm", trControl = controlObject)
      pred = as.numeric( predict(model , Xtest )  ) 
    } else if (model.label == "glm") {   ### glm
      model <- caret::train(y = Ytrain, x = Xtrain , method = "glm", trControl = controlObject)
      pred = as.numeric( predict(model , Xtest )  ) 
    } else if (model.label == "glmStepAIC") {   ### glmStepAIC
      model <- caret::train(y = Ytrain, x = Xtrain , method = "glmStepAIC", trControl = controlObject)
      pred = as.numeric( predict(model , Xtest )  ) 
    } else if (model.label == "rlm") {   ### RobustLinearReg
      model <- caret::train(y = Ytrain, x = Xtrain , method = "rlm", preProcess="pca", trControl = controlObject, ... )
      pred = as.numeric( predict(model , Xtest )  ) 
    } else if (model.label == "knn") {  ### KNN_Reg
      model <- caret::train(y = Ytrain, x = Xtrain , method = "knn", 
                     preProc = c("center", "scale"), 
                     tuneGrid = data.frame(.k = 1:10),
                     trControl = controlObject, ...)
      pred = as.numeric( predict(model , Xtest )  )
    } else if (model.label == "pls") {  ### PLS_Reg
      .tuneGrid = expand.grid(.ncomp = 1:10)
      model <- caret::train(y = Ytrain, x = Xtrain ,
                     method = "pls",
                     tuneGrid = .tuneGrid , 
                     trControl = controlObject, ...)
      pred = as.numeric( predict(model , Xtest )  )
    } else if (model.label == "ridge") {  ### Ridge_Reg
      ridgeGrid <- data.frame(.lambda = seq(0, .1, length = 15))
      if (best.tuning) ridgeGrid <- data.frame(.lambda = seq(0, .1, length = 25))
      model <- caret::train(y = Ytrain, x = Xtrain ,
                     method = "ridge",
                     tuneGrid = ridgeGrid,
                     trControl = controlObject, ...)
      pred = as.numeric( predict(model , Xtest )  )
    } else if (model.label == "enet") {  ### Enet_Reg
      enetGrid <- expand.grid(.lambda = c(0, 0.01, .1), .fraction = seq(.05, 1, length = 20))
      if (best.tuning) enetGrid <- expand.grid(.lambda = c(0, 0.01,.1,.5,.8), .fraction = seq(.05, 1, length = 30))
      model <- caret::train(y = Ytrain, x = Xtrain ,
                     method = "enet",
                     tuneGrid = enetGrid, 
                     trControl = controlObject, ...)
      pred = as.numeric( predict(model , Xtest )  )
    } else if (model.label == "svmRadial") {  ### SVM_Reg
      model <- caret::train(y = Ytrain, x = Xtrain ,
                     method = "svmRadial",
                     tuneLength = 15,
                     trControl = controlObject,...)
      pred = as.numeric( predict(model , Xtest )  )
    } else if (model.label == "treebag") {  ### BaggedTree_Reg
      model <- caret::train(y = Ytrain, x = Xtrain ,
                     method = "treebag",
                     trControl = controlObject,...)
      pred = as.numeric( predict(model , Xtest )  )
    } else if (model.label == "gbm") {  ### GBM
      gbmGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                             n.trees = seq(100, 1000, by = 50),
                             shrinkage = c(0.01, 0.1), 
                             n.minobsinnode = 10)
      model <- caret::train(y = Ytrain, x = Xtrain ,
                     method = "gbm",
                     tuneGrid = gbmGrid,
                     bag.fraction = 0.5 , 
                     verbose = FALSE,
                     trControl = controlObject, ... )
      pred = as.numeric( predict(model , Xtest )  )
    } else if (model.label == "rf") {  ### RandomForest_Reg
      .ntrees = 150
      if (best.tuning) .ntrees = 1000
      model <- caret::train(y = Ytrain, x = Xtrain ,
                     method = "rf",
                     tuneLength = 10,
                     ntrees = .ntrees,
                     importance = TRUE,
                     trControl = controlObject, ...)
      pred = as.numeric( predict(model , Xtest )  )
    } else if (model.label == "cubist") {  ### Cubist_Reg
      cubistGrid <- expand.grid(.committees = c(1, 5, 10, 50, 75, 90, 100),
                                .neighbors = c(0, 1, 3, 5, 7, 9))
      model <- caret::train(y = Ytrain, x = Xtrain ,
                     method = "cubist",
                     tuneGrid = cubistGrid,
                     trControl = controlObject, ...)
      pred = as.numeric( predict(model , Xtest )  )
    } else if (model.label == "avNNet") {  ### Neural Networks 
      nnetGrid <- expand.grid(.decay = c(0.001, .01, .1),
                              .size = seq(1, 27, by = 2),
                              .bag = FALSE)
      model <- caret::train(y = Ytrain, x = Xtrain ,
                     method = "avNNet",
                     tuneGrid = nnetGrid,
                     linout = TRUE,
                     trace = FALSE,
                     maxit = 1000,
                     trControl = controlObject,...)
      pred = as.numeric( predict(model , Xtest )  )  
    } else if (model.label == "xgbTreeGTJ") {  ### xgbTreeGTJ 
      param <- list("objective" = "reg:linear" ,
                    "min_child_weight" = 6 , 
                    "subsample" = 0.7 , 
                    "colsample_bytree" = 0.6 , 
                    "scale_pos_weight" = 0.8 , 
                    "silent" = 1 , 
                    "max_depth" = 8 , 
                    "max_delta_step" = 2 )
      
      param['eta'] = 0.02
      if (! is.null(xgb.eta)) param['eta'] = xgb.eta
      
      ## fix nrounds? 
      fix.nround = FALSE
      nrounds = 3000 
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
                                   xgb.metric.label = xgb.metric.label, 
                                   param = param,
                                   cv.nround = nrounds , 
                                   fix.nround = fix.nround, 
                                   nfold = min(5,nrow(Xtrain)) , 
                                   verbose=verbose)
      pred = xgb$pred
      early.stop = xgb$early.stop
      
      model = list(
        results = list( RMSLE=xgb$perf.cv  ),
        bestTune = data.frame(early.stop = xgb$early.stop)
      )
      
    } else if (model.label == "xgbTree") {  ### XGBoost 
      if (best.tuning) {
        param <- list("objective" = "reg:linear",
                      "gamma" = 0.7,  
                      "max_depth" = 20, 
                      "subsample" = 0.5 , ## suggested in ESLII
                      "nthread" = 10, 
                      "min_child_weight" = 1 , 
                      "colsample_bytree" = 0.5, 
                      "max_delta_step" = 1)
        param['eta'] = 0.05
        if (! is.null(xgb.eta)) param['eta'] = xgb.eta
        
        ## fix nrounds? 
        fix.nround = FALSE
        nrounds = 3000 
        fPar = list(...)
        if (length(fPar)>0) {
          if ('nrounds' %in% names(fPar)) {
            fix.nround = TRUE 
            nrounds = as.integer(fPar['nrounds'])
          }
        }
        
        if (verbose) cat('>> xgb fix.nround:',fix.nround,' - nrounds:',nrounds,'\n')
        if (verbose) cat(">> xgb params:\n")
        if (verbose) print(param)
        
        xgb = xgb_train_and_predict (train_set = Xtrain,
                                     y = Ytrain,  
                                     test_set = Xtest, 
                                     foldList = xgb.foldList, 
                                     xgb.metric.fun = xgb.metric.fun,
                                     xgb.maximize = xgb.maximize, 
                                     xgb.metric.label = xgb.metric.label, 
                                     param = param,
                                     cv.nround = nrounds , 
                                     fix.nround = fix.nround, 
                                     nfold = min(5,nrow(Xtrain)) , 
                                     verbose=verbose)
        pred = xgb$pred
        early.stop = xgb$early.stop
        
        model = list(
          results = list( RMSLE=xgb$perf.cv  ),
          bestTune = data.frame(early.stop = xgb$early.stop)
        )
        
      } else {
        model <- caret::train(y = Ytrain, x = Xtrain ,
                       method = "xgbTree",
                       trControl = controlObject, ... ) 
        pred = as.numeric( predict(model , Xtest )  )
      }
    } else {
      stop("unrecognized model.label!")
    }
    return(list(model=model,pred=pred))
  }
  
  ##
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
  }
  
  ##
  if (verbose) cat(">> ",model.label,": time elapsed:",secs," secs. [min:",secs/60,"] [hours:",secs/(60*60),"]\n")
  if (verbose) {
    print(model)
  }
  
  ##
  return(list(pred = pred, model = model, secs = secs))
}

#' Plot predicted values vs. observed / residual values.  
#' 
#' @param observed the observed output variables (numeric vector). 
#' @param predicted the predicted values (numeric vector). 
#' @param main a string as a title for the plot  
#' 
#' @examples
#' obs = 1:10 
#' preds = obs + runif(length(obs)) 
#' ff.plotPerformance.reg(observed = obs , predicted = preds, main="Predicted vs. observed/residual")
#' @export

ff.plotPerformance.reg <- function(observed,predicted,main=NULL) {
  par(mfrow=c(1,2))
  
  residualValues <- observed - predicted
  
  # Observed values versus predicted values
  # It is a good idea to plot the values on a common scale.
  axisRange <- extendrange(c(observed, predicted))
  plot(observed, predicted,
       ylim = axisRange,
       xlim = axisRange)
  # Add a 45 degree reference line
  abline(0, 1, col = "darkgrey", lty = 2)
  
  # Predicted values versus residuals
  plot(predicted, residualValues, ylab = "residual")
  abline(h = 0, col = "darkgrey", lty = 2)
  
  if (! is.null(main)) {
    mtext(text = main,side = 3, line = -2, outer = TRUE , cex = 1.5 , font = 2 )  
  }
  
  par(mfrow=c(1,1))
}

#' Given a tuned regression model, finds more performant tuning configurations using Nelder/Mead, quasi-Newton and conjugate-gradient algorithms. 
#' 
#' @param y the output variable as numeric vector
#' @param Xtrain the encoded \code{data.frame} of train data. Must be a \code{data.frame} of \code{numeric}
#' @param caretModelName a string specifying which model to use. Possible values are \code{'lm'}, \code{'bayesglm'}, 
#' \code{'glm'}, \code{'glmStepAIC'}, \code{'rlm'}, \code{'knn'}, \code{'pls'}, \code{'ridge'}, \code{'enet'}, 
#' \code{'svmRadial'}, \code{'treebag'}, \code{'gbm'}, \code{'rf'}, \code{'cubist'}, \code{'avNNet'}, 
#' \code{'xgbTreeGTJ'}, \code{'xgbTree'}
#' @param controlObject a list of values that define how this function acts. Must be a caret \code{trainControl} object 
#' @param verbose \code{TRUE} to enable verbose mode. 
#' @param parallelize \code{TRUE} to enable parallelization (require \code{parallel}). 
#' @param bestTune a \code{data.frame} with best tuned parameters of specified model. 
#' @param max_secs the max number of seconds as time constraint 
#' @param method the method to use. Possible values are \code{c('Nelder-Mead', 'BFGS', 'CG', 'L-BFGS-B', 'SANN')}. 
#' @param useInteger \code{TRUE} if the tuning grid is composed of integers and not of continuous numbers.
#' @param seed a user specified seed. Useful for replicable execution (e.g. passing the same seed to the \code{\link{ff.verifyBlender}} function) 
#' if the control object involves random steps for creating resamples.    
#' 
#' @examples
#' ## suppress warnings raised because there few obs 
#' warn_def = getOption('warn')
#' options(warn=-1)
#'
#' ## data 
#' Xtrain <- data.frame( a = rep(1:5 , each = 2), b = 10:1, 
#' c = rep(as.Date(c("2007-06-22", "2004-02-13")),5) )
#' Xtest <- data.frame( a = rep(2:6 , each = 2), b = 1:10, 
#' c = rep(as.Date(c("2007-03-01", "2004-05-23")),5) )
#' Ytrain = 1:10 + runif(nrow(Xtrain))
#'
#' ## encode datasets 
#' l = ff.makeFeatureSet(Xtrain,Xtest,c("C","N","D"))
#' Xtrain = l$traindata
#' Xtest = l$testdata
#'
#' ## make a caret control object 
#' controlObject <- trainControl(method = "repeatedcv", 
#' repeats = 1, number = 2)
#'
#' ## train and predict 
#' tp = ff.trainAndPredict.reg(Ytrain=Ytrain ,
#'                           Xtrain=Xtrain , 
#'                           Xtest=Xtest , 
#'                           model.label = "cubist" , 
#'                           controlObject=controlObject)
#'
#' pred_test = tp$pred
#' model = tp$model
#' secs = tp$secs
#'
#' ## blender 
#' gBlender = ff.blend(bestTune = tp$model$bestTune, 
#'                                 caretModelName = "cubist" , 
#'                                 Xtrain = Xtrain , 
#'                                 y = Ytrain, controlObject = tp$model$control, 
#'                                 max_secs = 3, 
#'                                 seed = 123,
#'                                 method = c("Nelder-Mead", "BFGS", "CG", "L-BFGS-B", "SANN"),
#'                                 useInteger = TRUE, 
#'                                 parallelize = TRUE, 
#'                                 verbose = FALSE)
#' ff.summaryBlender(gBlender)
#' ff.getBestBlenderPerformance(gBlender)
#' bestTune = ff.getBestBlenderTune(gBlender)
#' ff.verifyBlender (gBlender,Xtrain=Xtrain,y=Ytrain,seed=123,
#' controlObject=tp$model$control,caretModelname = "cubist")
#'
#' ## restore warnings 
#' options(warn=warn_def)
#' @export
#' @references \url{https://stat.ethz.ch/pipermail/r-devel/2010-August/058081.html}
#' @return a list of lists (one for each specified optimization method) with components \code{par} (best set of parameters found), 
#' \code{value} (the value of fn corresponding to par), \code{counts} (a two-element integer vector giving the number of calls to fn and gr respectively; 
#' this excludes those calls needed to compute the Hessian, if requested, and any calls to fn to compute a finite-difference approximation to the gradient), 
#' \code{convergence} (an integer code. 0 indicates successful completion which is always the case for SANN and Brent), \code{message} 
#' (a character string giving any additional information returned by the optimizer, or NULL), \code{seed} (the used seed). 
#' For further details see \code{\link[stats]{optim}}.

ff.blend = function(bestTune,
                                 caretModelName, 
                                 Xtrain,y,
                                 controlObject,
                                 max_secs=10*60,
                                 seed = NULL,
                                 method = c("Nelder-Mead", "BFGS", "CG", "L-BFGS-B", "SANN"),
                                 useInteger = TRUE, 
                                 parallelize = TRUE, 
                                 verbose=TRUE) {
  checkModelName(caretModelName)
  
  stopifnot( is.data.frame(bestTune) && ncol(bestTune) > 0)
  
  seed = ifelse(is.null(seed),sample(1:100000, 1),seed)
  
  ## compute init  
  ptm = proc.time()
  cubistGrid <- expand.grid(bestTune)
  set.seed(seed)
  model <- caret::train(y = y, x = Xtrain ,
                 method = caretModelName,
                 tuneGrid = cubistGrid,
                 trControl = controlObject)
  tm = proc.time() - ptm
  secs1 = as.numeric(tm[3])
  max_iter = floor(max_secs / ( 2*secs1) ) # https://stat.ethz.ch/pipermail/r-devel/2010-August/058081.html
  if (verbose) cat(caretModelName,':: 1 iteration[rmse:',model$results$RMSE,']:',secs1,'secs --> max iter ==',max_iter, '\n') 
  
  ## inner function to optimize 
  runBlender = function(x,Xtrain,y,controlObject,meth,useInteger) { 
    
    ##
    getScore = function(x,Xtrain,caretModelName,controlObject,useInteger,seed) {
      if (useInteger) x = floor(x)
      .grid <- as.data.frame(t(x))
      set.seed(seed)
      model <- caret::train(y = y, x = Xtrain ,
                     method = caretModelName,
                     tuneGrid = .grid,
                     trControl = controlObject)
      model$results$RMSE
    }
    
    ##
    if (verbose)  cat(meth,':: >>>> trying:',x,'\n')
    
    ## 
    rmse = plyr::failwith( 100000, getScore , quiet = !verbose) (x=x,Xtrain=Xtrain,
                                                                 caretModelName=caretModelName,
                                                                 controlObject=controlObject,
                                                                 useInteger=useInteger,
                                                                 seed=seed)
  }
  
  ## optimize wrapper 
  doOptim = function(m,max_iter,useInteger) {
    res = optim(par = setNames(object = as.numeric(bestTune) , nm = names(bestTune)), 
                fn = runBlender,
                method = method[m], 
                control=list(maxit=max_iter),
                Xtrain = Xtrain,
                y = y,
                controlObject=controlObject,
                meth=method[m],
                useInteger=useInteger)
    if (useInteger) res[['par']] = floor(res[['par']])
    res['method'] = method[m]
    res['seed'] = seed
    res['parnames'] = as.data.frame(as.character(names(bestTune)))
    return(res)
  }
  
  if (parallelize) { 
    parallel::mclapply( seq_along(method) , doOptim , max_iter = max_iter , useInteger = useInteger, 
                        mc.cores = min(ff.getMaxCuncurrentThreads(),length(method))  )
  } else {
    lapply( seq_along(method) , doOptim, max_iter = max_iter, useInteger = useInteger)
  }
}

#' Helper function that given a blender object returns a \code{numeric} vector of performances (one for each optimization method). 
#' 
#' @param blender a blender object  
#' 
#' @export
#' @return a \code{numeric} vector of performances (one for each optimization method)

ff.summaryBlender = function(blender) {
  stopifnot( is.list(blender) )
  return(setNames( object = unlist(lapply(blender , function(x) x$value)) , 
                   nm = (unlist(lapply(blender , function(x) x$method))) ))
}

#' Helper function that given a blender object returns the best optimization method. 
#' 
#' @param blender a blender object  
#' 
#' @export
#' @return a \code{numeric} of best score and as object name the best performant method name. 

ff.getBestBlenderPerformance = function(blender) {
  stopifnot( is.list(blender) )
  perf = ff.summaryBlender(blender)
  return(perf[which(perf==min(perf))])
}

#' Helper function that given a blender object returns the best tuning parameters found by the blender. 
#' 
#' @param blender a blender object  
#' @param truncate \code{TRUE} to cut at the first tuning best configuration in case there are more than one optimal tuning configurations.  
#' 
#' @export
#' @return a \code{data.frame} of the best tuning parameters. 

ff.getBestBlenderTune = function(blender,truncate=TRUE) {
  stopifnot( is.list(blender) )
  perf = ff.summaryBlender(blender)
  idx = which(perf==min(perf))
  if (truncate) {
    if (length(idx)>1) idx = idx[1]
    return(blender[[idx]]$par)
  } else {
    return(lapply(blender[idx],function(xx) xx$par))
  }
}

#' Helper function that given a blender object replicates the execution in order to verify performances. 
#' 
#' @param blender a blender object  
#' @param Xtrain the train set  
#' @param y the output variable as numeric vector 
#' @param seed the seed used by the blender, if applicable. If the blender used one, it is necessary for replicating blender performances. 
#' @param caretModelname a string specifying which model to use. Possible values are \code{'lm'}, \code{'bayesglm'}, 
#' \code{'glm'}, \code{'glmStepAIC'}, \code{'rlm'}, \code{'knn'}, \code{'pls'}, \code{'ridge'}, \code{'enet'}, 
#' \code{'svmRadial'}, \code{'treebag'}, \code{'gbm'}, \code{'rf'}, \code{'cubist'}, \code{'avNNet'}, 
#' \code{'xgbTreeGTJ'}, \code{'xgbTree'}. It must be the same model name used by the blender. 
#' @param controlObject a list of values that define how this function acts. Must be a caret \code{trainControl} object. It must be the same used by the blender.
#' @export
#' @return a \code{numeric} as difference in performance between blender and replicated execution.  

ff.verifyBlender = function(blender,Xtrain,y,seed=NULL,controlObject, caretModelname) {
  stopifnot( is.list(blender) )
  seed = ifelse(is.null(seed),sample(1:100000, 1),seed)
  bestTune = ff.getBestBlenderTune(blender)
  cubistGrid <- as.data.frame( t(bestTune) )
  set.seed(seed)
  model <- caret::train(y = y, x = Xtrain ,
                 method = caretModelname,
                 tuneGrid = cubistGrid,
                 trControl = controlObject)
  bperf = ff.getBestBlenderPerformance(blender)
  if (length(bperf)>1) bperf = bperf[1]
  return(bperf - model$results$RMSE)
}


#' Create an ensemble of a tuned regression model.  
#' 
#' @param y the output variable as numeric vector
#' @param Xtrain the encoded \code{data.frame} of train data. Must be a \code{data.frame} of \code{numeric}
#' @param Xtest the encoded \code{data.frame} of test data. Must be a \code{data.frame} of \code{numeric}
#' @param caretModelName a string specifying which model to use. Possible values are \code{'lm'}, \code{'bayesglm'}, 
#' \code{'glm'}, \code{'glmStepAIC'}, \code{'rlm'}, \code{'knn'}, \code{'pls'}, \code{'ridge'}, \code{'enet'}, 
#' \code{'svmRadial'}, \code{'treebag'}, \code{'gbm'}, \code{'rf'}, \code{'cubist'}, \code{'avNNet'}, 
#' \code{'xgbTreeGTJ'}, \code{'xgbTree'}
#' @param controlObject a list of values that define how this function acts. Must be a caret \code{trainControl} object 
#' @param verbose \code{TRUE} to enable verbose mode. 
#' @param parallelize \code{TRUE} to enable parallelization (require \code{parallel}). 
#' @param removePredictorsMakingIllConditionedSquareMatrix_forLinearModels \code{TRUE} for removing predictors making 
#' ill-conditioned square matrices in case of fragile linear models, i.e. \code{c('rlm','pls','ridge','enet')}.
#' @param bestTune a \code{data.frame} with best tuned parameters of specified model. 
#' @param ... arguments passed to the regression routine.  
#' 
#' @examples
#'
#' ## suppress warnings raised because there few obs 
#' warn_def = getOption('warn')
#' options(warn=-1)
#'
#' ## data 
#' Xtrain <- data.frame( a = rep(1:10 , each = 2), b = 20:1, 
#' c = rep(as.Date(c("2007-06-22", "2004-02-13")),10) )
#' Xtest <- data.frame( a = rep(2:11 , each = 2), b = 1:20, 
#' c = rep(as.Date(c("2007-03-01", "2004-05-23")),10) )
#' Ytrain = 1:20 + runif(nrow(Xtrain))
#' 
#' ## encode datasets 
#' l = ff.makeFeatureSet(Xtrain,Xtest,c("C","N","D"))
#' Xtrain = l$traindata
#' Xtest = l$testdata
#'
#' ## make a caret control object 
#' controlObject <- trainControl(method = "repeatedcv", repeats = 1, number = 2)
#'
#' tp = ff.trainAndPredict.reg(Ytrain=Ytrain ,
#'                           Xtrain=Xtrain , 
#'                           Xtest=Xtest , 
#'                           model.label = "cubist" , 
#'                           controlObject=controlObject)
#'
#' pred_test = tp$pred
#' model = tp$model
#' secs = tp$secs
#'
#' ## create ensemble 
#' en = ff.createEnsemble(Xtrain = Xtrain, 
#'                       Xtest = Xtest, 
#'                       y = Ytrain, 
#'                       bestTune = tp$model$bestTune , 
#'                       caretModelName = "cubist" , 
#'                       parallelize = TRUE, 
#'                       removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = TRUE, 
#'                       controlObject = tp$model$control)
#' predTrain = en$predTrain
#' predTest = en$predTest
#'
#' ## restore warnings 
#' options(warn=warn_def)
#'
#' @export
#' @return a list of train and test predictions.  
#' 

ff.createEnsemble = function(Xtrain,
                             Xtest,
                             y,
                             caretModelName, 
                             bestTune,
                             removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = TRUE, 
                             controlObject, 
                             parallelize = TRUE,
                             verbose=TRUE , 
                             ... ) {
  checkModelName(caretModelName)
  stopifnot( ! is.null(controlObject) )
  stopifnot( is.atomic(y) && length(y) == nrow(Xtrain) )
  stopifnot( (is.null(bestTune)) || (is.data.frame(bestTune) && ncol(bestTune) > 0) )
  
  ##
  predTrain = rep(NA,nrow(Xtrain))
  predTest = rep(NA,nrow(Xtest))
  index = controlObject$index
  indexOut = controlObject$indexOut
  
  ## train
  doFold = function(i,
                    index,
                    indexOut,
                    Xtrain,
                    y,
                    caretModelName , 
                    bestTune, 
                    removePredictorsMakingIllConditionedSquareMatrix_forLinearModels, 
                    verbose, 
                    ... ) {
    train_i = Xtrain[ index[[i]] , ]
    y_i = y[ index[[i]] ]
    test_i = Xtrain[ indexOut[[i]] , ]
    
    ##
    fs = removePredictorsMakingIllConditionedSquareMatrix_IFFragileLinearModel(Xtrain=train_i, 
                                                                               Xtest=test_i, 
                                                                               model.label=caretModelName,
                                                                               removePredictorsMakingIllConditionedSquareMatrix_forLinearModels=removePredictorsMakingIllConditionedSquareMatrix_forLinearModels)
    train_i = fs$Xtrain
    test_i = fs$Xtest
    
    model = NULL
    internalControlObject = caret::trainControl(method = "none", summaryFunction = controlObject$summaryFunction )
    
    if (! is.null(bestTune) ) {
      model <- caret::train(y = y_i, x = train_i ,
                     method = caretModelName,
                     tuneGrid = bestTune,
                     trControl = internalControlObject , ...)
    } else if (identical(caretModelName,"rlm")) {
      model <- caret::train(y = y_i, x = train_i ,
                     method = caretModelName, 
                     preProcess="pca" , 
                     trControl = internalControlObject , ...)
    } else {
      model <- caret::train(y = y_i, x = train_i ,
                     method = caretModelName,
                     trControl = internalControlObject , ...)
    }
    
    return(pred=predict(model,test_i))
  }
  
  train_list = NULL
  if (parallelize) { 
    train_list = parallel::mclapply( seq_along(index) , 
                                     doFold , 
                                     index = index,
                                     indexOut = indexOut, 
                                     Xtrain = Xtrain, 
                                     y = y, 
                                     caretModelName = caretModelName, 
                                     bestTune = bestTune,
                                     removePredictorsMakingIllConditionedSquareMatrix_forLinearModels=removePredictorsMakingIllConditionedSquareMatrix_forLinearModels,
                                     verbose = verbose , 
                                     ... ,  
                                     mc.cores = min(length(index),ff.getMaxCuncurrentThreads())  )
  } else {
    train_list = lapply( seq_along(index) , 
                         doFold , 
                         index = index,
                         indexOut = indexOut, 
                         Xtrain = Xtrain, 
                         y = y,
                         caretModelName = caretModelName, 
                         bestTune = bestTune, 
                         removePredictorsMakingIllConditionedSquareMatrix_forLinearModels=removePredictorsMakingIllConditionedSquareMatrix_forLinearModels,
                         verbose = verbose,
                         ... )
  }
  
  ## predTrain 
  lapply(seq_along(train_list) , function(i) {
    predTrain[indexOut[[i]]] <<- train_list[[i]]
  })
  stopifnot( sum(is.na(predTrain))==0 )
  
  ## predTest
  fs = removePredictorsMakingIllConditionedSquareMatrix_IFFragileLinearModel(Xtrain=Xtrain, 
                                                                             Xtest=Xtest, 
                                                                             model.label=caretModelName,
                                                                             removePredictorsMakingIllConditionedSquareMatrix_forLinearModels=removePredictorsMakingIllConditionedSquareMatrix_forLinearModels)
  Xtrain = fs$Xtrain
  Xtest = fs$Xtest
  
  internalControlObject = caret::trainControl(method = "none", summaryFunction = controlObject$summaryFunction )
  model = NULL
  
  if (! is.null(bestTune) ) { 
    model <- caret::train(y = y, x = Xtrain ,
                   method = caretModelName,
                   tuneGrid = bestTune,
                   trControl = internalControlObject , ...)
  } else if (identical(caretModelName,"rlm")) {
    model <- caret::train(y = y, x = Xtrain ,
                   method = caretModelName, 
                   preProcess="pca" , 
                   trControl = internalControlObject , ...)
  } else {
    model <- caret::train(y = y, x = Xtrain ,
                   method = caretModelName, 
                   trControl = internalControlObject , ...)
  }
  
  ##
  predTest = predict(model,Xtest)
  stopifnot( sum(is.na(predTest))==0 )
  
  ##
  return(list(predTrain = predTrain , predTest = predTest))
}

