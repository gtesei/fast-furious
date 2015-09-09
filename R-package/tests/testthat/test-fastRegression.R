context("fastRegression")

# test_that('XGBoost', { 
#   #skip_on_cran()
#   
#   warn_def = getOption('warn')
#   options(warn=-1)
#   
#   ## data 
#   Xtrain <- data.frame( a = rep(1:5 , each = 2), b = 1:10, c = rep(as.Date(c("2007-06-22", "2004-02-13")),5) )
#   Xtest <- data.frame( a = rep(2:6 , each = 2), b= 1:10, c = rep(as.Date(c("2007-03-01", "2004-05-23")),5) )
#   Ytrain = 1:10 
#   
#   ## encode datasets 
#   l = ff.makeFeatureSet(Xtrain,Xtest,c('C','N','D'))
#   Xtrain = l$traindata
#   Xtest = l$testdata
#   
#   ## make a caret control object 
#   controlObject <- trainControl(method = "repeatedcv", repeats = 1, number = 2)
#   
#   ## xgbTreeGTJ best tuning 
#   tp = NULL
#   set.seed(123)
#   tp = ff.trainAndPredict.reg(Ytrain=Ytrain ,
#                              Xtrain=Xtrain , 
#                              Xtest=Xtest , 
#                              model.label = 'xgbTreeGTJ' , 
#                              controlObject=NULL, 
#                              best.tuning = T, 
#                              verbose=T, 
#                              xgb.eta = 0.5)  
#   
#   
#   
#   pred_test = tp$pred
#   model = tp$model
#   secs = tp$secs
#   
#   cat(">>>> length(pred_test): ",length(pred_test),"\n")
#   cat(">>>> nrow(Xtest): ",nrow(Xtest),"\n")
#   
#   expect_equal(is.null(tp),FALSE)
#   expect_equal(length(pred_test),nrow(Xtest))
#   expect_equal(secs>0,T)
#   
#   ## xgbTreeGTJ variant 
#   set.seed(123)
#   tp = ff.trainAndPredict.reg(Ytrain=Ytrain ,
#                              Xtrain=Xtrain , 
#                              Xtest=Xtest , 
#                              model.label = 'xgbTreeGTJ' , 
#                              controlObject=NULL, 
#                              best.tuning = F, 
#                              removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = F, 
#                              xgb.metric.fun = RMSE.xgb, 
#                              xgb.maximize =FALSE, 
#                              xgb.metric.label = 'rmse', 
#                              xgb.foldList = NULL,
#                              xgb.eta = 0.5, 
#                              verbose=T)  
#   
#   
#   
#   pred_test = tp$pred
#   model = tp$model
#   secs = tp$secs
#   
#   expect_equal(length(pred_test),nrow(Xtest))
#   expect_equal(secs>0,T)
#   
#   ## xgbTree variant 
#   set.seed(123)
#   tp = ff.trainAndPredict.reg(Ytrain=Ytrain ,
#                              Xtrain=Xtrain , 
#                              Xtest=Xtest , 
#                              model.label = 'xgbTree' , 
#                              controlObject=controlObject, 
#                              best.tuning = F, 
#                              removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = F, 
#                              xgb.metric.fun = RMSE.xgb, 
#                              xgb.maximize =FALSE, 
#                              xgb.metric.label = 'rmse', 
#                              xgb.foldList = NULL,
#                              xgb.eta = 0.5)  
#   
#   
#   
#   pred_test = tp$pred
#   model = tp$model
#   secs = tp$secs
#   
#   expect_equal(length(pred_test),nrow(Xtest))
#   expect_equal(secs>0,T)
#   
#   ## xgbTree variant 
#   set.seed(123)
#   tp = ff.trainAndPredict.reg(Ytrain=Ytrain ,
#                              Xtrain=Xtrain , 
#                              Xtest=Xtest , 
#                              model.label = 'xgbTree' , 
#                              controlObject=NULL, 
#                              best.tuning = TRUE, 
#                              removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = F, 
#                              xgb.metric.fun = RMSE.xgb, 
#                              xgb.maximize =FALSE, 
#                              xgb.metric.label = 'rmse', 
#                              xgb.foldList = NULL,
#                              xgb.eta = 0.5, 
#                              verbose=T)  
#   
#   
#   
#   pred_test = tp$pred
#   model = tp$model
#   secs = tp$secs
#   
#   cat(">>>> length(pred_test): ",length(pred_test),"\n")
#   cat(">>>> nrow(Xtest): ",nrow(Xtest),"\n")
#   
#   expect_equal(length(pred_test),nrow(Xtest))
#   expect_equal(secs>0,T)
#   
#   ## restore warnings 
#   options(warn=warn_def)
#   
# })


test_that('best tuning TRUE', {
  #skip_on_cran()
  warn_def = getOption('warn')
  options(warn=-1)
  
  ## data 
  Xtrain <- data.frame( a = rep(1:5 , each = 2), b = 1:10, c = rep(as.Date(c("2007-06-22", "2004-02-13")),5) )
  Xtest <- data.frame( a = rep(2:6 , each = 2), b= 1:10, c = rep(as.Date(c("2007-03-01", "2004-05-23")),5) )
  Ytrain = 1:10 + runif(nrow(Xtrain))
  
  ## encode datasets 
  l = ff.makeFeatureSet(Xtrain,Xtest,c('C','N','D'))
  Xtrain = l$traindata
  Xtest = l$testdata
  
  models = c('bayesglm','glm','treebag','rf')
  
  ## make a caret control object 
  controlObject <- trainControl(method = "repeatedcv", repeats = 1, number = 2)
  
  lapply(models , function(m) {
    #for (m in models) {
    cat(">>> model:",m,"\n")
    
    
    tp = ff.trainAndPredict.reg(Ytrain=Ytrain ,
                               Xtrain=Xtrain , 
                               Xtest=Xtest , 
                               model.label = m , 
                               controlObject=controlObject, 
                               best.tuning = T)  
    
    
    
    pred_test = tp$pred
    model = tp$model
    secs = tp$secs
    
    expect_equal(length(pred_test),nrow(Xtest))
    expect_equal(secs>0,T)
  })
  
  ## restore warnings 
  options(warn=warn_def)
  
})

test_that('base test case', {
  #skip_on_cran()
  
  ff.setMaxCuncurrentThreads(1)
  
  ## suppress warnings raised because there few obs 
  warn_def = getOption('warn')
  options(warn=-1)
  
  ## data 
  Xtrain <- data.frame( a = rep(1:5 , each = 2), b = 1:10, c = rep(as.Date(c("2007-06-22", "2004-02-13")),5) )
  Xtest <- data.frame( a = rep(2:6 , each = 2), b= 1:10, c = rep(as.Date(c("2007-03-01", "2004-05-23")),5) )
  Ytrain = 1:10 + runif(nrow(Xtrain))
  
  ## encode datasets 
  l = ff.makeFeatureSet(Xtrain,Xtest,c('C','N','D'))
  Xtrain = l$traindata
  Xtest = l$testdata
  
  ## make a caret control object 
  controlObject <- trainControl(method = "repeatedcv", repeats = 1, number = 2)
  model.label = "knn"
  tp = ff.trainAndPredict.reg(Ytrain=Ytrain ,
                             Xtrain=Xtrain , 
                             Xtest=Xtest , 
                             model.label = model.label , 
                             controlObject=controlObject)
  
  pred_test = tp$pred
  model = tp$model
  secs = tp$secs
  
  expect_equal(length(pred_test),nrow(Xtest))
  expect_equal(secs>0,T)
  
  ## parallelize FALSE
  en = ff.createEnsemble(Xtrain = Xtrain, 
                         Xtest = Xtest, 
                         y = Ytrain, 
                         bestTune = tp$model$bestTune , 
                         caretModelName = model.label , 
                         parallelize = F, 
                         removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = T, 
                         controlObject = tp$model$control)
  predTrain = en$predTrain
  predTest = en$predTest
  
  expect_equal(length(predTrain),nrow(Xtrain))
  expect_equal(length(predTest),nrow(Xtest))
  
  ## removePredictorsMakingIllConditionedSquareMatrix_forLinearModels FALSE
  en = ff.createEnsemble(Xtrain = Xtrain, 
                         Xtest = Xtest, 
                         y = Ytrain, 
                         bestTune = tp$model$bestTune , 
                         caretModelName = model.label , 
                         parallelize = F, 
                         removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = F, 
                         controlObject = tp$model$control)
  predTrain = en$predTrain
  predTest = en$predTest
  
  expect_equal(length(predTrain),nrow(Xtrain))
  expect_equal(length(predTest),nrow(Xtest))
  
  ## verbose FALSE
  en = ff.createEnsemble(Xtrain = Xtrain, 
                         Xtest = Xtest, 
                         y = Ytrain, 
                         bestTune = tp$model$bestTune , 
                         caretModelName = model.label , 
                         parallelize = F, 
                         removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = F, 
                         controlObject = tp$model$control,
                         verbose = F)
  
  predTrain = en$predTrain
  predTest = en$predTest
  
  expect_equal(length(predTrain),nrow(Xtrain))
  expect_equal(length(predTest),nrow(Xtest))
  
  ## blender 
  cat(">>> Testing blender ... \n")
  methods = c("Nelder-Mead")
  gBlender = ff.blend(bestTune = tp$model$bestTune, 
                                   caretModelName = model.label , 
                                   Xtrain = Xtrain , 
                                   y = Ytrain, controlObject = tp$model$control, 
                                   max_secs = 1.5, 
                                   seed = 123,
                                   method = methods,
                                   useInteger = T, 
                                   parallelize = F, 
                                   verbose = F)
  
  expect_equal(length(gBlender),length(methods))
  
  sb = ff.summaryBlender(gBlender)
  expect_equal(length(sb),length(methods))
  
  bbp = ff.getBestBlenderPerformance(gBlender)
  expect_equal(is.null(bbp),F)
  
  bestTune = ff.getBestBlenderTune(gBlender)
  expect_equal(is.null(bestTune),F)
  
  bestTune = ff.getBestBlenderTune(gBlender,truncate = F)
  expect_equal(is.null(bestTune),F)
  
  diff = ff.verifyBlender (gBlender,Xtrain=Xtrain,y=Ytrain,seed=123,controlObject=tp$model$control,caretModelname = model.label)
  expect_equal(length(diff),1)
  
  ## restore warnings 
  options(warn=warn_def)
  
  ## restore default 
  ff.setMaxCuncurrentThreads(2)
  
})

