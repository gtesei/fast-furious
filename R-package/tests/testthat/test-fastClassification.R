test_that('base test case', {
  #skip_on_cran()
  
  ## suppress warnings raised because there few obs 
  warn_def = getOption('warn')
  options(warn=-1)
  
  ## data 
  Xtrain <- data.frame( a = rep(1:10 , each = 2), b = 20:1, c = rep(as.Date(c("2007-06-22", "2004-02-13")),10) , d = 20:1)
  Xtest <- data.frame( a = rep(2:11 , each = 2), b = 1:20, c = rep(as.Date(c("2007-03-01", "2004-05-23")),10) , d = 1:20)
  Ytrain = c(rep(1,10),rep(0,10))
  
  ## encode data sets 
  l = ff.makeFeatureSet(Xtrain,Xtest,c("C","N","D","N"))
  Xtrain = l$traindata
  Xtest = l$testdata
  
  ## make a caret control object 
  controlObject <- trainControl(method = "repeatedcv", repeats = 2, number = 3 , summaryFunction = twoClassSummary , classProbs = TRUE)
  tp = ff.trainAndPredict.class(Ytrain=Ytrain ,
                                Xtrain=Xtrain , 
                                Xtest=Xtest, 
                                model.label = "svmRadial" , 
                                controlObject=controlObject, 
                                verbose=T , 
                                best.tuning=T)
  
  pred_test = tp$pred
  model = tp$model
  elapsed.secs = tp$secs
  
  bestTune = l$model$bestTune
  best_ROC = max(tp$model$results$ROC)
  
  expect_equal(length(pred_test),nrow(Xtest))
  expect_equal(elapsed.secs>0,TRUE)
  
  ## make a caret control object 
  controlObject <- trainControl(method = "repeatedcv", repeats = 2, number = 3 , summaryFunction = twoClassSummary , classProbs = TRUE)
  tp = ff.trainAndPredict.class(Ytrain=Ytrain ,
                                Xtrain=Xtrain , 
                                Xtest=Xtest, 
                                model.label = "libsvm" , 
                                controlObject=controlObject, 
                                verbose=T , 
                                best.tuning=T)
  
  pred_test = tp$pred
  model = tp$model
  elapsed.secs = tp$secs
  
  bestTune = l$model$bestTune
  best_ROC = max(tp$model$results$ROC)
  
  expect_equal(length(pred_test),nrow(Xtest))
  expect_equal(elapsed.secs>0,TRUE)
  
  ## make a caret control object 
  controlObject <- trainControl(method = "repeatedcv", repeats = 2, number = 3 , summaryFunction = twoClassSummary , classProbs = TRUE)
  tp = ff.trainAndPredict.class(Ytrain=Ytrain ,
                                Xtrain=Xtrain , 
                                Xtest=Xtest, 
                                model.label = "glmnet_alpha_0.5" , 
                                controlObject=controlObject, 
                                verbose=T , 
                                best.tuning=T)
  
  pred_test = tp$pred
  model = tp$model
  elapsed.secs = tp$secs
  
  bestTune = l$model$bestTune
  best_ROC = max(tp$model$results$ROC)
  
  expect_equal(length(pred_test),nrow(Xtest))
  expect_equal(elapsed.secs>0,TRUE)
  
  ## restore warnings 
  options(warn=warn_def)
  
})