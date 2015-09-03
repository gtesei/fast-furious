context("featureFilter")

test_that('zero variance predictors are removed', {
  #skip_on_cran()
  Xtrain <- data.frame( a = rep(1:3 , each = 2), b = c(4:1,6,6), c = rep(1,6))
  l = ff.featureFilter (traindata = Xtrain,
                               testdata = NULL,
                               removeOnlyZeroVariacePredictors=TRUE)
  expect_equal(ncol(l$traindata),2)
  
  l = ff.featureFilter (traindata = NULL,
                        testdata = Xtrain,
                        removeOnlyZeroVariacePredictors=TRUE,
                        performVarianceAnalysisOnTrainSetOnly = FALSE)
  expect_equal(ncol(l$testdata),2)
  
  l = ff.featureFilter (traindata = NULL,
                        testdata = Xtrain,
                        removeOnlyZeroVariacePredictors=FALSE,
                        performVarianceAnalysisOnTrainSetOnly = FALSE)
  expect_equal(ncol(l$testdata),2)
  
  l = ff.featureFilter (traindata = Xtrain,
                        testdata = Xtrain,
                        removeOnlyZeroVariacePredictors=FALSE,
                        performVarianceAnalysisOnTrainSetOnly = FALSE)
  expect_equal(ncol(l$testdata),2)
  
  l = ff.featureFilter (traindata = Xtrain,
                        testdata = Xtrain,
                        removeOnlyZeroVariacePredictors=FALSE,
                        performVarianceAnalysisOnTrainSetOnly = FALSE, 
                        featureScaling = FALSE, 
                        verbose=FALSE)
  expect_equal(ncol(l$testdata),2)
  
})

test_that('removing high correlated predictors', {
  #skip_on_cran()
  Xtrain <- data.frame( a = rep(1:3 , each = 2), b = c(4:1,6,6))
  Xtrain = cbind(Xtrain,c = Xtrain$a)
  l = ff.featureFilter (traindata = Xtrain,
                        testdata = Xtrain,
                        removeOnlyZeroVariacePredictors=TRUE,
                        removeHighCorrelatedPredictors = TRUE, 
                        performVarianceAnalysisOnTrainSetOnly = FALSE, 
                        featureScaling = FALSE, 
                        verbose=TRUE)
  expect_equal(ncol(l$testdata),2)
})

test_that('removing predictors making ill conditioned saquare matrices', {
  #skip_on_cran()
  Xtrain <- data.frame( a = rep(1:3 , each = 2), b = c(4:1,6,6))
  Xtrain = cbind(Xtrain,c = Xtrain$a)
  l = ff.featureFilter (traindata = Xtrain,
                        testdata = Xtrain,
                        removeOnlyZeroVariacePredictors=TRUE,
                        removePredictorsMakingIllConditionedSquareMatrix = TRUE, 
                        removeHighCorrelatedPredictors = FALSE, 
                        performVarianceAnalysisOnTrainSetOnly = FALSE, 
                        featureScaling = FALSE, 
                        verbose=TRUE)
  expect_equal(ncol(l$testdata),2)
  expect_equal(ncol(l$traindata),2)
})

test_that('removing correlated predictors below threshold', {
  #skip_on_cran()
  Xtrain <- data.frame( a = rep(1:3 , each = 2), b = c(4:1,6,6) )
  set.seed(123)
  Xtrain$c = Xtrain$a + runif(nrow(Xtrain))
  l = ff.featureFilter (traindata = Xtrain,
                        testdata = Xtrain,
                        removeOnlyZeroVariacePredictors=FALSE,
                        y = Xtrain$a, 
                        correlationThreshold = 0.75 ,  
                        removePredictorsMakingIllConditionedSquareMatrix = FALSE, 
                        removeHighCorrelatedPredictors = FALSE, 
                        performVarianceAnalysisOnTrainSetOnly = FALSE, 
                        featureScaling = FALSE, 
                        verbose=FALSE)
  
  expect_equal(ncol(l$testdata),3)
  
  expect_error(ff.featureFilter (traindata = Xtrain,
                                     testdata = Xtrain,
                                     removeOnlyZeroVariacePredictors=TRUE,
                                     y = Xtrain$a, 
                                     correlationThreshold = 0.75 ,  
                                     removePredictorsMakingIllConditionedSquareMatrix = FALSE, 
                                     removeHighCorrelatedPredictors = FALSE, 
                                     performVarianceAnalysisOnTrainSetOnly = FALSE, 
                                     featureScaling = FALSE, 
                                     verbose=FALSE)) 
  
  expect_error(ff.featureFilter (traindata = Xtrain,
                                 testdata = Xtrain,
                                 removeOnlyZeroVariacePredictors=TRUE,
                                 y = NULL, 
                                 correlationThreshold = 0.75 ,  
                                 removePredictorsMakingIllConditionedSquareMatrix = FALSE, 
                                 removeHighCorrelatedPredictors = FALSE, 
                                 performVarianceAnalysisOnTrainSetOnly = FALSE, 
                                 featureScaling = FALSE, 
                                 verbose=FALSE)) 
  
})

test_that('removing identical predictors', {
  #skip_on_cran()
  
  data = data.frame( a = rep(1:3 , each = 2), b = c(4:1,6,6), c = rep(1,6) , d=rep(1:3 , each = 2))
  data.poly = ff.poly(x=data,n=3,direction = 0)
  Xtrain = data.poly[1:3,]
  Xtrain = data.poly[4:6,]
  
  l = ff.featureFilter (traindata = Xtrain,
                    testdata = Xtrain,
                    removeOnlyZeroVariacePredictors=TRUE,
                    y = NULL, 
                    correlationThreshold = NULL ,  
                    removePredictorsMakingIllConditionedSquareMatrix = FALSE, 
                    removeIdenticalPredictors = TRUE,
                    removeHighCorrelatedPredictors = FALSE, 
                    performVarianceAnalysisOnTrainSetOnly = FALSE, 
                    featureScaling = FALSE, 
                    verbose=TRUE)
  expect_equal(object = ncol(l$traindata) , 10)
  expect_equal(object = ncol(l$testdata) , 10)
  
  l = ff.featureFilter (traindata = Xtrain,
                        testdata = Xtrain,
                        removeOnlyZeroVariacePredictors=TRUE,
                        y = NULL, 
                        correlationThreshold = NULL ,  
                        removePredictorsMakingIllConditionedSquareMatrix = FALSE, 
                        removeIdenticalPredictors = TRUE,
                        removeHighCorrelatedPredictors = FALSE, 
                        performVarianceAnalysisOnTrainSetOnly = FALSE, 
                        featureScaling = FALSE, 
                        verbose=FALSE)
  expect_equal(object = ncol(l$traindata) , 10)
  expect_equal(object = ncol(l$testdata) , 10)
})

test_that('removing correlated predictors below threshold', {
  #skip_on_cran()
  Xtrain <- data.frame( a = rep(1:3 , each = 2), b = c(4:1,6,6), c = rep(1,6) )
  Xtest <-  Xtrain + runif(nrow(Xtrain))
  data = rbind(Xtrain,Xtest)
  
  data.poly = ff.poly(x=data,n=2,direction = 0)
  expect_equal(ncol(data.poly),9)
  expect_equal(nrow(data.poly),nrow(data))
  
  data.poly = ff.poly(x=data,n=2,direction = 1)
  expect_equal(ncol(data.poly),6)
  expect_equal(nrow(data.poly),nrow(data))
  
  data.poly = ff.poly(x=data,n=2,direction = -1)
  expect_equal(ncol(data.poly),6)
  expect_equal(nrow(data.poly),nrow(data))
})

test_that('correlation filter', {
  #skip_on_cran()
  Xtrain <- data.frame( a = rep(1:3 , each = 2), b = c(4:1,6,6), c = rep(1,6) )
  Xtest <-  Xtrain + runif(nrow(Xtrain))
  y = 1:6
  
  l = ff.corrFilter(Xtrain=Xtrain,Xtest=Xtest,y=y,rel_th=0.5 , method = 'spearman')
  expect_equal(ncol(l$Xtrain),1)
  
  l = NULL
  l = ff.corrFilter(Xtrain=Xtrain,Xtest=Xtest,y=y,rel_th=0.5 , method = 'spearman')
  expect_equal(ncol(l$Xtrain),1)
  
  l = NULL
  l = ff.corrFilter(Xtrain=Xtrain,Xtest=Xtest,y=y,abs_th=1 , rel_th = NULL , method = 'spearman')
  expect_equal(ncol(l$Xtrain),1)
  
  l = NULL
  l = ff.corrFilter(Xtrain=Xtrain,Xtest=Xtest,y=y,abs_th=1 , rel_th = NULL , method = 'kendall')
  expect_equal(ncol(l$Xtrain),1)
  
  l = NULL
  l = ff.corrFilter(Xtrain=Xtrain,Xtest=Xtest,y=y,abs_th=1 , rel_th = NULL , method = 'pearson')
  expect_equal(ncol(l$Xtrain),1)
  
  expect_error(ff.corrFilter(Xtrain=Xtrain,Xtest=Xtest,y=y,abs_th=1 , rel_th = NULL , method = 'pippo'))
  
  expect_error(ff.corrFilter(Xtrain=Xtrain,Xtest=Xtest,y=y,abs_th=1 , rel_th = 4 , method = 'pearson'))
  
})




