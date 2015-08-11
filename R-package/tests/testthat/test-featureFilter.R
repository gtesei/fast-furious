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

test_that('removing high correlated predictors', {
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
                        verbose=TRUE)
  
  expect_equal(ncol(l$testdata),3)
})




