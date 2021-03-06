test_that('pca', {
  #skip_on_cran()
  
  ## suppress warnings raised because there few obs 
  warn_def = getOption('warn')
  options(warn=-1)
  
  ## data 
  Xtrain <- data.frame( a = rep(1:10 , each = 2), b = 20:1, c = rep(as.Date(c("2007-06-22", "2004-02-13")),10) , d = 20:1)
  Xtest <- data.frame( a = rep(2:11 , each = 2), b = 1:20, c = rep(as.Date(c("2007-03-01", "2004-05-23")),10) , d = 1:20)
  
  ## encode data sets 
  l = ff.makeFeatureSet(Xtrain,Xtest,c("C","N","D","N"))
  Xtrain = l$traindata
  Xtest = l$testdata
  
  ffPCA = ff.pca(Xtrain = Xtrain , Xtest = Xtest , center = TRUE , scale. = TRUE , removeZeroVarPredictors = TRUE , 
         varThreshold = 0.95 , doPlot = FALSE , verbose = TRUE)
  
  expect_equal(object = ffPCA$numComp , expected = 14)
  expect_equal(object = ffPCA$numComp.elbow , expected = 2)
  expect_equal(object = ffPCA$numComp.threshold , expected = 11)
  
  ## restore warnings 
  options(warn=warn_def)
  
})

test_that('k-means', {
  #skip_on_cran()
  
  ## suppress warnings raised because there few obs 
  warn_def = getOption('warn')
  options(warn=-1)
  
  ## data 
  Xtrain <- data.frame( a = rep(1:10 , each = 2), b = 20:1, c = rep(as.Date(c("2007-06-22", "2004-02-13")),10) , d = 20:1)
  Xtest <- data.frame( a = rep(2:11 , each = 2), b = 1:20, c = rep(as.Date(c("2007-03-01", "2004-05-23")),10) , d = 1:20)
  
  ## encode data sets 
  l = ff.makeFeatureSet(Xtrain,Xtest,c("C","N","D","N"))
  Xtrain = l$traindata
  Xtest = l$testdata
  
  MAX_CENTERS = 10 
  
  ffKmeans = ff.kmeans(Xtrain = Xtrain , Xtest = Xtest , max_centers = MAX_CENTERS , nstart = 5 , verbose = TRUE)
  
  expect_lt(object = ffKmeans$best_n_cluters , expected = ffKmeans$max_centers)
  expect_equal(object = length(ffKmeans$K.train),expected = nrow(Xtrain))
  expect_equal(object = length(ffKmeans$K.test),expected = nrow(Xtest))
  expect_equal(object = ffKmeans$max_centers ,expected = MAX_CENTERS)
  
  ## restore warnings 
  options(warn=warn_def)
  
})