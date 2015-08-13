context("featureEncode")

test_that('encode categorical feature', {
  #skip_on_cran()
  Xtrain <- data.frame( a = rep(1:3 , each = 2), b = c(4:1,6,6), c = letters[1:6])
  Xtest <- data.frame( a = rep(2:4 , each = 2), b = c(1:4,6,6), c = letters[6:1])
  l = ff.encodeCategoricalFeature (Xtrain$c , Xtest$c , "c")
  expect_equal(ncol(l$traindata),6)
  expect_equal(ncol(l$testdata),6)
  expect_equal(nrow(l$traindata),6)
  expect_equal(nrow(l$testdata),6)
  l = ff.encodeCategoricalFeature (Xtrain$b, Xtest$b , "b" , asNumericSequence = T)
  expect_equal(l$traindata$b_6[6],1)
  l = ff.encodeCategoricalFeature (Xtrain$b, Xtest$b , "b" , levels = c(4:1,6))
  expect_equal(l$traindata$b_6[6],1)
})

test_that('extract date feature', {
  #skip_on_cran()
  Xtrain <- data.frame( a = rep(1:3 , each = 2), b = 6:1, c = rep(as.Date(c("2007-06-22", "2004-02-13")),3) )
  Xtest <- data.frame( a = rep(2:4 , each = 2), b = 1:6, c = rep(as.Date(c("2007-03-01", "2004-05-23")),3) )
  l = ff.extractDateFeature(Xtrain$c,Xtest$c)
  expect_equal(length(l$traindata),6)
  expect_equal(length(l$testdata),6)
  expect_equal(l$traindata[2],0)
})

test_that('make feature set', {
  #skip_on_cran()
  Xtrain <- data.frame( a = rep(1:3 , each = 2), b = 6:1, c = rep(as.Date(c("2007-06-22", "2004-02-13")),3) )
  Xtest <- data.frame( a = rep(2:4 , each = 2), b = 1:6, c = rep(as.Date(c("2007-03-01", "2004-05-23")),3) )
  meta = c('C','N','D')
  
  ## no scaling 
  l = ff.makeFeatureSet(data.train = Xtrain , data.test = Xtest , meta = meta)
  
  expect_equal(nrow(l$traindata),nrow(Xtrain))
  expect_equal(nrow(l$testdata),nrow(Xtest))
  
  expect_equal(ncol(Xtrain)+3,ncol(l$traindata))
  expect_equal(ncol(Xtest)+3,ncol(l$testdata))
  
  l = ff.makeFeatureSet(data.train = Xtrain , data.test = Xtest , meta = meta , parallelize = TRUE)
  
  expect_equal(nrow(l$traindata),nrow(Xtrain))
  expect_equal(nrow(l$testdata),nrow(Xtest))
  
  expect_equal(ncol(Xtrain)+3,ncol(l$traindata))
  expect_equal(ncol(Xtest)+3,ncol(l$testdata))
  
  ## scaling 
  l = ff.makeFeatureSet(data.train = Xtrain , data.test = Xtest , meta = meta , scaleNumericFeatures = TRUE)
  
  expect_equal(nrow(l$traindata),nrow(Xtrain))
  expect_equal(nrow(l$testdata),nrow(Xtest))
  
  expect_equal(ncol(Xtrain)+3,ncol(l$traindata))
  expect_equal(ncol(Xtest)+3,ncol(l$testdata))
  
  ## scaling and parallelize 
  l = ff.makeFeatureSet(data.train = Xtrain , data.test = Xtest , meta = meta , scaleNumericFeatures = TRUE ,parallelize = TRUE)
  
  expect_equal(nrow(l$traindata),nrow(Xtrain))
  expect_equal(nrow(l$testdata),nrow(Xtest))
  
  expect_equal(ncol(Xtrain)+3,ncol(l$traindata))
  expect_equal(ncol(Xtest)+3,ncol(l$testdata))
  
  ## scaling and ! parallelize 
  l = ff.makeFeatureSet(data.train = Xtrain , data.test = Xtest , meta = meta , scaleNumericFeatures = TRUE ,parallelize = FALSE)
  
  expect_equal(nrow(l$traindata),nrow(Xtrain))
  expect_equal(nrow(l$testdata),nrow(Xtest))
  
  expect_equal(ncol(Xtrain)+3,ncol(l$traindata))
  expect_equal(ncol(Xtest)+3,ncol(l$testdata))
  
})


