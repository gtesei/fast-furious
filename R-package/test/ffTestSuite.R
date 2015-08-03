
library(RUnit) 

## test params 
ff_test_param = list(
  verbose = T 
  )

## before testing menv (make env)
base_path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/'
source(paste(base_path,'R-package/menv.R',sep=''))
runTestFile('/Users/gino/kaggle/fast-furious/gitHub/fast-furious/R-package/test/Test_menv.R')

## test all 
ff_test_suite <- defineTestSuite("Fast-Furious Test Suite",
                                 dirs = c('/Users/gino/kaggle/fast-furious/gitHub/fast-furious/R-package/test') , 
                                testFileRegexp = 'Test_\\w+\\.R'  )

test.result <- runTestSuite(ff_test_suite)
printTextProtocol(test.result)
