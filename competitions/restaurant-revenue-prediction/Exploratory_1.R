library(caret)
library(Hmisc)
library(data.table)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)
library(fBasics)

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/restaurant-revenue-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/restaurant-revenue-prediction/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/restaurant-revenue-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/restaurant-revenue-prediction/"
  } else if (type == "process") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/data_process"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/data_process/"
  } else {
    stop("unrecognized type.")
  }
  
  if (file.exists(base.path1))  {
    ret = paste0(base.path1,"/")
  } else {
    ret = base.path2
  }
  
  ret
}

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
getPvalueTypeIError = function(x,y) {
  test = NA
  pvalue = NA
  
  ## type casting and understanding stat test 
  if (class(x) == "integer") x = as.numeric(x)
  if (class(y) == "integer") y = as.numeric(y)
  
  if ( class(x) == "factor" & class(y) == "numeric" ) {
    # C -> Q
    test = "ANOVA"
  } else if (class(x) == "factor" & class(y) == "factor" ) {
    # C -> C
    test = "CHI-SQUARE"
  } else if (class(x) == "numeric" & class(y) == "numeric" ) {
    test = "PEARSON"
  }  else {
    # Q -> C 
    # it performs anova test x ~ y 
    test = "ANOVA"
    tmp = x 
    x = y 
    y = tmp 
  }
  
  ## performing stat test and computing p-value
  if (test == "ANOVA") {                
    test.anova = aov(y~x)
    pvalue = summary(test.anova)[[1]][["Pr(>F)"]][1]
  } else if (test == "CHI-SQUARE") {    
    test.chisq = chisq.test(x = x , y = y)
    pvalue = test.chisq$p.value
  } else {                             
    ###  PEARSON
    test.corr = cor.test(x =  x , y =  y)
    pvalue = test.corr$p.value
  }
  
  pvalue
}

buildData.basic = function(train.raw , test.raw) {
  ## remove id 
  train = train.raw[ , -1] 
  test = test.raw[ , -1] 
  
  ## 2014 should be the target year ... so use open date to misure the number of years between open date and the target year 
  train$years.to.target = 2014 - year(as.Date( train.raw[,2] , "%m/%d/%Y"))
  test$years.to.target = 2014 - year(as.Date( test.raw[,2] , "%m/%d/%Y"))
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## City
  l = encodeCategoricalFeature (train[,1] , test[,1] , colname.prefix = "city" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## City Group 
  l = encodeCategoricalFeature (train[,1] , test[,1] , colname.prefix = "city.group" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## Type
  l = encodeCategoricalFeature (train[,1] , test[,1] , colname.prefix = "type" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## extracting y 
  y = train[,38]
  train = train[,-38]
  
  ## P29
  l = encodeCategoricalFeature (train[,29] , test[,29] , colname.prefix = "P29" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -29]
  test = test[ , -29]
  
  ## P35
  l = encodeCategoricalFeature (train[,35] , test[,35] , colname.prefix = "P35" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -35]
  test = test[ , -35]
  
  ## P25
  l = encodeCategoricalFeature (train[,25] , test[,25] , colname.prefix = "P25" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -25]
  test = test[ , -25]
  
  ## P36
  l = encodeCategoricalFeature (train[,36] , test[,36] , colname.prefix = "P36" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -36]
  test = test[ , -36]
  
  #   ## P31
  #   l = encodeCategoricalFeature (train[,31] , test[,31] , colname.prefix = "P31" , asNumeric=F)
  #   tr = l[[1]]
  #   ts = l[[2]]
  #   
  #   train = cbind(train,tr)
  #   test = cbind(test,ts)
  #   
  #   train = train[ , -31]
  #   test = test[ , -31]
  
#   # P27
#   l = encodeCategoricalFeature (train[,27] , test[,27] , colname.prefix = "P27" , asNumeric=F)
#   tr = l[[1]]
#   ts = l[[2]]
#   
#   train = cbind(train,tr)
#   test = cbind(test,ts)
#   
#   train = train[ , -27]
#   test = test[ , -27]
  
  #   ## P17
  #   l = encodeCategoricalFeature (train[,17] , test[,17] , colname.prefix = "P17" , asNumeric=F)
  #   tr = l[[1]]
  #   ts = l[[2]]
  #   
  #   train = cbind(train,tr)
  #   test = cbind(test,ts)
  #   
  #   train = train[ , -17]
  #   test = test[ , -17]
  #   
  #   ## P5
  #   l = encodeCategoricalFeature (train[,5] , test[,5] , colname.prefix = "P5" , asNumeric=F)
  #   tr = l[[1]]
  #   ts = l[[2]]
  #   
  #   train = cbind(train,tr)
  #   test = cbind(test,ts)
  #   
  #   train = train[ , -5]
  #   test = test[ , -5]
  
  ## 
  list(train,y,test)
}

predict.train.k.folds = function (traindata , 
                                  y , 
                                  model.label = "RandomForest_Reg", 
                                  k = 6 ) {
  ### train set 
  folds = kfolds(k,dim(traindata)[1])
  
  pred.1.train = rep(NA,dim(traindata)[1])
  
  for(j in 1:k) {  
    if (verbose) cat("--k-fold:: ",j, "/",k , "\n")
    traindata.train <- traindata[ folds != j,]
    traindata.y.train = y[folds != j]
    
    traindata.xval <- traindata[folds == j,]
    traindata.y.xval = y[folds == j]
    
    ###
    pred.1.train.fold = reg.trainAndPredict( traindata.y.train , 
                                             traindata.train , 
                                             traindata.xval , 
                                             model.label , 
                                             controlObject, 
                                             best.tuning = T)
    
    pred.1.train.fold = ifelse(pred.1.train.fold >= 1150 , pred.1.train.fold , 1150) 
    
    pred.1.train[folds == j] = pred.1.train.fold
  } ### end of k-fold 
  
  ## check 
  if (sum(is.na(pred.1.train)) > 0)
    stop("something wrong (NAs) in tip.train")
  
  pred.1.train
}

#######
verbose = T
source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))
source(paste0( getBasePath("process") , "/Resample_Lib.R"))
source(paste0( getBasePath("process") , "/Regression_Lib.R"))
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))

controlObject <- trainControl(method = "boot", number = 200)

#######
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

train.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train.csv" , sep=''))) 

test.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                      "test.csv" , sep='')))


####### basic feature processing 
l = buildData.basic(train.raw , test.raw)
train = l[[1]]
y = l[[2]]
test = l[[3]]

#####

#describe(train)




