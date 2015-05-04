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

buildData.basic = function(train.raw , test.raw , features_to_encode , remove_enc) {
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
  
  ## encoding 
  toRemove = NULL
  
  for (i in features_to_encode) {
    
    ft = paste0("P",i)
    
    l = encodeCategoricalFeature (train[,i] , test[,i] , colname.prefix = ft , asNumeric=F)
    tr = l[[1]]
    ts = l[[2]]
    
    train = cbind(train,tr)
    test = cbind(test,ts)
    
    toRemove = c(toRemove,i)
  }
  
  ## high revenue combinations 
  train$hrc = ifelse(train$P1 == 5 & train$P8 == 3 , 1 , 0) 
  test$hrc = ifelse(test$P1 == 5 & test$P8 == 3 , 1 , 0) 
  
  ## removing 
  if (remove_enc) {
    train = train[ , -toRemove]
    test = test[ , -toRemove]
  }
  
  list(train,y,test)
}

buildData.analysis = function(train.raw , test.raw) {
  ## remove id 
  train = train.raw[ , -1] 
  test = test.raw[ , -1] 
  
  ## 2014 should be the target year ... so use open date to misure the number of years between open date and the target year 
  train$years.to.target = 2014 - year(as.Date( train.raw[,2] , "%m/%d/%Y"))
  test$years.to.target = 2014 - year(as.Date( test.raw[,2] , "%m/%d/%Y"))
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## extracting y 
  y = train[,41]
  train = train[,-41]
  
  list(train,y,test)
}

####### 
verbose = T

source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))
source(paste0( getBasePath("process") , "/Resample_Lib.R"))
source(paste0( getBasePath("process") , "/Regression_Lib.R"))
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))

controlObject <- trainControl(method = "repeatedcv", repeats = 10, number = 30)

cat(">>> resampling:: repeatedcv 10 30 \n")

RegModels = c("Average" , "Mode",  
              "LinearReg", "RobustLinearReg", 
              "PLS_Reg" , "Ridge_Reg" , "Enet_Reg" , 
              "KNN_Reg", 
              "SVM_Reg", 
              "BaggedTree_Reg"
              , "RandomForest_Reg"
              , "Cubist_Reg" 
              #, "NNet"
) 

#RegModels = c("Average","Mode")

cat("****** Available regression models ******\n") 
print(RegModels)

#######
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

train.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train.csv" , sep=''))) 

test.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                      "test.csv" , sep='')))

###

#FEATURES_NUMERIC = c(2,3,4,13,26,27,28,29)

# for (i in 1:37) {
#   if (i %in% FEATURES_NUMERIC) next
#   ft = paste0("P",i)
#   vals = sort(unique(c(train.raw[,ft],test.raw[,ft])))
#   cat("\n>>> ",ft,":",vals," \n")
# } 


####### analysis 
l = buildData.analysis(train.raw , test.raw)
train.an = l[[1]]
y = l[[2]]
test.an = l[[3]]


##### test.nc 
cols = dim(test.nc)[2]
test.nc = as.data.frame(matrix(NA,100,cols))
colnames(test.nc) = colnames(test.an)
for (col in 1:cols) {
  val.test = sort(unique(test.an[,col])) 
  val.train = sort(unique(train.an[,col])) 
  val.nc = val.test[! val.test %in% val.train]
  if (length(val.nc) > 0) {
    test.nc[(1:length(val.nc)),col] = val.nc
  }
  cat(">>> ",colnames(test.an)[col],": found ",length(val.nc)," \n")
}

##### test.nc.det
test.nc.det = as.data.frame(matrix(0,dim(test.an)[1],cols))
colnames(test.nc.det) = colnames(test.an)
y.nc.det = rep(0,dim(test.an)[1])
for (col in 1:cols) {
  vals = test.nc[,col]
  vals = na.omit(vals)
  cat(">>> ",colnames(test.an)[col],": vals[",length(vals),"] ",vals," ... \n")
  if ( length(vals) > 0) {
    for (row in 1:(dim(test.an)[1]) ) {
      test.nc.det[row,col] = ifelse( test.an[row,col] %in% vals , 1 , 0) 
      y.nc.det[row] = y.nc.det[row] + test.nc.det[row,col]
    }
  }
}

###
year.to.target.an = data.frame(year.to.target = train.an$years.to.target , 
                               revenues = y)

year.to.target.clusters = ddply(year.to.target.an, .(year.to.target) , function(x) c(rev.mean = mean(x$revenues) , 
                                                                                     rev.sd = sd(x$revenues)) ) 
par(mfrow=c(2,1))
plot(year.to.target.clusters$year.to.target , year.to.target.clusters$rev.mean , type="l")
plot(year.to.target.clusters$year.to.target , year.to.target.clusters$rev.sd , type="l")

year.to.target.clusters$zscore = (year.to.target.clusters$rev.mean/year.to.target.clusters$rev.sd)

####
write.csv(test.nc,
          quote=FALSE, 
          file=paste(getBasePath("data"),"test_not_covered.csv",sep='') ,
          row.names=FALSE)

write.csv(test.nc.det,
          quote=FALSE, 
          file=paste(getBasePath("data"),"test_not_covered_detail.csv",sep='') ,
          row.names=FALSE)

write.csv(y.nc.det,
          quote=FALSE, 
          file=paste(getBasePath("data"),"ytest_not_covered_detail.csv",sep='') ,
          row.names=FALSE)


####### basic feature processing 
#l = buildData.basic(train.raw , test.raw , c(35,25,36,31,17,5) , T)
l = buildData.basic(train.raw , test.raw , c(35,25,36) , T)
train = l[[1]]
y = l[[2]]
test = l[[3]]

## process data 
l = featureSelect (train,test,y=y,
                   removeOnlyZeroVariacePredictors = T,
                   performVarianceAnalysisOnTrainSetOnly = T,
                   correlationRhreshold = NA
)
traindata = l[[1]]
testdata = l[[2]]

## pred.1 
# pred = 
#   reg.trainAndPredict( y , 
#                        traindata , 
#                        testdata , 
#                        "BaggedTree_Reg" , 
#                        controlObject, 
#                        best.tuning = T)


fit <- train(y = y, x = traindata ,
             method = "treebag",
             trControl = controlObject)

pred = as.numeric( predict(fit , testdata )  )

featImp = varImp(fit , scale = F)
cat("************** Predictor Importance - not scaled ")
featImp

cat("************** Predictor Importance - scaled ")
featImp.scaled = varImp(fit , scale = T)
featImp.scaled


cat(">>> Storing on disk ... \n")

write.csv(data.frame(Id = test.raw$Id , Prediction = pred ),
          quote=FALSE, 
          file=paste(getBasePath("data"),"pred__6_feat_cat.csv",sep='') ,
          row.names=FALSE)



