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

####### 
verbose = T

source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))
source(paste0( getBasePath("process") , "/Resample_Lib.R"))
source(paste0( getBasePath("process") , "/Regression_Lib.R"))
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))

controlObject <- trainControl(method = "repeatedcv", repeats = 10, number = 30)

cat(">>> resampling:: repeatedcv 10 30 \n")

#######
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

train.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train.csv" , sep=''))) 

test.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                      "test.csv" , sep='')))

sub = as.data.frame( fread(paste(getBasePath("data") , 
                                      "mySub_boost_2_1_71.csv" , sep='')))

cat(">> training observations:",dim(train.raw)[1]," \n")

####### basic feature processing 
l = buildData.basic(train.raw , test.raw , c(35,25,36) , T)
train = l[[1]]
y = l[[2]]
test = l[[3]]


#####
ytt.train = data.frame(year.to.target = train$years.to.target , 
                               revenues = y)

ytt.test = data.frame(year.to.target = test$years.to.target , 
                       revenues = sub$Prediction)

ytt.train.cl = ddply(ytt.train, .(year.to.target) , function(x) c(rev.mean.train = mean(x$revenues) , 
                                                                  rev.sd.train = sd(x$revenues), 
                                                                  num.train = length(x$revenues)
                                                                  ) )

ytt.train.cl$num.train.perc = (ytt.train.cl$num.train/length(train$P25_6))

ytt.test.cl = ddply(ytt.test, .(year.to.target) , function(x) c(rev.mean.test = mean(x$revenues) , 
                                                                rev.sd.test = sd(x$revenues) , 
                                                                num.test = length(x$revenues)
                                                                ) )

ytt.test.cl$num.test.perc = (ytt.test.cl$num.test/length(test$P2))

ytt.comp = merge(x = ytt.train.cl , y = ytt.test.cl , by = "year.to.target")

ytt.comp$delta = ytt.comp$rev.mean.train - ytt.comp$rev.mean.test

###
## some comparisons ... 
cat("******** BEFORE ADJIUSTING \n")
pred = sub$Prediction
cat("min  y train = ",min(y) , " vs. min pred = ", min(pred)  , " \n")
cat("max  y train = ",max(y) , " vs. max pred = ", max(pred)  , " \n")
cat("mean y train = ",mean(y), " vs. mean pred = ",mean(pred) ,  "\n")

###
perc.th = 0.05
for (yy in ytt.comp$year.to.target) {
  perc = ytt.comp[ytt.comp$year.to.target == yy , ]$num.train.perc 
  cat(">> processing year",yy," - perc ",perc," - ")
  if (perc >=perc.th) {
    delta = ytt.comp[ytt.comp$year.to.target == yy , ]$delta
    sub$Prediction[test$years.to.target == yy ] = sub$Prediction[test$years.to.target == yy ] + delta
    cat(">> modified ",sum(test$years.to.target == yy) , "submissions ")
  }
  cat("\n")
}

## some comparisons ... 
cat("******** AFTER ADJIUSTING \n")
pred = sub$Prediction
cat("min  y train = ",min(y) , " vs. min pred = ", min(pred)  , " \n")
cat("max  y train = ",max(y) , " vs. max pred = ", max(pred)  , " \n")
cat("mean y train = ",mean(y), " vs. mean pred = ",mean(pred) ,  "\n")

write.csv(sub,
          quote=FALSE, 
          file=paste(getBasePath("data"),"mySub_boost_2_1_71___adjiusted.csv",sep='') ,
          row.names=FALSE)
