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


#####
grid = data.frame(Id.train = train.raw$Id , 
                  n.train = NA , 
                  n.test = NA, 
                  n.test.1 = NA , 
                  n.test.2 = NA , 
                  n.test.3 = NA , 
                  n.test.4 = NA , 
                  n.test.5 = NA , 
                  n.test.6 = NA , 
                  n.test.7 = NA , 
                  n.test.8 = NA , 
                  n.test.9 = NA , 
                  n.test.10 = NA , 
                  n.test.11 = NA , 
                  n.test.12 = NA , 
                  n.test.13 = NA , 
                  n.test.14 = NA , 
                  n.test.15 = NA , 
                  n.test.16 = NA , 
                  n.test.17 = NA , 
                  n.test.18 = NA , 
                  n.test.19 = NA , 
                  n.test.20 = NA , 
                  n.test.21 = NA , 
                  n.test.22 = NA , 
                  n.test.23 = NA , 
                  n.test.24 = NA , 
                  n.test.25 = NA , 
                  n.test.26 = NA , 
                  n.test.27 = NA , 
                  n.test.28 = NA , 
                  n.test.29 = NA , 
                  n.test.30 = NA , 
                  n.test.31 = NA , 
                  n.test.32 = NA , 
                  n.test.33 = NA , 
                  n.test.34 = NA , 
                  n.test.35 = NA , 
                  n.test.36 = NA , 
                  n.test.37 = NA 
                  )

for (id in train.raw$Id) {
  grid[grid$Id.train==id,]$n.train = dim(train.raw[
  train.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
  train.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
  train.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
  train.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
  train.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
  train.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
  train.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
  train.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
  train.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
  train.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
  train.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
  train.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
  train.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
  train.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
  train.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
  train.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
  train.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
  train.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
  train.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
  train.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
  train.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
  train.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
  train.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
  train.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
  train.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
  train.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
  train.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
  train.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
  train.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
  train.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
  train.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
  train.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
  train.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
  train.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
  train.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
  train.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
  train.raw$P37 == train.raw[train.raw$Id == id,]$P37 
  ,])[1]
  
  # n.test
  grid[grid$Id.train==id,]$n.test = dim(test.raw[
      test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
      test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
        test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
        test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
        test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
        test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
        test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
        test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
        test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
        test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
        test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
        test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
        test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.1
  grid[grid$Id.train==id,]$n.test.1 = dim(test.raw[
    #test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
      test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.2
  grid[grid$Id.train==id,]$n.test.2 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    #test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.3
  grid[grid$Id.train==id,]$n.test.3 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      #test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.4
  grid[grid$Id.train==id,]$n.test.4 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      #test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.5
  grid[grid$Id.train==id,]$n.test.5 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      #test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.6
  grid[grid$Id.train==id,]$n.test.6 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      #test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.7
  grid[grid$Id.train==id,]$n.test.7 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      #test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.8
  grid[grid$Id.train==id,]$n.test.8 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      #test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.9
  grid[grid$Id.train==id,]$n.test.9 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      #test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  
  # n.test.10
  grid[grid$Id.train==id,]$n.test.10 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      #test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.11
  grid[grid$Id.train==id,]$n.test.11 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      #test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.12
  grid[grid$Id.train==id,]$n.test.12 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      #test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.13
  grid[grid$Id.train==id,]$n.test.13 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      #test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.14
  grid[grid$Id.train==id,]$n.test.14 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      #test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.15
  grid[grid$Id.train==id,]$n.test.15 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      #test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.16
  grid[grid$Id.train==id,]$n.test.16 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      #test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.17
  grid[grid$Id.train==id,]$n.test.17 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      #test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.18
  grid[grid$Id.train==id,]$n.test.18 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      #test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.19
  grid[grid$Id.train==id,]$n.test.19 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      #test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.20
  grid[grid$Id.train==id,]$n.test.20 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      #test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.21
  grid[grid$Id.train==id,]$n.test.21 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      #test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.22
  grid[grid$Id.train==id,]$n.test.22 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      #test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.23
  grid[grid$Id.train==id,]$n.test.23 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      #test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.24
  grid[grid$Id.train==id,]$n.test.24 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      #test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.25
  grid[grid$Id.train==id,]$n.test.25 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      #test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.26
  grid[grid$Id.train==id,]$n.test.26 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      #test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.27
  grid[grid$Id.train==id,]$n.test.27 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      #test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.28
  grid[grid$Id.train==id,]$n.test.28 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      #test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.29
  grid[grid$Id.train==id,]$n.test.29 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      #test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  
  # n.test.30
  grid[grid$Id.train==id,]$n.test.30 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      #test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.31
  grid[grid$Id.train==id,]$n.test.31 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      #test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.32
  grid[grid$Id.train==id,]$n.test.32 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      #test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.33
  grid[grid$Id.train==id,]$n.test.33 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      #test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.34
  grid[grid$Id.train==id,]$n.test.34 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      #test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.35
  grid[grid$Id.train==id,]$n.test.35 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      #test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.36
  grid[grid$Id.train==id,]$n.test.36 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      #test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
      test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  # n.test.37
  grid[grid$Id.train==id,]$n.test.37 = dim(test.raw[
    test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
    test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
      test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
      test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
      test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
      test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
      test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
      test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
      test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
      test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
      test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
      test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
      test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
      test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
      test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
      test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
      test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
      test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
      test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
      test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
      test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
      test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
      test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
      test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
      test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
      test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
      test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
      test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
      test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
      test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
      test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
      test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
      test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
      test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
      test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
      test.raw$P36 == train.raw[train.raw$Id == id,]$P36 
      #test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
    ,])[1]
  
  
#     if (grid[grid$Id.train==id,]$n.test > 0) {
#       cat("modifying sub ... \n")
#       sub[
#         test.raw$P1 == train.raw[train.raw$Id == id,]$P1 & 
#           test.raw$P2 == train.raw[train.raw$Id == id,]$P2 & 
#           test.raw$P3 == train.raw[train.raw$Id == id,]$P3 & 
#           test.raw$P4 == train.raw[train.raw$Id == id,]$P4 & 
#           test.raw$P5 == train.raw[train.raw$Id == id,]$P5 & 
#           test.raw$P6 == train.raw[train.raw$Id == id,]$P6 & 
#           test.raw$P7 == train.raw[train.raw$Id == id,]$P7 & 
#           test.raw$P8 == train.raw[train.raw$Id == id,]$P8 & 
#           test.raw$P9 == train.raw[train.raw$Id == id,]$P9 & 
#           test.raw$P10 == train.raw[train.raw$Id == id,]$P10 & 
#           test.raw$P11 == train.raw[train.raw$Id == id,]$P11 & 
#           test.raw$P12 == train.raw[train.raw$Id == id,]$P12 & 
#           test.raw$P13 == train.raw[train.raw$Id == id,]$P13 & 
#           test.raw$P14 == train.raw[train.raw$Id == id,]$P14 & 
#           test.raw$P15 == train.raw[train.raw$Id == id,]$P15 & 
#           test.raw$P16 == train.raw[train.raw$Id == id,]$P16 & 
#           test.raw$P17 == train.raw[train.raw$Id == id,]$P17 & 
#           test.raw$P18 == train.raw[train.raw$Id == id,]$P18 & 
#           test.raw$P19 == train.raw[train.raw$Id == id,]$P19 & 
#           test.raw$P20 == train.raw[train.raw$Id == id,]$P20 & 
#           test.raw$P21 == train.raw[train.raw$Id == id,]$P21 & 
#           test.raw$P22 == train.raw[train.raw$Id == id,]$P22 & 
#           test.raw$P23 == train.raw[train.raw$Id == id,]$P23 & 
#           test.raw$P24 == train.raw[train.raw$Id == id,]$P24 & 
#           test.raw$P25 == train.raw[train.raw$Id == id,]$P25 & 
#           test.raw$P26 == train.raw[train.raw$Id == id,]$P26 & 
#           test.raw$P27 == train.raw[train.raw$Id == id,]$P27 & 
#           test.raw$P28 == train.raw[train.raw$Id == id,]$P28 & 
#           test.raw$P29 == train.raw[train.raw$Id == id,]$P29 & 
#           test.raw$P30 == train.raw[train.raw$Id == id,]$P30 & 
#           test.raw$P31 == train.raw[train.raw$Id == id,]$P31 & 
#           test.raw$P32 == train.raw[train.raw$Id == id,]$P32 & 
#           test.raw$P33 == train.raw[train.raw$Id == id,]$P33 &
#           test.raw$P34 == train.raw[train.raw$Id == id,]$P34 &
#           test.raw$P35 == train.raw[train.raw$Id == id,]$P35 &
#           test.raw$P36 == train.raw[train.raw$Id == id,]$P36 &
#           test.raw$P37 == train.raw[train.raw$Id == id,]$P37 
#         ,]$Prediction = train.raw[train.raw$Id == id,]$revenue
#     }
}

cat("************************\n")
print(grid)

grid.res = apply(grid,2,function(x) sum(x))
cat("************************\n")
print(grid.res)

# cat("********** storing on disk \n")
# write.csv(sub,
#           quote=FALSE, 
#           file=paste(getBasePath("data"),"pred_adjiusted_last.csv",sep='') ,
#           row.names=FALSE)
