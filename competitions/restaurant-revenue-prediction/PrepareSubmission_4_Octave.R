
library(data.table)

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
  
  list(train,y,test)
}



####
verbose = T
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))

#######

train.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train.csv" , sep=''))) 

test.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                      "test.csv" , sep='')))


####### basic feature processing 
l = buildData.basic(train.raw , test.raw)
train = l[[1]]
y = l[[2]]
test = l[[3]]


sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))


submitOctave = function (fn_oct = "pred_epsilon_SVR.csv" , 
                         fn_sub = "mySub_epsilon_SVR.csv" , 
                         y , 
                         doPlot = T) {
  
  sub = as.data.frame( fread(paste(getBasePath("data") , 
                                   fn_oct  , sep=''))) 
  
  
  #### min
  y_min = min(y)
  cat("min y train = ",y_min, " vs. min sub = ",min(sub$V1) , " --> adjiusting ... \n")
  sub$V1 = ifelse(sub$V1  >= y_min , sub$V1  , y_min)
  cat("---> min y train = ",min(y), " vs. min sub = ",min(sub$V1) , " ... \n")
  
  #### max 
  y_max = max(y)
  cat("max y train = ",y_max, " vs. max sub = ",max(sub$V1) , " --> adjiusting ... \n")
  sub$V1 = ifelse(sub$V1  <= y_max , sub$V1  , y_max)
  cat("---> max y train = ",max(y), " vs. max sub = ",max(sub$V1) , " ... \n")
  
  #####
  y_mean = mean(y)
  cat("mean y train = ",y_mean, " vs. mean sub = ",mean(sub$V1) , " --> NOT adjiusting ... \n")
#   cat("mean y train = ",y_mean, " vs. mean sub = ",mean(sub$V1) , " --> adjiusting ... \n")
#   sub$V1 = sub$V1 + (y_mean-mean(sub$V1))
#   cat("mean y train = ",y_mean, " vs. mean sub = ",mean(sub$V1) , " --> adjiusting ... \n")
  
  ### storing on disk 
  write.csv(data.frame(Id = sampleSubmission$Id , Prediction = sub$V1),
            quote=FALSE, 
            file=paste(getBasePath("data"), fn_sub ,sep='') ,
            row.names=FALSE)
  
  if (doPlot) {
    par(mfrow=c(1,2))
    hist(y)
    
    hist(sub$V1)
  }
  
}


submitOctave (fn_oct = "pred_epsilon_SVR.csv" , 
                         fn_sub = "mySub_epsilon_SVR.csv" , 
                         y)


submitOctave (fn_oct = "pred_nu_SVR.csv" , 
              fn_sub = "mySub_nu_SVR.csv" , 
              y)





