library(caret)
library(Hmisc)
library(data.table)
library(verification)
library(pROC)
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
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/walmart-recruiting-sales-in-stormy-weather"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/walmart-recruiting-sales-in-stormy-weather/"
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

getTrain = function () {
  path = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/train.csv"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/train.csv"
  
  if (file.exists(base.path1))  {
    path = base.path1
  } else if (file.exists(base.path2)) {
    path = base.path2
  } else {
    stop('impossible load train.csv')
  }
  
  cat("loading train data ... ")
  trdata = as.data.frame(fread(path))
  #cat("converting date ...")
  #trdata$date = as.Date(trdata$date,"%Y-%m-%d")
  trdata
} 
getTest = function () {
  path = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/test.csv"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/test.csv"
  
  if (file.exists(base.path1))  {
    path = base.path1
  } else if (file.exists(base.path2)) {
    path = base.path2
  } else {
    stop('impossible load train.csv')
  }
  
  cat("loading train data ... ")
  trdata = as.data.frame(fread(path))
  #cat("converting date ...")
  #trdata$date = as.Date(trdata$date,"%Y-%m-%d")
  trdata
} 

getTrainClosestDates.fast = function (testdata.header , traindata.header) {
  train.date = traindata.header$as.date
  test.date = testdata.header$as.date
  
  train.closesest = rep(as.Date("1900-01-01", "%Y-%m-%d"),length(testdata.header$as.date))
  min.diff = rep(-1,length(testdata.header$as.date))
  
  i = 1 
  while (i <= length(train.closesest) ) {
    td = test.date[i]
    
    md = min( abs(td - train.date) )
    while (! ((td-md) %in% train.date) ) md = md - 1 
    train.closesest[i] = td-md
    min.diff[i] = md
    
    i = i + 1 
    while ( (i <= length(train.closesest)) 
            & (test.date[i] == (test.date[i-1]+1))  ) {
      train.closesest[i] = train.closesest[i-1]
      min.diff[i] = min.diff[i-1]+1
      i = i + 1 
    }
  }
 list(train.closesest,min.diff)
}

getTrainClosestDates = function (testdata.header , traindata.header) {
  train.date = traindata.header$as.date
  test.date = testdata.header$as.date
  
  train.closesest = rep(as.Date("1900-01-01", "%Y-%m-%d"),length(testdata.header$as.date))
  min.diff = rep(-1,length(testdata.header$as.date))
  
  i = 1 
  while (i <= length(train.closesest) ) {
    td = test.date[i]
    
    md = min( abs(td - train.date) )
    while (! ((td-md) %in% train.date) ) md = md - 1 
    train.closesest[i] = td-md
    min.diff[i] = md
    
    i = i + 1 
  }
  list(train.closesest,min.diff)
}

##################
verbose = T 
source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))
source(paste0( getBasePath("process") , "/Resample_Lib.R"))
source(paste0( getBasePath("process") , "/Regression_Lib.R"))

##################
train = getTrain()
test = getTest()
keys = as.data.frame( fread(paste(getBasePath("data") , 
                                  "key.csv" , sep='')))
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

weather = as.data.frame( fread(paste(getBasePath("data") , 
                                     "weather.imputed.basic.17.9.csv" , sep=''))) ## <<<< TODO use weather.imputed.all.<perf>.csv

######
RegModels = c("Average" , "Mode",  
              "LinearReg", "RobustLinearReg", 
              "PLS_Reg" , "Ridge_Reg" , "Enet_Reg" , 
              "KNN_Reg", 
              #"SVM_Reg", 
              "BaggedTree_Reg"
              #, "RandomForest_Reg"
              #, "Cubist_Reg"
) 

###
sub = NULL
grid = NULL
controlObject <- trainControl(method = "boot", number = 100)
#controlObject <- trainControl(method = "repeatedcv" , repeats = 5 , number = 10)

###
stores.test = sort(unique(test$store_nbr))
items.test = sort(unique(test$item_nbr))

st = 1
it = 9 
stat = keys[keys$store_nbr == st,]$station_nbr 
##############
pred = NULL

## testdata
testdata = test[test$store_nbr == st & test$item_nbr == it ,  ]
testdata$station_nbr = stat
testdata = merge(x = testdata,y = weather, by=c("station_nbr","date"))
testdata.header = testdata[,c(1,2,3,4)]
testdata = testdata[,-c(1,2,3,4)]

## traindata
traindata = train[train$store_nbr == st & train$item_nbr == it ,  ]
traindata$station_nbr = stat
traindata = merge(x = traindata,y = weather, by=c("station_nbr","date"))
traindata.header = traindata[,c(1,2,3,4,5)]
traindata.y = traindata[,5]
traindata = traindata[,-c(1,2,3,4,5)]

## 
traindata.header$as.date = as.Date(traindata.header$date, "%Y-%m-%d")
traindata.header$data_type = 1

testdata.header$as.date = as.Date(testdata.header$date, "%Y-%m-%d")
testdata.header$data_type = 2 

data = rbind(traindata.header[,-5],testdata.header)

par(mfrow=c(1,1))
plot(data$as.date,data$data_type , xlab = "date", ylab = "1 = train / 2 = test") 

####
train.st = train[train$store_nbr == st , ]
is_in_train_set = apply(testdata.header , 1 , function(x) ifelse(x[2] %in% train$date , 1 , 0) )
is_in_train_set
is_in_train_set_same_store = apply(testdata.header , 1 , function(x) ifelse(x[2] %in% train.st$date , 1 , 0) )
is_in_train_set_same_store
is_in_traindata_set = apply(testdata.header , 1 , function(x) ifelse(x[2] %in% traindata.header$date , 1 , 0) )
is_in_traindata_set

unique(train[which(train$date == "2013-06-04") , ]$store_nbr)

plot(traindata.header$as.date , traindata.header$units, xlab = "date", ylab = "units sold" )
points(testdata.header$as.date, rep(mean(traindata.y),length(testdata.header$as.date)), pch=16, col="green")
legend("topleft", cex=.5  , legend = c("units sold in train set","mean as prediction in test set"),  
       pch=c(1,16) , col = c(1,"green"))

###
plot(traindata.header$as.date , traindata.header$units, xlab = "date", ylab = "units sold" )
points(testdata.header$as.date, rep(mean(traindata.y),length(testdata.header$as.date)), pch=16, col="green")

###
# Smoothed symmetrically:
f3 <- rep(1/3,3)
f3
y_3 <- filter(traindata.header$units, f3, sides=2)
lines(traindata.header$as.date, y_3, col=550 , lwd=2)


###
# Smoothed symmetrically:
f7 <- rep(1/7,7)
f7
y_7 <- filter(traindata.header$units, f7, sides=2)
lines(traindata.header$as.date, y_7, col="blue", lwd=2)

###
# Smoothed symmetrically:
f14 <- rep(1/14,14)
f14
y_14 <- filter(traindata.header$units, f14, sides=2)
lines(traindata.header$as.date, y_14, col="brown" , lwd=2)

legend("topleft", cex=.5 , legend = c("units sold in train set", "mean as prediction in test set", "moving average 3", "moving average 7", "moving average 14"),  
       pch=c(1,16,3,3,3) ,
       col = c(1,"green",550,"blue","brown")
        )

## 
corr = rep(NA,100)
ww = 100
win = 1:ww
for (w in win){
  cat("processing moving average " , w , " ... ")
  f.w <- rep(1/w,w)
  y.w <- filter(traindata.header$units, f.w, sides=2)
  el = c(c(1:floor(w/2)),c((929-floor(w/2)):929))
  c.w = cor(traindata.y[-el],y.w[-el]) 
  pval.w = cor.test(traindata.y[-el], y.w[-el], conf.level = 0.95)$p.value
  cat("cor=" , c.w , " - pval=", pval.w,"  \n")
  if (pval.w > 0.05) break 
  corr[w] = c.w
}

plot(corr , pch=16 , col="blue")


delta = corr[2:(length(corr))] - corr[1:(length(corr)-1)]
plot(delta)

candidates = NULL 
for (i in 1:(ww-2)) {
  if ( (delta[i] > 0 & delta[i+1] < 0) | 
         (delta[i] < 0 & delta[i+1] > 0)) {
    candidates = c(candidates,i) 
  }
}
candidates
corr[candidates]

##
l = getTrainClosestDates.fast (testdata.header , traindata.header)
train.closesest = l[[1]] 
min.diff = l[[2]]

l = getTrainClosestDates (testdata.header , traindata.header)
train.closesest.2 = l[[1]] 
min.diff.2 = l[[2]]

train.closesest-train.closesest.2
min.diff-min.diff.2


##
pi = 0.1
piT = 200
C = data.frame( date = traindata.header$as.date , val = rep(NA,length(traindata.header$as.date)) )
for (u in 1:length(traindata.header$as.date)) {
  cat("u:",u,"\n")
  dd = traindata.header[u,]$as.date
  
  ok = T
  ok = ok & (length(traindata.header[traindata.header$as.date == dd,]$units) > 0)
  ok = ok & (length(traindata.header[traindata.header$as.date == (dd-1),]$units) > 0)
  ok = ok & (length(traindata.header[traindata.header$as.date == (dd-1-piT),]$units) > 0)

  if (! ok ) next 
  
  C[dd-1,]$val =   (  traindata.header[traindata.header$as.date == dd,]$units + pi * (traindata.header[traindata.header$as.date == (dd-1),]$units -  traindata.header[traindata.header$as.date == (dd-1-piT),]$units ) ) / pi
}




