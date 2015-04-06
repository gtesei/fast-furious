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

### clusters 
clusters = ddply(train[,-1], .(store_nbr,item_nbr) , function(x) c(units = mean(x$units)) ) 
clusters$sold = ifelse(clusters$units > 0 , 1 , 0)
clusters.for.merge = clusters[,-3]

stores = unique(clusters$store_nbr)
items = unique(clusters$item_nbr)
stations = unique(keys$station_nbr)

store.station = ddply(keys , .(station_nbr) , function(x) c(stores = sum(x$store_nbr > 0)) )

## merge
train = merge(x = train,y = clusters.for.merge, by=c("store_nbr","item_nbr") , all.x=T, all.y=T)
train = merge(x = train,y = keys, by=c("store_nbr") , all.x=T, all.y=T)
train = merge(x = train,y = weather, by=c("station_nbr","date") , all.x=T, all.y=F)

train.sold = train[train$sold == 1,]
train.unsold = train[train$sold == 0,]

train.sold.mean = apply(train.sold[,-c(1,2,3,4)],2,mean)
train.sold.sd = apply(train.sold[,-c(1,2,3,4)],2,sd)

train.unsold.mean = apply(train.unsold[,-c(1,2,3,4)],2,mean)
train.unsold.sd = apply(train.unsold[,-c(1,2,3,4)],2,sd)

train.sold.mean
train.sold.sd
train.unsold.mean
train.unsold.sd

train.sd = apply(train[,-c(1,2,3,4)],2,mean)
z.score = abs( (train.sold.mean-train.unsold.mean)/train.sd )
z.score = sort(z.score)

## best combinations of store / product 
store.product.sold = ddply(train[,c(2,3,4,5)], .(store_nbr,item_nbr) , function(x) c(units = mean(x$units)) ) 
store.product.sold = store.product.sold[order(store.product.sold$units , decreasing = T),]
head(store.product.sold , n = 15)

## most sold products
product.sold = ddply(train[,c(2,4,5)], .(item_nbr) , function(x) c(units = mean(x$units)) ) 
product.sold = product.sold[order(product.sold$units , decreasing = T),]
head(product.sold , n = 15)

## best selling stations 
store.sold = ddply(train[,c(3,5)], .(store_nbr) , function(x) c(units = mean(x$units)) ) 
store.sold = store.sold[order(store.sold$units , decreasing = T),]
head(store.sold , n = 15)

## 


