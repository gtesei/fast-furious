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

###### correlation among products in the same store 
stores = sort(unique(train$store_nbr))
store_nbr = 45
train.st = train[train$store_nbr == store_nbr , ]
products = sort(unique(train.st$item_nbr))
corr.st = matrix(rep(0,length(products)*length(products)),length(products),length(products))
for (i in 1:length(products)) {
  for (j in 1:length(products)) {
    if (i >= j) next 
    train.st.i = train.st[train.st$item_nbr == i,]
    train.st.j = train.st[train.st$item_nbr == j,]
    mat = merge(x = train.st.i,y = train.st.j, by=c("store_nbr","date") , all.x=F, all.y=F)
    cov = cov(mat$units.x,mat$units.y)
    if (cov == 0) {
      corr.st[i,j] = 0
    } else {
      corr.st[i,j] = cor(mat$units.x,mat$units.y)
    }
  }
}

colnames(corr.st) = as.character(products)
rownames(corr.st) = as.character(products)

library(corrgram)
corrgram(corr.st)

library(corrplot)
col1 <- colorRampPalette(c("#7F0000", "red", "#FF7F00", "yellow", "white", "cyan", 
                           "#007FFF", "blue", "#00007F"))
col2 <- colorRampPalette(c("#67001F", "#B2182B", "#D6604D", "#F4A582", "#FDDBC7", 
                           "#FFFFFF", "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC", "#053061"))
col3 <- colorRampPalette(c("red", "white", "blue"))
col4 <- colorRampPalette(c("#7F0000", "red", "#FF7F00", "yellow", "#7FFF7F", 
                           "cyan", "#007FFF", "blue", "#00007F"))
wb <- c("white", "black")
#corrplot(corr.st, method = "circle")



sold.cols = sort(unique(which(corr.st > 0) %% 111))
corrplot(corr.st[sold.cols,sold.cols], order = "hclust", addrect = 2, 
         col = col1(200) , type = "lower"    )

mm = t(corr.st[sold.cols,sold.cols]) + corr.st[sold.cols,sold.cols]
corrplot.mixed( mm , col = col4(100))

###
keys[keys$station_nbr == 16 , ]

###
train.st.88 = train.st[train.st$item_nbr == 88,]
train.st.9 = train.st[train.st$item_nbr == 9,]
cor(train.st.88$units,train.st.9$units)
cor.test(train.st.88$units, train.st.9$units, conf.level = 0.99)


train.st.14.it.9 = train.st[train.st$item_nbr == 9,]




