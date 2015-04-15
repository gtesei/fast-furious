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

intrapolatePredTS <- function(train.chunk, test.chunk,doPlot=F) {
  data.chucks = rbind(train.chunk,test.chunk)
  data.chucks = data.chucks[order(data.chucks$as.date,decreasing = T),]
  test.idx = which(  is.na(data.chucks)   )
  ts.all = ts(data.chucks$units ,frequency = 365, start = c(2012,1) )
  x <- ts(ts.all,f=4)
  fit <- ts(rowSums(tsSmooth(StructTS(x))[,-2]))
  tsp(fit) <- tsp(x)
  if (doPlot) {
    plot(x)
    lines(fit,col=2)
  }
  pred = fit[test.idx]
  rmse = RMSE(pred = as.numeric(fit[!is.na(x)]) , obs = train.chunk$units)
  list(pred,rmse)
}

Models = c("Average" , "TS") 
##################
verbose = T 
doPlot = T
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

###
sub = NULL
grid = NULL

###
stores.test = sort(unique(test$store_nbr))
items.test = sort(unique(test$item_nbr))

for (st in stores.test) {
  stat = keys[keys$store_nbr == st,]$station_nbr 
  for (it in items.test) {
    cat (">>>> processing stores <",st,"> - station <",stat,">- item <",it,"> ... \n") 
    if (dim(test[test$store_nbr == st & test$item_nbr == it ,  ])[1] > 0) {
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
      
      ####### checking output variable 
      if (sum(traindata.y > 0) == 0) {
        cat("All units in training set are 0s ... setting prediction to all 0s ....\n")
        pred = rep(0,dim(testdata)[1])
        
        ### grid 
        .grid = data.frame(store = c(st) , 
                           item = c(it) , 
                           test.num = c(dim(testdata)[1]),
                           all0s=c(T) )
        tmp = data.frame(matrix( 0 , 1 ,  length(Models) ))
        colnames(tmp) = Models
        .grid = cbind(.grid , tmp)
        .grid$best.perf = 0
        .grid$best.model = "Average"
        
        if(is.null(grid)) grid = .grid 
        else grid = rbind(grid,.grid)
        
      } else {
        ### grid 
        .grid = data.frame(store = c(st) , 
                           item = c(it) , 
                           test.num = c(dim(testdata)[1]),
                           all0s=c(F) )
        tmp = data.frame(matrix( 0 , 1 ,  length(Models) ))
        colnames(tmp) = Models
        .grid = cbind(.grid , tmp)
        
        
        ### prediction 
        traindata.header$as.date = as.Date(traindata.header$date, "%Y-%m-%d")
        traindata.header$data_type = 1
        
        testdata.header$as.date = as.Date(testdata.header$date, "%Y-%m-%d")
        testdata.header$data_type = 2 

        train.chunk = traindata.header[,c(5,6,7)]
        test.chunk = testdata.header[,c(5,6)]
        test.chunk$units = NA
        
        l = pred = tryCatch({ 
          intrapolatePredTS (train.chunk, test.chunk,doPlot=T)
        } , error = function(err) { 
          print(paste("ERROR:  ",err))
          NULL
        })
        
        if (! is.null(l) ) {
          pred = l[[1]]
          
          ### completing grid 
          .grid$best.perf = l[[2]]
          .grid$best.model = "TS"
          
          if(is.null(grid)) grid = .grid 
          else grid = rbind(grid,.grid)
        } else {
          pred = rep(mean(train.chunk$units),dim(testdata.header)[1])
          
          .grid$best.perf = RMSE(obs = train.chunk$units , pred = rep(mean(train.chunk$units),dim(traindata.header)[1]))
          .grid$best.model = "Average"
          
          if(is.null(grid)) grid = .grid 
          else grid = rbind(grid,.grid)
        }
        
        pred = ifelse(pred >= 0, pred , 0 )
        ## some checks
        if (length(pred) != dim(testdata.header)[1]) 
          stop("length of pred != num rows of test set!!")
      }
      ## building submission 
      if (verbose) cat("Updating submission ... \n")
      id = apply(testdata.header,1,function(x) as.character(paste(x[3],"_",x[4],"_",x[2],sep='')) )  
      sub.chunck = data.frame(id = id , units = pred)
      if (is.null(sub)) {
        sub = sub.chunck
      } else {
        sub = rbind(sub,sub.chunck)
      }
    } else {
      cat (">> no prediction needed. \n")  
    }
  }
}


### perform some checks 
if (dim(sub)[1] != dim(sampleSubmission)[1]) 
  stop (paste("sampleSubmission has ",dim(sampleSubmission)[1]," vs sub that has ",dim(sub)[1]," rows!!"))

if ( sum(!(sub$id %in% sampleSubmission$id)) > 0 ) 
  stop("sub has some ids different from sampleSubmission ids !!")

if ( sum(!(sampleSubmission$id %in% sub$id)) > 0 ) 
  stop("sampleSubmission has some ids different from sub ids !!")

### storing on disk 
write.csv(sub,quote=FALSE, 
          file=paste(getBasePath("data"),"mySubTS.csv",sep='') ,
          row.names=FALSE)

cat("<<<<< submission correctly stored on disk >>>>>\n") 

## grid 
head(grid)
write.csv(grid,quote=FALSE, 
          file=paste(getBasePath("data"),"mySubTS_grid.csv",sep='') ,
          row.names=FALSE)
cat("<<<<< performance grid correctly stored on disk >>>>>\n") 
