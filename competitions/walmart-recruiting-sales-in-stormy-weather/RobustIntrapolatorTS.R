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
intrapolatePredTS <- function(train.chunk, test.chunk,doPlot=F) {
  data.chucks = rbind(train.chunk,test.chunk)
  data.chucks = data.chucks[order(data.chucks$as.date,decreasing = F),]
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

smoothSignal = function(sign) {
  #### smoth the signal 
  sign.sm = NULL 
  
  mm = mean(sign)
  
  mm.up.max = mm + 3 * sd(sign)
  mm.up.min = mm + sd(sign)
  
  mm.down.min = mm - 3 * sd(sign)
  mm.dowm.max = mm - sd(sign)
  
  iidx = sort(1:floor(mm.up.max - mm.up.min),decreasing = T)
  
  best.rmse = -1
  best.ii = -1
  
  for (ii in iidx) {
    th.up = mm.up.min+ii
    th.dowm = mm.dowm.max-ii
    ##cat(">>ii=",ii,"   th.up=",th.up,"    th.dowm=",th.dowm,"\n")
    
    sign.sm = ifelse(sign > th.up , th.up , sign)
    #sign.sm = ifelse(sign < th.dowm , th.dowm , sign)
    sign.sm[test.idx] = NA
    
    ts.all = ts(as.numeric(sign.sm) ,frequency = 365, start = c(2012,1) )
    x <- ts(sign.sm,f=4)
    fit.s <- ts(rowSums(tsSmooth(StructTS(x))[,-2]))
    tsp(fit.s) <- tsp(x)
    x[test.idx] = fit.s[test.idx]
    
    rmse.sm = RMSE(pred = as.numeric(fit.s[train.idx]), obs = x[train.idx])
    ##cat(">> rmse smooth signal:",rmse,"\n")
    
    rmse.or = RMSE(pred = as.numeric(fit.s[train.idx]), obs = sign[train.idx])
    ##cat(">> rmse smooth fit vs. orig signal:",rmse,"\n")
    
    if (best.rmse < 0 || rmse.or < best.rmse) {
      best.rmse = rmse.or
      best.ii = ii 
    }
  }
  
  list(best.rmse,best.ii,mm.up.min,mm.down.min)
}

rebuildSignal <- function(train.chunk, test.chunk,doPlot=F,st,it,pred.orig) {
  data.chucks = rbind(train.chunk,test.chunk)
  data.chucks = data.chucks[order(data.chucks$as.date,decreasing = F),]
  test.idx = which(  is.na(data.chucks$units)   )
  train.idx = which(  ! is.na(data.chucks$units)   )
  ts.all = ts(data.chucks$units ,frequency = 365, start = c(2012,1) )
  x <- ts(ts.all,f=4)
  fit <- ts(rowSums(tsSmooth(StructTS(x))[,-2]))
  tsp(fit) <- tsp(x)
  
  signal.rebuilt = x 
  signal.rebuilt[test.idx] = fit[test.idx]
  
  rmse = RMSE(pred = as.numeric(fit[train.idx]), obs = signal.rebuilt[train.idx])
  
  if (doPlot) {
    plot(signal.rebuilt)
    lines(fit,col=2)
    
    ### pred 
    signPred = signal.rebuilt
    signPred[test.idx] = pred.orig
    signPred[train.idx] = NA
    
    lines(signPred,col='green' , lwd = 4)
    
    title(paste("Station ",st, " -   Item " ,it,sep='')   )
  }
  
  list(signal.rebuilt,fit[test.idx],rmse)
}

#########
verbose = T 

source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))
source(paste0( getBasePath("process") , "/Resample_Lib.R"))
source(paste0( getBasePath("process") , "/Regression_Lib.R"))
source(paste0( getBasePath("process") , "/SelectBestPredictors_Lib.R"))

#########
train = getTrain()
test = getTest()

sub = as.data.frame( fread(paste(getBasePath("data") , 
                                    "sub_fix_37_5 2.csv" , sep='')))

sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

keys = as.data.frame( fread(paste(getBasePath("data") , 
                                  "key.csv" , sep='')))

weather = as.data.frame( fread(paste(getBasePath("data") , 
                                     "weather.imputed.full.17.8.csv" , sep=''))) 

grid =  as.data.frame( fread(paste(getBasePath("data") , 
                                   "grid_ts_models.csv" , sep='')))

stores.test = sort(unique(test$store_nbr))
items.test = sort(unique(test$item_nbr))


################
verbose = T 
################

grid = grid[!grid$all0s,]
grid = grid[order(grid$best.perf , decreasing = T),]
grid$mod =F


for (i in 1:nrow(grid)) {
  st = grid[i,]$store
  it = grid[i,]$item
  stat = keys[keys$store_nbr == st,]$station_nbr 
  cat ("\n >>>> [",i,"/",nrow(grid),"] Adjiusting st=",st," -  it=",it," ... \n")
  
  ########
  pred = NULL
  update.sub = F 
  
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
  y = traindata[,5]
  traindata = traindata[,-c(1,2,3,4,5)]
  
  ######## visulaze
  traindata.header$as.date = as.Date(traindata.header$date, "%Y-%m-%d")
  traindata.header$data_type = 1
  
  testdata.header$as.date = as.Date(testdata.header$date, "%Y-%m-%d")
  testdata.header$data_type = 2 
  
  train.chunk = traindata.header[,c(5,6,7)]
  test.chunk = testdata.header[,c(5,6)]
  test.chunk$units = NA
  
#   if (st == 37 & it ==5) {
#     cat(">> removing outlier for st = 37 and it = 5")
#     y = ifelse(y > 1000, 200, y)
#     train.chunk$units = y
#   } 
  
  #####
  id = sub[grep(x = sub$id , pattern = paste("^",st,"_",it,"_",sep='') ), ]$id
  pred.orig = sub[sub$id %in% id, ]$units 
  
  cat(">>pred.orig:",pred.orig,"\n")
  cat(">>mean(pred.orig):",mean(pred.orig),"\n")
  
  #######
  data.chucks = rbind(train.chunk,test.chunk)
  data.chucks = data.chucks[order(data.chucks$as.date,decreasing = F),]
  test.idx = which(  is.na(data.chucks$units)   )
  train.idx = which(  ! is.na(data.chucks$units)   )
  ts.all = ts(data.chucks$units ,frequency = 365, start = c(2012,1) )
  x <- ts(ts.all,f=4)
  
  fit = NULL
  fit <- tryCatch({ 
    ts(rowSums(tsSmooth(StructTS(x))[,-2]))
  } , error = function(err) { 
    print(paste("ERROR:  ",err))
    NULL
  })
  
  if (is.null(fit)) next 
  
  tsp(fit) <- tsp(x)
  
  signal.rebuilt = x 
  signal.rebuilt[test.idx] = fit[test.idx]
  
  rmse = RMSE(pred = as.numeric(fit[train.idx]), obs = signal.rebuilt[train.idx])
  
  #### plot
  #plot(signal.rebuilt)
  #lines(fit,col=2)
    
  ### pred 
  signPred = signal.rebuilt
  signPred[test.idx] = pred.orig
  signPred[train.idx] = NA
    
  #lines(signPred,col='green' , lwd = 4)
    
  #title(paste("Station ",st, " -   Item " ,it,sep='')   )
  
  list(signal.rebuilt,fit[test.idx],rmse)
  #######
  sign = signal.rebuilt
  fit.base.test = fit[test.idx]
  cat(">> fit.base.test:",fit.base.test,"\n")
  cat(">> mean(fit.base.test):",mean(fit.base.test),"\n")
  cat(">> rmse rebuilt signal:",rmse,"\n")
  
  best.rmse = best.ii = mm.up.min = mm.down.min = -1 
  
  l = tryCatch({  
    smoothSignal(sign) 
  } , error = function(err) { 
    print(paste("ERROR:  ",err))
    NULL
  })
  
  if (! is.null(l) ) { 
    best.rmse = l[[1]]
    best.ii = l[[2]]
    mm.up.min = l[[3]]
    mm.down.min = l[[4]]
  } else {
    next
  }
  
  cat(">>> best rmse:",best.rmse,"  best iidx:",best.ii,"\n")
  
  th.up = mm.up.min+best.ii
  #th.dowm = mm.dowm.max-best.ii
  sign.sm = ifelse(sign > th.up , th.up , sign)
  #sign.sm = ifelse(sign < th.dowm , th.dowm , sign)
  sign.sm[test.idx] = NA
  
  #####
  ts.all = ts(as.numeric(sign.sm) ,frequency = 365, start = c(2012,1) )
  x <- ts(sign.sm,f=4)
  fit.s <- ts(rowSums(tsSmooth(StructTS(x))[,-2]))
  tsp(fit.s) <- tsp(x)
  x[test.idx] = fit.s[test.idx]
  
  if (rmse > best.rmse) {
    cat(">>> that's better than the original signal !! \n")
    grid[i,]$mod = T
    update.sub = T 
    pred = fit.s[test.idx] 
    cat(">>> pred:\n")
    print(pred)
    cat(">>> mean<pred>:",mean(pred),"\n")
  }
  
  plot(sign,col='black')
  lines(x,col='grey')
  lines(fit,col=2)
  lines(fit.s,col="green")

  rmse = RMSE(pred = as.numeric(fit.s[train.idx]), obs = x[train.idx])
  cat(">> rmse smooth signal:",rmse,"\n")

  rmse = RMSE(pred = as.numeric(fit.s[train.idx]), obs = sign[train.idx])
  cat(">> rmse smooth fit vs. orig signal:",rmse,"\n")
  
  if (update.sub)  {
    if (verbose) cat("Updating submission ... \n")
    pred = ifelse(pred >= 0, pred , 0 )
    id = sub[grep(x = sub$id , pattern = paste("^",st,"_",it,"_",sep='') ), ]$id
    sub[sub$id %in% id, ]$units = pred 
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
          file=paste(getBasePath("data"),"sub_adjust_TSS.csv",sep='') ,
          row.names=FALSE)

write.csv(grid,quote=FALSE, 
          file=paste(getBasePath("data"),"grid_adjust_TSS.csv",sep='') ,
          row.names=FALSE)

cat("<<<<< submission correctly stored on disk >>>>>\n")

n.mod = sum(grid$mod)

cat(">>>>> done ",n.mod," updates of original submissions <<< type grid_adjust_TSS.csv to see which>>> \n")

