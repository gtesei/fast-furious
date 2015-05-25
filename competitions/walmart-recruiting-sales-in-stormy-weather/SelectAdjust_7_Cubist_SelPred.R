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

rebuildSignal <- function(train.chunk, test.chunk,doPlot=F,st,it) {
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
    
    ### xx 
    xx <- ts(ts.all,f=2)
    fit.xx <- ts(rowSums(tsSmooth(StructTS(xx))[,-2]))
    tsp(fit.xx) <- tsp(x)
    lines(fit.xx,col='green')
    
    title(paste("Station ",st, " -   Item " ,it,sep='')   )
  }
  
  list(signal.rebuilt,fit,fit.xx,rmse)
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

num.adj = 50
grid$choice = 'base'
grid = grid[1:num.adj,]

for (i in 1:num.adj) {
  st = grid[i,]$store
  it = grid[i,]$item
  
  stat = keys[keys$store_nbr == st,]$station_nbr 
  cat ("\n >>>> Adjiusting st=",st," -  it=",it," ... \n")
  
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
  traindata.y = traindata[,5]
  traindata = traindata[,-c(1,2,3,4,5)]
  
  ######## visulaze
  traindata.header$as.date = as.Date(traindata.header$date, "%Y-%m-%d")
  traindata.header$data_type = 1
  
  testdata.header$as.date = as.Date(testdata.header$date, "%Y-%m-%d")
  testdata.header$data_type = 2 
  
  train.chunk = traindata.header[,c(5,6,7)]
  test.chunk = testdata.header[,c(5,6)]
  test.chunk$units = NA
  
  #####
  data.chucks = rbind(train.chunk,test.chunk)
  data.chucks = data.chucks[order(data.chucks$as.date,decreasing = F),]
  test.idx = which(  is.na(data.chucks)   )
  
  ### original prediction 
  id = sub[grep(x = sub$id , pattern = paste("^",st,"_",it,"_",sep='') ), ]$id
  pred.orig = sub[sub$id %in% id, ]$units 
  
  l = tryCatch({ 
    rebuildSignal (train.chunk, test.chunk,doPlot=T,st=st,it=it)
  } , error = function(err) { 
    print(paste("ERROR:  ",err))
    NULL
  })
  
  if (! is.null(l) ) { 
    sign = l[[1]]
    fit.base = l[[2]]
    fit.green = l[[3]]
    rmse = l[[4]]
    cat(">> rmse from rebuilding signal:",rmse,"\n")
    
    #     update.sub = T 
    #     pred = as.numeric(fit.green)[test.idx]
    #     grid[i,]$choice = 'fit.green'
    
    
    stop("hhhh")
    
    ####### include trend feature <<<<<<<<<<<<<<
    l = featureSelect (traindata,testdata,
                       
                       removeOnlyZeroVariacePredictors=T,
                       removePredictorsMakingIllConditionedSquareMatrix = F, 
                       removeHighCorrelatedPredictors = F, 
                       
                       featureScaling = T)
    traindata = l[[1]]
    testdata = l[[2]]
    
#     traindata$trend = as.numeric(fit.base)[-test.idx]
#     testdata$trend = fit.base[test.idx]
    
    ####
   if (st == 37 & it ==5) {
     cat(">> removing outlier for st = 37 and it = 5")
     traindata.y = ifelse(traindata.y > 1000, 200, traindata.y)
     next
   } 

    ts=createTimeSlices(traindata.y, 
                           initialWindow = floor(36 * dim(traindata)[1]/100) , 
                           horizon = floor(12 * dim(traindata)[1]/100), 
                           fixedWindow = F)
    
    myTimeControl.bg <- trainControl(method = "timeslice", 
                                  initialWindow = floor(36 * dim(traindata)[1]/100) , 
                                  horizon = floor(12 * dim(traindata)[1]/100), 
                                  fixedWindow = TRUE , savePred=T, 
                                  index=ts$train , indexOut=ts$test )

    #############################
    predictors = NULL
    for (i in 1:5) {
      predictors.reg.linear  = getPvalueFeatures( features = traindata, response = traindata.y , p = i , 
                                                  pValueAdjust = F, pValueAdjustMethod = 0.05, 
                                                  verbose = T)
      
      if (is.null(predictors)) predictors = predictors.reg.linear[! is.na(predictors.reg.linear$pValue),]
      else predictors = rbind(predictors, predictors.reg.linear[! is.na(predictors.reg.linear$pValue),])
    }
    
    print(predictors[predictors$is.significant,])
    form = as.Formula(paste("y",paste(as.character(predictors[predictors$is.significant,]$label) , collapse = '+'),sep=' ~ '))
    
    model.5 <- train(form, data = cbind(traindata,y=traindata.y) , 
                     method = "cubist",
                     tuneGrid = expand.grid(.committees = c(1, 5, 10, 50, 75, 100),
                                            .neighbors = c(0, 1, 3, 5, 7, 9)),
                     trControl = myTimeControl.bg)
    perf.5 = min(model.5$results$RMSE)

    cat("***************** PERFORMANCES [init=",grid[i,]$best.perf,"]***************** \n")
    cat("perf.5:",perf.5,"\n")

    pred.5 = as.numeric( predict(model.5 , testdata )  )

    #if ( perf.5 < (grid[i,]$best.perf*1.4) ) {
    if ( perf.5 < rmse *1.1) {
      cat(">> in this case Cubist model performs better, so let's change submissions ... \n ")
      update.sub = T
      pred = pred.5
      grid[i,]$choice = 'cubist'
    }
    
  } else {
    ### what to to in case there's an errror in signal re-building 
    cat(">>> some problems occurred during signal rebuilding. No adjustements have been done.\n") 
  }
  
  ###### Updating submission
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
          file=paste(getBasePath("data"),"sub_adjust_cubist__selPred.csv",sep='') ,
          row.names=FALSE)

write.csv(grid,quote=FALSE, 
          file=paste(getBasePath("data"),"grid_adjust_cubist__more_predictors.csv",sep='') ,
          row.names=FALSE)

cat("<<<<< submission correctly stored on disk >>>>>\n")

n.mod = sum(grid$choice == 'cubist')

cat(">>>>> done ",n.mod," updates of original submissions <<< type grid to see which>>> \n")



