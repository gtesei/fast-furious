library(caret)
library(Hmisc)
library(data.table)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)
library(fBasics)

library(devtools)
install_github('caretEnsemble', 'zachmayer') #Install zach's caretEnsemble package
library(caretEnsemble)

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
  test.idx = which(  is.na(data.chucks)   )
  ts.all = ts(data.chucks$units ,frequency = 365, start = c(2012,1) )
  x <- ts(ts.all,f=4)
  fit <- ts(rowSums(tsSmooth(StructTS(x))[,-2]))
  tsp(fit) <- tsp(x)
  
  signal.rebuilt = x 
  signal.rebuilt[test.idx] = fit[test.idx]
  
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
  
  list(signal.rebuilt,fit,fit.xx)
}

#########
verbose = T 

source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))
source(paste0( getBasePath("process") , "/Resample_Lib.R"))
source(paste0( getBasePath("process") , "/Regression_Lib.R"))

#########
train = getTrain()
test = getTest()

sub = as.data.frame( fread(paste(getBasePath("data") , 
                                    "mySub_adj_10122.csv" , sep='')))

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
sub2 = sub 

###
# sub.wk = sub 
# sub.wk$st = NA
# sub.wk$it = NA 
# 
# idsplt = strsplit(sub$id,split = '_')
# sub.wk$st = as.numeric(lapply(idsplt, function(x) as.numeric(x[[1]]) ) )
# sub.wk$it = as.numeric(lapply(idsplt, function(x) as.numeric(x[[2]]) ) )
# 
# sub.group <- ddply(sub.wk, .(st,it), summarize , tot = sum(units) )
# sub.group = sub.group[order(sub.group$tot , decreasing = T),]

grid = grid[!grid$all0s,]
grid = grid[order(grid$best.perf , decreasing = T),]

num.adj = 10 
grid$choice = NA
grid = grid[1:num.adj,]

for (i in 1:num.adj) {
  st = grid[i,]$store
  it = grid[i,]$item
  stat = keys[keys$store_nbr == st,]$station_nbr 
  cat ("\n >>>> Adjiusting st=",st," -  it=",it," ... \n")
  
  ########
  pred = NULL
  pred2 = NULL
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
    
    #     update.sub = T 
    #     pred = as.numeric(fit.green)[test.idx]
    #     grid[i,]$choice = 'fit.green'
    
    ####### include trend feature <<<<<<<<<<<<<<
    l = featureSelect (traindata,testdata,featureScaling = T)
    traindata = l[[1]]
    testdata = l[[2]]
    
#     traindata$trend = as.numeric(fit.base)[-test.idx]
#     testdata$trend = fit.base[test.idx]
    
    ####
    ts=createTimeSlices(traindata.y, 
                           initialWindow = floor(36 * dim(traindata)[1]/100) , 
                           horizon = floor(12 * dim(traindata)[1]/100), 
                           fixedWindow = F)
    
    myTimeControl.bg <- trainControl(method = "timeslice", 
                                  initialWindow = floor(36 * dim(traindata)[1]/100) , 
                                  horizon = floor(12 * dim(traindata)[1]/100), 
                                  fixedWindow = TRUE , savePred=T, 
                                  index=ts$train , indexOut=ts$test )
    

    myTimeControl <- trainControl(method = "timeslice", 
                                     initialWindow = floor(36 * dim(traindata)[1]/100) , 
                                     horizon = floor(12 * dim(traindata)[1]/100), 
                                     fixedWindow = TRUE , savePred=T)
    
    
    model_list_big <- caretList(x = traindata, y = traindata.y, 
                                trControl=myTimeControl.bg,
                                methodList=c('lm')
    )
    
    ####
    model.1 <- train(y = traindata.y, x = traindata , method = "lm", trControl = myTimeControl)
    perf.1 = min(model.1$results$RMSE)
    
    model.2 <- train(y = traindata.y, x = traindata , method = "knn", preProc = c("center", "scale"), 
                     tuneGrid = data.frame(.k = 1:10),
                     trControl = myTimeControl.bg)
    perf.2 = min(model.2$results$RMSE)
    
    model.3 <- train(y = traindata.y, x = traindata,
                     method = "pls",
                     tuneGrid = expand.grid(.ncomp = 1:10) , 
                     trControl = myTimeControl.bg)
    perf.3 = min(model.3$results$RMSE)
    
    model.4 <- train(y = traindata.y, x = traindata , method = "rlm", trControl = myTimeControl.bg)
    perf.4 = min(model.4$results$RMSE)
    
    model.5 <- train(y = traindata.y, x = traindata, 
                     method = "cubist",
                     tuneGrid = expand.grid(.committees = c(1, 5, 10, 50, 75, 100),
                                            .neighbors = c(0, 1, 3, 5, 7, 9)),
                     trControl = myTimeControl.bg)
    perf.5 = min(model.5$results$RMSE)
    
    model.6 <- train(y = traindata.y, x = traindata, 
                 method = "treebag",
                 trControl = myTimeControl.bg)
    perf.6 = min(model.6$results$RMSE)
    
    model.7 <- train(y = traindata.y, x = traindata, 
                 method = "enet",
                 tuneGrid = expand.grid(.lambda = c(0, 0.01,.1,.5,.8), .fraction = seq(.05, 1, length = 30)) , 
                 trControl = myTimeControl.bg)
    perf.7 = min(model.7$results$RMSE)
    
    model.8 <- train(y = traindata.y, x = traindata, 
                 method = "svmRadial",
                 tuneLength = 15,
                 trControl = myTimeControl.bg)
    perf.8 = min(model.8$results$RMSE)
    
    #### ensemble 
    model_list_big[['knn']] <- model.2
    model_list_big[['pls']] <- model.3
    model_list_big[['rlm']] <- model.4
    model_list_big[['cubist']] <- model.5
    model_list_big[['treebag']] <- model.6
    model_list_big[['enet']] <- model.7
    model_list_big[['svmRadial']] <- model.8
    
    #greedy <- caretEnsemble(model_list_big, iter=1000L)
    #sort(greedy$weights, decreasing=TRUE)
    #perf.greedy = as.numeric(greedy$error)
    
    linear <- caretStack(model_list_big, method='glm', trControl=myTimeControl.bg)
    perf.linear = as.numeric(linear$error$RMSE)
    
    cat("***************** PERFORMANCES [init=",grid[i,]$best.perf,"]***************** \n")
    cat("perf.1:",perf.1,"\n")
    cat("perf.2:",perf.2,"\n")
    cat("perf.3:",perf.3,"\n")
    cat("perf.4:",perf.4,"\n")
    cat("perf.5:",perf.5,"\n")
    cat("perf.6:",perf.6,"\n")
    cat("perf.7:",perf.7,"\n")
    cat("perf.8:",perf.8,"\n")
    #cat("perf.greedy:",perf.greedy,"\n")
    cat("perf.linear:",perf.linear,"\n")
    
    ### predictions
    pred.1 = as.numeric( predict(model.1 , testdata )  )
    pred.2 = as.numeric( predict(model.2 , testdata )  )
    pred.3 = as.numeric( predict(model.3 , testdata )  )
    pred.4 = as.numeric( predict(model.4 , testdata )  )
    pred.5 = as.numeric( predict(model.5 , testdata )  )
    pred.6 = as.numeric( predict(model.6 , testdata )  )
    pred.7 = as.numeric( predict(model.7 , testdata )  )
    pred.8 = as.numeric( predict(model.8 , testdata )  )
    #pred.greedy = as.numeric( predict(greedy , testdata )  )
    pred.linear = as.numeric( predict(linear , testdata )  )
    
    #### choice best predictions 
    perfs = c(perf.1,perf.2,perf.3,perf.4,perf.5,perf.6,perf.7,perf.8,perf.linear)
    idx = which(perfs == min(perfs))
    
    update.sub = T
    if (idx == 1) {
      pred = pred.1
      cat(">> won model.1 \n") 
    } else if (idx == 2) {
      pred = pred.2
      cat(">> won model.2 \n") 
    } else if (idx == 3) {
      pred = pred.3
      cat(">> won model.3 \n") 
    } else if (idx == 4) {
      pred = pred.4
      cat(">> won model.4 \n") 
    } else if (idx == 5) {
      pred = pred.5
      cat(">> won model.5 \n") 
    } else if (idx == 6) {
      pred = pred.6
      cat(">> won model.6 \n") 
    } else if (idx == 7) {
      pred = pred.7
      cat(">> won model.7 \n") 
    } else if (idx == 8) {
      pred = pred.8
      cat(">> won model.8 \n") 
    } else if (idx == 9) {
      pred = pred.linear
      cat(">> won linear ensemble \n") 
    } else stop("bad index")

  #### choice best predictions 2
  perfs = c(perf.1,perf.2,perf.3,perf.4,perf.5,perf.6,perf.7,perf.8)
  idx = which(perfs == min(perfs))
  
  update.sub = T
  if (idx == 1) {
    pred2 = pred.1
    cat(">> won model.1 \n") 
  } else if (idx == 2) {
    pred2 = pred.2
    cat(">> won model.2 \n") 
  } else if (idx == 3) {
    pred2 = pred.3
    cat(">> won model.3 \n") 
  } else if (idx == 4) {
    pred2 = pred.4
    cat(">> won model.4 \n") 
  } else if (idx == 5) {
    pred2 = pred.5
    cat(">> won model.5 \n") 
  } else if (idx == 6) {
    pred2 = pred.6
    cat(">> won model.6 \n") 
  } else if (idx == 7) {
    pred2 = pred.7
    cat(">> won model.7 \n") 
  } else if (idx == 8) {
    pred2 = pred.8
    cat(">> won model.8 \n") 
  } else stop("bad index")
    
  } else {
    ### what to to in case there's an errror in signal re-building 
    cat(">>> some problems occurred during signal rebuilding. No adjustements have been done.\n") 
  }
  
  ###### Updating submission
  if (update.sub)  {
    
    if (verbose) cat("Updating submission ... & sub2 ... \n")
    pred = ifelse(pred >= 0, pred , 0 )
    id = sub[grep(x = sub$id , pattern = paste("^",st,"_",it,"_",sep='') ), ]$id
    sub[sub$id %in% id, ]$units = pred 
    
    pred2 = ifelse(pred2 >= 0, pred2 , 0 )
    sub2[sub2$id %in% id, ]$units = pred2 
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
          file=paste(getBasePath("data"),"sub_adjust.csv",sep='') ,
          row.names=FALSE)

write.csv(sub2,quote=FALSE, 
          file=paste(getBasePath("data"),"sub_adjust_2.csv",sep='') ,
          row.names=FALSE)

write.csv(grid,quote=FALSE, 
          file=paste(getBasePath("data"),"grid_adjust.csv",sep='') ,
          row.names=FALSE)

cat("<<<<< submission correctly stored on disk >>>>>\n")



