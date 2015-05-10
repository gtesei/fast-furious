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

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
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

rebuildSignal <- function(train.chunk, test.chunk,doPlot=F) {
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
    plot(x)
    lines(fit,col=2)
    lines(signal.rebuilt,col='brown')
  }
  
  list(signal.rebuilt,fit)
}
get.best.arima <- function(x.ts, maxord = c(1, 1, 1, 1, 1, 1)) {
  best.aic <- 1e+08
  n <- length(x.ts)
  for (p in 0:maxord[1]) 
    for (d in 0:maxord[2]) 
      for (q in 0:maxord[3]) 
        for (P in 0:maxord[4]) 
          for (D in 0:maxord[5]) 
            for (Q in 0:maxord[6]) {
              
              tryCatch({
                fit <- arima(x.ts, order = c(p, d, q), seas = list(order = c(P, 
                                                                             D, Q), frequency(x.ts)), method = "CSS")
                fit.aic <- -2 * fit$loglik + (log(n) + 1) * length(fit$coef)
                if (fit.aic < best.aic) {
                  best.aic <- fit.aic
                  best.fit <- fit
                  best.model <- c(p, d, q, P, D, Q)
                }
                
              }, error = function(e) {
                
              })
            }
  list(best.aic, best.fit, best.model)
}
buildLinearRegSeas = function(myts) {
  Time = time(myts)
  Seas = cycle(myts)
  lm = lm(myts ~ Time)
  lmSeas = lm(myts ~ 0 + Time + factor(Seas))
  list(lmSeas, lm)
}
predictLinearRegSeas = function(valts, regBoundle, freq = 12) {
  lm = regBoundle[[2]]
  lmSeas = regBoundle[[1]]
  
  new.t = as.vector(time(valts))
  
  pred.lm = lm$coeff[1] + lm$coeff[2] * new.t
  beta = c(rep(coef(lmSeas)[2:13], floor(length(valts)/freq)), coef(lmSeas)[2:((length(valts)%%freq) + 
                                                                                 1)])
  pred.lmSeas = lmSeas$coeff[1] * new.t + beta
  
  list(pred.lmSeas, pred.lm)
}

#####
Models = c("Average" , "TS") 
#####
verbose = T 
doPlot = T
source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))
source(paste0( getBasePath("process") , "/Resample_Lib.R"))
source(paste0( getBasePath("process") , "/Regression_Lib.R"))

#####
train = getTrain()
test = getTest()
keys = as.data.frame( fread(paste(getBasePath("data") , 
                                  "key.csv" , sep='')))
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

weather = as.data.frame( fread(paste(getBasePath("data") , 
                                     "weather.imputed.full.17.8.csv" , sep=''))) 

#####
sub = NULL
grid = NULL

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
      
      if (sum(traindata.y > 0) == 0) {
        cat("All units in training set are 0s ... setting prediction to all 0s ....\n")
        pred = rep(0,dim(testdata)[1])
        
        ### grid 
        .grid = data.frame(store = c(st) , 
                           item = c(it) , 
                           test.num = c(dim(testdata)[1]),
                           all0s=c(T) )
        
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
        
        
        ### prediction 
        traindata.header$as.date = as.Date(traindata.header$date, "%Y-%m-%d")
        traindata.header$data_type = 1
        
        testdata.header$as.date = as.Date(testdata.header$date, "%Y-%m-%d")
        testdata.header$data_type = 2 
        
        train.chunk = traindata.header[,c(5,6,7)]
        test.chunk = testdata.header[,c(5,6)]
        test.chunk$units = NA
        
        ##### re-build the signal with intrapolations 
        data.chucks = rbind(train.chunk,test.chunk)
        data.chucks = data.chucks[order(data.chucks$as.date,decreasing = F),]
        test.idx = which(  is.na(data.chucks)   )
        
        l = tryCatch({
          rebuildSignal(train.chunk, test.chunk,doPlot=F)
        } , error = function(err) { 
          print(paste("ERROR:  ",err))
          NULL
        })
        
        if (! is.null(l) ) {
          sign = l[[1]]
          fit.base = l[[2]]
          
          ###### models & predictions 
          pred.ar = tryCatch({
            mod.ar = ar(sign)
            predict(mod.ar, n.ahead = length(sign))
          } , error = function(err) { 
            print(paste("ERROR:  ",err))
            rep(0,length(test.idx))
          })
          
          #mod.hw.mul = HoltWinters(sign, seasonal = "mul")
          #pred.hw.mul = predict(mod.hw.mul, n.ahead = length(ts_val))
          
          pred.hw.add = tryCatch({
            mod.hw.add = HoltWinters(sign, seasonal = "add")
            predict(mod.hw.add, n.ahead = length(sign))
          } , error = function(err) { 
            print(paste("ERROR:  ",err))
            rep(0,length(test.idx))
          })
          
          pred.reg = tryCatch({
            regBoundle = buildLinearRegSeas(sign)
            #mod.reg = regBoundle[[1]]
            predRegBoundle = predictLinearRegSeas(sign, regBoundle)
            
            predRegBoundle[[2]]
          } , error = function(err) { 
            print(paste("ERROR:  ",err))
            rep(0,length(test.idx))
          })
          
          pred.reg.2 = tryCatch({
            regBoundle = buildLinearRegSeas(sign)
            #mod.reg = regBoundle[[1]]
            predRegBoundle = predictLinearRegSeas(sign, regBoundle)
            
            predRegBoundle[[1]]
          } , error = function(err) { 
            print(paste("ERROR:  ",err))
            rep(0,length(test.idx))
          })
          
          pred.arima = tryCatch({
            mod.arima <- get.best.arima(sign, maxord = c(2, 2, 2, 2, 2, 2))[[2]]
            predict(mod.arima, n.ahead = length(sign))$pred
          } , error = function(err) { 
            print(paste("ERROR:  ",err))
            rep(0,length(test.idx))
          })
          
          #mod.arima.log <- get.best.arima(log(sign), maxord = c(2, 2, 2, 2, 2, 2))[[2]]
          #pred.arima.log <- exp(predict(mod.arima.log, n.ahead = length(ts_val))$pred)
          
          pred.avg = rep(mean(sign),length(sign))
          pred.mode = rep(Mode(sign),length(sign))
          
          ###### performances 
          perf.ar = RMSE(obs = sign,pred = as.vector(pred.ar$pred) )
          perf.hw.add = RMSE(obs = sign,pred = as.vector(pred.hw.add) )
          perf.reg = RMSE(obs = sign,pred = as.vector(pred.reg) )
          perf.reg.2 = RMSE(obs = sign,pred = as.vector(pred.reg.2) )
          perf.arima = RMSE(obs = sign,pred = as.vector(pred.arima) )
          perf.base = RMSE(obs = sign,pred = as.vector(fit.base) )
          perf.avg = RMSE(obs = sign,pred = pred.avg )
          perf.mode = RMSE(obs = sign,pred = pred.mode )
          
          cat("perf.ar:",perf.ar,"\n") 
          cat("perf.hw.add:",perf.hw.add,"\n") 
          cat("perf.reg:",perf.reg,"\n") 
          cat("perf.reg.2:",perf.reg.2,"\n") 
          cat("perf.arima:",perf.arima,"\n") 
          cat("perf.base:",perf.base,"\n") 
          cat("perf.avg:",perf.avg,"\n") 
          cat("perf.mode:",perf.mode,"\n") 
          
          perfs = c(perf.ar,perf.hw.add,perf.reg,perf.reg.2,perf.arima,perf.base,perf.avg,perf.mode)
          idx.winner = which( perfs == min(perfs[!is.na(perfs)]) )
          best.perf = perfs[idx.winner]
          
          models = c("ar","hw.add","reg","reg.2","arima","base.fit","Average","Mode")
          best.model = models[idx.winner]
          
          ### prediction  
          pred = NULL
          if (idx.winner==1)  pred = as.vector(pred.ar$pred)[test.idx]
          else if (idx.winner==2)  pred = as.vector(pred.hw.add)[test.idx]
          else if (idx.winner==3)  pred = as.vector(pred.reg)[test.idx]
          else if (idx.winner==4)  pred = as.vector(pred.reg.2)[test.idx]
          else if (idx.winner==5)  pred = as.vector(pred.arima)[test.idx]
          else if (idx.winner==6)  pred = as.vector(fit.base)[test.idx]
          else if (idx.winner==7)  pred = as.vector(pred.avg)[test.idx]
          else if (idx.winner==8)  pred = as.vector(pred.mode)[test.idx]
          else stop("who did win??")
          
          ###### plot
          if (doPlot) {
            plot(sign)
            lines(as.vector(pred.ar$pred),col=2)
            lines(as.vector(pred.hw.add),col=3)
            lines(as.vector(pred.reg),col=3)
            lines(as.vector(pred.reg.2),col=4)
            lines(as.vector(pred.arima),col=5)
            lines(as.vector(fit.base),col=6)
          }
          
          ###### update grid 
          .grid$best.perf = best.perf
          .grid$best.model = best.model
          
          if(is.null(grid)) grid = .grid 
          else grid = rbind(grid,.grid)
          
        } else {
          ## can't rebuild teh signal, so just use the mean as prediction 
          pred = rep(mean(train.chunk$units),dim(testdata.header)[1])
        }
        
        ## some checks
        pred = ifelse(pred >= 0, pred , 0 )
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
          file=paste(getBasePath("data"),"sub_ts_models.csv",sep='') ,
          row.names=FALSE)

cat("<<<<< submission correctly stored on disk >>>>>\n")

## grid 
head(grid)
write.csv(grid,quote=FALSE, 
          file=paste(getBasePath("data"),"grid_ts_models.csv",sep='') ,
          row.names=FALSE)
cat("<<<<< performance grid correctly stored on disk >>>>>\n") 

