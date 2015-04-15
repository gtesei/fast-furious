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

getTrainClosestDates = function (testdata.header , traindata.header) {
  train.date = traindata.header$as.date
  test.date = testdata.header$as.date
  
  train.closesest = rep(as.Date("1900-01-01", "%Y-%m-%d"),length(testdata.header$as.date))
  min.diff = rep(-1,length(testdata.header$as.date))
  
  i = 1 
  while (i <= length(train.closesest) ) {
    td = test.date[i]
    
    md = min( abs(td - train.date) )
    while (! ((td-md) %in% train.date) ) md = md + 1 
    train.closesest[i] = td-md
    min.diff[i] = -1 * md
    
    i = i + 1 
    while ( (i <= length(train.closesest)) 
            & (test.date[i] == (test.date[i-1]+1))  ) {
      train.closesest[i] = train.closesest[i-1]
      min.diff[i] = min.diff[i-1]-1
      i = i + 1 
    }
  }
 list(train.closesest,min.diff)
}

getPerformance = function(pred, val) {
  res = pred - val
  MAE = sum(abs(res))/length(val)
  RSS = sum(res^2)
  MSE = RSS/length(val)
  RMSE = sqrt(MSE)
  perf = data.frame(MAE, RSS, MSE, RMSE)
}


splitTrainXvat = function(tser, perc_train) {
  ntrain = floor(length(as.vector(tser)) * perc_train)
  nval = length(as.vector(tser)) - ntrain
  
  ttrain = ts(as.vector(tser[1:ntrain]), start = start(tser), frequency = frequency(tser))
  tval = ts(as.vector(tser[ntrain + 1:nval]), start = end(ttrain) + deltat(tser), 
            frequency = frequency(tser))
  
  stopifnot(length(ttrain) == ntrain)
  stopifnot(length(tval) == nval)
  
  list(ttrain, tval)
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
                fit <- arima(x.ts, order = c(p, d, q), seas = 
                               list(order = c(P, D, Q), frequency(x.ts)), method = "CSS")
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
                                     "weather.imputed.basic.17.9.csv" , sep=''))) 

# winner.model = as.data.frame( fread(paste(getBasePath("data") , 
#                                           "mySub_grid.csv" , sep='')))
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

st = 33 ## store 33/ item 44 is the best selling combination 
it = 44 ## store 33/ item 44 is the best selling combination 

stat = keys[keys$store_nbr == st,]$station_nbr 

# cat ("winner model for store",st," - item", it, "is ",
#      winner.model[winner.model$store == st & winner.model$item == it, ]$best.model,"with RMSE on test set",
#      winner.model[winner.model$store == st & winner.model$item == it, ]$best.perf," \n")
# winner.model[winner.model$store == st & winner.model$item == it, ]
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
ww = 20
win = 1:ww
for (w in win){
  cat("processing moving average " , w , " ... ")
  f.w <- rep(1/w,w)
  y.w <- filter(traindata.header$units, f.w, sides=2)
  el = c(c(1:ceil(w/2)),c((length(y.w)-ceil(w/2)):length(y.w)))
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

#######################################################################################
sub = NULL
l = getTrainClosestDates (testdata.header , traindata.header)
train.closesest = l[[1]] 
min.diff = l[[2]]

date.struct = data.frame(test.data = testdata.header$as.date,train.closesest=train.closesest,min.diff=min.diff) 
xd = date.struct[date.struct$min.diff == -1,]$train.closesest

##
y = traindata.header[traindata.header$as.date <= xd[1],]$units

for (ii in 2:length(xd) ) {
  cat ("iteration <<",ii,"/ ",length(xd),">> \n")
  perf = NULL
  if (ii > 2) {
    more = traindata.header[traindata.header$as.date <= xd[ii] & traindata.header$as.date > xd[ii-1],]$units
    y = c(y,more)
  } 
  for (ww in 1:8) {
    
    cat ("moving average <<",ww,">> \n")
    
    my.ts = ts(y, frequency = 365, start = c(2012,1))
    test.dates = testdata.header[testdata.header$as.date <= xd[ii] & testdata.header$as.date > xd[ii-1],]
    my.ts.test = ts(rep(0,length(test.dates)), frequency = 365, start = (end(my.ts)+deltat(my.ts))  )
    
    data = splitTrainXvat(my.ts, 0.7)
    ts.train = data[[1]]
    ts.val = data[[2]]
    
    fma <- rep(1/ww,ww)
    fma
    y_ma <- filter(ts.train, fma, sides=1)
    ts.train = na.omit(y_ma)
    
    ## REG 
    regBoundle = buildLinearRegSeas(ts.train)
    mod.reg = regBoundle[[1]]
    mod.reg.2 = regBoundle[[2]]
    
    predRegBoundle = predictLinearRegSeas(ts.val, regBoundle , freq = 365)
    pred.reg = predRegBoundle[[2]]
    pred.reg.2 = predRegBoundle[[1]]
    
    ## AR
    mod.ar = ar(ts.train)
    pred.ar = predict(mod.ar, n.ahead = length(ts.val))
    
    ## ARIMA
    mod.arima <- get.best.arima(ts.train, maxord = c(2, 2, 2, 2, 2, 2))[[2]]
    pred.arima <- predict(mod.arima, n.ahead = length(ts.val))$pred
    
    ## ARIMA LOG 
    mod.arima.log <- get.best.arima(ifelse( log(ts.train) >= 0 , log(ts.train) , 0) , maxord = c(2, 2, 2, 2, 2, 2))[[2]]
    pred.arima.log <- exp(predict(mod.arima.log, n.ahead = length(ts.val))$pred)
    
#     ts.plot(my.ts, 
#             ts(pred.reg, start = start(ts.val) , frequency = frequency(ts.train)) , 
#             ts(pred.reg.2, start = start(ts.val) , frequency = frequency(ts.train)) , 
#             ts(as.numeric(pred.ar$pred), start = start(ts.val) , frequency = frequency(ts.train)) ,
#             ts(as.numeric(pred.arima), start = start(ts.val) , frequency = frequency(ts.train)) ,
#             ts(as.numeric(pred.arima.log), start = start(ts.val) , frequency = frequency(ts.train)) ,
#             col = 1:7, lty = 1:7)
#     legend("topleft", c("original serie", "Reg", "Reg.2", "AR" , "ARIMA" , "ARIMA.LOG"), lty = 1:7, 
#            col = 1:7 , cex=.5)
#     title( paste("Prediction on cross validation set (ma=",ww,")",sep='')  )
    
    perf = rbind (perf , cbind(data.frame(ma=ww) , data.frame(model="Reg") , getPerformance(pred = pred.reg, val = ts.val)) )
    perf = rbind (perf , cbind(data.frame(ma=ww) ,data.frame(model="Reg.2") , getPerformance(pred = pred.reg.2, val = ts.val)) )
    perf = rbind (perf , cbind(data.frame(ma=ww) ,data.frame(model="AR") , getPerformance(pred = as.numeric(pred.ar$pred), val = ts.val)) )
    perf = rbind (perf , cbind(data.frame(ma=ww) ,data.frame(model="ARIMA") , getPerformance(pred = as.numeric(pred.arima), val = ts.val)) )
    perf = rbind (perf , cbind(data.frame(ma=ww) ,data.frame(model="ARIMA.LOG") , getPerformance(pred = as.numeric(pred.arima.log), val = ts.val)) )
    perf = rbind (perf , cbind(data.frame(ma=ww) , data.frame(model="MEAN") , getPerformance(pred = as.numeric(mean(ts.train)), val = ts.val)) )
    
    print(perf)
    
  }
  
  print(perf)
  
  cat(">>>>>>>>>>>>>>>>>>> The winner is ....... \n")
  print(perf[perf$RMSE == min(perf$RMSE), ])
  winner.model = as.character(perf[perf$RMSE == min(perf$RMSE), ]$model)
  cat(">>  updating serie ....... \n")
  cat(">>  length before update =  ",length(y)," \n")
  
  ww.win = as.numeric(perf[perf$RMSE == min(perf$RMSE), ]$ma)
  fma <- rep(1/ww.win,ww.win)
  fma
  y_ma <- filter(my.ts, fma, sides=1)
  ts.train = na.omit(y_ma)
  
  pred.test = NULL
  
  if ( winner.model == "Reg" ) {
    regBoundle = buildLinearRegSeas(ts.train)
    mod.reg = regBoundle[[1]]
    mod.reg.2 = regBoundle[[2]]
    
    predRegBoundle = predictLinearRegSeas(my.ts.test, regBoundle , freq = 365)
    pred.reg = predRegBoundle[[2]]
    pred.reg.2 = predRegBoundle[[1]]
    
    pred.test=pred.reg
    y = c(y,pred.test)
  } else if ( winner.model == "Reg.2" ) {
    regBoundle = buildLinearRegSeas(ts.train)
    mod.reg = regBoundle[[1]]
    mod.reg.2 = regBoundle[[2]]
    
    predRegBoundle = predictLinearRegSeas(my.ts.test, regBoundle , freq = 365)
    pred.reg = predRegBoundle[[2]]
    pred.reg.2 = predRegBoundle[[1]]
    
    pred.test = pred.reg.2
    y = c(y,pred.test)
  } else if ( winner.model == "AR" ) {
    mod.ar = ar(ts.train)
    pred.ar = predict(mod.ar, n.ahead = length(my.ts.test))
    
    pred.test = as.numeric(pred.ar$pred)
    y = c(y,pred.test)
  } else if ( winner.model == "ARIMA" ) {
    mod.arima <- get.best.arima(my.ts, maxord = c(2, 2, 2, 2, 2, 2))[[2]]
    pred.arima <- predict(mod.arima, n.ahead = length(my.ts.test))$pred
    
    pred.test = as.numeric(pred.arima)
    y = c(y,pred.test)
  } else if ( winner.model == "ARIMA.LOG" ) {
    mod.arima.log <- get.best.arima(ifelse( log(ts.train) >= 0 , log(ts.train) , 0) , maxord = c(2, 2, 2, 2, 2, 2))[[2]]
    pred.arima.log <- exp(predict(mod.arima.log, n.ahead = length(my.ts.test))$pred)
    
    pred.test = as.numeric(pred.arima.log)
    y = c(y,pred.test)
  } else {  ## MEAN 
    
    pred.test = rep(as.numeric(mean(ts.train)),length(my.ts.test))
    y = c(y,pred.test) 
  }
  cat(">>  length after update =  ",length(y)," \n")
  
#   ts.plot(my.ts, ts(as.numeric(pred.test), start = start(my.ts.test) , frequency = frequency(ts.train)) ,
#           col = 1:2, lty = 1:2)
#   legend("topleft", c("train", "pred.test"), lty = 1:2, col = 1:2 , cex=.5)
#   title("Prediction on test set")
  #pacf(pred.test)
  if (is.null(sub) ) {
    sub = pred.test
  } else {
    sub = c(sub,pred.test)
  }
}

cat ("check: length sub == length test data:",(length(sub) == dim(testdata.header)[1]),"\n")
cat ("check: length y == length train data:",(length(y) == dim(traindata.header)[1]),"\n")

################################## model 
# pi = 0.1
# piT = 200
# C = data.frame( date = traindata.header$as.date , val = rep(NA,length(traindata.header$as.date)) )
# for (u in 1:length(traindata.header$as.date)) {
#   cat("u:",u,"\n")
#   dd = traindata.header[u,]$as.date
#   
#   ok = T
#   ok = ok & (length(traindata.header[traindata.header$as.date == dd,]$units) > 0)
#   ok = ok & (length(traindata.header[traindata.header$as.date == (dd-1),]$units) > 0)
#   ok = ok & (length(traindata.header[traindata.header$as.date == (dd-1-piT),]$units) > 0)
# 
#   if (! ok ) next 
#   
#   C[dd-1,]$val =   (  traindata.header[traindata.header$as.date == dd,]$units + pi * (traindata.header[traindata.header$as.date == (dd-1),]$units -  traindata.header[traindata.header$as.date == (dd-1-piT),]$units ) ) / pi
# }
################################## end model 
y = traindata.header[traindata.header$as.date <= xd[1],]$units
my.ts = ts(y, frequency = 365, start = c(2012,1))
test.dates = testdata.header[testdata.header$as.date <= xd[2] & testdata.header$as.date > xd[1],]
my.ts.test = ts(rep(NA,length(test.dates)), frequency = 365, start = (end(my.ts)+deltat(my.ts))  )
myzoo = ts(c(my.ts,my.ts.test) , frequency = 365, start = c(2012,1) )
x <- ts(myzoo,f=4)
fit <- ts(rowSums(tsSmooth(StructTS(x))[,-2]))
tsp(fit) <- tsp(x)
plot(x)
lines(fit,col=2)

### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> sembra ok e veloce 
train.chunk = traindata.header[,c(5,6,7)]
test.chunk = testdata.header[,c(5,6)]
test.chunk$units = NA
data.chucks = rbind(train.chunk,test.chunk)
data.chucks = data.chucks[order(data.chucks$as.date,decreasing = T),]
test.idx = which(  is.na(data.chucks)   )
ts.all = ts(data.chucks$units ,frequency = 365, start = c(2012,1) )
x <- ts(ts.all,f=4)
fit <- ts(rowSums(tsSmooth(StructTS(x))[,-2]))
tsp(fit) <- tsp(x)
plot(x)
lines(fit,col=2)
sub = fit[test.idx]


mySub_grid <- read.csv("~/Documents/Kaggle/fast-furious/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/mySub_grid.csv")
mySub <- read.csv("~/Documents/Kaggle/fast-furious/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/mySub.csv")

head(mySub_grid[order(mySub_grid$best.perf , decreasing = T),] , 30)


