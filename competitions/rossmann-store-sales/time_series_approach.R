library(data.table)
library(xgboost)
library(fastfurious)

### FUNCS
RMSPE = function(pred, obs) {
  ignIdx = which(obs==0)
  if (length(ignIdx)>0) {
    obs = obs[-ignIdx]
    pred = pred[-ignIdx]
  }
  
  stopifnot(sum(obs==0)==0)
  
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)
  
  rmspe = sqrt(mean( ((1-pred/obs)^2) ) )
  return (rmspe)
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

doAR <- function (tr,perc_train,te,doPlot=T) {
  start_date = min(tr$asDate)
  start_date_test = min(te$asDate)
  
  ts.tr = ts(tr$Sales ,frequency = 365, start = c(2000+as.integer(format(start_date, "%y")),1)  )
  
  l = splitTrainXvat(tser=ts.tr,perc_train=perc_train)
  
  ##
  mod.ar = ar(l$ttrain)
  pred.ar = predict(mod.ar, n.ahead = length(l$tval))
  
  rmspe.xval = RMSPE(pred=as.numeric(pred.ar$pred), obs=as.numeric(l$tval))
  
  if (doPlot) {
    ts.plot(ts.tr,pred.ar$pred,col = c(1,3) , lty = c(1,3) , ylab="Sales")
  }
  
  ##
  mod.ar = ar(ts.tr)
  pred  = as.numeric(predict(mod.ar, n.ahead = nrow(te))$pred)
  
  return(list(rmspe.xval=rmspe.xval,pred=pred))
  
}

doReg <- function (tr,perc_train,te,doPlot=T,doSeas=F) {
  start_date = min(tr$asDate)
  start_date_test = min(te$asDate)
  
  ts.tr = ts(tr$Sales ,frequency = 365, start = c(2000+as.integer(format(start_date, "%y")),1)  )
  
  l = splitTrainXvat(tser=ts.tr,perc_train=perc_train)
  
  ##
  regBoundle = buildLinearRegSeas(l$ttrain)
  predRegBoundle = predictLinearRegSeas(l$tval, regBoundle,freq = 365)
  if (doSeas) {
    mod.reg = regBoundle[[1]]
    pred.xval <- predRegBoundle[[2]]
  } else {
    mod.reg = regBoundle[[2]]
    pred.xval <- predRegBoundle[[1]]
  }
  
  rmspe.xval = RMSPE(pred=pred.xval, obs=as.numeric(l$tval))
  
  if (doPlot) {
    t2= ts(pred.xval,start = start(l$tval), frequency = 365)
    ts.plot(ts.tr,t2,col = c(1,3) , lty = c(1,3) , ylab="Sales")
  }
  
  ##
  te.tr = ts(rep(NA,nrow(te)) ,frequency = 365, start = c(2000+as.integer(format(start_date, "%y")),1)  )
  regBoundle = buildLinearRegSeas(ts.tr)
  predRegBoundle = predictLinearRegSeas(te.tr, regBoundle, freq = 365)
  if (doSeas) { 
    mod <- regBoundle[[1]]
    pred <- predRegBoundle[[2]]
  } else {
    mod <- regBoundle[[2]]
    pred <- predRegBoundle[[1]]
  }
  
  return(list(rmspe.xval=rmspe.xval,pred=pred))
  
}
mod.hw <- function (tr,perc_train,te,doPlot=T,doMul=F) {
  start_date = min(tr$asDate)
  start_date_test = min(te$asDate)
  
  ts.tr = ts(tr$Sales ,frequency = 365, start = c(2000+as.integer(format(start_date, "%y")),1)  )
  
  l = splitTrainXvat(tser=ts.tr,perc_train=perc_train)
  
  ##
  if (doMul) {
    mod <- HoltWinters(l$ttrain, seasonal = "mul")
    pred.xval <- predict(mod, n.ahead = length(l$tval))
  } else {
    mod <- HoltWinters(l$ttrain, seasonal = "add")
    pred.xval <- predict(mod, n.ahead = length(l$tval))
  }
  
  rmspe.xval = RMSPE(pred=pred.xval, obs=as.numeric(l$tval))
  
  if (doPlot) {
    ts.plot(ts.tr,pred.xval,col = c(1,3) , lty = c(1,3) , ylab="Sales")
  }
  
  ##
  if (doMul) { 
    mod <- HoltWinters(ts.tr, seasonal = "mul")
    pred <- predict(mod, n.ahead = nrow(te))
  } else {
    mod <- HoltWinters(ts.tr, seasonal = "add")
    pred <- predict(mod, n.ahead = nrow(te))
  }
  
  return(list(rmspe.xval=rmspe.xval,pred=pred))
  
}
doArima <- function (tr,perc_train,te,doPlot=T,doLog=F) {
  start_date = min(tr$asDate)
  start_date_test = min(te$asDate)
  
  ts.tr = ts(tr$Sales ,frequency = 365, start = c(2000+as.integer(format(start_date, "%y")),1)  )
  
  l = splitTrainXvat(tser=ts.tr,perc_train=perc_train)
  
  ##
  if (doLog) {
    mod.arima <- get.best.arima(log(l$ttrain), maxord = c(2, 2, 2, 2, 2, 2))[[2]]
    pred.arima.xval <- exp(predict(mod.arima, n.ahead = length(l$tval))$pred)
  } else {
    mod.arima <- get.best.arima(l$ttrain, maxord = c(2, 2, 2, 2, 2, 2))[[2]]  
    pred.arima.xval <- predict(mod.arima, n.ahead = length(l$tval))$pred
  }
  
  
  rmspe.xval = RMSPE(pred=pred.arima.xval, obs=as.numeric(l$tval))
  
  if (doPlot) {
    ts.plot(ts.tr,pred.arima.xval,col = c(1,3) , lty = c(1,3) , ylab="Sales")
  }
  
  ##
  if (doLog) { 
    mod.arima <- get.best.arima(log(ts.tr), maxord = c(2, 2, 2, 2, 2, 2))[[2]]
    pred <- exp(predict(mod.arima, n.ahead = nrow(te))$pred)
  } else {
    mod.arima <- get.best.arima(ts.tr, maxord = c(2, 2, 2, 2, 2, 2))[[2]]
    pred <- predict(mod.arima, n.ahead = nrow(te))$pred  
  }
  
  return(list(rmspe.xval=rmspe.xval,pred=pred))
}

get.best.arima <- function(x.ts, maxord = c(1, 1, 1, 1, 1, 1)) {
  best.aic <- 1e+08
  best.fit <- NULL
  best.model <- NULL
  n <- length(x.ts)
  for (p in 0:maxord[1]) 
    for (d in 0:maxord[2]) 
      for (q in 0:maxord[3]) 
        for (P in 0:maxord[4]) 
          for (D in 0:maxord[5]) 
            for (Q in 0:maxord[6]) {
              
              tryCatch({
                fit <- arima(x = x.ts, order = c(p, d, q), seas = list(order = c(P, D, Q), frequency(x.ts)), method = "CSS")
                fit.aic <- -2 * fit$loglik + (log(n) + 1) * length(fit$coef)
                if (fit.aic < best.aic) {
                  best.aic <- fit.aic
                  best.fit <- fit
                  best.model <- c(p, d, q, P, D, Q)
                }
                
              }, error = function(e) {
                #print(e)
              })
            }
  list(best.aic, best.fit, best.model)
}

splitTrainXvat = function(tser, perc_train) {
  ntrain = floor(length(as.vector(tser)) * perc_train)
  nval = length(as.vector(tser)) - ntrain
  
  ttrain = ts(as.vector(tser[1:ntrain]), start = start(tser), frequency = frequency(tser))
  tval = ts(as.vector(tser[ntrain + 1:nval]), start = end(ttrain) + deltat(tser), 
            frequency = frequency(tser))
  
  stopifnot(length(ttrain) == ntrain)
  stopifnot(length(tval) == nval)
  
  list(ttrain=ttrain, tval=tval)
}

intrapolate_ts = function(tr,perc_train,te,doPlot=T) {
  start_date = min(tr$asDate)
  start_date_test = min(te$asDate)
  
  ts.tr = ts(tr$Sales ,frequency = 365, start = c(2000+as.integer(format(start_date, "%y")),1)  )
  
  l = splitTrainXvat(tser=ts.tr,perc_train=perc_train)
  ts.all = ts(   c(as.numeric(l$ttrain),rep(NA,length(l$tval))) , frequency = 365, 
                 start = c(2000+as.integer(format(start_date, "%y")),1)   )
  
  ##x <- ts(ts.all,f=frequency(ts.tr))
  x <- ts(ts.all,f=50)
  fit <- ts(rowSums(tsSmooth(StructTS(x))[,-2]))
  tsp(fit) <- tsp(x)
  
  signal.rebuilt = x 
  te_tm_idx = which(is.na(signal.rebuilt))
  signal.rebuilt[te_tm_idx] = fit[te_tm_idx]
  
  if (doPlot) {
    predts = ts(as.numeric(signal.rebuilt),start = start(ts.tr),frequency = frequency(ts.tr))
    ts.plot(ts.tr,predts,col = c(1,3) , lty = c(1,3) , ylab="Sales")
  }
  
  rmspe.xval = RMSPE(pred=signal.rebuilt[te_tm_idx], obs=as.numeric(l$tval))
  
  ## pred
  ts.all = ts(   c(as.numeric(ts.tr),rep(NA,nrow(te))) , frequency = 365, 
                 start = c(2000+as.integer(format(start_date, "%y")),1)   )
  
  #x <- ts(ts.all,f=frequency(ts.tr))
  x <- ts(ts.all,f=50)
  fit <- ts(rowSums(tsSmooth(StructTS(x))[,-2]))
  tsp(fit) <- tsp(x)
  
  signal.rebuilt = x 
  te_tm_idx = which(is.na(signal.rebuilt))
  pred = fit[te_tm_idx]
  
  return(list(rmspe.xval=rmspe.xval,pred=pred))
}

### CONF 
DO_PLOT = F 

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/rossmann-store-sales')
ff.bindPath(type = 'code' , sub_path = 'competitions/rossmann-store-sales')
ff.bindPath(type = 'elab' , sub_path = 'dataset/rossmann-store-sales/elab') 

ff.bindPath(type = 'ensembles' , sub_path = 'dataset/rossmann-store-sales/ensembles/') 

## Xtrain / Xtest / Ytrain 
cat(">>> loading Xtrain / Xtest / Ytrain  ... \n")
Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_bench.csv" , sep='') , stringsAsFactors = F))
Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_bench.csv" , sep='') , stringsAsFactors = F))
Ytrain = Xtrain$Sales
test_id = Xtest$Id

Xtrain$Sales <- Ytrain
Xtest$Sales <- NA

Xtrain$Id = 1:nrow(Xtrain)
Xtest$Id = (nrow(Xtrain)+1):(nrow(Xtrain)+nrow(Xtest))

Xtrain$asDate = as.Date( paste(paste0("20",Xtrain$year),Xtrain$month,Xtrain$day,sep='-'), "%Y-%m-%d")
Xtest$asDate = as.Date( paste(paste0("20",Xtest$year),Xtest$month,Xtest$day,sep='-'), "%Y-%m-%d")

## stores 
stores_test = sort(unique(Xtest$Store))
predTest = rep(NA,nrow(Xtest))
perf.scores = data.frame(store=stores_test,intrap=NA,arima=NA,arima.log=NA,mod.hw.mul=NA,mod.hw.add=NA,reg=NA,reg2=NA,ar=NA)
for (i in seq_along(stores_test)) {
  cat(">>> processing store ",stores_test[i]," - [",i,"/",length(stores_test),"] ...\n") 
  trIdx = which(Xtrain$Store==stores_test[i])
  teIdx = which(Xtest$Store==stores_test[i])
  
  stopifnot(max(Xtrain[trIdx,]$asDate) < min(Xtest[teIdx,]$asDate))
  
  tr = Xtrain[trIdx,,drop=F]
  tr = tr[order(tr$asDate,decreasing = F),]
  
  te = Xtest[teIdx,,drop=F]
  te = te[order(te$asDate,decreasing = F),]
  
  perc_train = floor(100*nrow(tr)/(nrow(te)+nrow(tr)))/100
  
  #   #### StructTS 
  #   te$Sales <- NA 
  #   intr = intrapolate_ts(tr=tr,perc_train=perc_train,te=te,doPlot=F)
  #   perf.scores[perf.scores$store==stores_test[i],]$intrap=intr$rmspe.xval
  #   cat(">>>        StructTS:",intr$rmspe.xval,"\n")
  #   
  #   te$Sales = intr$pred
  #   te = te[order(te$Id,decreasing = F),]
  #   pred.1 = te$Sales
  
    #### ARIMA
  #     te$Sales <- NA 
  #     arima.bag = doArima(tr=tr,perc_train=perc_train,te=te,doPlot=T,doLog=F)
  #     perf.scores[perf.scores$store==stores_test[i],]$arima=arima.bag$rmspe.xval
  #     cat(">>>        ARIMA:",arima.bag$rmspe.xval,"\n")
  #     
  #     te$Sales = arima.bag$pred
  #     te = te[order(te$Id,decreasing = F),]
  #     pred.2 = te$Sales
  #   
  #   #### ARIMA.log
  #   te$Sales <- NA 
  #   arima.log.bag = doArima(tr=tr,perc_train=perc_train,te=te,doPlot=T,doLog=T)
  #   perf.scores[perf.scores$store==stores_test[i],]$arima.log=arima.log.bag$rmspe.xval
  #   cat(">>>        ARIMA.log:",arima.log.bag$rmspe.xval,"\n")
  #   
  #   te$Sales = arima.log.bag$pred
  #   te = te[order(te$Id,decreasing = F),]
  #   pred.3 = te$Sales
  
  #### mod.hw.mul
  te$Sales <- NA 
  mod.hw.mul.bag = plyr::failwith( NULL, mod.hw , quiet = F)(tr=tr,perc_train=perc_train,te=te,doPlot=DO_PLOT,doMul=T)
  if (!is.null(mod.hw.mul.bag)) {
    perf.scores[perf.scores$store==stores_test[i],]$mod.hw.mul=mod.hw.mul.bag$rmspe.xval
    cat(">>>        mod.hw.mul:",mod.hw.mul.bag$rmspe.xval,"\n")
    
    te$Sales = mod.hw.mul.bag$pred
    te = te[order(te$Id,decreasing = F),]
    pred.4 = te$Sales
  }
  
  #### mod.hw.add
  te$Sales <- NA 
  mod.hw.add.bag = plyr::failwith( NULL,mod.hw, quiet = F)(tr=tr,perc_train=perc_train,te=te,doPlot=DO_PLOT,doMul=F)
  if (!is.null(mod.hw.add.bag)) {
    perf.scores[perf.scores$store==stores_test[i],]$mod.hw.add=mod.hw.add.bag$rmspe.xval
    cat(">>>        mod.hw.add:",mod.hw.add.bag$rmspe.xval,"\n")
    
    te$Sales = mod.hw.add.bag$pred
    te = te[order(te$Id,decreasing = F),]
    pred.5 = te$Sales
  }
  
  #### reg
  te$Sales <- NA 
  reg.bag = plyr::failwith( NULL,doReg, quiet = F)(tr=tr,perc_train=perc_train,te=te,doPlot=DO_PLOT,doSeas=F)
  if (!is.null(reg.bag)) {
    perf.scores[perf.scores$store==stores_test[i],]$reg=reg.bag$rmspe.xval
    cat(">>>        reg:",reg.bag$rmspe.xval,"\n")
    
    te$Sales = reg.bag$pred
    te = te[order(te$Id,decreasing = F),]
    pred.6 = te$Sales
  }
  
  #### reg2
  te$Sales <- NA 
  reg2.bag = plyr::failwith( NULL,doReg, quiet = F)(tr=tr,perc_train=perc_train,te=te,doPlot=DO_PLOT,doSeas=T)
  if (!is.null(reg2.bag)) {
    perf.scores[perf.scores$store==stores_test[i],]$reg2=reg2.bag$rmspe.xval
    cat(">>>        reg2:",reg2.bag$rmspe.xval,"\n")
    
    te$Sales = reg2.bag$pred
    te = te[order(te$Id,decreasing = F),]
    pred.7 = te$Sales
  }
  
  #### ar
  te$Sales <- NA 
  ar.bag = plyr::failwith( NULL,doAR, quiet = F)(tr=tr,perc_train=perc_train,te=te,doPlot=DO_PLOT)
  if (!is.null(ar.bag)) {
    perf.scores[perf.scores$store==stores_test[i],]$ar=ar.bag$rmspe.xval
    cat(">>>        ar:",ar.bag$rmspe.xval,"\n")
    
    te$Sales = ar.bag$pred
    te = te[order(te$Id,decreasing = F),]
    pred.8 = te$Sales
  }
  
  ####################
  best_score = min(perf.scores[perf.scores$store==stores_test[i],],na.rm = T)
  idx = (which(perf.scores[perf.scores$store==stores_test[i],]==best_score) - 1)
  
  cat(">>>        Winner model:",colnames(perf.scores)[idx+1]," - ",best_score,"\n")
  
  ##
  predTest[teIdx] <- get(paste0("pred.",idx))
  predTest[teIdx][predTest[teIdx]<0] <- 0 
}

## write prediction on disk 
stopifnot(sum(is.na(predTest))==0)
stopifnot(sum(predTest==Inf)==0)
submission <- data.frame(Id=test_id)
submission$Sales <- predTest
stopifnot(nrow(submission)==41088)
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file= paste(ff.getPath("elab") , "sub_StructTS.csv", sep='') ,
          row.names=FALSE)

