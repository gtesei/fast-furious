library(xts)
########################### functions 
get.best.arima <- function(x.ts, maxord = c(1,1,1,1,1,1))
  {
  best.aic <- 1e8
  n <- length(x.ts)
  for (p in 0:maxord[1]) for(d in 0:maxord[2]) for(q in 0:maxord[3])
    for (P in 0:maxord[4]) for(D in 0:maxord[5]) for(Q in 0:maxord[6])
      {
      fit <- arima(x.ts, order = c(p,d,q),
                   seas = list(order = c(P,D,Q),
                               frequency(x.ts)), method = "CSS")
      fit.aic <- -2 * fit$loglik + (log(n) + 1) * length(fit$coef)
      if (fit.aic < best.aic)
        {
        
        best.aic <- fit.aic
        best.fit <- fit
        best.model <- c(p,d,q,P,D,Q)
        }
      }
  list(best.aic, best.fit, best.model)
  }

buildHarmonicModel = function(myts) {  
  Time = time(myts)
  terms = c("Time","I(Time^2)","COS[,1]","SIN[,1]",
            "COS[,2]","SIN[,2]","COS[,3]","SIN[,3]",
            "COS[,4]","SIN[,4]","COS[,5]","SIN[,5]", "COS[,6]","SIN[,6]" )
  SIN = COS = matrix(nr = length(myts) , nc = 6)
  for (i in 1:6) {
    COS[,i] = cos(2*pi*i*Time)
    SIN[,i] = sin(2*pi*i*Time)
    }
  Tscal = Time 
  mod.all = lm(myts ~ Time + I(Time^2) 
               + COS[,1] + SIN[,1] + COS[,2] + SIN[,2] 
               + COS[,3] + SIN[,3] + COS[,4] + SIN[,4] 
               + COS[,5] + SIN[,5] + COS[,6] + SIN[,6] )
  tscore = coef(mod.all) / sqrt(diag(vcov(mod.all)))
  fmla <- as.formula(paste("myts ~ " , paste(terms[abs(tscore)>2] , collapse= "+")))
  mod = lm(fmla)
  mod.res.ar = ar(resid(mod) , method="mle")
  list(mod.res.ar,mod)
} 

predictHarmonicModel = function(valts , boundle){
  mod = boundle[[2]] 
  mod.res.ar = boundle[[1]]

  Time.val = time(valts)
  SIN = COS = matrix(nr = length(valts) , nc = 6)
  for (i in 1:6) {
    COS[,i] = cos(2*pi*i*Time.val)
    SIN[,i] = sin(2*pi*i*Time.val)
    }
  new.t.scal = Time.val
  res.ar = predict( mod.res.ar , n.ahead=length(valts) )
  pred = mod$coeff[1] + mod$coeff[2] * new.t.scal +  
    mod$coeff[3] * I(new.t.scal^2) + mod$coeff[4] * SIN[, 1] + 
    mod$coeff[5] * SIN[, 2] + mod$coeff[6] * SIN[, 3] 
  pred.res.ar = as.vector(pred) + as.vector(res.ar$pred)  
  list(pred,pred.res.ar)
}

buildLinearRegSeas = function(myts){
  Time = time(myts)
  Seas = cycle(myts)
  lm = lm(myts ~ Time )
  lmSeas = lm(myts ~ 0 + Time + factor(Seas) )
  list(lmSeas,lm)
}
predictLinearRegSeas = function(valts,regBoundle) {
  lm = regBoundle[[2]]
  lmSeas = regBoundle[[1]]
  
  new.t = as.vector(time(valts))
  
  pred.lm = lm$coeff[1] + lm$coeff[2] * new.t
  beta = c(rep(coef(lmSeas)[2:13] , floor (length(valts)/12) ) , coef(lmSeas)[2:((length(valts) %% 12)+1)] )
  pred.lmSeas = lmSeas$coeff[1]*new.t + beta 
  
  list(pred.lmSeas,pred.lm)
}

compareModels = function(ts_train,ts_val,doPlot=T){
  ###### models 
  mod.ar = ar(ts_train)
  mod.hw.mul = HoltWinters(ts_train , seasonal= "mul")
  mod.hw.add = HoltWinters(ts_train , seasonal= "add")
  regBoundle = buildLinearRegSeas(ts_train)
  mod.reg = regBoundle[[1]] 
  mod.reg.2 = regBoundle[[2]]
#   harmonicBoundle = buildHarmonicModel(ts_train)
#   mod.reg.3 = harmonicBoundle[[2]]
#   mod.reg.3.res.ar = harmonicBoundle[[1]]
  mod.arima <- get.best.arima( ts_train, maxord = c(2,2,2,2,2,2))[[2]]
  mod.arima.log <- get.best.arima( log(ts_train), maxord = c(2,2,2,2,2,2))[[2]]
  
  models = c(mod.ar,mod.hw.mul,mod.hw.add,mod.reg,mod.reg.2,
#              mod.reg.3,mod.reg.3.res.ar,
             mod.arima,mod.arima.log)
  
  ###### predictions 
  pred.ar = predict( mod.ar , n.ahead=length(ts_val) )
  pred.hw.mul = predict( mod.hw.mul , n.ahead=length(ts_val) ) 
  pred.hw.add = predict( mod.hw.add , n.ahead=length(ts_val) ) 
  predRegBoundle = predictLinearRegSeas(ts_val,regBoundle)
  pred.reg = predRegBoundle[[2]]
  pred.reg.2 = predRegBoundle[[1]]
#   predHarmonicBoundle = predictHarmonicModel (ts_val , harmonicBoundle)
#   pred.reg.3 = predHarmonicBoundle[[1]]
#   pred.reg.3.res.ar = predHarmonicBoundle[[2]]
  pred.arima <- predict(mod.arima, n.ahead = length(ts_val))$pred
  pred.arima.log <- exp(predict(mod.arima.log, n.ahead = length(ts_val))$pred)
  
  ####### performance  
  perf.ar = cbind( type = c("AR") , getPerformance(as.vector(pred.ar$pred) , as.vector(ts_val)))
  perf.hw.mul = cbind( type = c("HW.mul") , getPerformance(as.vector(pred.hw.mul) , as.vector(ts_val)))
  perf.hw.add = cbind( type = c("HW.add") , getPerformance(as.vector(pred.hw.add) , as.vector(ts_val)))
  perf.reg = cbind( type = c("Reg") , getPerformance(as.vector(pred.reg) , as.vector(ts_val) ))
  perf.reg.2 = cbind( type = c("Reg.seas") , getPerformance(as.vector(pred.reg.2) , as.vector(ts_val) ))
#   perf.reg.3 = cbind( type = c("Reg.harm") , getPerformance(as.vector(pred.reg.3) , as.vector(ts_val) ))
#   perf.reg.3.res.ar = cbind( type = c("Reg.harm.res.ar") , getPerformance(as.vector(pred.reg.3.res.ar) , as.vector(ts_val) ))
  perf.arima = cbind( type = c("SARIMA") ,getPerformance(as.vector(pred.arima) , as.vector(ts_val) ) )
  perf.arima.log = cbind( type = c("SARIMA.log") ,getPerformance(as.vector(pred.arima.log) , as.vector(ts_val) ) )
  
  perf = rbind(
    perf.ar , perf.hw.mul, perf.hw.add, perf.reg, perf.reg.2, 
#     perf.reg.3, perf.reg.3.res.ar, 
    perf.arima, perf.arima.log
    )
  
  if (doPlot) {
    ts_all = ts( c(as.vector(ts_train),as.vector(ts_val)) , start=start(ts_train) , frequency=frequency(ts_train) )
    ts.plot(ts_all, 
            pred.ar$pred , 
            pred.hw.mul , 
            pred.hw.add , 
            ts(pred.reg , start=start(ts_val) , frequency=frequency(ts_all)), 
            col=1:5 , lty=1:5 )
    legend("topleft", c("TS", "AR" , "HW.mul" , "HW.add" , "Reg" ), 
           lty=1:5, col=1:5)
    
    ts.plot(ts_all , 
            ts(pred.reg.2 , start=start(ts_val) , frequency=frequency(ts_all)) , 
            #ts(pred.reg.3 , start=start(ts_val) , frequency=frequency(ts_all)) ,
            #ts(pred.reg.3.res.ar , start=start(ts_val) , frequency=frequency(ts_all)) ,
            ts(pred.arima , start=start(ts_val) , frequency=frequency(ts_all)) ,
            ts(pred.arima.log , start=start(ts_val) , frequency=frequency(ts_all)) ,
            col=1:4 , lty=1:4 )
    legend("topleft", c("TS","Reg.seas" , 
#                         "Reg.harm" , "Reg.harm.seas.ar" , 
                        "SARIMA" , "SARIMA.log"), 
           lty=1:4, col=1:4)
    }
  
  list(perf[order(perf$MAE) , ] , models[order(perf$MAE)])
}

getPerformance = function(pred , val) {
    res = pred - val
    MAE = sum(abs(res)) / length(val) 
  	RSS = sum(res^2)
    #TSS = sum(  (val-mean(val) )^2   ) 
  	MSE = RSS / length(val)
  	RMSE = sqrt(MSE)
  	#R2 = 1 - ( RSS  /  TSS )
  	
  	perf = data.frame(MAE,RSS,MSE,RMSE)
}

splitTrainXvat = function( tser , perc_train ) {
  ntrain = floor(length(as.vector(tser)) * perc_train)
  nval = length(as.vector(tser)) - ntrain
  
  ttrain = ts(as.vector(tser[1:ntrain]) , start = start(tser) , frequency=frequency(tser) )
  tval = ts(as.vector(tser[ntrain+1:nval]) , start=end(ttrain) + deltat(tser) , frequency=frequency(tser))
  
  stopifnot ( length(ttrain) == ntrain)
  stopifnot ( length(tval) == nval)
  
  list(ttrain , tval)
}
###############################

encodeSaleTrain = function(x) {
	mods = as.numeric(lapply(strsplit(as.character(x$module_category), "M"), function(x) as.numeric(x[2])))
	modMatrix = matrix(rep(0, dim(x)[1] * 9), nrow = dim(x)[1], ncol = 9)
	for (i in 1:length(mods)) {
		modMatrix[i, mods[i]] = 1
	}
	modDF = data.frame(modMatrix)
	colnames(modDF) = paste("M", 1:9, sep = "")

	comps = as.numeric(lapply(strsplit(as.character(x$component_category), "P"), function(x) as.numeric(x[2])))
	comMatrix = matrix(rep(0, dim(x)[1] * 31), nrow = dim(x)[1], ncol = 31)
	for (i in 1:length(comps)) {
		comMatrix[i, comps[i]] = 1
	}
	comDF = data.frame(comMatrix)
	colnames(comDF) = paste("P", 1:31, sep = "")

	ret = x[, -(1:2)]
	ret = data.frame(modDF, comDF, ret)
}

exploreRepTrain = function(x) {
	mods = as.numeric(lapply(strsplit(as.character(x$module_category), "M"), function(x) as.numeric(x[2])))
	comps = as.numeric(lapply(strsplit(as.character(x$component_category), "P"), function(x) as.numeric(x[2])))
	modComp = rep(0,length(comps))
	for (i in 1:length(comps)) {
		modComp[i] = as.numeric(paste(mods[i],comps[i],sep=''))
	}
	
	year_s = as.numeric(lapply(strsplit(as.character(x$year.month.sale), "/"), function(x) as.numeric(x[1])))
	month_s = as.numeric(lapply(strsplit(as.character(x$year.month.sale), "/"), function(x) as.numeric(x[2])))
	
	year_r = as.numeric(lapply(strsplit(as.character(x$year.month.repair), "/"), function(x) as.numeric(x[1])))
	month_r = as.numeric(lapply(strsplit(as.character(x$year.month.repair), "/"), function(x) as.numeric(x[2])))
	
	tt_s = as.yearmon(strptime(   paste(x$year.month.sale,'/15',sep='')  , format = '%Y/%m/%d') ) 
	tt_r = as.yearmon(strptime(   paste(x$year.month.repair,'/15',sep='')  , format = '%Y/%m/%d') ) 
	 
	ts = ( as.yearmon(strptime(   paste(x$year.month.sale,'/15',sep='')  , format = '%Y/%m/%d') )  - 
	      as.yearmon(strptime(   '2005/2/15'                       , format = '%Y/%m/%d') )
	    ) * 12
	    
	tr = ( as.yearmon(strptime(   paste(x$year.month.repair,'/15',sep='')  , format = '%Y/%m/%d') )  - 
	      as.yearmon(strptime(   '2005/2/15'                       , format = '%Y/%m/%d') )
	    ) * 12
    delta = tr-ts
	data = data.frame(x, modComp, year_s, month_s , year_r, month_r, tt_s , tt_r, ts , tr , delta)
}

repMeanAndSd = function(x) {
  
	data = exploreRepTrain(x)
	
	umodComp = unique(data$modComp)
	
	cum = matrix(  1:(length(umodComp)*3) , ncol=3 , nrow=length(umodComp) )
	rrow = 1
	for (mod in umodComp) {
		  cum[rrow , 1] = mod
		  
		  deltas = data[data$modComp==mod , 14]
		  reps = data[data$modComp==mod , 5]
		  wdeltas = deltas * reps
		  mu = sum(wdeltas) / sum(reps)
		  sigma = ifelse( sum(reps) == 1   ,  0,   sqrt( sum((wdeltas-reps*mu)^2)/(sum(reps) -1) ))
		  
		  cum[rrow , 2] = mu
		  cum[rrow , 3] = sigma
		  		  
		  rrow = rrow + 1
	}
	
	ret = data.frame(cum)
	colnames(ret) = c('modComp','mu','sigma')
	ret
}

exploreSaleTrain = function(x) {
	mods = as.numeric(lapply(strsplit(as.character(x$module_category), "M"), function(x) as.numeric(x[2])))
	comps = as.numeric(lapply(strsplit(as.character(x$component_category), "P"), function(x) as.numeric(x[2])))
	modComp = rep(0,length(comps))
	for (i in 1:length(comps)) {
		modComp[i] = as.numeric(paste(mods[i],comps[i],sep=''))
	}
	
	year = as.numeric(lapply(strsplit(as.character(x$year.month), "/"), function(x) as.numeric(x[1])))
	month = as.numeric(lapply(strsplit(as.character(x$year.month), "/"), function(x) as.numeric(x[2])))
	
	tt = as.yearmon(strptime(   paste(x$year.month,'/15',sep='')  , format = '%Y/%m/%d') ) 
	 
	t = ( as.yearmon(strptime(   paste(x$year.month,'/15',sep='')  , format = '%Y/%m/%d') )  - 
	      as.yearmon(strptime(   '2005/1/15'                       , format = '%Y/%m/%d') )
	    ) * 12
  
	data = data.frame(x, modComp, year,month , tt , t)
}

makeCumSales = function(x) {
  
	data = exploreSaleTrain(x)
	
	umodComp = unique(data$modComp)
	cum = matrix(  1:(38*length(umodComp)*3) , ncol=3 , nrow=38*length(umodComp) )

      ####init
	rrow = 1
	for (mod in umodComp) {
		for (m in 0:37) {
		  cum[rrow , 1] = mod
		  cum[rrow , 2] = m
		  cum[rrow , 3] = 0		  
		  rrow = rrow + 1
		}
	}
		
	#### fill matrix
	rrow = 1
	for (mod in umodComp) {
		for (m in 0:37) {
		  cum[rrow , 1] = mod
		  cum[rrow , 2] = m
		  ## NB: consideriamo le vendite cumulate al netto dei ritiri (=sales_log < 0)
		  ##cum[rrow , 3] = ifelse(m>0,cum[ (rrow-1), 3] + sum(data[data$modComp == mod & data$t == m , 4  ]) , sum(data[data$modComp == mod & data$t == m , 4  ]) ) 		
              cum[rrow , 3] = sum(data[data$modComp == mod & data$t <= (m+0.5) & data$t > (m-0.5) , 5  ])  		  
		  rrow = rrow + 1
		}
	}
	
	ret = data.frame(cum)
	colnames(ret) = c('modComp','t','cum_sales')
	ret
}

makeCumRepairs = function(x) {
  
	data = exploreRepTrain(x)
	
	umodComp = unique(data$modComp)
	uperiods = unique(data$year.month.repair)
	cum = matrix(  1:(length(uperiods)*length(umodComp)*3) , ncol=3 , nrow=length(uperiods)*length(umodComp) )
	
	####init
	rrow = 1
	for (mod in umodComp) {
		for (m in 0:(length(uperiods)-1)) {
		  cum[rrow , 1] = mod
		  cum[rrow , 2] = m
		  cum[rrow , 3] = 0 		  
		  rrow = rrow + 1
		}
	}
		
	#### fill matrix
	rrow = 1
	for (mod in umodComp) {
		for (m in 0:(length(uperiods)-1)) {
		  cum[rrow , 1] = mod
		  cum[rrow , 2] = m
		  cum[rrow , 3] = sum(data[data$modComp == mod & data$tr <= (m+0.5) & data$tr > (m-0.5) , 5  ])  		  
		  rrow = rrow + 1
		}
	}
	
	ret = data.frame(cum)
	colnames(ret) = c('modComp','t','cum_repairs')
	ret
}


treatOutTargetIdMap = function(x) {
	mods = as.numeric(lapply(strsplit(as.character(x$module_category), "M"), function(x) as.numeric(x[2])))
	comps = as.numeric(lapply(strsplit(as.character(x$component_category), "P"), function(x) as.numeric(x[2])))
	modComp = rep(0,length(comps))
	for (i in 1:length(comps)) {
		modComp[i] = as.numeric(paste(mods[i],comps[i],sep=''))
	}	
	 
	tt_r = as.yearmon(strptime(   paste(x$year,'/',x$month,'/15',sep='')  , format = '%Y/%m/%d') ) 
	 	    
	tr = ( as.yearmon(strptime(   paste(x$year,'/',x$month,'/15',sep='')   , format = '%Y/%m/%d') )  - 
	      as.yearmon(strptime(   '2005/1/15'                       , format = '%Y/%m/%d') )
	    ) * 12
      id = 1:(dim(x)[1])
	data = data.frame( id , x, modComp, tt_r, tr )
}




## load data set 
repTrainFn = "dataset/pakdd-cup-2014/RepairTrain.zat"
saleTrainFn = "dataset/pakdd-cup-2014/SaleTrain.zat"
outTargetIdMapFn = "dataset/pakdd-cup-2014/Output_TargetID_Mapping.zat"
sampleSubFn = "dataset/pakdd-cup-2014/SampleSubmission.zat"

repTrain = read.csv(repTrainFn)
saleTrain = read.csv(saleTrainFn)
outTargetIdMap = read.csv(outTargetIdMapFn)
sampleSub = read.csv(sampleSubFn)

## exploration 
es = exploreSaleTrain(saleTrain)
cs = makeCumSales(saleTrain)

er = exploreRepTrain(repTrain)
cr = makeCumRepairs(repTrain)

st = repMeanAndSd(repTrain)
plot(st$modComp,st$mu)
par(mfrow=c(2,1))
plot(st$modComp,st$mu)
plot(st$modComp,st$sigma)

outTargetIdMap  = treatOutTargetIdMap(outTargetIdMap)

write.csv(cr,quote=F,row.names=F,file="dataset/pakdd-cup-2014/cr.csv")
write.csv(cs,quote=F,row.names=F,file="dataset/pakdd-cup-2014/cs.csv")

##focus on mod = 224
er224 = er[er$modComp == 224,]
cr224 = cr[cr$modComp == 224,]
cs224 = cs[cs$modComp == 224,]
cr224.ts = ts(cr224$cum_repairs, start = c(2005, 2), freq = 12)
data = splitTrainXvat(cr224.ts, 0.7)
ts.train = data[[1]]
ts.val = data[[2]]
comparisons = compareModels(ts.train,ts.val,doPlot=T)
comparisons[1]

##focus on mod = 29
er29 = er[er$modComp == 29,]
cr29 = cr[cr$modComp == 29,]

cr29.ts = ts(cr29$cum_repairs, start = c(2005, 2), freq = 12)
cs29 = cs[cs$modComp == 29,]
cs29.ts = ts(cs29$cum_sales, start = c(2005, 1), freq = 12)
data = splitTrainXvat(cr29.ts, 0.7)
ts.train = data[[1]]
ts.val = data[[2]]
comparisons = compareModels(ts.train,ts.val,doPlot=T)
comparisons[1]

