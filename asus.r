library(xts)


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
	      as.yearmon(strptime(   '2005/1/15'                       , format = '%Y/%m/%d') )
	    ) * 12
	    
	tr = ( as.yearmon(strptime(   paste(x$year.month.repair,'/15',sep='')  , format = '%Y/%m/%d') )  - 
	      as.yearmon(strptime(   '2005/1/15'                       , format = '%Y/%m/%d') )
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
	rrow = 1
	for (mod in umodComp) {
		for (m in 0:37) {
		  cum[rrow , 1] = mod
		  cum[rrow , 2] = m
		  ## NB: consideriamo le vendite cumulate al netto dei ritiri (=sales_log < 0)
		  cum[rrow , 3] = ifelse(m>0,cum[ (rrow-1), 3] + sum(data[data$modComp == mod & data$t == m , 4  ]) , sum(data[data$modComp == mod & data$t == m , 4  ]) ) 		  
		  rrow = rrow + 1
		}
	}
	
	ret = data.frame(cum)
	colnames(ret) = c('modComp','t','cum_sales')
	ret
}

## load data set 
repTrainFn = "dataset/pakdd-cup-2014/RepairTrain.csv"
saleTrainFn = "dataset/pakdd-cup-2014/SaleTrain.csv"
outTargetIdMapFn = "dataset/pakdd-cup-2014/Output_TargetID_Mapping.csv"
#sampleSubFn = "dataset/pakdd-cup-2014/SampleSubmission.csv"

repTrain = read.csv(repTrainFn)
saleTrain = read.csv(saleTrainFn)
outTargetIdMap = read.csv(outTargetIdMapFn)
#sampleSub = read.csv(sampleSubFn)

## exploration 
es = exploreSaleTrain(saleTrain)
cs = makeCumSales(saleTrain)

er = exploreRepTrain(repTrain)

st = repMeanAndSd(repTrain)
plot(st$modComp,st$mu)
par(mfrow=c(2,1))
plot(st$modComp,st$mu)
plot(st$modComp,st$sigma)


