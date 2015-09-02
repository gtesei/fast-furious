library(data.table)
library(plyr)
library(Hmisc)
library(fastfurious)

getPvalueTypeIError = function(x,y) {
  test = NA
  pvalue = NA
  estimate = NA
  interpretation = NA 
  
  ## type casting and understanding stat test 
  if (class(x) == "integer") x = as.numeric(x)
  if (class(y) == "integer") y = as.numeric(y)
  
  if ( class(x) == "factor" & class(y) == "numeric" ) {
    # C -> Q
    test = "ANOVA"
  } else if (class(x) == "factor" & class(y) == "factor" ) {
    # C -> C
    test = "CHI-SQUARE"
  } else if (class(x) == "numeric" & class(y) == "numeric" ) {
    test = "PEARSON"
  }  else {
    # Q -> C 
    # it performs anova test x ~ y 
    test = "ANOVA"
    tmp = x 
    x = y 
    y = tmp 
  }
  
  ## performing stat test and computing p-value
  if (test == "ANOVA") {                
    test.anova = aov(y~x)
    pvalue = summary(test.anova)[[1]][["Pr(>F)"]][1]
    estimate = NULL
    if (pvalue < 0.5) {
      interpretation = 'means differ'
    } else {
      interpretation = 'data do not give you any reason to conclude that means differ'
    }
  } else if (test == "CHI-SQUARE") {    
    test.chisq = chisq.test(x = x , y = y)
    pvalue = test.chisq$p.value
    estimate = NULL
  } else {                             
    ###  PEARSON
    test.corr = cor.test(x =  x , y =  y , method = 'pearson')
    pvalue = test.corr$p.value
    estimate = test.corr$estimate
    if (pvalue < 0.5) {
      interpretation = 'there is correlation'
    } else {
      interpretation = 'data do not give you any reason to conclude that the correlation is real'
    }
  }
  
  return(list(test=test,pvalue=pvalue,estimate=estimate,interpretation=interpretation))
}

RMSLE = function(pred, obs) {
  if (sum(pred<0)>0) {
    pred = ifelse(pred >=0 , pred , 1.5)
  }
  rmsle = sqrt(    sum( (log(pred+1) - log(obs+1))^2 )   / length(pred))
  return (rmsle)
}
cluster_by = function(predictor.train,predictor.test,num_bids = 8,verbose=T) {
  
  data = as.vector(c(predictor.train,predictor.test))
  
  if (num_bids>8) {
    num_bids = 20
    split_16 = as.numeric(cut2(data, g=num_bids))
    
    return( list(levels.train = split_16[1:length(predictor.train)] , levels.test = split_16[(length(predictor.train)+1):length(data)] , 
                 theresolds = NULL) ) 
  } else {
    ## clustering by quantity 
    if (verbose) {
      print(describe(predictor.train))
      print(describe(predictor.test))
    }
    q = as.numeric(quantile(data, probs = ((1:num_bids)/num_bids)))
    
    ## counting cluster card 
    num=rep(0,num_bids)
    for (i in 1:num_bids)
      if (i == 1) {
        num[i] = sum(data<=q[i])
      } else {
        num[i] = sum(data<=q[i] & data>q[i-1])
      }
    if (verbose) print(describe(num))
    
    ## mapping quantity to cluster qty 
    qty2lev = data.frame(qty = sort(unique(data)) , lev = NA)
    for (i in 1:nrow(qty2lev)) {
      for (k in 1:length(q)) {
        if (k == 1) {
          if (qty2lev[i,]$qty <= q[1])  {
            qty2lev[i,]$lev = 1
            break
          } 
        } else {
          if (qty2lev[i,]$qty <= q[k] & qty2lev[i,]$qty > q[k-1] )  {
            qty2lev[i,]$lev = k
            break
          } 
        }
      }
    }
    
    ## mapping qty_lev on data 
    if (verbose) cat(">> mapping qty_lev to data ... \n")
    tr_qty_lev = rep(NA,length(predictor.train))
    for (i in 1:length(predictor.train))
      tr_qty_lev[i] = qty2lev[qty2lev$qty==predictor.train[i],]$lev
    
    ts_qty_lev = rep(NA,length(predictor.test))
    for (i in 1:length(predictor.test))
      ts_qty_lev[i] = qty2lev[qty2lev$qty==predictor.test[i],]$lev
    
    return( list(levels.train = tr_qty_lev , levels.test = ts_qty_lev , theresolds = q) ) 
  }
}

cluster_by_diamter = function(predictor.train,predictor.test,num_bids = 4,verbose=T) {
  
  data = as.vector(c(predictor.train,predictor.test))
  
  if (num_bids>4) {
   stop(paste0('too many bids:',num_bids))
  } else {
    ## clustering by quantity 
    if (verbose) {
      print(describe(predictor.train))
      print(describe(predictor.test))
    }
    q = as.numeric(quantile(data, probs = ((1:num_bids)/num_bids)))
    
    ## counting cluster card 
    num=rep(0,num_bids)
    for (i in 1:num_bids)
      if (i == 1) {
        num[i] = sum(data<=q[i])
      } else {
        num[i] = sum(data<=q[i] & data>q[i-1])
      }
    if (verbose) print(describe(num))
    
    ## mapping quantity to cluster qty 
    qty2lev = data.frame(qty = sort(unique(data)) , lev = NA)
    for (i in 1:nrow(qty2lev)) {
      for (k in 1:length(q)) {
        if (k == 1) {
          if (qty2lev[i,]$qty <= q[1])  {
            qty2lev[i,]$lev = 1
            break
          } 
        } else {
          if (qty2lev[i,]$qty <= q[k] & qty2lev[i,]$qty > q[k-1] )  {
            qty2lev[i,]$lev = k
            break
          } 
        }
      }
    }
    
    ## mapping qty_lev on data 
    if (verbose) cat(">> mapping qty_lev to data ... \n")
    tr_qty_lev = rep(NA,length(predictor.train))
    for (i in 1:length(predictor.train))
      tr_qty_lev[i] = qty2lev[qty2lev$qty==predictor.train[i],]$lev
    
    ts_qty_lev = rep(NA,length(predictor.test))
    for (i in 1:length(predictor.test))
      ts_qty_lev[i] = qty2lev[qty2lev$qty==predictor.test[i],]$lev
    
    return( list(levels.train = tr_qty_lev , levels.test = ts_qty_lev , theresolds = q) ) 
  }
}

perf_plot = function(Xtrain,cl=NULL) {
  
  if (is.null(cl)) Xtrain_cl = Xtrain
  else Xtrain_cl = Xtrain[Xtrain$qty_lev == cl,]
  
  rmsle_cl = RMSLE(pred = Xtrain_cl$best_ensenble,Xtrain_cl$cost)
  
  if (is.null(cl)) cat(">>> global RMSLE:",rmsle_cl,"\n")
  else cat('>>> Cluster',cl,'RMSLE:',rmsle_cl,'\n')
  
  if (is.null(cl))  ff.plotPerformance.reg(observed = Xtrain$cost,predicted = Xtrain$best_ensenble,
                                          main=paste('RMSLE:',format(x = rmsle_cl , digits = 4),' - all clusters, i.e. ',
                                                     nrow(Xtrain),' observations',sep='') )
  else  ff.plotPerformance.reg(observed = Xtrain_cl$cost,predicted = Xtrain_cl$best_ensenble,
                         main=paste('RMSLE:',format(x = rmsle_cl , digits = 4),' - cluster n.',cl,' i.e ',
                                    nrow(Xtrain_cl),' obs',sep='') )
}

plot_order_by = function(Xtrain,cl=NULL,main=NULL) {
  
  if (! is.null(cl)) Xtrain_cl = Xtrain[Xtrain$qty_lev == cl,]
  else Xtrain_cl = Xtrain
  
  Xtrain_cl = Xtrain_cl[order(Xtrain_cl$cost , decreasing = F),]
  par(mfrow=c(1,1))
  plot( x = 1:nrow(Xtrain_cl) , y = Xtrain_cl$cost , type = 'o'  , xlab ='order by cost', ylab = 'cost/absolute error' )
  lines(x = 1:nrow(Xtrain_cl) , y = Xtrain_cl$adiff ,  type = "l" , col = 'red' , ylab = 'absolute error')

  if (! is.null(main)) {
    mtext(text = main,side = 3, line = -2, outer = TRUE , cex = 1.5 , font = 2 )  
  }
}

################# FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/caterpillar-tube-pricing/competition_data')
ff.bindPath(type = 'elab' , sub_path = 'dataset/caterpillar-tube-pricing/elab')
ff.bindPath(type = 'docs' , sub_path = 'dataset/caterpillar-tube-pricing/docs')
ff.bindPath(type = 'process' , sub_path = 'data_process')

ff.bindPath(type = 'ensemble_4' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/ensemble_4') 

source(paste0( ff.getPath("process") , "/Regression_Lib.R"))

################# DATA 

sample_submission = as.data.frame( fread(paste(ff.getPath("data") , "sample_submission.csv" , sep=''))) 
train_set = as.data.frame( fread(paste(ff.getPath("data") , "train_set.csv" , sep=''))) 
test_set = as.data.frame( fread(paste(ff.getPath("data") , "test_set.csv" , sep=''))) 

tube = as.data.frame( fread(paste(ff.getPath("data") , "tube.csv" , sep='')))

best_enseble = as.data.frame( fread(paste(ff.getPath("ensemble_4") , "1_ws_0_1_1_1.csv" , sep='')))

# # bill_of_materials
# bill_of_materials = as.data.frame( fread(paste(ff.getPath("data") , "bill_of_materials.csv" , sep='')))
# components = as.data.frame( fread(paste(ff.getPath("data") , "components.csv" , sep='')))
# 
# # specs 
# specs = as.data.frame( fread(paste(ff.getPath("data") , "specs.csv" , sep='')))

################# MAKE DATA SUITABLE FOR ANALYSIS 

## tube 
tube[is.na(tube$material_id) , 'material_id'] = 'UNKNOWN' 
tube$material_id = factor(tube$material_id)

tube$end_a_1x = factor(tube$end_a_1x)
tube$end_a_2x = factor(tube$end_a_2x)
tube$end_x_1x = factor(tube$end_x_1x)
tube$end_x_2x = factor(tube$end_x_2x)
tube$end_a = factor(tube$end_a)
tube$end_x = factor(tube$end_x)

## train_set 
train_set$supplier = factor(train_set$supplier)
train_set$quote_date = as.Date(train_set$quote_date)
train_set$bracket_pricing = factor(train_set$bracket_pricing)

Xtrain = merge(x = train_set , y = tube , by = 'tube_assembly_id' , all = F)
Xtrain$best_ensenble = best_enseble[1:nrow(Xtrain),]$assemble

Xtest = merge(x = test_set , y = tube , by = 'tube_assembly_id' , all = F)

## clustering 
cls = cluster_by(predictor.train=Xtrain$quantity,
                 predictor.test=Xtest$quantity,
                 num_bids = 8,
                 verbose=T)

Xtrain$qty_lev = cls$levels.train
Xtest$qty_lev = cls$levels.test

### average performance 
perf_plot(Xtrain=Xtrain,cl=NULL)

### cluster N. 8 performance 
perf_plot(Xtrain=Xtrain,cl=8)

### cluster N. 1 performance 
perf_plot(Xtrain=Xtrain,cl=1)

### performance metrics 
Xtrain$adiff = abs(Xtrain$cost-Xtrain$best_ensenble)
Xtrain$adiffp = abs(Xtrain$cost-Xtrain$best_ensenble)/Xtrain$cost
Xtrain$rmsle = unlist(Map(f = RMSLE , pred = Xtrain$best_ensenble , obs = Xtrain$cost))

## order by cost 
plot_order_by(Xtrain=Xtrain,main='All clusters')

## cl 1 
plot_order_by(Xtrain=Xtrain,cl=1, main = 'Cluster 1')

## cl 8
plot_order_by(Xtrain=Xtrain,cl=8 , main = 'Cluster 8')

#### find the predictors most correlated with errors (adiff/adiffp/rmsle) , by cluster 
###  ESCLUDE: quote_date /  tube_assembly_id / cost / best_ensenble     
tridx = grep(pattern = 'quote_date|tube_assembly_id|cost|best_ensenble|adiff|adiffp|rmsle',x = colnames(Xtrain))

####corr diamter ==  0.327
Xtrain$diameter2 = (Xtrain$diameter)^2
Xtrain$diameter3 = (Xtrain$diameter)^3
Xtrain$diameterLog = log(Xtrain$diameter)  ###### <<<<<<<<<
Xtrain$diameter1_16 = (Xtrain$diameter)^(1/16) ###### <<<<<<<<<
Xtrain$diameter1_32 = (Xtrain$diameter)^(1/32) ###### <<<<<<<<<
Xtrain$diameter1_2 = (Xtrain$diameter)^(1/2)  ###### <<<<<<<<<<<<<<< ############# 
Xtrain$diameter1_4 = (Xtrain$diameter)^(1/4)  ###### <<<<<<<<<<<<<<< ############# 
Xtrain$diameter1_8 = (Xtrain$diameter)^(1/8)  ###### <<<<<<<<<<<<<<< ############# 

Xtrain$wall2 = (Xtrain$wall)^2
Xtrain$wall3 = (Xtrain$wall)^3
Xtrain$wallLog = log(Xtrain$wall)
Xtrain$wall1_16 = (Xtrain$wall)^(1/16)
Xtrain$wall1_32 = (Xtrain$wall)^(1/32)
Xtrain$wall1_2 = (Xtrain$wall)^(1/2)  
Xtrain$wall1_4 = (Xtrain$wall)^(1/4)
Xtrain$wall1_8 = (Xtrain$wall)^(1/8)

Xtrainnum_boss2 = (Xtrain$num_boss)^2
Xtrain$num_boss3 = (Xtrain$num_boss)^3
Xtrain$num_bossLog = NULL
Xtrain$num_boss1_16 = (Xtrain$num_boss)^(1/16)
Xtrain$num_boss1_32 = (Xtrain$num_boss)^(1/32)
Xtrain$num_boss1_2 = (Xtrain$num_boss)^(1/2)  
Xtrain$num_boss1_4 = (Xtrain$num_boss)^(1/4)
Xtrain$num_boss1_8 = (Xtrain$num_boss)^(1/8)


terr = 'rmsle'

aa = lapply(Xtrain[,-tridx] , function(x) {
  getPvalueTypeIError (x=x,y=Xtrain[,terr])
})

aadf = data.frame(name = rep(NA,length(aa)) , test = rep(NA,length(aa)) , pvalue = rep(NA,length(aa)) , estimate = rep(NA,length(aa)) , interpretation = rep(NA,length(aa))   )
for (i in seq_along(aa)) {
  aadf[i,]$name = names(aa[i])
  aadf[i,]$test = aa[[i]]$test
  aadf[i,]$pvalue = aa[[i]]$pvalue
  aadf[i,]$estimate = aa[[i]]$estimate
  aadf[i,]$interpretation = aa[[i]]$interpretation
}

aadf = aadf[order(aadf$estimate , decreasing = T), ]


### adiff 
# $ diameter          :List of 4
# ..$ test          : chr "PEARSON"
# ..$ pvalue        : num 0
# ..$ estimate      : Named num 0.308
# .. ..- attr(*, "names")= chr "cor"
# ..$ interpretation: chr "there is correlation"

# $ wall              :List of 4
# ..$ test          : chr "PEARSON"
# ..$ pvalue        : num 0
# ..$ estimate      : Named num 0.213
# .. ..- attr(*, "names")= chr "cor"
# ..$ interpretation: chr "there is correlation"

# $ num_boss          :List of 4
# ..$ test          : chr "PEARSON"
# ..$ pvalue        : num 0
# ..$ estimate      : Named num 0.168
# .. ..- attr(*, "names")= chr "cor"

# $ qty_lev           :List of 4
# ..$ test          : chr "PEARSON"
# ..$ pvalue        : num 4.86e-151
# ..$ estimate      : Named num -0.15
# .. ..- attr(*, "names")= chr "cor"
# ..$ interpretation: chr "there is correlation"

### adiffp
# $ diameter          :List of 4
# ..$ test          : chr "PEARSON"
# ..$ pvalue        : num 0
# ..$ estimate      : Named num 0.241
# .. ..- attr(*, "names")= chr "cor"
# ..$ interpretation: chr "there is correlation"

## rmsle 
# $ diameter          :List of 4
# ..$ test          : chr "PEARSON"
# ..$ pvalue        : num 0
# ..$ estimate      : Named num 0.327
# .. ..- attr(*, "names")= chr "cor"
# ..$ interpretation: chr "there is correlation"

### cluster n. 1 
# $ diameter       :List of 4
# ..$ test          : chr "PEARSON"
# ..$ pvalue        : num 0
# ..$ estimate      : Named num 0.288
# .. ..- attr(*, "names")= chr "cor"
# ..$ interpretation: chr "there is correlation"
# $ wall           :List of 4
# ..$ test          : chr "PEARSON"
# ..$ pvalue        : num 0
# ..$ estimate      : Named num 0.27
# .. ..- attr(*, "names")= chr "cor"
# ..$ interpretation: chr "there is correlation"
# $ num_boss       :List of 4
# ..$ test          : chr "PEARSON"
# ..$ pvalue        : num 0
# ..$ estimate      : Named num 0.207
# .. ..- attr(*, "names")= chr "cor"
# ..$ interpretation: chr "there is correlation"

###>>>> the winner is diameter !!
###>>>> (2) wall (3) num_boss

rmsleD = ddply(Xtrain , .(diameter) , function(x) c( rmsle=RMSLE(pred=x$best_ensenble, obs=x$cost), 
                                                     num = nrow(x)) )
rmsleD = rmsleD[order(rmsleD$rmsle,decreasing = T),]

lm1 <- lm(rmsleD$rmsle ~ rmsleD$diameter)
lm1sum <- summary(lm1)
r2 <- lm1sum$adj.r.squared
p <- lm1sum$coefficients[2, 4]
cat("p==",p,"\n")
par(mfrow=c(2,1))
plot(x = rmsleD$diameter , y = rmsleD$rmsle , xlab = 'diamter' , ylab='rmsle' , main='whole train set')
text(x = 10,y = 0.9,paste0('p=',p),cex=1, pos=4, col="green")
abline(lm1 , col = 'green', lty=1, lwd=1.5 )

# diameter      rmsle  num
# ...
# 36    76.20 0.54159382  187
# ...
# 29    50.80 0.51877344  190
# ...
# 38    88.90 0.44983292  213
# ...
# 33    63.50 0.42121419  430
# 24    38.10 0.41133163  204
# ...
# 39   101.60 0.39694190  216
# 20    31.75 0.39294634  246
# ...
# 16    25.40 0.27122052 2779
# ...
# 9     15.88 0.23760594 3400

### cluster 1 
cl =1 
Xtrain_cl = Xtrain[Xtrain$qty_lev == cl,]
rmsleD = ddply(Xtrain_cl , .(diameter) , function(x) c( rmsle=RMSLE(pred=x$best_ensenble, obs=x$cost), 
                                                     num = nrow(x)) )
lm1 <- lm(rmsleD$rmsle ~ rmsleD$diameter)
lm1sum <- summary(lm1)
r2 <- lm1sum$adj.r.squared
p <- lm1sum$coefficients[2, 4]
cat("p==",p,"\n")
plot(x = rmsleD$diameter , y = rmsleD$rmsle , xlab = 'diamter' , ylab='rmsle' , main='quantity = 1')
text(x = 10,y = 0.9,paste0('p=',p),cex=1, pos=4, col="green")
abline(lm1 , col = 'green', lty=1, lwd=1.5 )

#mtext(text = 'Correlation rmsle vs. diamter',side = 3, line = -2, outer = TRUE , cex = 1.5 , font = 2 ) 

# diameter      rmsle  num
# ...
# 25    50.80 0.56977449   98
# ...
# 34    88.90 0.49026230  130
# 29    63.50 0.46686803  225
# ...
# 13    25.40 0.34980328  671
# ...
# 8     15.88 0.27337497  847
# 7     12.70 0.25015892 1068
# 3      6.35 0.23986605 1241
# 10    19.05 0.23007531  698
# ...
# 5      9.52 0.22713190 1256


### 4 clusters 
cls4 = cluster_by_diamter(predictor.train = Xtrain$diameter,predictor.test = Xtest$diameter , num_bids = 4 , verbose = T)

cls4$theresolds
#[1]   9.52  12.70  19.05 203.20

table(cls4$levels.train)
# 1     2     3     4 
# 13095  4829  6831  5458 

table(cls4$levels.test)
# 1     2     3     4 
# 13217  4633  7030  5355

### 3 clusters 
cls3 = cluster_by_diamter(predictor.train = Xtrain$diameter,predictor.test = Xtest$diameter , num_bids = 3 , verbose = T)

cls3$theresolds
#[1]   9.52  15.88 203.20

table(cls3$levels.train)
# 1     2     3 
# 13095  8229  8889 

table(cls3$levels.test)
# 1     2     3 
# 13217  8142  8876 

### 2 clusters 
cls2 = cluster_by_diamter(predictor.train = Xtrain$diameter,predictor.test = Xtest$diameter , num_bids = 2 , verbose = T)

cls2$theresolds
#[1]    12.7 203.2

table(cls2$levels.train)
# 1     2 
# 17924 12289 

table(cls2$levels.test)
# 1     2 
# 17850 12385 

#### ordina train set per errors, taglia top 30% , bottom 30% - valuta mean diff mean predictors 
##   ordina i predittori da quelli che hanno maggior impatto nella media a quelli cge hanno impatto minore 











