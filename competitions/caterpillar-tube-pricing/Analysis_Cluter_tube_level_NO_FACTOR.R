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

getData = function() {
  ## build technical feature set 
  tube = cbind(tube_base,bom_base)
  tube = cbind(tube,spec_enc)
  dim(tube) ## 180 (encoded) technical features  
  # [1] 21198   180
  
  ## putting quote_date in data set  
  head_train_set = train_enc_date
  head_test_set = test_enc_date
  
  ## build train_set and test_set 
  train_set = merge(x = head_train_set , y = tube , by = 'tube_assembly_id' , all = F)
  test_set = merge(x = head_test_set , y = tube , by = 'tube_assembly_id' , all = F)
  
  ######### feature scaling 
  #   cat(">>> Feature scaling ... \n")
  #   feature2scal = c(
  #     "quote_date"     ,    "annual_usage"   ,     "min_order_quantity"    ,       
  #     "quantity"       ,      "wall"         ,      "length"               , "num_bends"      ,     "bend_radius"     ,    
  #     "num_boss"       ,      "num_bracket"  ,      
  #     "CP_001_weight"  ,     "CP_002_weight"  ,     "CP_003_weight"  ,     "CP_004_weight"  ,    "CP_005_weight"   ,    "CP_006_weight"    ,  
  #     "CP_007_weight"  ,     "CP_008_weight"  ,     "CP_009_weight"  ,    "CP_010_weight"   ,    "CP_011_weight"   ,    "CP_012_weight"    ,  
  #     "CP_014_weight"  ,     "CP_015_weight"  ,     "CP_016_weight"  ,     "CP_017_weight"  ,     "CP_018_weight"  ,     "CP_019_weight"   ,   
  #     "CP_020_weight"  ,     "CP_021_weight"  ,     "CP_022_weight"  ,     "CP_023_weight"  ,     "CP_024_weight"  ,     "CP_025_weight"   ,   
  #     "CP_026_weight"  ,     "CP_027_weight"  ,      "CP_028_weight" ,      "CP_029_weight" ,      "OTHER_weight"  
  #   )
  #   
  #   trans.scal <- preProcess(rbind(train_set[,feature2scal],test_set[,feature2scal]),
  #                            method = c("center", "scale") )
  #   
  #   print(trans.scal)
  #   
  #   train_set[,feature2scal] = predict(trans.scal,train_set[,feature2scal])
  #   test_set[,feature2scal] = predict(trans.scal,test_set[,feature2scal])
  
  ####################
  train_set$diameterLog = log(train_set$diameter)
  train_set$diameter1_16 = (train_set$diameter)^(1/16)
  train_set$diameter1_32 = (train_set$diameter)^(1/32)
  train_set$diameter1_2 = (train_set$diameter)^(1/2)
  train_set$diameter1_4 = (train_set$diameter)^(1/4)
  train_set$diameter1_8 = (train_set$diameter)^(1/8)
  
  test_set$diameterLog = log(test_set$diameter)
  test_set$diameter1_16 = (test_set$diameter)^(1/16)
  test_set$diameter1_32 = (test_set$diameter)^(1/32)
  test_set$diameter1_2 = (test_set$diameter)^(1/2)
  test_set$diameter1_4 = (test_set$diameter)^(1/4)
  test_set$diameter1_8 = (test_set$diameter)^(1/8)
  ####################
  
  return(structure(list(
    train_set = train_set , 
    test_set = test_set 
  )))
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

sample_submission = as.data.frame( fread(paste(ff.getPath("data") , 
                                               "sample_submission.csv" , sep=''))) 

train_enc = as.data.frame( fread(paste(ff.getPath("elab") , 
                                       "train_enc.csv" , sep=''))) 

test_enc = as.data.frame( fread(paste(ff.getPath("elab") , 
                                      "test_enc.csv" , sep=''))) 

train_enc_date = as.data.frame( fread(paste(ff.getPath("elab") , 
                                            "train_enc_date.csv" , sep=''))) 

test_enc_date = as.data.frame( fread(paste(ff.getPath("elab") , 
                                           "test_enc_date.csv" , sep=''))) 

## tech props 
tube_base = as.data.frame( fread(paste(ff.getPath("elab") , 
                                       "tube_base.csv" , sep='')))

bom_base = as.data.frame( fread(paste(ff.getPath("elab") , 
                                      "bom_base.csv" , sep='')))

spec_enc = as.data.frame( fread(paste(ff.getPath("elab") , 
                                      "spec_enc.csv" , sep='')))

best_enseble = as.data.frame( fread(paste(ff.getPath("ensemble_4") , "1_ws_0_1_1_1.csv" , sep='')))

################# MAKE DATA SUITABLE FOR ANALYSIS 

dl = getData()

Xtrain = dl$train_set
Xtest = dl$test_set

### material_id 
l = ff.encodeCategoricalFeature (Xtrain$material_id , Xtest$material_id , colname.prefix = "material_id" , asNumeric=F)
Xtrain = cbind(Xtrain , l$traindata)
Xtest = cbind(Xtest , l$testdata)
Xtrain[,'material_id'] = NULL
Xtest[,'material_id'] = NULL
###

Xtrain$best_ensenble = best_enseble[1:nrow(Xtrain),]$assemble

## clustering 
cls = cluster_by(predictor.train=Xtrain$quantity,
                 predictor.test=Xtest$quantity,
                 num_bids = 8,
                 verbose=T)

Xtrain$qty_lev = cls$levels.train
Xtest$qty_lev = cls$levels.test

cls = cluster_by_diamter(predictor.train=Xtrain$diameter,
                 predictor.test=Xtest$diameter,
                 num_bids = 4,
                 verbose=T)

Xtrain$diam_lev = cls$levels.train
Xtest$diam_lev = cls$levels.test

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

Xtrain$power32 = (Xtrain$cost)^(1/32)
####corr diamter ==  0.327
#terr = 'rmsle'
#terr = 'cost'
terr = 'power32'
patt = paste0('quote_date|tube_assembly_id|best_ensenble|adiff|adiffp|',terr)
tridx = grep(pattern = patt,x = colnames(Xtrain))

aa = lapply(Xtrain[,-tridx] , function(x) {
  ret = tryCatch({ 
    getPvalueTypeIError (x=x,y=Xtrain[,terr]) 
    } , error = function(err) { 
      print(paste("ERROR:  ",err))
      
      l = list(
        test = 'PEARSON',
        pvalue=1, 
        estimate = 0, 
        interpretation = "error"
      )
      setNames(object = l , nm = names(x))
      l
    })
  return (ret)
})

aadf = data.frame(name = rep(NA,length(aa)) , test = rep(NA,length(aa)) , pvalue = rep(NA,length(aa)) , estimate = rep(NA,length(aa)) , interpretation = rep(NA,length(aa))   )
for (i in seq_along(aa)) {
  aadf[i,]$name = names(aa[i])
  aadf[i,]$test = aa[[i]]$test
  aadf[i,]$pvalue = aa[[i]]$pvalue
  aadf[i,]$estimate = aa[[i]]$estimate
  aadf[i,]$interpretation = aa[[i]]$interpretation
}

aadf = aadf[order(abs(aadf$estimate) , decreasing = T), ]













