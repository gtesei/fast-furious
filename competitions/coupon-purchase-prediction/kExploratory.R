### LIBS 
library(fastfurious)
library(data.table)
library(plyr)

### FUNCS 
wn = function (day.from = "2011-07-01",day) {
  return(1 + floor(as.numeric(as.Date(day)-as.Date(day.from))/7))
}

ff.encodeCategoricalFeature = function(data.train , 
                                       data.test , 
                                       colname.prefix, 
                                       asNumericSequence=F , 
                                       replaceWhiteSpaceInLevelsWith=NULL,
                                       levels = NULL) {
  
  stopifnot(is.atomic(data.train))
  stopifnot(is.atomic(data.test))
  
  ### assembling 
  data = c(data.test , data.train)
  
  ###
  fact_min = 1 
  fact_max = -1
  facts = NULL
  if (asNumericSequence) {
    if (! is.null(levels))
      stop("levels must bel NULL if you set up asNumericSequence to true.")
    fact_max = max(unique(data))
    fact_min = min(unique(data))
    facts = fact_min:fact_max
  } else {
    if(is.null(levels)) facts = sort(unique(data))
    else facts = levels 
    
    if (! is.null(replaceWhiteSpaceInLevelsWith) ) 
      colns = gsub(" ", replaceWhiteSpaceInLevelsWith , sort(unique(data)))
  }
  
  colns = facts
  mm = outer(data,facts,function(x,y) ifelse(x==y,1,0))
  colnames(mm) = paste(colname.prefix,"_",colns,sep='')  
  
  ##
  mm = as.data.frame(mm)
  
  ## reassembling 
  testdata = mm[1:(length(data.test)),]
  traindata = mm[((length(data.test))+1):(dim(mm)[1]),]
  
  return(list(traindata = traindata ,testdata = testdata))
}


################# FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/coupon-purchase-prediction/data')
ff.bindPath(type = 'elab' , sub_path = 'dataset/coupon-purchase-prediction/elab' , createDir = T)
ff.bindPath(type = 'process' , sub_path = 'data_process')

### DATA
coupon_detail_train = as.data.frame( fread(paste(ff.getPath("data") , "coupon_detail_train.csv" , sep='')))

### PROCS
coupon_detail_train$I_DATE_WN = wn(day.from = min(coupon_detail_train$I_DATE) , day = coupon_detail_train$I_DATE)

cat(">>> encoding week number ...\n")
l = ff.encodeCategoricalFeature(data.train = coupon_detail_train$I_DATE_WN , 
                                data.test = coupon_detail_train$I_DATE_WN , 
                                asNumericSequence=T, 
                                colname.prefix = 'I_DATE_WN')


coupon_detail_train = cbind(coupon_detail_train , l$traindata)
weeks = length(sort(unique(coupon_detail_train$I_DATE_WN)))

cat(">>> clustering by user [weeks == ",weeks,"] ...\n")
clust = ddply(coupon_detail_train , .(USER_ID_hash) , function(x) {
  ret = rep(NA,weeks)
  lapply ( 1:weeks , function(i){
    ret[i] <<- sum( x[,paste0('I_DATE_WN_',i)] )
  })
  setNames(object = ret , nm = paste0('I_DATE_WN_',1:weeks))
} )

## Max coupons sales per user/week 
cat(">>> Coupons max sales per user/week ...\n")
as = setNames(apply(X = clust[2:ncol(clust)] , MARGIN = 2 , FUN = max),1:weeks)
barplot(as, legend.text = F,
        main = "Max coupons sales per user by week", 
        col = terrain.colors(weeks),
        beside = TRUE,
        xlab = "Week",
        ylab = "Sales (units)")

text(wn(day = "2011-12-25"), 25, "Christmas" , cex=0.8 , col = 'purple' )
text(wn(day = "2011-11-26"), 28, "Thanksgiving day" , cex=0.8 , col = 'purple')
text(wn(day = "2012-06-30"), 25, "June 30th" , cex=0.8 , col = 'purple')
par(new=TRUE)
plot( x = 1:weeks , y = filter(as, rep(1, 5)),  type='l', pch=27 , col="blue", lty=6, lwd=2 , axes=FALSE , xlab = "",
      ylab = "") 

## Average coupons sales per user/week
cat(">>> Coupons average sales per user/week ...\n")
as = setNames(apply(X = clust[2:ncol(clust)] , MARGIN = 2 , FUN = mean),1:weeks)
barplot(as, legend.text = F,
        main = "Average coupons sales per user (at least 1 purchase) by week", 
        col = terrain.colors(weeks),
        beside = TRUE,
        xlab = "Week",
        ylab = "Sales (units)")
text(wn(day = "2011-12-25"), 0.2, "Christmas" , cex=0.8 , col = 'brown' )
text(wn(day = "2011-11-26"), 0.23, "Thanksgiving day" , cex=0.8 , col = 'brown')
text(wn(day = "2012-06-30"), 0.23, "June 30th" , cex=0.8 , col = 'brown')
par(new=TRUE)
plot( x = 1:weeks , y = filter(as, rep(1, 5)),  type='l', pch=27 , col="blue", lty=6, lwd=2 , axes=FALSE , xlab = "",
      ylab = "") 





