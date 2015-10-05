library(fastfurious)
library(data.table)
library(plyr)
library(Hmisc)

### FUNCS 
#' Compute the average precision at k
#'
#' This function computes the average precision at k
#' between two sequences
#'
#' @param k max length of predicted sequence
#' @param actual ground truth set (vector)
#' @param predicted predicted sequence (vector)
#' @export
apk <- function(k, actual, predicted)
{
  
  if( length(actual)==0 || length(predicted)==0 ) 
  {
    return(0.0)
  }
  
  
  score <- 0.0
  cnt <- 0.0
  for (i in 1:min(k,length(predicted)))
  {
    if (predicted[i] %in% actual && !(predicted[i] %in% predicted[0:(i-1)]))
    {
      cnt <- cnt + 1
      score <- score + cnt/i 
    }
  }
  score <- score / min(length(actual), k)
  score
}
#' Compute the mean average precision at k
#'
#' This function computes the mean average precision at k
#' of two lists of sequences.
#'
#' @param k max length of predicted sequence
#' @param actual list of ground truth sets (vectors)
#' @param predicted list of predicted sequences (vectors)
#' @export
mapk <- function (k, actual, predicted)
{
  if( length(actual)==0 || length(predicted)==0 ) 
  {
    return(0.0)
  }
  
  scores <- rep(0, length(actual))
  for (i in 1:length(scores))
  {
    scores[i] <- apk( k, unlist(actual[[i]]), unlist(predicted[[i]]) )
  }
  score <- mean(scores)
  score
}
compute_score = function(week_number,smooth_coeff,pred_prefix) {
  #week_number = 2 
  #smooth_coeff = 0 
  #pred_prefix = 'pred_NOWP_NOPA_NOLA_'
  
  ## pred 
  fn = paste(ff.getPath("elab_pred"),pred_prefix,week_number,"_",smooth_coeff,".csv",sep='')
  pred = as.data.frame( fread( fn , stringsAsFactors = F))
  
  ## labels 
  labels = getLabels(week_number=week_number, verbose=T)$labels 
  
  cmp = merge(x = labels , y = pred , by = 'USER_ID_hash' , all = F)
  stopifnot(nrow(cmp)==nrow(pred),nrow(cmp)==nrow(labels))
  
  predList = lapply(cmp$PURCHASED_COUPONS.y , function(x){
    strsplit(x , split = ' ')
  })
  
  actList = lapply(cmp$PURCHASED_COUPONS.x , function(x){
    strsplit(x , split = ' ')
  })
  return(mapk(k=10, actual=actList, predicted=predList))
}

compute_score_family = function(pred_prefix) {
  #week_number = 2 
  #smooth_coeff = 0 
  #pred_prefix = 'pred_NOWP_NOPA_NOLA_'
  
  ret = as.data.frame(matrix(rep(NA,9*9),ncol=9))
  colnames(ret) = paste0("smooth_coeff_",seq(from = -4,to = 4,by = 1))
  
  for (wn in 1:9) {
    for (sm in seq(from = -4,to = 4,by = 1)) {
      ret[wn,(sm+5)] = compute_score(week_number=wn,smooth_coeff=sm,pred_prefix)
    }
  }
  
  return(ret)
}
  
### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'code' , sub_path = 'competitions/coupon-purchase-prediction')
ff.bindPath(type = 'data' , sub_path = 'dataset/coupon-purchase-prediction/data')

### GLOBAL CONFIG 
debug = F

### DATA
source(paste0(ff.getPath('code'),'make_coupon_vector.R'))
ff.bindPath(type = 'elab' , sub_path = 'dataset/coupon-purchase-prediction/elab' , createDir = T)
ff.bindPath(type = 'elab_train' , sub_path = 'dataset/coupon-purchase-prediction/elab/train' , createDir = T)
ff.bindPath(type = 'elab_labels' , sub_path = 'dataset/coupon-purchase-prediction/elab/labels' , createDir = T)
ff.bindPath(type = 'elab_pred' , sub_path = 'dataset/coupon-purchase-prediction/elab/pred' , createDir = T)

### PROCESSING 
pred_prefix = list()
pred_prefix[[1]] = "pred_NOWP_" # no WP , no PREF_NAME -- no model 
pred_prefix[[2]] = "pred_NOWP_NOPA_NOLA_" # no WP , no PREF_NAME , no LA -- no model 
pred_prefix[[3]] = "pred_NOPA_NOLA_" # cut only PREF_NAME -- no model 
pred_prefix[[4]] = "pred_mod_0.2_0.6_1_0.6__"
pred_prefix[[5]] = "pred_mod_0.2_1_1_1__"
pred_prefix[[6]] = "pred_equal_"
pred_prefix[[7]] = "pred_weigh_"

## legend_prefix
legend_prefix = list()
legend_prefix[[1]] = "no WP - no PREF_NAME" 
legend_prefix[[2]] = "no WP - no PREF_NAME - no LA" 
legend_prefix[[3]] = "no PREF_NAME" 
legend_prefix[[4]] = "model 0.2/0.6/1/0.6" 
legend_prefix[[5]] = "model 0.2/1/1/1" 
legend_prefix[[6]] = "equal weight" 
legend_prefix[[7]] = "weighted" 

perf = list()

cat(">>> computing scores ... \n")
for (i in seq_along(pred_prefix)) {
  cat("*************************  ",pred_prefix[[i]],"*********************************** \n")
  perf[[i]] = compute_score_family(pred_prefix = pred_prefix[[i]])
}

## plot 
cat(">>> plotting ... \n")
#for (i in 1:3) perf[[i]] = as.data.frame(matrix(rep(i,9*10),ncol=9))
colors = c("blue" , "green" , "brown" , "red" , "orange")

#par(xpd=F)
axisRange <- extendrange( unlist(perf) )
plot(x = 1:9 , y = t(perf[[1]][1]) , type='l' , col=colors[1] , xlab = "week" , 
     ylab = "MAP@10" , ylim = axisRange, 
     main="MAP@10 performances" , bty='L')

for (i in 2:9) {
  par(new=TRUE)
  plot( x = 1:9 , y = t(perf[[1]][i]),  type='l', col=colors[1],  xlab = "", ylab = "" , ylim = axisRange )  
} 

for (j in 2:length(perf)) {
  for (i in 1:9) {
    par(new=TRUE)
    plot( x = 1:9 , y = t(perf[[j]][i]),  type='l', col=colors[j],  xlab = "", ylab = "" , ylim = axisRange)   
  }
}

#par(xpd=TRUE)
legend(x=6,y=0.03,legend = legend_prefix, lty=1,col = colors)



