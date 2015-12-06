library(data.table)
library(xgboost)
library(fastfurious)
library(Hmisc)

### FUNCS 
RMSLE = function(pred, obs) {
  if (sum(pred<0)>0) {
    pred = ifelse(pred >=0 , pred , 1.5)
  }
  rmsle = sqrt(    sum( (log(pred+1) - log(obs+1))^2 )   / length(pred))
  return (rmsle)
}

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/deloitte-western-australia-rental-prices/data')
ff.bindPath(type = 'code' , sub_path = 'competitions/deloitte-western-australia-rental-prices')
ff.bindPath(type = 'elab' , sub_path = 'dataset/deloitte-western-australia-rental-prices/elab' ,  createDir = T) 

ff.bindPath(type = 'ensemble_1' , sub_path = 'dataset/deloitte-western-australia-rental-prices/ensembles/ensemble_1',createDir = T) ## out 
ff.bindPath(type = 'best_tune_1' , sub_path = 'dataset/deloitte-western-australia-rental-prices/ensembles/best_tune_1',createDir = T) ## out 
ff.bindPath(type = 'submission_1' , sub_path = 'dataset/deloitte-western-australia-rental-prices/ensembles/pred_ensemble_1',createDir = T) ## out 

ff.bindPath(type = 'ensemble_2' , sub_path = 'dataset/deloitte-western-australia-rental-prices/ensembles/ensemble_2',createDir = T) ## out 
ff.bindPath(type = 'best_tune_2' , sub_path = 'dataset/deloitte-western-australia-rental-prices/ensembles/best_tune_2',createDir = T) ## out 
ff.bindPath(type = 'submission_2' , sub_path = 'dataset/deloitte-western-australia-rental-prices/ensembles/pred_ensemble_2',createDir = T) ## out 


############################################# 

id_1 = "submit36.csv"
id_2 = "avg_Nov2_15.csv"
#id_2 = "layer1_dataProcNAs4_ytranflog_modxgbTreeGTJ_eta0.02_max_depth9_tuneTRUE.csv"


pred_1 = as.data.frame( fread(paste(ff.getPath("elab") , id_1 , sep='') , stringsAsFactors = F))
pred_2 = as.data.frame( fread(paste(ff.getPath("elab") , id_2 , sep='') , stringsAsFactors = F))
#pred_2 = as.data.frame( fread(paste(ff.getPath("submission_1") , id_2 , sep='') , stringsAsFactors = F))


###
describe(pred_1$REN_BASE_RENT)
describe(pred_2$REN_BASE_RENT)
###
p1 <- hist(pred_1$REN_BASE_RENT)                     # centered at 4
p2 <- hist(pred_2$REN_BASE_RENT)                     # centered at 6
plot( p2, col=rgb(0,0,1,1/4) ,xlim = c(0,30000))  # first histogram
plot( p1, col=rgb(1,0,0,1/4), xlim = c(0,30000) , add=T)  # second

##
delta_distribution = data.frame(max_delta = seq(from = 50,to = 1500,by = 10) , rmse = NA)

for (i in seq_along(delta_distribution$max_delta) ) { 
  MAX_DELTA = delta_distribution$max_delta[i]
  delta = abs(pred_1$REN_BASE_RENT-pred_2$REN_BASE_RENT)
  
  delta_idx = which(delta<=MAX_DELTA)
  
  pred_1_overlap = pred_1$REN_BASE_RENT[delta_idx]
  pred_2_overlap = pred_2$REN_BASE_RENT[delta_idx]
  
  pred_test = pred_1$REN_BASE_RENT
  pred_test[delta_idx] = pred_2$REN_BASE_RENT[delta_idx]
  
  rmsle_overlap = RMSLE(pred=pred_test, obs=pred_1$REN_BASE_RENT)
  
  cat(">>> MAX_DELTA:",MAX_DELTA,"--> overlapping rate: ",length(delta_idx)/length(pred_1$REN_BASE_RENT),"--> RMSLE:",rmsle_overlap,"\n")
  delta_distribution[delta_distribution$max_delta==MAX_DELTA,]$rmse = rmsle_overlap
}

plot(x = delta_distribution$max_delta,y = delta_distribution$rmse , type = "l")

RMSLE(pred = pred_2$REN_BASE_RENT , obs = pred_1$REN_BASE_RENT) ## 0.5028865
## >>> MAX_DELTA: 130 --> overlapping rate:  0.504817 --> RMSLE: 0.1280602 

####
delta_distribution_perc = data.frame(max_delta = seq(from = 0,to = 1,length.out = 400) , err_perc = NA)

for (i in seq_along(delta_distribution_perc$max_delta) ) { 
  MAX_DELTA = delta_distribution_perc$max_delta[i]
  delta = abs((pred_1$REN_BASE_RENT-pred_2$REN_BASE_RENT)/pred_1$REN_BASE_RENT)
  
  delta_idx = which(delta<=MAX_DELTA)
  
  pred_1_overlap = pred_1$REN_BASE_RENT[delta_idx]
  pred_2_overlap = pred_2$REN_BASE_RENT[delta_idx]
  
  pred_test = pred_1$REN_BASE_RENT
  pred_test[delta_idx] = pred_2$REN_BASE_RENT[delta_idx]
  
  rmsle_overlap = RMSLE(pred=pred_test, obs=pred_1$REN_BASE_RENT)
  
  cat(">>> MAX_DELTA:",MAX_DELTA,"--> overlapping rate: ",length(delta_idx)/length(pred_1$REN_BASE_RENT),"--> RMSLE:",rmsle_overlap,"\n")
  delta_distribution_perc[delta_distribution_perc$max_delta==MAX_DELTA,]$rmse = rmsle_overlap
}

plot(x = delta_distribution_perc$max_delta,y = delta_distribution_perc$err_perc , type = "l")

## >>> MAX_DELTA: 0.2982456 --> overlapping rate:  0.5010099 --> RMSLE: 0.1230421 

### 
MAX_DELTA = 0.2982456
delta = abs((pred_1$REN_BASE_RENT-pred_2$REN_BASE_RENT)/pred_1$REN_BASE_RENT)
delta_idx = which(delta<=MAX_DELTA)

pred_1_overlap = pred_1$REN_BASE_RENT[delta_idx]
pred_2_overlap = pred_2$REN_BASE_RENT[delta_idx]

pred_test = pred_1$REN_BASE_RENT
pred_test[delta_idx] = 0.6 * pred_1$REN_BASE_RENT[delta_idx] + 0.4 * pred_2$REN_BASE_RENT[delta_idx]

## pred 
stopifnot(sum(is.na(pred_avg))==0)
stopifnot(sum(pred_avg==Inf)==0)
submission = pred_1
submission$REN_BASE_RENT <- pred_test
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "merge_60_40_perc_err029.csv" , sep='') ,
          row.names=FALSE)

####
describe(pred_1$REN_BASE_RENT)
describe(pred_2$REN_BASE_RENT)
describe(submission$REN_BASE_RENT)

#### pred with 13 outlier 
pred_test = pred_1$REN_BASE_RENT
pred_test[pred_2$REN_BASE_RENT>3100] = pred_2$REN_BASE_RENT[pred_2$REN_BASE_RENT>3100]

## pred 
stopifnot(sum(is.na(pred_test))==0)
stopifnot(sum(pred_test==Inf)==0)
submission = pred_1
submission$REN_BASE_RENT <- pred_test
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "merge_Paul_13_outlier.csv" , sep='') ,
          row.names=FALSE)

### max delta tra 0.3 e 0.4 
delta = abs((pred_1$REN_BASE_RENT-pred_2$REN_BASE_RENT)/pred_1$REN_BASE_RENT)
delta_idx = which(delta<=0.4 & delta>0.3)

pred_1_overlap = pred_1$REN_BASE_RENT[delta_idx]
pred_2_overlap = pred_2$REN_BASE_RENT[delta_idx]

pred_test = pred_1$REN_BASE_RENT
pred_test[delta_idx] = 0.6 * pred_1$REN_BASE_RENT[delta_idx] + 0.4 * pred_2$REN_BASE_RENT[delta_idx]

## pred 
stopifnot(sum(is.na(pred_test))==0)
stopifnot(sum(pred_test==Inf)==0)
submission = pred_1
submission$REN_BASE_RENT <- pred_test
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "merge_60_40_perc_err03_04.csv" , sep='') ,
          row.names=FALSE)


####
cor.test(x = pred_1$REN_BASE_RENT,y = pred_2$REN_BASE_RENT,method = "pearson")




