library(data.table)
library(xgboost)
library(fastfurious)

### CONFIG 

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


############################################# 1 

id_1 = "layer1_dataProcNAs4_modxgbTreeGTJ_eta0.02_max_depth9_tuneTRUE.csv"
id_2 = "layer1_dataProcbase_ytranflog_modxgbTreeGTJ_eta0.02_max_depth6_tuneTRUE.csv"

ens_1 = as.data.frame( fread(paste(ff.getPath("ensemble_1") , id_1 , sep='') , stringsAsFactors = F))
ens_2 = as.data.frame( fread(paste(ff.getPath("ensemble_1") , id_2 , sep='') , stringsAsFactors = F))

assemble = (2/3) * ens_1$assemble + (1/3) * ens_2$assemble

## assemble
stopifnot(sum(is.na(assemble))==0)
stopifnot(sum(assemble==Inf)==0)
write.csv(data.frame(id = seq_along(assemble) , assemble=assemble),
          quote=FALSE, 
          file= paste(ff.getPath("ensemble_1") , "avg_Nov_15.csv" , sep='') ,
          row.names=FALSE)


## pred 
n_tr = 834562
pred = assemble[(n_tr+1):(length(assemble))]
stopifnot(sum(is.na(pred))==0)
stopifnot(sum(pred==Inf)==0)
submission = as.data.frame( fread(paste(ff.getPath("submission_1") , id_1 , sep='') , stringsAsFactors = F))
submission$REN_BASE_RENT <- pred
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "avg_Nov_15.csv" , sep='') ,
          row.names=FALSE)

######
CUTOFF = 1100
fn = "avg_Nov_15.csv"

source( paste(ff.getPath("code"),"decorator.R",sep='') )


############################################# 2 

id_1 = "layer1_dataProcNAs4_ytranflog_modxgbTreeGTJ_eta0.02_max_depth9_tuneTRUE.csv"
id_2 = "avg_Nov_15_decorated_CUTOFF1100.csv"

pred_1 = as.data.frame( fread(paste(ff.getPath("submission_1") , id_1 , sep='') , stringsAsFactors = F))
pred_2 = as.data.frame( fread(paste(ff.getPath("submission_1") , id_2 , sep='') , stringsAsFactors = F))

#pred_avg = (4/5) * pred_1$REN_BASE_RENT + (1/5) * pred_2$REN_BASE_RENT
#pred_avg = (2/3) * pred_1$REN_BASE_RENT + (1/3) * pred_2$REN_BASE_RENT
pred_avg = (3/4) * pred_1$REN_BASE_RENT + (1/4) * pred_2$REN_BASE_RENT

## pred 
stopifnot(sum(is.na(pred_avg))==0)
stopifnot(sum(pred_avg==Inf)==0)
submission = pred_1
submission$REN_BASE_RENT <- pred_avg
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "avg_Nov2_15.csv" , sep='') ,
          row.names=FALSE)

############################################# 3 - after merge 

id_1 = "submit36.csv"
id_2 = "avg_Nov2_15.csv"


pred_1 = as.data.frame( fread(paste(ff.getPath("elab") , id_1 , sep='') , stringsAsFactors = F))
pred_2 = as.data.frame( fread(paste(ff.getPath("elab") , id_2 , sep='') , stringsAsFactors = F))

## merge 
pred_merge = merge(x = pred_1 , y = pred_2 , by = "REN_ID")

### 60 - 40 
pred_merge$REN_BASE_RENT = 0.6 * pred_merge$REN_BASE_RENT.x + 0.4 * pred_merge$REN_BASE_RENT.y

## pred 
submission = pred_merge[,c("REN_ID","REN_BASE_RENT")]
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "merge_60_40.csv" , sep='') ,
          row.names=FALSE)


### 65 - 35 
pred_merge$REN_BASE_RENT = 0.65 * pred_merge$REN_BASE_RENT.x + 0.35 * pred_merge$REN_BASE_RENT.y

## pred 
submission = pred_merge[,c("REN_ID","REN_BASE_RENT")]
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "merge_65_35.csv" , sep='') ,
          row.names=FALSE)

########
id_1 = "merge_60_40.csv"
id_2 = "layer1_dataProcNAs5_ytranflog_modxgbTreeGTJ_eta0.02_max_depth9_tuneTRUE.csv"

pred_1 = as.data.frame( fread(paste(ff.getPath("elab") , id_1 , sep='') , stringsAsFactors = F))
pred_2 = as.data.frame( fread(paste(ff.getPath("elab") , id_2 , sep='') , stringsAsFactors = F))

## merge 
pred_merge = merge(x = pred_1 , y = pred_2 , by = "REN_ID")

### 60 - 40 
pred_merge$REN_BASE_RENT = 0.65 * pred_merge$REN_BASE_RENT.x + 0.35 * pred_merge$REN_BASE_RENT.y

## pred 
submission = pred_merge[,c("REN_ID","REN_BASE_RENT")]
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "merge2_65_45.csv" , sep='') ,
          row.names=FALSE)

### 50 - 40 - 10  
id_1 = "layer1_dataProcNAs5_ytranflog_modxgbTreeGTJ_eta0.02_max_depth9_tuneTRUE.csv"
id_2 = "submit36.csv"
id_3 = "avg_Nov2_15.csv"

pred_1 = as.data.frame( fread(paste(ff.getPath("elab") , id_1 , sep='') , stringsAsFactors = F))
pred_2 = as.data.frame( fread(paste(ff.getPath("elab") , id_2 , sep='') , stringsAsFactors = F))
pred_3 = as.data.frame( fread(paste(ff.getPath("elab") , id_3 , sep='') , stringsAsFactors = F))

## merge 
pred_merge = merge(x = pred_1 , y = pred_2 , by = "REN_ID")
colnames(pred_merge) <- c("REN_ID","REN_BASE_RENT.1","REN_BASE_RENT.2")
pred_merge = merge(x = pred_merge , y = pred_3 , by = "REN_ID")
colnames(pred_merge) <- c("REN_ID","REN_BASE_RENT.1","REN_BASE_RENT.2","REN_BASE_RENT.3")

### 50 - 40 - 10  
pred_merge$REN_BASE_RENT = 0.5 * pred_merge$REN_BASE_RENT.1 + 0.4 * pred_merge$REN_BASE_RENT.2 + 0.1 * pred_merge$REN_BASE_RENT.3

## pred 
submission = pred_merge[,c("REN_ID","REN_BASE_RENT")]
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "merge2_50_40_10.csv" , sep='') ,
          row.names=FALSE)

#### removing outliers 
cut_off = 3500
sum(pred_1$REN_BASE_RENT>cut_off) 
sum(pred_2$REN_BASE_RENT>cut_off)  
sum(pred_3$REN_BASE_RENT>cut_off)  
sum(pred_merge$REN_BASE_RENT>cut_off) 

pred_merge$REN_BASE_RENT[pred_merge$REN_BASE_RENT>cut_off]
pred_merge$REN_BASE_RENT.1[pred_merge$REN_BASE_RENT>cut_off]
pred_merge$REN_BASE_RENT.2[pred_merge$REN_BASE_RENT>cut_off]
pred_merge$REN_BASE_RENT.3[pred_merge$REN_BASE_RENT>cut_off]

##
pred_merge$REN_BASE_RENT[pred_merge$REN_BASE_RENT>cut_off] <- pred_merge$REN_BASE_RENT.2[pred_merge$REN_BASE_RENT>cut_off]

## pred 
submission = pred_merge[,c("REN_ID","REN_BASE_RENT")]
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "merge2_50_40_10_removeOutlierOver_3500.csv" , sep='') ,
          row.names=FALSE)


### 60 - 40 
id_1 = "layer1_dataProcNAs5_ytranflog_modxgbTreeGTJ_eta0.02_max_depth9_tuneTRUE.csv"
id_2 = "submit36.csv"

pred_1 = as.data.frame( fread(paste(ff.getPath("elab") , id_1 , sep='') , stringsAsFactors = F))
pred_2 = as.data.frame( fread(paste(ff.getPath("elab") , id_2 , sep='') , stringsAsFactors = F))

#### removing outliers 
cut_off = 5000
sum(pred_1$REN_BASE_RENT>cut_off) 
sum(pred_2$REN_BASE_RENT>cut_off)  

pred_merge = merge(x = pred_1 , y = pred_2 , by = "REN_ID")
colnames(pred_merge)[2:3] <- c("REN_BASE_RENT_1","REN_BASE_RENT_2")

##
pred_merge[pred_merge$REN_BASE_RENT_1>cut_off,]

#train = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
test = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
#Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_NAs5.csv" , sep='') , stringsAsFactors = F))

test_merge = merge(x = test , y = pred_merge , by = "REN_ID")
#test_merge = merge(x = test_merge , y = Xtest , by = "REN_ID")
test_merge[test_merge$REN_BASE_RENT_1>cut_off,]

## 
pred_merge[pred_merge$REN_BASE_RENT_1>cut_off,]$REN_BASE_RENT_1 <- pred_merge[pred_merge$REN_BASE_RENT_1>cut_off,]$REN_BASE_RENT_2

##
pred_merge$REN_BASE_RENT = 0.6 * pred_merge$REN_BASE_RENT_1 + 0.4 * pred_merge$REN_BASE_RENT_2 

## pred 
submission = pred_merge[,c("REN_ID","REN_BASE_RENT")]
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "merge3_60_40_removeOutlierOver_5000.csv" , sep='') ,
          row.names=FALSE)

##
pred_merge$REN_BASE_RENT = 0.7 * pred_merge$REN_BASE_RENT_1 + 0.3 * pred_merge$REN_BASE_RENT_2 

## pred 
submission = pred_merge[,c("REN_ID","REN_BASE_RENT")]
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "merge3_70_30_removeOutlierOver_5000.csv" , sep='') ,
          row.names=FALSE)



### 50 - 50 
id_1 = "merge3_60_40_removeOutlierOver_5000.csv"
id_2 = "merge2_50_40_10_removeOutlierOver_3500.csv"

pred_1 = as.data.frame( fread(paste(ff.getPath("submission_1") , id_1 , sep='') , stringsAsFactors = F))
pred_2 = as.data.frame( fread(paste(ff.getPath("submission_1") , id_2 , sep='') , stringsAsFactors = F))

pred_merge = merge(x = pred_1 , y = pred_2 , by = "REN_ID")
colnames(pred_merge)[2:3] <- c("REN_BASE_RENT_1","REN_BASE_RENT_2")

pred_merge$REN_BASE_RENT = 0.5 * pred_merge$REN_BASE_RENT_1 + 0.5 * pred_merge$REN_BASE_RENT_2 

## pred 
submission = pred_merge[,c("REN_ID","REN_BASE_RENT")]
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "merge4_50_50.csv" , sep='') ,
          row.names=FALSE)

###
id_1 = "merge4_50_50.csv"

pred_1 = as.data.frame( fread(paste(ff.getPath("submission_1") , id_1 , sep='') , stringsAsFactors = F))

Ytrain = as.data.frame( fread(paste(ff.getPath("elab") , "Ytrain_NAs6.csv" , sep='') , stringsAsFactors = F))

pred_1$REN_BASE_RENT[pred_1$REN_BASE_RENT<400] <- pred_1$REN_BASE_RENT[pred_1$REN_BASE_RENT<400] * 0.9 

## pred 
submission = pred_1[,c("REN_ID","REN_BASE_RENT")]
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "merge4_50_50_09_down_400.csv" , sep='') ,
          row.names=FALSE)


###  paul_425_removeOutliersITS_7000.csv
id_1 = "merge4_50_50.csv"
id_2 = "paul_36_ES_removeOutliersITS_9000.csv"
pred_1 = as.data.frame( fread(paste(ff.getPath("submission_1") , id_1 , sep='') , stringsAsFactors = F))
pred_2 = as.data.frame( fread(paste(ff.getPath("submission_1") , id_2 , sep='') , stringsAsFactors = F))

pred_merge = merge(x = pred_1 , y = pred_2 , by = "REN_ID")
colnames(pred_merge)[2:3] <- c("REN_BASE_RENT_1","REN_BASE_RENT_2")

pred_merge$delta_ass = pred_merge$REN_BASE_RENT_1 - pred_merge$REN_BASE_RENT_2



###  paul_425_removeOutliersITS_7000
id_1 = "merge4_50_50.csv"
id_2 = "paul_425_removeOutliersITS_7000.csv"
pred_1 = as.data.frame( fread(paste(ff.getPath("submission_1") , id_1 , sep='') , stringsAsFactors = F))
pred_2 = as.data.frame( fread(paste(ff.getPath("submission_1") , id_2 , sep='') , stringsAsFactors = F))

pred_merge = merge(x = pred_1 , y = pred_2 , by = "REN_ID")
colnames(pred_merge)[2:3] <- c("REN_BASE_RENT_1","REN_BASE_RENT_2")

pred_merge$delta_ass = pred_merge$REN_BASE_RENT_1 - pred_merge$REN_BASE_RENT_2

pred_merge$REN_BASE_RENT = pred_merge$REN_BASE_RENT_1
pred_merge$REN_BASE_RENT[pred_merge$delta_ass<0] <- pred_merge$REN_BASE_RENT_2[pred_merge$delta_ass<0]

## pred 
submission = pred_merge[,c("REN_ID","REN_BASE_RENT")]
print(head(submission))
write.csv(submission,
          quote=FALSE, 
          file=paste(ff.getPath("submission_1") , "merge4_50_50_gambling2.csv" , sep='') ,
          row.names=FALSE)



