library(caret)
library(Hmisc)
library(data.table)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)
library(fBasics)
library(lattice)
require(xgboost)
require(methods)

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/facebook-recruiting-iv-human-or-bot"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/facebook-recruiting-iv-human-or-bot/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/facebook-recruiting-iv-human-or-bot"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/facebook-recruiting-iv-human-or-bot/"
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


#################
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep=''))) ## outcome = 0 human 

train = as.data.frame( fread(paste(getBasePath("data") , 
                                   "train.csv" , sep=''))) 

test = as.data.frame( fread(paste(getBasePath("data") , 
                                  "test.csv" , sep='')))

bids = as.data.frame( fread(paste(getBasePath("data") , 
                                  "bids.csv" , sep='')))

secure_humans = as.data.frame( fread(paste(getBasePath("data") , 
                                           "secure_humans_idx.csv" , sep='')))

X = as.data.frame( fread(paste(getBasePath("data") , 
                                           "Xfin4.csv" , sep='')))

##### Train / Test 
train.full = merge(x = bids , y = train , by="bidder_id"  )
test.full = merge(x = bids , y = test , by="bidder_id"  )
## REMOVE SECURE HUMANS 
cat(">>> remove secure humans ... \n")
print(dim(test.full))
test.full = test.full[! test.full$bidder_id %in% test[secure_humans$index ,]$bidder_id,] ## remove from test.set 
print(dim(test.full))

## train/test index , labels  

trind = 1:length(unique(train.full$bidder_id))
teind = (max(trind)+1):nrow(X)
X.full = merge(x=X , y=train , by="bidder_id")
y = X.full$outcome
rm(X.full)

#### Making feature: average bids per auction 
bids_per_auction = ddply(bids , .(bidder_id,auction) , function(x) c(bid.num=length(x$bid_id)) )
avg_bids_per_auction = ddply(bids_per_auction , .(bidder_id) , function(x) c(avg_bids_per_auction = sum(x$bid.num)/length(x$auction), 
                                                                             avg_bids_per_auction2 = mean(x$bid.num), 
                                                                             sd_bids_per_auction = sd(x$bid.num) , 
                                                                             quant_025 = as.numeric(quantile(x$bid.num , c(0.25))), 
                                                                             quant_05 = as.numeric(quantile(x$bid.num , c(0.50))),
                                                                             quant_075 = as.numeric(quantile(x$bid.num , c(0.75)))
                             ))
avg_bids_per_auction[is.na(avg_bids_per_auction$sd_bids_per_auction) , ]$sd_bids_per_auction = 0

####### Xfin5
bidder_bids_per_auction = as.data.frame(matrix(rep(0,nrow(X)*1),nrow=nrow(X),ncol=1))
colnames(bidder_bids_per_auction) = c("avg_bids")

for (bid in (1:max(teind))) {
  bidder_id = X[bid,]$bidder_id
  bidder_bids_per_auction[bid,'avg_bids'] = avg_bids_per_auction[avg_bids_per_auction$bidder_id == bidder_id,]$avg_bids_per_auction
  if (bid %% 100 == 0) cat("..[",bid,"/",max(teind),"]..")
}

############################ store feature set 
## elimino bid_seq 
X.base = X[,-grep("bid_seq" , colnames(X) )]
## aggiungo auction 
X.base = cbind(X.base , bidder_bids_per_auction )
cat(">>> dim X.base [con bidder_id]:",dim(X.base)," .... storing on disk \n")
write.csv(X.base,quote=FALSE, 
          file=paste(getBasePath("data"),"Xfin5.csv",sep='') ,
          row.names=FALSE)
############################ 

############################ Xfin5.2
# bidder_bids_per_auction = as.data.frame(matrix(rep(0,nrow(X)*2),nrow=nrow(X),ncol=2))
# colnames(bidder_bids_per_auction) = c("avg_bids","sd_bids")
# 
# for (bid in (1:max(teind))) {
#   bidder_id = X[bid,]$bidder_id
#   bidder_bids_per_auction[bid,'avg_bids'] = avg_bids_per_auction[avg_bids_per_auction$bidder_id == bidder_id,]$avg_bids_per_auction
#   bidder_bids_per_auction[bid,'sd_bids'] = avg_bids_per_auction[avg_bids_per_auction$bidder_id == bidder_id,]$sd_bids_per_auction
#   if (bid %% 100 == 0) cat("..[",bid,"/",max(teind),"]..")
# }
# 
# ############################ store feature set 
# ## elimino bid_seq 
# X.base = X[,-grep("bid_seq" , colnames(X) )]
# ## aggiungo auction 
# X.base = cbind(X.base , bidder_bids_per_auction )
# cat(">>> dim X.base [con bidder_id]:",dim(X.base)," .... storing on disk \n")
# write.csv(X.base,quote=FALSE, 
#           file=paste(getBasePath("data"),"Xfin5.2.csv",sep='') ,
#           row.names=FALSE)

############################

########################### Xfin5.3
# bidder_bids_per_auction = as.data.frame(matrix(rep(0,nrow(X)*3),nrow=nrow(X),ncol=3))
# colnames(bidder_bids_per_auction) = c("quant_025","quant_05","quant_075")
# 
# for (bid in (1:max(teind))) {
#   bidder_id = X[bid,]$bidder_id
#   bidder_bids_per_auction[bid,'quant_025'] = avg_bids_per_auction[avg_bids_per_auction$bidder_id == bidder_id,]$quant_025
#   bidder_bids_per_auction[bid,'quant_05'] = avg_bids_per_auction[avg_bids_per_auction$bidder_id == bidder_id,]$quant_05
#   bidder_bids_per_auction[bid,'quant_075'] = avg_bids_per_auction[avg_bids_per_auction$bidder_id == bidder_id,]$quant_075
#   if (bid %% 100 == 0) cat("..[",bid,"/",max(teind),"]..")
# }
# 
# ############################ store feature set 
# ## elimino bid_seq 
# X.base = X[,-grep("bid_seq" , colnames(X) )]
# ## aggiungo auction 
# X.base = cbind(X.base , bidder_bids_per_auction )
# cat(">>> dim X.base [con bidder_id]:",dim(X.base)," .... storing on disk \n")
# write.csv(X.base,quote=FALSE, 
#           file=paste(getBasePath("data"),"Xfin5.3.csv",sep='') ,
#           row.names=FALSE)

############################

## elimino bidder_id 
X.base = X[,-grep("bidder_id" , colnames(X) )]

## elimino bid_seq 
X.base = X.base[,-grep("bid_seq" , colnames(X) )]
  
## aggiungo auction 
X.base = cbind(X.base , bidder_bids_per_auction )
cat(">>> dim X.base [no bidder_id]:",dim(X.base),"\n")

######### XGboost 
x = as.matrix(X.base)
x = matrix(as.numeric(x),nrow(x),ncol(x))

##### xgboost --> set necessary parameter
param <- list("objective" = "binary:logistic",
              "eval_metric" = "auc",
              #"num_class" = 4,
              "eta" = 0.05,  ## suggested in ESLII
              "gamma" = 0.7,  
              "max_depth" = 25, 
              "subsample" = 0.5 , ## suggested in ESLII
              "nthread" = 10, 
              "min_child_weight" = 1 , 
              "colsample_bytree" = 0.5, 
              "max_delta_step" = 1)

cat(">>Params:\n")
print(param)

### Cross-validation 
cat(">>Cross Validation ... \n")
inCV = T
xval.perf = -1
bst.cv = NULL
early.stop = cv.nround = 1000
cat(">> cv.nround: ",cv.nround,"\n") 

while (inCV) {
    cat(">>> maximizing auc ...\n")
    bst.cv = xgb.cv(param=param, data = x[trind,], label = y, nfold = 5, nrounds=cv.nround )    
    print(bst.cv)
    early.stop = min(which(bst.cv$test.auc.mean == max(bst.cv$test.auc.mean) ))
    xval.perf = bst.cv[early.stop,]$test.auc.mean
    cat(">> early.stop: ",early.stop," [xval.perf:",xval.perf,"]\n") 
  
  if (early.stop < cv.nround) {
    inCV = F
    cat(">> stopping [early.stop < cv.nround=",cv.nround,"] ... \n") 
  } else {
    cat(">> redo-cv [early.stop == cv.nround=",cv.nround,"] with 2 * cv.nround ... \n") 
    cv.nround = cv.nround * 2 
  }
  
  gc()
}

### Prediction 
bst = xgboost(param = param, data = x[trind,], label = y, nrounds = early.stop) 

cat(">> Making prediction ... \n")
pred = predict(bst,x[teind,])

print(">> prediction <<")
print(mean(pred))

print(">> train set labels <<")
print(mean(y))

#### assembling 
sub = data.frame(bidder_id = X[teind,]$bidder_id , pred.xgb = pred)
sub.full = merge(x=sub,y=sampleSubmission,by="bidder_id" , all=T)
sub.full$prediction = ifelse( ! is.na(sub.full$pred.xgb) , sub.full$pred.xgb , 0 )
sub.full = sub.full[,-2]

#### writing on disk 
fn = paste("sub_xgboost_auction_xval" , xval.perf , ".csv" , sep='') 
cat(">> writing prediction on disk [",fn,"]... \n")
write.csv(sub.full,quote=FALSE, 
          file=paste(getBasePath("data"),fn,sep='') ,
          row.names=FALSE)









