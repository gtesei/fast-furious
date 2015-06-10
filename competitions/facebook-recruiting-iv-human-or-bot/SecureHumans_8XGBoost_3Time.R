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
                                           "Xfin2.csv" , sep='')))

time_bids = as.data.frame( fread(paste(getBasePath("data") , 
                                       "time_bids_all_auctions2.csv" , sep='')))

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

###### Fabbrico la nuova feature 
bidder_times = as.data.frame(matrix(rep(0,nrow(X)*1),nrow=nrow(X),ncol=1))
colnames(bidder_times) = c("bid_seq")

cat(">>> making features for train and test set ...\n")

bidder.fastest.vect = time_bids[!is.na(time_bids$bidder.fastest) & time_bids$min.time.seq == 0 ,]$bidder.fastest
bidder.slowest.vect = time_bids[!is.na(time_bids$bidder.slowest) ,]$bidder.slowest

bidder.last.vect = time_bids[!is.na(time_bids$bidder.last) ,]$bidder.last 
bidder.last.1.vect = time_bids[!is.na(time_bids$bidder.last.1),]$bidder.last.1
bidder.last.2.vect = time_bids[!is.na(time_bids$bidder.last.2),]$bidder.last.2
bidder.last.3.vect = time_bids[!is.na(time_bids$bidder.last.3),]$bidder.last.3
bidder.last.4.vect = time_bids[!is.na(time_bids$bidder.last.4),]$bidder.last.4
bidder.last.5.vect = time_bids[!is.na(time_bids$bidder.last.5),]$bidder.last.5

bidder.seqlongest.vect = time_bids[!is.na(time_bids$bidder.seqlongest) & time_bids$seq_long_max > 4 ,]$bidder.seqlongest

bidder.seqlongest.vect.20 = time_bids[!is.na(time_bids$bidder.seqlongest) & time_bids$seq_long_max > 20 ,]$bidder.seqlongest

bidder.first.vect = time_bids[!is.na(time_bids$bidder.first) ,]$bidder.first

for (bid in (1:max(teind))) {
  
  bidder_id = X[bid,]$bidder_id
  
  bidder.fastest = sum(bidder.fastest.vect == bidder_id )
  bidder.slowest = sum(bidder.slowest.vect == bidder_id )
  
  bidder.last = sum(bidder.last.vect == bidder_id )
  bidder.last.1 = sum(bidder.last.1.vect == bidder_id )
  bidder.last.2 = sum(bidder.last.2.vect == bidder_id )
  bidder.last.3 = sum(bidder.last.3.vect == bidder_id )
  bidder.last.4 = sum(bidder.last.4.vect == bidder_id )
  #bidder.last.5 = sum(bidder.last.5.vect == bidder_id )
  
  bidder.seqlongest = sum(bidder.seqlongest.vect == bidder_id )
  bidder.seqlongest.20 = sum(bidder.seqlongest.vect.20 == bidder_id )
  
  bidder.first = sum(bidder.first.vect == bidder_id )
  
#  auctions = length(unique(bids[bids$bidder_id == bidder_id,]$auction))
  
  bidder_times[bid , "bid_seq" ] = bidder.seqlongest
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(teind),"]..")
}


###### Assemblo le features 
X.base = cbind(X , bidder_times )
cat(">> writing X on disk .. \n")
write.csv(X.base,quote=FALSE, 
          file=paste(getBasePath("data"),"Xfin3.csv",sep='') ,
          row.names=FALSE)

## elimino bidder_id 
X.base = X[,-grep("bidder_id" , colnames(X) )]
  
## aggiungo nuove features  
X.base = cbind(X.base , bidder_times )
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
fn = paste("sub_xgboost_xval", xval.perf , ".csv" , sep='')
cat(">> writing prediction on disk [",fn,"]... \n")
write.csv(sub.full,quote=FALSE, 
          file=paste(getBasePath("data"),fn,sep='') ,
          row.names=FALSE)









