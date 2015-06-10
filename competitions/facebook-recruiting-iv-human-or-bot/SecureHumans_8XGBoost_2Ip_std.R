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

#### Making feature:  ip a bidder use
cat(">>> Making feature:  ip a bidder use ************* STANDARD VERSION  *************  ... \n")
ip_train = unique(train.full$ip)
ip_test = unique(test.full$ip)
cat(">> we have",length(ip_train),"different ip on train set\n")
cat(">> we have",length(ip_test),"different ip on test set\n")
cat(">> all ip in test set are in the train set:",sum(!ip_test %in% ip_train)==0,"\n")
cat(">> ip in test set not in the train set:",sum(!ip_test %in% ip_train),"(",
    sum(!ip_test %in% ip_train)/length(ip_test)*100,"%)\n")


## train 
bidder_ip.train = ddply(train.full[,c('bidder_id','ip')],
                           .(bidder_id) , function(x)  {
                             ddply(x, .(ip) , function(xx) {
                               length(xx[,1])
                             } ) 
                           })
colnames(bidder_ip.train)[3]='num'

## test 
bidder_ip.test = ddply(test.full[,c('bidder_id','ip')],
                          .(bidder_id) , function(x)  {
                            ddply(x, .(ip) , function(xx) {
                              length(xx[,1])
                            } ) 
                          })
colnames(bidder_ip.test)[3]='num'

## Making feature 
bidder_ip_types = as.data.frame(matrix(rep(0,nrow(X)*length(unique(ip_train))),
                                          nrow=nrow(X),
                                          ncol=length(unique(ip_train))))
colnames(bidder_ip_types) = paste("ip_",gsub(" ","_", ip_train),sep='')

## Train
cat(">>> making features for train set ...\n")
for (bid in trind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_ip.train[bidder_ip.train$bidder_id == bidder_id , c('ip','num')] , 
             .(ip) , function(x) c(num=sum(x[2])))
  
  #bidder_merch_types[bid,which(merchandise_train %in% aa$merchandise)] = aa$num 
  for (ii in 1:nrow(aa)) {
    bidder_ip_types[bid, which(ip_train %in% aa[ii,]$ip)  ] = aa[ii,]$num  
  }
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(trind),"]..")
}

## Test train features 
bid = sample(trind,1)
cat("\n>>>Testing train features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_ip.train[bidder_ip.train$bidder_id == bidder_id , c('ip','num')])
#print(bidder_ip_types[bid,])
if (sum(bidder_ip_types[bid,]) == 
      sum(bidder_ip.train[bidder_ip.train$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}

## Test
cat(">>> making features for test set ...\n")
for (bid in teind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_ip.test[bidder_ip.test$bidder_id == bidder_id , c('ip','num')] , 
             .(ip) , function(x) c(num=sum(x[2])))
  
  ##bidder_merch_types[bid,which(merchandise_train %in% aa$merchandise)] = aa$num 
  for (ii in 1:nrow(aa)) {
    bidder_ip_types[bid, which(ip_train %in% aa[ii,]$ip)  ] = aa[ii,]$num  
  }
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(teind),"]..")
}

## Test train features 
bid = sample(teind,1)
cat("\n>>>Testing train features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_ip.test[bidder_ip.test$bidder_id == bidder_id , c('ip','num')])
#print(bidder_ip_types[bid,])
if (sum(bidder_ip_types[bid,]) == 
      sum(bidder_ip.test[bidder_ip.test$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}


###### Assemblo le features 
# X.base = cbind(X , bidder_ip_types )
# cat(">> writing X on disk .. \n")
# write.csv(X.base,quote=FALSE, 
#           file=paste(getBasePath("data"),"Xfin5.csv",sep='') ,
#           row.names=FALSE)

## elimino bidder_id 
X.base = X[,-grep("bidder_id" , colnames(X) )]
  
## aggiungo ip 
X.base = cbind(X.base , bidder_ip_types)
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
fn = paste("sub_xgboost_ip_xval" , xval.perf , ".csv" , sep='') 
cat(">> writing prediction on disk [",fn,"]... \n")
write.csv(sub.full,quote=FALSE, 
          file=paste(getBasePath("data"),fn,sep='') ,
          row.names=FALSE)









