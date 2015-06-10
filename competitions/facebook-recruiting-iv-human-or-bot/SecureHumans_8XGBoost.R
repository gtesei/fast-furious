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
                                           "Xfin.csv" , sep='')))

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

#### Making feature:  Merchandise a bidder use
cat(">>> Making feature:  Merchandise a bidder use ... \n")
merchandise_train = unique(train.full$merchandise)
merchandise_test = unique(test.full$merchandise)
cat(">> we have",length(merchandise_train),"different merchandise on train set\n")
cat(">> we have",length(merchandise_test),"different merchandise on test set\n")
cat(">> all merchandise in test set are in the train set:",sum(!merchandise_test %in% merchandise_train)==0,"\n")

## train 
bidder_merch.train = ddply(train.full[,c('bidder_id','merchandise')],
                           .(bidder_id) , function(x)  {
                             ddply(x, .(merchandise) , function(xx) {
                               length(xx[,1])
                             } ) 
                           })
colnames(bidder_merch.train)[3]='num'

## test 
bidder_merch.test = ddply(test.full[,c('bidder_id','merchandise')],
                          .(bidder_id) , function(x)  {
                            ddply(x, .(merchandise) , function(xx) {
                              length(xx[,1])
                            } ) 
                          })
colnames(bidder_merch.test)[3]='num'

## Making feature 
bidder_merch_types = as.data.frame(matrix(rep(0,nrow(X)*length(unique(merchandise_train))),
                                          nrow=nrow(X),
                                          ncol=length(unique(merchandise_train))))
colnames(bidder_merch_types) = paste("merchandise_",gsub(" ","_", merchandise_train),sep='')

## Train
cat(">>> making features for train set ...\n")
for (bid in trind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_merch.train[bidder_merch.train$bidder_id == bidder_id , c('merchandise','num')] , 
             .(merchandise) , function(x) c(num=sum(x[2])))
  
  #bidder_merch_types[bid,which(merchandise_train %in% aa$merchandise)] = aa$num 
  for (ii in 1:nrow(aa)) {
    bidder_merch_types[bid, which(merchandise_train %in% aa[ii,]$merchandise)  ] = aa[ii,]$num  
  }
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(trind),"]..")
}

## Test train features 
bid = sample(trind,1)
cat("\n>>>Testing train features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_merch.train[bidder_merch.train$bidder_id == bidder_id , c('merchandise','num')])
print(bidder_merch_types[bid,])
if (sum(bidder_merch_types[bid,]) == 
      sum(bidder_merch.train[bidder_merch.train$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}

## Test
cat(">>> making features for test set ...\n")
for (bid in teind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_merch.test[bidder_merch.test$bidder_id == bidder_id , c('merchandise','num')] , 
             .(merchandise) , function(x) c(num=sum(x[2])))
  
  ##bidder_merch_types[bid,which(merchandise_train %in% aa$merchandise)] = aa$num 
  for (ii in 1:nrow(aa)) {
    bidder_merch_types[bid, which(merchandise_train %in% aa[ii,]$merchandise)  ] = aa[ii,]$num  
  }
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(teind),"]..")
}

## Test train features 
bid = sample(teind,1)
cat("\n>>>Testing train features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_merch.test[bidder_merch.test$bidder_id == bidder_id , c('merchandise','num')])
print(bidder_merch_types[bid,])
if (sum(bidder_merch_types[bid,]) == 
      sum(bidder_merch.test[bidder_merch.test$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}

### scaling 
# cat(">>> scaling ...\n")
# sums = as.numeric(apply(bidder_merch_types,1,function(x) sum(x[1:ncol(bidder_merch_types)]) ))
# sums0s.idx = which(sums == 0)
# for (i in 1:length(sums)) {
#   if (sums[i]==0) {
#     cat(">> i == ",i,"sums == 0, ...\n")
#   } else {
#     bidder_merch_types[i,] = bidder_merch_types[i,]/sums[i]
#   }
# }
# sums_after = NULL 
# if (length(-sums0s.idx) > 0) {
#   sums_after = as.numeric(apply(bidder_merch_types[-sums0s.idx,],1,function(x) sum(x[1:ncol(bidder_merch_types)]) ))
# } else {
#   sums_after = as.numeric(apply(bidder_merch_types,1,function(x) sum(x[1:ncol(bidder_merch_types)]) ))
# }
# if( sum( sums_after > 1.001) > 0 || sum( sums_after < 0.999) > 0 )
#   stop("something wrong")


###### Making feature: country  
cat(">>> country ... \n")
country_train = unique(train.full$country)
country_test = unique(test.full$country)
cat(">> we have",length(country_train),"different country on train set\n")
cat(">> we have",length(country_test),"different country on test set\n")
cat(">> all country in test set are in the train set:",sum(!country_test %in% country_train)==0,"\n")
cat(">> country in test set not in the train set:",sum(!country_test %in% country_train),"(",
    sum(!country_test %in% country_train)/length(country_test)*100,"%)\n")

## train 
bidder_country.train = ddply(train.full[,c('bidder_id','country')],
                           .(bidder_id) , function(x)  {
                             ddply(x, .(country) , function(xx) {
                               length(xx[,1])
                             } ) 
                           })
colnames(bidder_country.train)[3]='num'

## test 
bidder_country.test = ddply(test.full[,c('bidder_id','country')],
                          .(bidder_id) , function(x)  {
                            ddply(x, .(country) , function(xx) {
                              length(xx[,1])
                            } ) 
                          })
colnames(bidder_country.test)[3]='num'

## Making feature 
bidder_country_types = as.data.frame(matrix(rep(0,nrow(X)*length(unique(country_train))),
                                          nrow=nrow(X),
                                          ncol=length(unique(country_train))))
colnames(bidder_country_types) = paste("country_",gsub(" ","_", country_train),sep='')

## Train
cat(">>> making features for train set ...\n")
for (bid in trind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_country.train[bidder_country.train$bidder_id == bidder_id , c('country','num')] , 
             .(country) , function(x) c(num=sum(x[2])))
  
  ##bidder_country_types[bid,which(country_train %in% aa$country)] = aa$num 
  for (ii in 1:nrow(aa)) {
    bidder_country_types[bid, which(country_train %in% aa[ii,]$country)  ] = aa[ii,]$num  
  }
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(trind),"]..")
}

## Test train features 
bid = sample(trind,1)
cat("\n>>>Testing train features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_country.train[bidder_country.train$bidder_id == bidder_id , c('country','num')])
print(bidder_country_types[bid,])
if (sum(bidder_country_types[bid,]) == 
      sum(bidder_country.train[bidder_country.train$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}

## Test
cat(">>> making features for test set ...\n")
for (bid in teind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_country.test[bidder_country.test$bidder_id == bidder_id , c('country','num')] , 
             .(country) , function(x) c(num=sum(x[2])))
  
  ##bidder_country_types[bid,which(country_train %in% aa$country)] = aa$num 
  for (ii in 1:nrow(aa)) {
    bidder_country_types[bid, which(country_train %in% aa[ii,]$country)  ] = aa[ii,]$num  
  }
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(teind),"]..")
}

## Test train features 
bid = sample(teind,1)
cat("\n>>>Testing train features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_country.test[bidder_country.test$bidder_id == bidder_id , c('country','num')])
print(bidder_country_types[bid,])
if (sum(bidder_country_types[bid,]) == 
      sum(bidder_country.test[bidder_country.test$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}

### scaling 
# sums = as.numeric(apply(bidder_country_types,1,function(x) sum(x[1:ncol(bidder_country_types)]) ))
# cat(">>> scaling ...\n")
# sums0s.idx = which(sums == 0)
# for (i in 1:length(sums)) {
#   if (sums[i]==0) {
#     cat(">> i == ",i,"sums == 0, ...\n")
#   } else {
#     bidder_country_types[i,] = bidder_country_types[i,]/sums[i]
#   }
# }
# sums_after = NULL 
# if (length(-sums0s.idx) > 0) {
#   sums_after = as.numeric(apply(bidder_country_types[-sums0s.idx,],1,function(x) sum(x[1:ncol(bidder_country_types)]) ))
# } else {
#   sums_after = as.numeric(apply(bidder_country_types,1,function(x) sum(x[1:ncol(bidder_country_types)]) ))
# }
# if( sum( sums_after > 1.001) > 0 || sum( sums_after < 0.999) > 0 )
#   stop("something wrong")
### end of scaling 

## N. bids a bidder partecipated 
# cat(">>> Making feature: N. bids a bidder did ... \n")
# bidder_bids.train = ddply(train.full[,c('bidder_id','bid_id')],
#                               .(bidder_id) , function(x) c(bid.num = length(unique(x$bid_id))) )
# bidder_bids.test = ddply(test.full[,c('bidder_id','bid_id')],
#                              .(bidder_id) , function(x) c(bid.num = length(unique(x$bid_id))) )
# 
# bidder_bids = rbind(bidder_bids.train,bidder_bids.test)
# bidder_bids$bid.num = scale(bidder_bids$bid.num)

###### other  
cat(">>> device ... \n")
xx_train = unique(train.full$device)
xx_test = unique(test.full$device)
cat(">> we have",length(xx_train),"different device on train set\n")
cat(">> we have",length(xx_test),"different device on test set\n")
cat(">> all device in test set are in the train set:",sum(!xx_test %in% xx_train)==0,"\n")
cat(">> device in test set not in the train set:",sum(!xx_test %in% xx_train),"(",
    sum(!xx_test %in% xx_train)/length(xx_test)*100,"%)\n")

cat(">>> ip ... \n")
xx_train = unique(train.full$ip)
xx_test = unique(test.full$ip)
cat(">> we have",length(xx_train),"different ip on train set\n")
cat(">> we have",length(xx_test),"different ip on test set\n")
cat(">> all ip in test set are in the train set:",sum(!xx_test %in% xx_train)==0,"\n")
cat(">> ip in test set not in the train set:",sum(!xx_test %in% xx_train),"(",
    sum(!xx_test %in% xx_train)/length(xx_test)*100,"%)\n")

cat(">>> url ... \n")
xx_train = unique(train.full$url)
xx_test = unique(test.full$url)
cat(">> we have",length(xx_train),"different url on train set\n")
cat(">> we have",length(xx_test),"different url on test set\n")
cat(">> all url in test set are in the train set:",sum(!xx_test %in% xx_train)==0,"\n")
cat(">> url in test set not in the train set:",sum(!xx_test %in% xx_train),"(",
    sum(!xx_test %in% xx_train)/length(xx_test)*100,"%)\n")

#################
## index 
##  1     - bidder_id 
##  2     - auct.num (to scale) 
##  3:14  - auct_lev 
##  15:22 - merch_lev 
##  23    - dev.num (to scale)
##  24:35 - dev_lev 
##  36    - country.num (to scale)
##  37:48 - country_lev 
##  49    - ip.num (to scale)
##  50:55 - ip_lev 
##  56    - url.num (to scale)
##  57:62 - url_lev 
################
X.base = X[,-c(3:14,15:22,24:35,37:48,50:55,57:62)]
X.base = cbind(bidder_merch_types,X.base) 
X.base = cbind(bidder_country_types,X.base) 
cat(">>> dim X.base:",dim(X.base),"\n")
cat(">>> storing matrix on disk .. \n")
write.csv(X.base,quote=FALSE, 
          file=paste(getBasePath("data"),"Xfin2.csv",sep='') ,
          row.names=FALSE)

################ elimino tutti i lev 
X.base = X[,-c(1,3:14,15:22,24:35,37:48,50:55,57:62)]
X.base = cbind(bidder_merch_types,X.base) 
X.base = cbind(bidder_country_types,X.base) 
cat(">>> dim X.base [no bidder_id]:",dim(X.base),"\n")

##XGboost 
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
fn = "sub_xgboost.csv"
cat(">> writing prediction on disk [",fn,"]... \n")
write.csv(sub.full,quote=FALSE, 
          file=paste(getBasePath("data"),fn,sep='') ,
          row.names=FALSE)









