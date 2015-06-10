library(caret)
library(Hmisc)
library(data.table)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)
library(fBasics)
library(lattice)

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

################
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))


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
                                           "X1.csv" , sep='')))


train.full = merge(x = bids , y = train , by="bidder_id"  )
test.full = merge(x = bids , y = test , by="bidder_id"  )

########### Load libs 
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))

########### REMOVE SECURE HUMANS 
cat(">>> remove secure humans ... ")
print(dim(test.full))
test.full = test.full[! test.full$bidder_id %in% test[secure_humans$index ,]$bidder_id,] ## remove from test.set 
print(dim(test.full))

## train/test index  
trind = 1:length(unique(train.full$bidder_id))
teind = (max(trind)+1):nrow(X)

######## Make Feature: Merchandise a bidder use 
cat(">>> Making feature:  Merchandise a bidder use ... \n")
merchandise_train = unique(train.full$merchandise)
cat(">> we have",length(merchandise_train),"different merchandise on train set\n")
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

##
merch.outcome = ddply(train.full , .(merchandise) , function(x) c(zero=sum(x$outcome==0),one=sum(x$outcome==1)) )
merch.outcome$ratio = merch.outcome$one/(merch.outcome$zero+merch.outcome$one)
merch.outcome = merch.outcome[order(merch.outcome$ratio , decreasing = T) , ]
cat(">>> merchandise with only robots:",sum(merch.outcome$ratio==1),"\n")
cat(">>> merchandise with only humans:",sum(merch.outcome$ratio==0),"\n")
describe(merch.outcome$ratio)
merch.outcome$ratio_lev = cut2(merch.outcome$ratio , cuts=c(0,0.03,0.046,0.048,0.063,0.1,0.176,0.197,1) )
print(table(merch.outcome$ratio_lev))
merch.outcome$ratio_lev_int = unlist(lapply(merch.outcome$ratio_lev, function(x) {
  i = 0
  for (i in 1:12) 
    if (x == levels(x)[[i]]) 
      break 
  return(i)
}))
print(head(merch.outcome))

##
bidder_merch.train.full = merge(x=bidder_merch.train,y=merch.outcome,by='merchandise')
bidder_merch.test.full = merge(x=bidder_merch.test,y=merch.outcome,by='merchandise')

## Making feature 
bidder_merch_types = as.data.frame(matrix(rep(0,nrow(X)*length(unique(merch.outcome$ratio_lev_int))),
                                             nrow=nrow(X),
                                             ncol=length(unique(merch.outcome$ratio_lev_int))))
colnames(bidder_merch_types) = paste0("merch_lev",1:ncol(bidder_merch_types))

## Train
cat(">>> making features for train set ...\n")
for (bid in trind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_merch.train.full[bidder_merch.train.full$bidder_id == bidder_id , c('ratio_lev_int','num')] , 
             .(ratio_lev_int) , function(x) c(num=sum(x[2])))
  
  bidder_merch_types[bid,aa$ratio_lev_int] = aa$num 
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(trind),"]..")
}

## Test train features 
bid = sample(trind,1)
cat("\n>>>Testing train features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_merch.train.full[bidder_merch.train.full$bidder_id == bidder_id , c('ratio_lev_int','num')])
print(bidder_merch_types[bid,])
if (sum(bidder_merch_types[bid,]) == 
      sum(bidder_merch.train.full[bidder_merch.train.full$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}

## Test 
cat(">>> making features for test set ...\n")
for (bid in teind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_merch.test.full[bidder_merch.test.full$bidder_id == bidder_id , c('ratio_lev_int','num')] , 
             .(ratio_lev_int) , function(x) c(num=sum(x[2])))
  
  bidder_merch_types[bid,aa$ratio_lev_int] = aa$num 
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(teind),"]..")
}

## Test test features 
bid = sample(teind,1)
cat("\n>>>Testing test features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_merch.test.full[bidder_merch.test.full$bidder_id == bidder_id , c('ratio_lev_int','num')])
print(bidder_merch_types[bid,])
if (sum(bidder_merch_types[bid,]) == 
      sum(bidder_merch.test.full[bidder_merch.test.full$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}

### making percentages 
sums = as.numeric(apply(bidder_merch_types,1,function(x) sum(x[1:ncol(bidder_merch_types)]) ))
bidder_merch_types = bidder_merch_types / sums
sums_after = as.numeric(apply(bidder_merch_types,1,function(x) sum(x[1:ncol(bidder_merch_types)]) ))
if( sum( sums_after > 1.001) > 0 || sum( sums_after < 0.999) > 0 )
  stop("something wrong")

### Binding features 
X=cbind(X,bidder_merch_types)

cat(">>> storing on disk ...\n")
print(head(X))
write.csv(X,quote=FALSE, 
          file=paste(getBasePath("data"),"X2.csv",sep='') ,
          row.names=FALSE)





