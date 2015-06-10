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


train.full = merge(x = bids , y = train , by="bidder_id"  )
test.full = merge(x = bids , y = test , by="bidder_id"  )

########### Load libs 
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))

########### SECURE HUMANS 
secure_humans_idx = NULL 

cat(">>> Computing secure humans .... \n")

### bidders that never made a bid ...
cat("\n 1) bidders occurring in test set that never made a bid:",sum(!test$bidder_id %in% bids$bidder_id)," \n")
bidders = unique(test[! test$bidder_id %in% bids$bidder_id,]$bidder_id)
print(bidders)
secure_humans_idx = which(test$bidder_id %in% bidders)

### bidders that never made a bid in auctions occurring in train set 
bidder_alone = 'f7b6e7e8d3ac5b17ee7673699899e2e0dwvpa'
cat("\n 2) bidders that never made a bid in auctions occurring in bids where there's at least one bidder occurring in the train set :1 \n")
print(bidder_alone)
secure_humans_idx = c(secure_humans_idx,which(test$bidder_id == bidder_alone ))

test.full = test.full[-which(test.full$bidder_id == bidder_alone),] ## remove from test.set 

cat(">>> secure_humans_idx:  \n")
print(secure_humans_idx)

cat("********** storing on disk \n")
write.csv(data.frame(index = secure_humans_idx),
          quote=FALSE, 
          file=paste(getBasePath("data"),"secure_humans_idx.csv",sep='') ,
          row.names=FALSE)

######### MAKING FEATURES 

## N. auctions a bidder partecipated 
cat(">>> Making feature: N. auctions a bidder partecipated ... \n")
bidder_auctions.train = ddply(train.full[,c('bidder_id','auction')],
                        .(bidder_id) , function(x) c(auct.num = length(unique(x$auction))) )
bidder_auctions.test = ddply(test.full[,c('bidder_id','auction')],
                              .(bidder_id) , function(x) c(auct.num = length(unique(x$auction))) )

X = rbind(bidder_auctions.train,bidder_auctions.test)

trind = 1:nrow(bidder_auctions.train)
teind = (nrow(bidder_auctions.train)+1):nrow(X)

cat(">> bidder.auctions:",nrow(X),"rows\n")
cat(">> train+test:",nrow(train)+nrow(test),"\n")
cat(">> diff:",nrow(X)-nrow(train)-nrow(test),"\n")
cat(">> bidders in test set that never made a bid:",length(unique(test[! test$bidder_id %in% bids$bidder_id,]$bidder_id)),"\n")
cat(">> bidders in train set that never made a bid:",length(unique(train[! train$bidder_id %in% bids$bidder_id,]$bidder_id)),"\n")

rm(bidder_auctions.train)
rm(bidder_auctions.test)

### Auctions a bidder partecipated 
cat(">>> Making feature: type of auctions a bidder partecipated ... \n")
auctions_train = unique(train.full$auction)
cat(">> we have",length(auctions_train),"different auctions on train set\n")

## train 
bidder_auctions.train = ddply(train.full[,c('bidder_id','auction')],
                              .(bidder_id) , function(x)  {
                                ddply(x, .(auction) , function(xx) {
                                   length(xx[,1])
                                } ) 
                              })
colnames(bidder_auctions.train)[3]='num'

## test 
bidder_auctions.test = ddply(test.full[,c('bidder_id','auction')],
                              .(bidder_id) , function(x)  {
                                ddply(x, .(auction) , function(xx) {
                                  length(xx[,1])
                                } ) 
                              })
colnames(bidder_auctions.test)[3]='num'

##
auction.outcome = ddply(train.full , .(auction) , function(x) c(zero=sum(x$outcome==0),one=sum(x$outcome==1)) )
auction.outcome$ratio = auction.outcome$one/(auction.outcome$zero+auction.outcome$one)
auction.outcome = auction.outcome[order(auction.outcome$ratio , decreasing = T) , ]
cat(">>> auctions with only robots:",sum(auction.outcome$ratio==1),"\n")
cat(">>> auctions with only humans:",sum(auction.outcome$ratio==0),"\n")
describe(auction.outcome$ratio)
auction.outcome$ratio_lev = cut2(auction.outcome$ratio , g = 20 )
print(table(auction.outcome$ratio_lev))
auction.outcome$ratio_lev_int = unlist(lapply(auction.outcome$ratio_lev, function(x) {
  i = 0
  for (i in 1:12) 
    if (x == levels(x)[[i]]) 
      break 
  return(i)
}))
print(head(auction.outcome))

##
bidder_auctions.train.full = merge(x=bidder_auctions.train,y=auction.outcome,by='auction')
bidder_auctions.test.full = merge(x=bidder_auctions.test,y=auction.outcome,by='auction')

## Making feature 
bidder_auctions_types = as.data.frame(matrix(rep(0,nrow(X)*length(unique(auction.outcome$ratio_lev_int))),
                                   nrow=nrow(X),
                                   ncol=length(unique(auction.outcome$ratio_lev_int))))
colnames(bidder_auctions_types) = paste0("auct_lev",1:ncol(bidder_auctions_types))

## Train
cat(">>> making features for train set ...\n")
for (bid in trind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_auctions.train.full[bidder_auctions.train.full$bidder_id == bidder_id , c('ratio_lev_int','num')] , 
        .(ratio_lev_int) , function(x) c(num=sum(x[2])))
    
  bidder_auctions_types[bid,aa$ratio_lev_int] = aa$num 
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(trind),"]..")
}

## Test train features 
bid = sample(trind,1)
cat("\n>>>Testing train features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_auctions.train.full[bidder_auctions.train.full$bidder_id == bidder_id , c('ratio_lev_int','num')])
print(bidder_auctions_types[bid,])
if (sum(bidder_auctions_types[bid,]) == 
      sum(bidder_auctions.train.full[bidder_auctions.train.full$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}

## Test
cat(">>> making features for test set ...\n")
for (bid in teind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_auctions.test.full[bidder_auctions.test.full$bidder_id == bidder_id , c('ratio_lev_int','num')] , 
             .(ratio_lev_int) , function(x) c(num=sum(x[2])))
  
  bidder_auctions_types[bid,aa$ratio_lev_int] = aa$num 
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(teind),"]..")
}

## Test test features 
bid = sample(teind,1)
cat("\n>>>Testing test features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_auctions.test.full[bidder_auctions.test.full$bidder_id == bidder_id , c('ratio_lev_int','num')])
print(bidder_auctions_types[bid,])
if (sum(bidder_auctions_types[bid,]) == 
      sum(bidder_auctions.test.full[bidder_auctions.test.full$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}

### making percentages 
sums = as.numeric(apply(bidder_auctions_types,1,function(x) sum(x[1:ncol(bidder_auctions_types)]) ))
bidder_auctions_types = bidder_auctions_types / sums
sums_after = as.numeric(apply(bidder_auctions_types,1,function(x) sum(x[1:ncol(bidder_auctions_types)]) ))
if( sum( sums_after > 1.001) > 0 || sum( sums_after < 0.999) > 0 )
  stop("something wrong")

### Binding features 
X=cbind(X,bidder_auctions_types)

cat(">>> storing on disk ...\n")
print(head(X))
write.csv(X,quote=FALSE, 
          file=paste(getBasePath("data"),"X1.csv",sep='') ,
          row.names=FALSE)








