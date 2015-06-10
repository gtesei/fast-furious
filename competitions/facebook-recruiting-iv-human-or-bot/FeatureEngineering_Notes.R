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

#train.full$merchandise = factor(train.full$merchandise , levels = gsub(" ", "_", unique(train.full$merchandise)) )
#train.full$auction = factor(train.full$auction , levels = gsub(" ", "_", unique(train.full$auction)) )

#### auction level 
train.merchandise = ddply(train.full , .(auction,merchandise)  , function(x)  c(  ratio = sum(x$outcome==1)/length(x$outcome) )   )

l = encodeCategoricalFeature (train.merchandise$merchandise , NULL , colname.prefix = paste0("merchandise") 
                              , asNumericSequence=F
                              , replaceWhiteSpaceInLevelsWith='_')
tr = l[[1]]
ts = l[[2]]

train.merchandise.2 = cbind(train.merchandise,tr)

#toRemove = c(toRemove,i)

bidder.auctions = ddply(train.full,.(bidder_id) , function(x) c(auctions=length(unique(x$auction))) )
bidder.merchandise = ddply(train.full,.(bidder_id) , function(x) c(merchandise=length(unique(x$merchandise)) ))
bidder.merchandise = bidder.merchandise[order(bidder.merchandise$merchandise,decreasing = T),]

bidder.auctions = ddply(train.full,.(bidder_id,outcome) , function(x) c(auctions=length(unique(x$auction))) )

ddply(bidder.auctions , .(outcome) , function(x) c(auction.mean = mean(x$auctions) , auction.sd=sd(x$auctions) ) )

auction.outcome = ddply(train.full , .(auction) , function(x) c(zero=sum(x$outcome==0),one=sum(x$outcome==1)) )
auction.outcome$ratio = auction.outcome$one/(auction.outcome$zero+auction.outcome$one)
auction.outcome = auction.outcome[order(auction.outcome$ratio , decreasing = T) , ]
sum(auction.outcome$ratio==1)
sum(auction.outcome$ratio==0)
describe(auction.outcome)

####
auct_not_train = unique(test.full[! test.full$auction %in% train.full$auction,]$auction)
auct_train = unique(test.full[test.full$auction %in% train.full$auction,]$auction)

bidder_auct_not_train = unique(test.full[! test.full$auction %in% train.full$auction,]$bidder_id)

bidder_auct_train = unique(test.full[test.full$auction %in% train.full$auction,]$bidder_id)

bidder_auct_not_train[!bidder_auct_not_train %in% bidder_auct_train]





