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

cat(">>> secure_humans_idx:  \n")
print(secure_humans_idx)

cat("********** storing on disk \n")
write.csv(data.frame(index = secure_humans_idx),
          quote=FALSE, 
          file=paste(getBasePath("data"),"secure_humans_idx.csv",sep='') ,
          row.names=FALSE)

######### MAKING FEATURES 

## N. auctions a bidder partecipated 
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

## Auctions a bidder partecipated 
auctions_train = unique(train.full$auction)
cat(">> we have",length(auctions_train),"different auctions on train set\n")

bidder_auctions.train = ddply(train.full[,c('bidder_id','auction')],
                              .(bidder_id) , function(x)  {
                                ddply(x, .(auction) , function(xx) {
                                   length(xx[,1])
                                } ) 
                              })
colnames(bidder_auctions.train)[3]='num'
l = encodeCategoricalFeature ( bidder_auctions.train$auction , NULL , 'auction')
tr = l[[1]]
ts = l[[2]]





