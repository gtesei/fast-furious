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

source(paste0( getBasePath("process") , "/Classification_Lib.R"))
#################
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep=''))) ## outcome = 0 human 

train = as.data.frame( fread(paste(getBasePath("data") , 
                                   "train.csv" , sep=''))) 

bids = as.data.frame( fread(paste(getBasePath("data") , 
                                  "bids.csv" , sep='')))

sub_best = as.data.frame( fread(paste(getBasePath("data") , 
                                  "sub_best_LB0.914.csv" , sep='')))


####  best performant feature set 
X = as.data.frame( fread(paste(getBasePath("data") , 
                               "Xfin5.csv" , sep='')))

train.full = merge(x = bids , y = train , by="bidder_id"  )
trind = 1:length(unique(train.full$bidder_id))
teind = (max(trind)+1):nrow(X)

X.full = merge(x=X , y=train , by="bidder_id")
y = X.full$outcome
y.cat = factor(y)
levels(y.cat) = c("human","robot")

################# 

## auction and number of bids a bidder made in that auction 
bidder_auction_full = ddply(bids , .(bidder_id,auction) , function(x)  c(bids_num_by_bidder = length(x$bid_id) ) )

## auctions a bidder partecipated in 
bidder_auction = ddply(bids , .(bidder_id) , function(x)  c(auction_num = length(unique(x$auction)) ) )
bidder_auction = bidder_auction[order(bidder_auction$auction_num , decreasing = T), ]

## number of bids per auction 
bids_auction = ddply(bids , .(auction) , function(x)  c(bids_num = length(x$bid_id) ) )
bids_auction = bids_auction[order(bids_auction$bids_num, decreasing = T),]

## average number of bids a bidder made on auctions 
avg_bids = data.frame(bidder_id = X$bidder_id , avg_bids = X$avg_bids)

avg_bids_500 = avg_bids[which(avg_bids$avg_bids > 500) , ]
table(train[train$bidder_id %in% avg_bids_500$bidder_id , ]$outcome)/length(train[train$bidder_id %in% avg_bids_500$bidder_id , ]$outcome)
length(train[train$bidder_id %in% avg_bids_500$bidder_id , ]$bidder_id) 

length(sub_best[sub_best$bidder_id %in% avg_bids_500$bidder_id , ]$prediction) 
mean(sub_best[sub_best$bidder_id %in% avg_bids_500$bidder_id , ]$prediction)
max(sub_best[sub_best$bidder_id %in% avg_bids_500$bidder_id , ]$prediction)
min(sub_best[sub_best$bidder_id %in% avg_bids_500$bidder_id , ]$prediction)

### merge 
M1 = merge(x=avg_bids , y = bidder_auction , by='bidder_id' )
M2 = merge(x=bidder_auction_full,y=bids_auction,by='auction')
M3 = merge(x=M2,y=M1,by='bidder_id')

bidder_1_auction_small = M3[M3$avg_bids < 2 & M3$auction_num < 10 & M3$bids_num < 5000,]$bidder_id

table(train[train$bidder_id %in% bidder_1_auction_small , ]$outcome)/length(train[train$bidder_id %in% bidder_1_auction_small , ]$outcome)
length(train[train$bidder_id %in% bidder_1_auction_small , ]$bidder_id) 
length(train[train$bidder_id %in% bidder_1_auction_small & train$outcome == 1, ]$bidder_id) 

length(sub_best[sub_best$bidder_id %in% bidder_1_auction_small , ]$prediction) 
mean(sub_best[sub_best$bidder_id %in% bidder_1_auction_small , ]$prediction)
max(sub_best[sub_best$bidder_id %in% bidder_1_auction_small , ]$prediction)
min(sub_best[sub_best$bidder_id %in% bidder_1_auction_small , ]$prediction)

###
bidder_1 = M1[which(M1$avg_bids == 1 & M1$auction_num == 1) , ]$bidder_id 
## 0.98344371 0.01655629 
table(train[train$bidder_id %in% bidder_1 , ]$outcome)/length(train[train$bidder_id %in% bidder_1 , ]$outcome)
train[train$bidder_id %in% bidder_1 & train$outcome==1, ]$bidder_id

mean(sub_best[sub_best$bidder_id %in% bidder_1 , ]$prediction)
max(sub_best[sub_best$bidder_id %in% bidder_1 , ]$prediction)
min(sub_best[sub_best$bidder_id %in% bidder_1 , ]$prediction)


###
bidder_bid_1_in_auction_1 = M3[which(M3$auction_num == 1 & M3$avg_bids == 1 ),]$bidder_id
#0.98344371 0.01655629 
table(train[train$bidder_id %in% bidder_bid_1_in_auction_1 , ]$outcome)/length(train[train$bidder_id %in% bidder_bid_1_in_auction_1 , ]$outcome)
length(train[train$bidder_id %in% bidder_bid_1_in_auction_1 , ]$bidder_id)

length(sub_best[sub_best$bidder_id %in% bidder_bid_1_in_auction_1 , ]$prediction) ##754 ~ 18% ... che effetto fa se ci metti uno 0!! 
mean(sub_best[sub_best$bidder_id %in% bidder_bid_1_in_auction_1 , ]$prediction)
max(sub_best[sub_best$bidder_id %in% bidder_bid_1_in_auction_1 , ]$prediction)
min(sub_best[sub_best$bidder_id %in% bidder_bid_1_in_auction_1 , ]$prediction)

###### update 
#sub_best[sub_best$bidder_id %in% bidder_bid_1_in_auction_1 , ]$prediction = 0  ### peggiorativo !!! 
#sub_best[sub_best$bidder_id %in% avg_bids_500$bidder_id , ]$prediction = 1 ## non pervenuto
sub_best[sub_best$bidder_id %in% bidder_1_auction_small , ]$prediction = 0


print(">> prediction <<")
print(mean(sub_best$prediction))

print(">> train set labels <<")
print(mean(y))

write.csv(sub_best,quote=FALSE, 
          file=paste(getBasePath("data"),"sub_bidder_1_auction_small.csv",sep='') ,
          row.names=FALSE)







