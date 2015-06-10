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
                                   "sub_BEST_LB_0.92.csv" , sep='')))

#sub_best = as.data.frame( fread(paste(getBasePath("data") , 
#                                      "sub_xgboost_stress_xval0.916393.csv" , sep='')))


#### best performant feature set 
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

## auction.outcome
auction.outcome = ddply(train.full , .(auction) , function(x) c(zero=sum(x$outcome==0),one=sum(x$outcome==1)) )
auction.outcome$ratio = auction.outcome$one/(auction.outcome$zero+auction.outcome$one)
auction.outcome = auction.outcome[order(auction.outcome$ratio , decreasing = T) , ]
cat(">>> auctions with only robots:",length(unique(auction.outcome[auction.outcome$ratio==1,]$auction)),"\n")
describe(auction.outcome$ratio)
only_robot_auctions = unique(auction.outcome[auction.outcome$ratio==1,]$auction)

bidder_in_only_robot_auctions = unique(bids[bids$auction %in% only_robot_auctions,]$bidder_id)

## device.outcome
device.outcome = ddply(train.full , .(device) , function(x) c(zero=sum(x$outcome==0),one=sum(x$outcome==1)) )
device.outcome$ratio = device.outcome$one/(device.outcome$zero+device.outcome$one)
device.outcome = device.outcome[order(device.outcome$ratio , decreasing = T) , ]
cat(">>> devices with only robots:",sum(device.outcome$ratio==1),"\n")
cat(">>> devices with only humans:",sum(device.outcome$ratio==0),"\n")
describe(device.outcome$ratio)
only_robot_device = unique(device.outcome[device.outcome$ratio==1,]$device)

bidder_with_only_robot_device = unique(bids[bids$device %in% only_robot_device,]$bidder_id)

## country.outcome
country.outcome = ddply(train.full , .(country) , function(x) c(zero=sum(x$outcome==0),one=sum(x$outcome==1)) )
country.outcome$ratio = country.outcome$one/(country.outcome$zero+country.outcome$one)
country.outcome = country.outcome[order(country.outcome$ratio , decreasing = T) , ]
cat(">>> country with only robots:",sum(country.outcome$ratio==1),"\n")
cat(">>> country with only humans:",sum(country.outcome$ratio==0),"\n")
describe(country.outcome$ratio)
only_robot_country = unique(country.outcome[country.outcome$ratio==1,]$country)

bidder_with_only_robot_country = unique(bids[bids$country %in% only_robot_country,]$bidder_id)

####

mean(avg_bids[avg_bids$bidder_id %in% bidder_with_only_robot_device , ]$avg_bids)  ##31.08629


bidder_with_device_robots_and_30avg = avg_bids[avg_bids$bidder_id %in% bidder_with_only_robot_device & avg_bids$avg_bids > 30, ]$bidder_id 

sub_best[sub_best$bidder_id %in% bidder_with_device_robots_and_30avg , ]$prediction = 1

mean(sub_best[sub_best$bidder_id %in% bidder_with_only_robot_device , ]$prediction) ## 0.1525629 , 291 bidders



sub_best[sub_best$bidder_id %in% bidder_with_only_robot_country , ]$prediction

print(">> prediction <<")
print(mean(sub_best$prediction))

print(">> train set labels <<")
print(mean(y))

write.csv(sub_best,quote=FALSE, 
          file=paste(getBasePath("data"),"sub_bidder_1_smanett.csv",sep='') ,
          row.names=FALSE)







