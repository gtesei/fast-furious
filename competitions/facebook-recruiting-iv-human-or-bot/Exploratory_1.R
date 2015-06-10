library(caret)
library(Hmisc)
library(data.table)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)
library(fBasics)

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

#######

sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

train = as.data.frame( fread(paste(getBasePath("data") , 
                                   "train.csv" , sep=''))) 

test = as.data.frame( fread(paste(getBasePath("data") , 
                                  "test.csv" , sep='')))

bids = as.data.frame( fread(paste(getBasePath("data") , 
                                  "bids.csv" , sep='')))

######## exploring train set ... 
train.full = merge(x = bids , y = train , by="bidder_id")
train.full = train.full[order(train.full$auction , train.full$time, decreasing = F),]

auctions = unique(train.full$auction)
cat(">> there are ",length(auctions), "different auctions - average bids x auctions = ",dim(train.full)[1]/length(auctions),"... \n" )


######## distribution of 1s and 0s in time ... 
time.bids = data.frame(auction = auctions , 
                       num.0 = NA , 
                       num.1 = NA , 
                       time.min = NA , 
                       time.max = NA, 
                       time.delta = NA, 
                       time.delta.perc = NA , 
                       time.mean = NA , 
                       time.mean.perc = NA , 
                       time.0.mean = NA , 
                       time.1.mean = NA , 
                       time.0.mean.perc = NA , 
                       time.1.mean.perc = NA , 
                       time.0.sd = NA , 
                       time.1.sd = NA, 
                       
                       min.time.seq = NA, 
                       bid.fastest = NA, 
                       bidder.fastest = NA, 
                       bidder.last = NA 
                       )
i = 0 
for (auct in auctions ) {
  ## auct (e.g. 38di3)
  i = i + 1 
  cat (">>> processing ",auct," [",i,"/",length(auctions),"] ... ")
  data = train.full[train.full$auction == auct , ]
  
  ###
  bidders = bids[bids$auction == auct , ]$bidder_id 
  times = bids[bids$auction == auct , ]$time 
  times.r = shift(v = times , 1)
  min.time.seq = min( (times - times.r)[-1] )
  bid.fastest = min(which(   (times - times.r)[-1]  ==   min( (times - times.r)[-1] ) ) + 1 )
  
  bidder.fastest = bidders[bid.fastest]
  bidder.last = bidders[length(bidders)]
  
  
  ###
  num.0 = sum(data$outcome == 0) 
  num.1 = sum(data$outcome == 1)  
  time.min = min(bids[bids$auction == auct ,]$time)
  time.max = max(bids[bids$auction == auct ,]$time)
  time.delta =  time.max - time.min
  time.delta.perc = time.delta / time.max
  time.mean = mean(bids[bids$auction == auct ,]$time)
  time.mean.perc = time.mean / time.max
  time.0.mean = mean(data[data$outcome == 0 , ]$time)
  time.1.mean = mean(data[data$outcome == 1 , ]$time)
  time.0.mean.perc = time.0.mean / time.max
  time.1.mean.perc = time.1.mean /  time.max
  time.0.sd = sd(data[data$outcome == 0 , ]$time)
  time.1.sd = sd(data[data$outcome == 1 , ]$time)
  
  ### 
  time.bids[time.bids$auction == auct , ]$num.0 = num.0 
  time.bids[time.bids$auction == auct , ]$num.1 = num.1 
  time.bids[time.bids$auction == auct , ]$time.min = time.min 
  time.bids[time.bids$auction == auct , ]$time.max = time.max 
  time.bids[time.bids$auction == auct , ]$time.delta = time.delta 
  time.bids[time.bids$auction == auct , ]$time.delta.perc = time.delta.perc 
  time.bids[time.bids$auction == auct , ]$time.mean = time.mean 
  time.bids[time.bids$auction == auct , ]$time.mean.perc = time.mean.perc 
  time.bids[time.bids$auction == auct , ]$time.0.mean = time.0.mean 
  time.bids[time.bids$auction == auct , ]$time.1.mean = time.1.mean 
  time.bids[time.bids$auction == auct , ]$time.0.mean.perc = time.0.mean.perc 
  time.bids[time.bids$auction == auct , ]$time.1.mean.perc = time.1.mean.perc 
  time.bids[time.bids$auction == auct , ]$time.0.sd = time.0.sd 
  time.bids[time.bids$auction == auct , ]$time.1.sd = time.1.sd 
    
  time.bids[time.bids$auction == auct , ]$min.time.seq = min.time.seq
  time.bids[time.bids$auction == auct , ]$bid.fastest = bid.fastest
  time.bids[time.bids$auction == auct , ]$bidder.fastest = bidder.fastest
  time.bids[time.bids$auction == auct , ]$bidder.last = bidder.last
  
  if (i %% 1000 == 0) {
    cat("\n>>> writing on disk ... \n")
    write.csv(time.bids,file=paste(getBasePath("data") , 
                                   "time_bids01.csv" , sep=''), quote=FALSE,row.names=FALSE)
  }
  
  if (i == 5) {
    print(head(time.bids))
  }
}

time.bids$num.tot = time.bids$num.0 + time.bids$num.1

cat("\n>>> writing on disk ... \n")
write.csv(time.bids,file=paste(getBasePath("data") , 
                          "time_bids01.csv" , sep=''), quote=FALSE,row.names=FALSE)
