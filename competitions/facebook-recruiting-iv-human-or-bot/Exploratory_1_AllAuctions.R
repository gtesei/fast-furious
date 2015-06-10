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
bids = as.data.frame( fread(paste(getBasePath("data") , 
                                  "bids.csv" , sep='')))
bids = bids[order(bids$auction , bids$time, decreasing = F),]

######## exploring train set ... 
auctions = unique(bids$auction)
cat(">> there are ",length(auctions), "different auctions - average bids x auctions = ",nrow(bids)/length(auctions),"... \n" )


######## distribution of 1s and 0s in time ... 
time.bids = data.frame(auction = auctions , 
                       
                       bid.num = NA, 
                       
                       time.min = NA , 
                       time.max = NA, 
                       time.delta = NA, 
                       time.delta.perc = NA , 
                       time.mean = NA , 
                       time.mean.perc = NA , 
                       
                       min.time.seq = NA, 
                       bidder.fastest = NA, 
                       
                       max.time.seq = NA, 
                       bidder.slowest = NA, 
                       
                       bidder.last.5 = NA, 
                       bidder.last.4 = NA, 
                       bidder.last.3 = NA, 
                       bidder.last.2 = NA, 
                       bidder.last.1 = NA, 
                       
                       bidder.last = NA,  
                       bidder.first = NA,  
                       
                       bidder.seqlongest  = NA, 
                       seq_long_max = NA
                       )
i = 0 
for (auct in auctions ) {
  ## auct (e.g. 38di3)
  i = i + 1 
  cat (">>> processing ",auct," [",i,"/",length(auctions),"] ... ")
  data = bids[bids$auction == auct , ]
  
  ###
  bid.num = nrow(bids[bids$auction == auct , ])
  
  ###
  bidders = bids[bids$auction == auct , ]$bidder_id 
  times = bids[bids$auction == auct , ]$time 
  times.r = shift(v = times , 1)
  
  min.time.seq = min( (times - times.r)[-1] )
  max.time.seq = max( (times - times.r)[-1] )
  
  bid.fastest = min(which(  (times - times.r)[-1]  == min.time.seq ) + 1 )
  bid.slowest = min(which(  (times - times.r)[-1]  == max.time.seq ) + 1 )
  
  bidder.fastest = bidders[bid.fastest]
  bidder.slowest = bidders[bid.slowest]
  
  bidder.last.5 = ifelse(length(bidders) > 5, bidders[length(bidders)-5] , NA)
  bidder.last.4 = ifelse(length(bidders) > 4, bidders[length(bidders)-4] , NA)
  bidder.last.3 = ifelse(length(bidders) > 3, bidders[length(bidders)-3] , NA)
  bidder.last.2 = ifelse(length(bidders) > 2, bidders[length(bidders)-2] , NA)
  bidder.last.1 = ifelse(length(bidders) > 1, bidders[length(bidders)-1] , NA)
  
  bidder.last = bidders[length(bidders)]
  bidder.first = ifelse(length(bidders) > 0, bidders[1] , NA)
  
  ## bidder.seqlongest
  bidder.seqlongest = curr.bidder = data$bidder_id[1] 
  seq_long_max = curr.long = 1
  
  for (bbs in data$bidder_id[-1]) {
    if (bbs == curr.bidder) {
      curr.long = curr.long + 1 
    } else {
      if (curr.long > seq_long_max) {
        seq_long_max = curr.long 
        bidder.seqlongest = curr.bidder 
      } else {
        
      }
      curr.long = 1 
      curr.bidder = bbs
    }
  }
  
  ######
  time.min = min(bids[bids$auction == auct ,]$time)
  time.max = max(bids[bids$auction == auct ,]$time)
  time.delta =  time.max - time.min
  time.delta.perc = time.delta / time.max
  time.mean = mean(bids[bids$auction == auct ,]$time)
  time.mean.perc = time.mean / time.max
  
  ##################################### 
  time.bids[time.bids$auction == auct , ]$bid.num = bid.num 
  
  time.bids[time.bids$auction == auct , ]$time.min = time.min 
  time.bids[time.bids$auction == auct , ]$time.max = time.max 
  time.bids[time.bids$auction == auct , ]$time.delta = time.delta 
  time.bids[time.bids$auction == auct , ]$time.delta.perc = time.delta.perc 
  time.bids[time.bids$auction == auct , ]$time.mean = time.mean 
  time.bids[time.bids$auction == auct , ]$time.mean.perc = time.mean.perc 
  
  ###
  time.bids[time.bids$auction == auct , ]$min.time.seq = min.time.seq
  time.bids[time.bids$auction == auct , ]$bidder.fastest = bidder.fastest
  
  time.bids[time.bids$auction == auct , ]$max.time.seq = max.time.seq
  time.bids[time.bids$auction == auct , ]$bidder.slowest = bidder.slowest
  
  time.bids[time.bids$auction == auct , ]$bidder.last.5 = bidder.last.5
  time.bids[time.bids$auction == auct , ]$bidder.last.4 = bidder.last.4
  time.bids[time.bids$auction == auct , ]$bidder.last.3 = bidder.last.3
  time.bids[time.bids$auction == auct , ]$bidder.last.2 = bidder.last.2
  time.bids[time.bids$auction == auct , ]$bidder.last.1 = bidder.last.1
  
  time.bids[time.bids$auction == auct , ]$bidder.last = bidder.last
  time.bids[time.bids$auction == auct , ]$bidder.first = bidder.first
  
  time.bids[time.bids$auction == auct , ]$bidder.seqlongest = bidder.seqlongest
  time.bids[time.bids$auction == auct , ]$seq_long_max = seq_long_max
  
  if (i %% 1000 == 0) {
    cat("\n>>> writing on disk ... \n")
    write.csv(time.bids,file=paste(getBasePath("data") , 
                                   "time_bids_all_auctions2.csv" , sep=''), quote=FALSE,row.names=FALSE)
  }
  
  if (i == 5) {
    print(head(time.bids))
  }
}

cat("\n>>> writing on disk ... \n")
write.csv(time.bids,file=paste(getBasePath("data") , 
                          "time_bids_all_auctions2.csv" , sep=''), quote=FALSE,row.names=FALSE)
