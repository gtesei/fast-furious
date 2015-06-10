library(caret)
library(Hmisc)
library(data.table)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)
library(fBasics)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

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

X = as.data.frame( fread(paste(getBasePath("data") , 
                               "Xfin4.csv" , sep='')))

######## exploring train set ... 
all_auctions = unique(bids$auction)
cat(">> there are ",length(all_auctions), "different auctions - average bids x auctions = ",nrow(bids)/length(all_auctions),"... \n" )

all_bidders = (X$bidder_id)
cat(">> there are ",length(all_bidders), "different bidders - average bidders x auctions = ",nrow(bids)/length(all_bidders),"... \n" )


### bids matrices 
seq_matrix = matrix(0, nrow = length(all_bidders)  , ncol = 4  )
rownames(seq_matrix) = all_bidders 
colnames(seq_matrix) = c("seq_min","seq_max","seq_mean","seq_mode") 

for (i in 1:nrow(X) ) {
  bidder_id = X[i,]$bidder_id 
  
  data = bids[bids$bidder_id == bidder_id , ]
  bid.num = nrow(data)
  auctions = unique(data$auction)
  bidder_i = data.frame(auction = auctions , seq_min = 0 , seq_max = 0 , seq_mean = 0 , seq_mode = 0)
  
  cat (">>> processing ",bidder_id," [",i,"/",length(all_bidders),"] [ auctions:",length(auctions),"] [ bid.num:",bid.num,"]... \n")
  
  ### processing auctions 
  for (auction in auctions) { 
    data_auction = data[data$auction == auction , ]
    bidder.vect = bids[bids$auction == auction , ]$bidder_id
    ############################ compute seq_min / seq_max for that auction 
    max.seq = 1
    min.seq = nrow(data_auction)
    curr = 0 
    seq = NULL
    for (bbd in bidder.vect) {
      if (bbd == bidder_id) {
        curr = curr + 1 
      } else {
        if (curr > 0 & curr > max.seq)  {
          max.seq = curr 
        } 
        if (curr > 0 & curr < min.seq) {
          min.seq = curr 
        }
        if(curr>0) {
          seq = c(seq,curr)
        }
        curr = 0 
      } 
    }
    
    if (curr>0) {
      seq = c(seq,curr)
      
      if (curr > 0 & curr > max.seq)  {
        max.seq = curr 
      } 
      if (curr > 0 & curr < min.seq) {
        min.seq = curr 
      }
      curr = 0
    }
    ############################
    if (min(seq) != min.seq | max(seq) != max.seq) {
      stop("something wrong") 
    }
    
    bidder_i[bidder_i$auction == auction,]$seq_min = min.seq 
    bidder_i[bidder_i$auction == auction,]$seq_max = max.seq 
    bidder_i[bidder_i$auction == auction,]$seq_mean = mean(seq) 
    bidder_i[bidder_i$auction == auction,]$seq_mode = Mode(seq) 
  } ### end of processing auctions
  
  ## update matrix
  seq_matrix[i,"seq_min"] = mean(bidder_i$seq_min)
  seq_matrix[i,"seq_max"] = mean(bidder_i$seq_max)
  seq_matrix[i,"seq_mean"] = mean(bidder_i$seq_mean)
  seq_matrix[i,"seq_mode"] = Mode(bidder_i$seq_mode)
  
  if (i %% 1000 == 0) {
    cat("\n>>> writing on disk ... \n")
    write.csv(as.data.frame(seq_matrix),quote=FALSE, 
              file=paste(getBasePath("data"),"seq_matrix.csv",sep='') ,
              row.names=F)
  }
  
  if (i == 20) {
    print(head(seq_matrix,20))
  }
}

cat("\n>>> writing on disk ... \n")
write.csv(as.data.frame(seq_matrix),quote=FALSE, 
          file=paste(getBasePath("data"),"seq_matrix.csv",sep='') ,
          row.names=F)

