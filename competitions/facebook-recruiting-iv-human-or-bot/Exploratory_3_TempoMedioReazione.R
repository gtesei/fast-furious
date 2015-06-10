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
bids = bids[order(bids$auction , bids$time, decreasing = F), ]
bids$id = 1:nrow(bids)

#bids_matrix = as.matrix(bids) 


X = as.data.frame( fread(paste(getBasePath("data") , 
                               "Xfin5.csv" , sep='')))

######## exploring train set ... 
cat("************* MEAN TIME REACTION *************\n")
all_auctions = unique(bids$auction)
cat(">> there are ",length(all_auctions), "different auctions - average bids x auctions = ",nrow(bids)/length(all_auctions),"... \n" )

all_bidders = (X$bidder_id)
cat(">> there are ",length(all_bidders), "different bidders - average bidders x auctions = ",nrow(bids)/length(all_bidders),"... \n" )


### bids matrices 
react_matrix = matrix(0, nrow = length(all_bidders)  , ncol = 3  )
rownames(react_matrix) = all_bidders 
colnames(react_matrix) = c("react_vs_all","react_vs_me","react_vs_other") 

for (i in 1:nrow(X) ) {
  bidder_id = X[i,]$bidder_id 
  
  data = bids[bids$bidder_id == bidder_id , ]
  bid.num = nrow(data)
  auctions = unique(data$auction)
  
  cat (">>> processing ",bidder_id," [",i,"/",length(all_bidders),"] [ auctions:",length(auctions),"] [ bid.num:",bid.num,"]... \n")
  
  seq_vs_all = NULL
  seq_vs_me = NULL
  seq_vs_other = NULL
  for (j in 1:nrow(data)) {
    id = data[j,'id']
    time = data[j,'time']
    auction = data[j,'auction']
    
    if (id == 1) next 
    
    id_prev = id-1
    #bid_previous = bids[bids$id == id_prev,]
    bid_previous = bids[id_prev,]
    
    time_prev = bid_previous$time
    bidder_id_prev = bid_previous$bidder_id 
    auction_prev = bid_previous$auction    

   bid_previous 
    
    
    if (auction != auction_prev) next  
    
    react_time = time - time_prev
    
    seq_vs_all = c(seq_vs_all,react_time)
    
    if (bidder_id != bidder_id_prev) {
      seq_vs_other = c(seq_vs_other,react_time)
    } else {
      seq_vs_me = c(seq_vs_me,react_time)
    }
  }
  
  ## update matrix
  react_matrix[i,"react_vs_all"] = ifelse(is.null(seq_vs_all) , NA , mean(seq_vs_all))
  react_matrix[i,"react_vs_me"] = ifelse(is.null(seq_vs_me) , NA , mean(seq_vs_me))
  react_matrix[i,"react_vs_other"] = ifelse(is.null(seq_vs_other) , NA , mean(seq_vs_other))
  
  cat ("> react_vs_all =",react_matrix[i,"react_vs_all"]," - react_vs_me =",react_matrix[i,"react_vs_me"], " - react_vs_other =",react_matrix[i,"react_vs_other"],"...\n")
  
  if (i %% 1000 == 0) {
    cat("\n>>> writing on disk ... \n")
    write.csv(as.data.frame(react_matrix),quote=FALSE, 
              file=paste(getBasePath("data"),"react_matrix.csv",sep='') ,
              row.names=F)
  }
  
  if (i == 20) {
    print(head(react_matrix,20))
  }
}

########### handling NAs 
cat("\n>>> Handling NAs ...\n")

## react_vs_all
nas = sum(is.na(react_matrix[,'react_vs_all']))
avg = mean(react_matrix[,'react_vs_all'] , na.rm = T) 
cat(">> react_vs_all has ",nas," NAs and average ",avg,"  BEFORE filling them with the average ..\n")

idx.na = as.integer(which(is.na(react_matrix[,'react_vs_all'])))
react_matrix[idx.na,'react_vs_all'] = avg 

nas = sum(is.na(react_matrix[,'react_vs_all']))
avg = mean(react_matrix[,'react_vs_all'] ) 
cat(">> react_vs_all has ",nas," NAs and average ",avg,"  AFTER ..\n")

## react_vs_me
nas = sum(is.na(react_matrix[,'react_vs_me']))
avg = mean(react_matrix[,'react_vs_me'] , na.rm = T) 
cat(">> react_vs_me has ",nas," NAs and average ",avg,"  BEFORE filling them with 0 ..\n")

idx.na = as.integer(which(is.na(react_matrix[,'react_vs_me'])))
react_matrix[idx.na,'react_vs_me'] = 0 

nas = sum(is.na(react_matrix[,'react_vs_me']))
avg = mean(react_matrix[,'react_vs_me'] ) 
cat(">> react_vs_me has ",nas," NAs and average ",avg,"  AFTER ..\n")

## react_vs_other
nas = sum(is.na(react_matrix[,'react_vs_other']))
avg = mean(react_matrix[,'react_vs_other'] , na.rm = T) 
cat(">> react_vs_other has ",nas," NAs and average ",avg,"  BEFORE filling them with the average ..\n")

idx.na = as.integer(which(is.na(react_matrix[,'react_vs_other'])))
react_matrix[idx.na,'react_vs_other'] = avg 

nas = sum(is.na(react_matrix[,'react_vs_other']))
avg = mean(react_matrix[,'react_vs_other'] ) 
cat(">> react_vs_other has ",nas," NAs and average ",avg,"  AFTER ..\n")

## check 
nas = sum( is.na(react_matrix) )
cat(">>> final check: react_matrix has ",nas," NAs \n")

########### storing on disk 
cat("\n>>> writing on disk ... \n")
write.csv(as.data.frame(react_matrix),quote=FALSE, 
          file=paste(getBasePath("data"),"react_matrix.csv",sep='') ,
          row.names=F)

