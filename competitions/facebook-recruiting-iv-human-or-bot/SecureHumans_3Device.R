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
                                           "X2.csv" , sep='')))


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

## Making feature: N. device a bidder used 
cat(">>> Making feature: N. device a bidder used ... \n")
bidder_device.train = ddply(train.full[,c('bidder_id','outcome','device')],
                              .(bidder_id,outcome) , function(x) c(dev.num = length(unique(x$device))) )

cat(">>> on average robots uses ",mean(bidder_device.train[bidder_device.train$outcome==1,]$dev.num),"different devices [sd=",
    sd(bidder_device.train[bidder_device.train$outcome==1,]$dev.num),"] \n")
cat(">>> on average humans uses ",mean(bidder_device.train[bidder_device.train$outcome==0,]$dev.num),"different devices [sd=",
    sd(bidder_device.train[bidder_device.train$outcome==0,]$dev.num),"] \n")

bidder_device.train = ddply(train.full[,c('bidder_id','device')],
                            .(bidder_id) , function(x) c(dev.num = length(unique(x$device))) )
bidder_device.test = ddply(test.full[,c('bidder_id','device')],
                            .(bidder_id) , function(x) c(dev.num = length(unique(x$device))) )

bidder_device = rbind(bidder_device.train,bidder_device.test)
X = cbind(X,bidder_device[,'dev.num' , drop=F])

rm(bidder_device)
rm(bidder_device.train)
rm(bidder_device.test)

### Making feature: type of device a bidder used 
cat(">>> Making feature: type of device a bidder used  ... \n")
devices = unique(train.full$device)
cat(">> we have",length(devices),"different devices on train set\n")

## train 
bidder_device.train = ddply(train.full[,c('bidder_id','device')],
                              .(bidder_id) , function(x)  {
                                ddply(x, .(device) , function(xx) {
                                  length(xx[,1])
                                } ) 
                              })
colnames(bidder_device.train)[3]='num'

## test 
bidder_device.test = ddply(test.full[,c('bidder_id','device')],
                             .(bidder_id) , function(x)  {
                               ddply(x, .(device) , function(xx) {
                                 length(xx[,1])
                               } ) 
                             })
colnames(bidder_device.test)[3]='num'

##
device.outcome = ddply(train.full , .(device) , function(x) c(zero=sum(x$outcome==0),one=sum(x$outcome==1)) )
device.outcome$ratio = device.outcome$one/(device.outcome$zero+device.outcome$one)
device.outcome = device.outcome[order(device.outcome$ratio , decreasing = T) , ]
cat(">>> devices with only robots:",sum(device.outcome$ratio==1),"\n")
cat(">>> devices with only humans:",sum(device.outcome$ratio==0),"\n")
describe(device.outcome$ratio)
device.outcome$ratio_lev = cut2(device.outcome$ratio , cuts=c(0,0.002,0.02008,0.0371,0.0591,0.1035,0.2286,0.4462,0.6585,0.8378,0.90,0.999) )
print(table(device.outcome$ratio_lev))
device.outcome$ratio_lev_int = unlist(lapply(device.outcome$ratio_lev, function(x) {
  i = 0
  for (i in 1:12) 
    if (x == levels(x)[[i]]) 
      break 
  return(i)
}))
print(head(device.outcome))

##
bidder_device.train.full = merge(x=bidder_device.train,y=device.outcome,by='device')
bidder_device.test.full = merge(x=bidder_device.test,y=device.outcome,by='device')

## Making feature 
bidder_devicde_types = as.data.frame(matrix(rep(0,nrow(X)*length(unique(device.outcome$ratio_lev_int))),
                                             nrow=nrow(X),
                                             ncol=length(unique(device.outcome$ratio_lev_int))))
colnames(bidder_devicde_types) = paste0("dev_lev",1:ncol(bidder_devicde_types))

## Train
cat(">>> making features for train set ...\n")
for (bid in trind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_device.train.full[bidder_device.train.full$bidder_id == bidder_id , c('ratio_lev_int','num')] , 
             .(ratio_lev_int) , function(x) c(num=sum(x[2])))
  
  bidder_devicde_types[bid,aa$ratio_lev_int] = aa$num 
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(trind),"]..")
}

## Test train features 
bid = sample(trind,1)
cat("\n>>>Testing train features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_device.train.full[bidder_device.train.full$bidder_id == bidder_id , c('ratio_lev_int','num')])
print(bidder_devicde_types[bid,])
if (sum(bidder_devicde_types[bid,]) == 
      sum(bidder_device.train.full[bidder_device.train.full$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}

## Test
cat(">>> making features for train set ...\n")
for (bid in teind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_device.test.full[bidder_device.test.full$bidder_id == bidder_id , c('ratio_lev_int','num')] , 
             .(ratio_lev_int) , function(x) c(num=sum(x[2])))
  
  bidder_devicde_types[bid,aa$ratio_lev_int] = aa$num 
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(teind),"]..")
}

## Test train features 
bid = sample(teind,1)
cat("\n>>>Testing test features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_device.test.full[bidder_device.test.full$bidder_id == bidder_id , c('ratio_lev_int','num')])
print(bidder_devicde_types[bid,])
if (sum(bidder_devicde_types[bid,]) == 
      sum(bidder_device.test.full[bidder_device.test.full$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}

### making percentages 
sums = as.numeric(apply(bidder_devicde_types,1,function(x) sum(x[1:ncol(bidder_devicde_types)]) ))
sums0s.idx = which(sums == 0)

for (i in 1:length(sums)) {
  if (sums[i]==0) {
    cat(">> i == ",i,"sums == 0, ...\n")
  } else {
    bidder_devicde_types[i,] = bidder_devicde_types[i,]/sums[i]
  }
} 

sums_after = as.numeric(apply(bidder_devicde_types[-sums0s.idx,],1,function(x) sum(x[1:ncol(bidder_devicde_types)]) ))
if( sum( sums_after > 1.001) > 0 || sum( sums_after < 0.999) > 0 )
  stop("something wrong")

### Binding features 
X=cbind(X,bidder_devicde_types)

cat(">>> storing on disk ...\n")
print(head(X))
write.csv(X,quote=FALSE, 
          file=paste(getBasePath("data"),"X3.csv",sep='') ,
          row.names=FALSE)




