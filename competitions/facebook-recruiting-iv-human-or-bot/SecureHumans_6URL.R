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
                                           "X5.csv" , sep='')))


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

## Making feature: N. ips a bidder used 
cat(">>> Making feature: N. ips a bidder used ... \n")
bidder_url.train = ddply(train.full[,c('bidder_id','outcome','url')],
                              .(bidder_id,outcome) , function(x) c(num = length(unique(x$url))) )

cat(">>> on average robots uses ",mean(bidder_url.train[bidder_url.train$outcome==1,]$num),"different urls [sd=",
    sd(bidder_url.train[bidder_url.train$outcome==1,]$num),"] \n")
cat(">>> on average humans uses ",mean(bidder_url.train[bidder_url.train$outcome==0,]$num),"different urls [sd=",
    sd(bidder_url.train[bidder_url.train$outcome==0,]$num),"] \n")

bidder_url.train = ddply(train.full[,c('bidder_id','url')],
                            .(bidder_id) , function(x) c(url.num = length(unique(x$url))) )
bidder_url.test = ddply(test.full[,c('bidder_id','url')],
                            .(bidder_id) , function(x) c(url.num = length(unique(x$url))) )

bidder_url = rbind(bidder_url.train,bidder_url.test)
X = cbind(X,bidder_url[,'url.num' , drop=F])

rm(bidder_url)
rm(bidder_url.train)
rm(bidder_url.test)

### Making feature: type of url a bidder used 
cat(">>> Making feature: type of url a bidder used  ... \n")
urls = unique(train.full$url)
cat(">> we have",length(urls),"different urls on train set\n")

## train 
bidder_url.train = ddply(train.full[,c('bidder_id','url')],
                              .(bidder_id) , function(x)  {
                                ddply(x, .(url) , function(xx) {
                                  length(xx[,1])
                                } ) 
                              })
colnames(bidder_url.train)[3]='num'

## test 
bidder_url.test = ddply(test.full[,c('bidder_id','url')],
                             .(bidder_id) , function(x)  {
                               ddply(x, .(url) , function(xx) {
                                 length(xx[,1])
                               } ) 
                             })
colnames(bidder_url.test)[3]='num'

##
url.outcome = ddply(train.full , .(url) , function(x) c(zero=sum(x$outcome==0),one=sum(x$outcome==1)) )
url.outcome$ratio = url.outcome$one/(url.outcome$zero+url.outcome$one)
url.outcome = url.outcome[order(url.outcome$ratio , decreasing = T) , ]
cat(">>> url with only robots:",sum(url.outcome$ratio==1),"\n")
cat(">>> url with only humans:",sum(url.outcome$ratio==0),"\n")
describe(url.outcome$ratio)
url.outcome$ratio_lev = cut2(url.outcome$ratio , cuts=c(0,0.0004,0.1,0.4,0.85,0.993,1) )
print(table(url.outcome$ratio_lev))
url.outcome$ratio_lev_int = unlist(lapply(url.outcome$ratio_lev, function(x) {
  i = 0
  for (i in 1:12) 
    if (x == levels(x)[[i]]) 
      break 
  return(i)
}))
print(head(url.outcome))
cat("...\n")
print(tail(url.outcome))

##
bidder_url.train.full = merge(x=bidder_url.train,y=url.outcome,by='url')
bidder_url.test.full = merge(x=bidder_url.test,y=url.outcome,by='url')

## Making feature 
bidder_url_types = as.data.frame(matrix(rep(0,nrow(X)*length(unique(url.outcome$ratio_lev_int))),
                                             nrow=nrow(X),
                                             ncol=length(unique(url.outcome$ratio_lev_int))))
colnames(bidder_url_types) = paste0("url_lev",1:ncol(bidder_url_types))

## Train
cat(">>> making features for train set ...\n")
for (bid in trind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_url.train.full[bidder_url.train.full$bidder_id == bidder_id , c('ratio_lev_int','num')] , 
             .(ratio_lev_int) , function(x) c(num=sum(x[2])))
  
  bidder_url_types[bid,aa$ratio_lev_int] = aa$num 
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(trind),"]..")
}

## Test train features 
bid = sample(trind,1)
cat("\n>>>Testing train features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_url.train.full[bidder_url.train.full$bidder_id == bidder_id , c('ratio_lev_int','num')])
print(bidder_url_types[bid,])
if (sum(bidder_url_types[bid,]) == 
      sum(bidder_url.train.full[bidder_url.train.full$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}

## Test
cat(">>> making features for train set ...\n")
for (bid in teind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_url.test.full[bidder_url.test.full$bidder_id == bidder_id , c('ratio_lev_int','num')] , 
             .(ratio_lev_int) , function(x) c(num=sum(x[2])))
  
  bidder_url_types[bid,aa$ratio_lev_int] = aa$num 
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(teind),"]..")
}

## Test train features 
bid = sample(teind,1)
cat("\n>>>Testing test features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_url.test.full[bidder_url.test.full$bidder_id == bidder_id , c('ratio_lev_int','num')])
print(bidder_url_types[bid,])
if (sum(bidder_url_types[bid,]) == 
      sum(bidder_url.test.full[bidder_url.test.full$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}

### making percentages 
sums = as.numeric(apply(bidder_url_types,1,function(x) sum(x[1:ncol(bidder_url_types)]) ))
sums0s.idx = which(sums == 0)

for (i in 1:length(sums)) {
  if (sums[i]==0) {
    cat(">> i == ",i,"sums == 0, ...\n")
  } else {
    bidder_url_types[i,] = bidder_url_types[i,]/sums[i]
  }
} 

sums_after = as.numeric(apply(bidder_url_types[-sums0s.idx,],1,function(x) sum(x[1:ncol(bidder_url_types)]) ))
if( sum( sums_after > 1.001) > 0 || sum( sums_after < 0.999) > 0 )
  stop("something wrong")

### Binding features 
X=cbind(X,bidder_url_types)

cat(">>> storing on disk ...\n")
print(head(X))
cat("...\n")
print(tail(X))

write.csv(X,quote=FALSE, 
          file=paste(getBasePath("data"),"X6.csv",sep='') ,
          row.names=FALSE)
