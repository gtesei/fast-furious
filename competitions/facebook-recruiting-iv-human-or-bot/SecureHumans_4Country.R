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
                                           "X3.csv" , sep='')))


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

## Making feature: N. county a bidder used 
cat(">>> Making feature: N. country a bidder used ... \n")
bidder_country.train = ddply(train.full[,c('bidder_id','outcome','country')],
                              .(bidder_id,outcome) , function(x) c(num = length(unique(x$country))) )

cat(">>> on average robots uses ",mean(bidder_country.train[bidder_country.train$outcome==1,]$num),"different countries [sd=",
    sd(bidder_country.train[bidder_country.train$outcome==1,]$num),"] \n")
cat(">>> on average humans uses ",mean(bidder_country.train[bidder_country.train$outcome==0,]$num),"different countries [sd=",
    sd(bidder_country.train[bidder_country.train$outcome==0,]$num),"] \n")

bidder_country.train = ddply(train.full[,c('bidder_id','country')],
                            .(bidder_id) , function(x) c(country.num = length(unique(x$country))) )
bidder_country.test = ddply(test.full[,c('bidder_id','country')],
                            .(bidder_id) , function(x) c(country.num = length(unique(x$country))) )

bidder_country = rbind(bidder_country.train,bidder_country.test)
X = cbind(X,bidder_country[,'country.num' , drop=F])

rm(bidder_country)
rm(bidder_country.train)
rm(bidder_country.test)

### Making feature: type of country a bidder used 
cat(">>> Making feature: type of country a bidder used  ... \n")
country = unique(train.full$country)
cat(">> we have",length(country),"different countries on train set\n")

## train 
bidder_country.train = ddply(train.full[,c('bidder_id','country')],
                              .(bidder_id) , function(x)  {
                                ddply(x, .(country) , function(xx) {
                                  length(xx[,1])
                                } ) 
                              })
colnames(bidder_country.train)[3]='num'

## test 
bidder_country.test = ddply(test.full[,c('bidder_id','country')],
                             .(bidder_id) , function(x)  {
                               ddply(x, .(country) , function(xx) {
                                 length(xx[,1])
                               } ) 
                             })
colnames(bidder_country.test)[3]='num'

##
country.outcome = ddply(train.full , .(country) , function(x) c(zero=sum(x$outcome==0),one=sum(x$outcome==1)) )
country.outcome$ratio = country.outcome$one/(country.outcome$zero+country.outcome$one)
country.outcome = country.outcome[order(country.outcome$ratio , decreasing = T) , ]
cat(">>> country with only robots:",sum(country.outcome$ratio==1),"\n")
cat(">>> country with only humans:",sum(country.outcome$ratio==0),"\n")
describe(country.outcome$ratio)
country.outcome$ratio_lev = cut2(country.outcome$ratio , cuts=c(0,0.00001,0.0119,0.0279,0.0526,0.0944,0.1250,0.1872,0.2453,0.3824,0.6667,0.99,1) )
print(table(country.outcome$ratio_lev))
country.outcome$ratio_lev_int = unlist(lapply(country.outcome$ratio_lev, function(x) {
  i = 0
  for (i in 1:12) 
    if (x == levels(x)[[i]]) 
      break 
  return(i)
}))
print(head(country.outcome))
cat("...\n")
print(tail(country.outcome))

##
bidder_country.train.full = merge(x=bidder_country.train,y=country.outcome,by='country')
bidder_country.test.full = merge(x=bidder_country.test,y=country.outcome,by='country')

## Making feature 
bidder_country_types = as.data.frame(matrix(rep(0,nrow(X)*length(unique(country.outcome$ratio_lev_int))),
                                             nrow=nrow(X),
                                             ncol=length(unique(country.outcome$ratio_lev_int))))
colnames(bidder_country_types) = paste0("country_lev",1:ncol(bidder_country_types))

## Train
cat(">>> making features for train set ...\n")
for (bid in trind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_country.train.full[bidder_country.train.full$bidder_id == bidder_id , c('ratio_lev_int','num')] , 
             .(ratio_lev_int) , function(x) c(num=sum(x[2])))
  
  bidder_country_types[bid,aa$ratio_lev_int] = aa$num 
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(trind),"]..")
}

## Test train features 
bid = sample(trind,1)
cat("\n>>>Testing train features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_country.train.full[bidder_country.train.full$bidder_id == bidder_id , c('ratio_lev_int','num')])
print(bidder_country_types[bid,])
if (sum(bidder_country_types[bid,]) == 
      sum(bidder_country.train.full[bidder_country.train.full$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}

## Test
cat(">>> making features for train set ...\n")
for (bid in teind) {
  bidder_id = X[bid,]$bidder_id
  
  aa = ddply(bidder_country.test.full[bidder_country.test.full$bidder_id == bidder_id , c('ratio_lev_int','num')] , 
             .(ratio_lev_int) , function(x) c(num=sum(x[2])))
  
  bidder_country_types[bid,aa$ratio_lev_int] = aa$num 
  
  if (bid %% 100 == 0) cat("..[",bid,"/",max(teind),"]..")
}

## Test train features 
bid = sample(teind,1)
cat("\n>>>Testing test features ... bid:",bid,"\n")
bidder_id = X[bid,]$bidder_id
print(bidder_country.test.full[bidder_country.test.full$bidder_id == bidder_id , c('ratio_lev_int','num')])
print(bidder_country_types[bid,])
if (sum(bidder_country_types[bid,]) == 
      sum(bidder_country.test.full[bidder_country.test.full$bidder_id == bidder_id , c('num')]) ) {
  cat(">>> OK\n")
} else {
  stop("something wrong!")
}

### making percentages 
sums = as.numeric(apply(bidder_country_types,1,function(x) sum(x[1:ncol(bidder_country_types)]) ))
sums0s.idx = which(sums == 0)

for (i in 1:length(sums)) {
  if (sums[i]==0) {
    cat(">> i == ",i,"sums == 0, ...\n")
  } else {
    bidder_country_types[i,] = bidder_country_types[i,]/sums[i]
  }
} 

sums_after = as.numeric(apply(bidder_country_types[-sums0s.idx,],1,function(x) sum(x[1:ncol(bidder_country_types)]) ))
if( sum( sums_after > 1.001) > 0 || sum( sums_after < 0.999) > 0 )
  stop("something wrong")

### Binding features 
X=cbind(X,bidder_country_types)

cat(">>> storing on disk ...\n")
print(head(X))
cat("...\n")
print(tail(X))
write.csv(X,quote=FALSE, 
          file=paste(getBasePath("data"),"X4.csv",sep='') ,
          row.names=FALSE)
