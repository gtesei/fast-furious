library(binhf)
library(fBasics)
library(lattice)
require(xgboost)
require(methods)
library(data.table)
library(plyr)

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/caterpillar-tube-pricing/competition_data"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/caterpillar-tube-pricing/competition_data/"
  } else if(type == "submission") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/caterpillar-tube-pricing"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/caterpillar-tube-pricing/"
  } else if(type == "elab") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/caterpillar-tube-pricing/elab"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/caterpillar-tube-pricing/elab/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/caterpillar-tube-pricing"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/caterpillar-tube-pricing/"
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

################# FAST-FURIOUS SOURCES
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))


################# DATA IN 
sample_submission = as.data.frame( fread(paste(getBasePath("data") , 
                                               "sample_submission.csv" , sep=''))) 

train_set = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train_set.csv" , sep=''))) 

test_set = as.data.frame( fread(paste(getBasePath("data") , 
                                      "test_set.csv" , sep=''))) 


################# DATA OUT 
train_enc = NULL 
test_enc = NULL

train_enc_date = NULL  # with quote_date 
test_enc_date = NULL   # with quote_date  

################# PROCESSING 

train_enc = train_set 
test_enc = test_set

train_enc_date = train_set  # with quote_date 
test_enc_date = test_set   # with quote_date  

## quote_date
all_date = as.Date(c(train_set$quote_date , test_set$quote_date))
all_date = sort(all_date)
head(all_date)
length(all_date)
#[1] 60448

all_date_uniq = as.Date(unique(c(train_set$quote_date , test_set$quote_date)))
all_date_uniq = sort(all_date_uniq)
head(all_date_uniq)
length(all_date_uniq)
#[1] 2413

# count 
hd = rep(0,length(all_date_uniq))
for (i in 1:length(all_date_uniq)) 
  hd[i] = sum(all_date == all_date_uniq[i])

plot(all_date_uniq,hd , type = 'h' , cex=1, col='blue' , 
     lwd = 2, 
     xlab="Quote Date" , ylab="Frequency" )

drange = all_date_uniq[length(all_date_uniq)] - all_date_uniq[1]
print(drange)

#  in prima approssimazioni rimuoviam quote_date
#  TODO a prendere la differenza tra [quote_day - min(quote)] e vedere come perfroma il modello 

train_enc[,'quote_date'] = NULL
test_enc[,'quote_date'] = NULL

train_enc_date$quote_date = as.numeric(as.Date(train_enc_date$quote_date) - rep(all_date_uniq[1] , nrow(train_enc_date))) 
test_enc_date$quote_date = as.numeric(as.Date(test_enc_date$quote_date) - rep(all_date_uniq[1] , nrow(test_enc_date))) 

## supplier
l = encodeCategoricalFeature (train_set$supplier , test_set$supplier , colname.prefix = "supplier" , asNumeric=F)
train_enc = cbind(train_enc , l$traindata)
test_enc = cbind(test_enc , l$testdata)
train_enc[,'supplier'] = NULL
test_enc[,'supplier'] = NULL

train_enc_date = cbind(train_enc_date , l$traindata)
test_enc_date = cbind(test_enc_date , l$testdata)
train_enc_date[,'supplier'] = NULL
test_enc_date[,'supplier'] = NULL

## bracket_pricing
train_enc$bracket_pricing = ifelse(train_enc$bracket_pricing == 'Yes' , 1 , 0)
test_enc$bracket_pricing = ifelse(test_enc$bracket_pricing == 'Yes' , 1 , 0)

train_enc_date$bracket_pricing = ifelse(train_enc_date$bracket_pricing == 'Yes' , 1 , 0)
test_enc_date$bracket_pricing = ifelse(test_enc_date$bracket_pricing == 'Yes' , 1 , 0)

################# SAVE DATA OUT on disk 
cat(">> saving on disk ...\n")

# train_enc
write.csv(train_enc,
          quote=FALSE, 
          file=paste(getBasePath("elab"),'train_enc.csv',sep='') ,
          row.names=FALSE)

# test_enc
write.csv(test_enc,
          quote=FALSE, 
          file=paste(getBasePath("elab"),'test_enc.csv',sep='') ,
          row.names=FALSE)

# train_enc_date
write.csv(train_enc_date,
          quote=FALSE, 
          file=paste(getBasePath("elab"),'train_enc_date.csv',sep='') ,
          row.names=FALSE)

# test_enc_date
write.csv(test_enc_date,
          quote=FALSE, 
          file=paste(getBasePath("elab"),'test_enc_date.csv',sep='') ,
          row.names=FALSE)





