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
                                           "X6.csv" , sep='')))

#################
## index 
##  1     - bidder_id 
##  2     - auct.num (to scale) 
##  3:14  - auct_lev 
##  15:22 - merch_lev 
##  23    - dev.num (to scale)
##  24:35 - dev_lev 
##  36    - country.num (to scale)
##  37:48 - country_lev 
##  49    - ip.num (to scale)
##  50:55 - ip_lev 
##  56    - url.num (to scale)
##  57:62 - url_lev 
################

### Test
cat(">>> just performing a little testing ... \n")
i = sample(nrow(X),1)
cat(">>> bidder_id:",X[i,1],"\n")
if (sum(X[i,3:14])==1 & sum(X[i,15:22])==1 & sum(X[i,24:35])==1 & sum(X[i,39:48])==1 & sum(X[i,50:55]) & sum(X[i,57:62])==1) {
  cat(">>> OK! \n")
} else {
  stop("something wrong. Maybe one of ~ 1350/6600 unlukely cases ...")
}

### Scaling 2,23,36,49,56
cols = c(2,23,36,49,56)
cat(">>> performing feature scaling of features ",colnames(X)[cols],"...\n")
trans = preProcess(x = X[,cols] , method = c("center","scale")  )
X[,cols] = predict(trans,X[,cols])

### Test
cat(">>> just performing a little testing ... \n")
i = sample(nrow(X),1)
cat(">>> bidder_id:",X[i,1],"\n")
if (sum(X[i,3:14])==1 & sum(X[i,15:22])==1 & sum(X[i,24:35])==1 & sum(X[i,39:48])==1 & sum(X[i,50:55]) & sum(X[i,57:62])==1) {
  cat(">>> OK! \n")
} else {
  stop("something wrong. Maybe one of ~ 1350/6600 unlukely cases ...")
}

#### storing on disk 
cat(">>> storing on disk ... \n")

write.csv(X,quote=FALSE, 
          file=paste(getBasePath("data"),"Xfin.csv",sep='') ,
          row.names=FALSE)
