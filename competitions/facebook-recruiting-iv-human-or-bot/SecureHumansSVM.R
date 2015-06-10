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
                               "Xfin5.csv" , sep='')))

## train/test index , labels  
train.full = merge(x = bids , y = train , by="bidder_id"  )
trind = 1:length(unique(train.full$bidder_id))
teind = (max(trind)+1):nrow(X)

X.full = merge(x=X , y=train , by="bidder_id")
y = X.full$outcome
y.cat = factor(y)
levels(y.cat) = c("human","robot")

rm(train.full)
rm(X.full)
rm(bids)

############

X.base = X[,-grep("bidder_id" , colnames(X) )]

x = as.matrix(X.base)
x = matrix(as.numeric(x),nrow(x),ncol(x))
sigmaRangeReduced <- sigest(x[trind,])
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1], .C = 2^(seq(-4, 4)))

rm(x)

cat(">>> training ... \n")
svmRFitCost <- train(x = X.base[trind,] , y = y.cat , 
                     method = "svmRadial",
                     metric = "ROC",
                     #class.weights = c(human = 7, robot = 3),
                     tuneGrid = svmRGridReduced,
                     trControl = trainControl(method = "repeatedcv", number = 5 , 
                                              summaryFunction = twoClassSummary , classProbs = TRUE)
                     )

perf.roc = max(svmRFitCost$results$ROC)
cat(">>> ROC:",perf.roc,"\n")

cat(">> predicting ... \n")
pred = predict(svmRFitCost , X.base[teind,])

rm(X.base)
#####
sub = data.frame(bidder_id = X[teind,]$bidder_id , pred.xgb = pred)
sub.full.base = merge(x=sub,y=sampleSubmission,by="bidder_id" , all=T)
sub.full.base$prediction = ifelse( ! is.na(sub.full.base$pred.xgb) , sub.full.base$pred.xgb , 0 )
sub.full.base = sub.full.base[,-2]

fn = paste("sub_svm_xval" , perf.roc , ".csv" , sep='') 
cat(">> writing prediction on disk [",fn,"]... \n")
write.csv(sub.full.base,quote=FALSE, 
          file=paste(getBasePath("data"),fn,sep='') ,
          row.names=FALSE)


