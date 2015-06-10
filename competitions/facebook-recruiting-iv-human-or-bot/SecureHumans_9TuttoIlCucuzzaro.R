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

#######
verbose = T
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))
source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))
source(paste0( getBasePath("process") , "/Resample_Lib.R"))
source(paste0( getBasePath("process") , "/Classification_Lib.R"))

ClassModels = c(
  "LogisticReg" , 
  "LDA" , 
  "PLSDA" , 
  "PMClass" , 
  "NSC" , 
#  "NNetClass" , 
  "SVMClass" , 
  "KNNClass" , 
  "ClassTrees" , 
  "BoostedTreesClass" , 
  "BaggingTreesClass" 
) 

controlObject <- trainControl(method = "repeatedcv", number = 5 , summaryFunction = twoClassSummary , classProbs = TRUE)


## train/test index , labels  
train.full = merge(x = bids , y = train , by="bidder_id"  )
trind = 1:length(unique(train.full$bidder_id))
teind = (max(trind)+1):nrow(X)
X.full = merge(x=X , y=train , by="bidder_id")
y = X.full$outcome

rm(train.full)
rm(X.full)
rm(bids)

############

X = X[,-grep("bidder_id" , colnames(X) )]

y.cat = factor(y)
levels(y.cat) = c("human","robot")

l = trainAndPredict.kfold.class (k=3,X[trind,],
                                 y.cat,
                                 fact.sign = 'robot', 
                                 ClassModels,
                                 controlObject, 
                                 verbose = T , 
                                 doPlot = T)
model.winner = l[[1]]
.grid = l[[2]]
perf.kfold = l[[3]]

cat("model winner: ",model.winner,"\n")
cat("\n*****\n")
print(perf.kfold)
cat("\n*****\n")
print(.grid)













