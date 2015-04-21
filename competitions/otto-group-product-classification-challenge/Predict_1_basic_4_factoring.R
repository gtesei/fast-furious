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
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/otto-group-product-classification-challenge"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/otto-group-product-classification-challenge/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/otto-group-product-classification-challenge"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/otto-group-product-classification-challenge/"
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

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#######
verbose = T
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))
source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))
source(paste0( getBasePath("process") , "/Resample_Lib.R"))
source(paste0( getBasePath("process") , "/Classification_Lib.R"))

ClassModels = c(
  "Mode",
  "LogisticReg" , 
  "LDA" , 
  "PLSDA" , 
  "PMClass" , 
  "NSC" , 
  "NNetClass" , 
  "SVMClass" , 
  "KNNClass" , 
  "ClassTrees" , 
  "BoostedTreesClass" , 
  "BaggingTreesClass" 
) 

sub = NULL
grid = NULL
controlObject <- trainControl(method = "boot", number = 30 , 
                              summaryFunction = twoClassSummary , classProbs = TRUE)

#######
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

train.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                       "encoded_train_reduced.csv" , sep=''))) 

test.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                      "encoded_test_reduced.csv" , sep='')))

y = as.data.frame( fread(paste(getBasePath("data") , 
                                      "encoded_y.csv" , sep='')))

y = as.numeric(y$y)
y.class = unique(y)

########
cat("\n*********\n")
for (j in y.class) {
  cat("y == ",j,":" , sum(y == j)," -  ",sum(y == j)/length(y) , " \n")
} 
cat("*********\n")
########

j = 6 
cat(">>> building train set for Class <<",as.character(j),">> ... ")

train.1 = train.raw[(y == j), ]
train.0 = train.raw[sample(x = which(y != j) , size = dim(train.1)[1] ) , ]
train.j = rbind(train.0,train.1)

y.j = c( rep(0,dim(train.1)[1]) , rep(1,dim(train.1)[1]) ) 
y.cat = factor(y.j) 
levels(y.cat) = c("other_classes","this_class")

### shuffle 
idx = sample(x = 1:length(y.j) , size = length(y.j))
train.j = train.j[idx,]
y.cat = y.cat[idx]

cat(">>> finding best model for Class <<",as.character(j),">> ... ")

train.j = train.j[1:400,]
y.cat = y.cat[1:400]
### k-fold 
l = trainAndPredict.kfold.class (k=4,train.j,
                                 y.cat,
                                 fact.sign = 'this_class', 
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










