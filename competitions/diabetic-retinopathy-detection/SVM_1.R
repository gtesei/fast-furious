
library(caret)
library(Hmisc)
library(verification)
library(pROC)
library(kernlab)
library(subselect)
library(plyr)
library(data.table)


ScoreQuadraticWeightedKappa = function (preds, obs) {
  
  min.rating = 0
  max.rating = 4
  
  #obs <- factor(obs, levels <- 0:4)
  #preds <- factor(preds, levels <- 0:4)
  
  obs = mapvalues(obs, from = c("No_DR","Mild","Moderate","Severe","Proliferative_DR"), to = min.rating:max.rating)
  preds = mapvalues(preds, from = c("No_DR","Mild","Moderate","Severe","Proliferative_DR"), to = min.rating:max.rating)
  
  confusion.mat <- table(data.frame(obs, preds))
  confusion.mat <- confusion.mat/sum(confusion.mat)

  histogram.a <- table(obs)/length(table(obs))
  histogram.b <- table(preds)/length(table(preds))
  
  expected.mat <- histogram.a %*% t(histogram.b)
  
  expected.mat <- expected.mat/sum(expected.mat)
  
  labels <- as.numeric(as.vector(names(table(obs))))
  weights <- outer(labels, labels, FUN <- function(x, y) (x - y)^2)
  kappa <- 1 - sum(weights * confusion.mat)/sum(weights * expected.mat)
  #print(kappa)
  kappa
}

costSummary <- function (data, lev = NULL, model = NULL) {
  if (is.character(data$obs)) data$obs <- factor(data$obs,
                                                 levels = lev)
  c(postResample(data[, "pred"], data[, "obs"]),
    ScoreQuadraticWeightedKappa = ScoreQuadraticWeightedKappa(data[, "pred"], data[, "obs"]))
}


getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/diabetic-retinopathy-detection"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/diabetic-retinopathy-detection/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/diabetic-retinopathy-detection"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/diabetic-retinopathy-detection/"
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

#########

train_labels = as.data.frame( fread(paste(getBasePath("data") , 
                                         "trainLabels.csv" , sep=''))) 
                                            

train_data = as.data.frame( fread(paste(getBasePath("data") , 
                                               'feat_gen_2000.csv' , sep=''))) 

#########

y.cat = factor(train_data$level) 
levels(y.cat) = c("No_DR","Mild","Moderate","Severe","Proliferative_DR")

ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                     summaryFunction = costSummary)

svmRFitCost <- train( x = train_data[,-c(1,2)], y = y.cat , 
                     method = "svmRadial",
                     metric = "ScoreQuadraticWeightedKappa",
                     maximize = T,
                     preProc = c("center", "scale"),
                     class.weights = c(No_DR = 1, Mild = 1, Moderate = 1, Severe = 1 , Proliferative_DR = 1),
                     tuneLength = 15,
                     trControl = ctrl)




