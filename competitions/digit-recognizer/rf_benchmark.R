# makes the random forest submission

library(randomForest)

getBasePath = function (type = "data" , ds = "" , gen="") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/digit-recognizer"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/digit-recognizer/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/data_pre_process"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/digit-recognizer/"
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

train <- read.csv(paste(getBasePath(),"train.csv",sep=""), header=TRUE)
test <- read.csv(paste(getBasePath(),"test.csv",sep=""), header=TRUE)

labels <- as.factor(train[,1])
train <- train[,-1]

train = train[1:100,]
labels = labels[1:100]

rf <- randomForest(train, labels, xtest=test, ntree=1000)
predictions <- levels(labels)[rf$test$predicted]

write(predictions, file=paste(getBasePath(),"rf_benchmark_fresh.csv",sep=""), ncolumns=1) 

rf <- randomForest(train, labels, xtest=train, ntree=1000)
pred.train = rf$test$predicted
acc.train = sum(pred.train == labels) / length(labels) 


