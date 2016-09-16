library(Hmisc)
library(data.table)
library(FeatureHashing)
library(xgboost)
library(plyr)
library(Matrix)

cat(Sys.time())
cat("Reading data\n")
###
prefix = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/predicting-red-hat-business-value/'
train <- fread(paste0(prefix,"act_train.csv"), header=TRUE)
test <- fread(paste0(prefix,"act_test.csv"), header=TRUE)
people <- fread(paste0(prefix,"people.csv"), header=TRUE)
sample_submission <- fread(paste0(prefix,"sample_submission.csv"), header=TRUE)

str(sample_submission)

explore_basic <- function(ds,name) {
  cat("--------------------- ",name," \n")
  str(ds)
  cat("NA:",sum(is.na(ds)))  
}

explore_basic(people,"people")
explore_basic(train,"train")
explore_basic(test,"test")

cat("JOIN train:",sum(!train$people_id %in% people$people_id),"\n")
cat("JOIN train:",sum(!test$people_id %in% people$people_id),"\n")

cat("train-test-people_id:",sum(train$people_id %in% test$people_id),"\n")

cat("train-test-activity_id:",sum(train$activity_id %in% test$activity_id),"\n") #0 
length(unique(train$activity_id)) # 2,197,291 = nrow(train) 
length(unique(test$activity_id)) # 498,687 = nrow(test) 


cat("train-test-activity_category:",sum(train$activity_category %in% test$activity_category),"\n") # 2,197,291 

length(unique(train$activity_category)) # 7 
length(unique(test$activity_category)) # 7 


cat("train-person-obs-avg:",mean(ddply(train, .(people_id) , function(x)c(avg=nrow(x)))$avg),"\n") ##14.52322
cat("test-person-obs-avg:",mean(ddply(test, .(people_id) , function(x)c(avg=nrow(x)))$avg),"\n") ##13.18476 

cat("people-person-obs-avg:",mean(ddply(people, .(people_id) , function(x)c(avg=nrow(x)))$avg),"\n") ##1


cat("train-person-otcome-avg:",mean(ddply(train, .(people_id) , function(x)c(avg=mean(x$outcome)))$avg),"\n") ##0.4355076 
cat("train-person-otcome-std:",mean(ddply(train, .(people_id) , function(x)c(std=sd(x$outcome)))$std,na.rm = T),"\n") ##0.02187992 

train <- train[order(train$people_id,decreasing = T),]
train[100:200,c(1,15), with = FALSE]

## is date important?    
cat("train-person-date-avg:",mean(ddply(train, .(people_id) , function(x)c(avg=mean(as.Date(x$date))))$avg),"\n") ##19406.57 
cat("train-person-date-std:",mean(ddply(train, .(people_id) , function(x)c(std=sd(as.Date(x$date))))$std,na.rm = T),"\n") ##33.58658 






