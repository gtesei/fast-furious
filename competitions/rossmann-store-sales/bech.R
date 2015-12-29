library(readr)
library(xgboost)
library(fastfurious)

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/rossmann-store-sales')
ff.bindPath(type = 'code' , sub_path = 'competitions/rossmann-store-sales')
ff.bindPath(type = 'elab' , sub_path = 'dataset/rossmann-store-sales/elab') 

ff.bindPath(type = 'ensemble_1' , sub_path = 'dataset/rossmann-store-sales/ensembles/ensemble_1',createDir = T) ## out 
ff.bindPath(type = 'best_tune_1' , sub_path = 'dataset/rossmann-store-sales/ensembles/best_tune_1',createDir = T) ## out 
ff.bindPath(type = 'submission_1' , sub_path = 'dataset/rossmann-store-sales/ensembles/pred_ensemble_1',createDir = T) ## out 

## DATA 
cat("reading the train and test data\n")
train <- read_csv(paste(ff.getPath("data") , "train.csv" , sep=''))
test  <- read_csv(paste(ff.getPath("data") , "test.csv" , sep=''))
store <- read_csv(paste(ff.getPath("data") , "store.csv" , sep=''))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train <- merge(train,store)
test <- merge(test,store)

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0

cat("train data column names and details\n")
names(train)
str(train)
summary(train)
cat("test data column names and details\n")
names(test)
str(test)
summary(test)

# looking at only stores that were open in the train set
# may change this later
train <- train[ which(train$Open=='1'),]
train <- train[ which(train$Sales!='0'),]

# seperating out the elements of the date column for the train set
train$month <- as.integer(format(train$Date, "%m"))
train$year <- as.integer(format(train$Date, "%y"))
train$day <- as.integer(format(train$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train <- train[,-c(3,8)]

# seperating out the elements of the date column for the test set
test$month <- as.integer(format(test$Date, "%m"))
test$year <- as.integer(format(test$Date, "%y"))
test$day <- as.integer(format(test$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
test <- test[,-c(4,7)]

feature.names <- names(train)[c(1,2,5:19)]
cat("Feature Names\n")
feature.names


cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
tra<-train[,feature.names]
RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}
nrow(train)
h<-sample(nrow(train),10000)

### exporting to data 
train_export <- tra 
train_export$Sales <- train$Sales
test_export <- test[,feature.names]
test_export$Id <- test$Id

write_csv(train_export, paste(ff.getPath("elab") , "Xtrain_bench.csv" , sep=''))
write_csv(test_export, paste(ff.getPath("elab") , "Xtest_bench.csv" , sep=''))
### end exporting to data 

##
dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=log(train$Sales+1)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log(train$Sales+1)[-h])
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster = "gbtree",
                eta                 = 0.005, # 0.06, #0.01,
                max_depth           = 10, #changed from default of 8
                subsample           = 0.9, # 0.7
                colsample_bytree    = 0.7 # 0.7
                #num_parallel_tree   = 2
                # alpha = 0.0001, 
                # lambda = 1
)
cat("****** param ***** \n")
print(param)
#### 
clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 14000, #300, #280, #125, #250, # changed from 300
                    verbose             = 0,
                    early.stop.round    = 100,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)
pred1 <- exp(predict(clf, data.matrix(test[,feature.names]))) -1
submission <- data.frame(Id=test$Id, Sales=pred1)
cat("saving the submission file\n")
write_csv(submission, paste(ff.getPath("elab") , "bech_eta0005_nround14000.csv" , sep=''))












