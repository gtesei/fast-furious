library(data.table)
library(xgboost)
library(fastfurious)

set.seed(1)
### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/springleaf-marketing-respons')
ff.bindPath(type = 'sub' , sub_path = 'dataset/springleaf-marketing-respons/sub',createDir = T)

cat("reading the train and test data\n")

train = coupon_list_train = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
test = coupon_list_train = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))

#### id: ncol == 1 ---> scarta diretto 
#### 0-variance preds in train = 208,214,839,846,1427 -->> tutti NA o 0-variance preds --> scarta diretto
# train2 = na.omit(train)
# for (i in 1:ncol(train2)) {
#   if (length(unique(train2[,i]))==1) {
#     cat (">>>> ",i,"\n")
#   }
# }


train = train[,-c(1,208,214,839,846,1427)]
test_id = test$ID
test = test[,-c(1,208,214,839,846,1427)]
####

feature.names <- names(train)[2:ncol(train)-1]

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    cat(">>> ",f," is character \n")
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("replacing missing values with -1\n")
train[is.na(train)] <- -1
test[is.na(test)]   <- -1

#cat("sampling train to get around 8GB memory limitations\n")
#train <- train[sample(nrow(train), 40000),]
gc()

cat("training a XGBoost classifier\n")
clf <- xgboost(data        = data.matrix(train[,feature.names]),
               label       = train$target,
               #nrounds     = 20,
               #nrounds     = 40,
               #nrounds     = 80,
               nrounds     = 60,
               objective   = "binary:logistic",
               eval_metric = "auc")

cat("making predictions in batches due to 8GB memory limitation\n")
submission <- data.frame(ID=test_id)
submission$target <- NA
for (rows in split(1:nrow(test), ceiling((1:nrow(test))/10000))) {
  submission[rows, "target"] <- predict(clf, data.matrix(test[rows,feature.names]))
}

cat("saving the submission file\n")
write.csv(submission,
          quote=FALSE, 
          file=paste0(ff.getPath("sub"),"xgboost_submission.csv") ,
          row.names=FALSE)
