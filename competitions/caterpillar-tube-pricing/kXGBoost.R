# This is implementation of XGboost model in R


# library required

library(data.table)
library(xgboost)
library(Matrix)
library(methods)

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



# Importing data into R
train  <- as.data.frame( fread(paste(getBasePath("data") , "train_set.csv" , sep='')))
test  <- as.data.frame( fread(paste(getBasePath("data") , "test_set.csv" , sep='')))
bom  <- as.data.frame( fread(paste(getBasePath("data") , "bill_of_materials.csv" , sep='')))
specs  <- as.data.frame( fread(paste(getBasePath("data") , "specs.csv" , sep='')))
tube  <- as.data.frame( fread(paste(getBasePath("data") , "tube.csv" , sep='')))

# you must know why I am using set.seed()
set.seed(546)

# Merging the data
train$id  <- -(1:nrow(train))
test$cost  <- 0

data  <- rbind(train,test)

data  <- merge(data,tube,by="tube_assembly_id",all = T)
data  <- merge(data,bom,by="tube_assembly_id",all = T)
data  <- merge(data,specs,by="tube_assembly_id",all = T)

# extracting year and month for quote_date

data$quote_date  <- strptime(data$quote_date,format = "%Y-%m-%d", tz="GMT")
data$year <- year(as.IDate(data$quote_date))
data$month <- month(as.IDate(data$quote_date))
data$week <- week(as.IDate(data$quote_date))

# dropping variables
data$quote_date  <- NULL
data$tube_assembly_id  <- NULL


# converting NA in to '0' and '" "' for mode Matrix Generation

for(i in 1:ncol(data)){
  if(is.numeric(data[,i])){
    data[is.na(data[,i]),i] = 0
  }else{
    data[,i] = as.character(data[,i])
    data[is.na(data[,i]),i] = " "
    data[,i] = as.factor(data[,i])
  }
}


# converting data.frame to sparse matrix for modelling

train  <- data[which(data$id < 0), ]
test  <- data[which(data$id > 0), ]

ids  <- test$id
cost  <- train$cost

#dropping some more variables

train$id  <- NULL 
test$id  <- NULL
#train$cost  <- 0
test$cost  <- NULL


# this is a very crude way of generating sparse matrix and might take a bit time
# if anybody has a better way feel free to comment;)

tr.mf  <- model.frame(as.formula(paste("cost ~",paste(names(train),collapse = "+"))),train)
tr.m  <- model.matrix(attr(tr.mf,"terms"),data = train)
tr  <- Matrix(tr.m)
t(tr)


te.mf  <- model.frame(as.formula(paste("~",paste(names(test),collapse = "+"))),test)
te.m  <- model.matrix(attr(te.mf,"terms"),data = test)
te  <- Matrix(te.m)
t(te)


# generating xgboost model 

# tr.x  <- xgb.DMatrix(tr,lable=log(names(train)+1))
cost.log  <- log(cost+1) # treating cost as log transfromation is working good on this data set

tr.x  <- xgb.DMatrix(tr,label = cost.log)
te.x  <- xgb.DMatrix(te)


# parameter selection
par  <-  list(booster = "gblinear",
              objective = "reg:linear",
              min_child_weight = 6,
              gamma = 2,
              subsample = 0.85,
              colsample_bytree = 0.75,
              max_depth = 10,
              verbose = 1,
              scale_pos_weight = 1)


#selecting number of Rounds
n_rounds= 200


#modeling

x.mod.t  <- xgb.train(params = par, data = tr.x , nrounds = n_rounds)
pred  <- predict(x.mod.t,te.x)
head(pred)

for(i in 1:50){
  x.mod.t  <- xgb.train(par,tr.x,n_rounds)
  pred  <- cbind(pred,predict(x.mod.t,te.x))
}

pred.sub  <- exp(rowMeans(pred))-1


# generating data frame for submission
sub.file = data.frame(id = ids, cost = pred.sub)
sub.file = aggregate(data.frame(cost = sub.file$cost), by = list(id = sub.file$id), mean)

write.csv(sub.file,
          quote=FALSE, 
          file=paste(getBasePath("submission"),'submit_gen_xgb.csv',sep='') ,
          row.names=FALSE)