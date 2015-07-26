library(readr)
library(tm)

library(NLP)

require(xgboost)
require(methods)

require(plyr)

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/crowdflower-search-relevance"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/crowdflower-search-relevance/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/crowdflower-search-relevance"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/ocrowdflower-search-relevance/"
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

ScoreQuadraticWeightedKappa = function (preds, dtrain) {
  
  obs <- getinfo(dtrain, "label")
  
  min.rating = 1
  max.rating = 4
  
  obs <- factor(obs, levels <- min.rating:max.rating)
  preds <- factor(preds, levels <- min.rating:max.rating)
  confusion.mat <- table(data.frame(obs, preds))
  confusion.mat <- confusion.mat/sum(confusion.mat)
  histogram.a <- table(obs)/length(table(obs))
  histogram.b <- table(preds)/length(table(preds))
  expected.mat <- histogram.a %*% t(histogram.b)
  expected.mat <- expected.mat/sum(expected.mat)
  labels <- as.numeric(as.vector(names(table(obs))))
  weights <- outer(labels, labels, FUN <- function(x, y) (x - y)^2)
  kappa <- 1 - sum(weights * confusion.mat)/sum(weights * expected.mat)
  
  return(list(metric = "qwk", value = kappa))
}

#### Data 
sampleSubmission <- read_csv(paste(getBasePath("data") , "sampleSubmission.csv" , sep=''))
train <- read_csv(paste(getBasePath("data") , "train.csv" , sep=''))
test  <- read_csv(paste(getBasePath("data") , "test.csv" , sep=''))
digest_df = read_csv(paste(getBasePath("data") , "base_matrix059.csv" , sep=''))

cat (">>> digest_df: ",dim(digest_df)," ... \n")

#### preparing xboost 
x = as.matrix(digest_df)
x = matrix(as.numeric(x),nrow(x),ncol(x))
rm(digest_df)

trind = 1:nrow(train)
teind = (nrow(train)+1):nrow(x)

y = train$median_relevance-1 

##### xgboost --> set necessary parameter
param <- list("objective" = "multi:softmax",
              "num_class" = 4,
              "eta" = 0.05,  
              "gamma" = 0.7,  
              "max_depth" = 25, 
              "subsample" = 0.5 , ## suggested in ESLII
              "nthread" = 10, 
              "min_child_weight" = 1 , 
              "colsample_bytree" = 0.5, 
              "max_delta_step" = 1)


##param['eval_metric'] = 'qwk'


cat(">>Params:\n")
print(param)

### Cross-validation 
cat(">>Cross Validation ... \n")
inCV = T
xval.perf = -1
bst.cv = NULL
early.stop = cv.nround = 3000 

cat(">> cv.nround: ",cv.nround,"\n") 

while (inCV) {
    cat(">>> maximizing ScoreQuadraticWeightedKappa ...\n")
    
    bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                    nfold = 5, nrounds=cv.nround , 
                    feval = ScoreQuadraticWeightedKappa , maximize = T)
    
    print(bst.cv)
    early.stop = which(bst.cv$test.qwk.mean == max(bst.cv$test.qwk.mean) )
    xval.perf = bst.cv[early.stop,]$test.qwk.mean
    cat(">> early.stop: ",early.stop," [xval.perf:",xval.perf,"]\n") 
  
  if (early.stop < cv.nround) {
    inCV = F
    cat(">> stopping [early.stop < cv.nround=",cv.nround,"] ... \n") 
  } else {
    cat(">> redo-cv [early.stop == cv.nround=",cv.nround,"] with 2 * cv.nround ... \n") 
    cv.nround = cv.nround * 2 
  }
  
  gc()
}

### Prediction 
bst = NULL

cat(">>> maximizing ScoreQuadraticWeightedKappa ...\n")
dtrain <- xgb.DMatrix(x[trind,], label = y)
watchlist <- list(train = dtrain)
bst = xgb.train(param = param, dtrain , 
                  nrounds = early.stop, watchlist = watchlist , 
                  feval = ScoreQuadraticWeightedKappa , maximize = T , verbose = 1)

cat(">> Making prediction ... \n")
pred = predict(bst,x[teind,])
pred = pred + 1 

print(">> prediction <<")
print(table(pred))

print(">> train set labels <<")
print(table(y+1))

fn = paste("sub_2gen___xval",xval.perf,".csv",sep='')
cat(">> writing prediction on disk [",fn,"]... \n")
write_csv(data.frame(id = sampleSubmission$id , prediction = pred) , paste(getBasePath("data") , fn , sep=''))

