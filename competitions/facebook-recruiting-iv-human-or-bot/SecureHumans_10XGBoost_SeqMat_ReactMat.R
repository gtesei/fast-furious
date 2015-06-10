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

source(paste0( getBasePath("process") , "/Classification_Lib.R"))
#################
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep=''))) ## outcome = 0 human 

train = as.data.frame( fread(paste(getBasePath("data") , 
                                   "train.csv" , sep=''))) 

bids = as.data.frame( fread(paste(getBasePath("data") , 
                                  "bids.csv" , sep='')))

seq_matrix = as.data.frame( fread(paste(getBasePath("data") , 
                                        "seq_matrix.csv" , sep='')))

react_matrix = as.data.frame( fread(paste(getBasePath("data") , 
                                        "react_matrix.csv" , sep='')))

#### best performant feature set 
X = as.data.frame( fread(paste(getBasePath("data") , 
                               "Xfin5.csv" , sep='')))

train.full = merge(x = bids , y = train , by="bidder_id"  )
trind = 1:length(unique(train.full$bidder_id))
teind = (max(trind)+1):nrow(X)

X.full = merge(x=X , y=train , by="bidder_id")
y = X.full$outcome
y.cat = factor(y)
levels(y.cat) = c("human","robot")


################# Model 
## elimino bidder_id 
X.base = X[,-grep("bidder_id" , colnames(X) )]
cat(">>> dim X.base [no bidder_id]:",dim(X.base),"\n")

## inserire nuove feature 

##### seq
X.base$seq_mean = seq_matrix$seq_mean ##0.91 
#X.base$seq_max = seq_matrix$seq_max
#X.base$seq_mode = seq_matrix$seq_mode
#X.base$seq_min = seq_matrix$seq_min ##0.906

### react 
X.base$react_vs_all = react_matrix$react_vs_all
#X.base$react_vs_me = react_matrix$react_vs_me
#X.base$react_vs_other = react_matrix$react_vs_other
cat(">>> mean react_vs_all=",mean(X.base$react_vs_all),"..\n")

cat(">>> dim X.base [seq_mean]:",dim(X.base),"\n")

######### XGboost 
x = as.matrix(X.base)
x = matrix(as.numeric(x),nrow(x),ncol(x))

##### xgboost --> set necessary parameter
param <- list("objective" = "binary:logistic",
              "eval_metric" = "auc",
              "eta" = 0.01,  ## suggested in ESLII
              #"gamma" = 0.7,  
              "gamma" = 3,  
              "max_depth" = 6, 
              "subsample" = 0.5 , ## suggested in ESLII
              "nthread" = 10, 
              #"min_child_weight" = 1 , 
              "min_child_weight" = 0 , 
              "colsample_bytree" = 0.5, 
              "max_delta_step" = 0
              #"max_delta_step" = 1
              )

cv.nround = 3000

### echo 
cat(">>Params:\n")
print(param)
cat(">> cv.nround: ",cv.nround,"\n") 

### Cross-validation 
cat(">>Cross Validation ... \n")
inCV = T
xval.perf = -1
bst.cv = NULL
early.stop = -1

while (inCV) {
  cat(">>> maximizing auc ...\n")
  bst.cv = xgb.cv(param=param, data = x[trind,], label = y, nfold = 5, nrounds=cv.nround )    
  print(bst.cv)
  early.stop = min(which(bst.cv$test.auc.mean == max(bst.cv$test.auc.mean) ))
  xval.perf = bst.cv[early.stop,]$test.auc.mean
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
bst = xgboost(param = param, data = x[trind,], label = y, nrounds = early.stop) 

cat(">> Making prediction ... \n")
pred = predict(bst,x[teind,])
pred.train = predict(bst,x[trind,])

print(">> prediction <<")
print(mean(pred))

print(">> train set labels <<")
print(mean(y))

#### assembling submission - no probs recalibration 
sub = data.frame(bidder_id = X[teind,]$bidder_id , pred.xgb = pred)
sub.full.base = merge(x=sub,y=sampleSubmission,by="bidder_id" , all=T)
sub.full.base$prediction = ifelse( ! is.na(sub.full.base$pred.xgb) , sub.full.base$pred.xgb , 0 )
sub.full.base = sub.full.base[,-2]

## writing on disk 
#fn = paste("sub_xgboost_seqMat_ALL_xval" , xval.perf , ".csv" , sep='') 
#fn = paste("sub_xgboost_reactMat_vsOther_xval" , xval.perf , ".csv" , sep='') 
fn = paste("sub_xgboost_reactVsAll_seqMean_xval" , xval.perf , ".csv" , sep='') 
cat(">> writing prediction on disk [",fn,"]... \n")
write.csv(sub.full.base,quote=FALSE, 
          file=paste(getBasePath("data"),fn,sep='') ,
          row.names=FALSE)


