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

################# Config 
base.submission = F 
stress.model = T
recalib.bayes = F
recalib.sigmoid = T 
recalib.killer = F 


################# Model 
## elimino bidder_id 
X.base = X[,-grep("bidder_id" , colnames(X) )]

cat(">>> dim X.base [no bidder_id]:",dim(X.base),"\n")

######### XGboost 
x = as.matrix(X.base)
x = matrix(as.numeric(x),nrow(x),ncol(x))

##### xgboost --> set necessary parameter
param <- list("objective" = "binary:logistic",
              "eval_metric" = "auc",
              "eta" = 0.01,  ## suggested in ESLII
              "gamma" = 0.7,  
              "max_depth" = 6, 
              "subsample" = 0.5 , ## suggested in ESLII
              "nthread" = 10, 
              "min_child_weight" = 1 , 
              "colsample_bytree" = 0.5, 
              "max_delta_step" = 1)

cv.nround = 2500
if (stress.model) {
  cat(">> stressing model ... \n") 
#   param['eta'] = 0.025
#   param['gamma'] = 5
  param['eta'] = 0.05
  param['gamma'] = 1
  param['max_delta_step'] = 0
  param['subsample'] = 0.5
  param['min_child_weight'] = 1
  param['colsample_bytree'] = 0.5
} 
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

sub.full.base = NULL
if (base.submission | stress.model) {
#### assembling submission - no probs recalibration 
sub = data.frame(bidder_id = X[teind,]$bidder_id , pred.xgb = pred)
sub.full.base = merge(x=sub,y=sampleSubmission,by="bidder_id" , all=T)
sub.full.base$prediction = ifelse( ! is.na(sub.full.base$pred.xgb) , sub.full.base$pred.xgb , 0 )
sub.full.base = sub.full.base[,-2]

## writing on disk 
fn = paste("sub_xgboost_stress_xval" , xval.perf , ".csv" , sep='') 
cat(">> writing prediction on disk [",fn,"]... \n")
write.csv(sub.full.base,quote=FALSE, 
          file=paste(getBasePath("data"),fn,sep='') ,
          row.names=FALSE)
}

################# Recalibrating probs 
if (recalib.bayes) {
  cat("recalibrating probabilities with Bayes ... \n") 
  print(">> prediction (test) **** BEFORE ****<<")
  print(mean(pred))
  print(">> prediction (train) <<")
  print(mean(pred.train))
  print(">> train set labels <<")
  print(mean(y))
  cat("applying NaiveBayes ... \n")
  
  train.df = data.frame(class = y.cat , prob = as.numeric( pred.train ))
  
  require (klaR)
  BayesCal <- NaiveBayes( class ~ prob  , data = train.df, usekernel = TRUE)
  BayesProbs <- predict(BayesCal, newdata = data.frame(prob = as.numeric( pred )))
  BayesProbs.robot <- BayesProbs$posterior[, "robot"]
  
  print(">> prediction (test) **** AFTER ****<<")
  print(mean(BayesProbs.robot))
  
  #### assembling submission - no probs recalibration 
  sub = data.frame(bidder_id = X[teind,]$bidder_id , pred.xgb = BayesProbs.robot)
  sub.full = merge(x=sub,y=sampleSubmission,by="bidder_id" , all=T)
  sub.full$prediction = ifelse( ! is.na(sub.full$pred.xgb) , sub.full$pred.xgb , 0 )
  sub.full = sub.full[,-2]
  
  ## writing on disk 
  fn = paste("sub_xgboost_stress_rb_xval" , xval.perf , ".csv" , sep='') 
  cat(">> writing prediction on disk [",fn,"]... \n")
  write.csv(sub.full,quote=FALSE, 
            file=paste(getBasePath("data"),fn,sep='') ,
            row.names=FALSE)
  
} 


sigmoidProbs = NULL
sub.full = NULL 

sigmoidProbs.2 = NULL 
sub.full.2 = NULL 

if (recalib.sigmoid) {
  cat("recalibrating probabilities with sigmoid ... \n") 
  print(">> prediction (test) **** BEFORE ****<<")
  print(mean(pred))
  print(">> prediction (train) <<")
  print(mean(pred.train))
  print(">> train set labels <<")
  print(mean(y))
  cat("applying sigmoid ... \n")
  
  train.df = data.frame(class = y.cat , prob = as.numeric( pred.train ))
  sigmoidalCal <- glm(  class ~ prob  , data = train.df , family = binomial)
  sigmoidProbs <- predict(sigmoidalCal, newdata = data.frame( prob = as.numeric( pred )), type = "response")
  
  print(">> prediction (test) **** AFTER ****<<")
  print(mean(sigmoidProbs))
  
  #### assembling submission - no probs recalibration 
  sub = data.frame(bidder_id = X[teind,]$bidder_id , pred.xgb = sigmoidProbs)
  sub.full = merge(x=sub,y=sampleSubmission,by="bidder_id" , all=T)
  sub.full$prediction = ifelse( ! is.na(sub.full$pred.xgb) , sub.full$pred.xgb , 0 )
  sub.full = sub.full[,-2]
  
  ## writing on disk 
  fn = paste("sub_xgboost_stress_rs_xval" , xval.perf , ".csv" , sep='') 
  cat(">> writing prediction on disk [",fn,"]... \n")
  write.csv(sub.full,quote=FALSE, 
            file=paste(getBasePath("data"),fn,sep='') ,
            row.names=FALSE)
  
  ######## ^ 2 
  train.df = data.frame(class = y.cat , prob = as.numeric( pred.train ))
  sigmoidalCal <- glm(  class ~ I(prob^2)  , data = train.df , family = binomial)
  sigmoidProbs.2 <- predict(sigmoidalCal, newdata = data.frame( prob = as.numeric( pred )), type = "response")
  
  print(">> prediction (test) **** AFTER ^2 ****<<")
  print(mean(sigmoidProbs.2))
  
  #### assembling submission - no probs recalibration 
  sub.2 = data.frame(bidder_id = X[teind,]$bidder_id , pred.xgb = sigmoidProbs.2)
  sub.full.2 = merge(x=sub.2,y=sampleSubmission,by="bidder_id" , all=T)
  sub.full.2$prediction = ifelse( ! is.na(sub.full.2$pred.xgb) , sub.full.2$pred.xgb , 0 )
  sub.full.2 = sub.full.2[,-2]
  
  ## writing on disk 
  fn = paste("sub_xgboost_stress_rs2_xval" , xval.perf , ".csv" , sep='') 
  cat(">> writing prediction on disk [",fn,"]... \n")
  write.csv(sub.full.2,quote=FALSE, 
            file=paste(getBasePath("data"),fn,sep='') ,
            row.names=FALSE)
  
}

pred.killer = NULL
sub.full.killer = NULL
if(recalib.killer) {
  cat("recalibrating probabilities with the killer transformation ... \n") 
  print(">> prediction (test) **** BEFORE ****<<")
  print(mean(pred))
  print(">> train set labels <<")
  print(mean(y))
  cat("applying the killer transformation ... \n")
  roc.train = get_auc (probs=pred.train , y = y.cat, fact.sign = 'robot', verbose=F, doPlot=F)
  cat(">> roc.train=",roc.train,"\n")
  cat(">> mean pred.train=",mean(pred.train),"\n")
  roc.train.killer = get_auc (probs=ifelse(pred.train>0.5,pred.train,pred.train^2) , y = y.cat, fact.sign = 'robot', verbose=F, doPlot=F)
  cat(">> roc.train.killer=",roc.train.killer,"\n")
  cat(">> mean pred.train.killer=",mean(ifelse(pred.train>0.5,pred.train,pred.train^2)),"\n")
  
  pred.killer = ifelse(pred > 0.5 , pred , pred^2) 
  
  print(">> prediction (test) **** AFTER Killer ****<<")
  print(mean(pred.killer))
  
  #### assembling submission - no probs recalibration 
  sub.killer = data.frame(bidder_id = X[teind,]$bidder_id , pred.xgb = pred.killer)
  sub.full.killer = merge(x=sub.killer,y=sampleSubmission,by="bidder_id" , all=T)
  sub.full.killer$prediction = ifelse( ! is.na(sub.full.killer$pred.xgb) , sub.full.killer$pred.xgb , 0 )
  sub.full.killer = sub.full.killer[,-2]
  
  ## writing on disk 
  fn = paste("sub_xgboost_stress_rkill_xval" , xval.perf , ".csv" , sep='') 
  cat(">> writing prediction on disk [",fn,"]... \n")
  write.csv(sub.full.killer,quote=FALSE, 
            file=paste(getBasePath("data"),fn,sep='') ,
            row.names=FALSE)
}

