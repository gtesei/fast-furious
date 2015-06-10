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

train.cv = function(param , cv.nround , k.fold = 5 , verbose = T) {
  ### echo 
  if (verbose) {
    cat(">>Params:\n")
    print(param)
    cat(">> cv.nround: ",cv.nround,"\n") 
  }
  
  ### Cross-validation 
  if (verbose) 
    cat(">>Cross Validation ... \n")
  inCV = T
  xval.perf = -1
  bst.cv = NULL
  early.stop = -1
  
  while (inCV) {
    if (verbose) 
      cat(">>> maximizing auc ...\n")
    bst.cv = xgb.cv(param=param, data = x[trind,], label = y, nfold = k.fold, nrounds=cv.nround )    
    print(bst.cv)
    early.stop = min(which(bst.cv$test.auc.mean == max(bst.cv$test.auc.mean) ))
    xval.perf = bst.cv[early.stop,]$test.auc.mean
    if (verbose) 
      cat(">> early.stop: ",early.stop," [xval.perf:",xval.perf,"]\n") 
    
    if (early.stop < cv.nround) {
      inCV = F
      if (verbose) 
        cat(">> stopping [early.stop < cv.nround=",cv.nround,"] ... \n") 
    } else {
      if (verbose) 
        cat(">> redo-cv [early.stop == cv.nround=",cv.nround,"] with 2 * cv.nround ... \n") 
      cv.nround = cv.nround * 2 
    }
    
    gc()
  }
  
  list(xval.perf , early.stop)
}

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

################# Model 
## elimino bidder_id 
X.base = X[,-grep("bidder_id" , colnames(X) )]
cat(">>> dim X.base [no bidder_id]:",dim(X.base),"\n")


################# Config 
xgbGrid  <- expand.grid(eta = c(0.09,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05) 
                        , gamma = c(0,0.5,0.7,1,2,5) 
                        , max_delta_step = c(0,1)
                        , subsample = c(0.5,1)
                        , min_child_weight = c(1,0)
                        , colsample_bytree = c(0.5,1)
                        , xval.perf = NA 
                        , early.stop = NA
                        )
iter = 5
cv.nround = 2500

######### XGboost 
x = as.matrix(X.base)
x = matrix(as.numeric(x),nrow(x),ncol(x))

##### xgboost --> set necessary parameter
param <- list("objective" = "binary:logistic",
              "eval_metric" = "auc",
              "eta" = 0.05,  ## suggested in ESLII
              "gamma" = 0.7,  
              "max_depth" = 6, 
              "subsample" = 0.5 , ## suggested in ESLII
              "nthread" = 10, 
              "min_child_weight" = 1 , 
              "colsample_bytree" = 0.5, 
              "max_delta_step" = 1)

for (i in 569:nrow(xgbGrid)) {
  eta = xgbGrid[i,]$eta 
  gamma = xgbGrid[i,]$gamma 
  max_delta_step = xgbGrid[i,]$max_delta_step 
  subsample = xgbGrid[i,]$subsample
  min_child_weight = xgbGrid[i,]$min_child_weight
  colsample_bytree = xgbGrid[i,]$colsample_bytree
  
  cat(">> processing [",i,"/",nrow(xgbGrid),"] [eta=",eta,"] [gamma=",gamma,"] [max_delta_step=",max_delta_step,"] [subsample=",subsample,"] [min_child_weight="
      ,min_child_weight,"] [colsample_bytree=",colsample_bytree,"]...\n") 
  
  param['eta'] = eta
  param['gamma'] = gamma
  param['max_delta_step'] = max_delta_step
  param['subsample'] = subsample
  param['min_child_weight'] = min_child_weight
  param['colsample_bytree'] = colsample_bytree
  
  xval.perf.vect = rep(NA,iter)
  early.stop.vect = rep(NA,iter)
  
  for (j in 1:iter) {
     l = train.cv(param , cv.nround , k.fold = 5 , verbose = T)
     xval.perf.vect[j] = l[[1]]
     early.stop.vect[j] = l[[2]]
     
  }
  
  ### update grid 
  xgbGrid[i,]$xval.perf = mean(xval.perf.vect) 
  xgbGrid[i,]$early.stop = mean(early.stop.vect)
}

xgbGrid = xgbGrid[order(xgbGrid$xval.perf , decreasing = T) , ]
cat("\n*************** TOP PERFORMANCE ***************\n")
print(head(xgbGrid))
cat("\n***************\n")
cat(">>> BEST configuration <<<<\n")
print(xgbGrid[1,]) 
write.csv(xgbGrid,quote=FALSE, 
          file=paste(getBasePath("data"),"xgb_perf_tuning_grid.csv",sep='') ,
          row.names=FALSE)




