library(caret)
library(data.table)
library(subselect)
library(plyr)
library(binhf)
library(fBasics)

library(xgboost)
library(methods)
library(magrittr)
require(stringr)

xgb.iter.eval <- function(booster, watchlist, iter, feval = NULL, prediction = FALSE) {
  if (class(booster) != "xgb.Booster.handle") {
    stop("xgb.eval: first argument must be type xgb.Booster")
  }
  if (typeof(watchlist) != "list") {
    stop("xgb.eval: only accepts list of DMatrix as watchlist")
  }
  for (w in watchlist) {
    if (class(w) != "xgb.DMatrix") {
      stop("xgb.eval: watch list can only contain xgb.DMatrix")
    }
  }
  if (length(watchlist) != 0) {
    if (is.null(feval)) {
      evnames <- list()
      for (i in 1:length(watchlist)) {
        w <- watchlist[i]
        if (length(names(w)) == 0) {
          stop("xgb.eval: name tag must be presented for every elements in watchlist")
        }
        evnames <- append(evnames, names(w))
      }
      msg <- .Call("XGBoosterEvalOneIter_R", booster, as.integer(iter), watchlist, 
                   evnames, PACKAGE = "xgboost")
    } else {
      msg <- paste("[", iter, "]", sep="")
      for (j in 1:length(watchlist)) {
        w <- watchlist[j]
        if (length(names(w)) == 0) {
          stop("xgb.eval: name tag must be presented for every elements in watchlist")
        }
        preds <- predict(booster, w[[1]])
        ret <- feval(preds, w[[1]])
        msg <- paste(msg, "\t", names(w), "-", ret$metric, ":", ret$value, sep="")
      }
    }
  } else {
    msg <- ""
  }
  if (prediction){
    preds <- predict(booster,watchlist[[2]])
    return(list(msg,preds))
  }
  return(msg)
}

#------------------------------------------
# helper functions for cross validation
#
my.xgb.cv.mknfold <- function(dall, nfold, param, stratified, folds) {
  if (nfold <= 1) {
    stop("nfold must be bigger than 1")
  }
  if(is.null(folds)) {
    if (exists('objective', where=param) && strtrim(param[['objective']], 5) == 'rank:') {
      stop("\tAutomatic creation of CV-folds is not implemented for ranking!\n",
           "\tConsider providing pre-computed CV-folds through the folds parameter.")
    }
    y <- getinfo(dall, 'label')
    randidx <- sample(1 : my.xgb.numrow(dall))
    if (stratified & length(y) == length(randidx)) {
      y <- y[randidx]
      #
      # WARNING: some heuristic logic is employed to identify classification setting!
      #
      # For classification, need to convert y labels to factor before making the folds,
      # and then do stratification by factor levels.
      # For regression, leave y numeric and do stratification by quantiles.
      if (exists('objective', where=param)) {
        # If 'objective' provided in params, assume that y is a classification label
        # unless objective is reg:linear
        if (param[['objective']] != 'reg:linear') y <- factor(y)
      } else {
        # If no 'objective' given in params, it means that user either wants to use
        # the default 'reg:linear' objective or has provided a custom obj function.
        # Here, assume classification setting when y has 5 or less unique values:
        if (length(unique(y)) <= 5) y <- factor(y)
      }
      folds <- xgb.createFolds(y, nfold)
    } else { 
      # make simple non-stratified folds
      kstep <- length(randidx) %/% nfold
      folds <- list()
      for (i in 1:(nfold-1)) {
        folds[[i]] = randidx[1:kstep]
        randidx = setdiff(randidx, folds[[i]])
      }
      folds[[nfold]] = randidx
    }
  }
  ret <- list()
  for (k in 1:nfold) {
    dtest <- slice(dall, folds[[k]])
    didx = c()
    for (i in 1:nfold) {
      if (i != k) {
        didx <- append(didx, folds[[i]])
      }
    }
    dtrain <- slice(dall, didx)
    bst <- xgb.Booster(param, list(dtrain, dtest))
    watchlist = list(train=dtrain, test=dtest)
    ret[[k]] <- list(dtrain=dtrain, booster=bst, watchlist=watchlist, index=folds[[k]])
  }
  return (ret)
}

my.xgb.numrow <- function(dmat) {
  nrow <- .Call("XGDMatrixNumRow_R", dmat, PACKAGE="xgboost")
  return(nrow)
}
# iteratively update booster with customized statistics
xgb.iter.boost <- function(booster, dtrain, gpair) {
  if (class(booster) != "xgb.Booster.handle") {
    stop("xgb.iter.update: first argument must be type xgb.Booster.handle")
  }
  if (class(dtrain) != "xgb.DMatrix") {
    stop("xgb.iter.update: second argument must be type xgb.DMatrix")
  }
  .Call("XGBoosterBoostOneIter_R", booster, dtrain, gpair$grad, gpair$hess, 
        PACKAGE = "xgboost")
  return(TRUE)
}

xgb.createFolds <- function(y, k = 10)
{
  if(is.numeric(y)) {
    ## Group the numeric data based on their magnitudes
    ## and sample within those groups.
    
    ## When the number of samples is low, we may have
    ## issues further slicing the numeric data into
    ## groups. The number of groups will depend on the
    ## ratio of the number of folds to the sample size.
    ## At most, we will use quantiles. If the sample
    ## is too small, we just do regular unstratified
    ## CV
    cuts <- floor(length(y)/k)
    if(cuts < 2) cuts <- 2
    if(cuts > 5) cuts <- 5
    y <- cut(y,
             unique(quantile(y, probs = seq(0, 1, length = cuts))),
             include.lowest = TRUE)
  }
  
  if(k < length(y)) {
    ## reset levels so that the possible levels and
    ## the levels in the vector are the same
    y <- factor(as.character(y))
    numInClass <- table(y)
    foldVector <- vector(mode = "integer", length(y))
    
    ## For each class, balance the fold allocation as far
    ## as possible, then resample the remainder.
    ## The final assignment of folds is also randomized.
    for(i in 1:length(numInClass)) {
      ## create a vector of integers from 1:k as many times as possible without
      ## going over the number of samples in the class. Note that if the number
      ## of samples in a class is less than k, nothing is producd here.
      seqVector <- rep(1:k, numInClass[i] %/% k)
      ## add enough random integers to get  length(seqVector) == numInClass[i]
      if(numInClass[i] %% k > 0) seqVector <- c(seqVector, sample(1:k, numInClass[i] %% k))
      ## shuffle the integers for fold assignment and assign to this classes's data
      foldVector[which(y == dimnames(numInClass)$y[i])] <- sample(seqVector)
    }
  } else foldVector <- seq(along = y)
  
  out <- split(seq(along = y), foldVector)
  names(out) <- NULL
  out
}

my.xgb.get.DMatrix <- function(data, label = NULL, missing = NULL) {
  inClass <- class(data)
  if (inClass == "dgCMatrix" || inClass == "matrix") {
    if (is.null(label)) {
      stop("xgboost: need label when data is a matrix")
    }
    if (is.null(missing)){
      dtrain <- xgb.DMatrix(data, label = label)
    } else {
      dtrain <- xgb.DMatrix(data, label = label, missing = missing)
    }
  } else {
    if (!is.null(label)) {
      warning("xgboost: label will be ignored.")
    }
    if (inClass == "character") {
      dtrain <- xgb.DMatrix(data)
    } else if (inClass == "xgb.DMatrix") {
      dtrain <- data
    } else {
      stop("xgboost: Invalid input of data")
    }
  }
  return (dtrain)
}

# construct a Booster from cachelist
xgb.Booster <- function(params = list(), cachelist = list(), modelfile = NULL) {
  if (typeof(cachelist) != "list") {
    stop("xgb.Booster: only accepts list of DMatrix as cachelist")
  }
  for (dm in cachelist) {
    if (class(dm) != "xgb.DMatrix") {
      stop("xgb.Booster: only accepts list of DMatrix as cachelist")
    }
  }
  handle <- .Call("XGBoosterCreate_R", cachelist, PACKAGE = "xgboost")
  if (length(params) != 0) {
    for (i in 1:length(params)) {
      p <- params[i]
      .Call("XGBoosterSetParam_R", handle, gsub("\\.", "_", names(p)), as.character(p),
            PACKAGE = "xgboost")
    }
  }
  if (!is.null(modelfile)) {
    if (typeof(modelfile) == "character") {
      .Call("XGBoosterLoadModel_R", handle, modelfile, PACKAGE = "xgboost")
    } else if (typeof(modelfile) == "raw") {
      .Call("XGBoosterLoadModelFromRaw_R", handle, modelfile, PACKAGE = "xgboost")      
    } else {
      stop("xgb.Booster: modelfile must be character or raw vector")
    }
  }
  return(structure(handle, class = "xgb.Booster.handle"))
}

# iteratively update booster with dtrain
xgb.iter.update <- function(booster, dtrain, iter, obj = NULL) {
  if (class(booster) != "xgb.Booster.handle") {
    stop("xgb.iter.update: first argument must be type xgb.Booster.handle")
  }
  if (class(dtrain) != "xgb.DMatrix") {
    stop("xgb.iter.update: second argument must be type xgb.DMatrix")
  }
  
  if (is.null(obj)) {
    .Call("XGBoosterUpdateOneIter_R", booster, as.integer(iter), dtrain, 
          PACKAGE = "xgboost")
  } else {
    pred <- predict(booster, dtrain)
    gpair <- obj(pred, dtrain)
    succ <- xgb.iter.boost(booster, dtrain, gpair)
  }
  return(TRUE)
}

xgb.cv.aggcv <- function(res, showsd = TRUE) {
  header <- res[[1]]
  ret <- header[1]
  for (i in 2:length(header)) {
    kv <- strsplit(header[i], ":")[[1]]
    ret <- paste(ret, "\t", kv[1], ":", sep="")
    stats <- c()
    stats[1] <- as.numeric(kv[2])    
    for (j in 2:length(res)) {
      tkv <- strsplit(res[[j]][i], ":")[[1]]
      stats[j] <- as.numeric(tkv[2])
    }
    ret <- paste(ret, sprintf("%f", mean(stats)), sep="")
    if (showsd) {
      ret <- paste(ret, sprintf("+%f", sd(stats)), sep="")
    }
  }
  return (ret)
}

my.xgb.cv = function (params = list(), data, nrounds, nfold, label = NULL, 
          missing = NULL, prediction = FALSE, showsd = TRUE, metrics = list(), 
          obj = NULL, feval = NULL, stratified = TRUE, folds = NULL, 
          verbose = T, ...) 
{
  
  if (typeof(params) != "list") {
    stop("xgb.cv: first argument params must be list")
  }
  if (!is.null(folds)) {
    if (class(folds) != "list" | length(folds) < 2) {
      stop("folds must be a list with 2 or more elements that are vectors of indices for each CV-fold")
    }
    nfold <- length(folds)
  }
  if (nfold <= 1) {
    stop("nfold must be bigger than 1")
  }
  if (is.null(missing)) {
    dtrain <- my.xgb.get.DMatrix(data, label)
  }
  else {
    dtrain <- my.xgb.get.DMatrix(data, label, missing)
  }
  params <- append(params, list(...))
  params <- append(params, list(silent = 1))
  for (mc in metrics) {
    params <- append(params, list(eval_metric = mc))
  }
  xgb_folds <- my.xgb.cv.mknfold(dtrain, nfold, params, stratified, 
                              folds)
  obj_type = params[["objective"]]
  mat_pred = FALSE
  if (!is.null(obj_type) && obj_type == "multi:softprob") {
    num_class = params[["num_class"]]
    if (is.null(num_class)) 
      stop("must set num_class to use softmax")
    predictValues <- matrix(0, my.xgb.numrow(dtrain), num_class)
    mat_pred = TRUE
  }
  else predictValues <- rep(0, my.xgb.numrow(dtrain))
  history <- c()
  ####
  dt = NULL 
  inCV = T 
  iit = 0  
  while (inCV) {
  ####
  ivect = ((iit*nrounds)+1):((iit+1)*nrounds)
  for (i in ivect) {
    msg <- list()
    for (k in 1:nfold) {
      fd <- xgb_folds[[k]]
      succ <- xgb.iter.update(fd$booster, fd$dtrain, i - 
                                1, obj)
      if (i < nrounds) {
        msg[[k]] <- xgb.iter.eval(fd$booster, fd$watchlist, 
                                  i - 1, feval) %>% stringr::str_split("\t") %>% .[[1]]
      }
      else {
        if (!prediction) {
          msg[[k]] <- xgb.iter.eval(fd$booster, fd$watchlist, 
                                    i - 1, feval) %>% stringr::str_split("\t") %>% .[[1]]
        }
        else {
          res <- xgb.iter.eval(fd$booster, fd$watchlist, 
                               i - 1, feval, prediction)
          if (mat_pred) {
            pred_mat = matrix(res[[2]], num_class, length(fd$index))
            predictValues[fd$index, ] <- t(pred_mat)
          }
          else {
            predictValues[fd$index] <- res[[2]]
          }
          msg[[k]] <- res[[1]] %>% stringr::str_split("\t") %>% 
            .[[1]]
        }
      }
    }
    ret <- xgb.cv.aggcv(msg, showsd)
    history <- c(history, ret)
    if (verbose) 
      paste(ret, "\n", sep = "") %>% cat
    #print(str(ret))
    #print(ret)
  }
  
  ##### costruisci quel cazzo che devi costruire .... 
  colnames <- stringr::str_split(string = history[1], pattern = "\t")[[1]] %>% 
    .[2:length(.)] %>% stringr::str_extract(".*:") %>% stringr::str_replace(":", 
                                                          "") %>% stringr::str_replace("-", ".")
  colnamesMean <- paste(colnames, "mean")
  if (showsd) 
    colnamesStd <- paste(colnames, "std")
  colnames <- c()
  if (showsd) 
    for (i in 1:length(colnamesMean)) colnames <- c(colnames, 
                                                    colnamesMean[i], colnamesStd[i])
  else colnames <- colnamesMean
  type <- rep(x = "numeric", times = length(colnames))
  dt <- read.table(text = "", colClasses = type, col.names = colnames) %>% 
    as.data.table
  split <- stringr::str_split(string = history, pattern = "\t")
  for (line in split) dt <- line[2:length(line)] %>% stringr::str_extract_all(pattern = "\\d*\\.+\\d*") %>% 
    unlist %>% as.numeric %>% as.list %>% {
      rbindlist(list(dt, .), use.names = F, fill = F)
    }
  
  ##### checkkati il gradiente!! 
  iit = iit + 1
  
  early.stop = which(dt$test.mlogloss.mean == min(dt$test.mlogloss.mean) )
  
  if (early.stop < ivect[nrounds]) {
    inCV = F
    cat(">> [eta:",params$eta,"] [",early.stop,"== early.stop < max.curr.iter ==",ivect[nrounds],"] [perf.test=",min(dt$test.mlogloss.mean),"] --> stopping [perf.test=",min(dt$test.mlogloss.mean),"] ... \n") 
  } else {
    cat(">> [eta:",params$eta,"] [early.stop == max.curr.iter ==",ivect[nrounds],"] [perf.test=",min(dt$test.mlogloss.mean),"] --> redo-cv ... another ",nrounds," rounds ... \n") 
  }
  
  gc() 
  } ##### end while inCV 
  
  ### finally 
  if (prediction) {
    return(list(dt = dt, pred = predictValues))
  }
  return(dt)
}


getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/otto-group-product-classification-challenge"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/otto-group-product-classification-challenge/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/otto-group-product-classification-challenge"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/otto-group-product-classification-challenge/"
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

#######
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

train = as.data.frame( fread(paste(getBasePath("data") , 
                                   "train.csv" , sep=''))) 

test = as.data.frame( fread(paste(getBasePath("data") , 
                                  "test.csv" , sep='')))


########
verbose = T

#########
train = train[,-1]
test = test[,-1]

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

train = train[,-ncol(train)]

trans.scal <- preProcess(rbind(train,test),
                        method = c("center", "scale") )

trans.scal
train = predict(trans.scal,train)
test = predict(trans.scal,test)

######
x = rbind(train,test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "eta" = 0.0001,  ## suggested in ESLII
              "gamma" = 0.5,  
              "max_depth" = 25, 
              "subsample" = 0.5 , ## suggested in ESLII
              "nthread" = 10, 
              
              "min_child_weight" = 1 , 
              "colsample_bytree" = 0.5, 
              "max_delta_step" = 1
              )

cat(">>Params:\n")
print(param)

# Run Cross Valication
cat(">>Cross validation ... \n")

inCV = T
early.stop = cv.nround = 500
bst.cv = NULL

#while (inCV) {
  
  cat(">> cv.nround: ",cv.nround,"\n") 
  bst.cv = my.xgb.cv(param=param, data = x[trind,], label = y, 
                  nfold = 5, nrounds=cv.nround)
#  print(bst.cv)
  early.stop = which(bst.cv$test.mlogloss.mean == min(bst.cv$test.mlogloss.mean) )
  cat(">> early.stop: ",early.stop," [test.perf:",bst.cv[early.stop,]$test.mlogloss.mean,"]\n") 
#  if (early.stop < cv.nround) {
#     inCV = F
#     cat(">> stopping [early.stop < cv.nround=",cv.nround,"] ... \n") 
#   } else {
#     cat(">> redo-cv [early.stop == cv.nround=",cv.nround,"] with 2 * cv.nround ... \n") 
#     cv.nround = cv.nround * 2 
#   }
#   gc()
# }

cat(">>Train the model ... \n")
# Train the model
nround = early.stop
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

# Make prediction
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
#write.csv(pred,file='submission.csv', quote=FALSE,row.names=FALSE)
write.csv(pred,file=paste(getBasePath("data") , 
                          "sub_xgb_boost_mod_eta_",param$eta,".csv" , sep=''), quote=FALSE,row.names=FALSE)

