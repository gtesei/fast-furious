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

bt_train_predict = function(Xtrain, y.cat , controlObject , Xval , fact.sign = 'robot' , verbose = F) {
  model <- train( x = Xtrain , y = y.cat,  
                  tuneGrid = expand.grid(.trials = c(1, (1:10)*10), .model = "tree", .winnow = c(TRUE, FALSE) ),
                  method = "C5.0",  metric = "ROC", trControl = controlObject)
  
  perf.roc = max(model$results$ROC)
  cat(">> ROC:",perf.roc,"\n")
  
  cat(">> predicting ... \n")
  pred = predict(model , Xval , type = "prob") [,fact.sign]
  
  list(perf=perf.roc , pred_xval=pred)
  
}
  
svm_train_predict = function(Xtrain, y.cat , controlObject , Xval , fact.sign = 'robot' , verbose = F) {
  x = as.matrix(Xtrain)
  x = matrix(as.numeric(x),nrow(x),ncol(x))
  sigmaRangeReduced <- sigest(x)
  svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1], .C = 2^(seq(-4, 4)))
  rm(x)
  
  cat(">>> SVM training ... \n")
  svmRFitCost <- train(x = Xtrain , y = y.cat , 
                       method = "svmRadial",
                       metric = "ROC",
                       #class.weights = c(human = 7, robot = 3),
                       tuneGrid = svmRGridReduced,
                       trControl = controlObject)
  perf.roc = max(svmRFitCost$results$ROC)
  cat(">> ROC:",perf.roc,"\n")
  
  cat(">> predicting ... \n")
  pred = predict(svmRFitCost , Xval , type = "prob") [,fact.sign]
  
  list(perf=perf.roc , pred_xval=pred)
}
  
xgb_train_predict = function(Xtrain, y , Xval , verbose = F) {
  cat(">>> dim Xtrain [no bidder_id]:",dim(Xtrain),"\n")
  
  ######### XGboost 
  x = as.matrix(Xtrain)
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
    bst.cv = xgb.cv(param=param, data = x, label = y, nfold = 5, nrounds=cv.nround , verbose = verbose)    
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
  xval = as.matrix(Xval)
  xval = matrix(as.numeric(xval),nrow(xval),ncol(xval))
  
  cat(">> training and making prediction on Xval ... \n")
  bst = xgboost(param = param, data = x , label = y, nrounds = early.stop , verbose = verbose) 
  
  pred = predict(bst,xval)
  
  list(early_stop = early.stop , perf=xval.perf , pred_xval=pred)
}

source(paste0( getBasePath("process") , "/Classification_Lib.R"))

#################
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep=''))) ## outcome = 0 human 

train = as.data.frame( fread(paste(getBasePath("data") , 
                                   "train.csv" , sep=''))) 

test = as.data.frame( fread(paste(getBasePath("data") , 
                                  "test.csv" , sep='')))

bids = as.data.frame( fread(paste(getBasePath("data") , 
                                  "bids.csv" , sep='')))

secure_humans = as.data.frame( fread(paste(getBasePath("data") , 
                                           "secure_humans_idx.csv" , sep='')))

X = as.data.frame( fread(paste(getBasePath("data") , 
                               "Xfin5.csv" , sep='')))

## train/test index , labels  
train.full = merge(x = bids , y = train , by="bidder_id"  )
trind = 1:length(unique(train.full$bidder_id))
teind = (max(trind)+1):nrow(X)

X.full = merge(x=X , y=train , by="bidder_id")
y = X.full$outcome
y.cat = factor(y)
levels(y.cat) = c("human","robot")

rm(train.full)
rm(X.full)
rm(bids)

X.base = X[,-grep("bidder_id" , colnames(X) )]

############ config
round_num = 3
controlObject <- trainControl(method = "repeatedcv", number = 5 , summaryFunction = twoClassSummary , classProbs = TRUE)

############
cat(">>> Human / Robot frequencies <<<")
table(y.cat)
table(y.cat)/length(y.cat)

robot_freq_nat = sum(y==1)/length(y)
human_freq_nat = 1 - robot_freq_nat

#robot_freq = c(0.5,0.35,0.20)
robot_freq = c(robot_freq_nat,0.10,0.15,0.20)

#models = c("xgB","SVMClass" , "BoostedTreesClass")
models = c("xgB")
metrics = c("auc","sens","spec")
rounds = c(as.character(1:round_num),"avg")

perf_grid = expand.grid(model = models , metric = metrics , round=rounds , robot_freq = robot_freq , performance = NA ) 

####
robots_idx = which(y==1)
humans_idx = which(y==0)

for (rf in robot_freq) {
  cat(">>> processing ",rf,"...\n") 
  
  ## xval data external layer (10%): 5% robot , 95% human 
  xval_robot_idx = sample(robots_idx, ceil(length(robots_idx) * 0.1 ) )
  xval_human_idx = sample(humans_idx, ceil(length(humans_idx) * 0.1 ) ) 
  xval_idx = sample(c(xval_human_idx,xval_robot_idx))
  
  #### 
  auc = sens = spec = rep(NA,length(models))
  for (rd in 1:round_num) {
    ## training data: rf % robot (1-rf) human 
    robots_idx_res = robots_idx[which( ! robots_idx %in% xval_robot_idx)]
    humans_idx_res = humans_idx[which( ! humans_idx %in% xval_human_idx)]
    
    train_robot_idx = robots_idx_res
    train_human_idx = sample(humans_idx_res, ceil(length(robots_idx_res) * ((1-rf)/rf) ) ) 
    
    train_idx = sample(c(train_robot_idx,train_human_idx))
    
    cat(">> processing round ",rd," [rf=",rf,"] ...\n") 
    for (md in 1:length(models)) {
      cat(">> processing model ",models[md]," [rf=",rf,"] [round=",rd,"]...\n") 
      if (md == 1) {
        ## xgB
        xgb = NULL
        xgb = tryCatch({ 
          xgb_train_predict( Xtrain = X.base[train_idx,] , y = y[train_idx] , Xval = X.base[xval_idx,] , verbose=F )
        } , error = function(err) { 
          print(paste("ERROR:  ",err))
          NULL
        })
        
        if( is.null(xgb) ) {
          perf_grid[perf_grid$model == models[md] & perf_grid$metric == 'auc' & perf_grid$round == rd & perf_grid$robot_freq == rf, 'performance'] = 0 ## penalize 
        } else {
          perf.xgb = get_auc (probs = xgb$pred_xval , y = y.cat[xval_idx], fact.sign = 'robot', verbose=F, doPlot=F)
          perf_grid[perf_grid$model == models[md] & perf_grid$metric == 'auc' & perf_grid$round == rd & perf_grid$robot_freq == rf, 'performance'] = perf.xgb
          cat(">> xgb xval (internal layer) perf=",xgb$perf," ... prediction on xval (external layer):",
              perf_grid[perf_grid$model == models[md] & perf_grid$metric == 'auc' & perf_grid$round == rd & perf_grid$robot_freq == rf, 'performance']
              ," \n") 
        }
      } else if (md == 2) {
        ## SVMClass
        svm = NULL
        svm = tryCatch({ 
          svm_train_predict( Xtrain = X.base[train_idx,] , y.cat = y.cat[train_idx] , controlObject = controlObject , Xval = X.base[xval_idx,] , verbose=F )
        } , error = function(err) { 
          print(paste("ERROR:  ",err))
          NULL
        })
        
        if( is.null(svm) ) {
          perf_grid[perf_grid$model == models[md] & perf_grid$metric == 'auc' & perf_grid$round == rd & perf_grid$robot_freq == rf, 'performance'] = 0 ## penalize 
        } else {
          perf.svm = get_auc (probs = svm$pred_xval , y = y.cat[xval_idx], fact.sign = 'robot', verbose=F, doPlot=F)
          perf_grid[perf_grid$model == models[md] & perf_grid$metric == 'auc' & perf_grid$round == rd & perf_grid$robot_freq == rf, 'performance'] = perf.svm
          cat(">> svm xval (internal layer) perf=",svm$perf," ... prediction on xval (external layer):",
              perf_grid[perf_grid$model == models[md] & perf_grid$metric == 'auc' & perf_grid$round == rd & perf_grid$robot_freq == rf, 'performance']
              ," \n") 
        }
      } else if (md == 3) {
        ## BoostedTreesClass
        bt = NULL
        bt = tryCatch({ 
          bt_train_predict( Xtrain = X.base[train_idx,] , y.cat = y.cat[train_idx] , controlObject = controlObject , Xval = X.base[xval_idx,] , verbose=F )
        } , error = function(err) { 
          print(paste("ERROR:  ",err))
          NULL
        })
        
        if( is.null(bt) ) {
          perf_grid[perf_grid$model == models[md] & perf_grid$metric == 'auc' & perf_grid$round == rd & perf_grid$robot_freq == rf, 'performance'] = 0 ## penalize 
        } else {
          perf.bt = get_auc (probs = bt$pred_xval , y = y.cat[xval_idx], fact.sign = 'robot', verbose=F, doPlot=F)
          perf_grid[perf_grid$model == models[md] & perf_grid$metric == 'auc' & perf_grid$round == rd & perf_grid$robot_freq == rf, 'performance'] = perf.bt
          cat(">> BoostedTrees xval (internal layer) perf=",bt$perf," ... prediction on xval (external layer):",
              perf_grid[perf_grid$model == models[md] & perf_grid$metric == 'auc' & perf_grid$round == rd & perf_grid$robot_freq == rf, 'performance']
              ," \n") 
        }
      } else {
        stop("md>3 not supported at the moment") 
      } 
    } ## end of models 
    
    ### computing averages 
    for (md in 1:length(models)) { 
      perf_grid[perf_grid$model == models[md] & perf_grid$metric == 'auc' & perf_grid$round == 'avg' & perf_grid$robot_freq == rf, ]$performance = mean(
        perf_grid[perf_grid$model == models[md] & perf_grid$metric == 'auc' & perf_grid$round != 'avg' & perf_grid$robot_freq == rf, ]$performance)
    }
  } ## end of rounds 
  cat(">> finished [robot frequency =",rf,"] \n") 
  print(perf_grid[perf_grid$metric == 'auc' & perf_grid$round == 'avg' & perf_grid$robot_freq == rf, ])
  
  model_winner = as.character (perf_grid[perf_grid$metric == 'auc' & perf_grid$round == 'avg' & perf_grid$robot_freq == rf, ]$model [which (
    perf_grid[perf_grid$metric == 'auc' & perf_grid$round == 'avg' & perf_grid$robot_freq == rf, ]$performance == 
           max(perf_grid[perf_grid$metric == 'auc' & perf_grid$round == 'avg' & perf_grid$robot_freq == rf, ]$performance))])
  cat(">> the winner is ... ",model_winner, " ... storing on disk ... \n") 
  write.csv(perf_grid,quote=FALSE, 
            file=paste(getBasePath("data"),"unbalanced_class_perf_grid.csv",sep='') ,
            row.names=FALSE)
}

## conslusions 
cat(">> CONCLUSIONS ... \n") 
print(perf_grid[perf_grid$metric == 'auc' & perf_grid$round == 'avg' , ])

model_winner = as.character (perf_grid[perf_grid$metric == 'auc' & perf_grid$round == 'avg' , ]$model [which (
  perf_grid[perf_grid$metric == 'auc' & perf_grid$round == 'avg' , ]$performance == 
    max(perf_grid[perf_grid$metric == 'auc' & perf_grid$round == 'avg' , ]$performance))])

freq_winner = as.numeric (perf_grid[perf_grid$metric == 'auc' & perf_grid$round == 'avg' , ]$robot_freq [which (
  perf_grid[perf_grid$metric == 'auc' & perf_grid$round == 'avg' , ]$performance == 
    max(perf_grid[perf_grid$metric == 'auc' & perf_grid$round == 'avg' , ]$performance))])

perf_winner = as.numeric (perf_grid[perf_grid$metric == 'auc' & perf_grid$round == 'avg' , ]$performance [which (
  perf_grid[perf_grid$metric == 'auc' & perf_grid$round == 'avg' , ]$performance == 
    max(perf_grid[perf_grid$metric == 'auc' & perf_grid$round == 'avg' , ]$performance))])


cat(">> the winner is ... MODEL =",model_winner, "  FREQUENCY =",freq_winner,  "  AUC=",perf_winner," ... storing on disk ... \n") 
write.csv(perf_grid,quote=FALSE, 
          file=paste(getBasePath("data"),"unbalanced_class_perf_grid.csv",sep='') ,
          row.names=FALSE)

### training the winning model with the winning freqeuncy
cat(">> training the winning model with the winning freqeuncy ... \n")

train_robot_idx = robots_idx
train_human_idx = sample(humans_idx, ceil(length(robots_idx) * ((1-freq_winner)/freq_winner) ) ) 
train_idx = sample(c(train_robot_idx,train_human_idx))

pred = NULL
xval.perf = -1 

if (model_winner == models[1]) {
  ## xgB
  xgb = NULL
  xgb = tryCatch({ 
    xgb_train_predict( Xtrain = X.base[train_idx,] , y = y[train_idx] , Xval = X.base[teind,] , verbose=F )
  } , error = function(err) { 
    print(paste("ERROR:  ",err))
    NULL
  })
  
  if( is.null(xgb) ) {
    stop("problem training the winning model") 
  } else {
    pred = xgb$pred_xval
    xval.perf = xgb$perf
  }
} else if (model_winner == models[2]) {
  ## SVMClass
  svm = NULL
  svm = tryCatch({ 
    svm_train_predict( Xtrain = X.base[train_idx,] , y.cat = y.cat[train_idx] , controlObject = controlObject , Xval = X.base[teind,] , verbose=F )
  } , error = function(err) { 
    print(paste("ERROR:  ",err))
    NULL
  })
  
  if( is.null(svm) ) {
    stop("problem training the winning model")  
  } else {
    pred = svm$pred_xval
    xval.perf = svm$perf
  }
} else if (model_winner == models[3]) {
  ## BoostedTreesClass
  bt = NULL
  bt = tryCatch({ 
    bt_train_predict( Xtrain = X.base[train_idx,] , y.cat = y.cat[train_idx] , controlObject = controlObject , Xval = X.base[teind,] , verbose=F )
  } , error = function(err) { 
    print(paste("ERROR:  ",err))
    NULL
  })
  
  if( is.null(bt) ) {
    stop("problem training the winning model") 
  } else {
    pred = bt$pred_xval
    xval.perf = bt$perf
  }
} else {
  stop("which model_winner??") 
} 

#### assembling submission - no probs recalibration 
sub = data.frame(bidder_id = X[teind,]$bidder_id , pred = pred)
sub.full.base = merge(x=sub,y=sampleSubmission,by="bidder_id" , all=T)
sub.full.base$prediction = ifelse( ! is.na(sub.full.base$pred) , sub.full.base$pred , 0 )
sub.full.base = sub.full.base[,-2]

## writing on disk 
fn = paste("sub_unbalanced_mod_",model_winner,"_xval" , xval.perf , ".csv" , sep='') 
cat(">> writing prediction on disk [",fn,"]... \n")
write.csv(sub.full.base,quote=FALSE, 
          file=paste(getBasePath("data"),fn,sep='') ,
          row.names=FALSE)
