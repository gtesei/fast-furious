library(data.table)
library(xgboost)
library(fastfurious)

### FUNCS
RMSPE.ens = function(pred, obs) {
  ignIdx = which(obs==0)
  if (length(ignIdx)>0) {
    obs = obs[-ignIdx]
    pred = pred[-ignIdx]
  }
  
  stopifnot(sum(obs==0)==0)
  
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)
  
  rmspe = sqrt(mean( ((1-pred/obs)^2) ) )
  return (rmspe)
}

### CONF
DO_GREEDY_PICKING = F

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/rossmann-store-sales')
ff.bindPath(type = 'code' , sub_path = 'competitions/rossmann-store-sales')
ff.bindPath(type = 'elab' , sub_path = 'dataset/rossmann-store-sales/elab') 

ff.bindPath(type = 'ensembles' , sub_path = 'dataset/rossmann-store-sales/ensembles/') 

## PROCS 

## Ytrain
train_raw = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_bench.csv" , sep='') , stringsAsFactors = F))
Ytrain = train_raw$Sales
rm(list=c("train_raw"))
gc()

## ens_dirs
ens_dirs = list.files( ff.getPath('ensembles') )
ens_dirs = ens_dirs[which(substr(x = ens_dirs, start = 1 , stop = nchar('ensemble_')) == "ensemble_")]
cat(">>> Found ",length(ens_dirs),"ensemble directory:",ens_dirs,"\n")

## ens_dir
ensembles_scores = NULL
for (i in 1:length(ens_dirs)) {
  ens_dir = paste0('ensemble_',i)
  stopifnot(ens_dir %in% ens_dirs) 
  ensembles_i = list.files( paste0(ff.getPath('ensembles') , ens_dir) )
  cat(">>> processing ",ens_dir," --> found ",length(ensembles_i),"ensembles...\n")
  if (length(ensembles_i) == 0) next 
  
  ensembles_scores_j = data.frame(ID = ensembles_i , layer = rep(i,length(ensembles_i)) , RMSPE=NA)
  
  for (j in ensembles_i) {
    sub_j = as.data.frame( fread( paste(ff.getPath('ensembles') , ens_dir, .Platform$file.sep,j, sep='') , stringsAsFactors = F))
    predTrain = sub_j[1:length(Ytrain),'assemble']
    ensembles_scores_j[ensembles_scores_j$ID==j,'RMSPE'] <- RMSPE.ens(pred=predTrain, obs=Ytrain)
    ensembles_scores_j <- ensembles_scores_j[order(ensembles_scores_j$RMSPE,decreasing = F),]
  }
  if (is.null(ensembles_scores)) {
    ensembles_scores = ensembles_scores_j
  } else {
    ensembles_scores = rbind(ensembles_scores,ensembles_scores_j)
  }
  rm(ensembles_scores_j)
}

ensembles_scores <- ensembles_scores[order(ensembles_scores$RMSPE,decreasing = F),]

## DO_GREEDY_PICKING 
if (DO_GREEDY_PICKING) {
  cat (">>> doing greedy picking ... \n")
  ensembles_train = NULL
  ensembles_test = NULL
  
  
  ## ensembles_train / ensembles_test 
  for (i in 1:length(ens_dirs)) {
    ens_dir = paste0('ensemble_',i)
    stopifnot(ens_dir %in% ens_dirs) 
    ensembles_i = list.files( paste0(ff.getPath('ensembles') , ens_dir) )
    cat(">>> processing ",ens_dir," --> found ",length(ensembles_i),"ensembles...\n")
    if (length(ensembles_i) == 0) next 
    
    for (j in ensembles_i) {
      sub_j = as.data.frame( fread( paste(ff.getPath('ensembles') , ens_dir, .Platform$file.sep,j, sep='') , stringsAsFactors = F))
      predTrain = sub_j[1:length(Ytrain),'assemble']
      predTest = sub_j[(length(Ytrain)+1):nrow(sub_j),'assemble']
      
      if (is.null(ensembles_train)) {
        ensembles_train = data.frame(V1=predTrain)
        ensembles_test = data.frame(V1=predTest)
        colnames(ensembles_train) = j
        colnames(ensembles_test) = j
      } else {
        ensembles_train_tmp = data.frame(V1=predTrain)
        ensembles_test_tmp = data.frame(V1=predTest)
        colnames(ensembles_train_tmp) = j
        colnames(ensembles_test_tmp) = j
        ensembles_train = cbind(ensembles_train,ensembles_train_tmp)
        ensembles_test = cbind(ensembles_test,ensembles_test_tmp)
      }
    }
  }
  
  
  ## Xtrain / Xtest / Ytrain 
  cat(">>> loading Xtrain / Xtest / Ytrain  ... \n")
  Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_bench.csv" , sep='') , stringsAsFactors = F))
  Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_bench.csv" , sep='') , stringsAsFactors = F))
  Ytrain = Xtrain$Sales
  test_id = Xtest$Id
  
  ## stores 
  stores_test = sort(unique(Xtest$Store))
  predTest = rep(NA,nrow(Xtest))
  for (i in seq_along(stores_test)) {
    cat(">>> processing store ",stores_test[i]," - [",i,"/",length(stores_test),"] ...\n") 
    trIdx = which(Xtrain$Store==stores_test[i])
    teIdx = which(Xtest$Store==stores_test[i])
    
    Ytrain_st = Ytrain[trIdx]
    ensembles_train_st = ensembles_train[trIdx,,drop=F]
    
    ## find best model 
    perf = unlist(lapply(ensembles_train_st,function(x) {
      return(RMSPE.ens(x,Ytrain_st))
    }))
    idxBestMod = as.numeric(which(perf==min(perf)))
    nmBestMod = names(ensembles_train_st)[idxBestMod]
    cat(">>> best model:",nmBestMod,"...\n")
    
    ## predict 
    predTest[teIdx] = as.numeric(ensembles_test[teIdx,idxBestMod])
  }
  
  ## write prediction on disk 
  stopifnot(sum(is.na(predTest))==0)
  stopifnot(sum(predTest==Inf)==0)
  submission <- data.frame(Id=test_id)
  submission$Sales <- predTest
  stopifnot(nrow(submission)==41088)
  print(head(submission))
  write.csv(submission,
            quote=FALSE, 
            file= paste(ff.getPath("elab") , "sub_greedy_pick.csv", sep='') ,
            row.names=FALSE)
}

