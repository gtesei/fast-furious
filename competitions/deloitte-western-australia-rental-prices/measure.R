library(data.table)
library(xgboost)
library(fastfurious)

### FUNCS
RMSLE = function(pred, obs) {
  if (sum(pred<0)>0) {
    pred = ifelse(pred >=0 , pred , 1.5)
  }
  rmsle = sqrt(    sum( (log(pred+1) - log(obs+1))^2 )   / length(pred))
  return (rmsle)
}

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/deloitte-western-australia-rental-prices')
ff.bindPath(type = 'code' , sub_path = 'competitions/deloitte-western-australia-rental-prices')
ff.bindPath(type = 'elab' , sub_path = 'dataset/deloitte-western-australia-rental-prices/elab') 

ff.bindPath(type = 'ensembles' , sub_path = 'dataset/deloitte-western-australia-rental-prices/ensembles/') 

## PROCS 

## Ytrain
Ytrain = as.data.frame( fread(paste(ff.getPath("elab") , "Ytrain_base.csv" , sep='') , stringsAsFactors = F))
Ytrain = Ytrain$Ytrain

## Xtrain 
Xtrain = NULL 

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
  
  ensembles_scores_j = data.frame(ID = ensembles_i , layer = rep(i,length(ensembles_i)) , RMSLE=NA)
  
  for (j in ensembles_i) {
    sub_j = as.data.frame( fread( paste(ff.getPath('ensembles') , ens_dir, .Platform$file.sep,j, sep='') , stringsAsFactors = F))
    predTrain = sub_j[1:length(Ytrain),'assemble']
    ensembles_scores_j[ensembles_scores_j$ID==j,'RMSLE'] <- RMSLE(pred=predTrain, obs=Ytrain)
    ensembles_scores_j <- ensembles_scores_j[order(ensembles_scores_j$RMSLE,decreasing = F),]
  }
  if (is.null(ensembles_scores)) {
    ensembles_scores = ensembles_scores_j
  } else {
    ensembles_scores = rbind(ensembles_scores,ensembles_scores_j)
  }
  rm(ensembles_scores_j)
}

ensembles_scores <- ensembles_scores[order(ensembles_scores$RMSLE,decreasing = F),]
  
  