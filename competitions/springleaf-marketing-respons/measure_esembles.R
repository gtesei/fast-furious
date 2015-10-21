library(data.table)
library(xgboost)
library(fastfurious)

### FUNCS
findMostFrequentPredictorOnError = function(predTrain) {
  pred = as.numeric(predTrain>0.5)
  
  train_raw_err = Xtrain[ pred != Ytrain , ]
  preds = colnames(train_raw_err)
  
  a = (lapply(seq_along(preds) , function(i){
    t = sort(table(train_raw_err[,i]),decreasing=T)
    ret = t[1] / sum(Xtrain[,i] == names(t[1]))
    return(ret)
  }))
  
  pred.name = NULL
  pred.val = 0 
  pred.occurr = 0 
  aa = lapply(seq_along(a) , function(i) {
    if (a[[i]] > pred.occurr) {
      pred.name <<- preds[i]
      pred.val <<- names(a[[i]])
      pred.occurr <<- as.numeric(a[[i]]) 
    }
  })
  
  return(list(pred.name=pred.name,pred.val=pred.val,pred.occurr=pred.occurr)) 
}


### CONFIG 
DO_ERR_ANALYSIS = F

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/springleaf-marketing-respons')
ff.bindPath(type = 'code' , sub_path = 'competitions/springleaf-marketing-respons')
ff.bindPath(type = 'elab' , sub_path = 'dataset/springleaf-marketing-respons/elab') 

ff.bindPath(type = 'ensembles' , sub_path = 'dataset/springleaf-marketing-respons/ensembles/') 

## PROCS 

## Ytrain
train_raw = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
Ytrain = train_raw$target
rm(list=c("train_raw"))
gc()

## Xtrain 
Xtrain = NULL 
if (DO_ERR_ANALYSIS) {
  Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_docproc2.csv" , sep='') , stringsAsFactors = F))  
}


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
  
  ensembles_scores_j = data.frame(ID = ensembles_i , layer = rep(i,length(ensembles_i)) , 
                                  AUC=NA, 
                                  err_an.pred_name = NA , 
                                  err_an.pred_val = NA , 
                                  err_an.pred_occur = NA ) 
  
  if (!DO_ERR_ANALYSIS) ensembles_scores_j = data.frame(ID = ensembles_i , layer = rep(i,length(ensembles_i)) , AUC=NA)
  
  for (j in ensembles_i) {
    sub_j = as.data.frame( fread( paste(ff.getPath('ensembles') , ens_dir, .Platform$file.sep,j, sep='') , stringsAsFactors = F))
    predTrain = sub_j[1:length(Ytrain),'assemble']
    roc_1 = verification::roc.area(Ytrain , predTrain )$A
    #l = fastfurious:::getCaretFactors(y=Ytrain)
    #roc_2 = as.numeric( pROC::auc(pROC::roc(response = l$y.cat, predictor = predTrain, levels = levels(l$y.cat) )))
    ensembles_scores_j[ensembles_scores_j$ID==j,'AUC'] <- roc_1
    if (DO_ERR_ANALYSIS) {
      err_an = findMostFrequentPredictorOnError(predTrain=predTrain)
      ensembles_scores_j[ensembles_scores_j$ID==j,'err_an.pred_name'] <- err_an$pred.name
      ensembles_scores_j[ensembles_scores_j$ID==j,'err_an.pred_val'] <- err_an$pred.val
      ensembles_scores_j[ensembles_scores_j$ID==j,'err_an.pred_occur'] <- err_an$pred.occurr
    }
    ensembles_scores_j <- ensembles_scores_j[order(ensembles_scores_j$AUC,decreasing = T),]
  }
  if (is.null(ensembles_scores)) {
    ensembles_scores = ensembles_scores_j
  } else {
    ensembles_scores = rbind(ensembles_scores,ensembles_scores_j)
  }
  rm(ensembles_scores_j)
}

ensembles_scores <- ensembles_scores[order(ensembles_scores$AUC,decreasing = T),]

if (DO_ERR_ANALYSIS) {
  cat(">>> writing on disk ... \n")
  write.csv(ensembles_scores,
            quote=FALSE, 
            file=paste(ff.getPath("elab"),"ensembles_scores.csv",sep=''),
            row.names=FALSE)
} 



