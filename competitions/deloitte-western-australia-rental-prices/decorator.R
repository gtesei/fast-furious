library(data.table)
library(fastfurious)

### FUNCS

### CONFIG 
if (!exists("CUTOFF")) CUTOFF = 100
cat(">>> CUTOFF =",CUTOFF,"days ...\n")

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/deloitte-western-australia-rental-prices/data')
ff.bindPath(type = 'code' , sub_path = 'competitions/deloitte-western-australia-rental-prices') 
ff.bindPath(type = 'elab' , sub_path = 'dataset/deloitte-western-australia-rental-prices/elab') 

ff.bindPath(type = 'submission_1' , sub_path = 'dataset/deloitte-western-australia-rental-prices/ensembles/pred_ensemble_1') 

################# PROCS
# if (!exists("fn")) fn = 'layer1_dataProcNAs4_ytranflog_modxgbTreeGTJ_eta0.02_max_depth9_tuneTRUE.csv'
# basePath = ff.getPath('submission_1')

if (!exists("fn")) fn = 'merge4_50_50.csv'
basePath = ff.getPath('submission_1')
  
fn.dec = paste(strsplit(x=fn,split=".csv")[[1]],"_decorated_CUTOFF",CUTOFF,".csv",sep='')
fn = paste0(basePath,fn)
fn.dec = paste0(basePath,fn.dec)
stopifnot(file.exists(fn))

## load data 
Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_base.csv" , sep='') , stringsAsFactors = F))
Xtrain = Xtrain[,c("VE_NUMBER","REN_ID","REN_DATE_EFF_FROM")]
Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_base.csv" , sep='') , stringsAsFactors = F))
Xtest = Xtest[,c("VE_NUMBER","REN_ID","REN_DATE_EFF_FROM")]
Ytrain = as.data.frame( fread(paste(ff.getPath("elab") , "Ytrain_base.csv" , sep='') , stringsAsFactors = F))
Ytrain = Ytrain$Ytrain
Xtrain = cbind(Xtrain,Ytrain=Ytrain)

submission = as.data.frame( fread( fn , stringsAsFactors = F))

#Xtest = merge(x = Xtest , y = submission , by="REN_ID" , all =  F )

## 
cat(">>> there are ",sum(unique(Xtest$VE_NUMBER) %in% unique(Xtrain$VE_NUMBER)),"VE_NUMBER in Xtest present also in Xtrain ... \n" )

VEs = unique(Xtest$VE_NUMBER) [unique(Xtest$VE_NUMBER) %in% unique(Xtrain$VE_NUMBER)]
Xtest = Xtest[Xtest$VE_NUMBER %in% VEs , ]

#Xtrain[Xtrain$VE_NUMBER==302,]
#Xtest[Xtest$VE_NUMBER==302,]

a = lapply( 1:nrow(Xtest) , function(i) {
#for (i in 1:nrow(Xtest)) {
  if (i == 1) cat(">>> starting ... [ 1 /",nrow(Xtest),"] ...\n")
  ven = Xtest[i,]$VE_NUMBER 
  dte = Xtest[i,]$REN_DATE_EFF_FROM 
  rid = Xtest[i,]$REN_ID 
  tr = Xtrain[Xtrain$VE_NUMBER == ven ,]
  dists = abs(tr$REN_DATE_EFF_FROM-dte)
  mdist = min(dists)
  if (mdist<CUTOFF) {
    pred_decor = tr[which(dists==mdist),"Ytrain"]
    if (length(pred_decor)>1) pred_decor =pred_decor[length(pred_decor)]
    submission[submission$REN_ID==rid,"REN_BASE_RENT"] <<- pred_decor  
  } 
  if (i %% 1000 == 0) cat(">>> [ ",i," /",nrow(Xtest),"] ...\n")
#}
})

cat(">>> writing on disk ... \n")
write.csv(submission,
          quote=FALSE, 
          file= fn.dec ,
          row.names=FALSE)