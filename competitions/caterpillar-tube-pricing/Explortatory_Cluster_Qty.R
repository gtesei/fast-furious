require(xgboost)
require(methods)
library(data.table)
library(plyr)
library(Hmisc)

library(lattice)
require(gridExtra) 
library(fastfurious)
library(parallel)

### FUNCS 
cluster_by = function(predictor.train,predictor.test,num_bids = 8,verbose=T) {
  require(Hmisc)
  
  ## clustering by quantity 
  if (verbose) {
    print(describe(predictor.train))
    print(describe(predictor.test))
  }
  
  data = as.vector(c(predictor.train,predictor.test))
  q = as.numeric(quantile(data, probs = ((1:num_bids)/num_bids)))
  
  ## counting cluster card 
  num=rep(0,num_bids)
  for (i in 1:num_bids)
    if (i == 1) {
      num[i] = sum(data<=q[i])
    } else {
      num[i] = sum(data<=q[i] & data>q[i-1])
    }
  if (verbose) print(describe(num))
  
  ## mapping quantity to cluster qty 
  qty2lev = data.frame(qty = sort(unique(data)) , lev = NA)
  for (i in 1:nrow(qty2lev)) {
    for (k in 1:length(q)) {
      if (k == 1) {
        if (qty2lev[i,]$qty <= q[1])  {
          qty2lev[i,]$lev = 1
          break
        } 
      } else {
        if (qty2lev[i,]$qty <= q[k] & qty2lev[i,]$qty > q[k-1] )  {
          qty2lev[i,]$lev = k
          break
        } 
      }
    }
  }
  
  ## mapping qty_lev on data 
  if (verbose) cat(">> mapping qty_lev to data ... \n")
  tr_qty_lev = rep(NA,length(predictor.train))
  for (i in 1:length(predictor.train))
    tr_qty_lev[i] = qty2lev[qty2lev$qty==predictor.train[i],]$lev
  
  ts_qty_lev = rep(NA,length(predictor.test))
  for (i in 1:length(predictor.test))
    ts_qty_lev[i] = qty2lev[qty2lev$qty==predictor.test[i],]$lev
  
  return( list(levels.train = tr_qty_lev , levels.test = ts_qty_lev , theresolds = q) )
}

################# FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/caterpillar-tube-pricing/competition_data')
ff.bindPath(type = 'process' , sub_path = 'data_process')
ff.bindPath(type = 'elab' , sub_path = 'dataset/caterpillar-tube-pricing/elab')
ff.bindPath(type = 'docs' , sub_path = 'dataset/caterpillar-tube-pricing/docs')
ff.bindPath(type = 'submission' , sub_path = 'dataset/caterpillar-tube-pricing')
ff.bindPath(type = 'submission_old' , sub_path = 'dataset/caterpillar-tube-pricing/submission')

source(paste0( ff.getPath("process") , "/FeatureSelection_Lib.R"))

################# SETTINGS
DEBUG_MODE = F

################# CLUSTER 
cluster_card = c(1,2,4,8)

################# DATA IN 

## elab 
train_enc = as.data.frame( fread(paste(ff.getPath("elab") , 
                                       "train_enc.csv" , sep=''))) 

test_enc = as.data.frame( fread(paste(ff.getPath("elab") , 
                                      "test_enc.csv" , sep=''))) 

train_enc_date = as.data.frame( fread(paste(ff.getPath("elab") , 
                                            "train_enc_date.csv" , sep=''))) 

test_enc_date = as.data.frame( fread(paste(ff.getPath("elab") , 
                                           "test_enc_date.csv" , sep=''))) 

## tech props 
tube_base = as.data.frame( fread(paste(ff.getPath("elab") , 
                                  "tube_base.csv" , sep='')))

bom_base = as.data.frame( fread(paste(ff.getPath("elab") , 
                                       "bom_base.csv" , sep='')))

spec_enc = as.data.frame( fread(paste(ff.getPath("elab") , 
                                       "spec_enc.csv" , sep='')))


####>>>>>>>>>> PROCESSING 
## build technical feature set 
tube = cbind(tube_base,bom_base)
tube = cbind(tube,spec_enc)
dim(tube) ## 180 (encoded) technical features  
# [1] 21198   180

## putting quote_date in data set  
head_train_set = train_enc_date
head_test_set = test_enc_date

## build train_set and test_set 
train_set = merge(x = head_train_set , y = tube , by = 'tube_assembly_id' , all = F)
test_set = merge(x = head_test_set , y = tube , by = 'tube_assembly_id' , all = F)

######### feature scaling 
cat(">>> Feature scaling ... \n")
feature2scal = c(
  "quote_date"     ,    "annual_usage"   ,     "min_order_quantity"    ,       
  "diameter"       ,      "wall"         ,      "length"               , "num_bends"      ,     "bend_radius"     ,    
  "num_boss"       ,      "num_bracket"  ,      
  "CP_001_weight"  ,     "CP_002_weight"  ,     "CP_003_weight"  ,     "CP_004_weight"  ,    "CP_005_weight"   ,    "CP_006_weight"    ,  
  "CP_007_weight"  ,     "CP_008_weight"  ,     "CP_009_weight"  ,    "CP_010_weight"   ,    "CP_011_weight"   ,    "CP_012_weight"    ,  
  "CP_014_weight"  ,     "CP_015_weight"  ,     "CP_016_weight"  ,     "CP_017_weight"  ,     "CP_018_weight"  ,     "CP_019_weight"   ,   
  "CP_020_weight"  ,     "CP_021_weight"  ,     "CP_022_weight"  ,     "CP_023_weight"  ,     "CP_024_weight"  ,     "CP_025_weight"   ,   
  "CP_026_weight"  ,     "CP_027_weight"  ,      "CP_028_weight" ,      "CP_029_weight" ,      "OTHER_weight"  
)

trans.scal <- preProcess(rbind(train_set[,feature2scal],test_set[,feature2scal]),
                         method = c("center", "scale") )

print(trans.scal)

train_set[,feature2scal] = predict(trans.scal,train_set[,feature2scal])
test_set[,feature2scal] = predict(trans.scal,test_set[,feature2scal])

######### 

######### 
ccor = function(df,y,mth) {
  mean(abs(unlist(lapply(df , function(x) {
    cor(x,y,method = mth)
  }))))
}
methods = c('pearson','spearman')

grid = expand.grid(cluster_card = cluster_card , corr = NA)

grid = cbind(grid , 
             setNames(as.data.frame(matrix(rep(NA,length(methods)*length(cluster_card)),
                                           ncol = length(methods))),paste0(methods,'_avg')))

grid = cbind(grid , 
             setNames(as.data.frame(matrix(rep(NA,length(methods)*max(cluster_card)*length(cluster_card)),
                                           ncol = max(cluster_card)*length(methods))),
                      c(outer(1:max(cluster_card),methods , FUN = 'paste' , sep='_'))))

grid['corr'] = NULL

for (cc in seq_along(cluster_card)) {
  cat(">>> CLUSTERING ",cluster_card[cc],"levels ...")
  
  ## clustering 
  cluster_levs = 1:cluster_card[cc]
  if (cluster_card[cc] > 1) {
    cls = cluster_by(predictor.train=train_set$quantity,
                     predictor.test=test_set$quantity,
                     num_bids = length(cluster_levs),
                     verbose=T)
    train_set$qty_lev = cls$levels.train
  } else {
    train_set$qty_lev = 1
  }
  
  ## compute the coefficient for each cluaster 
  obs = rep(NA,length(cluster_levs))
  for (cls in seq_along(cluster_levs)) {
    train_set_cl = train_set[train_set$qty_lev == cls,]
    cat('>>> train observations:',nrow(train_set_cl), ' \n')
    obs[cls] = nrow(train_set_cl)
    ##############
    ## DATA PROC 
    ##############
    
    ## tube_assembly_id , id 
    train_set_cl[, 'tube_assembly_id'] = NULL     
    train_set_cl[, 'material_id'] = NULL 
    
    ## y, data 
    y = train_set_cl$cost   
    train_set_cl[, 'cost'] = NULL 
    
    ####### remove zero variance predictors   
    l = featureSelect (train_set_cl,
                       NULL,
                       removeOnlyZeroVariacePredictors=F,
                       performVarianceAnalysisOnTrainSetOnly = T , 
                       removePredictorsMakingIllConditionedSquareMatrix = F, 
                       removeHighCorrelatedPredictors = F, 
                       featureScaling = F)
    train_set_cl = l$traindata
    ####### 
    corrs = setNames(unlist(lapply(methods,ccor,df = train_set_cl , y = y)),methods)
    for (mm in seq_along(methods)) {
      grid[grid$cluster_card == length(cluster_levs) , paste(cls,'_',methods[mm],sep='') ] = corrs[methods[mm]]
    }
  }
  stopifnot(sum(obs) == nrow(train_set))
  for (mm in seq_along(methods)) {
    grid[grid$cluster_card == length(cluster_levs) , paste0(methods[mm],'_avg') ] = 
       obs %*% t(as.matrix(grid[grid$cluster_card == length(cluster_levs) , paste(cluster_levs,'_',methods[mm],sep='') ])) / sum(obs)
  }
}

### write grid on disk 
write.csv(grid,quote=FALSE, 
          file=paste(ff.getPath("submission"),'grid_qty_corr.csv',sep='') ,
          row.names=FALSE)
