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
getData = function() {
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
  
  return(list(train_set=train_set,test_set=test_set))
}

buildGrid = function(ensemble = "ensemble_1" , best_tune = 'best_tune_1') {
  
  info = 4+1
  ## ensemble_1
  cat(">>>> ",ensemble,"...\n")
  mf = list.files( ff.getPath(ensemble) )
  grid = as.data.frame(matrix(rep(0,length(mf)*info), ncol = info))
  grid[,1] = mf 
  colnames(grid) = c('ensemble','rmse','rmsle','predictions_closest','predictions_closest_perc')
  
  pred_train = as.data.frame(matrix(rep(NA,(length(mf))*nrow(train_set)), ncol = (length(mf))))
  colnames(pred_train) = c(mf) 
  
  for (i in seq_along(mf)) {
    idxtg = grep(pattern = 'treebag' , x = mf[i] , value = F)
    if (length(idxtg) == 0) {
      tuneGrid = as.data.frame( fread(paste(ff.getPath(best_tune) , mf[i] , sep=''))) 
      grid[grid$ensemble==mf[i],'rmse'] = mean(tuneGrid$rmse,na.rm = T)
    }
    pred_i = as.data.frame( fread(paste(ff.getPath(ensemble) , mf[i] , sep='')))$assemble
    pred_i = pred_i[1:nrow(train_set)]
    pred_train[,mf[i]] = abs(train_set$cost - pred_i)
  }
  
  close = apply(X = pred_train,MARGIN = 1,function(x) {
    colnames(pred_train)[which(x == min(x,na.rm = T))]
  })
  
  mins = unlist(apply(X = pred_train,MARGIN = 1,function(x) {
    min(x,na.rm = T)
  }))
  
  for (i in seq_along(close)) {
    for (j in 1:length(close[[i]])) {
      grid[grid$ensemble == close[[i]][j] , 'predictions_closest'] <- 
        (grid[grid$ensemble == close[[i]][j] , 'predictions_closest'] + 1)
    } 
  }
  
  grid$predictions_closest_perc = (grid$predictions_closest / nrow(train_set))
  
  grid$rmse = ifelse(grid$rmse == 0 , NA , grid$rmse)
  for (i in seq_along(mf)) {
    pred_i = as.data.frame( fread(paste(ff.getPath(ensemble) , mf[i] , sep='')))$assemble
    pred_i = pred_i[1:nrow(train_set)]
    pred_i = ifelse(pred_i < 0 , 1.5 , pred_i)
    grid[grid$ensemble == mf[i] , 'rmsle'] = 
      sqrt(    sum( (log(pred_i+1) - log(train_set$cost+1))^2 )   /nrow(train_set))
  }  
  
  grid = grid[order(grid$rmsle , decreasing = F),]
  
  return(list(grid=grid,min_avg=mean(mins)))
}

################# FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/caterpillar-tube-pricing/competition_data')
ff.bindPath(type = 'elab' , sub_path = 'dataset/caterpillar-tube-pricing/elab')
ff.bindPath(type = 'docs' , sub_path = 'dataset/caterpillar-tube-pricing/docs')
ff.bindPath(type = 'best_tune_1' , sub_path = 'dataset/caterpillar-tube-pricing/best_tune_1') ## in 
ff.bindPath(type = 'best_tune_2' , sub_path = 'dataset/caterpillar-tube-pricing/best_tune_2') ## in 
ff.bindPath(type = 'best_tune_3' , sub_path = 'dataset/caterpillar-tube-pricing/best_tune_3') ## in 
ff.bindPath(type = 'ensemble_1' , sub_path = 'dataset/caterpillar-tube-pricing/ensemble_1') ## in
ff.bindPath(type = 'ensemble_2' , sub_path = 'dataset/caterpillar-tube-pricing/ensemble_2') ## in 
ff.bindPath(type = 'ensemble_3' , sub_path = 'dataset/caterpillar-tube-pricing/ensemble_3') ## in 


################# SETTINGS

################# DATA IN 
l = getData()
train_set = l$train_set
test_set = l$test_set

################# PROCESSING 
b1 = buildGrid(ensemble = "ensemble_1" , best_tune = 'best_tune_1')
b2 = buildGrid(ensemble = "ensemble_2" , best_tune = 'best_tune_2')
b3 = buildGrid(ensemble = "ensemble_3" , best_tune = 'best_tune_3')

grid_1 = b1$grid
grid_2 = b2$grid
grid_3 = b3$grid

write.csv(grid_1,
          quote=FALSE, 
          file=paste(ff.getPath("docs"), 'ensemble_diagnosys_1.csv' ,sep='') ,
          row.names=FALSE)

write.csv(grid_2,
          quote=FALSE, 
          file=paste(ff.getPath("docs"), 'ensemble_diagnosys_2.csv' ,sep='') ,
          row.names=FALSE)

write.csv(grid_3,
          quote=FALSE, 
          file=paste(ff.getPath("docs"), 'ensemble_diagnosys_3.csv' ,sep='') ,
          row.names=FALSE)

### Stats
cat(">>> mean rmsle ensemble 1: ",mean(grid_1$rmsle , na.rm = T) , "\n")
cat(">>> mean rmsle ensemble 2: ",mean(grid_2$rmsle , na.rm = T) , "\n") 
cat(">>> mean rmsle ensemble 3: ",mean(grid_3$rmsle , na.rm = T) , "\n") 

cat(">>> mean min distances ensemble 1: ",b1$min_avg, "\n") 
cat(">>> mean min distances ensemble 2: ",b2$min_avg, "\n") 
cat(">>> mean min distances ensemble 3: ",b3$min_avg, "\n") 




