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
  
  data = as.vector(c(predictor.train,predictor.test))
  
  if (num_bids>8) {
    num_bids = 20
    split_16 = as.numeric(cut2(data, g=num_bids))
    
    return( list(levels.train = split_16[1:length(predictor.train)] , levels.test = split_16[(length(predictor.train)+1):length(data)] , 
                 theresolds = NULL) ) 
  } else {
    ## clustering by quantity 
    if (verbose) {
      print(describe(predictor.train))
      print(describe(predictor.test))
    }
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
}
getData = function() {
  sample_submission = as.data.frame( fread(paste(ff.getPath("data") , 
                                                 "sample_submission.csv" , sep=''))) 
  
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
  
  return(list(train_set=train_set,test_set=test_set,sample_submission=sample_submission))
}

buildGrid = function(ensemble = "ensemble_1" , 
                     best_tune = 'best_tune_1' , 
                     best_solution_outdir='pred_ensemble_1' ,
                     best_solution_fn = 'best_greedy_8_clusters.csv' ) {
  
  info = 11+1
  ## ensemble_1
  cat(">>>> ",ensemble,"...\n")
  
  ## grid 
  mf = list.files( ff.getPath(ensemble) )
  
  ## discard previous best_solution_fn 
  toDiscardIdx = which(mf == best_solution_fn)
  if (length(toDiscardIdx) > 0) {
    mf = mf[-toDiscardIdx]
  }
  
  ## 
  models = length(mf) + 1 
  grid = as.data.frame(matrix(rep(0,models*info), ncol = info))
  grid[,1] = c(mf,best_solution_fn) 
  colnames(grid) = c('ensemble','rmsle','predictions_closest','predictions_closest_perc', 
                     paste0('rmsle_c',1:n_clusters))
  
  ## pred_train
  pred_train = as.data.frame(matrix(rep(NA,(length(mf))*nrow(train_set)), ncol = (length(mf))))
  colnames(pred_train) = c(mf) 
  for (i in seq_along(mf)) {
    pred_i = as.data.frame( fread(paste(ff.getPath(ensemble) , mf[i] , sep='')))$assemble
    pred_i = pred_i[1:nrow(train_set)]
    pred_i = ifelse(pred_i < 0 , 1.5 , pred_i)
    pred_train[,mf[i]] = pred_i
    
    ## >>>> rmsle 
    grid[grid$ensemble == mf[i] , 'rmsle'] = 
      sqrt( sum( (log( pred_i +1) - log(train_set$cost+1))^2 ) / length(pred_i) )
    
    ## compute RMSLE for each cluster 
    for (cl in 1:n_clusters) {
      idx = which(train_set$qty_lev == cl)
      pred_i_cl = pred_i[idx]
      y_cl = train_set[idx,]$cost
      grid[grid$ensemble == mf[i] , paste0('rmsle_c',cl)] = 
        sqrt( sum( (log(pred_i_cl+1) - log(y_cl+1))^2 ) / length(pred_i_cl))
    }
  }  
  
  ## best ensemble of each clusters 
  grid[grid$ensemble == best_solution_fn , paste0('rmsle_c',1:n_clusters)] = unlist(lapply( 
    grid[grid$ensemble != best_solution_fn , paste0('rmsle_c',1:n_clusters) ] , function(x) {
      min(x)
    }))
  
  best_idx = unlist(lapply( 
    grid[grid$ensemble != best_solution_fn , paste0('rmsle_c',1:n_clusters) ] , function(x) {
      m = which(x == min(x))
      if (length(m)>1) m = m[1]
      return(m)
    }))
  best_ensembles_name = grid[grid$ensemble != best_solution_fn , 'ensemble'] [best_idx]
  best_ensembles = data.frame(cluster = 1:n_clusters , ensemble = best_ensembles_name)
  
  pred_train_best = rep(NA,nrow(train_set))
  for (x in seq_along(best_idx)) {
    idx = which(train_set$qty_lev == x)
    pred_train_best[idx] = pred_train[idx , best_ensembles_name[x] ]
  }
  stopifnot(sum(is.na(pred_train_best))==0)
  
  pred_train[,best_solution_fn] = pred_train_best
  
  # best_rmsle <<<< NOTICE it is not the weigthed mean of the best ensemble of each cluster 
  best_rmsle = sqrt(    sum( (log(pred_train_best+1) - log(train_set$cost+1))^2 )   /length(pred_train_best))
  grid[grid$ensemble == best_solution_fn , 'rmsle'] = best_rmsle
  
  ## order grid by rmsle
  grid = grid[order(grid$rmsle , decreasing = F),]
  
  ## writing on disk best solution 
  if (! is.null(best_solution_outdir)) {
    cat(">> writing ",best_solution_fn,'on',ff.getPath(best_solution_outdir),'...\n')
    
    ## pred_test
    pred_test = as.data.frame(matrix(rep(NA,(length(mf))*nrow(test_set)), ncol = (length(mf))))
    colnames(pred_test) = c(mf) 
    for (i in seq_along(mf)) {
      pred_i = as.data.frame( fread(paste(ff.getPath(ensemble) , mf[i] , sep='')))$assemble
      pred_i = pred_i[(nrow(train_set)+1):(nrow(train_set)+nrow(test_set))]
      pred_i = ifelse(pred_i < 0 , 1.5 , pred_i)
      pred_test[,mf[i]] = pred_i
    }
    
    # pred_test_best
    pred_test_best = rep(NA,nrow(test_set))
    for (x in seq_along(best_idx)) {
      idx = which(test_set$qty_lev == x)
      pred_test_best[idx] = pred_test[idx , best_ensembles_name[x] ]
    }
    stopifnot(sum(is.na(pred_test_best))==0)
    stopifnot(sum(pred_test_best<0)==0)
    
    # write on disk prediction 
    write.csv(data.frame(id = sample_submission$id , cost=pred_test_best),
              quote=FALSE, 
              file=paste(ff.getPath(best_solution_outdir), best_solution_fn ,sep='') ,
              row.names=FALSE)
    
    # write on disk ensemble 
    assemble = c(pred_train_best,pred_test_best)
    write.csv(data.frame(id = seq_along(assemble) , assemble=assemble),
              quote=FALSE, 
              file=paste(ff.getPath(ensemble), best_solution_fn ,sep='') ,
              row.names=FALSE)
  }
  
  ## >>> min distance to y 
  min_dist_train = as.data.frame(matrix( rep(NA,(ncol(pred_train))*nrow(train_set)), ncol = ncol(pred_train) ))
  colnames(min_dist_train) = colnames(pred_train)
  for (i in colnames(pred_train)) {
    min_dist_train[,i] = abs(train_set$cost - pred_train[,i])
  }

  close = apply(X = min_dist_train,MARGIN = 1,function(x) {
    colnames(min_dist_train)[which(x == min(x,na.rm = T))]
  })
  
  for (i in seq_along(close)) {
    for (j in 1:length(close[[i]])) {
      grid[grid$ensemble == close[[i]][j] , 'predictions_closest'] <- 
        (grid[grid$ensemble == close[[i]][j] , 'predictions_closest'] + 1)
    } 
  }
  
  grid$predictions_closest_perc = (grid$predictions_closest / nrow(train_set))
  
  mins = unlist(apply(X = min_dist_train,MARGIN = 1,function(x) {
    min(x,na.rm = T)
  }))
  
  return(list( grid=grid , min_avg=mean(mins) , best_ensembles=best_ensembles ))
}

################# FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/caterpillar-tube-pricing/competition_data')
ff.bindPath(type = 'elab' , sub_path = 'dataset/caterpillar-tube-pricing/elab')
ff.bindPath(type = 'docs' , sub_path = 'dataset/caterpillar-tube-pricing/docs')

ff.bindPath(type = 'best_tune_1' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/best_tune_1') ## in 
ff.bindPath(type = 'best_tune_2' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/best_tune_2') ## in 
ff.bindPath(type = 'best_tune_3' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/best_tune_3') ## in 
ff.bindPath(type = 'best_tune_4' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/best_tune_4') ## in 
#ff.bindPath(type = 'best_tune_5' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/best_tune_5') ## in 

ff.bindPath(type = 'ensemble_1' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/ensemble_1') ## in
ff.bindPath(type = 'ensemble_2' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/ensemble_2') ## in 
ff.bindPath(type = 'ensemble_3' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/ensemble_3') ## in 
ff.bindPath(type = 'ensemble_4' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/ensemble_4') ## in 
#ff.bindPath(type = 'ensemble_5' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/ensemble_5') ## in 

ff.bindPath(type = 'pred_ensemble_1' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/pred_ensemble_1') ## out
ff.bindPath(type = 'pred_ensemble_2' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/pred_ensemble_2') ## out
ff.bindPath(type = 'pred_ensemble_3' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/pred_ensemble_3') ## out
ff.bindPath(type = 'pred_ensemble_4' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/pred_ensemble_4') ## out
#ff.bindPath(type = 'pred_ensemble_5' , sub_path = 'dataset/caterpillar-tube-pricing/ensembles/pred_ensemble_5') ## out

################# SETTINGS

################# DATA IN 
l = getData()
train_set = l$train_set
test_set = l$test_set
sample_submission = l$sample_submission

############# PROCESSING 
#clustering 
n_clusters = 8 
cls = cluster_by(predictor.train=train_set[,'quantity'],
                 predictor.test=test_set[,'quantity'],
                 num_bids = n_clusters,
                 verbose=F)

train_set$qty_lev = cls$levels.train
test_set$qty_lev = cls$levels.test

####### build grid layer n. 1 
b1 = buildGrid(ensemble = "ensemble_1" , best_tune = 'best_tune_1' , 
               best_solution_outdir='pred_ensemble_1', 
               best_solution_fn = 'best_greedy_8_clusters_1.csv')
ggrid_1 = b1$grid
cat(">>> mean rmsle ensemble 1: ",mean(ggrid_1$rmsle , na.rm = T) , "\n")
cat(">>> mean min distances ensemble 1: ",b1$min_avg, "\n") 
cat(">>> best ensembles for each cluster layer N. 1 \n") 
print(b1$best_ensembles)
write.csv(ggrid_1,
          quote=FALSE, 
          file=paste(ff.getPath("docs"), 'redo_ensemble_diagnosys_1.csv' ,sep='') ,
          row.names=FALSE)


####### build grid layer n. 2
b2 = buildGrid(ensemble = "ensemble_2" , best_tune = 'best_tune_2' , 
               best_solution_outdir='pred_ensemble_2' , 
               best_solution_fn = 'best_greedy_8_clusters_2.csv')
ggrid_2 = b2$grid
cat(">>> mean rmsle ensemble 2: ",mean(ggrid_2$rmsle , na.rm = T) , "\n") 
cat(">>> mean min distances ensemble 2: ",b2$min_avg, "\n") 
cat(">>> best ensembles for each cluster layer N. 2 \n") 
print(b2$best_ensembles)
write.csv(ggrid_2,
           quote=FALSE, 
           file=paste(ff.getPath("docs"), 'redo_ensemble_diagnosys_2.csv' ,sep='') ,
           row.names=FALSE)

####### build grid layer n. 3
b3 = buildGrid(ensemble = "ensemble_3" , best_tune = 'best_tune_3' , 
               best_solution_outdir='pred_ensemble_3' , 
               best_solution_fn = 'best_greedy_8_clusters_3.csv')
ggrid_3 = b3$grid
cat(">>> mean rmsle ensemble 3: ",mean(ggrid_3$rmsle , na.rm = T) , "\n") 
cat(">>> mean min distances ensemble 2: ",b3$min_avg, "\n") 
cat(">>> best ensembles for each cluster layer N. 3 \n") 
print(b3$best_ensembles)
write.csv(ggrid_3,
          quote=FALSE, 
          file=paste(ff.getPath("docs"), 'redo_ensemble_diagnosys_3.csv' ,sep='') ,
          row.names=FALSE)

####### build grid layer n. 4
b4 = buildGrid(ensemble = "ensemble_4" , best_tune = 'best_tune_4' , 
               best_solution_outdir='pred_ensemble_4' , 
               best_solution_fn = 'best_greedy_8_clusters_4.csv')
ggrid_4 = b4$grid
cat(">>> mean rmsle ensemble 4: ",mean(ggrid_4$rmsle , na.rm = T) , "\n") 
cat(">>> mean min distances ensemble 4: ",b4$min_avg, "\n") 
cat(">>> best ensembles for each cluster layer N. 4 \n") 
print(b4$best_ensembles)
write.csv(ggrid_4,
          quote=FALSE, 
          file=paste(ff.getPath("docs"), 'redo_ensemble_diagnosys_4.csv' ,sep='') ,
          row.names=FALSE)
