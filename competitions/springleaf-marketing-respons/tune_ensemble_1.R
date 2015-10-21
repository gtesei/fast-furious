library(data.table)
library(xgboost)
library(fastfurious)

### FUNCS

get_data_base = function () {
  cat("reading the train and test data\n")
  
  train = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
  test = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
  
  train = train[,-c(1,208,214,839,846,1427)]
  test_id = test$ID
  test = test[,-c(1,208,214,839,846,1427)]
  
  #####
  ## VAR_0246 VAR_0530 hanno solo -1 e NAs 
  ## ci sono inoltre ~50 predittori uguali 
  ####
  
  feature.names <- names(train)[2:ncol(train)-1]
  
  cat("assuming text variables are categorical & replacing them with numeric ids\n")
  for (f in feature.names) {
    if (class(train[[f]])=="character") {
      cat(">>> ",f," is character \n")
      levels <- unique(c(train[[f]], test[[f]]))
      train[[f]] <- as.integer(factor(train[[f]], levels=levels))
      test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
    }
  }
  
  cat("replacing missing values with -1.5\n")
  train[is.na(train)] <- -1.5
  test[is.na(test)]   <- -1.5
  
  return(list(
    Ytrain = train$target ,
    Xtrain = train[,feature.names] , 
    Xtest = test[,feature.names],  
    test_id = test_id
  ))
}

get_data_base_poly3_cut1000 = function () {
  
  db = get_data_base()
  
  ##
  data = rbind(db$Xtrain,db$Xtest)
  data.poly = ff.poly(x = data , n = 3)
  Xtrain = data.poly[1:nrow(train),]
  Xtest = data.poly[(nrow(train)+1):nrow(data),]
  
  l = ff.corrFilter(Xtrain=Xtrain,Xtest=Xtest,y=db$Ytrain,abs_th = 1000 , method = "spearman")
  Xtrain = l$Xtrain
  Xtest =l$Xtest
  
  return(list(
    Ytrain = db$Ytrain ,
    Xtrain = Xtrain , 
    Xtest = Xtest,  
    test_id = db$test_id
  ))
}

buildIDModelList = function(list) {
  stopifnot(length(list)>0)
  for (i in 1:length(list)) {
    fn = ""
    if (DEBUG) fn = "DEBUG_"
    for (j in 1:length(list[[i]])) {
      fn = paste(fn,names(list[[i]][j]),list[[i]][j],sep="")
      if (j < length(list[[i]])) {
        fn = paste(fn,"_",sep="")
      }
    }
    fn = paste0(fn,".csv")
    list[[i]]$id = fn 
  }
  return(list)
}

getData = function(mod) {
  Ytrain = NULL
  Xtrain = NULL
  Xtest = NULL
  test_id = NULL
  
  dataProc = mod$dataProc
  if (identical("base",dataProc)) {
    cat(">>> dataProc base ...\n")
    l = get_data_base()
    Ytrain = l$Ytrain
    Xtrain = l$Xtrain
    Xtest = l$Xtest 
    test_id = l$test_id 
    rm(l)
    gc()
  } else if (identical("base_poly3_cut1000",dataProc)) {
    l = get_data_base_poly3_cut1000()
    Ytrain = l$Ytrain
    Xtrain = l$Xtrain
    Xtest = l$Xtest 
    test_id = l$test_id 
    rm(l)
    gc()
  } else if (identical("doc_proc_2",dataProc)) {
    
    ## raw data 
    train_raw = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
    test_raw = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
    
    test_id = test_raw$ID 
    Ytrain = train_raw$target
    
    rm(list=c("train_raw","test_raw"))
    gc()
    
    ## elab data 
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_docproc2.csv" , sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_docproc2.csv" , sep='') , stringsAsFactors = F))
    
  } else if (identical("doc_proc_2_uc",dataProc)) {
    
    ## raw data 
    train_raw = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
    test_raw = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
    
    test_id = test_raw$ID 
    Ytrain = train_raw$target
    
    rm(list=c("train_raw","test_raw"))
    gc()
    
    ## elab data 
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_docproc2.csv" , sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_docproc2.csv" , sep='') , stringsAsFactors = F))
    
    ## 
    idx1 = which(Ytrain == 1)
    idx0 = which(Ytrain == 0)
    
    inr = length(idx1) / length(idx0)
    
    l0 = 1.5*length(idx1) ## ratio = 0.6666667 (instead of 0.3333333)
    
    idx0_new = sample(x = idx0 , size = l0 , replace = F)
    
    stopifnot(length(idx0_new)==length(unique(idx0_new)))
    
    Xtrain = Xtrain[c(idx1,idx0_new),]
    Ytrain = Ytrain[c(idx1,idx0_new)]
    
  } else if (identical("doc_proc_3_uc",dataProc)) {
    
    ## raw data 
    train_raw = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
    test_raw = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
    
    test_id = test_raw$ID 
    Ytrain = train_raw$target
    
    rm(list=c("train_raw","test_raw"))
    gc()
    
    ## elab data 
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_docproc3_clean.csv" , sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_docproc3_clean.csv" , sep='') , stringsAsFactors = F))
    
    ## 
    idx1 = which(Ytrain == 1)
    idx0 = which(Ytrain == 0)
    
    inr = length(idx1) / length(idx0)
    
    l0 = 1.5*length(idx1) ## ratio = 0.6666667 (instead of 0.3333333)
    
    idx0_new = sample(x = idx0 , size = l0 , replace = F)
    
    stopifnot(length(idx0_new)==length(unique(idx0_new)))
    
    Xtrain = Xtrain[c(idx1,idx0_new),]
    Ytrain = Ytrain[c(idx1,idx0_new)]
    
  } else if (identical("doc_proc_3",dataProc)) {
    
    ## raw data 
    train_raw = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
    test_raw = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
    
    test_id = test_raw$ID 
    Ytrain = train_raw$target
    
    rm(list=c("train_raw","test_raw"))
    gc()
    
    ## elab data 
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_docproc3_clean.csv" , sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_docproc3_clean.csv" , sep='') , stringsAsFactors = F))
    
  } else if (identical("CBDB",dataProc)) {
    ## raw data 
    train_raw = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
    test_raw = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
    
    test_id = test_raw$ID 
    Ytrain = train_raw$target
    
    rm(list=c("train_raw","test_raw"))
    gc()
    
    ## CBDB
    ff.bindPath(type = 'CBDB' , sub_path = 'dataset/springleaf-marketing-respons/elab/CBDB') 
    n.ds = mod$CBDB_num
    Xtrain = as.data.frame( fread(paste(ff.getPath("CBDB"),"Xtrain_",n.ds,".csv",sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("CBDB"),"Xtest_",n.ds,".csv",sep='') , stringsAsFactors = F))
    
  } else if (identical("poly2cut1000",dataProc)) {
    ## raw data 
    train_raw = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
    test_raw = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
    
    test_id = test_raw$ID 
    Ytrain = train_raw$target
    
    rm(list=c("train_raw","test_raw"))
    gc()
    
    ## Xtrain_docproc2_poly2cut1000.csv / Xtest_docproc2_poly2cut1000.csv
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab"),"Xtrain_docproc2_poly2cut1000.csv",sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab"),"Xtest_docproc2_poly2cut1000.csv",sep='') , stringsAsFactors = F))
    
    ## NAs 
    cat(">>> checking that predictors with NAs values are date predictors ... \n")
    feature.names <- colnames(Xtrain)
    predNA = unlist(lapply(1:length(feature.names) , function(i) {
      f = feature.names[i]
      sum(is.na(Xtrain[[f]]))>0 || sum(is.na(Xtest[[f]]))>0
    }))
    predNAIdx = which(predNA)
    if (length(predNAIdx)>0) {
      cat(">>> found ",length(predNAIdx),"predictors with NAs values:",feature.names[predNAIdx],"--> removing ...\n") 
      
      Xtrain = Xtrain[,-predNAIdx]
      Xtest = Xtest[,-predNAIdx]
    }
    
  } else if (identical("pca95var",dataProc)) {
    ## raw data 
    train_raw = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
    test_raw = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
    
    test_id = test_raw$ID 
    Ytrain = train_raw$target
    
    rm(list=c("train_raw","test_raw"))
    gc()
    
    ## Xtrain_docproc2_pca_95var.csv / Xtest_docproc2_pca_95var.csv
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab"),"Xtrain_docproc2_pca_95var.csv",sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab"),"Xtest_docproc2_pca_95var.csv",sep='') , stringsAsFactors = F))
    
    ## NAs 
    cat(">>> checking that predictors with NAs values are date predictors ... \n")
    feature.names <- colnames(Xtrain)
    predNA = unlist(lapply(1:length(feature.names) , function(i) {
      f = feature.names[i]
      sum(is.na(Xtrain[[f]]))>0 || sum(is.na(Xtest[[f]]))>0
    }))
    predNAIdx = which(predNA)
    if (length(predNAIdx)>0) {
      cat(">>> found ",length(predNAIdx),"predictors with NAs values:",feature.names[predNAIdx],"--> removing ...\n") 
      
      Xtrain = Xtrain[,-predNAIdx]
      Xtest = Xtest[,-predNAIdx]
    }
    
  } else if (identical("pcaElbow",dataProc)) {
    ## raw data 
    train_raw = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
    test_raw = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
    
    test_id = test_raw$ID 
    Ytrain = train_raw$target
    
    rm(list=c("train_raw","test_raw"))
    gc()
    
    ## Xtrain_docproc2_pca_elbow.csv / Xtest_docproc2_pca_elbow.csv
    Xtrain = as.data.frame( fread(paste(ff.getPath("elab"),"Xtrain_docproc2_pca_elbow.csv",sep='') , stringsAsFactors = F))
    Xtest = as.data.frame( fread(paste(ff.getPath("elab"),"Xtest_docproc2_pca_elbow.csv",sep='') , stringsAsFactors = F))
    
    ## NAs 
    cat(">>> checking that predictors with NAs values are date predictors ... \n")
    feature.names <- colnames(Xtrain)
    predNA = unlist(lapply(1:length(feature.names) , function(i) {
      f = feature.names[i]
      sum(is.na(Xtrain[[f]]))>0 || sum(is.na(Xtest[[f]]))>0
    }))
    predNAIdx = which(predNA)
    if (length(predNAIdx)>0) {
      cat(">>> found ",length(predNAIdx),"predictors with NAs values:",feature.names[predNAIdx],"--> removing ...\n") 
      
      Xtrain = Xtrain[,-predNAIdx]
      Xtest = Xtest[,-predNAIdx]
    }
    
  } else if (identical("default",dataProc) && mod$layer == 2) {
    prev_layer = mod$layer -1 
    
    ## raw data 
    train_raw = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
    test_raw = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
    
    test_id = test_raw$ID 
    Ytrain = train_raw$target
    
    rm(list=c("train_raw","test_raw"))
    gc()
    
    ##
    ff.bindPath(type = 'ensembles' , sub_path = 'dataset/springleaf-marketing-respons/ensembles/') 
    
    ## ens_dirs
    kword = paste0('ensemble_',prev_layer)
    ens_dirs = list.files( ff.getPath('ensembles') )
    ens_dirs = ens_dirs[which(substr(x = ens_dirs, start = 1 , stop = nchar(kword)) == kword)]
    cat(">>> Found ",length(ens_dirs),"ensemble directory:",ens_dirs,"\n")
    stopifnot(length(ens_dirs)==1)
    
    ## ens_dir
    ensembles_scores = NULL
    Xtrain = NULL
    Xtest = NULL
    
    ##
    ens_dir = paste0('ensemble_',prev_layer)
    stopifnot(ens_dir %in% ens_dirs) 
    ensembles_i = list.files( paste0(ff.getPath('ensembles') , ens_dir) )
    cat(">>> processing ",ens_dir," --> found ",length(ensembles_i),"ensembles...\n")
    ensembles_scores = data.frame(ID = ensembles_i , layer = rep(prev_layer,length(ensembles_i)) , AUC=NA) 
    Xtrain = data.frame(matrix(rep(NA,length(Ytrain)*length(ensembles_i)),ncol=length(ensembles_i),nrow=length(Ytrain)))
    Xtest = data.frame(matrix(rep(NA,length(test_id)*length(ensembles_i)),ncol=length(ensembles_i),nrow=length(test_id)))
    colnames(Xtrain) = ensembles_i
    colnames(Xtest) = ensembles_i
    
    for (j in ensembles_i) {
      sub_j = as.data.frame( fread( paste(ff.getPath('ensembles') , ens_dir, .Platform$file.sep,j, sep='') , stringsAsFactors = F))
      predTrain = sub_j[1:length(Ytrain),'assemble']
      predTest = sub_j[(length(Ytrain)+1):nrow(sub_j),'assemble']
      trIdx = which(colnames(Xtrain) == j)
      Xtrain[,trIdx] = predTrain
      Xtest[,trIdx] = predTest
      roc_1 = verification::roc.area(Ytrain , predTrain )$A
      #l = fastfurious:::getCaretFactors(y=Ytrain)
      #roc_2 = as.numeric( pROC::auc(pROC::roc(response = l$y.cat, predictor = predTrain, levels = levels(l$y.cat) )))
      ensembles_scores[ensembles_scores$ID==j,'AUC'] <- roc_1
      ensembles_scores <- ensembles_scores[order(ensembles_scores$AUC,decreasing = T),]
    }
    
    ## apply here threshold
    if (!is.null(mod$threshold)) {
      cat(">>> cutting at ",mod$threshold,"...\n")
      takeIdx = which(ensembles_scores$AUC >= mod$threshold)
      Xtrain = Xtrain[,takeIdx,drop=F]
      Xtest = Xtest[,takeIdx,drop=F]
    }
    
    ## apply here k-mens 
    if (!is.null(mod$kmeans)) {
      cat(">>> adding k-means ",mod$kmeans,"...\n")
      if (mod$kmeans==1) {
        kmeans = as.data.frame( fread(paste(ff.getPath("elab") , "k-means_1.csv" , sep='') , stringsAsFactors = F))  
        Xtrain = cbind(Xtrain , 
                       cluster = kmeans[1:nrow(Xtrain),'cluster'])
        Xtest = cbind(Xtest , 
                      cluster = kmeans[(nrow(Xtrain)+1):nrow(kmeans),'cluster'])
      } else if (mod$kmeans==3) {
        kmeans = as.data.frame( fread(paste(ff.getPath("elab") , "k-means_3.csv" , sep='') , stringsAsFactors = F))  
        Xtrain = cbind(Xtrain , 
                       cluster = kmeans[1:nrow(Xtrain),'cluster'] , 
                       cluster_1 = kmeans[1:nrow(Xtrain),'cluster_1'] , 
                       cluster_3 = kmeans[1:nrow(Xtrain),'cluster_3'])
        Xtest = cbind(Xtest , 
                      cluster = kmeans[(nrow(Xtrain)+1):nrow(kmeans),'cluster'], 
                      cluster_1 = kmeans[(nrow(Xtrain)+1):nrow(kmeans),'cluster_1'],
                      cluster_3 = kmeans[(nrow(Xtrain)+1):nrow(kmeans),'cluster_3'])
      } else {
        stop(paste0("unrecognized kmeans:",mod$kmean))
      }
    }
    
    ## apply here error analysis 
    if (!is.null(mod$err_an)) { 
      cat(">>> adding error analysis predictors ...\n")
      
      Xtrain_l1 = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_docproc2.csv" , sep='') , stringsAsFactors = F))
      Xtest_l1 = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_docproc2.csv" , sep='') , stringsAsFactors = F))
      
      errAn = as.data.frame( fread(paste(ff.getPath("elab") , "ensembles_scores.csv" , sep='') , stringsAsFactors = F))  
      errAn = unique(errAn[,c("err_an.pred_name","err_an.pred_val")])
      
      ###
      tr_preds = data.frame(matrix(rep(NA,nrow(errAn)*nrow(Xtrain_l1)),ncol=nrow(errAn),nrow=nrow(Xtrain_l1)))
      te_preds = data.frame(matrix(rep(NA,nrow(errAn)*nrow(Xtest_l1)),ncol=nrow(errAn),nrow=nrow(Xtest_l1)))
      colnames(tr_preds) <- colnames(te_preds) <- paste(errAn$err_an.pred_name,"_",errAn$err_an.pred_val,sep='')
      
      for (i in 1:ncol(tr_preds)) {
        tr_preds[i] = as.numeric(Xtrain_l1[,as.character(errAn$err_an.pred_name[i])] == errAn$err_an.pred_val[i])
        te_preds[i] = as.numeric(Xtest_l1[,as.character(errAn$err_an.pred_name[i])] == errAn$err_an.pred_val[i])
      }
      
      stopifnot( sum(is.na(tr_preds))==0 , sum(is.na(te_preds))==0 )
      
      Xtrain = cbind(Xtrain , tr_preds) 
      Xtest = cbind(Xtest , te_preds)
    }
    
  } else if (identical("default",dataProc) && mod$layer == 3) { 
    prev_layer = mod$layer -1 
    
    ## raw data 
    train_raw = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
    test_raw = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
    
    test_id = test_raw$ID 
    Ytrain = train_raw$target
    
    rm(list=c("train_raw","test_raw"))
    gc()
    
    ##
    ff.bindPath(type = 'ensembles' , sub_path = 'dataset/springleaf-marketing-respons/ensembles/') 
    ens_dirs = list.files( ff.getPath('ensembles') )
    
    Xtrain = NULL
    Xtest = NULL 
    for (layer in 1:prev_layer) {
      ens_dir = paste0('ensemble_',layer)
      
      ##
      stopifnot(ens_dir %in% ens_dirs) 
      ensembles_i = list.files( paste0(ff.getPath('ensembles') , ens_dir) )
      cat(">>> processing ",ens_dir," --> found ",length(ensembles_i),"ensembles...\n")
      
      Xtrain_i = data.frame(matrix(rep(NA,length(Ytrain)*length(ensembles_i)),ncol=length(ensembles_i),nrow=length(Ytrain)))
      Xtest_i = data.frame(matrix(rep(NA,length(test_id)*length(ensembles_i)),ncol=length(ensembles_i),nrow=length(test_id)))
      colnames(Xtrain_i) = ensembles_i
      colnames(Xtest_i) = ensembles_i
      
      for (j in ensembles_i) {
        sub_j = as.data.frame( fread( paste(ff.getPath('ensembles') , ens_dir, .Platform$file.sep,j, sep='') , stringsAsFactors = F))
        predTrain = sub_j[1:length(Ytrain),'assemble']
        predTest = sub_j[(length(Ytrain)+1):nrow(sub_j),'assemble']
        trIdx = which(colnames(Xtrain_i) == j)
        Xtrain_i[,trIdx] = predTrain
        Xtest_i[,trIdx] = predTest
      }
      
      Xtrain <- if(is.null(Xtrain)) Xtrain_i else cbind(Xtrain,Xtrain_i)
      Xtest <- if(is.null(Xtest)) Xtest_i else cbind(Xtest,Xtest_i)
    }
    
  } else {
    stop(paste0("unrecognized type of dataProc:",dataProc))
  }
  
  cat(">>> loaded Ytrain:",length(Ytrain),"\n")
  cat(">>> loaded Xtrain:",dim(Xtrain),"\n")
  cat(">>> loaded Xtest:",dim(Xtest),"\n")
  cat(">>> loaded test_id:",length(test_id),"\n")
  
  return(list(
    Ytrain = Ytrain, 
    Xtrain = Xtrain, 
    Xtest = Xtest, 
    test_id = test_id
    ))
}

### CONFIG 
DEBUG = F

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/springleaf-marketing-respons')
ff.bindPath(type = 'code' , sub_path = 'competitions/springleaf-marketing-respons')
ff.bindPath(type = 'elab' , sub_path = 'dataset/springleaf-marketing-respons/elab') 

ff.bindPath(type = 'ensemble_1' , sub_path = 'dataset/springleaf-marketing-respons/ensembles/ensemble_1',createDir = T) ## out 
ff.bindPath(type = 'best_tune_1' , sub_path = 'dataset/springleaf-marketing-respons/ensembles/best_tune_1',createDir = T) ## out 
ff.bindPath(type = 'submission_1' , sub_path = 'dataset/springleaf-marketing-respons/ensembles/pred_ensemble_1',createDir = T) ## out 

ff.bindPath(type = 'ensemble_2' , sub_path = 'dataset/springleaf-marketing-respons/ensembles/ensemble_2',createDir = T) ## out 
ff.bindPath(type = 'best_tune_2' , sub_path = 'dataset/springleaf-marketing-respons/ensembles/best_tune_2',createDir = T) ## out 
ff.bindPath(type = 'submission_2' , sub_path = 'dataset/springleaf-marketing-respons/ensembles/pred_ensemble_2',createDir = T) ## out 

ff.bindPath(type = 'ensemble_3' , sub_path = 'dataset/springleaf-marketing-respons/ensembles/ensemble_3',createDir = T) ## out 
ff.bindPath(type = 'best_tune_3' , sub_path = 'dataset/springleaf-marketing-respons/ensembles/best_tune_3',createDir = T) ## out 
ff.bindPath(type = 'submission_3' , sub_path = 'dataset/springleaf-marketing-respons/ensembles/pred_ensemble_3',createDir = T) ## out 

####
source(paste0(ff.getPath("code"),"fastClassification.R"))

################# MODELS 

modelList = list(
  
  ##############################################################################
  #                                    1 LAYER                                 #
  ##############################################################################
  
  #list(layer = 1  , dataProc = "base", mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T)
  
  
  #list(layer = 1  , dataProc = "doc_proc_2", mod = 'libsvm'  ,tune=T)
  
  #list(layer = 1  , dataProc = "CBDB", CBDB_num=1 , mod = 'xgbTreeGTJ'  , eta=0.02 , max_depth=2, tune=T), 
  
  #list(layer = 1  , dataProc = "poly2cut1000",       mod = 'xgbTreeGTJ'  , eta=0.02 , max_depth=3, tune=T) , 
  
  
  #list(layer = 1  , dataProc = "pca95var",           mod = 'xgbTreeGTJ'  , eta=0.02 , max_depth=3, tune=T) , 
  #list(layer = 1  , dataProc = "pcaElbow",           mod = 'xgbTreeGTJ'  , eta=0.02 , max_depth=3, tune=T) , 
  
  
  #list(layer = 1  , dataProc = "doc_proc_2", mod = 'xgbTreeGTJ'  , eta=0.02 , max_depth=2 , tune=T)
  #list(layer = 1  , dataProc = "doc_proc_2", mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T)
  list(layer = 1  , dataProc = "base", mod = 'xgbTreeGTJ'  , eta=0.01 , tune=T)
  
  ### on going 
  #list(layer = 1  , dataProc = "pca95var", mod = 'glm'   , tune=F)
  #list(layer = 1  , dataProc = "doc_proc_3_uc", mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T),
  #list(layer = 1  , dataProc = "doc_proc_2_uc", mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T)
  #list(layer = 1  , dataProc = "pca95var", mod = 'glmnet_alpha_0.5'   , tune=T),
  #list(layer = 1  , dataProc = "doc_proc_3", mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T),
  #list(layer = 1  , dataProc = "doc_proc_2", mod = 'glm'   , tune=F)
  ########
  
  ### TODO 
  # 
  # 
  
  #list(layer = 1  , dataProc = "CBDB", CBDB_num=3 ,  mod = 'glmnet_alpha_1'   , tune=T),
  #
  #list(layer = 1  , dataProc = "CBDB", CBDB_num=5 ,  mod = 'glmnet_alpha_0'   , tune=T),
  
  #list(layer = 1  , dataProc = "CBDB", CBDB_num=2 ,  mod = 'xgbTreeGTJ'  , eta=0.01 , tune=T) 
  ####
  
  ##############################################################################
  #                                    2 LAYER                                 #
  ##############################################################################
  
  ############## 
  #list(layer = 2  , dataProc = "default", mod = 'glmnet_alpha_1'  , tune=T), 
  #list(layer = 2  , dataProc = "default", mod = 'glmnet_alpha_0.4'  , tune=T), 
  #list(layer = 2  , dataProc = "default", mod = 'glmnet_alpha_0.5'  , tune=T), 
  #list(layer = 2  , dataProc = "default", mod = 'glmnet_alpha_0.6'  , tune=T), 
  #list(layer = 2  , dataProc = "default", mod = 'glmnet_alpha_0'  , tune=T),
  
  #list(layer = 2  , dataProc = "default", mod = 'glm'  ,tune=F),
  #list(layer = 2  , dataProc = "default", mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T), 
  #list(layer = 2  , dataProc = "default", mod = 'xgbTreeGTJ'  , eta=0.01 , tune=T), 
  #list(layer = 2  , dataProc = "default", mod = 'xgbTreeGTJ'  , eta=0.005 , tune=T)
  

  ### threshold 0.7 
  #list(layer = 2  , dataProc = "default", threshold = 0.7 , mod = 'glmnet_alpha_1'  , tune=T), 
  #list(layer = 2  , dataProc = "default", threshold = 0.7 , mod = 'glmnet_alpha_0.5'  , tune=T), 
  #list(layer = 2  , dataProc = "default", threshold = 0.7 , mod = 'glmnet_alpha_0'  , tune=T),
  
  #list(layer = 2  , dataProc = "default", threshold = 0.7 , mod = 'glm'  ,tune=F),
  #list(layer = 2  , dataProc = "default", threshold = 0.7 , mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T), 
  #list(layer = 2  , dataProc = "default", threshold = 0.7 , mod = 'libsvm'  ,tune=T),
  
  ### threshold 0.75
  #list(layer = 2  , dataProc = "default", threshold = 0.75 , mod = 'glmnet_alpha_1'  , tune=T), 
  #list(layer = 2  , dataProc = "default", threshold = 0.75 , mod = 'glmnet_alpha_0.5'  , tune=T), 
  #list(layer = 2  , dataProc = "default", threshold = 0.75 , mod = 'glmnet_alpha_0'  , tune=T), 
  
  #list(layer = 2  , dataProc = "default", threshold = 0.75 , mod = 'glm'  ,tune=F),
  #list(layer = 2  , dataProc = "default", threshold = 0.75 , mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T), 
  #list(layer = 2  , dataProc = "default", threshold = 0.75 , mod = 'libsvm'  ,tune=T), 
  
  ############## kmeans = 1 
  ### 
  #list(layer = 2  , dataProc = "default", kmeans = 1 , mod = 'glmnet_alpha_1'  , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 1 , mod = 'glmnet_alpha_0.5'  , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 1 , mod = 'glmnet_alpha_0'  , tune=T),
  #list(layer = 2  , dataProc = "default", kmeans = 1 , mod = 'glm'  ,tune=F),
  #list(layer = 2  , dataProc = "default", kmeans = 1 , mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 1 , mod = 'libsvm'  ,tune=T),
  
  ############## kmeans = 3 
  ### 
  #list(layer = 2  , dataProc = "default", kmeans = 3 , mod = 'glmnet_alpha_1'  , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 3 , mod = 'glmnet_alpha_0.5'  , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 3 , mod = 'glmnet_alpha_0'  , tune=T),
  #list(layer = 2  , dataProc = "default", kmeans = 3 , mod = 'glm'  ,tune=F),
  #list(layer = 2  , dataProc = "default", kmeans = 3 , mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 3 , mod = 'libsvm'  ,tune=T), 
  
  ############## err_an 
  ### 
  #list(layer = 2  , dataProc = "default", kmeans = 3 , err_an =T , mod = 'glmnet_alpha_1'  , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 3 , err_an =T , mod = 'glmnet_alpha_0.5'  , tune=T), 
  #list(layer = 2  , dataProc = "default", kmeans = 3 , err_an =T , mod = 'glmnet_alpha_0'  , tune=T),
  #list(layer = 2  , dataProc = "default", kmeans = 3 , err_an =T , mod = 'glm'  ,tune=F),
  #list(layer = 2  , dataProc = "default", kmeans = 3 , err_an =T , mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T)
  #list(layer = 2  , dataProc = "default", kmeans = 3 , err_an =T , mod = 'libsvm'  ,tune=T)
  
  ##############################################################################
  #                                    3 LAYER                                 #
  ##############################################################################
  
#   list(layer = 3  , dataProc = "default", mod = 'glmnet_alpha_1'  , tune=T), 
#   list(layer = 3  , dataProc = "default", mod = 'glmnet_alpha_0.4'  , tune=T), 
#   list(layer = 3  , dataProc = "default", mod = 'glmnet_alpha_0.5'  , tune=T), 
#   list(layer = 3  , dataProc = "default", mod = 'glmnet_alpha_0.6'  , tune=T), 
#   list(layer = 3  , dataProc = "default", mod = 'glmnet_alpha_0'  , tune=T),
#   
#   list(layer = 3  , dataProc = "default", mod = 'glm'  ,tune=F),
#   list(layer = 3  , dataProc = "default", mod = 'xgbTreeGTJ'  , eta=0.02 , tune=T), 
#   list(layer = 3  , dataProc = "default", mod = 'xgbTreeGTJ'  , eta=0.01 , tune=T), 
#   list(layer = 3  , dataProc = "default", mod = 'xgbTreeGTJ'  , eta=0.005 , tune=T)
  
  
)

modelList = buildIDModelList(modelList)

##############
## MAIN LOOP 
##############
ptm <- proc.time()
for (m in  seq_along(modelList) ) { 
  cat(">>> now processing:\n")
  print(modelList[[m]])
  
  ## data 
  data = getData(modelList[[m]])
  Ytrain = data$Ytrain
  Xtrain = data$Xtrain
  Xtest = data$Xtest 
  test_id = data$test_id 
  rm(data)
  gc()
  if (DEBUG) {
    cat("> debug .. \n")
    Xtrain = Xtrain[1:100,]
    #Xtest = Xtrain[,1:10]
    Ytrain = Ytrain[1:100]
    gc()
  }
  
  ##############
  ## TUNE 
  ##############
  controlObject = trainControl(method = "repeatedcv", repeats = 1, number = 4 , summaryFunction = twoClassSummary , classProbs = TRUE)
  l = ff.trainAndPredict.class ( Ytrain=Ytrain ,
                                 Xtrain=Xtrain , 
                                 Xtest=Xtest , 
                                 model.label=modelList[[m]]$mod , 
                                 controlObject=controlObject, 
                                 best.tuning = TRUE, 
                                 verbose = TRUE, 
                                 removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = F, 
                                 xgb.metric.fun = NULL, 
                                 xgb.maximize = TRUE, 
                                 metric.label = 'auc', 
                                 xgb.foldList = NULL,
                                 xgb.eta = modelList[[m]]$eta, 
                                 xgb.max_depth = modelList[[m]]$max_depth, 
                                 xgb.cv.default = FALSE)
  
  if ( !is.null(l$model) ) {
    roc_mod = max(l$model$results$ROC)
    bestTune = l$model$bestTune
    pred = l$pred
    pred.prob = l$pred.prob
    secs = l$secs 
    rm(l)
  } else {
    stop(paste('model',modelList[[m]]$mod,':error!'))
  }
  
  ## write prediction on disk 
  submission <- data.frame(ID=test_id)
  submission$target <- pred.prob
  print(head(submission))
  write.csv(submission,
            quote=FALSE, 
            file=paste0(ff.getPath(paste0("submission_",modelList[[m]]$layer)),modelList[[m]]$id) ,
            row.names=FALSE)
  
  ## write best tune on disk 
  tuneGrid = data.frame(model=modelList[[m]]$mod,secs=secs,ROC=roc_mod) 
  tuneGrid = cbind(tuneGrid,bestTune)
  write.csv(tuneGrid,
            quote=FALSE, 
            file=paste0(ff.getPath(paste0("best_tune_",modelList[[m]]$layer)),modelList[[m]]$id) ,
            row.names=FALSE)
  
  ##############
  ## ENSEMB 
  ##############
  cat(">>> Ensembling ... \n") 
  bestTune = NULL
  if (modelList[[m]]$tune) {
    bestTune = as.data.frame( fread(paste0(ff.getPath(paste0("best_tune_",modelList[[m]]$layer)),modelList[[m]]$id)))
    stopifnot( nrow(bestTune)>0 , !is.null(bestTune) )
  }
  submission = as.data.frame( fread(paste0(ff.getPath(paste0("submission_",modelList[[m]]$layer)),modelList[[m]]$id)))
  stopifnot( nrow(submission)>0 , !is.null(submission) )
 
  ## controlObject 
  nFolds = controlObject$number
  nrepeats =  controlObject$repeats
  index = caret::createMultiFolds(y=Ytrain, nFolds, nrepeats)
  indexOut <- lapply(index, function(training, allSamples) allSamples[-unique(training)], allSamples = seq(along = Ytrain))
  controlObject = trainControl(method = "repeatedcv", 
                               ## The method doesn't really matter
                               ## since we defined the resamples
                               index = index, 
                               indexOut = indexOut , 
                               summaryFunction = twoClassSummary , classProbs = TRUE)
  rm(list = c("index","indexOut"))
  
  ## adjust resamples for unbalanced class dataset
  if (identical("doc_proc_3_uc",modelList[[m]]$dataProc) || identical("doc_proc_2_uc",modelList[[m]]$dataProc)) {
    cat(">>> adjusting resamples for unbalanced classes ... \n")
    
    ## Ytrain
    cat (">>> loading whole Ytrain ... \n")
    train_raw = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
    Ytrain = train_raw$target
    rm(list=c("train_raw"))
    gc()
    
    if (identical("doc_proc_3_uc",modelList[[m]]$dataProc)) {
      cat (">>> loading whole Xtrain_docproc3.csv ... \n")
      ## Xtrain 
      Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_docproc3.csv" , sep='') , stringsAsFactors = F))
    } else {
      cat (">>> loading whole Xtrain_docproc2.csv ... \n")
      ## Xtrain
      Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_docproc2.csv" , sep='') , stringsAsFactors = F))
    }
    
    ## redo resamples 
    index = caret::createMultiFolds(y=Ytrain, nFolds, nrepeats)
    indexOut <- lapply(index, function(training, allSamples) allSamples[-unique(training)], allSamples = seq(along = Ytrain))
    controlObject = trainControl(method = "repeatedcv", 
                                 ## The method doesn't really matter
                                 ## since we defined the resamples
                                 index = index, 
                                 indexOut = indexOut , 
                                 summaryFunction = twoClassSummary , classProbs = TRUE)
    rm(list = c("index","indexOut"))
    
    ## filter resamples
    for (fI in seq_along(controlObject$index)) {
       
      idx1 = which( Ytrain[controlObject$index[fI][[1]]] == 1)
      idx0 = which( Ytrain[controlObject$index[fI][[1]]] == 0)
      
      inr = length(idx1) / length(idx0)
      
      l0 = 1.5*length(idx1) ## ratio = 0.6666667 (instead of 0.3333333)
      
      idx0_new = sample(x = idx0 , size = l0 , replace = F)
      
      inr = length(idx1) / length(idx0_new)
      
      stopifnot(length(idx0_new)==length(unique(idx0_new)))
      
      controlObject$index[fI][[1]] = c(idx1,idx0_new)
    }
    
  }  ## end-if unbalanced classes adjiusting 
  
  ## createEnsemble
  ff.setMaxCuncurrentThreads(4)
  ens = NULL 
  if (modelList[[m]]$mod == "xgbTreeGTJ") {
    ens = ff.createEnsemble (Xtrain = Xtrain,
                             Xtest = Xtest,
                             y = Ytrain,
                             caretModelName = 'xgbTree', 
                             predTest <- submission$target,
                             bestTune = expand.grid(
                               nrounds = bestTune$early.stop ,
                               max_depth = if (!is.null(modelList[[m]]$max_depth)) modelList[[m]]$max_depth else 8 ,  
                               eta = modelList[[m]]$eta ),
                             removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = F, 
                             controlObject = controlObject, 
                             parallelize = T,
                             verbose = T , 
                             regression = FALSE, 
                             
                             ### ... 
                             objective = "binary:logistic",
                             eval_metric = "auc", 
                             subsample = 0.7 , 
                             colsample_bytree = 0.6 , 
                             scale_pos_weight = 0.8 , 
                             #silent = 1 , 
                             max_delta_step = 2)
    
  } else if (modelList[[m]]$tune){
    ens = ff.createEnsemble (Xtrain = Xtrain,
                             Xtest = Xtest,
                             y = Ytrain,
                             caretModelName = modelList[[m]]$mod, 
                             predTest = submission$target,
                             bestTune = bestTune[, 4:ncol(bestTune) , drop = F], 
                             removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = F, 
                             controlObject = controlObject, 
                             parallelize = T,
                             verbose = T , 
                             regression = FALSE)
  } else {
    ens = ff.createEnsemble (Xtrain = Xtrain,
                             Xtest = Xtest,
                             y = Ytrain,
                             caretModelName = modelList[[m]]$mod, 
                             predTest = submission$target,
                             bestTune = NULL, 
                             removePredictorsMakingIllConditionedSquareMatrix_forLinearModels = F, 
                             controlObject = controlObject, 
                             parallelize = T,
                             verbose = T , 
                             regression = FALSE)
  }
  
  ## assemble 
  assemble = c(ens$predTrain,ens$predTest)
  write.csv(data.frame(id = seq_along(assemble) , assemble=assemble),
            quote=FALSE, 
            file=paste0(ff.getPath(paste0("ensemble_",modelList[[m]]$layer)),modelList[[m]]$id) ,
            row.names=FALSE)
  
}
####### end of loop 
tm = proc.time() - ptm
secs = as.numeric(tm[3])
cat(">>> MAIN LOOP >>> Time elapsed:",secs," secs. [",secs/60,"min.] [",secs/(60*60),"hours] \n")
############## end of MAIN LOOP 


