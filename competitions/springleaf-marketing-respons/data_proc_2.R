library(data.table)
library(xgboost)
library(fastfurious)

### FUNCS

getNode <- function (nodes,nI) {
  if (length(nodes)<nI) {
    return(nodes[sample(length(nodes),1)])
  } else {
    return(nodes[nI])
  }
}

### CONFIG 
#TASK = "base"
#TASK = "polycut"
#TASK = "pca"
#TASK = "cor"
#TASK = "CBDB"
#TASK = "K-MEANS"


### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/springleaf-marketing-respons')
ff.bindPath(type = 'code' , sub_path = 'competitions/springleaf-marketing-respons')

ff.bindPath(type = 'elab' , sub_path = 'dataset/springleaf-marketing-respons/elab',createDir = T) ## out 


### PROCS 
if (TASK == "base_removeNAs") { 
  cat(">>> loading data ... \n")
  train = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
  test = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
  
  train = train[,-c(1,208,214,839,846,1427,ncol(train))]
  test = test[,-c(1,208,214,839,846,1427)]
  
  feature.names <- colnames(train)
  predNA = unlist(lapply(1:length(feature.names) , function(i) {
    f = feature.names[i]
    sum(is.na(train[[f]]))>0 || sum(is.na(test[[f]]))>0
  }))
  
  predNAIdx = which(predNA)
  cat(">>> Removing predictors with NAs values: ",feature.names[predNAIdx], "... \n")
  
  train = train[,-predNAIdx]
  test = test[,-predNAIdx]
  
  cat('>>> date predictors with "" instead of NAs ... \n')
  toRemoveL = c("VAR_0073","VAR_0156","VAR_0157","VAR_0158","VAR_0159",
                "VAR_0166","VAR_0167","VAR_0168","VAR_0169", "VAR_0176","VAR_0177","VAR_0178","VAR_0179")
  toRemoveIdx = grep(pattern = paste(toRemoveL,collapse = "|") , x = colnames(train) )
  
  stopifnot(sum(colnames(train) != colnames(train))==0)
  
  train = train[,-toRemoveIdx]
  test = test[,-toRemoveIdx]
  
  cat(">>> loading meta.table ... \n")
  meta.table = as.data.frame( fread(paste(ff.getPath("elab") , "meta.data_compiled.csv" , sep='') , stringsAsFactors = F))
  cat("completing meta.table ... \n")
  meta.table[is.na(meta.table)] <- FALSE
  
  stopifnot(sum(colnames(train) != colnames(train))==0)
  
  feature.names <- colnames(train)
  mt = rep(NA,length(feature.names))
  for (i in 1:length(feature.names)) {
    f = feature.names[i]
    meta.table_f = meta.table[meta.table$feature.name == f , ]
    stopifnot(nrow(meta.table_f)>0)
    mt[i] <- if (meta.table_f$isCateg) "C" else if (meta.table_f$isDate) "D" else "N"
  }
  
  ##
  cat(">>> converting dates ... \n")
  for (i in  which(mt == "D")) {
    
    ## train
    date_tmp = rep(as.Date(strsplit("22MAR12:00:00:00" , split = ":")[[1]][1] , format = "%d%B%y"),nrow(train))
    al = lapply ( 1:nrow(train) , function(j) {
      #al = lapply ( 1:10 , function(j) {
      date_tmp[j] <<- as.Date(strsplit(train[j,feature.names[i]] , split = ":")[[1]][1] , format = "%d%B%y")
    })
    train[,feature.names[i]] = date_tmp
    
    ## test 
    date_tmp = rep(as.Date(strsplit("22MAR12:00:00:00" , split = ":")[[1]][1] , format = "%d%B%y"),nrow(test))
    al = lapply ( 1:nrow(test) , function(j) {
      date_tmp[j] <<- as.Date(strsplit(test[j,feature.names[i]] , split = ":")[[1]][1] , format = "%d%B%y")
    })
    test[,feature.names[i]] = date_tmp
  }
  
  ##
  cat(">>> handling categorical binary predictors or with too many levels ... \n")
  for (i in 1:length(feature.names)) {
    f = feature.names[i]
    if ( class(train[[f]])=="character" && mt[i] != "C" && mt[i] != "D" ) {
      print(f)
      levels <- unique(c(train[[f]], test[[f]]))
      train[[f]] <- as.integer(factor(train[[f]], levels=levels))
      test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
    }
  }
  
  cat(">>> checking that predictors with NAs values are date predictors ... \n")
  predNA = unlist(lapply(1:length(feature.names) , function(i) {
    f = feature.names[i]
    sum(is.na(train[[f]]))>0
  }))
  stopifnot(sum( sort(which(predNA)) != sort(which(mt == "D")) )==0)
  
  ##
  cat(">>> making feature set ... \n")
  ds = ff.makeFeatureSet(data.train = train[,feature.names] , 
                         data.test = test[,feature.names] , 
                         meta = mt , 
                         remove1DummyVarInCatPreds = T)
  
  
  ##
  cat(">>> imputing date predictors with NAs values with -1.5 ... \n")
  ds$traindata[is.na(ds$traindata)] = -10
  ds$testdata[is.na(ds$testdata)] = -10 
  
  
  ##
  cat(">>> removing 0-variance predictors and identical predictors ... \n")
  l = ff.featureFilter (ds$traindata,
                        ds$testdata,
                        removeOnlyZeroVariacePredictors=TRUE,
                        performVarianceAnalysisOnTrainSetOnly = TRUE , 
                        removePredictorsMakingIllConditionedSquareMatrix = FALSE, 
                        removeHighCorrelatedPredictors = FALSE, 
                        featureScaling = FALSE, 
                        verbose = TRUE)
  
  ## 
  cat(">>> writing on disk ... \n")
  
  write.csv(l$traindata,
            quote=FALSE, 
            file=paste0(ff.getPath("elab"),"Xtrain_docproc3_clean.csv"),
            row.names=FALSE)
  
  write.csv(l$testdata,
            quote=FALSE, 
            file=paste0(ff.getPath("elab"),"Xtest_docproc3_clean.csv"),
            row.names=FALSE)
  
} else if (TASK == "base") {
  cat(">>> loading data ... \n")
  train = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
  test = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
  
  train = train[,-c(1,208,214,839,846,1427)]
  test = test[,-c(1,208,214,839,846,1427)]
  feature.names <- names(train)[2:ncol(train)-1]
  
  cat(">>> loading meta.table ... \n")
  meta.table = as.data.frame( fread(paste(ff.getPath("elab") , "meta.data_compiled.csv" , sep='') , stringsAsFactors = F))
  cat("completing meta.table ... \n")
  meta.table[is.na(meta.table)] <- FALSE
  
  feature.names <- names(train)[2:ncol(train)-1]
  mt = rep(NA,length(feature.names))
  for (i in 1:length(feature.names)) {
    f = feature.names[i]
    meta.table_f = meta.table[meta.table$feature.name == f , ]
    stopifnot(nrow(meta.table_f)>0)
    mt[i] <- if (meta.table_f$isCateg) "C" else if (meta.table_f$isDate) "D" else "N"
  }
  
  cat(">>> replacing NAs with -1.5 ... \n")
  train[is.na(train)] <- -1.5
  test[is.na(test)]   <- -1.5
  
  ##
  cat(">>> converting dates ... \n")
  for (i in  which(mt == "D")) {
    
    ## train
    date_tmp = rep(as.Date(strsplit("22MAR12:00:00:00" , split = ":")[[1]][1] , format = "%d%B%y"),nrow(train))
    al = lapply ( 1:nrow(train) , function(j) {
      #al = lapply ( 1:10 , function(j) {
      date_tmp[j] <<- as.Date(strsplit(train[j,feature.names[i]] , split = ":")[[1]][1] , format = "%d%B%y")
    })
    train[,feature.names[i]] = date_tmp
    
    ## test 
    date_tmp = rep(as.Date(strsplit("22MAR12:00:00:00" , split = ":")[[1]][1] , format = "%d%B%y"),nrow(test))
    al = lapply ( 1:nrow(test) , function(j) {
      date_tmp[j] <<- as.Date(strsplit(test[j,feature.names[i]] , split = ":")[[1]][1] , format = "%d%B%y")
    })
    test[,feature.names[i]] = date_tmp
  }
  
  ##
  cat(">>> handling categorical binary predictors or with too many levels ... \n")
  for (i in 1:length(feature.names)) {
    f = feature.names[i]
    if ( class(train[[f]])=="character" && mt[i] != "C" && mt[i] != "D" ) {
      print(f)
      levels <- unique(c(train[[f]], test[[f]]))
      train[[f]] <- as.integer(factor(train[[f]], levels=levels))
      test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
    }
  }
  
  ## 
  cat(">>> checking that predictors with NAs values are date predictors ... \n")
  predNA = unlist(lapply(1:length(feature.names) , function(i) {
    f = feature.names[i]
    sum(is.na(train[[f]]))>0
  }))
  stopifnot(sum( sort(which(predNA)) != sort(which(mt == "D")) )==0)
  
  ##
  cat(">>> making feature set ... \n")
  ds = ff.makeFeatureSet(data.train = train[,feature.names] , 
                         data.test = test[,feature.names] , 
                         meta = mt , 
                         remove1DummyVarInCatPreds = T)
  
  ##
  cat(">>> imputing date predictors with NAs values with -1.5 ... \n")
  ds$traindata[is.na(ds$traindata)] = -1.5 
  ds$testdata[is.na(ds$testdata)] = -1.5 
  
  ##
  cat(">>> removing 0-variance predictors and identical predictors ... \n")
  l = ff.featureFilter (ds$traindata,
                        ds$testdata,
                        removeOnlyZeroVariacePredictors=TRUE,
                        performVarianceAnalysisOnTrainSetOnly = TRUE , 
                        removePredictorsMakingIllConditionedSquareMatrix = FALSE, 
                        removeHighCorrelatedPredictors = FALSE, 
                        featureScaling = FALSE, 
                        verbose = TRUE)
  
  ## 
  cat(">>> writing on disk ... \n")
  
  write.csv(l$traindata,
            quote=FALSE, 
            file=paste0(ff.getPath("elab"),"Xtrain_docproc2.csv"),
            row.names=FALSE)
  
  write.csv(l$testdata,
            quote=FALSE, 
            file=paste0(ff.getPath("elab"),"Xtest_docproc2.csv"),
            row.names=FALSE)
} else if (TASK == "polycut") {
  
  cat(">>> processing task:",TASK," ... \n")
  
  ## conf 
  poly.degree = 2
  abs_th = 1000
  
  ptm <- proc.time()
  ## Ytrain 
  cat(">>> loading raw data for extracting Ytrain ... \n")
  train_raw = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
  
  Ytrain = train_raw$target
  
  rm(list=c("train_raw")); gc()
  
  ## Xtrain / Xtest 
  cat(">>> loading Xtrain / Xtest  ... \n")
  Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_docproc2.csv" , sep='') , stringsAsFactors = F))
  Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_docproc2.csv" , sep='') , stringsAsFactors = F))
  n.train = nrow(Xtrain)
  
  ## ff.poly
  cat(">>> computing poly features [degree:",poly.degree,"]  ... \n")
  data = rbind(Xtrain,Xtest)
  rm(list=c("Xtrain","Xtest"))
  
  ##### data = ff.poly(x = data , n = poly.degree)
  ########################################
  n = poly.degree
  x <- data
  rm(list=c("data"))
  
  ##
  x.poly = as.data.frame(matrix(rep(0 , nrow(x)*ncol(x)*(n-1)) , nrow = nrow(x)))
  lapply(2:n,function(i){
    d = x 
    d[] <- lapply(X = x , FUN = function(x){
      return(x^i)
    })  
    colnames(d) = paste(colnames(x),'^',i,sep='')
    x.poly[,((i-2)*ncol(x)+1):((i-1)*ncol(x))] <<- d 
    colnames(x.poly)[((i-2)*ncol(x)+1):((i-1)*ncol(x))] <<- colnames(d)
  })  
  
  ##
  x.poly.2 = as.data.frame(matrix(rep(0 , nrow(x)*ncol(x)*(n-1)) , nrow = nrow(x)))
  lapply(2:n,function(i){
    d = x 
    d[] <- lapply(X = x , FUN = function(x){
      return(x^(1/i))
    })  
    colnames(d) = paste(colnames(x),'^1/',i,sep='')
    x.poly.2[,((i-2)*ncol(x)+1):((i-1)*ncol(x))] <<- d 
    colnames(x.poly.2)[((i-2)*ncol(x)+1):((i-1)*ncol(x))] <<- colnames(d)
  })
  
  ### 
  data = cbind(x,x.poly,x.poly.2)
  rm(list=c("x","x.poly","x.poly.2"))
  ########################################
  Xtrain = data[1:n.train,]
  Xtest = data[(n.train+1):nrow(data),]
  rm(list=c("data"))
  
  ## ff.corrFilter
  cat(">>> computing correlation with Ytrain [abs_th:",abs_th,"]  ... \n")
  l = ff.corrFilter(Xtrain=Xtrain,Xtest=Xtest,y=Ytrain,abs_th = abs_th , rel_th = NULL , method = "spearman")
  Xtrain = l$Xtrain
  Xtest =l$Xtest
  rm(list=c("l"))
  
  ## write on disk 
  cat(">>> writing on disk ... \n")
  write.csv(Xtrain,
            quote=FALSE, 
            file=paste0(ff.getPath("elab"),"Xtrain_docproc2_poly3cut1000.csv"),
            row.names=FALSE)
  write.csv(Xtest,
            quote=FALSE, 
            file=paste0(ff.getPath("elab"),"Xtest_docproc2_poly3cut1000.csv"),
            row.names=FALSE)
  
  ##
  tm = proc.time() - ptm
  secs = as.numeric(tm[3])
  cat(">>> time elapsed:",secs," secs. [min:",secs/60,"] [hours:",secs/(60*60),"]\n")
} else if (TASK == "pca") { 
  cat(">>> processing task:",TASK," ... \n")
  
  ## Xtrain / Xtest 
  cat(">>> loading Xtrain / Xtest  ... \n")
  Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_docproc2.csv" , sep='') , stringsAsFactors = F))
  Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_docproc2.csv" , sep='') , stringsAsFactors = F))
  n.train = nrow(Xtrain)
  
  ## Principal Components
  cat(">>> computing Principal Components  ... \n")
  data = rbind(Xtrain,Xtest)
  rm(list=c("Xtrain","Xtest"))
  
  #data = data[,-c("VAR_0212") ] ## eliminating 0-var predictors. >>> Note: this pred has 250737 different values but ~e-313 
  idxRem = grep(pattern = "VAR_0212" , x = colnames(data))
  cat(">>> eliminating predictor --> idx:",idxRem," - name:",colnames(data)[idxRem],"...\n")
  stopifnot(colnames(data)[idxRem]=="VAR_0212")
  data = data[,-idxRem]
  
  pr.out = prcomp(data, scale=T , retx = T)
  rm(list=c("data"))
  
  cat(">>> found ",ncol(pr.out$x),"PCs ... \n")
  pr.var = pr.out$sdev^2
  pve = pr.var/sum(pr.var)
  
  PC.num = which(-diff(pve) == max(-diff(pve))) + 1 
  
  cat(">>> Number of PCs to hold according to the elbow rule: first ",PC.num," PCs ... \n")
  
  cumVar <- cumsum(pve) 
  numComp95 <- max(2, which.max(cumVar > 0.95))
  
  cat(">>> Number of PCs to hold explaining 95% of variance: first ",numComp95," PCs ... \n")
  
  ## write on disk 
  cat(">>> writing on disk ... \n")
  write.csv(pr.out$x[1:n.train,1:PC.num,drop=F],
            quote=FALSE, 
            file=paste0(ff.getPath("elab"),"Xtrain_docproc2_pca_elbow.csv"),
            row.names=FALSE)
  write.csv(pr.out$x[((n.train+1):nrow(pr.out$x)),1:PC.num,drop=F],
            quote=FALSE, 
            file=paste0(ff.getPath("elab"),"Xtest_docproc2_pca_elbow.csv"),
            row.names=FALSE)
  
  write.csv(pr.out$x[1:n.train,1:numComp95,drop=F],
            quote=FALSE, 
            file=paste0(ff.getPath("elab"),"Xtrain_docproc2_pca_95var.csv"),
            row.names=FALSE)
  write.csv(pr.out$x[((n.train+1):nrow(pr.out$x)),1:numComp95,drop=F],
            quote=FALSE, 
            file=paste0(ff.getPath("elab"),"Xtest_docproc2_pca_95var.csv"),
            row.names=FALSE)
  
  ## plot 
  jpeg(filename=paste(ff.getPath("elab"),"pca.jpg",sep=''))
  plot(pve[1:numComp95] , xlab="Principal Component" , 
       ylab = "Proportion of Variance Explained" , ylim=c(0,1) , type='b' , 
       main ="Considered PCs explaining 95% Variance")
  dev.off()
} else if (TASK == "cor") {
  cat(">>> processing task:",TASK," ... \n")
  
  ## Xtrain / Xtest 
  cat(">>> loading Xtrain / Xtest  ... \n")
  Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_docproc2.csv" , sep='') , stringsAsFactors = F))
  Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_docproc2.csv" , sep='') , stringsAsFactors = F))
  
  ## Correlation 
  cat(">>> computing correlation  ... \n")
  data = rbind(Xtrain,Xtest)
  rm(list=c("Xtrain","Xtest"))
  
  corrMatrix = cor(x = data , method = "spearman")
  
  ## write on disk 
  cat(">>> writing on disk ... \n")
  write.csv(corrMatrix,
            quote=FALSE, 
            file=paste0(ff.getPath("elab"),"corr_spearman.csv"),
            row.names=FALSE)
} else if (TASK == "CBDB") {
  ## conf 
  reduction_factor = 5/8
  #minCorrIntraClusterASQuantile = 0.9
  minCorrIntraCluster = 0.95
  
  ## 
  cat(">>> processing task:",TASK," ... \n")
  require(Matrix)
  
  ff.bindPath(type = 'CBDB' , sub_path = 'dataset/springleaf-marketing-respons/elab/CBDB',createDir = T) ## out 
  
  ## loading Corr Matrix 
  corMM = as.data.frame( fread(paste(ff.getPath("elab") , "corr_spearman.csv" , sep='') , stringsAsFactors = F))
  
  ##
  stopifnot(sum(is.na(corMM))==0)
  
  ##
  rownames(corMM) = colnames(corMM)
  corM = Matrix(as.matrix(abs(corMM)))
  stopifnot(isSymmetric(corM))
  initCorrM.sym <- corM
  corM[lower.tri(corM, diag = TRUE)] <-0 
  initCorrM <- corM
  
  ##
  clusters = list()
  #minCorrIntraCluster = as.numeric(quantile(as.numeric(corM) , probs = minCorrIntraClusterASQuantile))
  
  cat(">>> number of cells in pred corr matrix >",minCorrIntraCluster,":" ,sum(initCorrM>=minCorrIntraCluster),"\n")
  
  ##
  reduction_iter = floor(ncol(initCorrM)*reduction_factor)
  for (reduction in 1:reduction_iter) { 
    cat(">>> reduction:",reduction,"/",reduction_iter," ..\n")
    
    cp = which (corM == max(corM) , arr.ind = TRUE)
    if (nrow(cp)>1) cp = cp[1,,drop=F]
    cn = colnames(corM)[cp]
    nocn = colnames(corM)[-cp]
    
    currCluster <- NULL
    if (  sum( substr(x=cn, start=1, stop=3) == "CL_")  == 0) { 
      ## new cluster 
      name = paste0("CL_",paste(cn,collapse = "_"))
      clusters[[name]] <- list( name = name , nodes=cn , corrIntra = min(initCorrM.sym[cn,cn]) )
      currCluster <- clusters[[name]]
      
    } else if (  sum( substr(x=cn, start=1, stop=3) == "CL_")  == 1) { 
      ## add to existing cluster 
      cl_name <- if (substr(x=cn[1], start=1, stop=3) == "CL_") cn[1] else cn[2]
      new_node <- if (substr(x=cn[1], start=1, stop=3) == "CL_") cn[2] else cn[1]
      
      new_nodes = c(clusters[[cl_name]]$nodes , new_node ) 
      new_corrIntra = min(initCorrM.sym[new_nodes,new_nodes])
      new_name = paste0("CL_",paste(new_nodes,collapse = "_"))
      
      clusters[[new_name]] <- list( name=new_name , nodes=new_nodes , corrIntra=new_corrIntra)
      clusters[[cl_name]] <- NULL
      
      currCluster <- clusters[[new_name]]
      
    } else if (  sum( substr(x=cn, start=1, stop=3) == "CL_")  == 2) {
      ## Fuse 2 clusters 
      new_nodes <- c(clusters[[cn[1]]]$nodes , clusters[[cn[2]]]$nodes ) 
      new_corrIntra <- min(initCorrM.sym[new_nodes,new_nodes])
      
      new_name = paste0("CL_",paste(new_nodes,collapse = "_"))
      clusters[[new_name]] <- list( name=new_name , nodes=new_nodes , corrIntra=new_corrIntra )
      
      clusters[[cn[1]]] <- NULL
      clusters[[cn[2]]] <- NULL
      
      currCluster <- clusters[[new_name]]
    }
    
    ## 
    if (currCluster$corrIntra < minCorrIntraCluster) {
      cat(">>> stop: reached minCorrIntraCluster:",currCluster$corrIntra,"<",minCorrIntraCluster,"! \n")
      break 
    }
    
    ##
    corMs <- Matrix(rep( 0 , (ncol(corM)-1)^2 ) , ncol = ncol(corM)-1 , nrow = nrow(corM)-1 )
    colnames(corMs) = c(nocn,  currCluster$name )
    rownames(corMs) = colnames(corMs)
    corMs[1:(nrow(corMs)-1),1:(ncol(corMs)-1) ] = corM[nocn,nocn] 
    
    for (ii in (1:(nrow(corMs)-1)) ) {
      ## nodes_row
      nodes_row <- NULL
      name_row <- colnames(corMs)[ii]
      if (  substr(x=name_row, start=1, stop=3) == "CL_" ) {
        nodes_row <- clusters[[name_row]]$nodes
      } else {
        nodes_row <- name_row
      }
      
      ## nodes_col 
      nodes_col <- currCluster$nodes
      
      ##
      stopifnot( length(setdiff(x = nodes_row , y = nodes_col)) == length(nodes_row) ) 
      stopifnot( length(setdiff(x = nodes_col , y = nodes_row)) == length(nodes_col) ) 
      
      ## 
      corMs[ii, ncol(corMs) ] <- mean(initCorrM.sym[nodes_row,nodes_col,drop=F])
    }
    
    stopifnot(max(corMs)<=1)
    corM <- corMs 
  }
  
  ## >>> check
  pred_tot = NULL
  for (i in seq_along(clusters)) {
    pred_tot = union(pred_tot , clusters[[i]]$nodes)
  }
  
  ## clusters must have distinct nodes 
  stopifnot(length(unique(pred_tot)) == length(pred_tot))
  
  ## leaves inside and outside clusters must be disjoint and they must be a partition of initial predictors's set  
  leaves = setdiff(x = colnames(initCorrM) , pred_tot)
  stopifnot(length(leaves) + length(pred_tot) == length(colnames(initCorrM)))
  
  ## info 
  cat(">>> clusters: ",length(clusters),"\n")
  cat(">>> leaves outside clusters: ",length(leaves),"\n")
  cat(">>> leaves inside clusters: ",length(pred_tot),"\n")
  
  cl.num = NULL
  for (i in seq_along(clusters)) cl.num = c(cl.num,length(clusters[[i]]$nodes))
  cl.num = sort(cl.num)
  cat(">>> distribution of number of nodes per cluster \n")
  print(quantile(cl.num))
  cat(">>> mean:",mean(cl.num)," - sd:",sd(cl.num),"\n")
  
  ## dataset generation 
  cat(">>> dataset generation ... \n")
  Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_docproc2.csv" , sep='') , stringsAsFactors = F))
  Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_docproc2.csv" , sep='') , stringsAsFactors = F))
  
  nr_ds = floor(mean(cl.num))
  for (i in 1:nr_ds) {
    cat(">>> ds:",i,"/",nr_ds,"..\n") 
    
    nodes_in_clusters = rep(NA,length(clusters))
    for (cl_j in seq_along(clusters)) {
      nodes_in_clusters[cl_j] = getNode(clusters[[cl_j]]$nodes,i)
    }
    
    all_nodes = c(leaves,nodes_in_clusters)
    
    Xtrain_i = Xtrain[,all_nodes]
    Xtest_i = Xtest[,all_nodes]
    
    cat(">>> writing on disk ... \n")
    write.csv(Xtrain_i,
              quote=FALSE, 
              file=paste(ff.getPath("CBDB"),"Xtrain_",i,".csv",sep=''),
              row.names=FALSE)
    write.csv(Xtest_i,
              quote=FALSE, 
              file=paste(ff.getPath("CBDB"),"Xtest_",i,".csv",sep=''),
              row.names=FALSE)
  }
} else if (TASK == "K-MEANS") { 
  cat(">>> processing task:",TASK," ... \n")
  
  N.CENTERS.MAX = 10 
  
  ## dataset generation 
  cat(">>> loading data ... \n")
  Xtrain = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_docproc2_pca_95var.csv" , sep='') , stringsAsFactors = F))
  Xtest = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_docproc2_pca_95var.csv" , sep='') , stringsAsFactors = F))
  
  data = rbind(Xtrain,Xtest)
  rm(list=c("Xtrain","Xtest"))
  
  cat(">>> computing withinss ...\n")
  wss = rep(NA,N.CENTERS.MAX)
  for (i in 1:N.CENTERS.MAX) wss[i] <- sum(kmeans(x = data , centers = i , nstart = 5)$withinss)
  
  ## jpeg
  jpeg(filename=paste(ff.getPath("elab"),"k-means.jpg",sep=''))
  plot(1:N.CENTERS.MAX, wss, type="l", xlab="Number of Clusters",ylab="Within groups sum of squares")
  dev.off()
  
  ## find best number of clusters 
  wss_delta = rep(0,length(wss)-1)
  for (i in 1:length(wss)-1) wss_delta[i] = (wss[i+1] - wss[i])/wss[i]
  wss_delta = wss_delta[2:length(wss_delta)] ## remove 1st clusters 
  n_cluters = which(wss_delta == min(wss_delta))+1
  cat(">>> best number of clusters:",n_cluters," ...\n")
  
  ## K1 
  kopt = kmeans(x = data , centers = n_cluters , nstart = 5)
  cat(">>> writing on disk ... \n")
  write.csv(data.frame( ID = seq_along(kopt$cluster), cluster=kopt$cluster),
            quote=FALSE, 
            file=paste(ff.getPath("elab"),"k-means_1.csv",sep=''),
            row.names=FALSE)
  
  ## K3
  if (n_cluters > 2) {
    cat(">>> making  K3  ... \n")
    n_cluters_1 = n_cluters -1 
    n_cluters_3 = n_cluters + 1 
    
    kopt_1 = kmeans(x = data , centers = n_cluters_1 , nstart = 5)
    kopt_3 = kmeans(x = data , centers = n_cluters_3 , nstart = 5)
    
    cat(">>> writing on disk ... \n")
    write.csv(data.frame( ID = seq_along(kopt$cluster), cluster_1=kopt_1$cluster , cluster=kopt$cluster , cluster_3=kopt_3$cluster),
              quote=FALSE, 
              file=paste(ff.getPath("elab"),"k-means_3.csv",sep=''),
              row.names=FALSE)
  } else {
    cat(">>> optimal number of clusters is 2 --> impossible to make K3 \n")
  }
}




