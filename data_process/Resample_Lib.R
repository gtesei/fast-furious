

verify.kfolds = function(k,folds,dataset,event.level) {
  correct = T
  for(j in 1:k) {  
    ## y 
    dataset.train = dataset[folds != j]
    dataset.xval = dataset[folds == j]
    
    if (sum(dataset.train == event.level ) == 0 | 
          sum(dataset.xval == event.level ) == 0  ) {
      correct = F
      break
    }
  }
  return(correct)
}

kfolds = function(k,data.length) {
  k = min(k,data.length)
  folds = rep(NA,data.length)
  labels = 1:data.length
  st = floor(data.length/k)
  al_labels = NULL
  for (s in 1:k) {
    x = NULL
    if (is.null(al_labels))
      x = sample(labels,st)
    else
      x = sample(labels[-al_labels],st)
    
    folds[x] = s
    if (is.null(al_labels))
      al_labels = x
    else
      al_labels = c(al_labels,x)
  }
  ss = 1
  for (s in 1:length(folds)){
    if (is.na(folds[s])) {
      folds[s] = ss
      ss = ss + 1
    } 
  }
  folds
}

trainAndPredict.kfold.reg.wallmart = function(k,traindata,traindata.y,RegModels,controlObject) {
  source(paste0( getBasePath("process") , "/Regression_Lib.R"))
  
  .grid = data.frame(store = c(st) , 
                     item = c(it) , 
                     test.num = c(dim(testdata)[1]),
                     all0s=c(F) )
  tmp = data.frame(matrix( 0 , 1 ,  length(RegModels) ))
  colnames(tmp) = RegModels
  .grid = cbind(.grid , tmp)
  
  ####### training and predicting <<<<<<<<<<<<<<
  k = 5
  folds = kfolds(k,dim(traindata)[1])
  
  perf.kfold = data.frame(matrix(rep(-1,(k*length(RegModels))),k,length(RegModels)))
  colnames(perf.kfold) = RegModels
  
  for(j in 1:k) {  
    if (verbose) cat("--k-fold:: ",j, "/",k , "\n")
    traindata.train <- traindata[ folds != j,]
    traindata.y.train = traindata.y[folds != j]
    
    traindata.xval <- traindata[folds == j,]
    traindata.y.xval = traindata.y[folds == j]
    
    ###
    for ( mo in 1:length(RegModels))  {
      if (verbose) cat("Trying ", RegModels[mo] , " ... ")
      model.label = RegModels[mo]
      
      pred = tryCatch({ 
        reg.trainAndPredict( traindata.y.train , 
                             traindata.train , 
                             traindata.xval , 
                             model.label , 
                             controlObject, 
                             best.tuning = F)
      } , error = function(err) { 
        print(paste("ERROR:  ",err))
        NULL
      })
      
      if(! is.null(pred)) { 
        perf.kfold[j,mo] = RMSE(pred = pred, obs = traindata.y.xval)
        if (verbose) cat("RMSE = ", perf.kfold[j,mo] , "\n")
      } else {
        perf.kfold[j,mo] = 1000000000 ## RMSE
        if (verbose) cat("(fake) RMSE = ", perf.kfold[j,mo] , "\n")
      }
      
    } ### end of model shot    
  } ### end of k-fold 
  
  #### results 
  for ( mo in 1:length(RegModels))  {
    .grid[1,(4+mo)] = mean(perf.kfold[,mo])
  }
  .grid$best.perf = min(.grid[1,(4+(1:length(RegModels)))])
  model.idx = which(.grid[1,(4+(1:length(RegModels)))] == .grid$best.perf)
  .grid$best.model = RegModels[model.idx]
  
  list(RegModels[model.idx],.grid,perf.kfold)
}

trainAndPredict.kfold.reg = function(k,traindata,traindata.y,RegModels,controlObject) {
  source(paste0( getBasePath("process") , "/Regression_Lib.R"))
  
  .grid = data.frame( train.num = c(dim(traindata)[1])  )
  tmp = data.frame(matrix( 0 , 1 ,  length(RegModels) ))
  colnames(tmp) = RegModels
  .grid = cbind(.grid , tmp)
  
  ####### training and predicting <<<<<<<<<<<<<<
  folds = kfolds(k,dim(traindata)[1])
  
  perf.kfold = data.frame(matrix(rep(-1,(k*length(RegModels))),k,length(RegModels)))
  colnames(perf.kfold) = RegModels
  
  for(j in 1:k) {  
    if (verbose) cat("--k-fold:: ",j, "/",k , "\n")
    traindata.train <- traindata[ folds != j,]
    traindata.y.train = traindata.y[folds != j]
    
    traindata.xval <- traindata[folds == j,]
    traindata.y.xval = traindata.y[folds == j]
    
    ###
    for ( mo in 1:length(RegModels))  {
      if (verbose) cat("Trying ", RegModels[mo] , " ... ")
      model.label = RegModels[mo]
      
      pred = tryCatch({ 
        reg.trainAndPredict( traindata.y.train , 
                             traindata.train , 
                             traindata.xval , 
                             model.label , 
                             controlObject, 
                             best.tuning = F)
      } , error = function(err) { 
        print(paste("ERROR:  ",err))
        NULL
      })
      
      if(! is.null(pred)) { 
        perf.kfold[j,mo] = RMSE(pred = pred, obs = traindata.y.xval)
        if (verbose) cat("RMSE = ", perf.kfold[j,mo] , "\n")
      } else {
        perf.kfold[j,mo] = 1000000000 ## RMSE
        if (verbose) cat("(fake) RMSE = ", perf.kfold[j,mo] , "\n")
      }
      
    } ### end of model shot    
  } ### end of k-fold 
  
  #### results 
  for ( mo in 1:length(RegModels))  {
    .grid[1,(1+mo)] = mean(perf.kfold[,mo])
  }
  .grid$best.perf = min(.grid[1,(1+(1:length(RegModels)))])
  model.idx = which(.grid[1,(1+(1:length(RegModels)))] == .grid$best.perf)
  .grid$best.model = RegModels[model.idx]
  
  list(RegModels[model.idx],.grid,perf.kfold)
}

trainAndPredict.kfold.class = function(k,traindata,
                                       traindata.y,
                                       fact.sign = 'preict', 
                                       ClassModels,controlObject, 
                                       verbose = T , 
                                       doPlot = F) {
  
  source(paste0( getBasePath("process") , "/Classification_Lib.R"))
  
  .grid = data.frame( train.num = c(dim(traindata)[1])  )
  tmp = data.frame(matrix( 0 , 1 ,  length(ClassModels) ))
  colnames(tmp) = ClassModels
  .grid = cbind(.grid , tmp)
  
  ####### training and predicting <<<<<<<<<<<<<<
  
  folds = kfolds(k,dim(traindata)[1])
  while ( ! verify.kfolds(k,folds,traindata.y,fact.sign) ) {
    if (verbose) cat("--k-fold:: generated bad folds :: retrying ... \n")
    folds = kfolds(k,dim(traindata)[1])
  }
  
  perf.kfold = data.frame(matrix(rep(-1,(k*length(ClassModels))),k,length(ClassModels)))
  colnames(perf.kfold) = ClassModels
  
  for(j in 1:k) {  
    if (verbose) cat("--k-fold:: ",j, "/",k , "\n")
    traindata.train <- traindata[ folds != j,]
    traindata.y.train = traindata.y[folds != j]
    
    traindata.xval <- traindata[folds == j,]
    traindata.y.xval = traindata.y[folds == j]
    
    ###
    for ( mo in 1:length(ClassModels))  {
      if (verbose) cat("Trying ", ClassModels[mo] , " ... ")
      model.label = ClassModels[mo]
      
      l = tryCatch({ 
        class.trainAndPredict ( traindata.y.train , 
                                          traindata.train , 
                                          traindata.xval , 
                                          fact.sign = fact.sign , 
                                          model.label , 
                                          controlObject, 
                                          best.tuning = F, 
                                          verbose = verbose)
        
      } , error = function(err) { 
        print(paste("ERROR:  ",err))
        NULL
      })
      
      if(! is.null(l)) {  
        pred.prob.train = l[[1]]
        pred.train = l[[2]] 
        pred.prob.test = l[[3]]
        pred.test = l[[4]]
        
        auc = measure.class ( pred.prob.train , 
                              pred.prob.test , 
                              pred.train , 
                              pred.test,
                              traindata.y.train, 
                              traindata.y.xval,
                              fact.sign = fact.sign , 
                              verbose = verbose, 
                              doPlot = doPlot,
                              label = model.label)
        
        perf.kfold[j,mo] = auc
        if (verbose) cat("AUC = ", perf.kfold[j,mo] , "\n")
      } else {
        perf.kfold[j,mo] = 0 ## AUC
        if (verbose) cat("(fake) AUC = ", perf.kfold[j,mo] , "\n")
      }
      
    } ### end of model shot    
  } ### end of k-fold 
  
  #### results 
  for ( mo in 1:length(RegModels))  {
    .grid[1,(1+mo)] = mean(perf.kfold[,mo])
  }
  .grid$best.perf = max(.grid[1,(1+(1:length(RegModels)))])
  model.idx = which(.grid[1,(1+(1:length(RegModels)))] == .grid$best.perf)
  .grid$best.model = ClassModels[model.idx]
  
  list(ClassModels[model.idx],.grid,perf.kfold)
}


