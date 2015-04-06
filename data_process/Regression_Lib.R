library(caret)

#### supported regression models 
All.RegModels = c("Average" , "Mode",  
                  "LinearReg", "RobustLinearReg", 
                  "PLS_Reg" , "Ridge_Reg" , "Enet_Reg" , 
                  "KNN_Reg", "SVM_Reg", "BaggedTree_Reg", "RandomForest_Reg", "Cubist_Reg") 

reg.trainAndPredict = function( YtrainingSet , XtrainingSet , 
                            testSet , 
                            model.label , 
                            controlObject, 
                            best.tuning = F) {
  pred = NULL
  
  ptm <- proc.time()
  if (model.label == "LinearReg") {   ### LinearReg
    fit <- train(y = YtrainingSet, x = XtrainingSet , method = "lm", trControl = controlObject)
    
    pred = as.numeric( predict(fit , testSet )  ) 
  } else if (model.label == "RobustLinearReg") {   ### RobustLinearReg
    fit <- train(y = YtrainingSet, x = XtrainingSet , method = "rlm", preProcess="pca", trControl = controlObject)
    
    pred = as.numeric( predict(fit , testSet )  ) 
  } else if (model.label == "KNN_Reg") {  ### KNN_Reg
    fit <- train(y = YtrainingSet, x = XtrainingSet , method = "knn", preProc = c("center", "scale"), 
                 tuneGrid = data.frame(.k = 1:10),
                 trControl = controlObject)
    
    pred = as.numeric( predict(fit , testSet )  )
  } else if (model.label == "PLS_Reg") {  ### PLS_Reg
    .tuneGrid = expand.grid(.ncomp = 1:10)
    if (best.tuning)  .tuneGrid = expand.grid(.ncomp = 1:30)
    
    fit <- train(y = YtrainingSet, x = XtrainingSet ,
                 method = "pls",
                 tuneGrid = .tuneGrid , 
                 trControl = controlObject)
    
    pred = as.numeric( predict(fit , testSet )  )
  } else if (model.label == "Ridge_Reg") {  ### Ridge_Reg
    ridgeGrid <- data.frame(.lambda = seq(0, .1, length = 15))
    if (best.tuning) data.frame(.lambda = seq(0, .1, length = 25))
      
    fit <- train(y = YtrainingSet, x = XtrainingSet ,
                 method = "ridge",
                 tuneGrid = ridgeGrid,
                 trControl = controlObject)
    
    pred = as.numeric( predict(fit , testSet )  )
  } else if (model.label == "Enet_Reg") {  ### Enet_Reg
    enetGrid <- expand.grid(.lambda = c(0, 0.01, .1), .fraction = seq(.05, 1, length = 20))
    if (best.tuning) enetGrid <- expand.grid(.lambda = c(0, 0.01,.1,.5,.8), .fraction = seq(.05, 1, length = 30))
    
    fit <- train(y = YtrainingSet, x = XtrainingSet ,
                 method = "enet",
                 tuneGrid = enetGrid)
    
    pred = as.numeric( predict(fit , testSet )  )
  } else if (model.label == "SVM_Reg") {  ### SVM_Reg
    fit <- train(y = YtrainingSet, x = XtrainingSet ,
                 method = "svmRadial",
                 tuneLength = 15,
                 trControl = controlObject)
    
    pred = as.numeric( predict(fit , testSet )  )
  } else if (model.label == "BaggedTree_Reg") {  ### BaggedTree_Reg
    fit <- train(y = YtrainingSet, x = XtrainingSet ,
                 method = "treebag",
                 trControl = controlObject)
    
    pred = as.numeric( predict(fit , testSet )  )
  } else if (model.label == "RandomForest_Reg") {  ### RandomForest_Reg
    gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, by = 2),
                           .n.trees = seq(100, 1000, by = 50),
                           .shrinkage = c(0.01, 0.1))
    
    fit <- train(y = YtrainingSet, x = XtrainingSet ,
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 trControl = controlObject, 
                 verbose = F)
    
    pred = as.numeric( predict(fit , testSet )  )
  } else if (model.label == "Cubist_Reg") {  ### Cubist_Reg
    cubistGrid <- expand.grid(.committees = c(1, 5, 10, 50, 75, 100),
                              .neighbors = c(0, 1, 3, 5, 7, 9))
    
    fit <- train(y = YtrainingSet, x = XtrainingSet ,
                 method = "cubist",
                 tuneGrid = cubistGrid,
                 trControl = controlObject)
    
    pred = as.numeric( predict(fit , testSet )  )
  } else if (model.label == "Average") {  ### Average 
    
    ltset = ifelse( ! is.null(dim(testSet)) , dim(testSet) , length(testSet) )
    pred = rep(mean(YtrainingSet),ltset)
    
  } else if (model.label == "Mode") {  ### Mode 
    
    ltset = ifelse( ! is.null(dim(testSet)) , dim(testSet) , length(testSet) )
    pred = rep(Mode(YtrainingSet),ltset)
    
  } else {
    stop("unrecognized model.label.")
  }
  tm = proc.time() - ptm
  cat("Time elapsed:",tm,"\n")
  
  pred
}

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}