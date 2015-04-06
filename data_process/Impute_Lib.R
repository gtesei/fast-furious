library(data.table) 
library(caret)

#### supported imputing models 
All.RegModels.impute = c("Average" , "Mode", "LinearReg","KNN_Reg", "PLS_Reg" , "Ridge_Reg" , "SVM_Reg", "Cubist_Reg") 
All.ClassModels.impute = c("Mode" , "SVMClass") 

#####
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
getPvalueTypeIError = function(x,y) {
  test = NA
  pvalue = NA
  
  ## type casting and understanding stat test 
  if (class(x) == "integer") x = as.numeric(x)
  if (class(y) == "integer") y = as.numeric(y)
  
  if ( class(x) == "factor" & class(y) == "numeric" ) {
    # C -> Q
    test = "ANOVA"
  } else if (class(x) == "factor" & class(y) == "factor" ) {
    # C -> C
    test = "CHI-SQUARE"
  } else if (class(x) == "numeric" & class(y) == "numeric" ) {
    test = "PEARSON"
  }  else {
    # Q -> C 
    # it performs anova test x ~ y 
    test = "ANOVA"
    tmp = x 
    x = y 
    y = tmp 
  }
  
  ## performing stat test and computing p-value
  if (test == "ANOVA") {                
    test.anova = aov(y~x)
    pvalue = summary(test.anova)[[1]][["Pr(>F)"]][1]
  } else if (test == "CHI-SQUARE") {    
    test.chisq = chisq.test(x = x , y = y)
    pvalue = test.chisq$p.value
  } else {                             
    ###  PEARSON
    test.corr = cor.test(x =  x , y =  y)
    pvalue = test.corr$p.value
  }
  
  pvalue
}

buildDecisionMatrix = function (data) {
  predictors.name = colnames(data)
  DecisionMatrix = data.frame(predictor = predictors.name)
  
  tmp = matrix(rep(NA,length(predictors.name)) , nrow = length(predictors.name) , ncol = length(predictors.name) )
  colnames(tmp) = paste0("pvalue_",predictors.name)
  tmp = as.data.frame(tmp)
  DecisionMatrix = cbind(DecisionMatrix , tmp)
  
  DecisionMatrix$NAs = as.numeric( apply(data,2,FUN = function(x) sum(is.na(x))) )
  
  tmp = matrix(rep(NA,length(predictors.name)) , nrow = length(predictors.name) , ncol = length(predictors.name) )
  colnames(tmp) = paste0("NA_",predictors.name)
  tmp = as.data.frame(tmp)
  DecisionMatrix = cbind(DecisionMatrix , tmp)
  
  pnum = length(predictors.name)
  #pvalue
  for (i in 1:pnum) {
    for (j in 1:pnum) {
      if (i == j ) {
        DecisionMatrix[i,(j+1)] = 0 
      } else {
        tmp = na.omit(data[,c(i,j)])
        DecisionMatrix[i,(j+1)] <- getPvalueTypeIError(x = tmp[,2], y = tmp[,1])
      }
    }
  }
  #NA
  for (i in 1:pnum) {
    for (j in 1:pnum) {
      if (i == j ) {
        DecisionMatrix[i,(j+pnum+2)] = -1 
      } else {
        toImpute = sum(is.na(data[,i]))
        if (toImpute == 0){
          DecisionMatrix[i,(j+pnum+2)] = -1 
        } else {
          DecisionMatrix[i,(j+pnum+2)] <- sum(is.na(data[,i]) & is.na(data[,j]) )  / sum(is.na(data[,i]))
        }
      }
    }
  }
  
  DecisionMatrix
}

findImputePredictors = function(DecisionMatrix,data) {
  
  predictors.name = DecisionMatrix$predictor
  pnum = length(predictors.name)
  
  ImputePredictors = data.frame(predictor = DecisionMatrix$predictor, NAs = DecisionMatrix$NAs , 
                                need.impute = ifelse(DecisionMatrix$NAs > 0 , T , F)  , 
                                predictors = rep(NA,pnum) , predictorIndex = rep(NA,pnum))
  
  for (i in 1:pnum) {
    if(ImputePredictors[i,]$need.impute) {
      candidates = which(DecisionMatrix[i,((pnum+3):(2*pnum+2))] == 0)
      stat.sign = which( DecisionMatrix[i,1+candidates ] < 0.05)
      predIdx = candidates[stat.sign] 
      ImputePredictors[i,]$predictors = paste(predictors.name[predIdx] , collapse = "-")
      ImputePredictors[i,]$predictorIndex = paste(predIdx , collapse = "-")
    } 
  }
  
  ImputePredictors$is.factor = as.logical(vapply(data, is.factor, logical(1) ))
  ImputePredictors
}

trainAndPredict = function( YtrainingSet , XtrainingSet , testSet , model ) {
  pred = NULL
  
  ptm <- proc.time()
  controlObject <- trainControl(method = "repeatedcv", number = 10 , repeats = 5)
  if (model == "LinearReg") {   ### LinearReg
    fit <- train(y = YtrainingSet, x = XtrainingSet , method = "lm", trControl = controlObject)
    pred = as.numeric( predict(fit , testSet )  ) 
  } else if (model == "KNN_Reg") {  ### KNN_Reg
    fit <- train(y = YtrainingSet, x = XtrainingSet , method = "knn", preProc = c("center", "scale"), 
                 tuneGrid = data.frame(.k = 1:10),
                 trControl = controlObject)
    pred = as.numeric( predict(fit , testSet )  )
  } else if (model == "PLS_Reg") {  ### PLS_Reg
    fit <- train(y = YtrainingSet, x = XtrainingSet ,
                     method = "pls",
                     ## The default tuning grid evaluates
                     ## components 1... tuneLength
                     tuneLength = 20,
                     trControl = controlObject,
                     preProc = c("center", "scale"))
    pred = as.numeric( predict(fit , testSet )  )
  } else if (model == "Ridge_Reg") {  ### Ridge_Reg
    ridgeGrid <- data.frame(.lambda = seq(0, .1, length = 15))
    fit <- train(y = YtrainingSet, x = XtrainingSet ,
                           method = "ridge",
                           ## Fir the model over many penalty values
                           tuneGrid = ridgeGrid,
                           trControl = controlObject,
                           ## put the predictors on the same scale
                           preProc = c("center", "scale"))
    pred = as.numeric( predict(fit , testSet )  )
  } else if (model == "SVM_Reg") {  ### SVM_Reg
    fit <- train(y = YtrainingSet, x = XtrainingSet ,
                       method = "svmRadial",
                       tuneLength = 15,
                       preProc = c("center", "scale"),
                       trControl = controlObject)
    pred = as.numeric( predict(fit , testSet )  )
  } else if (model == "Cubist_Reg") {  ### Cubist_Reg
    cubistGrid <- expand.grid(.committees = c(1, 5, 10, 50, 75, 100),
                              .neighbors = c(0, 1, 3, 5, 7, 9))
    fit <- train(y = YtrainingSet, x = XtrainingSet ,
                 method = "cubist",
                 tuneGrid = cubistGrid,
                 trControl = controlObject)
    pred = as.numeric( predict(fit , testSet )  )
  } else if (model == "Average") {  ### Average 
    
    ltset = ifelse( ! is.null(dim(testSet)) , dim(testSet) , length(testSet) )
    pred = rep(mean(YtrainingSet),ltset)
    
  } else if (model == "Mode") {  ### Mode 
    
    ltset = ifelse( ! is.null(dim(testSet)) , dim(testSet) , length(testSet) )
    pred = rep(Mode(YtrainingSet),ltset)
    
  } else if (model == "SVMClass") { ###  SVMClass
    fit <- train(y = YtrainingSet, x = XtrainingSet , method = "svmRadial", preProc = c("center", "scale"), 
                 tuneLength = 8 , trControl = controlObject)
    pred = predict(fit , testSet )
  } else {
    stop("unrecognized model.")
  }
  tm = proc.time() - ptm
  cat("Time elapsed:",tm,"\n")
  
  pred
}

findBestImputeModel = function(data, ImputePredictors , 
                               RegModels = All.RegModels.impute, 
                               ClassModels = All.ClassModels.impute , 
                               verbose = T , debug = F  ) {
  ## completing matrix 
  pnum = dim(data)[2] 
  modelNumber = max(length(RegModels),length(ClassModels))
  for (j in 1:modelNumber) {
    tmp = data.frame(x = rep(NA,pnum) , y=rep(NA,pnum))
    colnames(tmp) = c(paste0("Mod_",j) , paste0("Perf_Mod_",j))
    ImputePredictors = cbind(ImputePredictors , tmp)
  }
  
  ImputePredictors$winner = rep(NA,pnum)
  ImputePredictors$best_perf = rep(NA,pnum)
  
  ## sampling data 
  for (i in 1:pnum) {
    if (ImputePredictors[i,]$need.impute) {
      models = NULL
      if (ImputePredictors[i,]$is.factor) {
        models = ClassModels
      } else {
        models = RegModels
      }
      
      if (verbose) {
        cat("processing " , as.character(ImputePredictors[i,]$predictor) , " ...\n")
      }
      ImputeTestIdx = which(is.na(data[,i])) 
      ImputePredIdx = c(i, as.numeric(unlist( strsplit( ImputePredictors[i,]$predictorIndex , "-" ) ) ) )
      
      ImputeXTest = data[ImputeTestIdx,ImputePredIdx]
      ImputeXtrain = data[-ImputeTestIdx,ImputePredIdx]
      ImputeXtrain = na.omit(ImputeXtrain)
      
      if (length(ImputePredIdx) == 1) {
        ## degenerate case, i.e. there're no predictors 
        ## using average for regression and mode for classification 
        cat("There're no predictors available ... using average/mode ... \n")
        
        ## split train set 
        ImputeXtrain.train = NULL
        ImputeXtrain.test = NULL
        split = tryCatch({ createDataPartition(y = ImputeXtrain , p = 3/4 , list = F)
                          } , error = function(err) { 
                             print(paste("ERROR:  ",err))
                             NULL
                           })
        if(! is.null(split)) { 
          ImputeXtrain.train = ImputeXtrain[split]
          ImputeXtrain.test = ImputeXtrain[-split]
        } else {
          split = sample(x = 1:length(ImputeXtrain) , length(ImputeXtrain) * 3 / 4 , replace = F)
          ImputeXtrain.train = ImputeXtrain[split]
          ImputeXtrain.test = ImputeXtrain[-split]
        }
        
        ###
        pred = -1
        perf = -1
        model.deg = ""
        if (ImputePredictors[i,]$is.factor) {
          model.deg = "Mode"
          pred = rep(Mode(ImputeXtrain.train), length(ImputeXtrain.test))
          perf = sum(pred == ImputeXtrain.test) / length(ImputeXtrain.test)
          if (verbose) cat("Accuracy = ",perf,"\n")
        } else {
          model.deg = "Average"
          pred = rep(mean(ImputeXtrain.train) , length(ImputeXtrain.test))
          perf = RMSE(pred = pred, obs = ImputeXtrain.test)
          if (verbose) cat("RMSE = ", perf , "\n")
        }
        
        ###
        for (m in 1:length(models)) { 
          ImputePredictors[i,6+(m-1)*2+1] = model.deg
          ImputePredictors[i,6+(m-1)*2+2] = perf
        }
        
      } else {
      ## ordinary case, i.e. predictors >= 1 
        
      ## split train set 
      ImputeXtrain.train = NULL
      ImputeXtrain.test = NULL
      split = tryCatch({  createDataPartition(y = ImputeXtrain[,1] , p = 3/4 , list = F)
                        } , error = function(err) { 
                          print(paste("ERROR:  ",err))
                          NULL
                        })  
      if(! is.null(split)) { 
        ImputeXtrain.train = ImputeXtrain[split,]
        ImputeXtrain.test = ImputeXtrain[-split,]
      } else {
        split = sample(x = 1:length(ImputeXtrain[,1]) , length(ImputeXtrain[,1]) * 3 / 4 , replace = F)
        ImputeXtrain.train = ImputeXtrain[split,]
        ImputeXtrain.test = ImputeXtrain[-split,]
      }                    
      
      y.ImputeXtrain.train = ImputeXtrain.train[,1]
      train.ImputeXtrain.train = ImputeXtrain.train[,-1]
      
      y.ImputeXtrain.test = ImputeXtrain.test[,1]
      test.ImputeXtrain.test = ImputeXtrain.test[,-1]
      
      for (m in 1:length(models)) {
        if (verbose) {
          cat("trying " , models[m] , " ... ")
        }
        
        ##### trying models 
        pred = tryCatch({ 
          trainAndPredict ( y.ImputeXtrain.train , train.ImputeXtrain.train , test.ImputeXtrain.test , models[m] )
        } , error = function(err) { 
          print(paste("ERROR:  ",err))
          NULL
        })
        if(! is.null(pred)) { 
          perf = -1
          if (ImputePredictors[i,]$is.factor) {
            perf = sum(pred == y.ImputeXtrain.test) / length(pred)
            if (verbose) cat("Accuracy = ",perf,"\n")
          } else {
            perf = RMSE(pred = pred, obs = y.ImputeXtrain.test)
            if (verbose) cat("RMSE = ", perf , "\n")
          }
          ImputePredictors[i,6+(m-1)*2+1] = models[m]
          ImputePredictors[i,6+(m-1)*2+2] = perf
        } else {
          ### setting fake performances 
          ImputePredictors[i,6+(m-1)*2+1] = models[m]
          if (ImputePredictors[i,]$is.factor) {
            perf = 0 ## 0 accuracy 
            ImputePredictors[i,6+(m-1)*2+2] = perf
            if (verbose) cat("setting (fake) Accuracy = ", perf , "\n")
          } else {
            perf = 1000000000 ## RMSE
            ImputePredictors[i,6+(m-1)*2+2] =  perf
            if (verbose) cat("setting (fake)  RMSE = ", perf , "\n")
          }
        }
      }  #### end trying models 
      
     } ## end case >= 1 predictors 
    }
  }  #### end predictors 
  
  ## the winner is ... 
  for (i in 1:pnum) {
    if (ImputePredictors[i,]$need.impute) {
      models = NULL
      if (ImputePredictors[i,]$is.factor) {
        models = ClassModels
      } else {
        models = RegModels
      }
      bestModel = ""
      bestPerf = -1
      for (m in 1:length(models)) {
        #model = models[m]
        model = ImputePredictors[i,6+2*(m-1)+1]
        perf = ImputePredictors[i,6+2*(m-1)+2]
        if (bestPerf == -1) {
          bestModel = model
          bestPerf = perf 
        } else if (ImputePredictors[i,]$is.factor & bestPerf < perf) {
          bestModel = model
          bestPerf = perf
        } else if (bestPerf > perf) {
          bestModel = model
          bestPerf = perf
        }
      }
      ## setting winner model 
      ImputePredictors[i,]$winner = bestModel
      ImputePredictors[i,]$best_perf = bestPerf
    }
  }
  
  ImputePredictors
} 

predictAndImpute = function(data,ImputePredictors,
                            verbose = T , debug = F) {
  ## predicting 
  pnum = dim(data)[2]
  for (i in 1:pnum) {
    if (ImputePredictors[i,]$need.impute) {
      if (verbose) {
        cat("predicting on " , as.character(ImputePredictors[i,]$predictor) , " with "
            , ImputePredictors[i,]$winner, " ...\n")
      }
      ImputeTestIdx = which(is.na(data[,i])) 
      ImputePredIdx = c(i, as.numeric(unlist( strsplit( ImputePredictors[i,]$predictorIndex , "-" ) ) ) )
      ImputeXtest = data[ImputeTestIdx,ImputePredIdx]
      ImputeXtrain = data[-ImputeTestIdx,ImputePredIdx]
      ImputeXtrain = na.omit(ImputeXtrain)
      
      y.ImputeXtrain = NULL
      train.ImputeXtrain = NULL
      test.ImputeXtest = NULL
      
      if (length(ImputePredIdx) > 1) {
        y.ImputeXtrain = ImputeXtrain[,1]
        train.ImputeXtrain = ImputeXtrain[,-1]
        test.ImputeXtest = ImputeXtest[,-1]
      } else {
          y.ImputeXtrain = ImputeXtrain
          train.ImputeXtrain = ImputeXtrain 
          test.ImputeXtest = ImputeXtest
        }
      
      ## predicting ...
      pred = trainAndPredict ( y.ImputeXtrain , train.ImputeXtrain , test.ImputeXtest , ImputePredictors[i,]$winner )
      
      ## imputing ...
      data[ImputeTestIdx,i] = pred 
    }
  }
  
  data
}

blackGuido = function(data , 
                             RegModels = All.RegModels.impute, 
                             ClassModels = All.ClassModels.impute, 
                             verbose = T , 
                             debug = F ) {
  if (verbose) {
    cat("building decision matrix ... \n")
  }
  ## building decision matrix on data 
  DecisionMatrix = buildDecisionMatrix(data)
  if (verbose) {
    cat("****************** DecisionMatrix ****************** \n")
    print(DecisionMatrix)
  }
  
  ImputePredictors = findImputePredictors(DecisionMatrix,data)
  if (verbose) {
    cat("finding best models ... \n")
  }
  
  ## finding best models ...
  ImputePredictors = findBestImputeModel (data, ImputePredictors , 
                                          RegModels = RegModels , 
                                          ClassModels = ClassModels , 
                                          verbose = verbose , debug = debug  )
  
  if (verbose) {
    cat("****************** ImputePredictors ****************** \n")
    print(ImputePredictors)
  }
  
  ## predicting and imputing 
  data = predictAndImpute (data,ImputePredictors, verbose = verbose , debug = debug)
  
  list(data,ImputePredictors,DecisionMatrix)
}

prepare4Octave = function(data , verbose = T , debug = F ) {
  if (verbose) {
    cat("building decision matrix ... \n")
  }
  ## building decision matrix on data 
  DecisionMatrix = buildDecisionMatrix(data)
  if (verbose) {
    cat("****************** DecisionMatrix ****************** \n")
    print(DecisionMatrix)
  }
  
  ## models 
  RegModels = c("LinearReg","SVR") 
  ClassModels = c("LogisticReg","SVC")
  
  ImputePredictors = findImputePredictors(DecisionMatrix,data)
  
  ## completing matrix 
  pnum = dim(data)[2] 
  modelNumber = max(length(RegModels),length(ClassModels))
  for (j in 1:modelNumber) {
    tmp = data.frame(x = rep(NA,pnum) , y=rep(NA,pnum))
    tmp$x = as.character(tmp$x)
    tmp$y = as.numeric(tmp$y)
    colnames(tmp) = c(paste0("Mod_",j) , paste0("Perf_Mod_",j))
    ImputePredictors = cbind(ImputePredictors , tmp)
  }
  
  ImputePredictors$winner = as.character(rep(NA,pnum))
  ImputePredictors$best_perf = as.numeric(rep(NA,pnum))
  
  ImputePredictors
}


