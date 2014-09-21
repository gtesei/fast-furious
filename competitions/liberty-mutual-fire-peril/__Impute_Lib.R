library(data.table)

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/liberty-mutual-fire-peril/"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/liberty-mutual-fire-peril/"
  } else {
    stop("unrecognized type.")
  }
  
  if (file.exists(base.path1))  {
    ret = base.path1
  } else {
    ret = base.path2
  }
  
  ret
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
      candidates = which(DecisionMatrix[i,((pnum+3):(2*pnum+2)),] == 0)
      stat.sign = which( DecisionMatrix[i,1+candidates,] < 0.05)
      predIdx = candidates[stat.sign] 
      ImputePredictors[i,]$predictors = paste(predictors.name[predIdx] , collapse = ",")
      ImputePredictors[i,]$predictorIndex = paste(predIdx , collapse = ",")
    } 
  }
  
  ImputePredictors$is.factor = as.logical(vapply(data, is.factor, logical(1) ))
  ImputePredictors
}

trainAndPredict = function( form , trainingSet , testSet , model ) {
  pred = rep(NA,dim(testSet)[1])
  
  ptm <- proc.time()
  controlObject <- trainControl(method = "cv", number = 10)
  if (model == "LinearReg") {
    fit = linearReg <- train(form, data = trainingSet, method = "lm", trControl = controlObject)
    pred = as.numeric( predict(fit , testSet )  ) 
  } else if (model == "KNN_Reg") {
    fit <- train(form , data=trainingSet, method = "knn", preProc = c("center", "scale"), 
                 tuneGrid = data.frame(.k = 1:10),
                 trControl = controlObject)
    pred = as.numeric( predict(fit , testSet )  )
  } else if (model == "SVMClass") {
    fit <- train(form , data=trainingSet, method = "svmRadial", preProc = c("center", "scale"), 
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
                               RegModels = c("LinearReg","KNN_Reg") , ClassModels = c("SVMClass") , 
                               verbose = T , debug = F  ) {
  ## completing matrix 
  pnum = dim(data)[2] 
  modelNumber = max(length(RegModels),length(ClassModels))
  for (j in 1:modelNumber) {
    tmp = data.frame(x = rep(NA,pnum) , y=rep(NA,pnum))
    colnames(tmp) = c(paste0("Mod_",j) , paste0("Perf_Mod_",j))
    ImputePredictors = cbind(ImputePredictors , tmp)
  }
  
  ## sampling data 
  for (i in 1:pnum) {
    ## TODO resempling 10 volte 
    if (ImputePredictors[i,]$need.impute) {
      if (verbose) {
        cat("processing " , as.character(ImputePredictors[i,]$predictor) , " ...\n")
      }
      ImputeTestIdx = which(is.na(data[,i])) 
      ImputePredIdx = c(i, as.numeric(unlist( strsplit( ImputePredictors[i,]$predictorIndex , "," ) ) ) )
      ImputeXTest = data[ImputeTestIdx,ImputePredIdx]
      ImputeXtrain = data[-ImputeTestIdx,ImputePredIdx]
      ImputeXtrain = na.omit(ImputeXtrain)
      
      ## reduce train set in debug mode
      if (debug) {
        ImputeXtrain = ImputeXtrain[1:100,]
      }
      
      ## split train set 
      split = createDataPartition(y = ImputeXtrain[,1] , p = 0.6 , list = F)
      ImputeXtrain.train = ImputeXtrain[split,]
      ImputeXtrain.test = ImputeXtrain[-split,]
      
      modFormula = as.formula( paste0(ImputePredictors[i,]$predictor," ~ .") )
      models = NULL
      if (ImputePredictors[i,]$is.factor) {
        models = ClassModels
      } else {
        models = RegModels
      }
      
      for (m in 1:length(models)) {
        if (verbose) {
          cat("trying " , models[m] , " ...\n")
        }
        pred = trainAndPredict ( modFormula , ImputeXtrain.train , ImputeXtrain.test , models[m] )
        perf = -1
        if (ImputePredictors[i,]$is.factor) {
          perf = sum(pred == ImputeXtrain.test[,1]) / length(pred)
        } else {
          perf = RMSE(pred = pred, obs = ImputeXtrain.test[,1])
        }
        ImputePredictors[i,6+(m-1)*2+1] = models[m]
        ImputePredictors[i,6+(m-1)*2+2] = perf
      }
    }
  }
  
  ## the winner is ... 
  ImputePredictors$winner = rep(NA,pnum)
  ImputePredictors$best_perf = rep(NA,pnum)
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
        model = models[m]
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
        cat("predicting on " , as.character(ImputePredictors[i,]$predictor) , " ...\n")
      }
      ImputeTestIdx = which(is.na(data[,i])) 
      ImputePredIdx = c(i, as.numeric(unlist( strsplit( ImputePredictors[i,]$predictorIndex , "," ) ) ) )
      ImputeXtest = data[ImputeTestIdx,ImputePredIdx]
      ImputeXtrain = data[-ImputeTestIdx,ImputePredIdx]
      ImputeXtrain = na.omit(ImputeXtrain)
      
      ## reduce train set in debug mode
      if (debug) {
        ImputeXtrain = ImputeXtrain[1:100,]
      }
      
      ## predicting 
      bestModel = ImputePredictors[i,]$winner
      modFormula = as.formula( paste0(ImputePredictors[i,]$predictor," ~ .") )
      pred = trainAndPredict ( modFormula , ImputeXtrain , ImputeXtest , bestModel )
      
      ## imputing 
      data[ImputeTestIdx,i] = pred 
    }
  }
  
  data
}

imputeFastFurious = function(data , verbose = T , debug = F ) {
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
                                          RegModels = c("LinearReg","KNN_Reg") , ClassModels = c("SVMClass") , 
                                          verbose = verbose , debug = debug  )
  
  if (verbose) {
    cat("****************** ImputePredictors ****************** \n")
    print(ImputePredictors)
  }
  
  ## predicting and imputing 
  data = predictAndImpute (data,ImputePredictors, verbose = verbose , debug = debug)
  
  list(data,ImputePredictors,DecisionMatrix)
}

##############  Loading data sets (train, test, sample) ... 
source(paste0(getBasePath("code") , "__BestFinalPredictorSelector_Lib.R"))

Xtrain = as.data.frame(fread(paste0(getBasePath(), "Xtrain_reg.csv" ) , header = TRUE , sep=","  ))
Xtest = as.data.frame(fread(paste0(getBasePath(), "Xtest_reg.csv" ) , header = TRUE , sep="," ) )
ytrain = as.data.frame(fread(paste0(getBasePath(), "ytrain_reg.csv" ) , header = TRUE   ))


Xtrain$var4 = as.factor(Xtrain$var4)
Xtrain$dummy = as.factor(Xtrain$dummy)

Xtest$var4 = as.factor(Xtest$var4)
Xtest$dummy = as.factor(Xtest$dummy)

## imputing Xtest 
l = imputeFastFurious (data = Xtest , verbose = T , debug = F)
Xtest.imputed = l[[1]]
ImputePredictors = l[[2]]
DecisionMatrix = l[[3]]

write.csv(Xtest.imputed,quote=F,row.names=F,file=paste0(getBasePath(),"Xtest_reg_imputed.csv"))
write.csv(ImputePredictors,quote=F,row.names=F,file=paste0(getBasePath(),"Xtest_reg_imputeModel___ImputePredictors.csv"))


