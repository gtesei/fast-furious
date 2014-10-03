library(quantreg)
library(data.table)
library(glmnet)
library(class)
library(caret)


pScore <- function(x, y)
{
  numX <- length(unique(x))
  if(numX > 2)
  {
    ## With many values in x, compute a t-test
    out <- t.test(x ~ y)$p.value
  } else {
    ## For binary predictors, test the odds ratio == 1 via
    ## Fisher's Exact Test
    out <- fisher.test(factor(x), y)$p.value
  }
  out
}

pCorrection <- function (score, x, y)
{
  ## The options x and y are required by the caret package
  ## but are not used here
  score <- p.adjust(score, "bonferroni")
  ## Return a logical vector to decide which predictors
  ## to retain after the filter
  keepers <- (score <= 0.05)
  keepers
}

getPerfOnTestSet = function(x , y , k = 4) {
  reg.type = NA
  perf = NA
  
  if ( class(y) == "factor" & length(levels(y)) > 2 ) stop("TODO Multinomial support.")
  
  ## type casting and understanding stat test 
  if (class(x) == "integer") x = as.numeric(x)
  if (class(y) == "integer") y = as.numeric(y)
  
  if ( class(x) == "factor" & class(y) == "numeric" ) {
    # C -> Q
    reg.type = "LINEAR_REG"
  } else if (class(x) == "factor" & class(y) == "factor" ) {
    # C -> C
    reg.type = "LOGISTIC_REG"
  } else if (class(x) == "numeric" & class(y) == "numeric" ) {
    reg.type = "LINEAR_REG"
  }  else {
    # Q -> C 
    reg.type = "LOGISTIC_REG"
  }
  
  y = y[! is.na(x)]
  x = x[! is.na(x)]
  ## k-fold cross validation 
  folds = kfolds(k,length(x))
  cv=rep(x = -1 , k)
  
  if (reg.type == "LINEAR_REG") {
    for(j in 1:k) {  
      train.df = data.frame(y = y[folds != j] , x = x[folds != j])
      test.df = data.frame(x = x[folds == j])
      fit = lm(y ~ x , data=train.df)
      pred = predict(fit , test.df)
      cv[j]= mean(abs((y[folds == j] - pred)))
    }
  } else { #LOGISTIC_REG
    for(j in 1:k) {  
      train.df = data.frame(y = y[folds != j] , x = x[folds != j])
      test.df = data.frame(x = x[folds == j])
      fit = glm(y ~ x , data=train.df , family=binomial)
      pred.probs = predict(fit , test.df , type = "response")
      label0 = rownames(contrasts(y))[1]
      label1 = rownames(contrasts(y))[2]
      pred = ifelse(pred.probs > 0.5 , label1 , label0)
      cv[j]= mean( y[folds == j] == pred  )
    }
  }
  
  perf=mean(cv)
  perf
}

getPvalueFeatures = function(response,features , p = 1 , computePerfOnTrainSet = F , 
                             pValueAdjust = T, pValueAdjustMethod = "default" , 
                             verbose = F) {
  
  label.formula = rep(NA, dim(features)[2])
  pValue <- rep(NA, dim(features)[2])
  is.na <- rep(NA, dim(features)[2])
  perf.xval <- rep(NA, dim(features)[2])
  
  for (i in 1:(dim(features)[2])) {
    if (verbose) cat(i,"...")
    if ( p == 1) {
      label.formula[i] = colnames(features[i])
      pValue[i] <- getPvalueTypeIError(x = features[,i], y = response)
      if (computePerfOnTrainSet) {
        perf.xval[i] = getPerfOnTestSet(x = features[,i], y = response)
      }
    } else {
      ## label 
      if (p == 2) label.formula[i] = paste0(paste0("I(",colnames(features[i])),"^2)")
      if (p == 3) label.formula[i] = paste0(paste0("I(",colnames(features[i])),"^3)")
      if (p == 4) label.formula[i] = paste0(paste0("I(",colnames(features[i])),"^4)")
      if (p == 5) label.formula[i] = paste0(paste0("I(",colnames(features[i])),"^5)")
      if (p >  5) stop("p > 5 not supported.")
      
      ## pvalue , perf.xval
      if (class(features[,i]) == "factor") {
        pValue[i] = NA
        perf.xval[i] = NA
      } else {
        x.poly = features[,i]^p
        pValue[i] = getPvalueTypeIError(x = x.poly, y = response)
        if (computePerfOnTrainSet) {
          perf.xval[i] = getPerfOnTestSet(x = x.poly, y = response)
        }
      }
    }
    
    is.na[i] = sum(is.na(features[,i])) / length(features[,i]) 
  }
  if (verbose) cat("\n")
  
  ##p-value adjusting 
  if (length(pValue) > 1 & pValueAdjust) {
    if (verbose) {
      cat("adjusting p-value for multiple comparisons ... \n")
      cat("before adjusting: ", pValue, "\n")
    }
    
    if(pValueAdjustMethod == "default") {
      pValue = p.adjust(p = pValue)
    } else {
      pValue = p.adjust(p = pValue,method = pValueAdjustMethod) 
    }
    
    if (verbose) {
      cat("after adjusting: ", pValue, "\n")
    }
  }
  
  is.significant = ifelse(pValue < 0.05,T,F)
  data.frame(label = label.formula, pValue , is.significant , is.na , perf.xval)
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

getPredictorsMinPvalue = function (predPvalues, data , th = 0.05 , verbose = F) {
  var.name = NULL
  var.index = NULL
  
  if (verbose) cat ("processing ")
  
  for (i in 1:dim(predPvalues)[1]) {
    if ( ! predPvalues[i,]$is.significant ) break 
    
    var = as.character(predPvalues[i,]$label)
    if (verbose) {
      cat ( as.character(var),"...")
    }
    idx = as.numeric( grep(pattern = var  , x = colnames(data)) )
    
    var.name.tmp = c(var.name , var)
    var.index.tmp = c(var.index , idx)
    
    sumNA = getNasRows(data[,var.index.tmp])
    th.tmp = sumNA / dim(data)[1]
    
    if (th.tmp < th) {
      var.name = var.name.tmp
      var.index = var.index.tmp
    } else {
      next 
    }
  }
  if (verbose) cat("\n")
  list(var.name,var.index)
}

rowNa = function (row) {
  ifelse (sum(is.na(row)) > 0 , 1,0)
}
getNasRows = function (myData) {
  ret = 0
  if ( ! is.data.frame(myData) ) {
    ret = sum( is.na(myData) )
  } else { 
    ret = apply(myData,1, rowNa   )
    ret = sum(ret)
  }
  ret
}

getBestPredictors = function(train , response , test , 
                             polynomialDegree = 1 , NAthreshold = 0.05, 
                             pValueAdjust = T, pValueAdjustMethod = "default", 
                             verbose = F) {
  train.bkp = train 
  ## finding predictors with min p-values
  if (verbose) cat("finding min p-value predictors ... \n")
  predictors.reg.linear = getPvalueFeatures( features = train , response = response , p = polynomialDegree , 
                                             pValueAdjust = pValueAdjust, pValueAdjustMethod = pValueAdjustMethod, 
                                             verbose = verbose)
  predictors.reg.linear = predictors.reg.linear[order(predictors.reg.linear$pValue,decreasing = F),]
  
  l = getPredictorsMinPvalue(predPvalues = predictors.reg.linear , data = train , th = NAthreshold , verbose = verbose)
  var.name = l[[1]]
  var.index = l[[2]]
  
  if (verbose) {
    cat("min p-value predictors found:",var.name," \n")
    cat("reducing pair-wise correlations ... \n")
  }
  
  train = train[, var.index ]
  train = na.omit(train)
  
  ## reducing pair-wise correlations 
  CategoricalIdx = as.numeric(which( vapply(train, is.factor, logical(1) ) ) )
  CategoricalPredictors = train[,CategoricalIdx]
  train = train[,-CategoricalIdx] ## l'analisi di correlazioni non la faccio sulle variabili categoriche 
  PredToDel = findCorrelation(cor( train )) 
  if (verbose) {
    cat("removing ", paste(colnames(train) [PredToDel] , collapse=" " ) , " ... \n ")
  }
  train = train[,-PredToDel]
  train = cbind(train,CategoricalPredictors)
  
  ## indexes on initial train set / test set 
  IndexTrain = grep(paste(colnames(train),collapse="|") , colnames(train.bkp) ) 
  IndexTest = grep(paste(colnames(train),collapse="|") , colnames(test) ) 
  list(IndexTrain,IndexTest)
}

transfrom4Skewness = function (train,test) {
  library(e1071)
  
  no = which(as.numeric(lapply(train,is.factor)) == 1)
  noT = which(as.numeric(lapply(test,is.factor)) == 1)
  
  skewValuesTrainBefore <- apply(train[,-no], 2, skewness)
  cat("----> skewValues  before transformation (train) \n\n")
  print(skewValuesTrainBefore)
  
  skewValuesTestBefore <- apply(test[,-noT], 2, skewness)
  cat("----> skewValues  before transformation (test): \n")
  print(skewValuesTestBefore)
  
  idx = (1:15)[-no]
  for (i in idx) {
    varname = colnames(train)[i]
    cat("processing ",varname," ... \n")
    inTest = (sum(colnames(test) == varname) == 1)
    tr = BoxCoxTrans(train[,i])
    print(tr)
    if (! is.na(tr$lambda) ) {
      cat("tranforming train ... \n")
      newVal = predict(tr,train[,i])
      train[,i] = newVal
      cat("skewness after transformation (train): " , skewness(train[,i]), "  \n")
      if (inTest) {
        idxTest = which((colnames(test) == varname) == T )
        cat("tranforming test ... \n")
        newVal = predict(tr,test[,idxTest])
        test[,idxTest] = newVal
        cat("skewness after transformation (test): " , skewness(train[,i]), "  \n")
      } 
    }
  } 
  
  cat("---->  skewValues  before transformation (train) \n")
  print(skewValuesTrainBefore)
  skewValues <- apply(train[,-no], 2, skewness)
  cat("\n---->  skewValues  after transformation (train):  \n")
  print(skewValues)
  
  cat("\n\n\n---->  skewValues  before transformation (test) \n")
  print(skewValuesTestBefore)
  skewValues <- apply(test[,-noT], 2, skewness)
  cat("\n---->  skewValues  after transformation (test):  \n")
  print(skewValues)
  
  list(train,test) 
}


