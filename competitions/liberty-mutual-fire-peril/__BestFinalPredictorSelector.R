library(quantreg)
library(data.table)
library(glmnet)
library(class)
library(caret)

getData4Analysis = function(myData.tab) {
  myData = as.data.frame.matrix(myData.tab) 
  ## set NAs
  myData$var1 = ifelse(myData$var1 == "Z" , NA , myData$var1)
  myData$var2 = ifelse(myData$var2 == "Z" , NA , myData$var2)
  myData$var3 = ifelse(myData$var3 == "Z" , NA , myData$var3)
  myData$var4 = ifelse(myData$var4 == "Z" , NA , myData$var4)
  myData$var5 = ifelse(myData$var5 == "Z" , NA , myData$var5)
  myData$var6 = ifelse(myData$var6 == "Z" , NA , myData$var6)
  myData$var7 = ifelse(myData$var7 == "Z" , NA , myData$var7)
  myData$var8 = ifelse(myData$var8 == "Z" , NA , myData$var8)
  myData$var9 = ifelse(myData$var9 == "Z" , NA , myData$var9)
  
  ## set correct classes for regression 
  myData$var1 = as.numeric(myData$var1)
  myData$var2 = as.factor(myData$var2)
  myData$var3 = as.factor(myData$var3)
  
  ## TODO BETTER: perdi l'informazione sul secondo livello  
  #myData$var4_4 = factor(myData$var4 , ordered = T)
  myData$var4 = factor(myData$var4 )
  #myData$var4 = factor( ifelse(is.na(myData$var4), NA , substring(myData$var4 , 1 ,1) ) , ordered = T)
  #myData$var4 = factor( ifelse(is.na(myData$var4), NA , substring(myData$var4 , 1 ,1) ) )
  
  myData$var5 = as.factor(myData$var5)
  myData$var6 = as.factor(myData$var6)
  myData$var7 = as.numeric(myData$var7)
  myData$var8 = as.numeric(myData$var8)
  myData$var9 = as.factor(myData$var9)
  myData$dummy = as.factor(myData$dummy)
  
  if (! is.null(myData$target) ) {
    myData$target_0 = factor(ifelse(myData$target == 0,0,1))
  }
  
  ## garbage collector 
  gc() 
  
  ## return myData
  myData
}

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


getBasePath = function () {
  ret = ""
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"
  
  if (file.exists(base.path1))  {
    ret = base.path1
  } else {
    ret = base.path2
  }
  ret
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


##############  Loading data sets (train, test, sample) ... 
train.fn = "train.csv"
test.fn = "test.csv"
sampleSub.fn = "sampleSubmission.csv"

train.tab = fread(paste(getBasePath(),train.fn,sep="") , header = TRUE , sep=","  )
test.tab = fread(paste(getBasePath(),test.fn,sep="") , header = TRUE , sep=","  )

## data formatting 
train = getData4Analysis(train.tab)
test = getData4Analysis(test.tab)

train.bkp = train 
test.bkp = test 

dim(train)
dim(test)

cat("==========================> Finding best predictors for regression problem ... ")

l = getBestPredictors(train[ , -c(1,2,303)] , response = train$target , test , 
                      polynomialDegree = 1 , NAthreshold = 0.05, 
                      pValueAdjust = F, verbose = T )
BestPredictorTrainIndex = l[[1]]
BestPredictorTestIndex = l[[2]]

cat("on train set, found ", length(BestPredictorTrainIndex)  , "predictors:" , 
    paste(colnames(train[ , -c(1,2,303)]) [BestPredictorTrainIndex] , collapse=" " ) , " \n") 
cat("on test set, found ", length(BestPredictorTestIndex)  , "predictors:" , 
    paste(colnames(test) [BestPredictorTestIndex] , collapse=" " ) , " \n") 

ytrain = train$target
Xtrain = train[ , -c(1,2,303)] [,BestPredictorTrainIndex]
Xtest = test[,BestPredictorTestIndex]

## describe 
library(Hmisc)
describe(x = ytrain)
describe(x = Xtrain)
describe(x = Xtest)

write.csv(ytrain,quote=F,row.names=F,file=paste0(getBasePath(),"ytrain_reg.csv"))
write.csv(Xtrain,quote=F,row.names=F,file=paste0(getBasePath(),"Xtrain_reg.csv"))
write.csv(Xtest,quote=F,row.names=F,file=paste0(getBasePath(),"Xtest_reg.csv"))

cat("==========================> Finding best predictors for classification problem ... ")

l = getBestPredictors(train[ , -c(1,2,303)] , response = train$target_0 , test , 
                      polynomialDegree = 1 , NAthreshold = 0.05, 
                      pValueAdjust = T, pValueAdjustMethod = "bonferroni", verbose = T )
BestPredictorTrainIndex = l[[1]]
BestPredictorTestIndex = l[[2]]

cat("on train set, found ", length(BestPredictorTrainIndex)  , "predictors:" , 
    paste(colnames(train[ , -c(1,2,303)]) [BestPredictorTrainIndex] , collapse=" " ) , " \n") 
cat("on test set, found ", length(BestPredictorTestIndex)  , "predictors:" , 
    paste(colnames(test) [BestPredictorTestIndex] , collapse=" " ) , " \n") 

ytrain = train$target_0
Xtrain = train[ , -c(1,2,303)] [,BestPredictorTrainIndex]
Xtest = test[,BestPredictorTestIndex]

## describe 
library(Hmisc)
describe(x = ytrain)
describe(x = Xtrain)
describe(x = Xtest)

write.csv(ytrain,quote=F,row.names=F,file=paste0(getBasePath(),"ytrain_class.csv"))
write.csv(Xtrain,quote=F,row.names=F,file=paste0(getBasePath(),"Xtrain_class.csv"))
write.csv(Xtest,quote=F,row.names=F,file=paste0(getBasePath(),"Xtest_class.csv"))


