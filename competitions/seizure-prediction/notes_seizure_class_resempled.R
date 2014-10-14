
## notes_seizure_R
library(caret)
library(Hmisc)
library(data.table)
library(verification)
library(pROC)
library(kernlab)

getBasePath = function (type = "data" , ds="") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/seizure-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/seizure-prediction/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/data_pre_process"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/data_pre_process/"
  } else {
    stop("unrecognized type.")
  }
  
  if (file.exists(base.path1))  {
    ret = paste0(base.path1,"/")
  } else {
    ret = base.path2
  }
  
  if (ds != "" ) {
    ret = paste0(paste0(ret,ds),"_digest/")
  }
  ret
} 


split.random = function (Xtrain_mean_sd,Xtrain_quant,
             Xtrain_mean_sd.scaled,Xtrain_quant.scaled,
             Xtrain_mean_sd.reduced,Xtrain_quant.reduced,
             ytrain.cat,verbose=F) {
  
  
  ##set.seed(429494444)
  sd = sample(1:429494444, 1)
  if (verbose) cat("** Random split - using seed =",sd, " ... \n")
  set.seed(sd)
  forTraining <- createDataPartition(ytrain[,1], p = 3/4)[[1]]
  
  ## full 
  Xtrain_mean_sd.train <- Xtrain_mean_sd[ forTraining,]
  Xtrain_mean_sd.xval <- Xtrain_mean_sd[-forTraining,]
  Xtrain_quant.train <- Xtrain_quant[ forTraining,]
  Xtrain_quant.xval <- Xtrain_quant[-forTraining,]
  
  ## scaled 
  Xtrain_mean_sd.scaled.train <- Xtrain_mean_sd.scaled[ forTraining,]
  Xtrain_mean_sd.scaled.xval <- Xtrain_mean_sd.scaled[-forTraining,]
  Xtrain_quant.scaled.train <- Xtrain_quant.scaled[ forTraining,]
  Xtrain_quant.scaled.xval <- Xtrain_quant.scaled[-forTraining,]
  
  ## reduced 
  Xtrain_mean_sd.reduced.train <- Xtrain_mean_sd.reduced[ forTraining,]
  Xtrain_mean_sd.reduced.xval <- Xtrain_mean_sd.reduced[-forTraining,]
  Xtrain_quant.reduced.train <- Xtrain_quant.reduced[ forTraining,]
  Xtrain_quant.reduced.xval <- Xtrain_quant.reduced[-forTraining,]
  
  ## y 
  ytrain.cat.train = ytrain.cat[forTraining]
  ytrain.cat.xval = ytrain.cat[-forTraining]
  
  list(Xtrain_mean_sd.train,Xtrain_mean_sd.xval,Xtrain_quant.train,Xtrain_quant.xval,
       Xtrain_mean_sd.scaled.train,Xtrain_mean_sd.scaled.xval,Xtrain_quant.scaled.train,Xtrain_quant.scaled.xval,
       Xtrain_mean_sd.reduced.train,Xtrain_mean_sd.reduced.xval,Xtrain_quant.reduced.train,Xtrain_quant.reduced.xval,
       ytrain.cat.train,ytrain.cat.xval,sd)
}

predictAndMeasure = function(model,model.label,model.id,
                             trainingData, ytrain,
                             testData, yxval,
                             tm, grid = NULL,verbose=F, doPlot=F) {
  
  ## predictions - probs 
  pred.prob.train = predict(model , trainingData , type = "prob")[,'preict'] 
  pred.prob.xval = predict(model , testData , type = "prob")[,'preict']
  
  ## predictions - factors  
  pred.train = predict(model , trainingData )
  pred.xval = predict(model , testData )
  
  ## accuracy 
  acc.xval.all0 = sum(factor(rep('interict',length(yxval)) , levels = levels(yxval) ) == yxval) / length(yxval)
  acc.train = sum(ytrain == pred.train) / length(ytrain)
  acc.xval = sum(yxval == pred.xval) / length(yxval) 
  
  ## ROC 
  rocCurve <- pROC::roc(response = ytrain, predictor = pred.prob.train, levels = levels(ytrain) )
  roc.train = as.numeric( pROC::auc(rocCurve) )
  
  rocCurve <- pROC::roc(response = yxval, predictor = pred.prob.xval , levels = levels(yxval) )
  roc.xval = as.numeric( pROC::auc(rocCurve) )
  
  roc.train.2 = roc.area(as.numeric(ytrain == 'preict') , pred.prob.train )$A
  roc.xval.2 = roc.area(as.numeric(yxval == 'preict') , pred.prob.xval )$A
  
  roc.xval.min = min(roc.xval.2,roc.xval)
  
  ## logging 
  if (verbose) cat("******************* ", model.label, " <<" , model.id ,  ">>  --  Time elapsed(sec.):",tm[[3]], " \n")
  if (verbose) cat("** acc.train =",acc.train, " -  acc.xval =",acc.xval, " - acc.xval.all0 =",acc.xval.all0, "  \n")
  if (verbose) cat("** roc.train =",roc.train," -  roc.xval =",roc.xval,"  \n")
  if (verbose) cat("** roc.train.2 =",roc.train.2," -  roc.xval.2 =",roc.xval.2,"  \n")
  if (verbose) cat("** roc.xval.min =",roc.xval.min, " \n")
  
  ## poltting 
  if (doPlot) {
    plot(rocCurve, legacy.axes = TRUE , main = paste(model.label  
                                                     , " - acc.xval=",acc.xval
                                                     , " - roc.xval=" ,roc.xval  
                                                     , " - roc.xval.2=",roc.xval.2
                                                     , collapse ="" )   )
  }
  
  ### perf.grid 
  perf.grid = NULL
  if (is.null(grid)) { 
    perf.grid = data.frame(predictor = c(model.label) , model.id = c(model.id), 
                           acc.train = c(acc.train) , acc.xval = c(acc.xval) , acc.xval.all0 = c(acc.xval.all0), 
                           roc.train = c(roc.train) , roc.xval =c(roc.xval),
                           roc.train.2 = c(roc.train.2) , roc.xval.2 =c(roc.xval.2),
                           roc.xval.min = roc.xval.min , 
                           time = c(tm[[3]]))
  } else {
    .grid = data.frame(predictor = c(model.label) , model.id = c(model.id),
                       acc.train = c(acc.train) , acc.xval = c(acc.xval) , acc.xval.all0 = c(acc.xval.all0), 
                       roc.train = c(roc.train) , roc.xval =c(roc.xval),
                       roc.train.2 = c(roc.train.2) , roc.xval.2 =c(roc.xval.2),
                       roc.xval.min = roc.xval.min , 
                       time = c(tm[[3]]) )
    perf.grid = rbind(grid, .grid)
  }
  
  perf.grid
}

createValidationGrid = function (cases) {
  grid = matrix(rep(-1,cases*66),nrow=66,ncol=(5+cases)) 
  grid.df = as.data.frame(grid)
  
  seeds = sample( (1:429494444) , cases )
  
  cases.names = paste0("seed_",seeds)
  colnames(grid.df) = c("model" , "model.id" , "min", "max", "avg", cases.names)
  grid.df[,2] = 1:66
  
  list( grid.df ,  seeds ) 
}

######################################################## CONSTANTS 
LOGISTIC_REG_MEAN_SD = 1 
LOGISTIC_REG_QUANTILES = 2
LOGISTIC_REG_MEAN_SD_SCALED = 3
LOGISTIC_REG_QUANTILES_SCALED = 4
LOGISTIC_REG_MEAN_SD_REDUCED = 5
LOGISTIC_REG_QUANTILES_REDUCED = 6

LDA_MEAN_SD = 7
LDA_QUANTILES = 8
LDA_MEAN_SD_SCALED = 9 
LDA_QUANTILES_SCALED= 10 
LDA_MEAN_SD_REDUCED = 11
LDA_REG_QUANTILES_REDUCED = 12 

PLSDA_MEAN_SD = 13
PLSDA_QUANTILES = 14
PLSDA_MEAN_SD_SCALED = 15 
PLSDA_QUANTILES_SCALED= 16 
PLSDA_MEAN_SD_REDUCED = 17
PLSDA_REG_QUANTILES_REDUCED = 18 

PM_MEAN_SD = 19
PM_QUANTILES = 20
PM_MEAN_SD_SCALED = 21 
PM_QUANTILES_SCALED= 22 
PM_MEAN_SD_REDUCED = 23
PM_REG_QUANTILES_REDUCED = 24

NSC_MEAN_SD = 25
NSC_QUANTILES = 26
NSC_MEAN_SD_SCALED = 27 
NSC_QUANTILES_SCALED= 28 
NSC_MEAN_SD_REDUCED = 29
NSC_REG_QUANTILES_REDUCED = 30

NN_MEAN_SD = 31
NN_QUANTILES = 32
NN_MEAN_SD_SCALED = 33 
NN_QUANTILES_SCALED= 34 
NN_MEAN_SD_REDUCED = 35
NN_QUANTILES_REDUCED = 36

SVM_MEAN_SD = 37
SVM_QUANTILES = 38
SVM_MEAN_SD_SCALED = 39 
SVM_QUANTILES_SCALED= 40 
SVM_MEAN_SD_REDUCED = 41
SVM_QUANTILES_REDUCED = 42

KNN_MEAN_SD = 43
KNN_QUANTILES = 44
KNN_MEAN_SD_SCALED = 45 
KNN_QUANTILES_SCALED= 46 
KNN_MEAN_SD_REDUCED = 47
KNN_QUANTILES_REDUCED = 48

CLASS_TREE_MEAN_SD = 49
CLASS_TREE_QUANTILES = 50
CLASS_TREE_MEAN_SD_SCALED = 51 
CLASS_TREE_QUANTILES_SCALED= 52 
CLASS_TREE_MEAN_SD_REDUCED = 53
CLASS_TREE_QUANTILES_REDUCED = 54

BOOSTED_TREE_MEAN_SD = 55
BOOSTED_TREE_QUANTILES = 56
BOOSTED_TREE_MEAN_SD_SCALED = 57 
BOOSTED_TREE_QUANTILES_SCALED= 58 
BOOSTED_TREE_MEAN_SD_REDUCED = 59
BOOSTED_TREE_QUANTILES_REDUCED = 60

BAGGING_TREE_MEAN_SD = 61
BAGGING_TREE_QUANTILES = 62
BAGGING_TREE_MEAN_SD_SCALED = 63 
BAGGING_TREE_QUANTILES_SCALED= 64 
BAGGING_TREE_MEAN_SD_REDUCED = 65
BAGGING_TREE_QUANTILES_REDUCED = 66

######################################################## MAIN LOOP 
NUMBER_CASES = 10 
l= createValidationGrid(NUMBER_CASES)
validation.grid = l[[1]]
seeds = l[[2]]

############# model selection ... 
verbose = T
doPlot = F 

# controlObject <- trainControl(method = "repeatedcv", repeats = 5, number = 10 , 
#                               summaryFunction = twoClassSummary , classProbs = TRUE)

controlObject <- trainControl(method = "boot", number = 50 , 
                              summaryFunction = twoClassSummary , classProbs = TRUE)

##dss = c("Dog_1","Dog_2","Dog_3","Dog_4","Dog_5","Patient_1","Patient_2")
dss = c("Patient_2")


for (ds in dss) {
  
  cat("|---------------->>> processing data set <<",ds,">> ..\n")
  
  for (cs in 1:NUMBER_CASES) {
    
    cat("|-------------------------------->>> case ",cs,"/"  ,NUMBER_CASES ,  " ..\n")

  ######### loading data sets ...
  Xtrain_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"Xtrain_mean_sd.zat",sep="") , header = F , sep=","  ))
  Xtrain_quant = as.data.frame( fread(paste(getBasePath(type = "data" , ds=ds),"Xtrain_quant.zat",sep="") , header = F , sep=","  ))
  
  Xtest_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"Xtest_mean_sd.zat",sep="") , header = F , sep=","  ))
  Xtest_quant = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"Xtest_quant.zat",sep="") , header = F , sep=","  ))
  
  ytrain = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"ytrain.zat",sep="") , header = F , sep=","  ))
  
  ## names 
  colnames(Xtrain_mean_sd)  = colnames(Xtest_mean_sd) = paste("fmeansd",rep(1:((dim(Xtrain_mean_sd)[2]))) , sep = "")
  colnames(Xtrain_quant) = colnames(Xtest_quant) = paste("fquant",rep(1:((dim(Xtrain_quant)[2]))) , sep = "")
  
  ytrain.cat = factor(ytrain[,2]) 
  levels(ytrain.cat) = c("interict","preict")
  
  ######### making train / xval set ...
  ### A. reduced+scaled  
  Xtest_quant.reduced = Xtrain_quant.reduced = NULL 
  Xtest_mean_sd.reduced = Xtrain_mean_sd.reduced = NULL 
  
  # rmoving high correlated predictors on Xtrain_quant
  PredToDel = findCorrelation(cor( Xtrain_quant )) 
  cat("PLS:: on Xtrain_quant removing ",length(PredToDel), " predictors: ",paste(colnames(Xtrain_quant) [PredToDel] , collapse=" " ) , " ... \n ")
  Xtest_quant.reduced =  Xtest_quant  [,-PredToDel]
  Xtrain_quant.reduced = Xtrain_quant [,-PredToDel]
  
  # rmoving high correlated predictors Xtrain_mean_sd
  PredToDel = findCorrelation(cor( Xtrain_mean_sd )) 
  cat("PLS:: on Xtrain_mean_sd removing ",length(PredToDel), " predictors: ",paste(colnames(Xtrain_mean_sd) [PredToDel] , collapse=" " ) , " ... \n ")
  Xtest_mean_sd.reduced =  Xtest_mean_sd  [,-PredToDel]
  Xtrain_mean_sd.reduced = Xtrain_mean_sd [,-PredToDel]
  
  # feature scaling 
  scale.reduced.mean_sd = preProcess(Xtrain_mean_sd.reduced,method = c("center","scale"))
  scale.reduced.quant = preProcess(Xtrain_quant.reduced,method = c("center","scale"))
  
  Xtest_mean_sd.reduced = predict(scale.reduced.mean_sd,Xtest_mean_sd.reduced)
  Xtrain_mean_sd.reduced = predict(scale.reduced.mean_sd,Xtrain_mean_sd.reduced)
  Xtest_quant.reduced = predict(scale.reduced.quant,Xtest_quant.reduced)
  Xtrain_quant.reduced = predict(scale.reduced.quant,Xtrain_quant.reduced)
  
  ### B. scaled only  
  scale.mean_sd = preProcess(Xtrain_mean_sd,method = c("center","scale"))
  scale.quant = preProcess(Xtrain_quant,method = c("center","scale"))
  
  Xtrain_mean_sd.scaled = predict(scale.mean_sd,Xtrain_mean_sd)
  Xtest_mean_sd.scaled = predict(scale.mean_sd,Xtest_mean_sd)
  Xtrain_quant.scaled = predict(scale.quant,Xtrain_quant)
  Xtest_quant.scaled = predict(scale.quant,Xtest_quant)
  
  #### partitioning into train , xval ... 
  l = split.random(Xtrain_mean_sd,Xtrain_quant,
               Xtrain_mean_sd.scaled,Xtrain_quant.scaled,
               Xtrain_mean_sd.reduced,Xtrain_quant.reduced,
               ytrain.cat,verbose)
  
  Xtrain_mean_sd.train = l[[1]] 
  Xtrain_mean_sd.xval = l[[2]]
  Xtrain_quant.train = l[[3]] 
  Xtrain_quant.xval = l[[4]]
  
  Xtrain_mean_sd.scaled.train = l[[5]] 
  Xtrain_mean_sd.scaled.xval = l[[6]]
  Xtrain_quant.scaled.train = l[[7]] 
  Xtrain_quant.scaled.xval = l[[8]] 
  
  Xtrain_mean_sd.reduced.train = l[[9]] 
  Xtrain_mean_sd.reduced.xval = l[[10]]
  Xtrain_quant.reduced.train = l[[11]] 
  Xtrain_quant.reduced.xval = l[[12]]
  
  ytrain.cat.train = l[[13]] 
  ytrain.cat.xval = l[[14]]
  
  ######################################################## Logistic regression 
  ## 1. LOGISTIC_REG_MEAN_SD
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.train , y = ytrain.cat.train , 
                       method = "glm", metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Logistic Reg (Mean sd)",LOGISTIC_REG_MEAN_SD,
                               Xtrain_mean_sd.train, ytrain.cat.train,
                               Xtrain_mean_sd.xval, ytrain.cat.xval,
                               tm, grid = NULL,verbose=verbose, doPlot=doPlot)

  
  validation.grid [validation.grid$model.id == LOGISTIC_REG_MEAN_SD,]$model = "Logistic Reg (Mean sd)"
  validation.grid [validation.grid$model.id == LOGISTIC_REG_MEAN_SD,(5+cs)] = perf.grid[perf.grid$model.id==LOGISTIC_REG_MEAN_SD,]$roc.xval.2  
  
  ## 2. LOGISTIC_REG_QUANTILES
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.train , y = ytrain.cat.train , 
                    method = "glm", metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Logistic Reg (Quant)",LOGISTIC_REG_QUANTILES,
                                           Xtrain_quant.train, ytrain.cat.train,
                                           Xtrain_quant.xval, ytrain.cat.xval,
                                           tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == LOGISTIC_REG_QUANTILES,]$model = "Logistic Reg (Quant)"
  validation.grid [validation.grid$model.id == LOGISTIC_REG_QUANTILES,(5+cs)] = perf.grid[perf.grid$model.id==LOGISTIC_REG_QUANTILES,]$roc.xval.2
  
  ## 3. LOGISTIC_REG_MEAN_SD_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.scaled.train , y = ytrain.cat.train , 
                                   method = "glm", metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Logistic Reg (Mean sd scaled)",LOGISTIC_REG_MEAN_SD_SCALED,
                                           Xtrain_mean_sd.scaled.train, ytrain.cat.train,
                                           Xtrain_mean_sd.scaled.xval, ytrain.cat.xval,
                                           tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == LOGISTIC_REG_MEAN_SD_SCALED,]$model = "Logistic Reg (Mean sd scaled)"
  validation.grid [validation.grid$model.id == LOGISTIC_REG_MEAN_SD_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==LOGISTIC_REG_MEAN_SD_SCALED,]$roc.xval.2
  
  ## 4. LOGISTIC_REG_QUANTILES_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.scaled.train , y = ytrain.cat.train , 
                                   method = "glm", metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Logistic Reg (Quant scaled)",LOGISTIC_REG_QUANTILES_SCALED,
                                           Xtrain_quant.scaled.train, ytrain.cat.train,
                                           Xtrain_quant.scaled.xval, ytrain.cat.xval,
                                           tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == LOGISTIC_REG_QUANTILES_SCALED,]$model = "Logistic Reg (Quant scaled)"
  validation.grid [validation.grid$model.id == LOGISTIC_REG_QUANTILES_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==LOGISTIC_REG_QUANTILES_SCALED,]$roc.xval.2

  ## 5. LOGISTIC_REG_MEAN_SD_REDUCED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.reduced.train , y = ytrain.cat.train , 
                                   method = "glm", metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Logistic Reg (Mean sd reduced)",LOGISTIC_REG_MEAN_SD_REDUCED,
                                           Xtrain_mean_sd.reduced.train, ytrain.cat.train,
                                           Xtrain_mean_sd.reduced.xval, ytrain.cat.xval,
                                           tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == LOGISTIC_REG_MEAN_SD_REDUCED,]$model = "Logistic Reg (Mean sd reduced)"
  validation.grid [validation.grid$model.id == LOGISTIC_REG_MEAN_SD_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==LOGISTIC_REG_MEAN_SD_REDUCED,]$roc.xval.2
  
  ## 6. LOGISTIC_REG_QUANTILES_REDUCED 
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.reduced.train , y = ytrain.cat.train , 
                                   method = "glm", metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Logistic Reg (Quant reduced)",LOGISTIC_REG_QUANTILES_REDUCED,
                                           Xtrain_quant.reduced.train, ytrain.cat.train,
                                           Xtrain_quant.reduced.xval, ytrain.cat.xval,
                                           tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == LOGISTIC_REG_QUANTILES_REDUCED,]$model = "Logistic Reg (Quant reduced)"
  validation.grid [validation.grid$model.id == LOGISTIC_REG_QUANTILES_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==LOGISTIC_REG_QUANTILES_REDUCED,]$roc.xval.2
  
  print(perf.grid) 
  
  ######################################################## Linear Discriminant Analysis  
  ## 3. MEAN_SD_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.scaled.train , y = ytrain.cat.train,  method = "lda", metric = "ROC" , trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"LDA (Mean sd scaled)",LDA_MEAN_SD_SCALED,
                                 Xtrain_mean_sd.scaled.train, ytrain.cat.train,
                                 Xtrain_mean_sd.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == LDA_MEAN_SD_SCALED,]$model = "LDA (Mean sd scaled)"
  validation.grid [validation.grid$model.id == LDA_MEAN_SD_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==LDA_MEAN_SD_SCALED,]$roc.xval.2
  
  ## 4. QUANTILES_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.scaled.train , y = ytrain.cat.train, method = "lda", metric = "ROC" , trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"LDA (Quant scaled)",LDA_QUANTILES_SCALED,
                                 Xtrain_quant.scaled.train, ytrain.cat.train,
                                 Xtrain_quant.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == LDA_QUANTILES_SCALED,]$model = "LDA (Quant scaled)"
  validation.grid [validation.grid$model.id == LDA_QUANTILES_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==LDA_QUANTILES_SCALED,]$roc.xval.2
  
  ## 5. MEAN_SD_REDUCED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.reduced.train , y = ytrain.cat.train , method = "lda", metric = "ROC" , trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"LDA (Mean sd reduced)",LDA_MEAN_SD_REDUCED,
                                 Xtrain_mean_sd.reduced.train, ytrain.cat.train,
                                 Xtrain_mean_sd.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == LDA_MEAN_SD_REDUCED,]$model = "LDA (Mean sd reduced)"
  validation.grid [validation.grid$model.id == LDA_MEAN_SD_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==LDA_MEAN_SD_REDUCED,]$roc.xval.2
  
  ## 6. QUANTILES_REDUCED 
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.reduced.train , y = ytrain.cat.train, method = "lda", metric = "ROC" , trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"LDA (Quant reduced)",LDA_REG_QUANTILES_REDUCED,
                                 Xtrain_quant.reduced.train, ytrain.cat.train,
                                 Xtrain_quant.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == LDA_REG_QUANTILES_REDUCED,]$model = "LDA (Mean sd reduced)"
  validation.grid [validation.grid$model.id == LDA_REG_QUANTILES_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==LDA_REG_QUANTILES_REDUCED,]$roc.xval.2
  
  print(perf.grid) 
  
  ######################################################## Partial Least Squares Discriminant Analysis
  ## 3. MEAN_SD_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.scaled.train , y = ytrain.cat.train,  method = "pls", 
                  tuneGrid = expand.grid(.ncomp = 1:10), metric = "ROC" , trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"PLSDA (Mean sd scaled)",PLSDA_MEAN_SD_SCALED,
                                 Xtrain_mean_sd.scaled.train, ytrain.cat.train,
                                 Xtrain_mean_sd.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == PLSDA_MEAN_SD_SCALED,]$model = "PLSDA (Mean sd scaled)"
  validation.grid [validation.grid$model.id == PLSDA_MEAN_SD_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==PLSDA_MEAN_SD_SCALED,]$roc.xval.2
  
  ## 4. QUANTILES_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.scaled.train , y = ytrain.cat.train, method = "pls", 
                  tuneGrid = expand.grid(.ncomp = 1:10), metric = "ROC" , trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"PLSDA (Quant scaled)",PLSDA_QUANTILES_SCALED,
                                 Xtrain_quant.scaled.train, ytrain.cat.train,
                                 Xtrain_quant.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == PLSDA_QUANTILES_SCALED,]$model = "PLSDA (Quant scaled)"
  validation.grid [validation.grid$model.id == PLSDA_QUANTILES_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==PLSDA_QUANTILES_SCALED,]$roc.xval.2
  
  ## 5. MEAN_SD_REDUCED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.reduced.train , y = ytrain.cat.train , method = "pls", 
                  tuneGrid = expand.grid(.ncomp = 1:10), metric = "ROC" , trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"PLSDA (Mean sd reduced)",PLSDA_MEAN_SD_REDUCED,
                                 Xtrain_mean_sd.reduced.train, ytrain.cat.train,
                                 Xtrain_mean_sd.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == PLSDA_MEAN_SD_REDUCED,]$model = "PLSDA (Mean sd reduced)"
  validation.grid [validation.grid$model.id == PLSDA_MEAN_SD_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==PLSDA_MEAN_SD_REDUCED,]$roc.xval.2
  
  ## 6. QUANTILES_REDUCED 
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.reduced.train , y = ytrain.cat.train, method = "pls", 
                  tuneGrid = expand.grid(.ncomp = 1:10), metric = "ROC" , trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"PLSDA (Quant reduced)",PLSDA_REG_QUANTILES_REDUCED,
                                 Xtrain_quant.reduced.train, ytrain.cat.train,
                                 Xtrain_quant.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == PLSDA_REG_QUANTILES_REDUCED,]$model = "PLSDA (Quant reduced)"
  validation.grid [validation.grid$model.id == PLSDA_REG_QUANTILES_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==PLSDA_REG_QUANTILES_REDUCED,]$roc.xval.2
  
  print(perf.grid) 
  
  ######################################################## Penalized Models 
  glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1), .lambda = seq(.01, .2, length = 40))
  
  ## 3. MEAN_SD_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.scaled.train , y = ytrain.cat.train, 
                  method = "glmnet", tuneGrid = glmnGrid, metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Penalized Models (Mean sd scaled)",PM_MEAN_SD_SCALED,
                                 Xtrain_mean_sd.scaled.train, ytrain.cat.train,
                                 Xtrain_mean_sd.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == PM_MEAN_SD_SCALED,]$model = "Penalized Models (Mean sd scaled)"
  validation.grid [validation.grid$model.id == PM_MEAN_SD_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==PM_MEAN_SD_SCALED,]$roc.xval.2
  
  ## 4. QUANTILES_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.scaled.train , y = ytrain.cat.train, 
                  method = "glmnet", tuneGrid = glmnGrid, metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Penalized Models (Quant scaled)",PM_QUANTILES_SCALED,
                                 Xtrain_quant.scaled.train, ytrain.cat.train,
                                 Xtrain_quant.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == PM_QUANTILES_SCALED,]$model = "Penalized Models (Quant scaled)"
  validation.grid [validation.grid$model.id == PM_QUANTILES_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==PM_QUANTILES_SCALED,]$roc.xval.2
  
  ## 5. MEAN_SD_REDUCED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.reduced.train , y = ytrain.cat.train , 
                  method = "glmnet", tuneGrid = glmnGrid, metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Penalized Models (Mean sd reduced)",PM_MEAN_SD_REDUCED,
                                 Xtrain_mean_sd.reduced.train, ytrain.cat.train,
                                 Xtrain_mean_sd.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == PM_MEAN_SD_REDUCED,]$model = "Penalized Models (Mean sd reduced)"
  validation.grid [validation.grid$model.id == PM_MEAN_SD_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==PM_MEAN_SD_REDUCED,]$roc.xval.2
  
  ## 6. QUANTILES_REDUCED 
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.reduced.train , y = ytrain.cat.train, 
                  method = "glmnet", tuneGrid = glmnGrid, metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Penalized Models (Quant reduced)",PM_REG_QUANTILES_REDUCED,
                                 Xtrain_quant.reduced.train, ytrain.cat.train,
                                 Xtrain_quant.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == PM_REG_QUANTILES_REDUCED,]$model = "Penalized Models (Quant reduced)"
  validation.grid [validation.grid$model.id == PM_REG_QUANTILES_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==PM_REG_QUANTILES_REDUCED,]$roc.xval.2
  
  print(perf.grid) 
  
  ######################################################## Nearest Shrunken Centroids 
  nscGrid <- data.frame(.threshold = 0:25)
  
  ## 3. MEAN_SD_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.scaled.train , y = ytrain.cat.train, 
                  method = "pam", tuneGrid = nscGrid, metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Nearest Shrunken Centroids (Mean sd scaled)",NSC_MEAN_SD_SCALED,
                                 Xtrain_mean_sd.scaled.train, ytrain.cat.train,
                                 Xtrain_mean_sd.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == NSC_MEAN_SD_SCALED,]$model = "Nearest Shrunken Centroids (Mean sd scaled)"
  validation.grid [validation.grid$model.id == NSC_MEAN_SD_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==NSC_MEAN_SD_SCALED,]$roc.xval.2
  
  ## 4. QUANTILES_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.scaled.train , y = ytrain.cat.train, 
                  method = "pam", tuneGrid = nscGrid, metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Nearest Shrunken Centroids (Quant scaled)",NSC_QUANTILES_SCALED,
                                 Xtrain_quant.scaled.train, ytrain.cat.train,
                                 Xtrain_quant.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == NSC_QUANTILES_SCALED,]$model = "Nearest Shrunken Centroids (Quant scaled)"
  validation.grid [validation.grid$model.id == NSC_QUANTILES_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==NSC_QUANTILES_SCALED,]$roc.xval.2
  
  ## 5. MEAN_SD_REDUCED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.reduced.train , y = ytrain.cat.train , 
                  method = "pam", tuneGrid = nscGrid, metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Nearest Shrunken Centroids (Mean sd reduced)",NSC_MEAN_SD_REDUCED,
                                 Xtrain_mean_sd.reduced.train, ytrain.cat.train,
                                 Xtrain_mean_sd.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == NSC_MEAN_SD_REDUCED,]$model = "Nearest Shrunken Centroids (Mean sd reduced)"
  validation.grid [validation.grid$model.id == NSC_MEAN_SD_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==NSC_MEAN_SD_REDUCED,]$roc.xval.2
  
  ## 6. QUANTILES_REDUCED 
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.reduced.train , y = ytrain.cat.train, 
                  method = "pam", tuneGrid = nscGrid, metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Nearest Shrunken Centroids (Quant reduced)",NSC_REG_QUANTILES_REDUCED,
                                 Xtrain_quant.reduced.train, ytrain.cat.train,
                                 Xtrain_quant.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == NSC_REG_QUANTILES_REDUCED,]$model = "Nearest Shrunken Centroids (Quant reduced)"
  validation.grid [validation.grid$model.id == NSC_REG_QUANTILES_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==NSC_REG_QUANTILES_REDUCED,]$roc.xval.2
  
  print(perf.grid) 

  ######################################################## Neural Networks 
  nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
  maxSize <- max(nnetGrid$.size)

  ## 5. MEAN_SD_REDUCED
  numWts <- 1*(maxSize * ( (dim(Xtrain_mean_sd.reduced.train)[2]) + 1) + maxSize + 1)
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.reduced.train , y = ytrain.cat.train , 
                  method = "nnet", metric = "ROC", 
                  preProc = c( "spatialSign") , 
                  tuneGrid = nnetGrid , trace = FALSE , maxit = 2000 , MaxNWts = numWts, 
                  trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Neural Networks (Mean sd reduced)",NN_MEAN_SD_REDUCED,
                                 Xtrain_mean_sd.reduced.train, ytrain.cat.train,
                                 Xtrain_mean_sd.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == NN_MEAN_SD_REDUCED,]$model = "Neural Networks (Mean sd reduced)"
  validation.grid [validation.grid$model.id == NN_MEAN_SD_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==NN_MEAN_SD_REDUCED,]$roc.xval.2
  
  ## 6. QUANTILES_REDUCED 
  numWts <- 1*(maxSize * ( (dim(Xtrain_quant.reduced.train)[2]) + 1) + maxSize + 1)
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.reduced.train , y = ytrain.cat.train, 
                  method = "nnet", metric = "ROC", 
                  preProc = c("spatialSign") , 
                  tuneGrid = nnetGrid , trace = FALSE , maxit = 2000 , MaxNWts = numWts, 
                  trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Neural Networks (Quant reduced)",NN_QUANTILES_REDUCED,
                                 Xtrain_quant.reduced.train, ytrain.cat.train,
                                 Xtrain_quant.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == NN_QUANTILES_REDUCED,]$model = "Neural Networks (Quant reduced)"
  validation.grid [validation.grid$model.id == NN_QUANTILES_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==NN_QUANTILES_REDUCED,]$roc.xval.2

  print(perf.grid) 

  ######################################################## SVM
  ## 3. MEAN_SD_SCALED
  sigmaRangeReduced <- sigest(as.matrix(Xtrain_mean_sd.scaled.train))
  svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1], .C = 2^(seq(-4, 4)))
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.scaled.train , y = ytrain.cat.train, 
                  method = "svmRadial", tuneGrid = svmRGridReduced, metric = "ROC", fit = FALSE, trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"SVM (Mean sd scaled)",SVM_MEAN_SD_SCALED,
                                 Xtrain_mean_sd.scaled.train, ytrain.cat.train,
                                 Xtrain_mean_sd.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == SVM_MEAN_SD_SCALED,]$model = "SVM (Mean sd scaled)"
  validation.grid [validation.grid$model.id == SVM_MEAN_SD_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==SVM_MEAN_SD_SCALED,]$roc.xval.2
  
  ## 4. QUANTILES_SCALED
  sigmaRangeReduced <- sigest(as.matrix(Xtrain_quant.scaled.train))
  svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1], .C = 2^(seq(-4, 4)))
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.scaled.train , y = ytrain.cat.train, 
                  method = "svmRadial", tuneGrid = svmRGridReduced, metric = "ROC", fit = FALSE, trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"SVM (Quant scaled)",SVM_QUANTILES_SCALED,
                                 Xtrain_quant.scaled.train, ytrain.cat.train,
                                 Xtrain_quant.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == SVM_QUANTILES_SCALED,]$model = "SVM (Quant scaled)"
  validation.grid [validation.grid$model.id == SVM_QUANTILES_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==SVM_QUANTILES_SCALED,]$roc.xval.2
  
  ## 5. MEAN_SD_REDUCED
  sigmaRangeReduced <- sigest(as.matrix(Xtrain_mean_sd.reduced.train))
  svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1], .C = 2^(seq(-4, 4)))
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.reduced.train , y = ytrain.cat.train , 
                  method = "svmRadial", tuneGrid = svmRGridReduced, metric = "ROC", fit = FALSE, trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"SVM (Mean sd reduced)",SVM_MEAN_SD_REDUCED,
                                 Xtrain_mean_sd.reduced.train, ytrain.cat.train,
                                 Xtrain_mean_sd.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == SVM_MEAN_SD_REDUCED,]$model = "SVM (Mean sd reduced)"
  validation.grid [validation.grid$model.id == SVM_MEAN_SD_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==SVM_MEAN_SD_REDUCED,]$roc.xval.2
  
  ## 6. QUANTILES_REDUCED 
  sigmaRangeReduced <- sigest(as.matrix(Xtrain_quant.reduced.train))
  svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1], .C = 2^(seq(-4, 4)))
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.reduced.train , y = ytrain.cat.train, 
                  method = "svmRadial", tuneGrid = svmRGridReduced, metric = "ROC", fit = FALSE, trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"SVM (Quant reduced)",SVM_QUANTILES_REDUCED,
                                 Xtrain_quant.reduced.train, ytrain.cat.train,
                                 Xtrain_quant.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == SVM_QUANTILES_REDUCED,]$model = "SVM (Quant reduced)"
  validation.grid [validation.grid$model.id == SVM_QUANTILES_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==SVM_QUANTILES_REDUCED,]$roc.xval.2

  print(perf.grid) 

  ######################################################## KNN
  ## 3. MEAN_SD_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.scaled.train , y = ytrain.cat.train, 
                  method = "knn", 
                  tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)),
                  metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"KNN (Mean sd scaled)",KNN_MEAN_SD_SCALED,
                                 Xtrain_mean_sd.scaled.train, ytrain.cat.train,
                                 Xtrain_mean_sd.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == KNN_MEAN_SD_SCALED,]$model = "KNN (Mean sd scaled)"
  validation.grid [validation.grid$model.id == KNN_MEAN_SD_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==KNN_MEAN_SD_SCALED,]$roc.xval.2
  
  ## 4. QUANTILES_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.scaled.train , y = ytrain.cat.train, 
                  method = "knn", 
                  tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)),
                  metric = "ROC",  trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"KNN (Quant scaled)",KNN_QUANTILES_SCALED,
                                 Xtrain_quant.scaled.train, ytrain.cat.train,
                                 Xtrain_quant.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == KNN_QUANTILES_SCALED,]$model = "KNN (Quant scaled)"
  validation.grid [validation.grid$model.id == KNN_QUANTILES_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==KNN_QUANTILES_SCALED,]$roc.xval.2
  
  ## 5. MEAN_SD_REDUCED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.reduced.train , y = ytrain.cat.train , 
                  method = "knn", 
                  tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)),
                  metric = "ROC",  trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"KNN (Mean sd reduced)",KNN_MEAN_SD_REDUCED,
                                 Xtrain_mean_sd.reduced.train, ytrain.cat.train,
                                 Xtrain_mean_sd.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == KNN_MEAN_SD_REDUCED,]$model = "KNN (Mean sd reduced)"
  validation.grid [validation.grid$model.id == KNN_MEAN_SD_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==KNN_MEAN_SD_REDUCED,]$roc.xval.2
  
  ## 6. QUANTILES_REDUCED 
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.reduced.train , y = ytrain.cat.train, 
                  method = "knn", 
                  tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)),
                  metric = "ROC",  trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"KNN (Quant reduced)",KNN_QUANTILES_REDUCED,
                                 Xtrain_quant.reduced.train, ytrain.cat.train,
                                 Xtrain_quant.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == KNN_QUANTILES_REDUCED,]$model = "KNN (Quant reduced)"
  validation.grid [validation.grid$model.id == KNN_QUANTILES_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==KNN_QUANTILES_REDUCED,]$roc.xval.2
  
  print(perf.grid) 

  ######################################################## CLASSIFICATION TREES 
  ## 1. MEAN_SD
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.train , y = ytrain.cat.train , 
                  method = "rpart", tuneLength = 30, metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Class. Trees (Mean sd)",CLASS_TREE_MEAN_SD,
                                 Xtrain_mean_sd.train, ytrain.cat.train,
                                 Xtrain_mean_sd.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid ,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == CLASS_TREE_MEAN_SD,]$model = "Class. Trees (Mean sd)"
  validation.grid [validation.grid$model.id == CLASS_TREE_MEAN_SD,(5+cs)] = perf.grid[perf.grid$model.id==CLASS_TREE_MEAN_SD,]$roc.xval.2
 
  ## 2. QUANTILES
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.train , y = ytrain.cat.train , 
                  method = "rpart", tuneLength = 30, metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Class. Trees (Quant)",CLASS_TREE_QUANTILES,
                                 Xtrain_quant.train, ytrain.cat.train,
                                 Xtrain_quant.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == CLASS_TREE_QUANTILES,]$model = "Class. Trees (Quant)"
  validation.grid [validation.grid$model.id == CLASS_TREE_QUANTILES,(5+cs)] = perf.grid[perf.grid$model.id==CLASS_TREE_QUANTILES,]$roc.xval.2

  ## 3. MEAN_SD_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.scaled.train , y = ytrain.cat.train, 
                  method = "rpart", tuneLength = 30, metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Class. Trees (Mean sd scaled)",CLASS_TREE_MEAN_SD_SCALED,
                                 Xtrain_mean_sd.scaled.train, ytrain.cat.train,
                                 Xtrain_mean_sd.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == CLASS_TREE_MEAN_SD_SCALED,]$model = "Class. Trees (Mean sd scaled)"
  validation.grid [validation.grid$model.id == CLASS_TREE_MEAN_SD_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==CLASS_TREE_MEAN_SD_SCALED,]$roc.xval.2
  
  ## 4. QUANTILES_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.scaled.train , y = ytrain.cat.train, 
                  method = "rpart", tuneLength = 30, metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Class. Trees (Quant scaled)",CLASS_TREE_QUANTILES_SCALED,
                                 Xtrain_quant.scaled.train, ytrain.cat.train,
                                 Xtrain_quant.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == CLASS_TREE_QUANTILES_SCALED,]$model = "Class. Trees (Quant scaled)"
  validation.grid [validation.grid$model.id == CLASS_TREE_QUANTILES_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==CLASS_TREE_QUANTILES_SCALED,]$roc.xval.2

  ## 5. MEAN_SD_REDUCED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.reduced.train , y = ytrain.cat.train , 
                  method = "rpart", tuneLength = 30, metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Class. Trees (Mean sd reduced)",CLASS_TREE_MEAN_SD_REDUCED,
                                 Xtrain_mean_sd.reduced.train, ytrain.cat.train,
                                 Xtrain_mean_sd.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == CLASS_TREE_MEAN_SD_REDUCED,]$model = "Class. Trees (Mean sd reduced)"
  validation.grid [validation.grid$model.id == CLASS_TREE_MEAN_SD_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==CLASS_TREE_MEAN_SD_REDUCED,]$roc.xval.2
  
  ## 6. QUANTILES_REDUCED 
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.reduced.train , y = ytrain.cat.train, 
                  method = "rpart", tuneLength = 30, metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Class. Trees (Quant reduced)",CLASS_TREE_QUANTILES_REDUCED,
                                 Xtrain_quant.reduced.train, ytrain.cat.train,
                                 Xtrain_quant.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == CLASS_TREE_QUANTILES_REDUCED,]$model = "Class. Trees (Quant reduced)"
  validation.grid [validation.grid$model.id == CLASS_TREE_QUANTILES_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==CLASS_TREE_QUANTILES_REDUCED,]$roc.xval.2

  print(perf.grid) 

  ######################################################## BOOSTED TREES 
  ## 1. MEAN_SD
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.train , y = ytrain.cat.train , 
                  method = "C5.0",  metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Boosted Trees C5.0 (Mean sd)",BOOSTED_TREE_MEAN_SD,
                                 Xtrain_mean_sd.train, ytrain.cat.train,
                                 Xtrain_mean_sd.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid ,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == BOOSTED_TREE_MEAN_SD,]$model = "Boosted Trees C5.0 (Mean sd)"
  validation.grid [validation.grid$model.id == BOOSTED_TREE_MEAN_SD,(5+cs)] = perf.grid[perf.grid$model.id==BOOSTED_TREE_MEAN_SD,]$roc.xval.2
 
  ## 2. QUANTILES
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.train , y = ytrain.cat.train , 
                  method = "C5.0",  metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Boosted Trees C5.0 (Quant)",BOOSTED_TREE_QUANTILES,
                                 Xtrain_quant.train, ytrain.cat.train,
                                 Xtrain_quant.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == BOOSTED_TREE_QUANTILES,]$model = "Boosted Trees C5.0 (Quant)"
  validation.grid [validation.grid$model.id == BOOSTED_TREE_QUANTILES,(5+cs)] = perf.grid[perf.grid$model.id==BOOSTED_TREE_QUANTILES,]$roc.xval.2

  ## 3. MEAN_SD_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.scaled.train , y = ytrain.cat.train, 
                  method = "C5.0",  metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Boosted Trees C5.0 (Mean sd scaled)",BOOSTED_TREE_MEAN_SD_SCALED,
                                 Xtrain_mean_sd.scaled.train, ytrain.cat.train,
                                 Xtrain_mean_sd.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == BOOSTED_TREE_MEAN_SD_SCALED,]$model = "Boosted Trees C5.0 (Mean sd scaled)"
  validation.grid [validation.grid$model.id == BOOSTED_TREE_MEAN_SD_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==BOOSTED_TREE_MEAN_SD_SCALED,]$roc.xval.2
  
  ## 4. QUANTILES_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.scaled.train , y = ytrain.cat.train, 
                  method = "C5.0",  metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Boosted Trees C5.0 (Quant scaled)",BOOSTED_TREE_QUANTILES_SCALED,
                                 Xtrain_quant.scaled.train, ytrain.cat.train,
                                 Xtrain_quant.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == BOOSTED_TREE_QUANTILES_SCALED,]$model = "Boosted Trees C5.0 (Quant scaled)"
  validation.grid [validation.grid$model.id == BOOSTED_TREE_QUANTILES_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==BOOSTED_TREE_QUANTILES_SCALED,]$roc.xval.2

  ## 5. MEAN_SD_REDUCED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.reduced.train , y = ytrain.cat.train , 
                  method = "C5.0",  metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Boosted Trees C5.0 (Mean sd reduced)",BOOSTED_TREE_MEAN_SD_REDUCED,
                                 Xtrain_mean_sd.reduced.train, ytrain.cat.train,
                                 Xtrain_mean_sd.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == BOOSTED_TREE_MEAN_SD_REDUCED,]$model = "Boosted Trees C5.0 (Mean sd reduced)"
  validation.grid [validation.grid$model.id == BOOSTED_TREE_MEAN_SD_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==BOOSTED_TREE_MEAN_SD_REDUCED,]$roc.xval.2
  
  ## 6. QUANTILES_REDUCED 
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.reduced.train , y = ytrain.cat.train, 
                  tuneGrid = expand.grid(.trials = c(1, (1:10)*10), .model = "tree", .winnow = c(TRUE, FALSE) ),
                  method = "C5.0",  metric = "ROC", trControl = controlObject)
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Boosted Trees C5.0 (Quant reduced)",BOOSTED_TREE_QUANTILES_REDUCED,
                                 Xtrain_quant.reduced.train, ytrain.cat.train,
                                 Xtrain_quant.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == BOOSTED_TREE_QUANTILES_REDUCED,]$model = "Boosted Trees C5.0 (Quant reduced)"
  validation.grid [validation.grid$model.id == BOOSTED_TREE_QUANTILES_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==BOOSTED_TREE_QUANTILES_REDUCED,]$roc.xval.2

  print(perf.grid) 

  ######################################################## BAGGING TREES 
  ## 1. MEAN_SD
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.train , y = ytrain.cat.train , 
                  method = "bag",  metric = "ROC", trControl = controlObject, B = 50 , 
                  bagControl = bagControl(fit = plsBag$fit,
                                          predict = plsBag$pred,
                                          aggregate = plsBag$aggregate))
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Bagged Trees (Mean sd)",BAGGING_TREE_MEAN_SD,
                                 Xtrain_mean_sd.train, ytrain.cat.train,
                                 Xtrain_mean_sd.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid ,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == BAGGING_TREE_MEAN_SD,]$model = "Bagged Trees (Mean sd)"
  validation.grid [validation.grid$model.id == BAGGING_TREE_MEAN_SD,(5+cs)] = perf.grid[perf.grid$model.id==BAGGING_TREE_MEAN_SD,]$roc.xval.2
  
  ## 2. QUANTILES
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.train , y = ytrain.cat.train , 
                  method = "bag",  metric = "ROC", trControl = controlObject, B = 50 ,  
                  bagControl = bagControl(fit = plsBag$fit,
                                          predict = plsBag$pred,
                                          aggregate = plsBag$aggregate))
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Bagged Trees (Quant)",BAGGING_TREE_QUANTILES,
                                 Xtrain_quant.train, ytrain.cat.train,
                                 Xtrain_quant.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == BAGGING_TREE_QUANTILES,]$model = "Bagged Trees (Quant)"
  validation.grid [validation.grid$model.id == BAGGING_TREE_QUANTILES,(5+cs)] = perf.grid[perf.grid$model.id==BAGGING_TREE_QUANTILES,]$roc.xval.2
  
  ## 3. MEAN_SD_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.scaled.train , y = ytrain.cat.train, 
                  method = "bag",  metric = "ROC", trControl = controlObject, B = 50 ,  
                  bagControl = bagControl(fit = plsBag$fit,
                                          predict = plsBag$pred,
                                          aggregate = plsBag$aggregate))
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Bagged Trees (Mean sd scaled)",BAGGING_TREE_MEAN_SD_SCALED,
                                 Xtrain_mean_sd.scaled.train, ytrain.cat.train,
                                 Xtrain_mean_sd.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == BAGGING_TREE_MEAN_SD_SCALED,]$model = "Bagged Trees (Mean sd scaled)"
  validation.grid [validation.grid$model.id == BAGGING_TREE_MEAN_SD_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==BAGGING_TREE_MEAN_SD_SCALED,]$roc.xval.2
  
  ## 4. QUANTILES_SCALED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.scaled.train , y = ytrain.cat.train, 
                  method = "bag",  metric = "ROC", trControl = controlObject, B = 50 , 
                  bagControl = bagControl(fit = plsBag$fit,
                                          predict = plsBag$pred,
                                          aggregate = plsBag$aggregate))
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Bagged Trees (Quant scaled)",BAGGING_TREE_QUANTILES_SCALED,
                                 Xtrain_quant.scaled.train, ytrain.cat.train,
                                 Xtrain_quant.scaled.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == BAGGING_TREE_QUANTILES_SCALED,]$model = "Bagged Trees (Quant scaled)"
  validation.grid [validation.grid$model.id == BAGGING_TREE_QUANTILES_SCALED,(5+cs)] = perf.grid[perf.grid$model.id==BAGGING_TREE_QUANTILES_SCALED,]$roc.xval.2
  
  
  ## 5. MEAN_SD_REDUCED
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_mean_sd.reduced.train , y = ytrain.cat.train , 
                  method = "bag",  metric = "ROC", trControl = controlObject, B = 50 ,
                  bagControl = bagControl(fit = plsBag$fit,
                                          predict = plsBag$pred,
                                          aggregate = plsBag$aggregate))
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Bagged Trees (Mean sd reduced)",BAGGING_TREE_QUANTILES_REDUCED,
                                 Xtrain_mean_sd.reduced.train, ytrain.cat.train,
                                 Xtrain_mean_sd.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == BAGGING_TREE_QUANTILES_REDUCED,]$model = "Bagged Trees (Mean sd reduced)"
  validation.grid [validation.grid$model.id == BAGGING_TREE_QUANTILES_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==BAGGING_TREE_QUANTILES_REDUCED,]$roc.xval.2
  
  ## 6. QUANTILES_REDUCED 
  set.seed(476); ptm <- proc.time()
  model <- train( x = Xtrain_quant.reduced.train , y = ytrain.cat.train, 
                  method = "bag",  metric = "ROC", trControl = controlObject, tuneGrid = data.frame(vars = seq(1, 15, by = 2)), 
                  bagControl = bagControl(fit = plsBag$fit,
                                          predict = plsBag$pred,
                                          aggregate = plsBag$aggregate))
  tm = proc.time() - ptm
  perf.grid = predictAndMeasure (model,"Bagged Trees (Quant reduced)",BOOSTED_TREE_QUANTILES_REDUCED,
                                 Xtrain_quant.reduced.train, ytrain.cat.train,
                                 Xtrain_quant.reduced.xval, ytrain.cat.xval,
                                 tm, grid = perf.grid,verbose=verbose, doPlot=doPlot)
  
  validation.grid [validation.grid$model.id == BOOSTED_TREE_QUANTILES_REDUCED,]$model = "Bagged Trees (Quant reduced)"
  validation.grid [validation.grid$model.id == BOOSTED_TREE_QUANTILES_REDUCED,(5+cs)] = perf.grid[perf.grid$model.id==BOOSTED_TREE_QUANTILES_REDUCED,]$roc.xval.2
  
  print(perf.grid) 

  #################################################### the winner is ... 
  perf.grid = perf.grid[order(perf.grid$roc.xval.2, perf.grid$roc.xval , perf.grid$acc.xval , decreasing = T),] 
  model.id.winner = perf.grid[1,]$model.id
  model.label.winner = as.character(perf.grid[1,]$predictor) 
  if (verbose) cat("************ THE WINNER IS ",model.label.winner," <<",model.id.winner,">> \n") 
}

  if (verbose) cat("********************** END OF CASES ",cs,"/"  ,NUMBER_CASES ,  " ********************** \n") 
   
  validation.grid$max = apply(validation.grid[-(1:5)] , 1 , max)
  validation.grid$min = apply(validation.grid[-(1:5)] , 1 , min)
  validation.grid$avg = apply(validation.grid[-(1:5)] , 1 , mean)
  
  validation.grid = validation.grid[order(validation.grid$min , decreasing = T),]

  ##### saving on disk perf.grid ...
  write.csv(validation.grid,quote=FALSE,file=paste0(getBasePath(),paste0(ds,"_validation_grid_class.csv")), row.names=FALSE)

}





