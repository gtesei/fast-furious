## notes_seizure_R
library(caret)
library(Hmisc)
library(data.table)
library(verification)

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

######### load data sets 
Xtrain_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds="Dog_1"),"Xtrain_mean_sd.zat",sep="") , header = F , sep=","  ))
Xtrain_quant = as.data.frame( fread(paste(getBasePath(type = "data" , ds="Dog_1"),"Xtrain_quant.zat",sep="") , header = F , sep=","  ))

Xtest_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds="Dog_1"),"Xtest_mean_sd.zat",sep="") , header = F , sep=","  ))
Xtest_quant = as.data.frame(fread(paste(getBasePath(type = "data" , ds="Dog_1"),"Xtest_quant.zat",sep="") , header = F , sep=","  ))

ytrain = as.data.frame(fread(paste(getBasePath(type = "data" , ds="Dog_1"),"ytrain.zat",sep="") , header = F , sep=","  ))

######### quick explanatory analysis 
ls()
describe(Xtrain_mean_sd)
describe(Xtest_mean_sd)

describe(Xtrain_quant)
describe(Xtest_quant)

describe(ytrain)

ps = sample((dim(Xtrain_mean_sd)[2]),16)

featurePlot(x = Xtrain_mean_sd[,ps], y = ytrain[,1],
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth")) 

featurePlot(x = Xtrain_quant[,ps], y = ytrain[,1],
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth")) 

######### arrainging datasets  
Xtrain_mean_sd$time_before_seizure = ytrain[,1]
Xtrain_quant$time_before_seizure = ytrain[,1]

set.seed(975)
forTraining <- createDataPartition(Xtrain_mean_sd$time_before_seizure, p = 3/4)[[1]]
Xtrain_mean_sd.train <- Xtrain_mean_sd[ forTraining,]
Xtrain_mean_sd.xval <- Xtrain_mean_sd[-forTraining,]

Xtrain_quant.train <- Xtrain_quant[ forTraining,]
Xtrain_quant.xval <- Xtrain_quant[-forTraining,]

ytrain.cat = as.factor(ytrain[forTraining,2]);
ytest.cat = as.factor(ytrain[-forTraining,2]);

####
preictIdx = which(ytrain[,2] == 1)
interictIdx = which(ytrain[,2] == 0)
signIdx = 1:((dim(Xtrain_quant)[2] -1) / 7) * 7 
signal = Xtrain_quant[,signIdx]
signal.interict = Xtrain_quant[interictIdx,signIdx]
signal.preict = Xtrain_quant[preictIdx,signIdx]

signal.interict.mean = apply(signal.interict,2,mean)
signal.interict.sd = apply(signal.interict,2,sd)

signal.interict.min = apply(signal.interict,2,min)
signal.interict.max = apply(signal.interict,2,max)

signal.preict.mean = apply(signal.preict,2,mean)
signal.preict.sd = apply(signal.preict,2,sd)

signal.preict.min = apply(signal.preict,2,min)
signal.preict.max = apply(signal.preict,2,max)

df = data.frame(signal.interict.mean=signal.interict.mean,signal.interict.sd=signal.interict.sd,
                signal.preict.mean=signal.preict.mean, signal.preict.sd=signal.preict.sd, 
                signal.interict.min=signal.interict.min, signal.interict.max, 
                signal.preict.min=signal.preict.min, signal.preict.max=signal.preict.max)





######### models in action 
predictAndMeasure = function(model,model.label,trainingData,ytrain,testData,ytest,tm , 
                             ytrain.cat , ytest.cat ,   
                             grid = NULL,verbose=F) {
  pred.train = predict(model , trainingData) 
  RMSE.train = RMSE(obs = ytrain , pred = pred.train)
  
  pred = predict(model , testData) 
  RMSE.test = RMSE(obs = ytest , pred = pred)
  
  if (verbose) cat("****** RMSE(train) =",RMSE.train," -  RMSE(test) =",RMSE.test,"  --  Time elapsed(sec.):",tm[[3]], "...  \n")
  
  ## fitter.cat 
  train.cat = data.frame( cat = ytrain.cat , pr =  pred.train )
  #test.cat = data.frame(cat = ytest.cat , pr =  pred )
  test.cat = data.frame( pr =  pred )
  ctrl <- trainControl(summaryFunction = twoClassSummary, classProbs = TRUE)
  fitter.cat <- train( cat ~ pr ,  data=train.cat , method = "glm", metric = "ROC", trControl = ctrl)
  pred.train.cat = predict(fitter.cat, train.cat ) 
  pred.cat = predict(fitter.cat, test.cat ) 
  
  acc.train = sum(ytrain.cat == pred.train.cat) / length(ytrain.cat)
  acc.test = sum(ytest.cat == pred.cat) / length(ytest.cat)
  
  roc.train = roc.area(as.numeric(ytrain.cat == 1) , as.numeric(pred.train.cat == 1) )$A
  roc.test = roc.area(as.numeric(ytest.cat == 1) , as.numeric(pred.cat == 1) )$A
  
  ### perf.grid 
  perf.grid = NULL
  if (is.null(grid)) { 
    perf.grid = data.frame(predictor = c(model.label) , RMSE.train = c(RMSE.train) , RMSE.test = c(RMSE.test) , 
                           acc.train = c(acc.train) , acc.test = c(acc.test) , 
                           roc.train = c(roc.train) , roc.test =c(roc.test),
                           time = c(tm[[3]]))
  } else {
    .grid = data.frame(predictor = c(model.label) , RMSE.train = c(RMSE.train) , 
                       RMSE.test = c(RMSE.test) , 
                       acc.train = c(acc.train) , acc.test = c(acc.test) , 
                       roc.train = c(roc.train) , roc.test =c(roc.test),
                       time = c(tm[[3]]) )
    perf.grid = rbind(grid, .grid)
  }
  
  perf.grid
}

verbose = T
controlObject <- trainControl(method = "repeatedcv", repeats = 5, number = 10)

######################################################## linear regression 
if (verbose) cat("****** [Xtrain_mean_sd] linear regression ...  \n")
set.seed(669); ptm <- proc.time()
linearReg <- train(time_before_seizure ~  . , data = Xtrain_mean_sd.train, method = "lm", trControl = controlObject) 
if (verbose) linearReg
tm = proc.time() - ptm
perf.grid = predictAndMeasure (linearReg,"Linear Reg (mean sd)",Xtrain_mean_sd.train,
                               Xtrain_mean_sd.train$time_before_seizure,
                               Xtrain_mean_sd.xval,Xtrain_mean_sd.xval$time_before_seizure,
                               tm ,  ytrain.cat , ytest.cat ,    
                               grid = NULL , verbose)

if (verbose) cat("****** [Xtrain_quant] linear regression ...  \n")
set.seed(669); ptm <- proc.time()
linearReg.quant <- train(time_before_seizure ~  . , data = Xtrain_quant.train, method = "lm", trControl = controlObject) 
if (verbose) linearReg.quant
tm = proc.time() - ptm
perf.grid = predictAndMeasure (linearReg.quant,"Linear Reg (quantiles)",Xtrain_quant.train,
                               Xtrain_quant.train$time_before_seizure,
                               Xtrain_quant.xval,Xtrain_quant.xval$time_before_seizure, 
                               tm,  ytrain.cat , ytest.cat ,    
                               grid = perf.grid , verbose )

######################################################## Elastic Net
# if (verbose) cat("****** [Xtrain_mean_sd] Elastic Net ...  \n")
# set.seed(669); ptm <- proc.time()
# enetGrid <- expand.grid(.lambda = c(0, .001, .01, .1), .fraction = seq(0.05, 1, length = 20))
# enetModel <- train(time_before_seizure ~  . , data = Xtrain_mean_sd.train , method = "enet", preProc = c("center", "scale"), 
#                    tuneGrid = enetGrid, trControl = controlObject)
# if (verbose) enetModel
# tm = proc.time() - ptm
# perf.grid = predictAndMeasure (enetModel,"Elastic Net (mean sd)",Xtrain_mean_sd.train,Xtrain_mean_sd.train$time_before_seizure,
#                                Xtrain_mean_sd.xval,Xtrain_mean_sd.xval$time_before_seizure,
#                                tm ,  ytrain.cat , ytest.cat ,    
#                                grid = perf.grid , verbose)
# 
# if (verbose) cat("****** [Xtrain_quant] Elastic Net ...  \n")
# set.seed(669); ptm <- proc.time()
# enetModel.quant <- train(time_before_seizure ~  . , data = Xtrain_quant.train , method = "enet", preProc = c("center", "scale"), tuneGrid = enetGrid, trControl = controlObject)
# if (verbose) enetModel.quant
# tm = proc.time() - ptm
# perf.grid = predictAndMeasure (enetModel.quant,"Elastic Net (quantiles)",Xtrain_quant.train,Xtrain_quant.train$time_before_seizure,
#                                Xtrain_quant.xval,Xtrain_quant.xval$time_before_seizure, 
#                                tm,  ytrain.cat , ytest.cat ,    
#                                grid = perf.grid , verbose)

######################################################## Partial Least Squares
if (verbose) cat("****** [Xtrain_mean_sd] Partial Least Squares ...  \n")
set.seed(669); ptm <- proc.time()
plsModel <- train(time_before_seizure ~  . , data = Xtrain_mean_sd.train , method = "pls", preProc = c("center", "scale"), tuneLength = 15, trControl = controlObject)
if (verbose) plsModel
tm = proc.time() - ptm
perf.grid = predictAndMeasure (plsModel,"PLS (mean sd)",Xtrain_mean_sd.train,Xtrain_mean_sd.train$time_before_seizure,
                               Xtrain_mean_sd.xval,Xtrain_mean_sd.xval$time_before_seizure,
                               tm ,  ytrain.cat , ytest.cat ,    
                               grid = perf.grid , verbose)

if (verbose) cat("****** [Xtrain_quant] Partial Least Squares ...  \n")
set.seed(669); ptm <- proc.time()
plsModel.quant <- train(time_before_seizure ~  . , data = Xtrain_quant.train , method = "pls", preProc = c("center", "scale"), tuneLength = 15, trControl = controlObject)
if (verbose) plsModel.quant
tm = proc.time() - ptm
perf.grid = predictAndMeasure (plsModel.quant,"PLS (quantiles)",Xtrain_quant.train,Xtrain_quant.train$time_before_seizure,
                               Xtrain_quant.xval,Xtrain_quant.xval$time_before_seizure, 
                               tm,  ytrain.cat , ytest.cat ,    
                               grid = perf.grid , verbose)

######################################################## Support Vector Machines 
if (verbose) cat("****** [Xtrain_mean_sd] Support Vector Machines ...  \n")
set.seed(669); ptm <- proc.time()
svmRModel <- train(time_before_seizure ~  . , data = Xtrain_mean_sd.train ,  method = "svmRadial",
                   tuneLength = 15, preProc = c("center", "scale"),  trControl = controlObject)
if (verbose) svmRModel
tm = proc.time() - ptm
perf.grid = predictAndMeasure (svmRModel,"SVM (mean sd)",Xtrain_mean_sd.train,Xtrain_mean_sd.train$time_before_seizure,
                               Xtrain_mean_sd.xval,Xtrain_mean_sd.xval$time_before_seizure,
                               tm ,  ytrain.cat , ytest.cat ,    
                               grid = perf.grid , verbose)

if (verbose) cat("****** [Xtrain_quant] Support Vector Machines ...  \n")
set.seed(669); ptm <- proc.time()
svmRModel.quant <- train(time_before_seizure ~  . , data = Xtrain_quant.train ,  method = "svmRadial",
                         tuneLength = 15, preProc = c("center", "scale"),  trControl = controlObject)
if (verbose) svmRModel.quant
tm = proc.time() - ptm
perf.grid = predictAndMeasure (svmRModel.quant,"SVM (quantiles)",Xtrain_quant.train,Xtrain_quant.train$time_before_seizure,
                               Xtrain_quant.xval,Xtrain_quant.xval$time_before_seizure, 
                               tm,  ytrain.cat , ytest.cat ,    
                               grid = perf.grid , verbose)

######################################################## Bagged Tree
if (verbose) cat("****** [Xtrain_mean_sd] Bagged Tree ...  \n")
set.seed(669); ptm <- proc.time()
treebagModel <- train(time_before_seizure ~  . , data = Xtrain_mean_sd.train , method = "treebag", trControl = controlObject)

if (verbose) treebagModel
tm = proc.time() - ptm
perf.grid = predictAndMeasure (treebagModel,"Bagged Tree (mean sd)",Xtrain_mean_sd.train,Xtrain_mean_sd.train$time_before_seizure,
                               Xtrain_mean_sd.xval,Xtrain_mean_sd.xval$time_before_seizure,
                               tm , ytrain.cat , ytest.cat ,  
                               grid = perf.grid , verbose)

if (verbose) cat("****** [Xtrain_quant] Bagged Tree ...  \n")
set.seed(669); ptm <- proc.time()
treebagModel.quant <- train(time_before_seizure ~  . , data = Xtrain_quant.train , method = "treebag", trControl = controlObject)

if (verbose) treebagModel.quant
tm = proc.time() - ptm
perf.grid = predictAndMeasure (treebagModel.quant,"Bagged Tree (quantiles)",Xtrain_quant.train,Xtrain_quant.train$time_before_seizure,
                               Xtrain_quant.xval,Xtrain_quant.xval$time_before_seizure, 
                               tm, ytrain.cat , ytest.cat ,  
                               grid = perf.grid , verbose )

######################################################## Cond Inf Tree
# if (verbose) cat("****** [Xtrain_mean_sd] Cond Inf Tree ...  \n")
# set.seed(669); ptm <- proc.time()
# ctreeModel <- train(time_before_seizure ~  . , data = Xtrain_mean_sd.train,  method = "ctree", tuneLength = 10, trControl = controlObject)
# 
# if (verbose) ctreeModel
# tm = proc.time() - ptm
# perf.grid = predictAndMeasure (ctreeModel,"Cond Inf Tree (mean sd)",Xtrain_mean_sd.train,Xtrain_mean_sd.train$time_before_seizure,
#                                Xtrain_mean_sd.xval,Xtrain_mean_sd.xval$time_before_seizure,
#                                tm , ytrain.cat , ytest.cat ,  
#                                grid = perf.grid , verbose)
# 
# if (verbose) cat("****** [Xtrain_quant] Cond Inf Tree ...  \n")
# set.seed(669); ptm <- proc.time()
# ctreeModel.quant <- train(time_before_seizure ~  . , data = Xtrain_quant.train ,  method = "ctree", tuneLength = 10, trControl = controlObject)
# 
# if (verbose) ctreeModel.quant
# tm = proc.time() - ptm
# perf.grid = predictAndMeasure (ctreeModel.quant,"Cond Inf Tree (quantiles)",Xtrain_quant.train,Xtrain_quant.train$time_before_seizure,
#                                Xtrain_quant.xval,Xtrain_quant.xval$time_before_seizure, 
#                                tm, ytrain.cat , ytest.cat ,  
#                                grid = perf.grid , verbose )

######################################################## CART
if (verbose) cat("****** [Xtrain_mean_sd] CART ...  \n")
set.seed(669); ptm <- proc.time()
rpartModel <- train(time_before_seizure ~  . , data = Xtrain_mean_sd.train,  method = "rpart", tuneLength = 30, trControl = controlObject)

if (verbose) rpartModel
tm = proc.time() - ptm
perf.grid = predictAndMeasure (rpartModel,"CART (mean sd)",Xtrain_mean_sd.train,Xtrain_mean_sd.train$time_before_seizure,
                               Xtrain_mean_sd.xval,Xtrain_mean_sd.xval$time_before_seizure,
                               tm , ytrain.cat , ytest.cat ,  
                               grid = perf.grid , verbose)

if (verbose) cat("****** [Xtrain_quant] CART ...  \n")
set.seed(669); ptm <- proc.time()
rpartModel.quant <- train(time_before_seizure ~  . , data = Xtrain_quant.train , method = "rpart", tuneLength = 30, trControl = controlObject)

if (verbose) rpartModel.quant
tm = proc.time() - ptm
perf.grid = predictAndMeasure (rpartModel.quant,"CART (quantiles)",Xtrain_quant.train,Xtrain_quant.train$time_before_seizure,
                               Xtrain_quant.xval,Xtrain_quant.xval$time_before_seizure, 
                               tm, ytrain.cat , ytest.cat ,  
                               grid = perf.grid , verbose )

######################################################## NNET\if (verbose) 
# cat("****** [Xtrain_mean_sd] NNET ...  \n")
# set.seed(669); ptm <- proc.time()
# nnetGrid <- expand.grid(.decay = c(0.001, .01, .1), .size = seq(1, 27, by = 2), .bag = FALSE)
# nnetModel <- train(time_before_seizure ~  . , data = Xtrain_mean_sd.train , method = "avNNet", 
#                          tuneGrid = nnetGrid, preProc = c("center", "scale"), linout = TRUE, 
#                          trace = FALSE, maxit = 1000, trControl = controlObject)
# 
# if (verbose) nnetModel
# tm = proc.time() - ptm
# perf.grid = predictAndMeasure (nnetModel,"NNET (mean sd)",Xtrain_mean_sd.train,Xtrain_mean_sd.train$time_before_seizure,
#                                Xtrain_mean_sd.xval,Xtrain_mean_sd.xval$time_before_seizure,
#                                tm , ytrain.cat , ytest.cat ,  
#                                grid = perf.grid , verbose)
# 
# if (verbose) cat("****** [Xtrain_quant] NNET ...  \n")
# set.seed(669); ptm <- proc.time()
# nnetGrid <- expand.grid(.decay = c(0.001, .01, .1), .size = seq(1, 27, by = 2), .bag = FALSE)
# nnetModel.quant <- train(time_before_seizure ~  . , data = Xtrain_quant.train , method = "avNNet", 
#                    tuneGrid = nnetGrid, preProc = c("center", "scale"), linout = TRUE, 
#                    trace = FALSE, maxit = 1000, trControl = controlObject)
# if (verbose) nnetModel.quant
# tm = proc.time() - ptm
# perf.grid = predictAndMeasure (nnetModel.quant,"NNET (quantiles)",Xtrain_quant.train,Xtrain_quant.train$time_before_seizure,
#                                Xtrain_quant.xval,Xtrain_quant.xval$time_before_seizure, 
#                                tm, ytrain.cat , ytest.cat ,  
#                                grid = perf.grid , verbose )

######################################################## CUBIST
cat("****** [Xtrain_mean_sd] CUBIST ...  \n")
set.seed(669); ptm <- proc.time()
cubistGrid <- expand.grid(.committees = c(1, 5, 10, 50, 75, 100), .neighbors = c(0, 1, 3, 5, 7, 9))
cbModel <- train(time_before_seizure ~  . , data = Xtrain_mean_sd.train ,  method = "cubist", 
                       tuneGrid = cubistGrid, trControl = controlObject)

if (verbose) cbModel
tm = proc.time() - ptm
perf.grid = predictAndMeasure (cbModel,"CUBIST (mean sd)",Xtrain_mean_sd.train,Xtrain_mean_sd.train$time_before_seizure,
                               Xtrain_mean_sd.xval,Xtrain_mean_sd.xval$time_before_seizure,
                               tm , ytrain.cat , ytest.cat ,  
                               grid = perf.grid , verbose)

if (verbose) cat("****** [Xtrain_quant] CUBIST ...  \n")
set.seed(669)
cubistGrid <- expand.grid(.committees = c(1, 5, 10, 50, 75, 100), .neighbors = c(0, 1, 3, 5, 7, 9))
cbModel.quant <- train(time_before_seizure ~  . , data = Xtrain_quant.train ,  method = "cubist", 
                 tuneGrid = cubistGrid, trControl = controlObject)
if (verbose) cbModel.quant
tm = proc.time() - ptm
perf.grid = predictAndMeasure (cbModel.quant,"CUBIST (quantiles)",Xtrain_quant.train,Xtrain_quant.train$time_before_seizure,
                               Xtrain_quant.xval,Xtrain_quant.xval$time_before_seizure, 
                               tm, ytrain.cat , ytest.cat ,  
                               grid = perf.grid , verbose )

######################################################## RF
# cat("****** [Xtrain_mean_sd] RF ...  \n")
# set.seed(669)
# rfModel <- train(CompressiveStrength ~ ., data = trainingSet, method = "rf", tuneLength = 10, ntrees = 1000, 
#                  importance = TRUE, trControl = controlObject)
# 
# cat("****** [Xtrain_mean_sd] RF ...  \n")
# set.seed(669); ptm <- proc.time()
# rfModel <- train(time_before_seizure ~  . , data = Xtrain_mean_sd.train ,  
#                  method = "rf", tuneLength = 10, ntrees = 1000, 
#                  importance = TRUE, trControl = controlObject)
# 
# if (verbose) rfModel
# tm = proc.time() - ptm
# perf.grid = predictAndMeasure (rfModel,"RF (mean sd)",Xtrain_mean_sd.train,Xtrain_mean_sd.train$time_before_seizure,
#                                Xtrain_mean_sd.xval,Xtrain_mean_sd.xval$time_before_seizure,
#                                tm , ytrain.cat , ytest.cat ,  
#                                grid = perf.grid , verbose)
# 
# if (verbose) cat("****** [Xtrain_quant] NNET ...  \n")
# set.seed(669)
# cubistGrid <- expand.grid(.committees = c(1, 5, 10, 50, 75, 100), .neighbors = c(0, 1, 3, 5, 7, 9))
# rfModel.quant <- train(time_before_seizure ~  . , data = Xtrain_quant.train ,  
#                        method = "rf", tuneLength = 10, ntrees = 1000, 
#                        importance = TRUE, trControl = controlObject)
# if (verbose) rfModel.quant
# tm = proc.time() - ptm
# perf.grid = predictAndMeasure (rfModel.quant,"RF (quantiles)",Xtrain_quant.train,Xtrain_quant.train$time_before_seizure,
#                                Xtrain_quant.xval,Xtrain_quant.xval$time_before_seizure, 
#                                tm, ytrain.cat , ytest.cat ,  
#                                grid = perf.grid , verbose )


##### saving on disk 
write.csv(perf.grid,quote=FALSE,file=paste0(getBasePath(),"Dog_1_perf_grid_regress.csv"), row.names=FALSE)
