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

######### load data sets 
Xtrain_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds="Dog_1"),"Xtrain_mean_sd.zat",sep="") , header = F , sep=","  ))
Xtrain_quant = as.data.frame( fread(paste(getBasePath(type = "data" , ds="Dog_1"),"Xtrain_quant.zat",sep="") , header = F , sep=","  ))

Xtest_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds="Dog_1"),"Xtest_mean_sd.zat",sep="") , header = F , sep=","  ))
Xtest_quant = as.data.frame(fread(paste(getBasePath(type = "data" , ds="Dog_1"),"Xtest_quant.zat",sep="") , header = F , sep=","  ))

ytrain = as.data.frame(fread(paste(getBasePath(type = "data" , ds="Dog_1"),"ytrain.zat",sep="") , header = F , sep=","  ))

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

verbose = T
controlObject <- trainControl(method = "repeatedcv", repeats = 5, number = 10)

######################################################## linear regression 
# if (verbose) cat("****** [Xtrain_mean_sd] linear regression ...  \n")
# set.seed(669); ptm <- proc.time()
# linearReg <- train(time_before_seizure ~  . , data = Xtrain_mean_sd.train, method = "lm", trControl = controlObject) 
# if (verbose) linearReg
# tm = proc.time() - ptm
# perf.grid = predictAndMeasure (linearReg,"Linear Reg (mean sd)",Xtrain_mean_sd.train,
#                                Xtrain_mean_sd.train$time_before_seizure,
#                                Xtrain_mean_sd.xval,Xtrain_mean_sd.xval$time_before_seizure,
#                                tm ,  ytrain.cat , ytest.cat ,    
#                                grid = NULL , verbose)

if (verbose) cat("****** [Xtrain_quant] linear regression ...  \n")
set.seed(669); ptm <- proc.time()
linearReg.quant <- train(time_before_seizure ~  . , data = Xtrain_quant.train, method = "lm", trControl = controlObject) 
if (verbose) linearReg.quant
tm = proc.time() - ptm
# perf.grid = predictAndMeasure (linearReg.quant,"Linear Reg (quantiles)",Xtrain_quant.train,
#                                Xtrain_quant.train$time_before_seizure,
#                                Xtrain_quant.xval,Xtrain_quant.xval$time_before_seizure, 
#                                tm,  ytrain.cat , ytest.cat ,    
#                                grid = perf.grid , verbose )

##########
set.seed(669); ptm <- proc.time()
plsModel <- train(time_before_seizure ~  . , data = Xtrain_mean_sd.train , method = "pls", preProc = c("center", "scale"), 
                  tuneLength = 15, trControl = controlObject)
#if (verbose) plsModel
tm = proc.time() - ptm
##########

model = plsModel
trainingData = Xtrain_quant.train
ytrain = Xtrain_quant.train$time_before_seizure
testData = Xtrain_quant.xval
ytest = Xtrain_quant.xval$time_before_seizure
model.label = "Linear Reg (mean sd)"
grid = NULL


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
#fitter.cat <- train( cat ~ pr ,  data=train.cat , method = "glm", metric = "ROC", trControl = ctrl)
fitter.cat <- glm( cat ~ pr ,  data=train.cat , family = binomial)
pred.train.cat = predict(fitter.cat, newdata = train.cat , type = "response") 
pred.cat = predict(fitter.cat, newdata = test.cat , type = "response") 

acc.train = sum(    factor(ifelse(pred.train.cat > 0.5,1,0), levels=levels(ytrain.cat)) ==   ytrain.cat   ) / length(ytrain.cat)
acc.test = sum(     factor(ifelse(pred.cat > 0.5,1,0),       levels=levels(ytrain.cat)) ==   ytest.cat  ) / length(ytest.cat)

#roc.train = roc.area(as.numeric(ytrain.cat == 1) , as.numeric(pred.train.cat == 1) )$A
#roc.test = roc.area(as.numeric(ytest.cat == 1) , as.numeric(pred.cat == 1) )$A

rocCurve <- roc(response = ytrain.cat, predictor = as.numeric(pred.train.cat), levels = rev(levels(ytrain.cat)))
roc.train = as.numeric( auc(rocCurve) )

rocCurve <- roc(response = ytest.cat, predictor = as.numeric(pred.cat ), levels = rev(levels(ytest.cat)))
roc.test = as.numeric( auc(rocCurve) )

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

####### proviamo l'errore 
rocCurve <- roc(response = ytest.cat, predictor = as.numeric(pred.cat > 0.5), levels = rev(levels(ytest.cat)))
roc.test = as.numeric( auc(rocCurve) )
roc.test ## 0.3375 !!!!! 
ci.roc(rocCurve)
plot(rocCurve, legacy.axes = TRUE)

rocCurve <- roc(response = ytest.cat, predictor = as.numeric(pred.cat ), levels = rev(levels(ytest.cat)))
roc.test = as.numeric( auc(rocCurve) )
roc.test ## 0.7194444 !! 
ci.roc(rocCurve)
plot(rocCurve, legacy.axes = TRUE)
plot(pred.cat,ytest.cat) 


###########
library(randomForest)
rfModel <- randomForest(cat ~ pr ,  data=train.cat , ntree = 2000)
library(MASS) ## for the qda() function
qdaModel <- qda(cat ~ pr ,  data=train.cat)

qdaTrainPred <- predict(qdaModel, train.cat)
names(qdaTrainPred)
head(qdaTrainPred$class)
head(qdaTrainPred$posterior)

qdaTestPred <- predict(qdaModel, test.cat)
pred.train.cat  <- qdaTrainPred$posterior[,"1"]
pred.cat <- qdaTestPred$posterior[,"1"]
rocCurve <- roc(response = ytest.cat, predictor = as.numeric(pred.cat ), levels = rev(levels(ytest.cat)))
roc.test = as.numeric( auc(rocCurve) )
roc.test ## 0.7194444 !! 
ci.roc(rocCurve)
plot(rocCurve, legacy.axes = TRUE)


rfTestPred <- predict(rfModel, test.cat, type = "prob")
head(rfTestPred)
pred.cat <- rfTestPred[,"1"]
rocCurve <- roc(response = ytest.cat, predictor = as.numeric(pred.cat ), levels = rev(levels(ytest.cat)))
roc.test = as.numeric( auc(rocCurve) )
roc.test ## 0.7118056 !! 
ci.roc(rocCurve)
plot(rocCurve, legacy.axes = TRUE)


ctrl <- trainControl(summaryFunction = twoClassSummary, classProbs = TRUE)
set.seed(202)
sigmaRangeReduced <- sigest(as.matrix( pred.train ))
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                                  .C = 2^(seq(-4, 4)))
set.seed(476)
svmRModel <- train(cat ~ pr ,  data=train.cat , 
                     method = "svmRadial",
                     metric = "ROC",
                     preProc = c("center", "scale"),
                     tuneGrid = svmRGridReduced,
                     fit = FALSE,
                     trControl = ctrl)

pred.cat = predict(svmRModel, train.cat , type = "prob" )

svmRModel
