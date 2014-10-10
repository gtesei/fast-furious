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

ytrain.cat = as.factor(ytrain[forTraining,2])
ytest.cat = as.factor(ytrain[-forTraining,2])

levels(ytrain.cat) = levels(ytest.cat) = c("inter-ict","pre-ict")

verbose = T
controlObject <- trainControl(method = "repeatedcv", repeats = 5, number = 10)

######
if (verbose) cat("****** [Xtrain_quant] linear regression ...  \n")
set.seed(669); ptm <- proc.time()
linearReg.quant <- train(time_before_seizure ~  . , data = Xtrain_quant.train, method = "lm", trControl = controlObject) 
if (verbose) linearReg.quant
tm = proc.time() - ptm


model = linearReg.quant
trainingData = Xtrain_quant.train
ytrain = Xtrain_quant.train$time_before_seizure
testData = Xtrain_quant.xval
ytest = Xtrain_quant.xval$time_before_seizure
model.label = "Linear Reg (quant)"
grid = NULL


pred.train = predict(model , trainingData) 
RMSE.train = RMSE(obs = ytrain , pred = pred.train)
pred = predict(model , testData) 
RMSE.test = RMSE(obs = ytest , pred = pred)
if (verbose) cat("****** RMSE(train) =",RMSE.train," -  RMSE(test) =",RMSE.test,"  --  Time elapsed(sec.):",tm[[3]], "...  \n")

## fitter.cat 
train.cat = data.frame( cat = ytrain.cat , pr =  pred.train )
test.cat = data.frame( pr =  pred )

## glm 
fitter.cat <- glm( cat ~ pr ,  data=train.cat , family = binomial)
pred.train.glm = predict(fitter.cat, newdata = train.cat , type = "response") 
pred.test.glm = predict(fitter.cat, newdata = test.cat , type = "response") 

acc.train = sum(    factor(ifelse(pred.train.glm > 0.5,1,0), levels=levels(ytrain.cat)) ==   ytrain.cat   ) / length(ytrain.cat)
acc.test = sum(     factor(ifelse(pred.test.glm > 0.5,1,0),       levels=levels(ytrain.cat)) ==   ytest.cat  ) / length(ytest.cat)

rocCurve <- roc(response = ytrain.cat, predictor = as.numeric(pred.train.glm), levels = rev(levels(ytrain.cat)))
roc.train = as.numeric( auc(rocCurve) )

rocCurve <- roc(response = ytest.cat, predictor = as.numeric(pred.test.glm ), levels = rev(levels(ytest.cat)))
roc.test = as.numeric( auc(rocCurve) )


###########
library(randomForest)
rfModel <- randomForest(cat ~ pr ,  data=train.cat , ntree = 2000)
library(MASS) ## for the qda() function
qdaModel <- qda(cat ~ pr ,  data=train.cat)

pred.train.qda <- predict(qdaModel, train.cat)
names(pred.train.qda)
head(pred.train.qda$class)
head(pred.train.qda$posterior)

pred.test.qda <- predict(qdaModel, test.cat)
pred.test.qda.pre  <- pred.test.qda$posterior[,"pre-ict"]
rocCurve <- roc(response = ytest.cat, predictor = as.numeric(pred.test.qda.pre ), levels = rev(levels(ytest.cat)))
roc.test = as.numeric( auc(rocCurve) )
roc.test ## 0.7194444 !! 
ci.roc(rocCurve)
plot(rocCurve, legacy.axes = TRUE)



######### rf
pred.test.rf <- predict(rfModel, test.cat, type = "prob")
head(pred.test.rf)
pred.test.rf.pre <- pred.test.rf[,"pre-ict"]
rocCurve <- roc(response = ytest.cat, predictor = as.numeric(pred.test.rf.pre ), levels = rev(levels(ytest.cat)))
roc.test = as.numeric( auc(rocCurve) )
roc.test ## 0.7118056 !! 
ci.roc(rocCurve)
plot(rocCurve, legacy.axes = TRUE)


#### calibration 
calCurve <- calibration(ytest.cat ~ pred.test.qda.pre + pred.test.rf.pre )
calCurve
xyplot(calCurve, auto.key = list(columns = 2))

sigmoidalCal <- glm(relevel(ytest.cat, ref = "inter-ict") ~ pred.test.qda.pre , family = binomial)
coef(summary(sigmoidalCal)) 
sigmoidProbs <- predict(sigmoidalCal, newdata = pred.test.qda.pre, type = "response")
calCurve <- calibration(ytest.cat ~ pred.test.qda.pre + pred.test.rf.pre + sigmoidProbs)
calCurve
xyplot(calCurve, auto.key = list(columns = 2))


library(klaR)
dd = data.frame(class = ytest.cat , pr = pred.test.qda.pre)
BayesCal <- NaiveBayes(class ~ pr , data = dd,  usekernel = TRUE)
BayesProbs <- predict(BayesCal, newdata = dd[, "pr", drop = FALSE])
QDABayes <- BayesProbs$posterior[, "pre-ict"]
calCurve2 <- calibration(ytest.cat ~ pred.test.qda.pre + pred.test.rf.pre + QDABayes)
xyplot(calCurve2)
xyplot(calCurve2, auto.key = list(columns = 2))

