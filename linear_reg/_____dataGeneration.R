
getBasePath = function () {
  ret = ""
  base.path1 = "C:/docs/ff/gitHub/fast-furious"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious"
  
  if (file.exists(base.path1))  {
    ret = base.path1
  } else {
    ret = base.path2
  }
  
  ret
}

predictAndMeasure = function(model,model.label,trainingData,ytrain,testData,ytest,tm , grid = NULL,verbose=F) {
  pred = predict(model , trainingData) 
  RMSE.train = RMSE(obs = ytrain , pred = pred)
  
  pred = predict(model , testData) 
  RMSE.test = RMSE(obs = ytest , pred = pred)
  
  if (verbose) cat("******[WITHOUT TRANSFORMATIONS]  RMSE(train) =",RMSE.train," -  RMSE(test) =",RMSE.test,"  --  Time elapsed(sec.):",tm[[3]], "...  \n")
  
  perf.grid = NULL
  if (is.null(grid)) { 
    perf.grid = data.frame(predictor = c(model.label) , RMSE.train = c(RMSE.train) , RMSE.test = c(RMSE.test) , time = c(tm[[3]]))
  } else {
    .grid = data.frame(predictor = c(model.label) , RMSE.train = c(RMSE.train) , RMSE.test = c(RMSE.test) , time = c(tm[[3]]))
    perf.grid = rbind(grid, .grid)
  }
  
  perf.grid
}

#AppliedPredictiveModeling ::: solubility
library(caret)
library(AppliedPredictiveModeling)
data(solubility)

doBenchemark = F 
verbose = T 

cat("****** writing on disk data sets ... ", ls()[as.vector( (grep("sol",ls())) )], " ... \n")

write.csv(solTrainX,quote=F,row.names=F,file=paste0(getBasePath(),"/linear_reg/__solTrainX.zat"))
write.csv(solTestX,quote=F,row.names=F,file=paste0(getBasePath(),"/linear_reg/__solTestX.zat"))

write.csv(solTestXtrans,quote=F,row.names=F,file=paste0(getBasePath(),"/linear_reg/__solTestXtrans.zat"))
write.csv(solTrainXtrans,quote=F,row.names=F,file=paste0(getBasePath(),"/linear_reg/__solTrainXtrans.zat"))

write.csv(solTestY,quote=F,row.names=F,file=paste0(getBasePath(),"/linear_reg/__solTestY.zat"))
write.csv(solTrainY,quote=F,row.names=F,file=paste0(getBasePath(),"/linear_reg/__solTrainY.zat"))

if (! doBenchemark) stop("finished.")

cat("****** setting some bechmarks ...  \n")

trainingData = solTrainX
trainingData$Solubility = solTrainY

testData = solTestX
#testData$Solubility = solTestY

trainingData.trans = solTrainXtrans
trainingData.trans$Solubility = solTrainY

testData.trans = solTestXtrans
#testData$Solubility = solTestY

controlObject <- trainControl(method = "repeatedcv", repeats = 5, number = 10)


######################################################## linear regression 
if (verbose) cat("****** [WITHOUT TRANSFORMATIONS] linear regression ...  \n")
set.seed(669); ptm <- proc.time()
linearReg <- train(Solubility ~  . , data = trainingData, method = "lm", trControl = controlObject) 
if (verbose) linearReg
tm = proc.time() - ptm
perf.grid = predictAndMeasure (linearReg,"Linear Reg",trainingData,solTrainY,testData,solTestY,tm , grid = NULL , verbose)

if (verbose) cat("****** [WITH TRANSFORMATIONS] linear regression ...  \n")
set.seed(669); ptm <- proc.time()
linearReg.trans <- train(Solubility ~  . , data = trainingData.trans, method = "lm", trControl = controlObject) 
if (verbose) linearReg.trans
tm = proc.time() - ptm
perf.grid = predictAndMeasure (linearReg.trans,"Linear Reg (Trans)",trainingData.trans,solTrainY,testData.trans,solTestY,tm, grid = perf.grid)

######################################################## Elastic Net
if (verbose) cat("****** [WITHOUT TRANSFORMATIONS] Elastic Net ...  \n")
set.seed(669); ptm <- proc.time()
enetGrid <- expand.grid(.lambda = c(0, .001, .01, .1), .fraction = seq(0.05, 1, length = 20))
enetModel <- train(Solubility ~ . , data = trainingData , method = "enet", preProc = c("center", "scale"), tuneGrid = enetGrid, trControl = controlObject)
if (verbose) enetModel
tm = proc.time() - ptm
perf.grid = predictAndMeasure (enetModel,"Elastic Net",trainingData,solTrainY,testData,solTestY,tm , grid = perf.grid , verbose )

if (verbose) cat("****** [WITH TRANSFORMATIONS] Elastic Net ...  \n")
set.seed(669); ptm <- proc.time()
enetModel.trans <- train(Solubility ~ . , data = trainingData.trans , method = "enet", preProc = c("center", "scale"), tuneGrid = enetGrid, trControl = controlObject)
if (verbose) enetModel.trans
tm = proc.time() - ptm
perf.grid = predictAndMeasure (enetModel.trans,"Elastic Net (Trans)",trainingData.trans,solTrainY,testData.trans,solTestY,tm, grid = perf.grid , verbose )

######################################################## Partial Least Squares
if (verbose) cat("****** [WITHOUT TRANSFORMATIONS] Partial Least Squares ...  \n")
set.seed(669); ptm <- proc.time()
plsModel <- train(Solubility ~ . , data = trainingData , method = "pls", preProc = c("center", "scale"), tuneLength = 15, trControl = controlObject)
if (verbose) plsModel
tm = proc.time() - ptm
perf.grid = predictAndMeasure (plsModel,"PLS",trainingData,solTrainY,testData,solTestY,tm , grid = perf.grid , verbose )

if (verbose) cat("****** [WITH TRANSFORMATIONS] Partial Least Squares ...  \n")
set.seed(669); ptm <- proc.time()
plsModel.trans <- train(Solubility ~ . , data = trainingData.trans , method = "pls", preProc = c("center", "scale"), tuneLength = 15, trControl = controlObject)
if (verbose) plsModel.trans
tm = proc.time() - ptm
perf.grid = predictAndMeasure (plsModel.trans,"PLS (Trans)",trainingData.trans,solTrainY,testData.trans,solTestY,tm, grid = perf.grid , verbose )

######################################################## Support Vector Machines 
if (verbose) cat("****** [WITHOUT TRANSFORMATIONS] Support Vector Machines ...  \n")
set.seed(669); ptm <- proc.time()
svmRModel <- train(Solubility ~ . , data = trainingData, method = "svmRadial",
                   tuneLength = 15, preProc = c("center", "scale"),  trControl = controlObject)
if (verbose) svmRModel
tm = proc.time() - ptm
perf.grid = predictAndMeasure (svmRModel,"SVM",trainingData,solTrainY,testData,solTestY,tm , grid = perf.grid , verbose )

if (verbose) cat("****** [WITH TRANSFORMATIONS] Support Vector Machines ...  \n")
set.seed(669); ptm <- proc.time()
svmRModel.trans <- train(Solubility ~ . , data = trainingData.trans, method = "svmRadial",
                   tuneLength = 15, preProc = c("center", "scale"),  trControl = controlObject)
if (verbose) svmRModel.trans
tm = proc.time() - ptm
perf.grid = predictAndMeasure (svmRModel.trans,"SVM (Trans)",trainingData.trans,solTrainY,testData.trans,solTestY,tm, grid = perf.grid , verbose )

######################################################## Bagged Tree
if (verbose) cat("****** [WITHOUT TRANSFORMATIONS] Bagged Tree ...  \n")
set.seed(669); ptm <- proc.time()
treebagModel <- train(Solubility ~ . , data = trainingData, method = "treebag", trControl = controlObject)

if (verbose) treebagModel
tm = proc.time() - ptm
perf.grid = predictAndMeasure (treebagModel,"Bagged Tree",trainingData,solTrainY,testData,solTestY,tm , grid = perf.grid , verbose )

if (verbose) cat("****** [WITH TRANSFORMATIONS] Bagged Tree ...  \n")
set.seed(669); ptm <- proc.time()
treebagModel.trans <- train(Solubility ~ . , data = trainingData.trans, method = "treebag", trControl = controlObject)

if (verbose) treebagModel.trans
tm = proc.time() - ptm
perf.grid = predictAndMeasure (treebagModel.trans,"Bagged Tree (Trans)",trainingData.trans,solTrainY,
                               testData.trans,solTestY,tm, grid = perf.grid , verbose )

######################################################## Cond Inf Tree
if (verbose) cat("****** [WITHOUT TRANSFORMATIONS] Cond Inf Tree ...  \n")
set.seed(669); ptm <- proc.time()
ctreeModel <- train(Solubility ~ . , data = trainingData , method = "ctree", tuneLength = 10, trControl = controlObject)

if (verbose) ctreeModel
tm = proc.time() - ptm
perf.grid = predictAndMeasure (ctreeModel,"Cond Inf Tree",trainingData,solTrainY,testData,solTestY,tm , grid = perf.grid , verbose )

if (verbose) cat("****** [WITH TRANSFORMATIONS] Cond Inf Tree ...  \n")
set.seed(669); ptm <- proc.time()
ctreeModel.trans <- train(Solubility ~ . , data = trainingData.trans , method = "ctree", tuneLength = 10, trControl = controlObject)

if (verbose) ctreeModel.trans
tm = proc.time() - ptm
perf.grid = predictAndMeasure (ctreeModel.trans,"Cond Inf Tree (Trans)",trainingData.trans,solTrainY,
                               testData.trans,solTestY,tm, grid = perf.grid , verbose )

######################################################## CART
if (verbose) cat("****** [WITHOUT TRANSFORMATIONS] CART ...  \n")
set.seed(669); ptm <- proc.time()
rpartModel <- train(Solubility ~ . , data = trainingData , method = "rpart", tuneLength = 30, trControl = controlObject)

if (verbose) rpartModel
tm = proc.time() - ptm
perf.grid = predictAndMeasure (rpartModel,"Cond Inf Tree",trainingData,solTrainY,testData,solTestY,tm , grid = perf.grid , verbose )

if (verbose) cat("****** [WITH TRANSFORMATIONS] CART ...  \n")
set.seed(669); ptm <- proc.time()
rpartModel.trans <- train(Solubility ~ . , data = trainingData.trans , method = "rpart", tuneLength = 30, trControl = controlObject)

if (verbose) rpartModel.trans
tm = proc.time() - ptm
perf.grid = predictAndMeasure (rpartModel.trans,"Cond Inf Tree (Trans)",trainingData.trans,solTrainY,
                               testData.trans,solTestY,tm, grid = perf.grid , verbose )


###### performance plot 
allResamples <- resamples(list("Linear Reg" = linearReg, "Linear Reg (Trans)" = linearReg.trans, 
                               "SVM" = svmRModel , "SVM (Trans)" = svmRModel.trans , 
                               "PLS" = plsModel , "PLS (Trans)" = plsModel.trans , 
                               "Elastic Net" = enetModel , "Elastic Net (Trans)" = enetModel.trans , 
                               "Bagged Tree" = treebagModel , "Bagged Tree (Trans)" = treebagModel.trans , 
                               "Cond Inf Tree" = ctreeModel , "Cond Inf Tree (Trans)" = ctreeModel.trans
                               ))
parallelplot(allResamples)
parallelplot(allResamples , metric = "Rsquared")

## serializing bechmarks 
write.csv(perf.grid,quote=F,row.names=F,file=paste0(getBasePath(),"/linear_reg/__benchmarks.zat"))
