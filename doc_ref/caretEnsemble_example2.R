#Setup
rm(list = ls(all = TRUE)) 
gc(reset=TRUE)
set.seed(1234) #From random.org

#Libraries
library(caret)
library(devtools)
install_github('caretEnsemble', 'zachmayer') #Install zach's caretEnsemble package
library(caretEnsemble)

#Data
library(mlbench)
dat <- mlbench.xor(500, 2)
X <- data.frame(dat$x)
Y <- factor(ifelse(dat$classes=='1', 'Yes', 'No'))

#Split train/test
train <- runif(nrow(X)) <= .66

#Setup CV Folds
#returnData=FALSE saves some space
folds=5
repeats=1
myControl <- trainControl(method='cv', number=folds, repeats=repeats, 
                          returnResamp='none', classProbs=TRUE,
                          returnData=FALSE, savePredictions=TRUE, 
                          verboseIter=TRUE, allowParallel=TRUE,
                          summaryFunction=twoClassSummary,
                          index=createMultiFolds(Y[train], k=folds, times=repeats))
PP <- c('center', 'scale')

#Train some models
#model1 <- train(X[train,], Y[train], method='gbm', trControl=myControl,
#                tuneGrid=expand.grid(.n.trees=500, .interaction.depth=15, .shrinkage = 0.01))

model2 <- train(X[train,], Y[train], method='blackboost', trControl=myControl)
#model3 <- train(X[train,], Y[train], method='parRF', trControl=myControl)
model4 <- train(X[train,], Y[train], method='mlpWeightDecay', trControl=myControl, trace=FALSE, preProcess=PP)
model5 <- train(X[train,], Y[train], method='knn', trControl=myControl, preProcess=PP)
#model6 <- train(X[train,], Y[train], method='earth', trControl=myControl, preProcess=PP)
model7 <- train(X[train,], Y[train], method='glm', trControl=myControl, preProcess=PP)
model8 <- train(X[train,], Y[train], method='svmRadial', trControl=myControl, preProcess=PP)
model9 <- train(X[train,], Y[train], method='gam', trControl=myControl, preProcess=PP)
model10 <- train(X[train,], Y[train], method='glmnet', trControl=myControl, preProcess=PP)

#####
model_list_big <- caretList(x = X[train,], y = Y[train], 
  trControl=myControl,
  metric='ROC',
  methodList=c('blackboost')
)
model_list_big[['mlpWeightDecay']] <- model4
model_list_big[['knn']] <- model5
model_list_big[['glm']] <- model7
model_list_big[['svmRadial']] <- model8
model_list_big[['gam']] <- model9
model_list_big[['glmnet']] <- model10


#####
#Make a list of all the models
##all.models <- list(model1, model2, model3, model4, model5, model6, model7, model8, model9, model10)
all.models <- list( model2, model4, model5,  model7, model8, model9, model10)
names(all.models) <- sapply(all.models, function(x) x$method)
sort(sapply(all.models, function(x) min(x$results$ROC)))

#Make a greedy ensemble - currently can only use RMSE
#greedy <- caretEnsemble(all.models, iter=1000L)
greedy <- caretEnsemble(model_list_big, iter=1000L)
sort(greedy$weights, decreasing=TRUE)
greedy$error

#Make a linear regression ensemble
#linear <- caretStack(all.models, method='glm', trControl=trainControl(method='cv'))
linear <- caretStack(model_list_big, method='glm', trControl=trainControl(method='cv'))
linear$error

#Predict for test set:
library(caTools)
preds <- data.frame(sapply(all.models, function(x){predict(x, X[!train,], type='prob')[,2]}))
preds$ENS_greedy <- predict(greedy, newdata=X[!train,])
preds$ENS_linear <- predict(linear, newdata=X[!train,], type='prob')[,2]
sort(data.frame(colAUC(preds, Y[!train])))