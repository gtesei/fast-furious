
library(caret)

data = data.frame( a = (1:20 + runif(20)) , b = (21:40 + runif(20)) ) 
y = 1:20

set.seed(123)
tr = sample(1:20 , size = 10)

Xtrain = data[tr,]
ytrain = y[tr]

Xtrain = data[-tr,]
ytest = y[tr]

###
RMSLE = function(pred, obs) {
  #RMSE(pred = pred , obs = obs)
  sqrt(    sum( (log(pred+1) - log(obs+1))^2 )   /length(pred))
}

RMSLECostSummary <- function (data, lev = NULL, model = NULL) {
  c(postResample(data[, "pred"], data[, "obs"]),
    RMSLE = RMSLE(pred = data[, "pred"], obs = data[, "obs"]))
}

ctrl <- trainControl(method = "repeatedcv", 
                     repeats = 5, summaryFunction = RMSLECostSummary )

cubistGrid <- expand.grid(.committees = c(1, 5, 10, 50, 75, 100),
                          .neighbors = c(0, 1, 3, 5, 7, 9))

model <- train(y = ytrain, x = Xtrain , 
               method = "cubist", tuneGrid = cubistGrid, 
               trControl = ctrl , metric = 'RMSLE' , maximize = F)
model$bestTune
min(model$results$RMSE)
min(model$results$RMSLE)
# committees neighbors
# 30         75         9





