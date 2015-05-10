library(caret)
library(ggplot2)
library(pls)

data(economics)

timeSlices <- createTimeSlices(1:nrow(economics), 
                               initialWindow = 36, horizon = 12, fixedWindow = TRUE)

str(timeSlices,max.level = 1)

trainSlices <- timeSlices[[1]]
testSlices <- timeSlices[[2]]

plsFitTime <- train(unemploy ~ pce + pop + psavert,
                    data = economics[trainSlices[[1]],],
                    method = "pls",
                    preProc = c("center", "scale"))

####
pred <- predict(plsFitTime,economics[testSlices[[1]],])

true <- economics$unemploy[testSlices[[1]]]

plot(true, col = "red", ylab = "true (red) , pred (blue)", ylim = range(c(pred,true)))
points(pred, col = "blue")



####
for(i in 1:length(trainSlices)){
  plsFitTime <- train(unemploy ~ pce + pop + psavert,
                      data = economics[trainSlices[[i]],],
                      method = "pls",
                      preProc = c("center", "scale"))
  pred <- predict(plsFitTime,economics[testSlices[[i]],])
  
  
  true <- economics$unemploy[testSlices[[i]]]
  plot(true, col = "red", ylab = "true (red) , pred (blue)", 
       main = i, ylim = range(c(pred,true)))
  points(pred, col = "blue") 
}

####

myTimeControl <- trainControl(method = "timeslice", 
                              initialWindow = 36, horizon = 12, fixedWindow = TRUE)


plsFitTime <- train(unemploy ~ pce + pop + psavert,
                    data = economics,
                    method = "pls",
                    preProc = c("center", "scale"), 
                    trControl = myTimeControl)

plsFitTime$bestTune

