library(AppliedPredictiveModeling)
data(concrete)
str(concrete)

str(mixtures)

library(Hmisc)
describe( x =  concrete)
describe( x =  mixtures)

featurePlot(x = concrete[,-9] , y = concrete$CompressiveStrength , 
            between = list(x = 1 , y = 1)  , type = c("g" , "p" , "smooth") )


########### data processing  
library(plyr)
averaged = ddply (mixtures , .(Cement,BlastFurnaceSlag,FlyAsh,Water,Superplasticizer,CoarseAggregate,Age) ,
                  function(x) c(CompressiveStrength = mean (x$CompressiveStrength) ) )

library(caret)
forTraining = createDataPartition( averaged$CompressiveStrength , p = 3/4)[[1]] 

trainSet = averaged[forTraining , ]
testSet = averaged[-forTraining , ]

##### modeling 
modFormula = paste("CompressiveStrength ~ (.)^2 + I(Cement^2) + ",
                   "I(BlastFurnaceSlag^2) + (FlyAsh^2) + I(Water^2) + ", 
                   "I(Superplasticizer^2) + (CoarseAggregate^2) + I(Age^2) ")

modFormula = as.formula(modFormula)

controlObject = trainControl(method = "repeatedcv" , repeats = 5 , number = 10)

####
set.seed(669)
linearReg = train(modFormula , method = "lm" , data = trainSet , trControl = controlObject )
linearReg

set.seed(669)
plModel = train(modFormula , method = "pls" , data = trainSet , trControl = controlObject 
                , preProcess = c("center" , "scale") )
plModel

set.seed(669)
enetGrid = expand.grid(.lambda = c(0 , 0.001 , 0.01 , 0.1) , .fraction = seq(0.05,1,length = 20) )
enetModel = train(modFormula , data = trainSet , method = "enet" , preProcess = c("center","scale") , 
                  tuneGrid = enetGrid , trControl = controlObject)
enetModel


set.seed(669)
earthModel = train(CompressiveStrength  ~ . , data = trainSet , method = "earth" , trControl = controlObject , 
                   tuneGrid = expand.grid(.degree = 1 , .nprune = 2:25) )
earthModel

set.seed(669)
svmModel = train(CompressiveStrength  ~ . , data = trainSet , method = "svmRadial" , tuneLength = 15 
                 , preProcess = c("center","scale") , trControl = controlObject)
svmModel

nnetGrid = expand.grid(.decay=c(0.001,0.01,0.1) , 
                       .size = seq(1,27,by=2) , 
                       .bag=F)
set.seed(669)
nnetModel = train(CompressiveStrength  ~ . , data = trainSet , method = "avNNet" , tuneGrid = nnetGrid , 
                  preProcess=c("center","scale") , linout = T , trace = F , maxit = 1000 , 
                  trControl = controlObject) 
nnetModel

set.seed(669)
rpartModel = train(CompressiveStrength  ~ . , data = trainSet , method = "rpart" , tuneLength = 30 , 
                  trControl = controlObject) 
rpartModel


### .... 
cubistGrid = expand.grid(.committees = c(1,5,10,5075,100) , 
                         .neighbors = c(0,1,3,5,7,9)) 
set.seed(669) 
cbModel = train(CompressiveStrength  ~ . , data = trainSet , 
                              trControl = controlObject , method = "cubist" , tuneGrid = cubistGrid ) 
cbModel

###
age28Data = subset(trainSet , Age == 28)
pp1 = preProcess(age28Data[,-(8:9)] , c("center","scale") )
scaledTrain = predict( pp1, age28Data[,-(8:9)] )
set.seed(91)
startMixture = sample(  1:nrow(age28Data) , 1  )
starters = scaledTrain [ startMixture , 1:7 ]


pool = scaledTrain
index = maxDissim(starters , pool, 14)





