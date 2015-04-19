library(caret)
library(Hmisc)
library(data.table)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)
library(fBasics)

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/restaurant-revenue-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/restaurant-revenue-prediction/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/restaurant-revenue-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/restaurant-revenue-prediction/"
  } else if (type == "process") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/data_process"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/data_process/"
  } else {
    stop("unrecognized type.")
  }
  
  if (file.exists(base.path1))  {
    ret = paste0(base.path1,"/")
  } else {
    ret = base.path2
  }
  
  ret
}

buildData.basic = function(train.raw , test.raw) {
  ## remove id 
  train = train.raw[ , -1] 
  test = test.raw[ , -1] 
  
  ## 2014 should be the target year ... so use open date to misure the number of years between open date and the target year 
  train$years.to.target = 2014 - year(as.Date( train.raw[,2] , "%m/%d/%Y"))
  test$years.to.target = 2014 - year(as.Date( test.raw[,2] , "%m/%d/%Y"))
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## City
  l = encodeCategoricalFeature (train[,1] , test[,1] , colname.prefix = "city" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## City Group 
  l = encodeCategoricalFeature (train[,1] , test[,1] , colname.prefix = "city.group" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## Type
  l = encodeCategoricalFeature (train[,1] , test[,1] , colname.prefix = "type" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## extracting y 
  y = train[,38]
  train = train[,-38]
  
  list(train,y,test)
}

#######
verbose = T
source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))
source(paste0( getBasePath("process") , "/Resample_Lib.R"))
source(paste0( getBasePath("process") , "/Regression_Lib.R"))
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))

controlObject <- trainControl(method = "boot", number = 200)

RegModels = c("Average" , "Mode",  
              "LinearReg", "RobustLinearReg", 
              "PLS_Reg" , "Ridge_Reg" , "Enet_Reg" , 
              "KNN_Reg", 
              "SVM_Reg", 
              "BaggedTree_Reg"
              , "RandomForest_Reg"
              , "Cubist_Reg" 
              #, "NNet"
) 

cat("\nRegression models:\n")
print(RegModels)

#######
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

train.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train.csv" , sep=''))) 

test.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                      "test.csv" , sep='')))

####### basic feature processing 
l = buildData.basic(train.raw , test.raw)
train = l[[1]]
y = l[[2]]
test = l[[3]]

####### build the grid 
grid = as.data.frame(matrix(rep(NA,37*5),nrow = 37,ncol = 5))
colnames(grid) = c("predictor.cat","model.winner","best.perf","model.2","perf.2")
grid$predictor.cat = paste("P",1:37,sep='')

###### loop on P1-P37 

for (i in 1:37) {
  cat("********* Assuming categorical ",paste0("P",i)," ********* \n")
  ## making Pi categorical 
  l = encodeCategoricalFeature (train[,i] , test[,i] , colname.prefix = paste0("P",i) , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train.mod = cbind(train,tr)
  test.mod = cbind(test,ts)
  
  train.mod = train.mod[ , -i]
  test.mod = test.mod[ , -i]
  
  ####### feature selection <<<<<<<<<<<<<<
  l = featureSelect (train.mod,test.mod)
  traindata = l[[1]]
  testdata = l[[2]]
  
  ### k-fold 
  l = trainAndPredict.kfold.reg (k = 6,traindata,y,RegModels,controlObject)
  model.winner = l[[1]]
  .grid = l[[2]]
  perf.kfold = l[[3]]
  
  ## finding second model 
  avg = apply(perf.kfold,2,function(x) mean(x))
  avg.ord = sort(avg)
  mod.2.idx = which(avg == avg.ord[2])
  model.2 = colnames(perf.kfold)[mod.2.idx]
  model.2.perf = avg[mod.2.idx]
  
  ### results 
  if (verbose) {
    cat("****** RMSE - each model/fold ****** \n")
    print(perf.kfold)
    cat("\n****** RMSE - mean ****** \n")
    print(.grid)
    cat("\n>>>>>>>>>>>> The winner is ... ",model.winner," [RMSE:",.grid$best.perf,"]\n")
    cat(">>>>>>>>>>>> The second model is ... ",model.2," [RMSE:",model.2.perf,"]\n")
    cat("\n >>> updating grid \n")
  }
  
  grid[i,]$model.winner = model.winner
  grid[i,]$best.perf = .grid$best.perf
  grid[i,]$model.2 = model.2
  grid[i,]$perf.2 = model.2.perf
  
  if (verbose) 
    print(grid[i,])
}

## check 
if (sum(is.na(grid)) > 0) {
  print(grid)
  stop("something wrong (NAs) in grid")
}

##### display results 
grid = grid[order(grid$best.perf,decreasing = F),]
print(grid)
most.likely.cat = grid[1,]$predictor.cat
most.likely.cat.idx = as.numeric(strsplit(x = most.likely.cat , "P")[[1]][[2]])
most.likely.cat.mod = grid[1,]$model.winner

cat("******* Predictor most likely categorical: ",most.likely.cat," - index:",most.likely.cat.idx," ******* \n") 
cat("******* Winner model: ",most.likely.cat.mod," - RMSE:",grid[1,]$best.perf," ******* \n") 

##### choice the best guess and making prediction on test set 
if (verbose) 
  cat(">>> making prediction on test set ... \n") 

i = most.likely.cat.idx
cat("********* Assuming categorical ",paste0("P",i)," ********* \n")
## making Pi categorical 
l = encodeCategoricalFeature (train[,i] , test[,i] , colname.prefix = paste0("P",i) , asNumeric=F)
tr = l[[1]]
ts = l[[2]]

train.mod = cbind(train,tr)
test.mod = cbind(test,ts)

train.mod = train.mod[ , -i]
test.mod = test.mod[ , -i]

####### feature selection <<<<<<<<<<<<<<
l = featureSelect (train.mod,test.mod)
traindata = l[[1]]
testdata = l[[2]]

pred = reg.trainAndPredict( y , 
                            traindata , 
                            testdata , 
                            model.winner , 
                            controlObject, 
                            best.tuning = T)

pred = ifelse(pred >= 1150 , pred , 1150) ## TODO better 

### storing on disk 
write.csv(data.frame(Id = test.raw$Id , Prediction = pred),
          quote=FALSE, 
          file=paste(getBasePath("data"),"mySub_cat_guess.csv") ,
          row.names=FALSE)

write.csv(grid,
          quote=FALSE, 
          file=paste(getBasePath("data"),"grid_cat_guess.csv") ,
          row.names=FALSE)

cat("<<<<< submission/grid stored on disk >>>>>\n")






