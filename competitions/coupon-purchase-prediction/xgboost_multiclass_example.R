
library(xgboost)
library(methods)

data = iris 
#xgboost take features in [0,numOfClass)
y = ifelse(data$Species=='setosa',0,ifelse(data$Species=='versicolor',1,2))
data = data[,-5]

train = data[1:100 , ]
test = data[101:150,]

x = rbind(train,test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 3,
              "eta" = 0.005,  ## suggested in ESLII
              "gamma" = 0.5,  
              "max_depth" = 25, 
              "subsample" = 0.5 , ## suggested in ESLII
              "nthread" = 10, 
              
              "min_child_weight" = 1 , 
              "colsample_bytree" = 0.5, 
              "max_delta_step" = 1
)

cat(">>Params:\n")
print(param)

Run Cross Valication
cat(">>Cross validation ... \n")

inCV = T
early.stop = cv.nround = 300
bst.cv = NULL
bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                  nfold = 5, nrounds=cv.nround)
print(bst.cv)
early.stop = which(bst.cv$test.mlogloss.mean == min(bst.cv$test.mlogloss.mean) )
cat(">> early.stop: ",early.stop," [test.mlogloss.mean:",bst.cv[early.stop,]$test.mlogloss.mean,"]\n") 

cat(">>Train the model ... \n")
# Train the model
nround = early.stop
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

# Make prediction
pred = predict(bst,x[teind,])
pred = matrix(pred,3,length(pred)/3)
pred = t(pred)
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:3))
pred


  