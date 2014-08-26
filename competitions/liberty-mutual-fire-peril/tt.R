
######### Q -> C 
x = rnorm(n = 1000 , mean = 100 , sd =  20000)
y = factor(ifelse(x > 100 , "Yes" , "No" ))
train.df = data.frame(y = y , x = x)
mod = glm(y ~ x , family = binomial , data = train.df )
summary(mod)
x.test = rnorm(n = 500 , mean = 100 , sd =  20000)
test.df = data.frame(x = x.test)
pred = predict(mod , test.df , type = "response")
label0 = rownames(contrasts(y))[1]
label1 = rownames(contrasts(y))[2]
pred.test = ifelse(pred > 0.5 , label1 , label0)
y.test = ifelse(x.test > 100 , "Yes" , "No" )
table(y.test , pred.test)

test.anova = aov(x~y)
pvalue = summary(test.anova)[[1]][["Pr(>F)"]][1]
pvalue

## quadratic 
x = rnorm(n = 1000 , mean = 100 , sd =  50)
z = rnorm(n = 1000 , mean = 100 , sd =  50)
y = factor(ifelse( (x^2 + z^2) > 5000 , "Yes" , "No" ))
summary(y)
train.df = data.frame(y = y , x = x , z = z)

mod = glm(y ~ I(x^2) + I(z^2), family = binomial , data = train.df )
summary(mod)

x.test = rnorm(n = 500 , mean = 100 , sd =  50)
z.test = rnorm(n = 500 , mean = 100 , sd =  50)
test.df = data.frame(x = x.test , z = z.test)
pred = predict(mod , test.df , type = "response")
label0 = rownames(contrasts(y))[1]
label1 = rownames(contrasts(y))[2]
pred.test = ifelse(pred > 0.5 , label1 , label0)
y.test = factor(ifelse((x.test^2 + z.test^2) > 5000 , "Yes" , "No" ))
summary(y.test)
table(y.test , pred.test)

test.anova = aov(x~y)
pvalue = summary(test.anova)[[1]][["Pr(>F)"]][1]
pvalue

## interaction terms  
x = rnorm(n = 1000 , mean = 100 , sd =  50)
z = rnorm(n = 1000 , mean = 100 , sd =  50)
y = factor(ifelse( (x * z) > 5000 , "Yes" , "No" ))
summary(y)
train.df = data.frame(y = y , x = x , z = z)

mod = glm(y ~ x:z, family = binomial , data = train.df )
summary(mod)

x.test = rnorm(n = 500 , mean = 100 , sd =  50)
z.test = rnorm(n = 500 , mean = 100 , sd =  50)
test.df = data.frame(x = x.test , z = z.test)
pred = predict(mod , test.df , type = "response")
label0 = rownames(contrasts(y))[1]
label1 = rownames(contrasts(y))[2]
pred.test = ifelse(pred > 0.5 , label1 , label0)
y.test = factor(ifelse((x.test * z.test) > 5000 , "Yes" , "No" ))
summary(y.test)
table(y.test , pred.test)

test.anova = aov(x~y)
pvalue = summary(test.anova)[[1]][["Pr(>F)"]][1]
pvalue



######### Q -> Q
x = rnorm(n = 1000 , mean = 100 , sd =  20)
y = 3 * x + 3 
train.df = data.frame(y = y , x = x)
mod = lm(y ~ x ,  data = train.df )
summary(mod)
x.test = rnorm(n = 500 , mean = 100 , sd =  20)
y.test = 3 * x.test + 3 
test.df = data.frame(x = x.test)
pred = predict(mod , test.df)
mean(1/length(x.test)*(y.test - pred)^2)

test.corr = cor.test(x =  x , y =  y)
pvalue = test.corr$p.value
pvalue


### quadratic 
x = rnorm(n = 1000 , mean = 100 , sd =  20)
y = 3 * x^2 + 3 
train.df = data.frame(y = y , x = x)
mod = lm(y ~ x ,  data = train.df )
summary(mod)
x.test = rnorm(n = 500 , mean = 100 , sd =  20)
y.test = 3 * x.test^2 + 3 
test.df = data.frame(x = x.test)
pred = predict(mod , test.df)
mean(abs(y.test - pred))

test.corr = cor.test(x =  x , y =  y)
pvalue = test.corr$p.value
pvalue

### interaction terms  
x = rnorm(n = 1000 , mean = 100 , sd =  20)
z = rnorm(n = 1000 , mean = 100 , sd =  20)
y = 3 * x*z + 3 
train.df = data.frame(y = y , x = x , z = z)
mod = lm(y ~ x + z,  data = train.df )
summary(mod)
x.test = rnorm(n = 500 , mean = 100 , sd =  20)
z.test = rnorm(n = 500 , mean = 100 , sd =  20)
y.test = 3 * x.test*z.test + 3
test.df = data.frame(x = x.test , z = z.test)
pred = predict(mod , test.df)
mean(abs(y.test - pred))

test.corr = cor.test(x =  x , y =  y)
pvalue = test.corr$p.value
pvalue

test.corr = cor.test(x =  z , y =  y)
pvalue = test.corr$p.value
pvalue


######### C -> C 
x = factor(rep(1000,x=c("A","B","C") , 1000))
y = factor(ifelse(x == "A" , "Yes" , "No"))
train.df = data.frame(y = y , x = x)
mod = glm(y ~ x , data = train.df , family = binomial )
summary(mod)
x.test = factor(rep(1000,x=c("A","B","C") , 500))
y.test = factor(ifelse(x.test == "A" , "Yes" , "No"))
test.df = data.frame(y = y.test , x = x.test)
pred.probs = predict(mod , test.df , type = "response")
pred = ifelse(pred.probs > 0.5 , "Yes" , "No")
table(pred,y.test)
mean(y.test == pred)

test.chisq = chisq.test( x = x , y = y)
pvalue = test.corr$p.value
pvalue

######### C -> Q
x = factor(rep(1000,x=c("A","B","C") , 1000))
y = ifelse(x == "A" , 20 , 2)
train.df = data.frame(y = y , x = x)
mod = lm(y ~ x , data = train.df  )
summary(mod)
x.test = factor(rep(1000,x=c("A","B","C") , 500))
y.test = ifelse(x.test == "A" , 20 , 2)
test.df = data.frame(y = y.test , x = x.test)
pred = predict(mod , test.df )
mean(1/length(x.test)*(y.test - pred)^2)

test.anova = aov(y~x)
pvalue = summary(test.anova)[[1]][["Pr(>F)"]][1]
pvalue



###################
library(glmnet)
base.path = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"
results.linear.reg.fn = "results_reg_linear.csv"
c = read.csv(paste0(base.path,results.linear.reg.fn))
results.linear.reg.ord = results.linear.reg[order(results.linear.reg$perf , decreasing = T) , ]    
x.model = model.matrix(target ~ var13:geodemVar24 + var13:weatherVar47 + var8:var13 
                       + I(var13^2) + I(var13^3) + I(var13^4) + var10 + var4 + I(var10^2) + var10:var13 
                       + var13 + var13:geodemVar37 + I(var13^5) + I(var10^5) + var4_4 
                       + var8:var10 + var8:geodemVar24 + var10:geodemVar24 
                       + geodemVar13:weatherVar118 , train)[,-1]
cv.out = cv.glmnet(x.model , train$target , alpha = 0)



#### CAP 3 

library(caret)
library(corrplot)
library(e1071)
library(AppliedPredictiveModeling)
data(segmentationOriginal)
segData = subset(segmentationOriginal , Case == "Train")
cellID = segData$Cell
class = segData$Class
case = segData$Case
segData = segData[, -(1:3)]
skewness(segData$AngleCh1)
skewValues = apply(segData , 2 , skewness)
Ch1AreaTrans = BoxCoxTrans(segData$AreaCh1)
Ch1AreaTrans
head(segData$AreaCh1)
predict(Ch1AreaTrans , head(segData$AreaCh1))
(819^(-.9)-1)/(-.9)
pcaObject = prcomp(segData , center = T , scale. = T)
percentVariance = pcaObject$sdev^2/sum(pcaObject$sdev^2)*100
percentVariance
head(pcaObject$x[,1:5])
head(pcaObject$rotation[,1:5])
spatialSign(segData)
trans = preProcess(segData,method = c("BoxCox","center","scale","pca"))
trans
transformed = predict(trans,segData)
head(transformed[,1:5])
nearZeroVar(segData)
correlations = cor(segData)
dim(correlations)
correlations[1:4,1:4]
corrplot(correlations, order="hclust")
highCorr = findCorrelation(correlations, cutoff = .75)
length(highCorr)
head(highCorr)
filteredSegData = segData[,-highCorr]
dim(filteredSegData)

#### CAP 4
library(AppliedPredictiveModeling)
data(twoClassData)
str(predictors)
str(classes)

set.seed(1)
trainingRows = createDataPartition(classes , p = 0.8 , list = F)
head(trainingRows)
trainPredictors = predictors[trainingRows , ]
testPredictors = predictors[-trainingRows , ]
testClasses = classes[-trainingRows ]
str(trainPredictors)
str(testPredictors)

set.seed(1)
repeatedSplits = createDataPartition(classes , p = 0.8 , times = 3)
str(repeatedSplits)

set.seed(1)
cvSplits = createFolds(trainingRows , k = 10 , returnTrain = T )
str(cvSplits)
fold1 = cvSplits[[1]]
fold1
cvPredictors1 = trainPredictors[fold1 , ]
str(cvPredictors1)
nrow(cvPredictors1)
nrow(trainPredictors)
trainPredictorsMatrix = as.matrix(trainPredictors)
trainClasses <- classes[trainingRows]
testClasses <- classes[-trainingRows]
knn.fit = knn3(x = trainPredictorsMatrix , y = trainClasses)
knn.fit 
testPred = predict(knn.fit,newdata = testPredictors , type = "class")
head(testPred)
table(testClasses,testPred)

data(GermanCredit)
set.seed(1056)
svmFit = train(Class ~ . , data = GermanCredit , method = "svmRadial" , preProc = c("center","scale") , 
               tuneLenght = 10 , trControl = trainControl(method="repeatedcv", repeats=5) )


##### cap 5 
# Use the 'c' function to combine numbers into a vector
observed <- c(0.22, 0.83, -0.12, 0.89, -0.23, -1.30, -0.15, -1.4,
                  + 0.62, 0.99, -0.18, 0.32, 0.34, -0.30, 0.04, -0.87,
                  + 0.55, -1.30, -1.15, 0.20)

predicted <- c(0.24, 0.78, -0.66, 0.53, 0.70, -0.75, -0.41, -0.43,
                 + 0.49, 0.79, -1.19, 0.06, 0.75, -0.07, 0.43, -0.42,
                 + -0.25, -0.64, -1.26, -0.07)
residualValues <- observed - predicted
summary(residualValues)

# Observed values versus predicted values
# It is a good idea to plot the values on a common scale.
axisRange <- extendrange(c(observed, predicted))
plot(observed, predicted,
       ylim = axisRange,
       xlim = axisRange)
# Add a 45 degree reference line
abline(0, 1, col = "darkgrey", lty = 2)

# Predicted values versus residuals
plot(predicted, residualValues, ylab = "residual")
abline(h = 0, col = "darkgrey", lty = 2)

R2(predicted,observed)
RMSE(predicted,observed)
cor(predicted,observed)


############################
form = "target ~ var13:geodemVar24 + var13:weatherVar47 + var8:var13 + I(var13^2) + I(var13^3) + I(var13^4) + var10 + var4 + I(var10^2) + var10:var13 + var13 + var13:geodemVar37 + I(var13^5) + I(var10^5) + var4_4 + var8:var10" 
var.sel = c("target" ,  "var13", "geodemVar24",  "weatherVar47",  "var8", "var10",  "var4"  , "geodemVar37" , "var4_4" )
var.er = "target|var13|geodemVar24|weatherVar47|var8|var10|var4|geodemVar37|var4_4"
var.idx = grep(pattern = var.er , names (train.not.na) )







