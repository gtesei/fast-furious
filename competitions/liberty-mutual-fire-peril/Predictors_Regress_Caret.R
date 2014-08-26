library(quantreg)
library(data.table)
library(glmnet)
library(class)
library(caret)

WeightedGini <- function(solution, weights, submission){
  df = data.frame(solution = solution, weights = weights, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df$random = cumsum((df$weights/sum(df$weights)))
  totalPositive <- sum(df$solution * df$weights)
  df$cumPosFound <- cumsum(df$solution * df$weights)
  df$Lorentz <- df$cumPosFound / totalPositive
  n <- nrow(df)
  gini <- sum(df$Lorentz[-1]*df$random[-n]) - sum(df$Lorentz[-n]*df$random[-1])
  return(gini)
}

NormalizedWeightedGini <- function(solution, weights, submission) {
  WeightedGini(solution, weights, submission) / WeightedGini(solution, weights, solution)
}


encodeCategoricalFeature = function(ddata,i,facts.in=NULL) {
  fact_max = 0
  fact_min = 1 
  facts = NULL
  if (is.null(facts.in)) {
    facts = na.omit(unique(ddata[,i]))
    fact_max = length(facts)
  } else {
    fact_max = length(facts.in)
    facts = facts.in
  }
  
  mm = matrix(rep(0,dim(ddata)[1]),nrow=dim(ddata)[1],ncol=fact_max)
  col_name = colnames(ddata)[i]
  colnames(mm) = paste(paste(col_name,"_",sep=''),facts,sep='')
  for (j in fact_min:fact_max) {
    mm[,j] = ddata [,i] == facts[j]
  }  
  ddata = cbind(ddata,mm)
  #ddata = ddata[,-i]
}

bootstrap = function (traindata , labels , positive.ratio = 0.5) {
  pos.idx.set = which(labels == 1)
  neg.idx.set = which(labels == 0)
  
  traindata.pos.samples = floor(dim(traindata)[1] * positive.ratio)
  traindata.neg.samples = dim(traindata)[1] - traindata.pos.samples
  
  pos.idx = sample(pos.idx.set , traindata.pos.samples , T)
  neg.idx = sample(neg.idx.set , traindata.neg.samples , T)
  
  all.idx = c(pos.idx,neg.idx)
  
  traindata.boot = traindata[all.idx , ]
  traindata.boot
}

getPvalueTypeIError = function(x,y) {
  test = NA
  pvalue = NA
  
  ## type casting and understanding stat test 
  if (class(x) == "integer") x = as.numeric(x)
  if (class(y) == "integer") y = as.numeric(y)
  
  if ( class(x) == "factor" & class(y) == "numeric" ) {
    # C -> Q
    test = "ANOVA"
  } else if (class(x) == "factor" & class(y) == "factor" ) {
    # C -> C
    test = "CHI-SQUARE"
  } else if (class(x) == "numeric" & class(y) == "numeric" ) {
    test = "PEARSON"
  }  else {
    # Q -> C 
    # it performs anova test x ~ y 
    test = "ANOVA"
    tmp = x 
    x = y 
    y = tmp 
  }
  
  ## performing stat test and computing p-value
  if (test == "ANOVA") {                
    test.anova = aov(y~x)
    pvalue = summary(test.anova)[[1]][["Pr(>F)"]][1]
  } else if (test == "CHI-SQUARE") {    
    test.chisq = chisq.test(x = x , y = y)
    pvalue = test.chisq$p.value
  } else {                             
    ###  PEARSON
    test.corr = cor.test(x =  x , y =  y)
    pvalue = test.corr$p.value
  }
  
  pvalue
}

getPerfOnTestSet = function(x , y , k = 4) {
  reg.type = NA
  perf = NA
  
  if ( class(y) == "factor" & length(levels(y)) > 2 ) stop("TODO Multinomial support.")
  
  ## type casting and understanding stat test 
  if (class(x) == "integer") x = as.numeric(x)
  if (class(y) == "integer") y = as.numeric(y)
  
  if ( class(x) == "factor" & class(y) == "numeric" ) {
    # C -> Q
    reg.type = "LINEAR_REG"
  } else if (class(x) == "factor" & class(y) == "factor" ) {
    # C -> C
    reg.type = "LOGISTIC_REG"
  } else if (class(x) == "numeric" & class(y) == "numeric" ) {
    reg.type = "LINEAR_REG"
  }  else {
    # Q -> C 
    reg.type = "LOGISTIC_REG"
  }
  
  y = y[! is.na(x)]
  x = x[! is.na(x)]
  ## k-fold cross validation 
  folds = kfolds(k,length(x))
  cv=rep(x = -1 , k)
  
  if (reg.type == "LINEAR_REG") {
    for(j in 1:k) {  
      train.df = data.frame(y = y[folds != j] , x = x[folds != j])
      test.df = data.frame(x = x[folds == j])
      fit = lm(y ~ x , data=train.df)
      pred = predict(fit , test.df)
      cv[j]= mean(abs((y[folds == j] - pred)))
    }
  } else { #LOGISTIC_REG
    for(j in 1:k) {  
      train.df = data.frame(y = y[folds != j] , x = x[folds != j])
      test.df = data.frame(x = x[folds == j])
      fit = glm(y ~ x , data=train.df , family=binomial)
      pred.probs = predict(fit , test.df , type = "response")
      label0 = rownames(contrasts(y))[1]
      label1 = rownames(contrasts(y))[2]
      pred = ifelse(pred.probs > 0.5 , label1 , label0)
      cv[j]= mean( y[folds == j] == pred  )
    }
  }
  
  perf=mean(cv)
  perf
}

getLinerRegPerfXval = function(form, train, k=4) {
  
  #k = 4 
  #form = "target ~ var11 + var12 + var13 + var14 + var15 + var16 + var17 "
  
  folds = kfolds(k,dim(train)[1])
  cv=rep(x = -1 , k)
  
  for(j in 1:k) {  
    traindata = train[folds != j,]
    xvaldata = train[folds == j,]
    fit = lm(as.formula(form) , data=traindata)
    #fit = rq(as.formula(form) , data=traindata , tau = 0.1 )
    pred = predict(fit , xvaldata)
    pred = ifelse(is.na(pred) , 0 , pred)    ## TODO BETTER 
    #cv[j]= mean(abs((xvaldata$target - pred)))
    cv[j] = NormalizedWeightedGini (xvaldata$target, xvaldata$var11, pred)
  }
  
  perf=mean(cv)
  perf
}

getPvalueInteractionTerms = function(response,features,pvalueFeatures,pvalue.threshold = 0.05) {
  
  n = dim(features)[2] * (dim(features)[2] - 1) / 2 
  
  label.formula = rep(NA, n)
  pValue <- rep(NA, n)
  is.na <- rep(NA, n)
  perf.xval <- rep(NA, n)
  
  p = 1 
  for (i in 1:(dim(features)[2]) ) {
    if ( class(features[,i]) == "factor" ) next 
    if ( ! is.na(pvalueFeatures$pValue[pvalueFeatures$label == colnames(features[i])]) & pvalueFeatures$pValue[pvalueFeatures$label == colnames(features[i])] > pvalue.threshold) next 
    
    for (j in (i+1):(dim(features)[2]) ) {
      if ( class(features[,j]) == "factor" ) next 
      if ( ! is.na(pvalueFeatures$pValue[pvalueFeatures$label == colnames(features[j])]) & pvalueFeatures$pValue[pvalueFeatures$label == colnames(features[j])] > pvalue.threshold) next
      
      label.formula[p] = paste0(paste0( colnames(features[i]) , ":" ) , colnames(features[j]) )
      pValue[p] <- getPvalueTypeIError(x = (features[,i]*features[,j]), y = response)
      perf.xval[p] = getPerfOnTestSet(x = (features[,i]*features[,j]) , y = response)
      is.na[p] = sum(is.na(  features[,i]*features[,j]  )) / length(features[,i]) 
      
      p = p + 1
    }
  }
  
  is.significant = ifelse(pValue < 0.05,T,F)
  
  ## trim 
  idx = which(! is.na(label.formula) )
  label.formula = label.formula[idx]
  pValue <- pValue[idx]
  is.na <- is.na[idx]
  perf.xval <- perf.xval[idx]
  is.significant = is.significant[idx]
  
  ## return 
  data.frame(label = label.formula, pValue , is.significant , is.na , perf.xval)
}

getPvalueFeatures = function(response,features , p = 1) {
  
  label.formula = rep(NA, dim(features)[2])
  pValue <- rep(NA, dim(features)[2])
  is.na <- rep(NA, dim(features)[2])
  perf.xval <- rep(NA, dim(features)[2])
  
  for (i in 1:(dim(features)[2])) {
    if ( p == 1) {
      label.formula[i] = colnames(features[i])
      pValue[i] <- getPvalueTypeIError(x = features[,i], y = response)
      perf.xval[i] = getPerfOnTestSet(x = features[,i], y = response)
    } else {
      ## label 
      if (p == 2) label.formula[i] = paste0(paste0("I(",colnames(features[i])),"^2)")
      if (p == 3) label.formula[i] = paste0(paste0("I(",colnames(features[i])),"^3)")
      if (p == 4) label.formula[i] = paste0(paste0("I(",colnames(features[i])),"^4)")
      if (p == 5) label.formula[i] = paste0(paste0("I(",colnames(features[i])),"^5)")
      if (p >  5) stop("p > 5 not supported.")
      
      ## pvalue , perf.xval
      if (class(features[,i]) == "factor") {
        pValue[i] = NA
        perf.xval[i] = NA
      } else {
        x.poly = features[,i]^p
        pValue[i] = getPvalueTypeIError(x = x.poly, y = response)
        perf.xval[i] = getPerfOnTestSet(x = x.poly, y = response)
      }
    }
    
    is.na[i] = sum(is.na(features[,i])) / length(features[,i]) 
  }
  
  is.significant = ifelse(pValue < 0.05,T,F)
  data.frame(label = label.formula, pValue , is.significant , is.na , perf.xval)
}

kfolds = function(k,data.length) {
  k = min(k,data.length)
  folds = rep(NA,data.length)
  labels = 1:data.length
  st = floor(data.length/k)
  al_labels = NULL
  for (s in 1:k) {
    x = NULL
    if (is.null(al_labels))
      x = sample(labels,st)
    else
      x = sample(labels[-al_labels],st)
    
    folds[x] = s
    if (is.null(al_labels))
      al_labels = x
    else
      al_labels = c(al_labels,x)
  }
  ss = 1
  for (s in 1:length(folds)){
    if (is.na(folds[s])) {
      folds[s] = ss
      ss = ss + 1
    } 
  }
  folds
}

##############  Loading data sets (train, test, sample) ... 

#base.path = "C:/docs/ff/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"
base.path = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"

train.fn = "train.csv"
test.fn = "test.csv"
sampleSub.fn = "sampleSubmission.csv"

train.tab = fread(paste(base.path,train.fn,sep="") , header = TRUE , sep=","  )

######### train set transformations ...

train = as.data.frame.matrix(train.tab) 
## set NAs
train$var1 = ifelse(train$var1 == "Z" , NA , train$var1)
train$var2 = ifelse(train$var2 == "Z" , NA , train$var2)
train$var3 = ifelse(train$var3 == "Z" , NA , train$var3)
train$var4 = ifelse(train$var4 == "Z" , NA , train$var4)
train$var5 = ifelse(train$var5 == "Z" , NA , train$var5)
train$var6 = ifelse(train$var6 == "Z" , NA , train$var6)
train$var7 = ifelse(train$var7 == "Z" , NA , train$var7)
train$var8 = ifelse(train$var8 == "Z" , NA , train$var8)
train$var9 = ifelse(train$var9 == "Z" , NA , train$var9)

## set correct classes for regression 
train$var1 = as.numeric(train$var1)
train$var2 = as.factor(train$var2)
train$var3 = as.factor(train$var3)

## TODO BETTER: perdi l'informazione sul secondo livello  
#train$var4_4 = factor(train$var4 , ordered = T)
train$var4_4 = factor(train$var4 )
#train$var4 = factor( ifelse(is.na(train$var4), NA , substring(train$var4 , 1 ,1) ) , ordered = T)
train$var4 = factor( ifelse(is.na(train$var4), NA , substring(train$var4 , 1 ,1) ) )

train$var5 = as.factor(train$var5)
train$var6 = as.factor(train$var6)
train$var7 = as.numeric(train$var7)
train$var8 = as.numeric(train$var8)
train$var9 = as.factor(train$var9)
train$dummy = as.factor(train$dummy)

train$target_0 = factor(ifelse(train$target == 0,0,1))

## all0s model 
all0.mae = mean(abs(train$target))
all0.acc = mean(train$target_0 == 0)
all0.mae
all0.acc

### merge reg
p.thresold = 0.05 
na.thresold = 0.5 

### focusing only on no-NAs observation to asses the model 
var13.na = is.na(train$var13)
geodemVar24.na = is.na(train$geodemVar24)  
weatherVar47.na = is.na(train$weatherVar47) 
var8.na = is.na(train$var8) 
var10.na = is.na(train$var10)  
var4.na = is.na(train$var4) 
geodemVar37.na = is.na(train$geodemVar37)
var4_4.na = is.na(train$var4_4)
geodemVar13.na = is.na(train$geodemVar13)
weatherVar118.na = is.na(train$weatherVar118)
weatherVar227.na = is.na(train$weatherVar227)
var11.na = is.na(train$var11)
dummy.na = is.na(train$dummy)
geodemVar8.na = is.na(train$geodemVar8)
geodemVar26.na = is.na(train$geodemVar26)

all.na = geodemVar26.na | dummy.na | geodemVar8.na | var13.na | geodemVar24.na | weatherVar47.na | var8.na | var10.na | var4.na | geodemVar37.na | var4_4.na | geodemVar13.na | weatherVar118.na | weatherVar227.na  | var11.na 

sum(all.na) / dim(train)[1] ## 9% 
train.not.na = subset(train , ! all.na)

### pulizia ...
rm(train) 
rm(train.tab)
gc() 
var.er = "target|var13|geodemVar24|weatherVar47|var8|var10|var4|geodemVar37|var4_4|var11"
var.idx = grep(pattern = var.er , names (train.not.na) )
train.not.na = train.not.na[, var.idx]
train.not.na = train.not.na[,-11]

### formulas ...
form = "target ~ var13:geodemVar24 + var13:weatherVar47 + var8:var13 + I(var13^2) + I(var13^3) + I(var13^4) + var10 + var4 + I(var10^2) + var10:var13 + var13 + var13:geodemVar37 + I(var13^5) + I(var10^5) + var4_4 + var8:var10" 
form.error = "error ~ geodemVar19 + weatherVar67 + geodemVar30 + geodemVar29 + I(weatherVar133^2) + I(geodemVar19^2) + geodemVar30:weatherVar104 + geodemVar19:weatherVar67 + geodemVar19:weatherVar66"

####### normalizing 
# train.not.na.tmp = train.not.na[,-c(2,10)]
# trans <- preProcess(train.not.na.tmp, method = c("center", "scale") )
# train.not.na.tmp = predict(trans,train.not.na.tmp)
# train.not.na.tmp = cbind(train.not.na.tmp , var4 = train.not.na$var4)
# train.not.na.tmp = cbind(train.not.na.tmp, var4_4 = train.not.na$var4_4)
# train.not.na = train.not.na.tmp
# rm(train.not.na.tmp)

### matrix data 
x = model.matrix ( as.formula(form)  , train.not.na )[,-1]
y = train.not.na$target

## best lambda - mail model 
bestlam = 0.2494176 

## best lambda - error model 
bestlam.err = 0.9

##### main model 
ptm <- proc.time()

k = 10 
folds = kfolds(k,dim(train.not.na)[1])
cv=rep(x = -1 , k)
for(j in 1:k) {  
  traindata = train.not.na[folds != j,]
  xvaldata = train.not.na[folds == j,]
  fit = glmnet (x[folds != j,] , y[folds != j]  , alpha = 0, lambda = bestlam , thresh = 1e-12)
  pred = as.numeric( predict(fit , s = bestlam , newx = x[folds == j,] )  )
  prf = NormalizedWeightedGini (xvaldata$target, xvaldata$var11, pred)
  cv[j] = prf
  cat("finished iteration n. " , j ," got " , prf , " \n")   
}
perf.main=mean(cv)
perf.main.sd = sd(cv)
perf.main    # 0.31846
perf.main.sd # 0.09441328

tm = proc.time() - ptm
cat("Time elapsed in loop:",tm,"\n")

##### main model + error model 
k = 10
folds = kfolds(k,dim(train.not.na)[1])
cv=rep(x = -1 , k)
for(j in 1:k) {  
  #err.idx = (j %% k) + 1  
  
  #traindata = train.not.na[folds != j & folds != err.idx ,]
  traindata = train.not.na[folds != j ,]
  xvaldata = train.not.na[folds == j,]
  #errordata = train.not.na[folds == err.idx,]
    
  fit = glmnet (x[folds != j ,] , y[folds != j ]  , alpha = 0, lambda = bestlam , thresh = 1e-12)
  #pred.err = as.numeric( predict(fit , s = bestlam , newx = x[folds == err.idx,] )  )
  pred = as.numeric( predict(fit , s = bestlam , newx = x[folds == j,] )  )
  
  #error = pred.err - errordata$target
  error = as.numeric( predict(fit , s = bestlam , newx = x[folds != j ,] )  ) - traindata$target
  x.err = model.matrix ( as.formula(form.error)  , traindata )[,-1]
  form.error.2 = "target ~ geodemVar19 + weatherVar67 + geodemVar30 + geodemVar29 + I(weatherVar133^2) + I(geodemVar19^2) + geodemVar30:weatherVar104 + geodemVar19:weatherVar67 + geodemVar19:weatherVar66"
  x.err.xval = model.matrix ( as.formula(form.error.2)  , xvaldata )[,-1]
  fit.err = glmnet (x.err , error  , alpha = 0, lambda = bestlam.err , thresh = 1e-12)
  pred.error = as.numeric( predict(fit.err , s = bestlam.err , newx = x.err.xval )  )
  
  pred.final = pred - pred.error
  
  prf = NormalizedWeightedGini (xvaldata$target, xvaldata$var11, pred.final)
  cv[j] = prf
  cat("finished iteration n. " , j ," got " , prf , " \n")
}
perf.main.err=mean(cv)
perf.main.err.sd = sd(cv)
perf.main.err #0.3203446
perf.main.err.sd #0.101676


################################## CARET #####################

################# lm 
ptm <- proc.time()
k = 10 
folds = kfolds(k,dim(train.not.na)[1])
cv=rep(x = -1 , k)
for(j in 1:k) {  
  traindata = train.not.na[folds != j,]
  xvaldata = train.not.na[folds == j,]
  fit = train(as.formula(form) , data=traindata , method="lm" , preProcess=c("center","scale") 
              , trControl = trainControl(method = "cv" , number = 10) ) 
  pred = as.numeric( predict(fit , xvaldata )  )
  prf = NormalizedWeightedGini (xvaldata$target, xvaldata$var11, pred)
  cv[j] = prf
  cat("finished iteration n. " , j ," got " , prf , " \n")   
}
perf.lm=mean(cv)
perf.lm.sd = sd(cv)
perf.lm    ## 0.2810049
perf.lm.sd ## 0.09023256
tm = proc.time() - ptm
cat("Time elapsed in loop:",tm,"\n")

# finished iteration n.  1  got  0.3958944  
# finished iteration n.  2  got  0.30311
# finished iteration n.  3  got  0.3065531 
# finished iteration n.  4  got  0.3687263 
# finished iteration n.  5  got  0.1592254
# finished iteration n.  6  got  0.1430399
# finished iteration n.  7  got  0.2834239  
# finished iteration n.  8  got  0.2166975   
# finished iteration n.  9  got  0.3523733  

################# rlm 
ptm <- proc.time()
k = 10 
folds = kfolds(k,dim(train.not.na)[1])
cv=rep(x = -1 , k)
for(j in 1:k) {  
  traindata = train.not.na[folds != j,]
  xvaldata = train.not.na[folds == j,]
  fit = train(as.formula(form) , data=traindata , method="rlm" , preProcess=c("center","scale") 
              , trControl = trainControl(method = "cv" , number = 10) ) 
  pred = as.numeric( predict(fit , xvaldata )  )
  prf = NormalizedWeightedGini (xvaldata$target, xvaldata$var11, pred)
  cv[j] = prf
  cat("finished iteration n. " , j ," got " , prf , " \n")   
}
perf.lm=mean(cv)
perf.lm.sd = sd(cv)
perf.lm    
perf.lm.sd
tm = proc.time() - ptm
cat("Time elapsed in loop:",tm,"\n")

# Loading required package: MASS
# Error in train.default(x, y, weights = w, ...) : 
#   final tuning parameters could not be determined
# In addition: There were 12 warnings (use warnings() to see them)

################# pls 

ptm <- proc.time()
k = 10 
folds = kfolds(k,dim(train.not.na)[1])
cv=rep(x = -1 , k)
for(j in 1:k) {  
  traindata = train.not.na[folds != j,]
  xvaldata = train.not.na[folds == j,]
  fit = train(as.formula(form) , data=traindata , method="pls" , preProcess=c("center","scale") 
              , tuneLength=40 , trControl = trainControl(method = "cv" , number = 10) ) 
  pred = as.numeric( predict(fit , xvaldata )  )
  prf = NormalizedWeightedGini (xvaldata$target, xvaldata$var11, pred)
  cv[j] = prf
  cat("finished iteration n. " , j ," got " , prf , " \n")   
}
perf.lm=mean(cv)
perf.lm.sd = sd(cv)
perf.lm    
perf.lm.sd
tm = proc.time() - ptm
cat("Time elapsed in loop:",tm,"\n")

# finished iteration n.  1  got  0.3618932  
# finished iteration n.  2  got  0.2485195  
# finished iteration n.  3  got  0.2818237 
# finished iteration n.  4  got  0.2507527 
# finished iteration n.  5  got  0.3371699  
# finished iteration n.  6  got  0.2694991

################# ridge 
ptm <- proc.time()
ridgeGrid <- data.frame(.lambda = seq(0, .1, length = 15))

k = 10 
folds = kfolds(k,dim(train.not.na)[1])
cv=rep(x = -1 , k)
for(j in 1:k) {  
  traindata = train.not.na[folds != j,]
  xvaldata = train.not.na[folds == j,]
  fit = train(as.formula(form) , data=traindata , method="ridge" , preProcess=c("center","scale") 
              , tuneGrid = ridgeGrid , trControl = trainControl(method = "cv" , number = 10) ) 
  pred = as.numeric( predict(fit , xvaldata )  )
  prf = NormalizedWeightedGini (xvaldata$target, xvaldata$var11, pred)
  cv[j] = prf
  cat("finished iteration n. " , j ," got " , prf , " \n")
  fit
}
perf.ridge=mean(cv)
perf.ridge.sd = sd(cv)
perf.ridge    
perf.ridge.sd
tm = proc.time() - ptm
cat("Time elapsed in loop:",tm,"\n")

# finished iteration n.  1  got  0.4064847  -- 

################# MARS 
ptm <- proc.time()

marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)
# Fix the seed so that the results can be reproduced
set.seed(100)

k = 10 
folds = kfolds(k,dim(train.not.na)[1])
cv=rep(x = -1 , k)
for(j in 1:k) {  
  traindata = train.not.na[folds != j,]
  xvaldata = train.not.na[folds == j,]
  fit <- train(as.formula(form) , data=traindata,
                    method = "earth",
                    # Explicitly declare the candidate models to test
                    tuneGrid = marsGrid,
                    trControl = trainControl(method = "cv" , number = 10))
  pred = as.numeric( predict(fit , xvaldata )  )
  prf = NormalizedWeightedGini (xvaldata$target, xvaldata$var11, pred)
  cv[j] = prf
  cat("finished iteration n. " , j ," got " , prf , " \n")
  fit
}
perf.mars=mean(cv)
perf.mars.sd = sd(cv)
tm = proc.time() - ptm
cat("Time elapsed in loop:",tm,"\n")
perf.mars    
perf.mars.sd

