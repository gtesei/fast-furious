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

getPvalueFeatures = function(response,features , p = 1 , computePerfOnTrainSet = F) {
  
  label.formula = rep(NA, dim(features)[2])
  pValue <- rep(NA, dim(features)[2])
  is.na <- rep(NA, dim(features)[2])
  perf.xval <- rep(NA, dim(features)[2])
  
  for (i in 1:(dim(features)[2])) {
    print(i)
    if ( p == 1) {
      label.formula[i] = colnames(features[i])
      pValue[i] <- getPvalueTypeIError(x = features[,i], y = response)
      if (computePerfOnTrainSet) {
        perf.xval[i] = getPerfOnTestSet(x = features[,i], y = response)
      }
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
        if (computePerfOnTrainSet) {
          perf.xval[i] = getPerfOnTestSet(x = x.poly, y = response)
        }
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

myBootstrapSample = function(data,len=-1,p_target_pos=-1) {
  if (len <= 0 ) len = dim(data)[1]
  data.length = min(dim(data)[1],len)
  idx = 1:data.length
  if(p_target_pos < 0) {
    idx = sample(1:data.length, replace = TRUE)
  } else {
    p_stat = sum(data$target > 0) / dim(data)[1]
    cat("bootstrap --> perturbing stat distr from p_target_pos=",p_stat," to p_target_pos=",p_target_pos," ...\n")
    idx_pos = which(data$target > 0) 
    idx_neg = which(data$target <= 0)
    
    n_pos = floor(p_target_pos * dim(data)[1])
    n_neg = dim(data)[1] - n_pos
    
    idx_pos_boot = sample(idx_pos, replace = TRUE , size = n_pos)
    idx_neg_boot = sample(idx_neg, replace = TRUE , size = n_neg)
    
    idx = c(idx_pos_boot,idx_neg_boot)
  }

  data[idx,]
} 

######### train set transformations ...
getData = function(myData.tab , id = F) {
  myData = as.data.frame.matrix(myData.tab) 
  ## set NAs
  myData$var1 = ifelse(myData$var1 == "Z" , NA , myData$var1)
  myData$var2 = ifelse(myData$var2 == "Z" , NA , myData$var2)
  myData$var3 = ifelse(myData$var3 == "Z" , NA , myData$var3)
  myData$var4 = ifelse(myData$var4 == "Z" , NA , myData$var4)
  myData$var5 = ifelse(myData$var5 == "Z" , NA , myData$var5)
  myData$var6 = ifelse(myData$var6 == "Z" , NA , myData$var6)
  myData$var7 = ifelse(myData$var7 == "Z" , NA , myData$var7)
  myData$var8 = ifelse(myData$var8 == "Z" , NA , myData$var8)
  myData$var9 = ifelse(myData$var9 == "Z" , NA , myData$var9)

  ## set correct classes for regression 
  myData$var1 = as.numeric(myData$var1)
  myData$var2 = as.factor(myData$var2)
  myData$var3 = as.factor(myData$var3)
  
  ## TODO BETTER: perdi l'informazione sul secondo livello  
  #myData$var4_4 = factor(myData$var4 , ordered = T)
  myData$var4_4 = factor(myData$var4 )
  #myData$var4 = factor( ifelse(is.na(myData$var4), NA , substring(myData$var4 , 1 ,1) ) , ordered = T)
  myData$var4 = factor( ifelse(is.na(myData$var4), NA , substring(myData$var4 , 1 ,1) ) )
  
  myData$var5 = as.factor(myData$var5)
  myData$var6 = as.factor(myData$var6)
  myData$var7 = as.numeric(myData$var7)
  myData$var8 = as.numeric(myData$var8)
  myData$var9 = as.factor(myData$var9)
  myData$dummy = as.factor(myData$dummy)

  ### pulizia ...
  if (! id ) {
    var.er = "target|var13|geodemVar24|weatherVar47|var8|var10|var4|geodemVar37|var4_4|var11"
    var.idx = grep(pattern = var.er , names (myData) )
    myData = myData[, var.idx]
  } else {
    var.er = "id|target|var13|geodemVar24|weatherVar47|var8|var10|var4|geodemVar37|var4_4|var11"
    var.idx = grep(pattern = var.er , names (myData) )
    myData = myData[, var.idx]
  }
  
  ## garbage collector 
  gc() 
  
  ## return myData
  myData
}

getData4Analysis = function(myData.tab) {
  myData = as.data.frame.matrix(myData.tab) 
  ## set NAs
  myData$var1 = ifelse(myData$var1 == "Z" , NA , myData$var1)
  myData$var2 = ifelse(myData$var2 == "Z" , NA , myData$var2)
  myData$var3 = ifelse(myData$var3 == "Z" , NA , myData$var3)
  myData$var4 = ifelse(myData$var4 == "Z" , NA , myData$var4)
  myData$var5 = ifelse(myData$var5 == "Z" , NA , myData$var5)
  myData$var6 = ifelse(myData$var6 == "Z" , NA , myData$var6)
  myData$var7 = ifelse(myData$var7 == "Z" , NA , myData$var7)
  myData$var8 = ifelse(myData$var8 == "Z" , NA , myData$var8)
  myData$var9 = ifelse(myData$var9 == "Z" , NA , myData$var9)
  
  ## set correct classes for regression 
  myData$var1 = as.numeric(myData$var1)
  myData$var2 = as.factor(myData$var2)
  myData$var3 = as.factor(myData$var3)
  
  ## TODO BETTER: perdi l'informazione sul secondo livello  
  #myData$var4_4 = factor(myData$var4 , ordered = T)
  myData$var4 = factor(myData$var4 )
  #myData$var4 = factor( ifelse(is.na(myData$var4), NA , substring(myData$var4 , 1 ,1) ) , ordered = T)
  #myData$var4 = factor( ifelse(is.na(myData$var4), NA , substring(myData$var4 , 1 ,1) ) )
  
  myData$var5 = as.factor(myData$var5)
  myData$var6 = as.factor(myData$var6)
  myData$var7 = as.numeric(myData$var7)
  myData$var8 = as.numeric(myData$var8)
  myData$var9 = as.factor(myData$var9)
  myData$dummy = as.factor(myData$dummy)
  
  if (! is.null(myData$target) ) {
    myData$target_0 = factor(ifelse(myData$target == 0,0,1))
  }
  
  ## garbage collector 
  gc() 
  
  ## return myData
  myData
}

rowNa = function (row) {
  ifelse (sum(is.na(row)) > 0 , 1,0)
}
getNasRows = function (myData) {
  ret = apply(myData,1, rowNa   )
  as.logical(ret)
}

trainAndPredict = function( form , train , test , type="ridge",p_target_pos=-1) {
  ptm <- proc.time()
  pred = rep(NA,dim(test)[1])
  
  controlObject <- trainControl(method = "cv", number = 10)
  
  if (type == "ridge") { 
    ridgeGrid <- data.frame(.lambda = seq(0, .1, length = 15))
    fit = train(as.formula(form) , data=train , method="ridge" , preProcess=c("center","scale") 
                , tuneGrid = ridgeGrid , trControl = trainControl(method = "cv" , number = 10) ) 
    pred = as.numeric( predict(fit , test )  )
  } else if (type == "mars") {
    marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)
    fit <- train(as.formula(form) , data=train,
                 method = "earth",
                 # Explicitly declare the candidate models to test
                 tuneGrid = marsGrid,
                 trControl = trainControl(method = "cv" , number = 10))
    pred = as.numeric( predict(fit , test )  )
  } else if (type == "myRidge") { 
    
    train.tmp = train[,-c(1,2,10)]
    trans <- preProcess(train.tmp, method = c("center", "scale") )
    
    ## train 
    train.tmp = predict(trans,train.tmp)
    train.tmp = cbind(train.tmp , train[,1])
    names(train.tmp)[length(train.tmp)] = names(train)[1]
    train.tmp = cbind(train.tmp , train[,2])
    names(train.tmp)[length(train.tmp)] = names(train)[2]
    train.tmp = cbind(train.tmp , train[,10])
    names(train.tmp)[length(train.tmp)] = names(train)[10]
   
    ## test 
    test.tmp = test[,-c(1,9)]
    test.tmp = predict(trans,test.tmp)
    test.tmp = cbind(test.tmp , test[,1])
    names(test.tmp)[length(test.tmp)] = names(test)[1]
    test.tmp = cbind(test.tmp , test[,9])
    names(test.tmp)[length(test.tmp)] = names(test)[9]
    
    ## switch 
    train = train.tmp
    test = test.tmp 
    
#     BOOT_ITER = 25 
#     pred.boot=matrix(NA,dim(test)[1],BOOT_ITER, dimnames=list(NULL, paste(1:BOOT_ITER)))
#     for (i in 1:BOOT_ITER) {
#       train.boot = myBootstrapSample(data = train,len=-1,p_target_pos) 
      ### matrix data 
    x = model.matrix ( as.formula(form)  , train )[,-1]
    x.test = model.matrix ( as.formula(form)  , cbind(target = 0 , test) ) [,-1]
    y = train$target
    ## best lambda - mail model 
    bestlam = 0.2494176 
      
    fit = glmnet (x , y  , alpha = 0, lambda = bestlam , thresh = 1e-12)
    pred = as.numeric( predict(fit , s = bestlam , newx = x.test )  )
#    pred.boot[,i] = pred.r
#     }
#     
#     pred = apply(pred.boot ,1,mean)
    
  } else if (type == "pls") { 
    fit <- train(as.formula(form) , data=train, 
                 method = "pls" , 
                 preProcess=c("center","scale") , 
                 trControl= trainControl (method="cv",number=10) )
    pred = as.numeric( predict(fit , test )  )
  } else if (type == "knn") { 
    fit <- train(as.formula(form) , data=train, 
                 method = "knn",
                 # Center and scaling will occur for new predictions too
                 preProc = c("center", "scale"),
                 tuneGrid = data.frame(.k = 1:20),
                 trControl = trainControl(method = "cv" , number = 10))
    pred = as.numeric( predict(fit , test )  )
  } else if (type == "svm") { 
    fit <- train(as.formula(form) , data=train ,  
                     method = "svmRadial",
                     preProc = c("center", "scale"),
                     tuneLength = 14,
                     trControl =  controlObject )
    pred = as.numeric( predict(fit , test )  )
  } else if (type == "m5") { 
    fit <- train(as.formula(form) , data=train ,   method = "M5", trControl = controlObject )
    pred = as.numeric( predict(fit , test )  )
  } else if (type == "cubist") { 
    cubistGrid <- expand.grid(.committees = c(1, 5, 10, 50, 75, 100), .neighbors = c(0, 1, 3, 5, 7, 9)) 
    fit <- train(as.formula(form) , data=train , method = "cubist" , tuneGrid = cubistGrid, trControl = controlObject )
    pred = as.numeric( predict(fit , test )  )
  } else if (type == "ctree2") {  
    fit <- train(as.formula(form) , data=train , method = "ctree2" , 
                 tuneLength = 10,
                 trControl = trainControl(method = "cv" , number=10) )
    pred = as.numeric( predict(fit , test )  )
  } else if (type == "rf") { 
    fit <- train(as.formula(form) , data=train , method = "rf" )
    pred = as.numeric( predict(fit , test )  )
  } else if (type == "gbm") {
    gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, by = 2),
                           .n.trees = seq(100, 1100, by = 100),
                           .shrinkage = c(0.01, 0.1))
    fit <- train(as.formula(form) , data=train , method = "gbm", tuneGrid = gbmGrid, verbose = FALSE)
    pred = as.numeric( predict(fit , test )  )
  }
  
  tm = proc.time() - ptm
  cat("Time elapsed in loop:",tm,"\n")
  
  pred 
}

mergeTestSet = function (test.na , test.nona , pred.na , pred.nona,sampleSub.tab) {
  predI.nona = cbind(id=test.nona$id,pred=pred.nona)
  predI.na = cbind(id=test.na$id , pred=pred.na)
  predI = merge(predI.na,predI.nona,by=c("id","pred"),all=T)
  subI = merge(predI,sampleSub.tab , by=c("id"))
}

writeOnDisk = function (base.path , sub.fn, subI) {
  submit <- gzfile(paste(base.path,sub.fn,sep=""), "wt")
  write.table(data.frame(id=subI$id, target=subI$pred), submit, sep=",", row.names=F, quote=F)
  close(submit)
}

trainPredict4Sub = function(form,train,test.nona,test.na,sampleSub.tab,model) {
  pred.nona = trainAndPredict(form, train, test.nona[,-1], type = model )
  subI = mergeTestSet (test.na , test.nona , 0, pred.nona, sampleSub.tab) 
  writeOnDisk(base.path , paste0(paste0("sub_",model),".csv.gz") , subI)
}

trainPredict4Test = function(form,train,model,train.size=5000,test.size=5000,tests=NULL) {
  ptm <- Sys.time()
  print(ptm)
  tIdx = sample(1:dim(train)[1])
  train.idx = tIdx[1:train.size]
  test.idx = tIdx[(train.size+1):(train.size+test.size)]
  ttrain = train[train.idx,]
  ttest = train[test.idx,]
  pred = trainAndPredict(form, ttrain, ttest[,-1], type = model )
  prf = NormalizedWeightedGini (ttest$target, ttest$var11, pred)
  cat(model,"--> NormalizedWeightedGini: ",prf,"\n")
  tm = Sys.time() - ptm
  cat("Time (min.) elapsed:",tm,"\n")
  if (! is.null(tests)) {
    tests$NormalizedWeightedGini[tests$model==model] = prf
    tests$time[tests$model==model] = tm
  }
  tests
}

transfrom4Skewness = function (train,test) {
  library(e1071)
  
  no = which(as.numeric(lapply(train,is.factor)) == 1)
  noT = which(as.numeric(lapply(test,is.factor)) == 1)
  
  skewValuesTrainBefore <- apply(train[,-no], 2, skewness)
  cat("----> skewValues  before transformation (train) \n\n")
  print(skewValuesTrainBefore)
  
  skewValuesTestBefore <- apply(test[,-noT], 2, skewness)
  cat("----> skewValues  before transformation (test): \n")
  print(skewValuesTestBefore)
  
  idx = (1:15)[-no]
  for (i in idx) {
    varname = colnames(train)[i]
    cat("processing ",varname," ... \n")
    inTest = (sum(colnames(test) == varname) == 1)
    tr = BoxCoxTrans(train[,i])
    print(tr)
    if (! is.na(tr$lambda) ) {
      cat("tranforming train ... \n")
      newVal = predict(tr,train[,i])
      train[,i] = newVal
      cat("skewness after transformation (train): " , skewness(train[,i]), "  \n")
      if (inTest) {
        idxTest = which((colnames(test) == varname) == T )
        cat("tranforming test ... \n")
        newVal = predict(tr,test[,idxTest])
        test[,idxTest] = newVal
        cat("skewness after transformation (test): " , skewness(train[,i]), "  \n")
      } 
    }
  } 
  
  cat("---->  skewValues  before transformation (train) \n")
  print(skewValuesTrainBefore)
  skewValues <- apply(train[,-no], 2, skewness)
  cat("\n---->  skewValues  after transformation (train):  \n")
  print(skewValues)
  
  cat("\n\n\n---->  skewValues  before transformation (test) \n")
  print(skewValuesTestBefore)
  skewValues <- apply(test[,-noT], 2, skewness)
  cat("\n---->  skewValues  after transformation (test):  \n")
  print(skewValues)
  
  list(train,test) 
}

charInd = function (c) {
  ret = which(c == LETTERS)
}
var4FromFactor2Num = function (v) {
  letters = substr(v,1,1) 
  val = as.numeric(sapply(letters,charInd))
  val = val * 10 + as.numeric(substr(v,2,2))
}

rowNa = function (row) {
  ifelse (sum(is.na(row)) > 0 , 1,0)
}
getNasRows = function (myData) {
  ret = 0
  if ( ! is.data.frame(myData) ) {
    ret = sum( is.na(myData) )
  } else { 
    ret = apply(myData,1, rowNa   )
    ret = sum(ret)
  }
  ret
}

getVarClassification = function (predPvalues = predictors.class.linear2 , data = train , th = 0.05) {
  var.name = NULL
  var.index = NULL
  
  for (i in 1:dim(predPvalues)[1]) {
    if ( ! predPvalues[i,]$is.significant ) break 
    
    var = as.character(predPvalues[i,]$label)
    cat ("processing ",var," ...\n")
    idx = as.numeric( grep(pattern = var  , x = colnames(data)) )
    
    var.name.tmp = c(var.name , var)
    var.index.tmp = c(var.index , idx)
      
    sumNA = getNasRows(data[,var.index.tmp])
    th.tmp = sumNA / dim(data)[1]
    
    if (th.tmp < th) {
      var.name = var.name.tmp
      var.index = var.index.tmp
    } else {
      next 
    }
  }
  list(var.name,var.index)
}

##############  Loading data sets (train, test, sample) ... 

#base.path = "C:/docs/ff/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"
base.path = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"

train.fn = "train.csv"
test.fn = "test.csv"
sampleSub.fn = "sampleSubmission.csv"

train.tab = fread(paste(base.path,train.fn,sep="") , header = TRUE , sep=","  )
test.tab = fread(paste(base.path,test.fn,sep="") , header = TRUE , sep=","  )

## training on not NA values ... 
train = getData4Analysis(train.tab)
test = getData4Analysis(test.tab)

dim(train)
dim(test)

## Analysis 
predictors.class.linear2 = getPvalueFeatures( features = train[ , - c(1,2,303)] , response = train$target_0 )
predictors.class.linear2 = predictors.class.linear2[order(predictors.class.linear2$pValue,decreasing = F),]
# predictors.class.linear.fn = paste(base.path,"predictors_class_linear2.csv",sep="")
# write.csv(predictors.class.linear,quote=F,row.names=F,file=predictors.class.linear.fn)
# predictors.class.linear.ord[!is.na(predictors.class.linear.ord$is.significant) & predictors.class.linear.ord$is.na == 0 , ]
# predictors.class.linear.ord[ predictors.class.linear.ord$is.na == 0 , ]

l = getVarClassification(predPvalues = predictors.class.linear2 , data = train , th = 0.05)
var.name = l[[1]]
var.index = l[[2]]

var.er = "target"
for (i in 1:length(var.name)) {
  var.er = paste0(paste0(var.er,"|"),var.name[i])
}

var.er = "target|var11|var4|var13|var10|dummy|weatherVar104|weatherVar31|weatherVar110|weatherVar98|weatherVar69|geodemVar31|var17|geodemVar37"

### just retain the variable higly correlated without NAs -- questi sono per la regression 
#var.er = "target|var13|var11|var10|dummy|var17"
#var.er = "target|var13|var11|var10|dummy|var17|var4|weatherVar104|weatherVar31|weatherVar110|weatherVar98|weatherVar69|weatherVar190|weatherVar194"

## train 
var.idx = grep(pattern = var.er , names (train) )
train.old = train 
train = train[, var.idx]
train = na.omit(train)

mm = model.matrix ( target_0 ~ .  , train )[,-1]
mm.df = as.data.frame(mm)
mm.df = cbind(mm.df,targetPos = train$target_0)
fn = paste(base.path,"train_class.csv",sep="")
write.csv(mm.df,quote=F,row.names=F,file=fn)

## test 
var.idx = grep(pattern = var.er , names (test) )
test = test[, var.idx]
test$id = test.tab$id 
test = na.omit(test)

mm = model.matrix ( ~ .  , test )[,-1]
mm.df.test = as.data.frame(mm)
fn = paste(base.path,"test_class.csv",sep="")
write.csv(mm.df.test,quote=F,row.names=F,file=fn)

## skewness  
l = transfrom4Skewness(train,test)
ttrain = l[[1]]
mm = model.matrix ( target_0 ~ .  , ttrain )[,-1]
mm.df = as.data.frame(mm)
mm.df = cbind(mm.df,targetPos = ttrain$target_0)
fn = paste(base.path,"train_nn_no_skewness.csv",sep="")
write.csv(mm.df,quote=F,row.names=F,file=fn)

ttest = l[[2]]
mm = model.matrix ( ~ .  , ttest )[,-1]
mm.df.test = as.data.frame(mm)
fn = paste(base.path,"test_nn_no_skewness.csv",sep="")
write.csv(mm.df.test,quote=F,row.names=F,file=fn)


# ### classifier basato sulle medie - 
# m1 = rep(NA,5)
# m0 = rep(NA,5)
# 
# mmin = rep(NA,5) 
# mmax = rep(NA,5) 
# mmed = rep(NA,5) 
# upper = rep(NA,5) 
# lower = rep(NA,5)
# 
# for (i in 1:length(colnames(mm.df))) {
#   var.idx = grep(pattern = "target" , colnames(mm.df)[i] )
#   if (length(var.idx) == 0) {
#     print(colnames(mm.df)[i])
#     m1[i] = mean(mm.df[mm.df$targetPos == 1,i])
#     m0[i] = mean(mm.df[mm.df$targetPos == 0,i])
#     sdi = sd(mm.df[,i])
#     
#     mmin[i] = min(m1[i],m0[i])
#     mmax[i] = max(m1[i],m0[i])
#     mmed[i] = (mmin[i]+mmax[i])/2
#     upper[i] = ifelse (m1[i]  < m0[i] , 0 , 1) 
#     lower[i] = ifelse (upper[i] == 1 , 0 ,  1)
#     
#     cat("m1 == ",m1[i]," m0 == ",m0[i] , "sd == ",sdi,"\n")
#   }
# }
# 
# 
# pred = rep(NA,5000)
# for (j in 1:5000 ) {
#   vots = NULL
#   for (i in 2:6) {
#     
#     
#       if(mm.df[j,i] < mmed[i]) {
#         vots = c(vots,lower[i])
#       } else {
#         vots = c(vots,upper[i])
#       }
#     
#   }
#   pred[j] = median(vots)
#   if (j %% 1000 == 0 ) print(j)
# }




  





 







