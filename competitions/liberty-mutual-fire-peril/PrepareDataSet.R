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

getBestPredictors = function (predPvalues, data = train , th = 0.05) {
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

getBasePath = function () {
  ret = ""
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/liberty-mutual-fire-peril"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"
  
  if (file.exists(base.path1))  {
    ret = paste0(base.path1,"/")
  } else {
    ret = base.path2
  }
  ret
} 

prepare4Regression = function(train,test,pca = T,trasnf4skew = F , maxPredictors = 10) {
  
  ret = NULL 
  var.name = NULL
  var.index = NULL
  trainTransformed = NULL
  testTransformed = NULL
  
  fn.train = ""
  fn.test = "" 
  if (pca) { 
    if (! trasnf4skew) {
      fn.train = "train_reg_pca.csv"
      fn.test = "test_reg_pca.csv" 
    } else {
      fn.train = "train_reg_pca_noskew.csv"
      fn.test = "test_reg_pca_noskew.csv"
    }
  } else {
    if (! trasnf4skew) {
      fn.train = "train_reg.csv"
      fn.test = "test_reg.csv" 
    } else {
      fn.train = "train_reg_noskew.csv"
      fn.test = "test_reg_noskew.csv" 
    }
  }
  
  ## Analysis (p-values)
  predictors.reg.linear = getPvalueFeatures( features = train[ , - c(1,2,303)] , response = train$target )
  predictors.reg.linear = predictors.reg.linear[order(predictors.reg.linear$pValue,decreasing = F),]
  
  l = getBestPredictors(predPvalues = predictors.reg.linear , data = train , th = 0.05)
  var.name = l[[1]]
  var.index = l[[2]]
  
  if (pca) {
    
    ### pca 
    trainReg = train[,var.index]
    trainReg = na.omit(trainReg)
    del.idx = NULL
    for (i in 1:length(var.index)) {
      if (is.factor(trainReg[,i])) {
        cat("eliminating ",var.name[i]," (factor) ... \n")
        del.idx = c(del.idx,i)
      } 
    }
    trainReg = trainReg[,-del.idx] ## elimino var4 e dummy che sono factor 
    
    ### situazione plot
    trainReg.pca = prcomp(trainReg , center = T , scale. = T)  
    plot(trainReg.pca, type = "l")
    percVar = trainReg.pca$sd^2/sum(trainReg.pca$sd^2)
    maxPredictors = min(maxPredictors,length(percVar))
    sum(percVar[1:maxPredictors])  
    trainReg.pca = trainReg.pca$x[,1:maxPredictors]
    
    ## trans caret 
    trainRegTrans = NULL 
    if (! trasnf4skew) { 
      trainRegTrans = preProcess(trainReg, method = c("center","scale","pca") )
    } else {
      trainRegTrans = preProcess(trainReg, method = c("BoxCox", "center","scale","pca") )
    }
    trainReg.pca.caret = predict(trainRegTrans,trainReg)
    percVar = apply(trainReg.pca.caret,2,sd)^2/sum(apply(trainReg.pca.caret,2,sd)^2)
    maxPredictors = min(maxPredictors,length(percVar))
    sum(percVar[1:maxPredictors]) 
    
    #train (with NAs)
    trainReg = train[,var.index]
    trainRegFact = trainReg[,del.idx]
    trainReg = trainReg[,-del.idx]
    trainReg.pca = predict(trainRegTrans,trainReg)
    trainTransformed = cbind(target = train$target , trainReg.pca[,1:maxPredictors] , trainRegFact)
    
    ## test 
    var.index = var.index - 1
    testReg = test[,var.index]
    testRegFact = testReg[,del.idx]
    testReg = testReg[, -del.idx]
    testReg.pca = predict(trainRegTrans,testReg)
    testTransformed = cbind(testReg.pca[,1:maxPredictors] , testRegFact)
    
  } else {
    var.er = "target|var11|var4|var13|var10|dummy|weatherVar104|weatherVar31|weatherVar110|weatherVar98|weatherVar69|geodemVar31|var17|geodemVar37"
    
    ### just retain the variable higly correlated without NAs -- questi sono per la regression 
    #var.er = "target|var13|var11|var10|dummy|var17"
    #var.er = "target|var13|var11|var10|dummy|var17|var4|weatherVar104|weatherVar31|weatherVar110|weatherVar98|weatherVar69|weatherVar190|weatherVar194"
    
    ## train 
    var.idx = grep(pattern = var.er , names (train) )
    train.old = train 
    trainTransformed = train[, var.idx]  
    
    ## test 
    var.idx = grep(pattern = var.er , names (test) )
    testTransformed = test[, var.idx]
    
    if ( trasnf4skew ) {
      ## skewness  
      l = transfrom4Skewness(trainTransformed,testTransformed)
      trainTransformed = l[[1]]
      testTransformed = l[[2]]
      
    } 
  }
  
  ret = list(var.name,var.index,fn.train,fn.test,predictors.reg.linear,trainTransformed,testTransformed)  
}

prepare4Classification = function(train,test,pca = T,trasnf4skew = F) {
  
  ret = NULL 
  var.name = NULL
  var.index = NULL
  trainTransformed = NULL
  testTransformed = NULL
  
  fn.train = ""
  fn.test = "" 
  if (pca) { 
    if (! trasnf4skew) {
      fn.train = "train_class_pca.csv"
      fn.test = "test_class_pca.csv" 
    } else {
      fn.train = "train_class_pca_noskew.csv"
      fn.test = "test_class_pca_noskew.csv"
    }
  } else {
    if (! trasnf4skew) {
      fn.train = "train_class.csv"
      fn.test = "test_class.csv" 
    } else {
      fn.train = "train_class_noskew.csv"
      fn.test = "test_class_noskew.csv" 
    }
  }
  
  ## Analysis (p-values)
  predictors.class.linear = getPvalueFeatures( features = train[ , - c(1,2,303)] , response = train$target_0 )
  predictors.class.linear = predictors.class.linear[order(predictors.class.linear$pValue,decreasing = F),]
  
  l = getBestPredictors(predPvalues = predictors.class.linear , data = train , th = 0.05)
  var.name = l[[1]]
  var.index = l[[2]]
  
  if (pca) {
    
    ### pca 
     trainClass = train[,var.index]
     trainClass = na.omit(trainClass)
     trainClass = trainClass[, -c(2,5)] ## elimino var4 e dummy che sono factor 
#     trainClass.pca = prcomp(trainClass , center = T , scale. = T)  
#     plot(trainClass.pca, type = "l")
#     percVar = trainClass.pca$sd^2/sum(trainClass.pca$sd^2)
#     sum(percVar[1:11]) ## catturo 80% della variance 
#     trainClass.pca = trainClass.pca$x[,1:11]
    
    trainClassTrans = NULL 
    if (! trasnf4skew) { 
      trainClassTrans = preProcess(trainClass, method = c("center","scale","pca") )
    } else {
      trainClassTrans = preProcess(trainClass, method = c("BoxCox", "center","scale","pca") )
    }
    trainClass.pca.caret = predict(trainClassTrans,trainClass)
    percVar = apply(trainClass.pca.caret,2,sd)^2/sum(apply(trainClass.pca.caret,2,sd)^2)
    sum(percVar[1:10]) ## catturo 83% della variance
    
    #train
    trainClass = train[,var.index]
    trainClass = trainClass[, -c(2,5)]
    trainClass.pca = predict(trainClassTrans,trainClass)
    trainClass.pca = cbind(trainClass.pca[,1:11] , var4 = train$var4 , dummy = train$dummy)
    trainTransformed = cbind(target = train$target , trainClass.pca , targetPos = train$target_0)
    
    ## test 
    var.index = var.index - 1
    testClass = test[,var.index]
    testClass = testClass[, -c(2,5)]
    testClass.pca = predict(trainClassTrans,testClass)
    testClass.pca = cbind(testClass.pca[,1:11] , var4 = test$var4 , dummy = test$dummy)
    testTransformed = testClass.pca
    
  } else {
    var.er = "target|var11|var4|var13|var10|dummy|weatherVar104|weatherVar31|weatherVar110|weatherVar98|weatherVar69|geodemVar31|var17|geodemVar37"
    
    ### just retain the variable higly correlated without NAs -- questi sono per la regression 
    #var.er = "target|var13|var11|var10|dummy|var17"
    #var.er = "target|var13|var11|var10|dummy|var17|var4|weatherVar104|weatherVar31|weatherVar110|weatherVar98|weatherVar69|weatherVar190|weatherVar194"
    
    ## train 
    var.idx = grep(pattern = var.er , names (train) )
    train.old = train 
    trainTransformed = train[, var.idx]  
    
    ## test 
    var.idx = grep(pattern = var.er , names (test) )
    testTransformed = test[, var.idx]
    
    if ( trasnf4skew ) {
      ## skewness  
      l = transfrom4Skewness(trainTransformed,testTransformed)
      trainTransformed = l[[1]]
      testTransformed = l[[2]]
      
    } 
  }

  ret = list(var.name,var.index,fn.train,fn.test,predictors.class.linear,trainTransformed,testTransformed)  
}

##############  Loading data sets (train, test, sample) ... 
base.path = getBasePath()

train.fn = "train.csv"
test.fn = "test.csv"
sampleSub.fn = "sampleSubmission.csv"

train.tab = fread(paste(base.path,train.fn,sep="") , header = TRUE , sep=","  )
test.tab = fread(paste(base.path,test.fn,sep="") , header = TRUE , sep=","  )

## data formatting 
train = getData4Analysis(train.tab)
test = getData4Analysis(test.tab)

train.bkp = train 
test.bkp = test 

dim(train)
dim(test)

### preparing data for classification 
l = prepare4Classification (train,test,pca = T)
var.name = l[[1]]
var.index = l[[2]]
fn.train = l[[3]]
fn.test = l[[4]]
predictors.class.linear = l[[5]]
trainTranformed = l[[6]]
testTranformed = l[[7]] 

## write train 
write.csv(as.data.frame(model.matrix ( ~ . , na.omit(trainTranformed) )[,-1]),quote=F,row.names=F,file=paste0(getBasePath(),fn.train))

## write test 
write.csv(as.data.frame(model.matrix ( ~ . , na.omit(testTranformed) )[,-1]),quote=F,row.names=F,file=paste0(getBasePath(),fn.test))


### preparing data for regression 
l = prepare4Regression (train,test,pca = T)
var.name = l[[1]]
var.index = l[[2]]
fn.train = l[[3]]
fn.test = l[[4]]
predictors.reg.linear = l[[5]]
trainTranformed = l[[6]]
testTranformed = l[[7]] 

## write train 
write.csv(as.data.frame(model.matrix ( ~ . , na.omit(trainTranformed) )[,-1]),quote=F,row.names=F,file=paste0(getBasePath(),fn.train))

## write test 
write.csv(as.data.frame(model.matrix ( ~ . , na.omit(testTranformed) )[,-1]),quote=F,row.names=F,file=paste0(getBasePath(),fn.test))





 







