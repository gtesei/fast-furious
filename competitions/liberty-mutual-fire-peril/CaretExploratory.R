library(quantreg)
library(data.table)
library(glmnet)
library(class)
library(caret)

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

pScore <- function(x, y)
{
  numX <- length(unique(x))
  if(numX > 2)
  {
    ## With many values in x, compute a t-test
    out <- t.test(x ~ y)$p.value
  } else {
    ## For binary predictors, test the odds ratio == 1 via
    ## Fisher's Exact Test
    out <- fisher.test(factor(x), y)$p.value
  }
  out
}

pCorrection <- function (score, x, y)
{
  ## The options x and y are required by the caret package
  ## but are not used here
  score <- p.adjust(score, "bonferroni")
  ## Return a logical vector to decide which predictors
  ## to retain after the filter
  keepers <- (score <= 0.05)
  keepers
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

## Analysis (p-values)
predictors.reg.linear = getPvalueFeatures( features = train[ , - c(1,2,303)] , response = train$target )
predictors.reg.linear = predictors.reg.linear[order(predictors.reg.linear$pValue,decreasing = F),]

l = getBestPredictors(predPvalues = predictors.reg.linear , data = train , th = 0.05)
var.name = l[[1]]
var.index = l[[2]]

train = train[, c(2,var.index)]
train = na.omit(train)

CategoricalPredictors = train[,c(4,5)]
train = train[,-c(4,5)]## l'analisi di correlazioni non la faccio sulle variabili categoriche 
PredToDel = findCorrelation(cor( train )) 
train = train[,-PredtoDel]
train = cbind(train,CategoricalPredictors)

IndexTrain = grep(paste(colnames(train),collapse="|") , colnames(train.bkp) ) 
IndexTrain = IndexTrain[-length(IndexTrain)] # elimino target_0
IndexTest = grep(paste(colnames(train),collapse="|") , colnames(test.bkp) ) 

library(Hmisc)
describe(x = train)

## plot 
fn = paste0(base.path,"target_by_predictors.jpeg")
jpeg(filename=fn)
par(mfrow=c(12,2) , oma=c(1,1,0,0), mar=c(1,1,1,0), tcl=-0.1, mgp=c(0,0,0))
#par(mfrow=c(1,1))
for (i in 2:(dim(train)[2]) ) {
  cname = colnames(train)[i]
  label = paste0("target ~ ",cname)
  plot( y = train$target, x = train[,i], pch = 19, col = "blue", cex = 0.5, 
        main = label ,  xlab = cname, ylab = "target")
  if (is.numeric(train[,i])) hist(train[,i] , xlab = cname )
}
dev.off()



#par(mfrow=c(1,1))

cname = "predictors"
fn = paste0(base.path,paste0("box_target_by_",cname))
png(filename=fn)
par(mfrow=c(9,2) , oma=c(1,1,0,0), mar=c(1,1,1,0), tcl=-0.1, mgp=c(0,0,0))
for (i in 2:(dim(train)[2]) ) {
  cname = colnames(train)[i]
  label = paste0("target ~ ",cname)
  #fn = paste0(base.path,paste0("box_target_by_",cname))
  #png(filename=fn)
  boxplot(train$target ~ train[,i], 
          col = terrain.colors(length(train[,i]) , 
          alpha = 0.8), varwidth = TRUE,
          main = label ,  xlab = cname, ylab = "target")
  
}
dev.off()



library(lattice)
xyplot(train$target ~ train$var13 | train$var4, panel = function(x, y, ...) {
  panel.xyplot(x, y, ...)
  lm1 <- lm(y ~ x)
  lm1sum <- summary(lm1)
  r2 <- lm1sum$adj.r.squared
  p <- lm1sum$coefficients[2, 4]
  panel.abline(lm1)
  panel.text(labels = bquote(italic(R)^2 == .(format(r2, digits = 3))), x = 780, y = 0.15)
  panel.text(labels = bquote(italic(p) == .(format(p, digits = 3))), x = 770,  y = 0.2)
}, data = train, as.table = TRUE, xlab = "var13", ylab = "target",  main = "target ~ var13 | var4")


for (i in 2:(dim(train)[2]) ) {
  cname = colnames(train)[i]
  label = paste0(paste0("target ~ ",cname)," | var4")
  fn = paste0(paste0(base.path,paste0("lat_target_by_",cname)),"_given_var4")
  cat(fn,"...\n")
  
  dev.hold()
  jpeg(filename=fn,quality=100)
  xyplot(train$target ~ train[,i] | train$var4, panel = function(x, y, ...) {
  panel.xyplot(x, y, ...)
  lm1 <- lm(y ~ x)
  lm1sum <- summary(lm1)
  r2 <- lm1sum$adj.r.squared
  p <- lm1sum$coefficients[2, 4]
  panel.abline(lm1)
  panel.text(labels = bquote(italic(R)^2 == .(format(r2, digits = 3))), x = 780, y = 0.15)
  panel.text(labels = bquote(italic(p) == .(format(p, digits = 3))), x = 770,  y = 0.2)
}, data = train, as.table = TRUE, xlab = cname, ylab = "target",  main = label )
  dev.off()
  
}


train$weatherVar118.cut <- equal.count(train$weatherVar118, 2)
train$var11.cut <- equal.count(train$var11, 2)
fn = paste0(paste0(base.path,paste0("lat_target_by_","var11_var118.cut")),"_given_var4.jpg")
jpeg(filename=fn,quality=100)
xyplot(train$target ~ train$var4 | train$var11.cut, panel = function(x, y, ...) {
  panel.xyplot(x, y, ...)
  lm1 <- lm(y ~ x)
  lm1sum <- summary(lm1)
  r2 <- lm1sum$adj.r.squared
  p <- lm1sum$coefficients[2, 4]
  panel.abline(lm1)
  panel.text(labels = bquote(italic(R)^2 == .(format(r2, digits = 3))), x = 780, y = 0.15)
  panel.text(labels = bquote(italic(p) == .(format(p, digits = 3))), x = 770,  y = 0.2)
}, data = train, as.table = TRUE, xlab = "train$var11 | train$weatherVar118.cut", 
ylab = "target",  
main = "train$target ~ train$var11 | train$weatherVar118.cut" )
dev.off()


train$target_0 = ifelse(train$target > 0 , 1 , 0)
aa = prop.table(table(train$var4,train$target_0),1) 
var4.freq = data.frame(var4 = aa[,0] , lab_0 = aa[,2] , lab_1 = aa[,1])
var4.freq = var4.freq[order(var4.freq$lab_1),]


