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
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/otto-group-product-classification-challenge"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/otto-group-product-classification-challenge/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/otto-group-product-classification-challenge"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/otto-group-product-classification-challenge/"
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

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#######
verbose = T
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))

#######
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

train.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train.csv" , sep=''))) 

test.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                      "test.csv" , sep='')))

####### exploratory 
cat(">>> exploring test/train data ... \n")

expl.grid = as.data.frame(matrix(rep(NA,94*7),nrow = 94 , ncol = 7))
colnames(expl.grid) = c("var","min","avg","max","num.values","mode","contains.decinal")

## target 
expl.grid[1,]$var = "target"
expl.grid[1,]$num.values = length(unique(train.raw$target))
expl.grid[1,]$mode = Mode(train.raw$target)
expl.grid[1,]$contains.decinal = F

## input variables 
for (i in 1:93) {
  
  vals = c(train.raw[,(i+1)],test.raw[,(i+1)])
  
  expl.grid[i+1,]$var = paste0("feat_",i)
  expl.grid[i+1,]$min = min(vals)
  expl.grid[i+1,]$avg = mean(vals)
  expl.grid[i+1,]$max = max(vals)
  expl.grid[i+1,]$num.values = length(unique(vals))
  expl.grid[i+1,]$mode = Mode(vals)
  expl.grid[i+1,]$contains.decinal = (sum(vals%%1) > 0)
  
}

expl.grid$max_over_num_values = expl.grid$max/expl.grid$num.values

cat("**************************\n")
print(expl.grid)
cat("**************************\n")

############ encode y, train, test 

### y  - "Class_1" "Class_2" "Class_3" "Class_4" "Class_5" "Class_6" "Class_7" "Class_8" "Class_9" 
cat(">>> encoding y ... \n")
y = rep(NA,length(train.raw$target))
card = rep(-1,9)
for (i in 1:9) {
  y = ifelse(train.raw$target == paste0("Class_",i) , i , y )
  card[i] = sum(train.raw$target == paste0("Class_",i))
  cat (paste0("Class_",i) , ":",card[i]," \n")
}

## check
cat(">>> checking y ... \n")
if (sum(is.na(y)))
  stop("some problem encoding y:NAs")

for (i in 1:9) {
  nn = sum(y == i)
  cat (paste0("Class_",i) , ":",nn," \n")
  if (nn != card[i])
    stop("some problem encoding: different class cardinality")
}

### train , test 
cat(">>> encoding train / test set ... \n")

## remove id 
train = train.raw[ , -1] 
test = test.raw[ , -1]

## remove target on train set 
train = train[,-94]

## processing feat_1 - feat_93
for (i in 1:93) {
  name = paste0("feat_",i)
  
  cat(">> processing ",name," ... \n")
  
  ## encoding  
  l = encodeCategoricalFeature (train[,1] , test[,1] , colname.prefix = name , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -1]
  test = test[ , -1]
}

########### store on disk 
cat(">>> storing on disk ... \n")
write.csv(data.frame(y = y),
          quote=FALSE, 
          file=paste0(getBasePath("data"),"encoded_y.csv") ,
          row.names=FALSE)
write.csv(train,
          quote=FALSE, 
          file=paste0(getBasePath("data"),"encoded_train.csv") ,
          row.names=FALSE)

write.csv(test,
          quote=FALSE, 
          file=paste0(getBasePath("data"),"encoded_test.csv") ,
          row.names=FALSE)


########## store on disk for octave 
cat(">>> storing on disk for octave ... \n")
write.table(x=data.frame(y = y),
          quote=FALSE, 
          file=paste0(getBasePath("data"),"oct_y_encoded.csv") ,
          row.names=FALSE, 
          col.names=FALSE)

write.table(train,
          quote=FALSE, 
          file=paste0(getBasePath("data"),"oct_train_encoded.csv") ,
          row.names=FALSE,
          col.names=FALSE)

write.table(test,
            quote=FALSE, 
            file=paste0(getBasePath("data"),"oct_test_encoded.csv") ,
            row.names=FALSE,
            col.names=FALSE)
