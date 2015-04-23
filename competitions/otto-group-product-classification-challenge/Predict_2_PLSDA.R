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
source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))
source(paste0( getBasePath("process") , "/Resample_Lib.R"))
source(paste0( getBasePath("process") , "/Classification_Lib.R"))

sub = NULL
grid = NULL
controlObject <- trainControl(method = "boot", number = 30 , 
                              summaryFunction = twoClassSummary , classProbs = TRUE)

#######
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

train.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                       "encoded_train_reduced.csv" , sep=''))) 

test.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                      "encoded_test_reduced.csv" , sep='')))

y = as.data.frame( fread(paste(getBasePath("data") , 
                                      "encoded_y.csv" , sep='')))

y = as.numeric(y$y)
y.class = unique(y)

########
cat("\n*********\n")
for (j in y.class) {
  cat("y == ",j,":" , sum(y == j)," -  ",sum(y == j)/length(y) , " \n")
} 
cat("*********\n")
########

Class_j = matrix(rep(NA,nrow(sampleSubmission)*9),nrow=nrow(sampleSubmission),ncol = 9)

for ( j in 1:length(y.class)) {
  cat(">>> building train set for Class <<",as.character(j),">> ... \n")
  
  train.1 = train.raw[(y == j), ]
  train.0 = train.raw[sample(x = which(y != j) , size = dim(train.1)[1] ) , ]
  train.j = rbind(train.0,train.1)
  
  y.j = c( rep(0,dim(train.1)[1]) , rep(1,dim(train.1)[1]) ) 
  y.cat = factor(y.j) 
  levels(y.cat) = c("other_classes","this_class")
  
  ### shuffle 
  idx = sample(x = 1:length(y.j) , size = length(y.j))
  train.j = train.j[idx,]
  y.cat = y.cat[idx]
  
  cat(">>> training and predicting with PLSDA on Class <<",as.character(j),">> ... \n")
  
  l = tryCatch({ 
    class.trainAndPredict ( y.cat , 
                            train.j , 
                            test.raw , 
                            fact.sign = 'this_class' , 
                            "PLSDA" , 
                            controlObject, 
                            best.tuning = F, 
                            verbose = verbose)
    
  } , error = function(err) { 
    print(paste("ERROR:  ",err))
    NULL
  })
  
  
  pred.prob.train = l[[1]]
  pred.train = l[[2]] 
  pred.prob.test = l[[3]]
  pred.test = l[[4]]
  
  
  Class_j[,j] = pred.prob.test
  
}

sub = data.frame ( id = sampleSubmission$id , 
                   Class_1 = Class_j[ , 1] , 
                   Class_2 = Class_j[ , 2] , 
                   Class_3 = Class_j[ , 3] , 
                   Class_4 = Class_j[ , 4] , 
                   Class_5 = Class_j[ , 5] , 
                   Class_6 = Class_j[ , 6] , 
                   Class_7 = Class_j[ , 7] , 
                   Class_8 = Class_j[ , 8] , 
                   Class_9 = Class_j[ , 9] 
                   )

## check
cat(">>> checking sub ... \n")
if (sum(is.na(sub)))
  stop("some problem with sub:NAs")


# #### adjiusting --- pessima idea: ha fatto 10 !!!
sub.max.idx = apply(sub[,-1],1,function(x) which.max(x))

Class_j_mod = matrix(rep(0,nrow(sub)*9),nrow=nrow(sub),ncol = 9)

for (j in 1:length(sub$id)) 
  for (i in 1:9) {
    rec = sum(sub[j, 2:10]) - sub[j,(sub.max.idx[j]+1)]
    Class_j_mod[j,i] = ifelse(i == sub.max.idx[j] , sub[j,(i+1)],   (1 - sub[j,(sub.max.idx[j]+1)]) * sub[j,(i+1)] / rec )
  }

sub = data.frame ( id = sub$id , 
                   Class_1 = Class_j_mod[ , 1] , 
                   Class_2 = Class_j_mod[ , 2] , 
                   Class_3 = Class_j_mod[ , 3] , 
                   Class_4 = Class_j_mod[ , 4] , 
                   Class_5 = Class_j_mod[ , 5] , 
                   Class_6 = Class_j_mod[ , 6] , 
                   Class_7 = Class_j_mod[ , 7] , 
                   Class_8 = Class_j_mod[ , 8] , 
                   Class_9 = Class_j_mod[ , 9] 
)

### storing on disk 
write.csv(sub,quote=FALSE, 
          file=paste(getBasePath("data"),"mySub_PLSDA_2.csv",sep='') ,
          row.names=FALSE)

cat("<<<<< submission correctly stored on disk >>>>>\n") 











