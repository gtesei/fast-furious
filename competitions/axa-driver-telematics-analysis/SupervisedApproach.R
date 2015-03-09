library(caret)
library(Hmisc)
library(data.table)
library(verification)
library(pROC)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)
library(fBasics)
library(NbClust)

getSampleSubmission = function () {
  ret = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/axa-driver-telematics-analysis"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/"
  
  if (file.exists(base.path1))  {
    ret = base.path1
    ret = paste(ret,"/",sep="")
  } else if (file.exists(base.path2)) {
    ret = base.path2
  } else {
    ret = NA
  }
  
  path = paste( ret, "sampleSubmission.csv" ,  sep = "")
  sampleSubmission = as.data.frame(fread( path ))
  
  drv = as.numeric (lapply(sampleSubmission$driver_trip, function(x) as.numeric (strsplit(x,"_")[[1]][1])  ))
  trp = as.numeric (lapply(sampleSubmission$driver_trip, function(x) as.numeric (strsplit(x,"_")[[1]][2])  ))
  
  sampleSubmission$drv = drv 
  sampleSubmission$trip = trp
  
  sampleSubmission
}

storeSubmission = function (data , feat.label , main.clust.alg , sec.clust.alg) {
  ret = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/submission_supervised"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/submission_supervised/"
  
  if (file.exists(base.path1))  {
    ret = base.path1
    ret = paste(ret,"/",sep="")
  } else if (file.exists(base.path2)) {
    ret = base.path2
  } else {
    ret = NA
  }
  
  fn = paste(ret, feat.label, "_" , main.clust.alg , "_" , sec.clust.alg  , "_submission.csv", sep="")
  write.csv(data,quote=FALSE,file=fn, row.names=FALSE)
} 

recoverSubmission = function (feat.label , main.clust.alg , sec.clust.alg) {
  ret = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/submission"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/submission/"
  
  if (file.exists(base.path1))  {
    ret = base.path1
    ret = paste(ret,"/",sep="")
  } else if (file.exists(base.path2)) {
    ret = base.path2
  } else {
    ret = NA
  }
  
  fn = paste(ret, feat.label, "_" , main.clust.alg , "_" , sec.clust.alg  , "_submission.csv", sep="")
  sampleSubmission = as.data.frame(fread( fn ))
  
  drv = as.numeric (lapply(sampleSubmission$driver_trip, function(x) as.numeric (strsplit(x,"_")[[1]][1])  ))
  trp = as.numeric (lapply(sampleSubmission$driver_trip, function(x) as.numeric (strsplit(x,"_")[[1]][2])  ))
  
  sampleSubmission$drv = drv 
  sampleSubmission$trip = trp
  
  sampleSubmission
}

getTrips = function (drv = 0) {
  ret = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/drivers/"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/drivers/"
  
  base.path1 = paste( base.path1, as.character(drv) , sep = "") 
  base.path2 = paste( base.path2, as.character(drv) , "/" , sep = "") 
  
  if (file.exists(base.path1))  {
    
    ret = as.numeric(lapply(as.character(list.files(base.path1)), function(x) as.numeric ( substr(x, 1, nchar(x) - 4)) ) )
    ret = sort( ret , decreasing = F)
    
  } else if (file.exists(base.path2)) {
    
    ret = as.numeric(lapply(as.character(list.files(base.path2)), function(x) as.numeric(substr(x, 1, nchar(x) - 4)) ))
    ret = sort( ret , decreasing = F)
    
  } else {
    ret = NA
  }
  
  ret
}


getDrivers = function () {
  ret = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/drivers"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/drivers/"
  
  if (file.exists(base.path1))  {
    
    ret = as.numeric (lapply(list.files(base.path1), function(x) as.numeric (x))) 
    ret = sort (ret , decreasing = F) 
    
  } else if (file.exists(base.path2)) {
    
    ret = as.numeric (lapply(list.files(base.path2), function(x) as.numeric (x)))
    ret = sort (ret , decreasing = F) 
    
  } else {
    ret = NA
  }
  
  ret
}

getDigestedDrivers = function (label) {
  ret = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/digest"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/digest/"
  
  if (file.exists(base.path1))  {
    ret = base.path1
    ret = paste(ret,"/",sep="")
  } else if (file.exists(base.path2)) {
    ret = base.path2
  } else {
    ret = NA
  }
  
  list = as.numeric(lapply(as.character(list.files(ret,pattern = paste(label,"*",sep=""))), 
                           function(x) {
                             a = substr(x, 1, nchar(x) - 4)
                             b = substr(a, nchar(label)+1,nchar(a))
                             as.numeric (b)
                           } 
                           ) 
                    )
  
  sort( list , decreasing = F)
} 

getDigestedDriverData = function (label,drv) {
  ret = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/digest"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/digest/"
  
  if (file.exists(base.path1))  {
    ret = base.path1
    ret = paste(ret,"/",sep="")
  } else if (file.exists(base.path2)) {
    ret = base.path2
  } else {
    ret = NA
  }
  
  path = paste( ret, label , as.character(drv) , ".csv" ,  sep = "")
  as.data.frame(fread( path ))
}

logErrors = function (  feat.label ,  
                        main.clust.alg , sec.clust.alg , 
                        drv ) { 
  ret = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/submission"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/submission/"
  
  if (file.exists(base.path1))  {
    ret = base.path1
    ret = paste(ret,"/",sep="")
  } else if (file.exists(base.path2)) {
    ret = base.path2
  } else {
    ret = NA
  }
  
  fn = paste(ret, feat.label, "_" , main.clust.alg , "_" , sec.clust.alg  , "_errors.csv", sep="")
  data = NULL 
  if ( file.exists(fn) ) {
    data = as.data.frame(fread( fn )) 
  } else {
    data = data.frame(ERR = c(drv))
  }
  
  vect = as.numeric(data$ERR)
  vect = c(vect,drv)
  vect = sort(unique(vect))
  
  data = data.frame(ERR = vect)
  
  write.csv(data,quote=FALSE,file=fn, row.names=FALSE)
} 

######################### settings ans constants 
debug = F

#RECOVER_FROM = 1634 ## <<<<<<<------------- attenzione session recovering in corso ... 
ALL_ONES = c(1634)

## do only these drivers (for testing)
#DRIVERS = c(1634)

## digest types 
FEAT_SET = "features_red_" ### reduced data set

## models
sup.model = NULL 

SUP_SVM = "svm"
SUP_LOG  = "logistic_regression"
SUP_LDA = "lda"
SUP_PLSDA = "plsda"
SUP_NSC = "nsc"
SUP_NN = "neural_networks"
SUP_KNN = "knn"

######################### main loop 

sup.model = SUP_KNN

sub = getSampleSubmission()
sub$prob = -1
if (exists("RECOVER_FROM") && RECOVER_FROM > -1) {
  cat("|--------------------------------->>> recovering session [data set:",FEAT_SET,"][main clust alg:",
      MAIN_CLUST_METH,"][secondary clust alg:",SEC_CLUST_METH,"] from driver ",RECOVER_FROM," ... \n")
  sub = recoverSubmission  (FEAT_SET , MAIN_CLUST_METH , SEC_CLUST_METH)
  print(head(sub[sub$drv == (unique(sub$drv)[which(unique((sub$drv)) == RECOVER_FROM) -1]),]))
  print(head(sub[sub$drv == (RECOVER_FROM),]))
}

ALL_DRIVERS_ORIG = getDrivers() 
if (exists("DRIVERS")) 
  ALL_DRIVERS_ORIG = intersect(ALL_DRIVERS_ORIG,DRIVERS)

cat("|--------------------------------->>> found ",length(ALL_DRIVERS_ORIG)," drivers in original dataset ... \n")

DIGESTED_DRIVERS = getDigestedDrivers( FEAT_SET )  
if (exists("DRIVERS")) 
  DIGESTED_DRIVERS = intersect(DIGESTED_DRIVERS,DRIVERS)

cat("|--------------------------------->>> found ",length(DIGESTED_DRIVERS)," drivers in digested datasets [",FEAT_SET,"]... \n")

controlObject <- trainControl(method = "boot", number = 30 , 
                              summaryFunction = twoClassSummary , classProbs = TRUE)

error.num = 0
 
for ( drv in DIGESTED_DRIVERS  ) { 
  
  cat("|---------------->>> processing driver:  [",FEAT_SET,"] <<",drv,">>  ..\n")
  
  ## skip all ones after selecting all ones ...
  if (exists("ALL_ONES") && is.element(el = drv , set = ALL_ONES)) {
    sub[sub$drv==drv,]$prob = 1
    next
  }
  
  ## if you are recovering a session skip work already performed ...
  if (exists("RECOVER_FROM") && drv < RECOVER_FROM ) {
    next
  }
  
  data = getDigestedDriverData (FEAT_SET,drv)
  df <- scale(data[,-1]) 
  df.initial = df 
  #df = preProcess(as.matrix(data[,-1]),method = c("center","scale"))
  
  ## predicting  
  ##data$pred = 1 ## dummy 
  label = rep(1,dim(df)[1])
  drvs = sample(DIGESTED_DRIVERS[-which(drv == DIGESTED_DRIVERS)],100) 
  old.size = dim(df)[1]
  new.size = old.size
  for (ds in drvs ) {
    
    if ( new.size >= (5*old.size) ) break
      
    datas = getDigestedDriverData (FEAT_SET,ds)
    dfs <- scale(datas[,-1]) 
    df.tmp = tryCatch({
      rbind(df,dfs)
    } , error = function(err) { 
      print(paste("ERROR:  ",err))
      NULL
    })
    if(! is.null(df.tmp)) {
      new.size = dim(df.tmp)[1]
      cat ("old df was ",as.character(old.size) , " rows ... new is ",as.character(new.size) ,"\n")
      df = df.tmp
    }
  }
  
  if (new.size < (5*old.size) ) {
    cat("impossible building train set ... setting all ones ... \n")
    data$pred = 1 ## dummy 
    error.num = error.num + 1 
  } else {
    cat("training ... \n")
    label = factor(c(label,rep(0,(new.size-old.size))))
    levels(label) = c("isOther","isTheDriver")
    
    model = NULL
    if (sup.model == SUP_SVM) { ## svm 
      sigmaRangeReduced <- sigest(as.matrix(df))
      svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1], .C = 2^(seq(-4, 4)))
      model <- train( x = df , y = label,  
                      method = "svmRadial", tuneGrid = svmRGridReduced, 
                      metric = "ROC", fit = FALSE, trControl = controlObject)
      
      #     model <- train( x = df , y = label,  
      #                     method = "svmRadial", tuneLength = 15 , 
      #                     metric = "ROC", fit = FALSE, trControl = controlObject)
      
    } else if (sup.model == SUP_LOG) { ## logistic_regression
      model <- train( x = df , y = label , 
                      method = "glm", metric = "ROC", trControl = controlObject)
    } else if (sup.model == SUP_LDA) { ## lda
      model <- train( x = df , y = label,  
                      method = "lda", metric = "ROC" , trControl = controlObject)
    } else if (sup.model == SUP_PLSDA) { ## plsda
      model <- train( x = df , y = label,  
                      method = "pls", tuneGrid = expand.grid(.ncomp = 1:10), 
                      metric = "ROC" , trControl = controlObject)
    } else if (sup.model == SUP_NSC) { ## nsc
      nscGrid <- data.frame(.threshold = 0:25)
      model <- train( x = df , y = label,  
                      method = "pam", tuneGrid = nscGrid, 
                      metric = "ROC", trControl = controlObject)
    } else if (sup.model == SUP_NN) { ## nn
      nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
      maxSize <- max(nnetGrid$.size)
      numWts <- 1*(maxSize * ( (dim(df)[2]) + 1) + maxSize + 1)
      model <- train( x = df , y = label,  
                      method = "nnet", metric = "ROC", 
                      preProc = c( "spatialSign") , 
                      tuneGrid = nnetGrid , trace = FALSE , maxit = 2000 , 
                      MaxNWts = numWts, trControl = controlObject)
    } else if (sup.model == SUP_KNN) { ## knn
      model <- train( x = df , y = label,  
                      method = "knn", 
                      tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)),
                      metric = "ROC",  trControl = controlObject)
    }
    
    cat("predicting ... \n")
    data$pred = predict(model , df.initial , type = "prob") [,'isTheDriver'] 
  }
  

  ### update submission 
  cat("|----->>> updating submission ..  \n")
  
  dd = data[,grep(pattern = "trip|pred"  , x = colnames(data))]
  dd$drv = drv 
  
  m1 = merge(sub,dd,by=c("drv","trip") , all = T )
  m1$prob = ifelse(! is.na(m1$pred),m1$pred,m1$prob)
  m1 = m1[,-(grep(pattern = "pred"  , x = colnames(m1)))]

  sub = m1 
}

## store submission 
cat("|----->>> storing submission ..  \n")
storeSubmission (sub[,(grep(pattern = "driver_trip|prob"  , 
                            x = colnames(sub)))] , FEAT_SET , "__supervised_approach__" , sup.model)

## some statistics ... 
cat("|----------------------->>> some statistics ..  \n")
cat("|----------------------->>> correct == " , ifelse ( sum(sub$prob == -1) > 0  , "NO" , "YES" ) , " \n" ) 
p.mean = sum(sub$prob)/length(sub$prob)
cat("|----------------------->>> [" ,paste(sup.model,sep="") , "] AVERAGE 1s == ",
    p.mean," \n")
cat("|----------------------->>>  number of errors  == ", as.character(error.num), " \n")

