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
  write.csv(data,quote=FALSE,file=fn, row.names=FALSE)
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

ALL_ONES = c(1634)

## file types 
ERROR_PREFIX = "error"
SUBMISSION_PREFIX = "sub"

## digest types 
FEAT_SET = "features_red_" ### reduced data set

## clustering algorithms 
MAIN_CLUST_METH = "ward"
SEC_CLUST_METH = "kmeans"

######################### main loop 

sub = getSampleSubmission()
sub$prob = -1

#DRIVERS = c(1634)

ALL_DRIVERS_ORIG = getDrivers() 
if (exists("DRIVERS")) 
  ALL_DRIVERS_ORIG = intersect(ALL_DRIVERS_ORIG,DRIVERS)

cat("|--------------------------------->>> found ",length(ALL_DRIVERS_ORIG)," drivers in original dataset ... \n")

DIGESTED_DRIVERS = getDigestedDrivers( FEAT_SET )  
if (exists("DRIVERS")) 
  DIGESTED_DRIVERS = intersect(DIGESTED_DRIVERS,DRIVERS)

cat("|--------------------------------->>> found ",length(DIGESTED_DRIVERS)," drivers in digested datasets [",FEAT_SET,"]... \n")

for ( drv in DIGESTED_DRIVERS  ) { 
  
  cat("|---------------->>> processing driver:  [",FEAT_SET,"] <<",drv,">>  ..\n")
  
  if (exists("ALL_ONES") && is.element(el = drv , set = ALL_ONES)) {
    sub[sub$drv==drv,]$prob = 1
    next
  }
  
  data = getDigestedDriverData (FEAT_SET,drv)
  df <- scale(data[,-1]) 
  #df = preProcess(as.matrix(data[,-1]),method = c("center","scale"))
  
  ## clustering ..
  nc = NULL
  
  nc = tryCatch ({
    NbClust(df, distance = "euclidean", min.nc = 2, max.nc = 8, 
                 method = MAIN_CLUST_METH , index = "alllong")
  } , error = function(err) { 
    print(paste("ERROR:  ",err))
    logErrors(FEAT_SET, MAIN_CLUST_METH , SEC_CLUST_METH , drv )
      
    cat("|-------->>> trying with centroid method only ... \n")  
    NbClust(df, distance = "euclidean", min.nc = 2, max.nc = 8, 
                 method = SEC_CLUST_METH )
  })
  
  
  ## analyzing results ...
  part = as.numeric(nc$Best.partition)
  part.val = unique(part)
  cluster.perc = as.numeric(lapply(part.val,function(x) sum(part==x)/length(part)) )
  dominant.index = which(cluster.perc == max(cluster.perc))
  dominant.partition = part.val[dominant.index]
  
  cat("|-------->>> found ",length(part.val)," clusters ..\n") 
  for (c in part.val) {
    dominat.msg = ""
    if (c == dominant.index) dominat.msg = " --> DOMINANT PARTITION"
    cat("|----->>> cluster <<",as.character(cluster.perc[c]), ">> ", dominat.msg ,  " \n")
  }
  
  data$clust = part
  data$pred = ifelse(part == dominant.partition , 1 , 0)
  
  if (debug) {
#     pairs(df[,sample(1:(dim(df)[2]),size=(dim(df)[2]/2))], 
#           main = paste(FEAT_SET,MAIN_CLUST_METH,"_",SEC_CLUST_METH,sep="") 
#           , pch = 21, bg = c("red", "green3", "blue")[unclass(part)] )
    
    pairs(df[,sample(1:(dim(df)[2]),size=(dim(df)[2]/2))], 
          main = paste(FEAT_SET,MAIN_CLUST_METH,"_",SEC_CLUST_METH,sep="") 
          , pch = 21, bg = colors()[sample(1:(length(colors())),length(part.val))]  [unclass(part)] )
    
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
                            x = colnames(sub)))] , FEAT_SET , MAIN_CLUST_METH , SEC_CLUST_METH)

## some statistics ... 
cat("|----------------------->>> some statistics ..  \n")
cat("|----------------------->>> correct == " , ifelse ( sum(sub$prob == -1) > 0  , "NO" , "YES" ) , " \n" ) 
p.mean = sum(sub$prob)/length(sub$prob)
cat("|----------------------->>> [" ,paste(FEAT_SET,MAIN_CLUST_METH,"_",SEC_CLUST_METH,sep="") , "] AVERAGE PROB == ",
    p.mean," \n")







