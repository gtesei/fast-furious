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

######################### settings ans constants 
debug = F
DIGEST_LABEL = "features_red_" ### reduced data set

######################### main loop 

DRIVERS = c(12)

ALL_DRIVERS_ORIG = getDrivers() 
if (exists("DRIVERS")) 
  ALL_DRIVERS_ORIG = intersect(ALL_DRIVERS_ORIG,DRIVERS)

cat("|--------------------------------->>> found ",length(ALL_DRIVERS_ORIG)," drivers in original dataset ... \n")

DIGESTED_DRIVERS = getDigestedDrivers( DIGEST_LABEL )  
if (exists("DRIVERS")) 
  DIGESTED_DRIVERS = intersect(DIGESTED_DRIVERS,DRIVERS)

cat("|--------------------------------->>> found ",length(DIGESTED_DRIVERS)," drivers in digested datasets [",DIGEST_LABEL,"]... \n")

for ( drv in DIGESTED_DRIVERS  ) { 
  
  cat("|---------------->>> processing driver:  [",DIGEST_LABEL,"] <<",drv,">>  ..\n")
  
  data = getDigestedDriverData (DIGEST_LABEL,drv)
  df <- scale(data[,-1]) 
  #df = preProcess(as.matrix(data[,-1]),method = c("center","scale"))
  
  ## clustering ...
  ERR = F 
  nc = NULL
  
  nc = tryCatch ({
    NbClust(df, distance = "euclidean", min.nc = 2, max.nc = 8, 
                 method = "complete", index = "alllong")
  } , error = function(err) { 
    print(paste("ERROR:  ",err))
    cat("|-------->>> trying with centroid method only ... \n")  
    NbClust(df, distance = "euclidean", min.nc = 2, max.nc = 8, 
                 method = "centroid")
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
  
  data$ERR = ERR 
  data$clust = part
  data$pred = ifelse(part == dominant.partition , 1 , 0)
  
}




