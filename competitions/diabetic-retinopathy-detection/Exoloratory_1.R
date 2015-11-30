library(caret)
library(Hmisc)
library(data.table)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)
library(fBasics)
library(lattice)
require(xgboost)
require(methods)

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/diabetic-retinopathy-detection"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/diabetic-retinopathy-detection/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/diabetic-retinopathy-detection"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/diabetic-retinopathy-detection/"
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

#################
trainLabels = as.data.frame( fread(paste(getBasePath("data") , 
                                              "trainLabels.csv" , sep=''))) 

vessel_area_train = as.data.frame( fread(paste(getBasePath("data") , 
                           'vessel_area_train.csv' , sep=''))) 

###### 
vessel_perf = ddply(vessel_area_train , .(level) , function(x) c( 
  vessel_area.mean=mean(x$vessel_area) , 
  vessel_area.sd=sd(x$vessel_area) , 
  vessel_area_ratio.mean=mean(x$vessel_area_ratio) , 
  vessel_area_ratio.sd=sd(x$vessel_area_ratio) 
  ) )

print(vessel_perf)

###### 

get_id = function(img) {
  return(unlist(strsplit(x = img,split = '_'))[1])
}

ids = unlist(lapply(X = vessel_area_train$image, FUN =  get_id ))
vessel_area_train$patient = ids

levels = ddply(vessel_area_train , .(patient) , function(x) c(levels = length(unique(x$level))) )
perc = length(levels[levels$levels > 1 , ]$levels) / dim(levels)[1]
cat (">> percentage of patients with different diagnosis (left image has a different diagnosis of rigth image):",perc,"\n")




