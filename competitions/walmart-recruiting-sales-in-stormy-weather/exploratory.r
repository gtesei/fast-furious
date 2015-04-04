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


getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/walmart-recruiting-sales-in-stormy-weather"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/walmart-recruiting-sales-in-stormy-weather/"
  } else if (type == "preprocess") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/data_pre_process"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/data_pre_process/"
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

getTrain = function () {
  path = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/train.csv"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/train.csv"
  
  if (file.exists(base.path1))  {
    path = base.path1
  } else if (file.exists(base.path2)) {
    path = base.path2
  } else {
    stop('impossible load train.csv')
  }
  
  cat("loading train data ... ")
  trdata = as.data.frame(fread(path))
  cat("converting date ...")
  train$date = as.Date(train$date,"%Y-%m-%d")
  trdata
} 

getWeather = function () {
  path = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/weather.csv"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/weather.csv"
  
  if (file.exists(base.path1))  {
    path = base.path1
  } else if (file.exists(base.path2)) {
    path = base.path2
  } else {
    stop('impossible load train.csv')
  }
  
  cat("loading weather data ...")
  weather = as.data.frame(fread(path))
  
  ####
  weather[weather$tmax == "M",]$tmax = NA
  weather$tmax = as.numeric(weather$tmax)
  
  weather[weather$tmin == "M",]$tmin = NA
  weather$tmin = as.numeric(weather$tmin)
  
  weather[weather$tavg == "M",]$tavg = NA
  weather$tavg = as.numeric(weather$tavg)
  
  weather[weather$depart == "M",]$depart = NA
  weather$depart = as.numeric(weather$depart)
  
  weather[weather$dewpoint == "M",]$dewpoint = NA
  weather$dewpoint = as.numeric(weather$dewpoint)
  
  weather[weather$wetbulb == "M",]$wetbulb = NA
  weather$wetbulb = as.numeric(weather$wetbulb)
  
  weather[weather$heat == "M",]$heat = NA
  weather$heat = as.numeric(weather$heat)
  
  weather[weather$cool == "M",]$cool = NA
  weather$cool = as.numeric(weather$cool)
  
  weather[weather$sunrise == "-",]$sunrise = NA
  weather$sunrise = as.numeric(weather$sunrise)
  
  weather[weather$sunset == "-",]$sunset = NA
  weather$sunset = as.numeric(weather$sunset)
  
  weather$snowfall = gdata::trim(weather$snowfall)         
  weather[weather$snowfall == "T",]$snowfall = "0.01"
  weather[weather$snowfall == "M",]$snowfall = NA
  weather$snowfall = as.numeric(weather$snowfall)
  
  weather$preciptotal = gdata::trim(weather$preciptotal)    
  weather[weather$preciptotal == "T",]$preciptotal = "0.01"
  weather[weather$preciptotal == "M",]$preciptotal = NA
  weather$preciptotal = as.numeric(weather$preciptotal)
  
  weather[weather$stnpressure == "M",]$stnpressure = NA
  weather$stnpressure = as.numeric(weather$stnpressure)
  
  weather[weather$sealevel == "M",]$sealevel = NA
  weather$sealevel = as.numeric(weather$sealevel)
  
  weather[weather$resultspeed == "M",]$resultspeed = NA
  weather$resultspeed = as.numeric(weather$resultspeed)
  
  weather[weather$resultdir == "M",]$resultdir = NA
  weather$resultdir = as.numeric(weather$resultdir)
  
  weather[weather$avgspeed == "M",]$avgspeed = NA
  weather$avgspeed = as.numeric(weather$avgspeed)
  
  ###
  weather$TS = 0
  weather[grep(x = weather$codesum , pattern = "TS"), ]$TS = 1
  
  weather$GR = 0
  weather[grep(x = weather$codesum , pattern = "GR" , fixed = T), ]$GR = 1
  
  weather$RA = 0
  weather[grep(x = weather$codesum , pattern = "RA" , fixed = T), ]$RA = 1
  
  weather$DZ = 0
  weather[grep(x = weather$codesum , pattern = "DZ" , fixed = T), ]$DZ = 1
  
  weather$SN = 0
  weather[grep(x = weather$codesum , pattern = "SN" , fixed = T), ]$SN = 1
  
  weather$SG = 0
  weather[grep(x = weather$codesum , pattern = "SG" , fixed = T), ]$SG = 1
  
  weather$GS = 0
  weather[grep(x = weather$codesum , pattern = "GS" , fixed = T), ]$GS = 1
  
  weather$PL = 0
  weather[grep(x = weather$codesum , pattern = "PL" , fixed = T), ]$PL = 1
  
  weather$FG_PLUS = 0
  weather[grep(x = weather$codesum , pattern = "FG+" , fixed = T), ]$FG_PLUS = 1
  
  weather$FG = 0
  weather[grep(x = weather$codesum , pattern = "FG" , fixed = T), ]$FG = 1
  
  weather$BR = 0
  weather[grep(x = weather$codesum , pattern = "BR" , fixed = T), ]$BR = 1
  
  weather$UP = 0
  weather[grep(x = weather$codesum , pattern = "UP" , fixed = T), ]$UP = 1
  
  weather$HZ = 0
  weather[grep(x = weather$codesum , pattern = "HZ" , fixed = T), ]$HZ = 1
  
  weather$FU = 0
  weather[grep(x = weather$codesum , pattern = "FU" , fixed = T), ]$FU = 1
  
  weather$DU = 0
  weather[grep(x = weather$codesum , pattern = "DU" , fixed = T), ]$DU = 1
  
  weather$SS = 0
  weather[grep(x = weather$codesum , pattern = "SS" , fixed = T), ]$SS = 1
  
  weather$SQ = 0
  weather[grep(x = weather$codesum , pattern = "SQ" , fixed = T), ]$SQ = 1
  
  weather$FZ = 0
  weather[grep(x = weather$codesum , pattern = "FZ" , fixed = T), ]$FZ = 1
  
  weather$MI = 0
  weather[grep(x = weather$codesum , pattern = "MI" , fixed = T), ]$MI = 1
  
  weather$PR = 0
  weather[grep(x = weather$codesum , pattern = "PR" , fixed = T), ]$PR = 1
  
  weather$BC = 0
  weather[grep(x = weather$codesum , pattern = "BC" , fixed = T), ]$BC = 1
  
  weather$BL = 0
  weather[grep(x = weather$codesum , pattern = "BL" , fixed = T), ]$BL = 1
  
  weather$VC = 0
  weather[grep(x = weather$codesum , pattern = "VC" , fixed = T), ]$VC = 1
  
  weather = weather[, -(grep(x = colnames(weather) ,  pattern = "codesum" )) ]
  
  weather
} 

##################
verbose = T 
source(paste0( getBasePath("preprocess") , "/Impute_Lib.R"))


##################
train = getTrain()
weather = getWeather()


## imputing Xtest 
l = imputeFastFurious (data = weather[,-c(1,2)] , verbose = T , debug = F)
Xtest.imputed = l[[1]]
ImputePredictors = l[[2]]
DecisionMatrix = l[[3]]


































