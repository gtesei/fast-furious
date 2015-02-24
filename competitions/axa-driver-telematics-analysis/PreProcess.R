
library(caret)
library(Hmisc)
library(data.table)
library(verification)
library(pROC)
library(kernlab)
library(subselect)
library(plyr)

getTripPath = function (driver = 0 , trip = 0) {
  ret = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/drivers/"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/drivers/"
  
  base.path1 = paste( base.path1, as.character(driver) , "/" , as.character(trip) , ".csv" , sep = "") 
  base.path2 = paste( base.path2, as.character(driver) , "/" , as.character(trip) , ".csv" , sep = "") 
  
  if (file.exists(base.path1))  {
    ret = base.path1
  } else if (file.exists(base.path2)) {
    ret = base.path2
  } else {
    ret = NA
  }
  
  ret
} 







######################### main loop 

DRIVERS = c(1,2)
TRIPS = c(1,2)

driver = 1
trip = 1

while (! is.na(getTripPath(driver,trip))) {
  trip = 1
  if ( ! (exists("DRIVERS") & ! is.element(driver,DRIVERS))  ) {
     
    while (! is.na(getTripPath(driver,trip))) {
      if ( ! (exists("TRIPS") & ! is.element(trip,TRIPS))  ) {
        
        cat("|---------------->>> processing driver: <<",driver,">>  <<",trip,">> ..\n")
      }
      
      trip = trip + 1 
    } 
  }
  
  driver = driver + 1 
  
  
  
  
} 


