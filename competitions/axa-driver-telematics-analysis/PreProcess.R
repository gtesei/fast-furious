
library(caret)
library(Hmisc)
library(data.table)
library(verification)
library(pROC)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)

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

######################### settings ans constants 
debug = T

######################### main loop 

DRIVERS = c(1,2)
TRIPS = c(1,2)

drv = 1
trip = 1

while (! is.na(getTripPath(drv,trip))) {
  
  if ( ! (exists("DRIVERS") & ! is.element(drv,DRIVERS))  ) {
     
    while (! is.na(getTripPath(drv,trip))) {
      if ( ! (exists("TRIPS") & ! is.element(trip,TRIPS))  ) {
        
        ############ TRIP PROCESSING - begin 
        cat("|---------------->>> processing driver: <<",drv,">>  <<",trip,">> ..\n")
        
        #### load data 
        trdata = as.data.frame(fread(getTripPath(drv,trip)))
        if (debug) print(head(trdata))
        
        #### extract 1-tier features  
        
        ## rho 
        trdata$rho = apply(trdata,1,function(x) {
              sqrt(x[1]^2 + x[2]^2)
          	})
        
        ## alpha 
        trdata$alpha = apply(trdata,1,function(x) {
          ret = 0 
          if (x[1] == 0) ret = 0
          else ret = atan(x[2]/x[1])
          ret
        })
        
        ## vx 
        trdata$vx =  trdata$x - shift(trdata$x,1)
        trdata$vx[1] = 0
        
        ## vy
        trdata$vy =  trdata$y - shift(trdata$y,1)
        trdata$vy[1] = 0
        
        ## V 
        trdata$V =  apply(trdata,1,function(x) {
          sqrt(x[5]^2 + x[6]^2)
        })
        
        ## ax 
        trdata$ax =  trdata$vx - shift(trdata$vx,1)
        trdata$ax[1] = 0
        
        ## ay
        trdata$ay =  trdata$vy - shift(trdata$vy,1)
        trdata$ay[1] = 0
        
        ## A 
        trdata$A =  apply(trdata,1,function(x) {
          sqrt(x[8]^2 + x[9]^2)
        })
        #### extract 2-tier features  
        
        #### update featutures set   
        
        if (debug) print(head(trdata))
        ############ TRIP PROCESSING - end 
      }
      
      trip = trip + 1 
    } 
  }
  
  drv = drv + 1 
  trip = 1
} 


