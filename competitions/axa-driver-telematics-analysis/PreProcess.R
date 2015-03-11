
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

store = function (driver = 0 , data , label) {
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
  
  fn = paste(ret,label,"_",driver,".csv",sep="")
  write.csv(data,quote=FALSE,file=fn, row.names=FALSE)
  fn
} 

findMainFrequencyComponent  = function (fs , Time , sign, doPlot = F) { 
  t = seq(from = 0, to = (Time-1/fs), by= 1/fs)    
  #n = 2^ceil(log(length(sign),2))
  n = length(sign)
  
  FFT = fft(sign)/n
  MOD = Mod(FFT)
  power = MOD^2
  fm = min(which( power == max(power)))  * (fs/n)  
  if (doPlot) {
    f = seq(from = 0 , to = (n-1) , by = 1) * (fs/n)  
    plot(x=f,y=power,type="l" , xlab="Hz",ylab="u.m. of observed phenomenon")
    #     magn.1 <- MOD[1:(length(MOD)/2)]
    #     x.axis <- 1:length(magn.1)/Time
    #     plot(x=x.axis,y=magn.1,type="l" , xlab="Hz",ylab="u.m. of observed phenomenon")
  }
  
  fm
}

powerSpect  = function (fs,Time,sign) { 
  t = seq(from = 0, to = (Time-1/fs), by= 1/fs)    
  n = length(t)
  
  FFT = fft(sign)/n
  #sum( FFT * Conj(FFT) )
  sum(Mod(FFT)^2)
}

findSpectralEdgeFrequency = function (fs,Time,sign,edge=0.5) {
  # fs = Sample frequency (Hz)
  # Time = secs sample
  # x = signal
  
  t = seq(from = 0, to = (Time-1/fs), by= 1/fs)    
  n = length(t)
  
  swaveX = fft(sign) / n
  powerFunction =  Mod(swaveX)^2
  power = sum(powerFunction) 
  
  nyquistfreq = fs/2
  
  spect = seq(from = 0 , to = nyquistfreq , by = fs/n)
  
  se = -1
  for (sp in 1:length(spect)) {
    pp = sum(powerFunction[0:sp])
    if (pp >= power * edge) {
      se = spect[sp]
      break
    }
  } 
  
  se
}

findSpectralEdges = function (fs,Time,sign,th=c(0.25,0.50,0.75)) {
  # fs = Sample frequency (Hz)
  # Time = secs sample
  # x = signal
  
  edges = rep(-1,length(th))
  
  t = seq(from = 0, to = (Time-1/fs), by= 1/fs)    
  n = length(t)
  
  swaveX = fft(sign) / n
  powerFunction =  Mod(swaveX)^2
  power = sum(powerFunction) / 2 
  
  nyquistfreq = fs/2
  
  spect = seq(from = 0 , to = nyquistfreq , by = fs/n)
  
  for (sp in 1:length(spect)) {
    pp = sum(powerFunction[0:sp])
    for (ee in 1:length(edges)) {
      if (2* pp >= power * th[ee]) {
        if (edges[ee] == -1) edges[ee] = spect[sp]
        if (sum((edges) == -1) == 0 ) break
      }
    }
  } 
  
  edges
}

######################### settings ans constants 
debug = F

######################### main loop 

ALL_ONES = c(1634)
FROM = 1634 

#DRIVERS = c(1634)
##TRIPS = c(1,26)

ALL_DRIVERS = getDrivers() 
if (exists("DRIVERS")) 
  ALL_DRIVERS = intersect(ALL_DRIVERS,DRIVERS)
  
cat("|--------------------------------->>> found ",length(ALL_DRIVERS)," drivers ... \n")

for ( drv in ALL_DRIVERS  ) {
  
    if (exists("ALL_ONES") && is.element(el = drv , set = ALL_ONES)) next 
    if (exists("FROM") && drv < FROM ) next 
  
    trips = getTrips(drv) 
    if (exists("TRIPS")) 
      trips = intersect(trips,TRIPS)
    
    features = data.frame(trip = trips, 
                          #### features eliminate perchè dipendenti dalla durata del sampling 
                          #rho.mean = -1 , rho.std = -1 , rho.kur = -1, rho.skew =-1, 
                          #alpha.mean = -1 , alpha.std = -1 , alpha.kur = -1, alpha.skew =-1, 
                          
                          V.mean = -1 , V.std = -1 , V.kur = -1, V.skew =-1, 
                          V.msc = -1 , 
                          #### features eliminate perchè dipendenti dalla durata del sampling 
                          #V.pow = -1 , 
                          V.pow25 = -1 , V.pow50 = -1 , V.pow75 = -1,
                          V.30 = -1 , V.60 = -1 , V.90 = -1 , 
                          
                          A.mean =-1, A.std =-1, A.kur =-1, A.skew =-1, 
                          A.msc = -1 , 
                          #### features eliminate perchè dipendenti dalla durata del sampling 
                          #A.pow = -1 , 
                          A.pow25 = -1 , A.pow50 = -1 , A.pow75 = -1)
    
    cat("|---------------->>> found <<",length(trips),">>  trips for driver <<",drv,">> ..\n")
    for (trip in trips) {
        
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
#         rho.mean = mean(trdata$rho)
#         rho.std = sd(trdata$rho)
#         rho.skew = skewness(trdata$rho)
#         rho.kur = kurtosis(trdata$rho)
#         
#         alpha.mean = mean(trdata$alpha)
#         alpha.std = sd(trdata$alpha)
#         alpha.skew = skewness(trdata$alpha)
#         alpha.kur = kurtosis(trdata$alpha)
        
        ## V
        V.mean = mean(trdata$V)
        V.std = sd(trdata$V)
        V.skew = skewness(trdata$V)
        V.kur = kurtosis(trdata$V)
        
        ## V.msc 
        V.msc = findMainFrequencyComponent(fs=1,Time=dim(trdata)[1],sign=trdata$V,doPlot=F)
        
        ## V.pow 
        #V.pow = powerSpect(fs=1,Time=dim(trdata)[1],sign=trdata$V) 

        V.quant = as.numeric(quantile(trdata$V, seq(0.3,1, by = 0.3)))
        
        edges = findSpectralEdges (fs=1,Time=dim(trdata)[1],sign=trdata$V)
        
        V.pow25 = edges[1]
        V.pow50 = edges[2]
        V.pow75 = edges[3]
        
        ## A
        A.mean = mean(trdata$A)
        A.std = sd(trdata$A)
        A.skew = skewness(trdata$A)
        A.kur = kurtosis(trdata$A)
        
        ## A.msc 
        A.msc = findMainFrequencyComponent(fs=1,Time=dim(trdata)[1],sign=trdata$A,doPlot=F)
        
        ## A.pow 
        #A.pow = powerSpect(fs=1,Time=dim(trdata)[1],sign=trdata$A) 
        
        edges = findSpectralEdges (fs=1,Time=dim(trdata)[1],sign=trdata$A)
        
        A.pow25 = edges[1]
        A.pow50 = edges[2]
        A.pow75 = edges[3]
        
        #### update featutures set   
#### features eliminate perchè dipendenti dalla durata del sampling 
#         features[features$trip == trip , ]$rho.mean = rho.mean
#         features[features$trip == trip , ]$rho.std = rho.std
#         features[features$trip == trip , ]$rho.skew = rho.skew
#         features[features$trip == trip , ]$rho.kur = rho.kur
#         
#         features[features$trip == trip , ]$alpha.mean = alpha.mean
#         features[features$trip == trip , ]$alpha.std = alpha.std
#         features[features$trip == trip , ]$alpha.skew = alpha.skew
#         features[features$trip == trip , ]$alpha.kur = alpha.kur
        
        ## V 
        features[features$trip == trip , ]$V.mean = V.mean
        features[features$trip == trip , ]$V.std = V.std
        features[features$trip == trip , ]$V.skew = V.skew
        features[features$trip == trip , ]$V.kur = V.kur
        
        features[features$trip == trip , ]$V.msc = V.msc
#### features eliminate perchè dipendenti dalla durata del sampling 
        #features[features$trip == trip , ]$V.pow = V.pow
        
        features[features$trip == trip , ]$V.pow25 = V.pow25
        features[features$trip == trip , ]$V.pow50 = V.pow50
        features[features$trip == trip , ]$V.pow75 = V.pow75

        features[features$trip == trip , ]$V.30 = V.quant[1]
        features[features$trip == trip , ]$V.60 = V.quant[2]
        features[features$trip == trip , ]$V.90 = V.quant[3]
        
        ## A
        features[features$trip == trip , ]$A.mean = A.mean
        features[features$trip == trip , ]$A.std = A.std
        features[features$trip == trip , ]$A.skew = A.skew
        features[features$trip == trip , ]$A.kur = A.kur
        
        features[features$trip == trip , ]$A.msc = A.msc
#### features eliminate perchè dipendenti dalla durata del sampling 
        #features[features$trip == trip , ]$A.pow = A.pow
        
        features[features$trip == trip , ]$A.pow25 = A.pow25
        features[features$trip == trip , ]$A.pow50 = A.pow50
        features[features$trip == trip , ]$A.pow75 = A.pow75
        
        features = features[with(features, order(trip)), ]
        
        if (debug) print(head(trdata))
        
        ############ TRIP PROCESSING - end 
      }
    
    cat("********* building reduced matrix .... \n")
        
    #### features reduction 
    features.red = features 
    
    ### 1 - removing columns with NAs values both in features and features.red
    features.na = apply(features,2,function(x) sum(is.na(x)) > 0 )
    if ( sum(sum(features.na) > 0 ) ) {
      cat("removing columns with NAs values both in features and features.red: ",colnames(features.red)[features.na]," ... \n ")
      
      features.red = features.red[-which(features.na)]
      features = features[-which(features.na)]
    } 
    
    ### 2 - removing near zero var predictors 
    PredToDel = nearZeroVar(features.red)
    if (length(PredToDel) > 0) {
      cat("removing ",length(PredToDel)," nearZeroVar predictors: ", paste(colnames(features.red) [PredToDel] , collapse=" " ) , " ... \n ")
      features.red  =  features.red  [,-PredToDel]
    }
    
    ### 3 - removing predictors that make ill-conditioned square matrix
    PredToDel = trim.matrix( cov( features.red ) )
    if (length(PredToDel$numbers.discarded) > 0  ) {
    #if (length(PredToDel$numbers.discarded) > 0 & (dim(features.red)[2]-length(PredToDel$numbers.discarded)>3) ) {
      cat("removing ",length(PredToDel$numbers.discarded)," predictors that make ill-conditioned square matrix: ", paste(colnames(features.red) [PredToDel$numbers.discarded] , collapse=" " ) , " ... \n ")
      features.red  =  features.red  [,-PredToDel$numbers.discarded]
    }
    
    # 4 - rmoving high correlated predictors 
    PredToDel = findCorrelation(cor( features.red )) 
    #if (  (dim(features.red)[2]-length(PredToDel)>3)  ) {
      cat("PLS:: on features.red removing ",length(PredToDel), " predictors: ",paste(colnames(features.red) [PredToDel] , collapse=" " ) , " ... \n ")
      features.red =  features.red  [,-PredToDel]
    #}
    
    if (debug) print(head(features)) 
    if (debug) print(head(features.red)) 
    
    ## store 
    store(drv,features,"features")
    store(drv,features.red,"features_red")
    
    ## some statistics 
    #features
    mean = apply(features,2,mean)
    cat("****** mean features ******\n")
    print(mean)
    cat("****** sd features ******\n")
    sd = apply (features,2,sd)
    print(sd)
#     cat("****** zeta score features ******\n")
#     zeta.score = apply(features,2,function(x) {
#       (x-mean(x))/sd(x)
#     }) 
#     print(zeta.score)
    
    #features.red
    mean = apply(features.red,2,mean)
    cat("****** mean features.red ******\n")
    print(mean)
    cat("****** sd features.red ******\n")
    sd = apply (features.red,2,sd)
    print(sd)
#     cat("****** zeta score features.red ******\n")
#     zeta.score = apply(features.red,2,function(x) {
#       (x-mean(x))/sd(x)
#     }) 
#     print(zeta.score)
} 


