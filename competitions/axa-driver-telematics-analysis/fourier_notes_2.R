
library(caret)
library(Hmisc)
library(data.table)
library(verification)
library(pROC)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)

getTrips = function (drv = 0) {
  ret = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/drivers/"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/drivers/"
  
  base.path1 = paste( base.path1, as.character(drv) , sep = "") 
  base.path2 = paste( base.path2, as.character(drv) , "/" , sep = "") 
  
  if (file.exists(base.path1))  {
    
    ret = order(as.numeric(lapply(as.character(list.files(base.path1)), function(x) as.numeric ( substr(x, 1, nchar(x) - 4)) ) 
                           , decreasing = T))
    
  } else if (file.exists(base.path2)) {
    
    ret = order(as.numeric(lapply(as.character(list.files(base.path2)), function(x) as.numeric(substr(x, 1, nchar(x) - 4)) ) 
                           , decreasing = T))
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
    
    ret = order (as.numeric (lapply(list.files(base.path1), function(x) as.numeric (x)) , decreasing = F) )
    
  } else if (file.exists(base.path2)) {
    
    ret = order (as.numeric (lapply(list.files(base.path2), function(x) as.numeric (x)) , decreasing = F) )
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


##
sign = seq(1:100) 
powerSpect(1,100,sign) ## 3383.5+0i come Octave 
findMainFrequencyComponent(1,100,sign,T)

##
xs <- seq(-2*pi,2*pi,pi/100)
wave.1 <- 10* sin(6*pi*xs)
wave.2 <- 20 * sin(4*pi*xs)
sign = wave.1 + wave.2 
plot(xs,sign,type="l")
powerSpect(1,100,sign)
findMainFrequencyComponent(1,100,sign,T)
findSpectralEdgeFrequency (1,100,sign) 

##
Time = 10 
fs = 100 
t = seq(from = 0 , to = (Time-(1/fs)) ,   by = (1/fs) ) 
sign = (1.3)*sin(2*pi*15*t) + (1.7)*sin(2*pi*40*(t-2))
plot(sign,type="l")
powerSpect(fs,Time,sign)
findMainFrequencyComponent(fs,Time,sign,T)
findSpectralEdgeFrequency (fs,Time,sign) 
findSpectralEdgeFrequency (fs,Time,sign,0.25) 
findSpectralEdgeFrequency (fs,Time,sign,0.1) 

findSpectralEdges (fs,Time,sign)




