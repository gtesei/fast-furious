##### Complex Wave
xs <- seq(-2*pi,2*pi,pi/100)
wave.1 <- sin(3*xs)
wave.2 <- sin(10*xs)
par(mfrow = c(1, 2))
plot(xs,wave.1,type="l",ylim=c(-1,1)); abline(h=0,lty=3)
plot(xs,wave.2,type="l",ylim=c(-1,1)); abline(h=0,lty=3)

## linear combination 
wave.3 <- 0.5 * wave.1 + 0.25 * wave.2
plot(xs,wave.3,type="l"); title("Eg complex wave"); abline(h=0,lty=3)

## overflowed, non-linear complex wave
wave.4 <- wave.3
wave.4[wave.3>0.5] <- 0.5
plot(xs,wave.4,type="l",ylim=c(-1.25,1.25)); title("overflowed, non-linear complex wave"); abline(h=0,lty=3)

## Repeating pattern
repeat.xs     <- seq(-2*pi,0,pi/100)
wave.3.repeat <- 0.5*sin(3*repeat.xs) + 0.25*sin(10*repeat.xs)
plot(xs,wave.3,type="l"); title("Repeating pattern")
points(repeat.xs,wave.3.repeat,type="l",col="red"); abline(h=0,v=c(-2*pi,0),lty=3)


##### Fourier Trasform 
plot.fourier <- function(fourier.series, f.0, ts) {
  w <- 2*pi*f.0
  trajectory <- sapply(ts, function(t) fourier.series(t,w))
  plot(ts, trajectory, type="l", xlab="time", ylab="f(t)"); abline(h=0,lty=3)
}

# An eg
plot.fourier(function(t,w) {sin(w*t)}, 1, ts=seq(0,1,1/100)) 

## example x costruire un segnale nel dominio del tempo date le caratteristiche nel dominio in freq
acq.freq <- 100                    # data acquisition frequency (Hz)
time     <- 6                      # measuring time interval (seconds)
ts       <- seq(0,time,1/acq.freq) # vector of sampling time-points (s) 
f.0      <- 1/time                 # fundamental frequency (Hz)

dc.component       <- 0
component.freqs    <- c(3,10)      # frequency of signal components (Hz)
component.delay    <- c(0,0)       # delay of signal components (radians)
component.strength <- c(.5,.25)    # strength of signal components

f <- function(t,w) { 
  dc.component + 
    sum( component.strength * sin(component.freqs*w*t + component.delay)) 
}

plot.fourier(f,f.0,ts)  

## Phase Shifts
component.delay <- c(pi/2,0)       # delay of signal components (radians)
plot.fourier(f,f.0,ts)

## DC Components
dc.component <- -2
plot.fourier(f,f.0,ts)

##### Fast Fourier Transform (FFT)
## the Discrete Fourier Transform (DFT) which requires \(O(n^2)\) operations (for \(n\) samples)
## the Fast Fourier Transform (FFT) which requires \(O(n.log(n))\) operations

## basic 
library(stats)
fft(c(1,1,1,1)) / 4  # to normalize
fft(1:4) / 4  

### example 
set.seed(101)
acq.freq <- 200
time     <- 1
w        <- 2*pi/time
ts       <- seq(0,time,1/acq.freq)
trajectory <- 3*rnorm(101) + 3*sin(3*w*ts)
plot(trajectory, type="l")

X.k <- fft(trajectory)

plot.frequency.spectrum <- function(X.k, xlimits=c(0,length(X.k))) {
  plot.data  <- cbind(0:(length(X.k)-1), Mod(X.k))
  
  # TODO: why this scaling is necessary?
  plot.data[2:length(X.k),2] <- 2*plot.data[2:length(X.k),2] 
  
  plot(plot.data, t="h", lwd=2, main="", 
       xlab="Frequency (Hz)", ylab="Strength", 
       xlim=xlimits, ylim=c(0,max(Mod(plot.data[,2]))))
}

plot.frequency.spectrum(X.k,xlimits=c(0,acq.freq/2))

### library GeneCycle 
library(GeneCycle)

f.data <- GeneCycle::periodogram(trajectory)
harmonics <- 1:(acq.freq/2)

plot(f.data$freq[harmonics]*length(trajectory), 
     f.data$spec[harmonics]/sum(f.data$spec), 
     xlab="Harmonics (Hz)", ylab="Amplitute Density", type="h")

######
require(stats)
#Domain setup
T = 10 
dt <- 1/100 #s
n <- T/dt
freq = 66
df <- 1/T
t <- seq(0,T,by=dt) #also try ts function
#CREATE OUR TIME SERIES DATA
y <- 10*sin(2*pi*freq*t) +4* sin(2*pi*20*t)

#CREATE OUR FREQUENCY ARRAY
f <- 1:length(t)/T

#FOURIER TRANSFORM WORK
Y <- fft(y)
mag <- sqrt(Re(Y)^2+Im(Y)^2)*2/n
phase <- atan(Im(Y)/Re(Y))
Yr <- Re(Y)
Yi <- Im(Y)

#PLOTTING
layout(matrix(c(1,2), 2, 1, byrow = TRUE))
plot(t,y,type="l",xlim=c(0,T))
plot(f[1:length(f)/2],mag[1:length(f)/2],type="l")

which(mag == max(mag)) * dt ## max comp freq ?


#######
---------------------------------------------------------------------------
  # R-script
acq.freq <- 4000       # data acquisition frequency (Hz)
sig1.freq <- 50           # frequency of 1st signal component (Hz)
sig2.freq <- 130        # frequency of 2nd signal component (Hz)
time <- 5                    # measuring time interval (s)

# vector of sampling time-points (s)
smpl.int <- (1:(time*acq.freq))/acq.freq  

# data vector containing two frequencies (2nd frequ with phase shift)
data <- 10 * sin(sig1.freq*smpl.int*2*pi)+20 * sin(sig2.freq*smpl.int*2*pi+pi/2)

plot(data,type="l")

# calculate fft of data
test <- fft(data)

# extract magnitudes and phases
magn <- Mod(test) # sqrt(Re(test)*Re(test)+Im(test)*Im(test))
phase <- Arg(test) # atan(Im(test)/Re(test))

# select only first half of vectors
magn.1 <- magn[1:(length(magn)/2)]
#phase.1 <- Arg(test)[1:(length(test)/2)]

# plot various vectors

# plot magnitudes as analyses by R
plot(magn,type="l")

# plot first half of magnitude vector
plot(magn.1,type="l")

# generate x-axis with frequencies
x.axis <- 1:length(magn.1)/time

# plot magnitudes against frequencies
plot(x=x.axis,y=magn.1,type="l")

#### find freq with max magnitude 
mi = which(magn.1 == max(magn.1))
x.axis[mi]


