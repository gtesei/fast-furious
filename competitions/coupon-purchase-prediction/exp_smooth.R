


weeks = 0:50

axisRange <- extendrange( seq(from = 0, to = 1 , by = 0.01))

plot(x = weeks , y = rep(1,length(weeks)) , type='l' , col='red' , xlab = expression(paste(Delta,"weeks")), 
     ylab = "sales" , ylim = axisRange, 
     main="Sales exponential smoothed")

par(new=TRUE)
exp_smooth_0.75 = 1/50 * log(0.75)
sales_smooth_0.75 = exp((weeks)*exp_smooth_0.75)
plot( x = weeks , y = sales_smooth_0.75,  type='l', pch=27 , col="blue", lty=6, lwd=2  , xlab = "", ylab = "" , ylim = axisRange)

par(new=TRUE)
exp_smooth_0.5 = 1/50 * log(1/2)
sales_smooth_05 = exp((weeks)*exp_smooth_0.5)
plot( x = weeks , y = sales_smooth_05,  type='l', pch=27 , col="blue", lty=6, lwd=2  , xlab = "", ylab = "" , ylim = axisRange)

par(new=TRUE)
exp_smooth_0.25 = 1/50 * log(1/4)
sales_smooth_0.25 = exp((weeks)*exp_smooth_0.25)
plot( x = weeks , y = sales_smooth_0.25,  type='l', pch=27 , col="blue", lty=6, lwd=2, xlab = "", ylab = "", ylim = axisRange)

par(new=TRUE)
exp_smooth_0.05 = 1/50 * log(0.05)
sales_smooth_0.05 = exp((weeks)*exp_smooth_0.05)
plot( x = weeks , y = sales_smooth_0.05,  type='l', pch=27 , col="blue", lty=6, lwd=2, xlab = "", ylab = "", ylim = axisRange)

par(new=TRUE)
exp_smooth2_0.25 = 1/50 * log(0.25)
sales_smooth2_0.25 = exp((50-weeks)*exp_smooth2_0.25)
plot( x = weeks , y = sales_smooth2_0.25,  type='l', pch=27 , col="green", lty=6, lwd=2 , xlab = "", ylab = "", ylim = axisRange) 

par(new=TRUE)
exp_smooth2_0.5 = 1/50 * log(1/2)
sales_smooth2_0.5 = exp((50-weeks)*exp_smooth2_0.5)
plot( x = weeks , y = sales_smooth2_0.5,  type='l', pch=27 , col="green", lty=6, lwd=2 , xlab = "", ylab = "", ylim = axisRange) 

par(new=TRUE)
exp_smooth2_0.75 = 1/50 * log(0.75)
sales_smooth2_0.75 = exp((50-weeks)*exp_smooth2_0.75)
plot( x = weeks , y = sales_smooth2_0.75,  type='l', pch=27 , col="green", lty=6, lwd=2 , xlab = "", ylab = "", ylim = axisRange) 

par(new=TRUE)
exp_smooth2_0.05 = 1/50 * log(0.05)
sales_smooth2_0.05 = exp((50-weeks)*exp_smooth2_0.05)
plot( x = weeks , y = sales_smooth2_0.05,  type='l', pch=27 , col="green", lty=6, lwd=2 , xlab = "", ylab = "", ylim = axisRange) 



