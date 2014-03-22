
getPerformance = function(pred , val) {
  	res = pred - val
  	MAE = sum(abs(res)) / length(val) 
  	RSS = sum(res^2)
  	MSE = RSS / length(val)
  	RMSE = sqrt(MSE)
  	R2 = 1 - ( RSS/ ( sum((val-mean(val))^2) ) )
  	
  	perf = data.frame(MAE,RSS,MSE,RMSE,R2)
}



################
# CAP 1 
################

############# passengers aggregate annual series and seasonal values  
data(AirPassengers)
AP = AirPassengers
par(mfrow=c(3,1))
plot(AP , ylab = "Passengers (1000's)")
plot(aggregate(AP) , ylab="Aggregated annual series")
boxplot(AP ~ cycle(AP) , ylab = "Boxplot of seasonal values")


