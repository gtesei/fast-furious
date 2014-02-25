imputeNAMean <- function(dd) {
  for (i in 1:length(dd)) {
    goodIndices = complete.cases(dd[,i])	 
    badIndices = is.na(dd[,i])
    m = mean (dd[,i][goodIndices])
    dd[,i][badIndices] <- m 
  }
  dd
}
data <- read.csv('test_v2.zat')
sum(is.na(data))
data <- imputeNAMean(data)
sum(is.na(data))
write.csv(data,quote=FALSE,file='test_impute_mean.csv',row.names=FALSE)

