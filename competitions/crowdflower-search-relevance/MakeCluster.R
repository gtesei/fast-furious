library(readr)
library(tm)
library(NLP)
require(xgboost)
require(methods)
require(plyr)
library(NbClust)
library(cluster)
library(data.table)
library(fpc)


getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/crowdflower-search-relevance"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/crowdflower-search-relevance/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/crowdflower-search-relevance"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/ocrowdflower-search-relevance/"
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

####### load 
cat(">>> loading queries and query_vect .... \n")
load(file=paste(getBasePath("data") , "query_vect" , sep=''))
load(file=paste(getBasePath("data") , "queries" , sep=''))

###### 
# query_vect_df = as.data.frame(query_vect)
# 
# nb = NbClust(query_vect, min.nc=2, max.nc=15 
#           , distance = "euclidean"
#           , method = "kmeans")

# We take the first iteration
wss <- sum(kmeans(query_vect,centers=1)$withinss)
# We take iteration 2 to 15
for (i in 2:50) wss[i] <- sum(kmeans(query_vect,centers=i)$withinss)

# We plot the 15 withinss values. One for each k
par(mfrow=c(1,1))
plot(1:50, wss, type="l", xlab="Number of Clusters",ylab="Within groups sum of squares")

wss_delta = rep(0,length(wss)-1)
for (i in 1:length(wss)-1) wss_delta[i] = (wss[i+1] - wss[i])/wss[i+1]

n_cluters = which(wss_delta == min(wss_delta))

### 
kopt = kmeans(query_vect,centers=n_cluters)

my_cluster = kopt$cluster
save(my_cluster,file=paste(getBasePath("data") , "my_cluster_knn" , sep=''))

