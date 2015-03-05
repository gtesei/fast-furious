
library(NbClust)

library(cluster)
library(data.table)
library(fpc)

wssplot <- function(data, nc=15, seed=1234) {
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nc) {
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i)$withinss)}
    plot(1:nc, wss, type="b", xlab="Number of Clusters", 
         ylab="Within groups sum of squares")
}

storeGrid = function (data , label) {
  ret = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/axa-driver-telematics-analysis"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/axa-driver-telematics-analysis/"
  
  if (file.exists(base.path1))  {
    ret = base.path1
    ret = paste(ret,"/",sep="")
  } else if (file.exists(base.path2)) {
    ret = base.path2
  } else {
    stop("where are u working??")
  }
  
  fn = paste(ret , label  , ".csv", sep="")
  write.csv(data,quote=FALSE,file=fn, row.names=FALSE)
}

loadGrid = function (label) {
  ret = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/axa-driver-telematics-analysis"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/axa-driver-telematics-analysis/"
  
  if (file.exists(base.path1))  {
    ret = base.path1
    ret = paste(ret,"/",sep="")
  } else if (file.exists(base.path2)) {
    ret = base.path2
  } else {
    stop("where are u working??")
  }
  
  fn = paste(ret , label  , ".csv", sep="")
  as.data.frame(fread( fn ))
}

predictNumCentrioids = function(data 
                                , dist = c("euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski")
                                , agg.method = c("ward.D","ward.D2", "single","complete","average","mcquitty","median","centroid","kmeans")
                                , index = c("all","alllong")) {
  
  ## predicting number of clusters 
  simGrid <- expand.grid(dist = dist, agg.method = agg.method , index = index , 
                         clust.num.true = centroids , clust.num.pred = -1 )
  
  
  for (i in (1:(dim(simGrid)[1]))) {
    scenario.lab = paste("[dist=",as.character(simGrid[i,]$dist),"] [agg.method="
                         ,as.character(simGrid[i,]$agg.method),"] [index=",as.character(simGrid[i,]$index),"]",sep="")
    
    cat(i,"/",dim(simGrid)[1],"||----------------->> scenario ",scenario.lab, "\n")
    
    nc = tryCatch ({
      NbClust(data[,-1], min.nc=2, max.nc=15 
              , distance = as.character(simGrid[i,]$dist) 
              , method=as.character(simGrid[i,]$agg.method) 
              , index = as.character(simGrid[i,]$index))
    } , error = function(err) { 
      print(paste("ERROR:  ",err))
      NULL 
    })
    
    if (is.null(nc)) {
      
    } else {
      part = as.numeric(nc$Best.partition)
      part.val = unique(part)
      clust.num.pred = length(part.val)
      
      simGrid[i,]$clust.num.pred = clust.num.pred
      
      cat ("|- predicted ",as.character(simGrid[i,]$clust.num.pred)," vs ",as.character(simGrid[i,]$clust.num.true),"(true) \n")
      
      
      pairs(data[,-1], main = paste("simulated data - basic scenario - predict ", scenario.lab,sep=""),
            pch = 21, bg = MY_COLORS[ 1:clust.num.pred ]  [unclass(part)] )
    }
    
  }
  
  simGrid
} 

predictNumCentrioidsFromGrid = function(data , simGrid) {
  
  for (i in (1:(dim(simGrid)[1]))) {
    scenario.lab = paste("[dist=",as.character(simGrid[i,]$dist),"] [agg.method="
                         ,as.character(simGrid[i,]$agg.method),"] [index=",as.character(simGrid[i,]$index),"]",sep="")
    
    cat(i,"/",dim(simGrid)[1],"||----------------->> scenario ",scenario.lab, "\n")
    
    nc = tryCatch ({
      NbClust(data[,-1], min.nc=2, max.nc=15 
              , distance = as.character(simGrid[i,]$dist) 
              , method=as.character(simGrid[i,]$agg.method) 
              , index = as.character(simGrid[i,]$index))
    } , error = function(err) { 
      print(paste("ERROR:  ",err))
      NULL 
    })
    
    if (is.null(nc)) {
      
    } else {
      part = as.numeric(nc$Best.partition)
      part.val = unique(part)
      clust.num.pred = length(part.val)
      
      simGrid[i,]$clust.num.pred = clust.num.pred
      
      cat ("|- predicted ",as.character(simGrid[i,]$clust.num.pred)," vs ",as.character(simGrid[i,]$clust.num.true),"(true) \n")
      
      
      pairs(data[,-1], main = paste("simulated data - basic scenario - predict ", scenario.lab,sep=""),
            pch = 21, bg = MY_COLORS[ 1:clust.num.pred ]  [unclass(part)] )
    }
    
  }
  
  simGrid
}

generateData = function(vars = 2 , centroids = 4 , obs = 200 , range = 0:10 , centroids.sep.fact = 1 
                        , var1.dummy = F) {
  
  centroids.sep = (2*centroids+2) / centroids.sep.fact
  
  centroids.center = (max(range) - min(range))  / centroids * (1:centroids) 
  centroids.disp = (centroids.center[2] - centroids.center[1]) * 1/ centroids.sep
  centroids.obs = obs / centroids
  
  v1 = c()
  if (! var1.dummy) {
    for (cen in (1:centroids)) {
      v1 = c(v1 , rnorm(n = centroids.obs , mean = centroids.center[cen] , sd = centroids.disp) )
    }
  } else {
    v1 = c(v1 , rnorm(n = obs , mean = ((max(range) - min(range))/2) , sd = ((max(range) - min(range))/3)) )
  }
  
  
  v2 = c()
  for (cen in (1:centroids)) {
    v2 = c(v2 , rnorm(n = centroids.obs , mean = centroids.center[cen] , sd = centroids.disp) )
  }
  
  data = data.frame( v1 = v1 , v2 = v2  ) 
  
  if (vars > 2) {
    for (vv in (1:(vars-2)) ) {
      v1 = c()
      for (cen in (1:centroids)) {
        v1 = c(v1 , rnorm(n = centroids.obs , mean = centroids.center[cen] , sd = centroids.disp) )
      }
      
      data = cbind(data, xx = v1)
    }
  }
  
  colnames(data) = paste("v" , 1:vars , sep="")
  
  ## cluster label 
  c.names = c()
  for (cen in (1:centroids)) {
    c.names = c(c.names,rep(cen,centroids.obs))
  }
  data = cbind(cluster = c.names,data)
  
  ## randomize ..
  data = data[sample((dim(data)[1])) , ]
  rownames(data) = 1:(dim(data)[1])
  data
}

####################################################################################
MY_COLORS = c( "green3", "red" , "yellow" , "blue" , "black" , "brown") 

## data 
data(wine, package="rattle")
head(wine)

df <- scale(wine[-1])                                 

wssplot(df)

set.seed(1234)
df <- scale(features.red[,-1])    

## kmeans - all 
nc <- NbClust(df, min.nc=2, max.nc=15, method="kmeans")
table(nc$Best.nc[1,])

part = as.numeric(nc$Best.partition)
part.val = unique(part)

pairs(df[, sample(1:(dim(df)[2]) , size=(dim(df)[2]/2)) ], main = "wine", 
      pch = 21, bg = MY_COLORS[ sample(1:(length(MY_COLORS)), size=length(part.val)) ]  [unclass(part)] )

barplot(table(nc$Best.nc[1,]),
          xlab="Numer of Clusters", ylab="Number of Criteria",
          main="Number of Clusters Chosen by 26 Criteria")

set.seed(1234)
fit.km <- kmeans(df, length(part.val), nstart=25)                           
fit.km$size


part = fit.km$cluster
part.val = unique(part)

pairs(df[, sample(1:(dim(df)[2]) , size=(dim(df)[2]/2)) ], main = "wine", 
      pch = 21, bg = MY_COLORS[ sample(1:(length(MY_COLORS)), size=length(part.val)) ]  [unclass(part)] )

aggregate(wine[-1], by=list(cluster=fit.km$cluster), mean)

######################
centroids = 4
vars = 3

## 1 - simulated data - basic scenario 
data = generateData(vars = vars , centroids = centroids)

pairs(data[,-1], main = "simulated data - basic scenario - ", 
      pch = 21, bg = MY_COLORS[ 1:centroids ]  [unclass(data$cluster)] )

simGrid = predictNumCentrioids (data 
                                , dist = c("euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski")
                                , agg.method = c("ward.D","ward.D2", "single","complete","average","mcquitty","median","centroid","kmeans")
                                , index = c("all","alllong"))

storeGrid (simGrid , "basic_scenario")

### --> **conclusione** con l'esclusione della distanza binary per il resto tutte te previsioni sono corrette
### -->                 scartiamo la distanza binary 

## 2 - simulated data - scenario - smaller intra-centriods distance 
data = generateData(vars = vars , centroids = centroids , centroids.sep.fact = 3)

pairs(data[,-1], main = "simulated data - basic scenario - ", 
      pch = 21, bg = MY_COLORS[ 1:centroids ]  [unclass(data$cluster)] )


simGrid = predictNumCentrioids (data 
                                    , dist = c("euclidean", "maximum", "manhattan", "canberra", "minkowski")
                                    , agg.method = c("ward.D","ward.D2", "single","complete","average","mcquitty","median","centroid","kmeans")
                                    , index = c("all","alllong"))

storeGrid (simGrid , "scenario_smaller_dist_3")

simGrid = loadGrid("scenario_smaller_dist_3")
ok = simGrid$clust.num.true == simGrid$clust.num.pred

### --> **conclusione** escludiamo circa 30 combinazioni 
simGrid = simGrid[ok,]
rownames(simGrid) = 1:(dim(simGrid)[1])
simGrid$clust.num.pred = -1 
simGrid[simGrid$agg.method=="ward.D2",]$agg.method = "ward" ### mac 
#simGrid = simGrid[simGrid$agg.method=="kmeans",]

## 3 - simulated data - scenario - introducing a dummy features 
data = generateData(vars = vars , centroids = centroids , centroids.sep.fact = 2 , var1.dummy = T)

pairs(data[,-1], main = "simulated data - basic scenario - ", 
      pch = 21, bg = MY_COLORS[ 1:centroids ]  [unclass(data$cluster)] )

simGrid = predictNumCentrioidsFromGrid (data , simGrid)
storeGrid (simGrid , "scenario_smaller_dist_2_var1dummy")


