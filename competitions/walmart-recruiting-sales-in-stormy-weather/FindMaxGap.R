library(data.table)

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/walmart-recruiting-sales-in-stormy-weather"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/walmart-recruiting-sales-in-stormy-weather/"
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

getTrain = function () {
  path = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/train.csv"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/train.csv"
  
  if (file.exists(base.path1))  {
    path = base.path1
  } else if (file.exists(base.path2)) {
    path = base.path2
  } else {
    stop('impossible load train.csv')
  }
  
  cat("loading train data ... ")
  trdata = as.data.frame(fread(path))
  #cat("converting date ...")
  #trdata$date = as.Date(trdata$date,"%Y-%m-%d")
  trdata
} 
getTest = function () {
  path = ""
  
  base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/test.csv"
  base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/test.csv"
  
  if (file.exists(base.path1))  {
    path = base.path1
  } else if (file.exists(base.path2)) {
    path = base.path2
  } else {
    stop('impossible load train.csv')
  }
  
  cat("loading train data ... ")
  trdata = as.data.frame(fread(path))
  #cat("converting date ...")
  #trdata$date = as.Date(trdata$date,"%Y-%m-%d")
  trdata
} 

#########
verbose = T 

#########
train = getTrain()
test = getTest()

model.grid = as.data.frame( fread(paste(getBasePath("data") , 
                                        "mySub_grid.csv" , sep='')))

ts.grid = as.data.frame( fread(paste(getBasePath("data") , 
                                     "mySubTS_grid.csv" , sep='')))

model.sub = as.data.frame( fread(paste(getBasePath("data") , 
                                       "mySub.csv" , sep='')))

sub = as.data.frame( fread(paste(getBasePath("data") , 
                                 "mySub_adj.csv" , sep='')))

sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

keys = as.data.frame( fread(paste(getBasePath("data") , 
                                  "key.csv" , sep='')))

weather = as.data.frame( fread(paste(getBasePath("data") , 
                                     "weather.imputed.basic.17.9.csv" , sep=''))) ## <<<< TODO use weather.imputed.all.<perf>.csv



stores.test = sort(unique(test$store_nbr))
items.test = sort(unique(test$item_nbr))

################
verbose = T 
################

###### find mean sold units in training set 
train.mean = ddply(train , 
                   .(store_nbr,item_nbr),
                   function(x) c(units.mean.train=mean(x$units))
                   )


###### find mean sold units in test set 
sub.stores = NULL
sub.items = NULL

tokens = (unlist(strsplit(sub$id, "_")))
cat("processing tokens ... \n")
sub.stores = as.numeric(tokens[(1:length(tokens) %% 3 == 1)])
sub.items = as.numeric(tokens[(1:length(tokens) %% 3 == 2)])
cat("processed tokens ... \n")

sub$store_nbr = sub.stores
sub$item_nbr = sub.items

sub.mean = ddply(sub , 
                   .(store_nbr,item_nbr),
                   function(x) c(units.mean.sub=mean(x$units))
)

#########
data = merge(x = train.mean, y = sub.mean, by=c("store_nbr","item_nbr") , all = T) 
data$delta = ( data$units.mean.train - data$units.mean.sub )
data$delta.abs = abs(data$delta) 
data = data[order(data$delta.abs , decreasing = T),]
head(data)

