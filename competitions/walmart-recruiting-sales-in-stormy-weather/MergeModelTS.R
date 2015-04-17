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
rain = getTrain()
test = getTest()

model.grid = as.data.frame( fread(paste(getBasePath("data") , 
                                  "mySub_grid.csv" , sep='')))

ts.grid = as.data.frame( fread(paste(getBasePath("data") , 
                                        "mySubTS_grid.csv" , sep='')))

model.sub = as.data.frame( fread(paste(getBasePath("data") , 
                                     "mySub.csv" , sep='')))

ts.sub = as.data.frame( fread(paste(getBasePath("data") , 
                                     "mySubTS.csv" , sep='')))

sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

keys = as.data.frame( fread(paste(getBasePath("data") , 
                                  "key.csv" , sep='')))

stores.test = sort(unique(test$store_nbr))
items.test = sort(unique(test$item_nbr))

################

sub = NULL

for (st in stores.test) {
  stat = keys[keys$store_nbr == st,]$station_nbr 
  for (it in items.test) {
    pred = NULL
    
    cat (">>>> processing stores <",st,"> - station <",stat,">- item <",it,"> ... \n") 
    
    model.best.perf = model.grid[model.grid$store == st & model.grid$item == it , ]$best.perf
    ts.best.perf = ts.grid[ts.grid$store == st & ts.grid$item == it , ]$best.perf
    
    if (model.best.perf <= ts.best.perf) {
      cat (">>>> winning model <",model.grid[model.grid$store == st & model.grid$item == it , ]$best.model
           ,"> ... \n") 
      pred = model.sub[grep(x = model.sub$id , pattern = paste("^",st,"_",it,"_",sep='') ), ]$units 
    } else {
      cat (">>>> winning TS <",ts.grid[ts.grid$store == st & ts.grid$item == it , ]$best.model
           ,"> ... \n") 
      pred = ts.sub[grep(x = ts.sub$id , pattern = paste("^",st,"_",it,"_",sep='') ), ]$units 
    }
    
    ###### Updating submission
    if (verbose) cat("Updating submission ... \n")
    id = ts.sub[grep(x = ts.sub$id , pattern = paste("^",st,"_",it,"_",sep='') ), ]$id
    sub.chunck = data.frame(id = id , units = pred)
    if (is.null(sub)) {
      sub = sub.chunck
    } else {
      sub = rbind(sub,sub.chunck)
    }
  }
}

### perform some checks 
if (dim(sub)[1] != dim(sampleSubmission)[1]) 
  stop (paste("sampleSubmission has ",dim(sampleSubmission)[1]," vs sub that has ",dim(sub)[1]," rows!!"))

if ( sum(!(sub$id %in% sampleSubmission$id)) > 0 ) 
  stop("sub has some ids different from sampleSubmission ids !!")

if ( sum(!(sampleSubmission$id %in% sub$id)) > 0 ) 
  stop("sampleSubmission has some ids different from sub ids !!")

### storing on disk 
write.csv(sub,quote=FALSE, 
          file=paste(getBasePath("data"),"mySub_merge_model_ts.csv",sep='') ,
          row.names=FALSE)

cat("<<<<< submission correctly stored on disk >>>>>\n") 





