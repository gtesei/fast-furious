library(caret)
library(Hmisc)
library(data.table)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)
library(fBasics)

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/restaurant-revenue-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/restaurant-revenue-prediction/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/restaurant-revenue-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/restaurant-revenue-prediction/"
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

addObservation = function(train.raw , years.to.taget = 11 , 
                          data  , 
                          yd 
                          ) {
  train.new = train.raw
  
  dt = NULL 
  if ( years.to.taget == 11  ) {
    dt = '10/13/2003'
  } else if (years.to.taget == 13 ) {
    dt = '10/13/2001'
  } else if (years.to.taget == 19 ) {
    dt = '10/13/1995'
  }
  
  tt = data.frame(
    Id = max(train.new$Id)+1 , 
    City = Mode(data[,'City']), 
    'City Group' = Mode(data[,'City Group']), 
    'Open Date' = dt  , 
    Type = Mode(data[,'Type']), 
    P1 = Mode(data[,'P1']), 
    P2 = Mode(data[,'P2']), 
    P3 = Mode(data[,'P3']), 
    P4 = Mode(data[,'P4']), 
    P5 = Mode(data[,'P5']), 
    P6 = Mode(data[,'P6']), 
    P7 = Mode(data[,'P7']), 
    P8 = Mode(data[,'P8']), 
    P9 = Mode(data[,'P9']), 
    P10 = Mode(data[,'P10']), 
    P11 = Mode(data[,'P11']), 
    P12 = Mode(data[,'P12']), 
    P13 = Mode(data[,'P13']), 
    P14 = Mode(data[,'P14']), 
    P15 = Mode(data[,'P15']), 
    P16 = Mode(data[,'P16']), 
    P17 = Mode(data[,'P17']), 
    P18 = Mode(data[,'P18']), 
    P19 = Mode(data[,'P19']), 
    P20 = Mode(data[,'P20']), 
    P21 = Mode(data[,'P21']), 
    P22 = Mode(data[,'P22']), 
    P23 = Mode(data[,'P23']), 
    P24 = Mode(data[,'P24']), 
    P25 = Mode(data[,'P25']), 
    P26 = Mode(data[,'P26']), 
    P27 = Mode(data[,'P27']), 
    P28 = Mode(data[,'P28']), 
    P29 = Mode(data[,'P29']), 
    P30 = Mode(data[,'P30']), 
    P31 = Mode(data[,'P31']), 
    P32 = Mode(data[,'P32']), 
    P33 = Mode(data[,'P33']), 
    P34 = Mode(data[,'P34']), 
    P35 = Mode(data[,'P35']), 
    P36 = Mode(data[,'P36']), 
    P37 = Mode(data[,'P37']), 
    revenue = mean(yd)
  )
  colnames(tt) [3] = 'City Group'
  colnames(tt) [4] = 'Open Date'
  tt
  
  train.new = rbind(train.new , tt)
  
  train.new
}

buildData.analysis = function(train.raw , test.raw) {
  ## remove id 
  train = train.raw[ , -1] 
  test = test.raw[ , -1] 
  
  ## 2014 should be the target year ... so use open date to misure the number of years between open date and the target year 
  train$years.to.target = 2014 - year(as.Date( train.raw[,2] , "%m/%d/%Y"))
  test$years.to.target = 2014 - year(as.Date( test.raw[,2] , "%m/%d/%Y"))
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## extracting y 
  y = train[,41]
  train = train[,-41]
  
  list(train,y,test)
}

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

####### 
verbose = T

#######
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

train.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train.csv" , sep=''))) 

test.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                      "test.csv" , sep='')))

####### analysis 
l = buildData.analysis(train.raw , test.raw)
train = l[[1]]
y = l[[2]]
test = l[[3]]


train.new = addObservation (train.raw , years.to.taget = 11 , 
                          data = train[ train$years.to.target == 10 | train$years.to.target == 12 , ] , 
                          yd = y[ train$years.to.target == 10 | train$years.to.target == 12  ]
)

train.new = addObservation (train.new , years.to.taget = 13 , 
                            data = train[ train$years.to.target == 12 , ] , 
                            yd = y[ train$years.to.target == 12   ]
)

train.new = addObservation (train.new , years.to.taget = 19 , 
                            data = train[ train$years.to.target == 18  , ] , 
                            yd = y[ train$years.to.target == 18    ]
)


### write on disk 
write.csv(train.new,
          quote=FALSE, 
          file=paste(getBasePath("data"),"train_modified.csv",sep='') ,
          row.names=FALSE)
