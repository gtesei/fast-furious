library(data.table)
library(fastfurious)
library(plyr)
library(Hmisc)
library(Matrix)

### FUNCS

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/rossmann-store-sales')
ff.bindPath(type = 'code' , sub_path = 'competitions/rossmann-store-sales')
ff.bindPath(type = 'elab' , sub_path = 'dataset/rossmann-store-sales/elab' ,  createDir = T) 

## DATA 
train = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
test = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))
store = as.data.frame( fread(paste(ff.getPath("data") , "store.csv" , sep='') , stringsAsFactors = F))
sample_submission = as.data.frame( fread(paste(ff.getPath("data") , "sample_submission.csv" , sep='') , stringsAsFactors = F))

## PROCS 
sum(is.na(train)) #0
sum(is.na(test)) #11
sum(is.na(store)) #1799

sum(is.na(test$Open)) ## 11 - tutti dello stesso store <- imputare con 1 

## test NAs 
test$Open[is.na(test$Open)] <- 1 

# store | CompetitionOpenSinceDate
store$CompetitionOpenSinceDate <- as.Date(NA) 
for (i in 1:nrow(store)) {
  if (is.na(store[i,]$CompetitionOpenSinceMonth)) next 
  store[i,]$CompetitionOpenSinceDate <- as.Date(paste(store[i,]$CompetitionOpenSinceYear,"-",store[i,]$CompetitionOpenSinceMonth,"-1",sep=''), format = '%Y-%m-%d')
}

# store | Promo2SinceDate
store$Promo2SinceDate <- as.Date(NA) 
for (i in 1:nrow(store)) {
  if (is.na(store[i,]$Promo2SinceWeek)) next 
  store[i,]$Promo2SinceDate <- as.Date(paste(store[i,]$Promo2SinceYear,"-",store[i,]$Promo2SinceWeek,"-1",sep=''), format = '%Y-%U-%u')
}

### train 
trst <- merge(x = train , y = store , by = "Store" , all = F)
tsst <- merge(x = test , y = store , by = "Store" , all = F)

stopifnot(nrow(trst) == nrow(train))
stopifnot(nrow(tsst) == nrow(test))

## 
cat(">>> focusing on open days ... \n")
trst <- trst[trst$Open == 1 , ]

## 
cat(">>> focusing on test stores  ... \n")
stores_test = sort(intersect(x = unique(trst$Store) , y = unique(tsst$Store)))
trst = trst[trst$Store %in% stores_test , ]

### Date  
trst$Date <- as.Date(trst$Date)
tsst$Date <- as.Date(tsst$Date)

trst$month <- as.integer(format(trst$Date, "%m"))
trst$year <- as.integer(format(trst$Date, "%y"))
trst$day <- as.integer(format(trst$Date, "%d"))

tsst$month <- as.integer(format(tsst$Date, "%m"))
tsst$year <- as.integer(format(tsst$Date, "%y"))
tsst$day <- as.integer(format(tsst$Date, "%d"))

l = ff.extractDateFeature(trst$Date,tsst$Date)
trst$dateNum <- l$traindata
tsst$dateNum <- l$testdata

## StateHoliday / SchoolHoliday 
levels <- unique(c(trst$StateHoliday,tsst$StateHoliday))
trst$StateHoliday <- as.integer(factor(trst$StateHoliday, levels=levels))
tsst$StateHoliday <- as.integer(factor(tsst$StateHoliday, levels=levels))

trst$SchoolHoliday <- as.numeric(trst$SchoolHoliday)
tsst$SchoolHoliday <- as.numeric(tsst$SchoolHoliday)

## StoreType / Assortment 
levels <- unique(c(trst$StoreType,tsst$StoreType))
trst$StoreType <- as.integer(factor(trst$StoreType, levels=levels))
tsst$StoreType <- as.integer(factor(tsst$StoreType, levels=levels))

levels <- unique(c(trst$Assortment,tsst$Assortment))
trst$Assortment <- as.integer(factor(trst$Assortment, levels=levels))
tsst$Assortment <- as.integer(factor(tsst$Assortment, levels=levels))

### CompetitionDistancePred 
## train
cat(">>> CompetitionDistancePred train set ...\n")
trst$CompetitionDistancePred <- NA 
a <- lapply(1:nrow(trst),function(i){
  if (i == 1) cat(">>> starting [",i,"/",nrow(trst),"] ...\n")
  if (i %% 5000 == 0) cat(">>> [",i,"/",nrow(trst),"] ...\n")
  
  train_i = trst[i,]
  
  if (is.na(train_i$CompetitionDistance)) {
    trst[i,]$CompetitionDistancePred <<- -1
  } else if (is.na(train_i$CompetitionOpenSinceDate)) {
    trst[i,]$CompetitionDistancePred <<- train_i$CompetitionDistance
  } else if (train_i$Date < train_i$CompetitionOpenSinceDate) {
    trst[i,]$CompetitionDistancePred <<- -1
  } else {
    trst[i,]$CompetitionDistancePred <<- train_i$CompetitionDistance
  }
}) 

# cheks 
stopifnot(  sum( is.na(trst$CompetitionDistancePred))==0)
stopifnot(  sum( is.na(trst$CompetitionDistance) & (trst$CompetitionDistancePred != -1)) == 0  ) 
stopifnot(  sum( is.na(trst$CompetitionOpenSinceDate) & !is.na(trst$CompetitionDistance) & (trst$CompetitionDistancePred == -1)) == 0  ) 
stopifnot(  sum( !is.na(trst$CompetitionOpenSinceDate) & !is.na(trst$CompetitionDistance) & (trst$Date < trst$CompetitionOpenSinceDate) & (trst$CompetitionDistancePred != -1)) == 0  )
stopifnot(  sum( !is.na(trst$CompetitionOpenSinceDate) & !is.na(trst$CompetitionDistance) & (trst$Date >= trst$CompetitionOpenSinceDate) & (trst$CompetitionDistancePred == -1)) == 0  ) 

## test 
cat(">>> CompetitionDistancePred test set ...\n")
tsst$CompetitionDistancePred <- NA 
a <- lapply(1:nrow(tsst),function(i){
  if (i == 1) cat(">>> starting [",i,"/",nrow(tsst),"] ...\n")
  if (i %% 5000 == 0) cat(">>> [",i,"/",nrow(tsst),"] ...\n")
  
  if (is.na(tsst[i,]$CompetitionDistance)) {
    tsst[i,]$CompetitionDistancePred <<- -1
  } else if (is.na(tsst[i,]$CompetitionOpenSinceDate)) {
    tsst[i,]$CompetitionDistancePred <<- tsst[i,]$CompetitionDistance
  } else if (tsst[i,]$Date < tsst[i,]$CompetitionOpenSinceDate) {
    tsst[i,]$CompetitionDistancePred <<- -1
  } else {
    tsst[i,]$CompetitionDistancePred <<- tsst[i,]$CompetitionDistance
  }
}) 

# cheks 
stopifnot(  sum( is.na(tsst$CompetitionDistancePred))==0)
stopifnot(  sum( is.na(tsst$CompetitionDistance) & (tsst$CompetitionDistancePred != -1)) == 0  ) 
stopifnot(  sum( is.na(tsst$CompetitionOpenSinceDate) & !is.na(tsst$CompetitionDistance) & (tsst$CompetitionDistancePred == -1)) == 0  ) 
stopifnot(  sum( !is.na(tsst$CompetitionOpenSinceDate) & !is.na(tsst$CompetitionDistance) & (tsst$Date < tsst$CompetitionOpenSinceDate) & (tsst$CompetitionDistancePred != -1)) == 0  )
stopifnot(  sum( !is.na(tsst$CompetitionOpenSinceDate) & !is.na(tsst$CompetitionDistance) & (tsst$Date >= tsst$CompetitionOpenSinceDate) & (tsst$CompetitionDistancePred == -1)) == 0  ) 


### Promo2Pred / Promo2SinceMonths
MI = c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec")
## train 
cat(">>> Promo2Pred train set ...\n")
trst$Promo2Pred <- 0 
trst$Promo2SinceMonths <- 0 
a <- lapply(1:nrow(trst),function(i) {
  if (i == 1) cat(">>> starting [",i,"/",nrow(trst),"] ...\n")
  if (i %% 5000 == 0) cat(">>> [",i,"/",nrow(trst),"] ...\n")
  
  train_i = trst[i,]
  
  if (train_i$Promo2 == 1) {
    if (train_i$Promo2SinceDate < train_i$Date) {
      mVect = unlist(strsplit(x = train_i$PromoInterval,split = ","))
      mVectIdx = rep(NA,length(mVect))
      for (j in 1:length(mVect)) mVectIdx[j] = which(mVect[j]==MI)
      month_ok = as.integer(format(train_i$Date, "%m")) %in% mVectIdx
      if (month_ok) {
        trst[i,]$Promo2Pred <<- 1
      }
      
      ## Promo2SinceMonths
      trst[i,]$Promo2SinceMonths <<- ( as.numeric(train_i$Date - train_i$Promo2SinceDate) / 30)
    }
  }
}) 

# checks 
stopifnot(sum(trst$Promo2==0 & trst$Promo2Pred!=0)==0)
stopifnot(sum(trst$Promo2==1 & trst$Date < trst$Promo2SinceDate & trst$Promo2Pred==1)==0)
stopifnot(sum(trst$Promo2==1 & trst$Date > trst$Promo2SinceDate & trst$Promo2SinceMonths==0)==0)

## test 
cat(">>> Promo2Pred test set ...\n")
tsst$Promo2Pred <- 0 
tsst$Promo2SinceMonths <- 0 
a <- lapply(1:nrow(tsst),function(i){
  if (i == 1) cat(">>> starting [",i,"/",nrow(tsst),"] ...\n")
  if (i %% 5000 == 0) cat(">>> [",i,"/",nrow(tsst),"] ...\n")
  
  if (tsst[i,]$Promo2 == 1) {
    if (tsst[i,]$Promo2SinceDate < tsst[i,]$Date) {
      mVect = unlist(strsplit(x = tsst[i,]$PromoInterval,split = ","))
      mVectIdx = rep(NA,length(mVect))
      for (j in 1:length(mVect)) mVectIdx[j] = which(mVect[j]==MI)
      month_ok = as.integer(format(tsst[i,]$Date, "%m")) %in% mVectIdx
      if (month_ok) {
        tsst[i,]$Promo2Pred <<- 1
      }
      
      ## Promo2SinceMonths
      tsst[i,]$Promo2SinceMonths <<- ( as.numeric(tsst[i,]$Date - tsst[i,]$Promo2SinceDate) / 30)
    }
  }
}) 

# checks 
stopifnot(sum(tsst$Promo2==0 & tsst$Promo2Pred!=0)==0)
stopifnot(sum(tsst$Promo2==1 & tsst$Date < tsst$Promo2SinceDate & tsst$Promo2Pred==1)==0)
stopifnot(sum(tsst$Promo2==1 & tsst$Date > tsst$Promo2SinceDate & tsst$Promo2SinceMonths==0)==0)

### resampling
tr_meta = ddply(trst , .(Store) , function(x) c(min_tr_date=min(x$Date),
                                        max_tr_date=max(x$Date), 
                                        tr_obs = nrow(x)
                                        ) )
tr_meta$tr_obs = as.numeric(tr_meta$tr_obs)
tr_meta$tr_days = as.numeric(tr_meta$max_tr_date-tr_meta$min_tr_date)
tr_meta$tr_months = tr_meta$tr_days / 30 

#
ts_meta = ddply(tsst , .(Store) , function(x) c(min_ts_date=min(x$Date),
                                                max_ts_date=max(x$Date), 
                                                ts_obs = nrow(x)
) )
ts_meta$ts_obs = as.numeric(ts_meta$ts_obs)
ts_meta$ts_days = as.numeric(ts_meta$max_ts_date-ts_meta$min_ts_date)
ts_meta$ts_months = ts_meta$ts_days / 30 

#
mmeta = merge(x = tr_meta , y = ts_meta , by = "Store" , all = F)
print(describe(mmeta))

# obs_ratio: # obs in test set / # obs in train set 
obs_ratio = 48 / 903.3 # ~ 5.3% 

### ---> let's do 10% cross-validation ratio --> 941 days * 0.1 =~ 94 days 
### -->  train period = "2013-01-01/02"  , "2015-04-28"
### -->  xval  period = "2015-04-29"     , "2015-07-31" 

####>>>>>>>>>> let's generate datesets  
trst_gen = trst

pred_test = c("Id","Store", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", 
              "Assortment", "month" , "year", "day", "dateNum", "CompetitionDistancePred" , "Promo2Pred" , "Promo2SinceMonths")

pred_train = c(pred_test,"Sales")


## Id train 
trst_gen$Id <- 1:nrow(trst_gen)
  
## resampling 
trst_gen_tr = trst_gen[trst_gen$Date < as.Date("2015-04-29") , ]
trst_gen_xval = trst_gen[trst_gen$Date >= as.Date("2015-04-29") , ]


## checks 
stopifnot(sum(is.na(trst_gen_tr[,pred_train]))==0)
stopifnot(sum(is.na(trst_gen_xval[,pred_train]))==0)
stopifnot(sum(is.na(tsst[,pred_test]))==0)

## write on disk 
cat(">>> writing on disk ... \n")
write.csv(trst_gen_tr[,pred_train],
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"Xtrain3_tr.csv"),
          row.names=FALSE)
write.csv(trst_gen_xval[,pred_train],
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"Xtrain3_xval.csv"),
          row.names=FALSE)
write.csv(tsst[,pred_test],
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"Xtest3.csv"),
          row.names=FALSE)

###########################################################################>>>>>>>>> 4

####### SalesLastYears3Days
### ---> let's do 10% cross-validation ratio --> 941 days * 0.1 =~ 94 days 
### -->  train period = "2013-01-01/02"  , "2015-04-28"
### -->  xval  period = "2015-04-29"     , "2015-07-31" 
### -->  NAs(=-1)     = "2013-01-01/02"  , "2014-01-02"

### train 
cat(">>> SalesLastYears3Days / SalesLastYears7Days -- train set ... \n")
date_min <- as.Date("2014-01-02")
trst$SalesLastYears3Days <- -1 
trst$SalesLastYears7Days <- -1 
a <- lapply(1:nrow(trst),function(i) {
  ##
  if (i == 1) cat(">>> starting [",i,"/",nrow(trst),"] ...\n")
  if (i %% 5000 == 0) cat(">>> [",i,"/",nrow(trst),"] ...\n")
  
  date <- trst[i,]$Date
  
  if (date > date_min) { 
    ##
    train_i = trst[trst$Store==trst[i,]$Store , ]
    
    ## date_1year_ago
    year <- as.integer(format(date, "%Y")) - 1
    month <- as.integer(format(date, "%m"))
    day <- as.integer(format(date, "%d"))
    date_1year_ago <- as.Date(paste(year,"-",month,"-",day,sep=''), format = '%Y-%m-%d')
    
    ## SalesLastYears3Days
    date_1year_ago_3dM <- date_1year_ago-1
    date_1year_ago_3dP <- date_1year_ago+1
    sales_3d <- train_i[train_i$Date >= date_1year_ago_3dM & train_i$Date <= date_1year_ago_3dP,]$Sales
    sales_3d <- sales_3d[!is.na(sales_3d)]
    sales_3d <- sales_3d[sales_3d>0]
    if(length(sales_3d)>0) {
      trst[i,]$SalesLastYears3Days <<- mean(sales_3d)
    } 
    
    ## SalesLastYears7Days
    date_1year_ago_7dM <- date_1year_ago-3
    date_1year_ago_7dP <- date_1year_ago+3
    sales_7d <- train_i[train_i$Date >= date_1year_ago_7dM & train_i$Date <= date_1year_ago_7dP,]$Sales
    sales_7d <- sales_7d[!is.na(sales_7d)]
    sales_7d <- sales_7d[sales_7d>0]
    if(length(sales_7d)>0) {
      trst[i,]$SalesLastYears7Days <<- mean(sales_7d)
    }
  }
})

### test 
cat(">>> SalesLastYears3Days / SalesLastYears7Days -- test set ... \n")
tsst$SalesLastYears3Days <- -1 
tsst$SalesLastYears7Days <- -1 
a <- lapply(1:nrow(tsst),function(i) {
  ##
  if (i == 1) cat(">>> starting [",i,"/",nrow(tsst),"] ...\n")
  if (i %% 5000 == 0) cat(">>> [",i,"/",nrow(tsst),"] ...\n")
  
  ##
  date <- tsst[i,]$Date
  
  ##
  train_i = trst[trst$Store==tsst[i,]$Store , ]
  
  ## date_1year_ago
  year <- as.integer(format(date, "%Y")) - 1
  month <- as.integer(format(date, "%m"))
  day <- as.integer(format(date, "%d"))
  date_1year_ago <- as.Date(paste(year,"-",month,"-",day,sep=''), format = '%Y-%m-%d')
  
  ## SalesLastYears3Days
  date_1year_ago_3dM <- date_1year_ago-1
  date_1year_ago_3dP <- date_1year_ago+1
  sales_3d <- train_i[train_i$Date >= date_1year_ago_3dM & train_i$Date <= date_1year_ago_3dP,]$Sales
  sales_3d <- sales_3d[!is.na(sales_3d)]
  sales_3d <- sales_3d[sales_3d>0]
  if(length(sales_3d)>0) {
    tsst[i,]$SalesLastYears3Days <<- mean(sales_3d)
  } 
  
  ## SalesLastYears7Days
  date_1year_ago_7dM <- date_1year_ago-3
  date_1year_ago_7dP <- date_1year_ago+3
  sales_7d <- train_i[train_i$Date >= date_1year_ago_7dM & train_i$Date <= date_1year_ago_7dM,]$Sales
  sales_7d <- sales_7d[!is.na(sales_7d)]
  sales_7d <- sales_7d[sales_7d>0]
  if(length(sales_7d)>0) {
    tsst[i,]$SalesLastYears7Days <<- mean(sales_7d)
  } 
})


####>>>>>>>>>> let's generate datesets  
trst_gen = trst

pred_test = c("Id","Store", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", 
              "Assortment", "month" , "year", "day", "dateNum", "CompetitionDistancePred" , "Promo2Pred" , "Promo2SinceMonths", 
              "SalesLastYears3Days", "SalesLastYears7Days")

pred_train = c(pred_test,"Sales")

## Id train 
trst_gen$Id <- 1:nrow(trst_gen)

## resampling 
trst_gen_tr = trst_gen[trst_gen$Date < as.Date("2015-04-29") , ]
trst_gen_xval = trst_gen[trst_gen$Date >= as.Date("2015-04-29") , ]


## checks 
stopifnot(sum(is.na(trst_gen_tr[,pred_train]))==0)
stopifnot(sum(is.na(trst_gen_xval[,pred_train]))==0)
stopifnot(sum(is.na(tsst[,pred_test]))==0)

## write on disk 
cat(">>> writing on disk ... \n")
write.csv(trst_gen_tr[,pred_train],
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"Xtrain4_tr.csv"),
          row.names=FALSE)
write.csv(trst_gen_xval[,pred_train],
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"Xtrain4_xval.csv"),
          row.names=FALSE)
write.csv(tsst[,pred_test],
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"Xtest4.csv"),
          row.names=FALSE)


