library(data.table)
library(fastfurious)
library(Hmisc)
library(Matrix)

### FUNCS
ff.predNA = function(data,asIndex=TRUE) {
  stopifnot(identical(class(data),"data.frame"))
  feature.names <- colnames(data)
  predNA = unlist(lapply(1:length(feature.names) , function(i) {
    sum(is.na(data[,i]))>0 
  }))  
  if (asIndex) return(predNA)
  else return(feature.names[predNA])
}

ff.obsNA = function(data) {
  stopifnot(identical(class(data),"data.frame"))
  obsNAs =  which(is.na(data)) %% nrow(data) 
  return(obsNAs)
}
### CONFIG 

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
bench = as.data.frame( fread(paste(ff.getPath("data") , "bench.csv" , sep='') , stringsAsFactors = F))

##
sum(is.na(train)) #0
sum(is.na(test)) #11
sum(is.na(store)) #1799

dim(train[train$Sales == 0 , ]) #~100% dei casi (tranne 54 osservazioni) le vendite a 0 si spiegano con il fatto che il negozio e' chiuso

dim(test[test$Open == 0 , ]) # 5995 = 15% del test set ha Open == 0 --> vendite a 0 

## testing on bechmark ... ok 
# testE = merge(x = test , y = bench , by = "Id" , all = F)
# testE = testE[,c("Id","Open","Sales")]
# testE[is.na(testE$Open) , ]$Open <- 1 
# testE[testE$Open==0 , ]$Sales <- 0 
# 
# write.csv(testE[,c("Id","Sales")],
#           quote=FALSE, 
#           file=paste0(ff.getPath("elab"),"bench_impr.csv"),
#           row.names=FALSE)
###  >>> got 1.03 in LB .... 

describe(train[train$Open==1,])

## >> store 
predNAs = ff.predNA(data = store,asIndex=F)
for (pp in predNAs) {
  perc = sum(is.na(store[,pp]))/nrow(store)
  cat("[",pp,"]:",perc,"-- NAs",sum(is.na(store[,pp])),"\n")
}
# [ CompetitionDistance ]: 0.002690583 -- NAs 3 
# [ CompetitionOpenSinceMonth ]: 0.3174888 -- NAs 354 
# [ CompetitionOpenSinceYear ]: 0.3174888 -- NAs 354 
# [ Promo2SinceWeek ]: 0.4878924 -- NAs 544 
# [ Promo2SinceYear ]: 0.4878924 -- NAs 544 

## NB 
sum(is.na(store$CompetitionOpenSinceMonth) & !is.na(store$CompetitionOpenSinceYear)) # 0
sum(! is.na(store$CompetitionOpenSinceMonth) & is.na(store$CompetitionOpenSinceYear)) # 0

sum(is.na(store$Promo2SinceWeek) & !is.na(store$Promo2SinceYear)) # 0
sum(! is.na(store$Promo2SinceWeek) & is.na(store$Promo2SinceYear)) # 0

sum( store$Promo2==1 & is.na(store$Promo2SinceWeek)) # 0

## imputing 

# CompetitionDistance
store$CompetitionDistance[is.na(store$CompetitionDistance)] <- 0 

# CompetitionOpenSinceDate
store$CompetitionOpenSinceDate <- as.Date(NA) 
for (i in 1:nrow(store)) {
  if (is.na(store[i,]$CompetitionOpenSinceYear)) next 
  store[i,]$CompetitionOpenSinceDate <- as.Date(paste(store[i,]$CompetitionOpenSinceYear,"-",store[i,]$CompetitionOpenSinceMonth,"-15",sep=''), 
                                                 format = '%Y-%m-%d')
}
min(store$CompetitionOpenSinceDate,na.rm = T) # "1900-01-15" probably an error ... using this value to impute 
store$CompetitionOpenSinceDate[is.na(store$CompetitionOpenSinceDate)] <- as.Date("1900-01-15")

# Promo2SinceDate
store$Promo2SinceDate <- as.Date(NA) 
for (i in 1:nrow(store)) {
  if (is.na(store[i,]$Promo2SinceWeek)) next 
  store[i,]$Promo2SinceDate <- as.Date(paste(store[i,]$Promo2SinceYear,"-",store[i,]$Promo2SinceWeek,"-1",sep=''), 
                                                format = '%Y-%U-%u')
}


## feat. sel. 
store$CompetitionOpenSinceMonth <- NULL
store$CompetitionOpenSinceYear <- NULL

store$Promo2SinceWeek <- NULL
store$Promo2SinceYear <- NULL

## >> test 
predNAs = ff.predNA(data = test,asIndex=F)
for (pp in predNAs) {
  perc = sum(is.na(test[,pp]))/nrow(test)
  cat("[",pp,"]:",perc,"-- NAs",sum(is.na(test[,pp])),"\n")
}
# [ Open ]: 0.0002677181 -- NAs 11 

## imputing 
test$Open[is.na(test$Open)] <- 1 

## >> merge 
##    >> only Open==1 
Xtrain = merge(x = train[train$Open==1,] , y = store , by = "Store" , all = F)
Xtest = merge(x = test , y = store , by = "Store" , all = F)

## 
Xtrain$Date = as.Date(Xtrain$Date)
Xtest$Date = as.Date(Xtest$Date)

##
Xtrain$Customers <- NULL

## cut Xtest 
str(Xtest[Xtest$Open==0,])
Xtest_Open_0 = Xtest[Xtest$Open==0, c("Id","Open")]
Xtest_Open_0$Sales <- 0 
Xtest_Open_0$Open <- NULL

Xtest = Xtest[!Xtest$Id %in% Xtest_Open_0$Id , ]

Xtrain$Open <- NULL
Xtest$Open <- NULL

##
Xtrain$hasCompetition <- as.integer(Xtrain$Date>Xtrain$CompetitionOpenSinceDate) 
Xtest$hasCompetition <- as.integer(Xtest$Date>Xtest$CompetitionOpenSinceDate) 

Xtrain$CompetitionOpenSinceDate <- NULL
Xtest$CompetitionOpenSinceDate <- NULL


# > unique(store$PromoInterval)
# [1] ""                 "Jan,Apr,Jul,Oct"  "Feb,May,Aug,Nov"  "Mar,Jun,Sept,Dec"

MI = c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec")

## Xtrain
Xtrain$hasPromo2 <- 0
##for (i in 1:nrow(Xtrain)) {
a = lapply(1:nrow(Xtrain) , function(i) {
  if (i == 1) cat(">>> starting [",i,"/",nrow(Xtrain),"] ...\n")
  if (i %% 5000 == 0) cat(">>> [",i,"/",nrow(Xtrain),"] ...\n")
  if ( Xtrain[i,]$Promo2 ==1 ) {
    if (Xtrain[i,]$Promo2SinceDate < Xtrain[i,]$Date) {
      mVect = unlist(strsplit(x = Xtrain[i,]$PromoInterval,split = ","))
      mVectIdx = rep(NA,length(mVect))
      for (j in 1:length(mVect)) mVectIdx[j] = (which(mVect[j]==MI))
      month_ok = (as.integer(format(Xtrain[i,]$Date, "%m")) %in% mVectIdx)
      if (month_ok) {
        Xtrain[i,]$hasPromo2 <<- 1
      }
    }
  }
})

## Xtest
Xtest$hasPromo2 <- 0
##for (i in 1:nrow(Xtest)) {
a = lapply(1:nrow(Xtest) , function(i) {
  if (i == 1) cat(">>> starting [",i,"/",nrow(Xtest),"] ...\n")
  if (i %% 5000 == 0) cat(">>> [",i,"/",nrow(Xtest),"] ...\n")
  if ( Xtest[i,]$Promo2 ==1 ) {
    if (Xtest[i,]$Promo2SinceDate < Xtest[i,]$Date) {
      mVect = unlist(strsplit(x = Xtest[i,]$PromoInterval,split = ","))
      mVectIdx = rep(NA,length(mVect))
      for (j in 1:length(mVect)) mVectIdx[j] = (which(mVect[j]==MI))
      month_ok = (as.integer(format(Xtest[i,]$Date, "%m")) %in% mVectIdx)
      if (month_ok) {
        Xtest[i,]$hasPromo2 <<- 1
      }
    }
  }
})

## clean 
Xtrain$Promo2 <- NULL
Xtrain$PromoInterval <- NULL
Xtrain$Promo2SinceDate <- NULL

Xtest$Promo2 <- NULL
Xtest$PromoInterval <- NULL
Xtest$Promo2SinceDate <- NULL

## categ 
feature.names = colnames(Xtrain)
for (f in feature.names) {
  if (class(Xtrain[,f])=="character") {
    cat(">>> ",f," is character \n")
    levels <- unique(c(Xtrain[,f],Xtest[,f]))
    Xtrain[,f] <- as.integer(factor(Xtrain[,f], levels=levels))
    Xtest[,f] <- as.integer(factor(Xtest[,f], levels=levels))
  }
}


## adjiust 
Xtrain$CompetitionDistance <- Xtrain$hasCompetition * Xtrain$CompetitionDistance
Xtest$CompetitionDistance <- Xtest$hasCompetition * Xtest$CompetitionDistance

## encode date 
l = ff.extractDateFeature(data.train = Xtrain$Date , data.test = Xtest$Date)
Xtrain$Date <- l$traindata
Xtest$Date <- l$testdata

## write on disk 
cat(">>> writing on disk ... \n")
write.csv(Xtrain,
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"Xtrain.csv"),
          row.names=FALSE)
write.csv(Xtest,
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"Xtest.csv"),
          row.names=FALSE)

write.csv(Xtest_Open_0,
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"Xtest_Open_0.csv"),
          row.names=FALSE)







