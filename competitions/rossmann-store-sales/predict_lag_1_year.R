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

test$Date <- as.Date(test$Date)

##
report = data.frame(row = 1:nrow(test) , year_back = NA , avg_days = NA , gen_avg = 0) 

## test 
cat(">>> lag 1 year ... \n")
test$Sales <- NA 
for(i in 1:nrow(test)) {
  if (i == 1) cat(">>> starting [",i,"/",nrow(test),"] ...\n")
  if (i %% 5000 == 0) cat(">>> [",i,"/",nrow(test),"] ...\n")
  
  if(test[i,]$Open==0) {
    test[i,]$Sales <- 0 
    next
  }
  
  #cat(">>>> i ==",i,"\n")
  
  ## 
  train_i = train[train$Store==test[i,]$Store , ]
  
  date <- test[i,]$Date
  
  ############################################# time machine year 1 
  backYear = 1 
  #cat(">>>> backYear ==",backYear,"\n")
  year <- as.integer(format(date, "%Y")) - backYear
  month <- as.integer(format(date, "%m"))
  day <- as.integer(format(date, "%d"))
  
  date_1year_ago <- as.Date(paste(year,"-",month,"-",day,sep=''), format = '%Y-%m-%d')
  
  ## SalesLastYear1d
#   sales_1d <- train_i[train_i$Date == date_1year_ago ,]$Sales  
#   sales_1d <- sales_1d[sales_1d>0]
#   if (length(sales_1d)>0) {
#     test[i,]$Sales <- sales_1d
#     #cat(">>>> sales_1d \n")
#     report[report$row==i,]$year_back <- backYear
#     report[report$row==i,]$avg_days <- 1
#     next 
#   } 
  
  ## SalesLastYear3days
#   date_1year_ago_3dM <- date_1year_ago-1
#   date_1year_ago_3dP <- date_1year_ago+1
#   sales_3d <- train_i[train_i$Date >= date_1year_ago_3dM & train_i$Date <= date_1year_ago_3dP,]$Sales
#   sales_3d <- sales_3d[!is.na(sales_3d)]
#   sales_3d <- sales_3d[sales_3d>0]
#   if(length(sales_3d)>0) {
#     test[i,]$Sales <- mean(sales_3d)  
#     #cat(">>>> sales_3d \n")
#     report[report$row==i,]$year_back <- backYear
#     report[report$row==i,]$avg_days <- 3
#     next
#   } 
  
  ## SalesLastYear7days
  date_1year_ago_3dM <- date_1year_ago-3
  date_1year_ago_3dP <- date_1year_ago+3
  sales_7d <- train_i[train_i$Date >= date_1year_ago_3dM & train_i$Date <= date_1year_ago_3dP,]$Sales
  sales_7d <- sales_7d[!is.na(sales_7d)]
  sales_7d <- sales_7d[sales_7d>0]
  if(length(sales_7d)>0) {
    test[i,]$Sales <-  mean(sales_7d)   
    #cat(">>>> sales_7d \n")
    report[report$row==i,]$year_back <- backYear
    report[report$row==i,]$avg_days <- 7
    next 
  } 
  
  ############################################# time machine year 2
  backYear = 2 
  #cat(">>>> backYear ==",backYear,"\n")
  year <- as.integer(format(date, "%Y")) - backYear
  month <- as.integer(format(date, "%m"))
  day <- as.integer(format(date, "%d"))
  
  date_1year_ago <- as.Date(paste(year,"-",month,"-",day,sep=''), format = '%Y-%m-%d')
  
  ## SalesLastYear1d
#   sales_1d <- train_i[train_i$Date == date_1year_ago ,]$Sales  
#   sales_1d <- sales_1d[sales_1d>0]
#   if (length(sales_1d)>0) {
#     test[i,]$Sales <- sales_1d
#     #cat(">>>> sales_1d \n")
#     report[report$row==i,]$year_back <- backYear
#     report[report$row==i,]$avg_days <- 1
#     next 
#   } 
  
  ## SalesLastYear3days
#   date_1year_ago_3dM <- date_1year_ago-1
#   date_1year_ago_3dP <- date_1year_ago+1
#   sales_3d <- train_i[train_i$Date >= date_1year_ago_3dM & train_i$Date <= date_1year_ago_3dP,]$Sales
#   sales_3d <- sales_3d[!is.na(sales_3d)]
#   sales_3d <- sales_3d[sales_3d>0]
#   if(length(sales_3d)>0) {
#     test[i,]$Sales <- mean(sales_3d)  
#     #cat(">>>> sales_3d \n")
#     report[report$row==i,]$year_back <- backYear
#     report[report$row==i,]$avg_days <- 3
#     next
#   } 
  
  ## SalesLastYear7days
  date_1year_ago_3dM <- date_1year_ago-3
  date_1year_ago_3dP <- date_1year_ago+3
  sales_7d <- train_i[train_i$Date >= date_1year_ago_3dM & train_i$Date <= date_1year_ago_3dP,]$Sales
  sales_7d <- sales_7d[!is.na(sales_7d)]
  sales_7d <- sales_7d[sales_7d>0]
  if(length(sales_7d)>0) {
    test[i,]$Sales <-  mean(sales_7d)   
    #cat(">>>> sales_7d \n")
    report[report$row==i,]$year_back <- backYear
    report[report$row==i,]$avg_days <- 7
    next 
  } 
  
  ## finally 
  test[i,]$Sales <- mean(train_i[train_i$Open==1,]$Sales)
  report[report$row==i,]$gen_avg <- 1
}

##
cat(">>> Report \n")
describe(report)

## checks 
stopifnot(sum(is.na(test[,c("Id","Sales")]))==0)

## write on disk 
cat(">>> writing on disk ... \n")
write.csv(test[,c("Id","Sales")],
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"lag1Y.csv"),
          row.names=FALSE)
write.csv(report,
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"report_lag1Y.csv"),
          row.names=FALSE)


