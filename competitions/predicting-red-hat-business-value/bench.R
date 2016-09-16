library(data.table)
library(FeatureHashing)
library(xgboost)
library(dplyr)
library(Matrix)

cat(Sys.time())
cat("Reading data\n")   

prefix = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/predicting-red-hat-business-value/'
train <- fread(paste0(prefix,"act_train.csv"), header=TRUE)
test <- fread(paste0(prefix,"act_test.csv"), header=TRUE)
people <- fread(paste0(prefix,"people.csv"), header=TRUE)
sample_submission <- fread(paste0(prefix,"sample_submission.csv"), header=TRUE)

cat(Sys.time())
cat("Processing data\n")

people$char_1<-NULL #unnecessary duplicate to char_2
names(people)[2:length(names(people))]=paste0('people_',names(people)[2:length(names(people))])

p_logi <- names(people)[which(sapply(people, is.logical))]
for (col in p_logi) set(people, j = col, value = as.numeric(people[[col]]))

#reducing group_1 dimension
people$people_group_1[people$people_group_1 %in% names(which(table(people$people_group_1)==1))]='group unique'


#reducing char_10 dimension
#unique.char_10=
#  rbind(
#    select(train,people_id,char_10),
#    select(test,people_id,char_10)) %>% group_by(char_10) %>% 
#  summarize(n=n_distinct(people_id)) %>% 
#  filter(n==1) %>% 
#  select(char_10) %>%
#  as.matrix() %>% 
#  as.vector()

#train$char_10[train$char_10 %in% unique.char_10]='type unique'
#test$char_10[test$char_10 %in% unique.char_10]='type unique'

d1 <- merge(train, people, by = "people_id", all.x = T)
d2 <- merge(test, people, by = "people_id", all.x = T)
Y <- d1$outcome
d1$outcome <- NULL

row.train=nrow(train)
gc(verbose=FALSE)

D=rbind(d1,d2)
D$i=1:dim(D)[1]


###uncomment this for CV run
#set.seed(120)
#unique_p <- unique(d1$people_id)
#valid_p  <- unique_p[sample(1:length(unique_p), 40000)]
#valid <- which(d1$people_id %in% valid_p)
#model <- (1:length(d1$people_id))[-valid]

test_activity_id=test$activity_id
rm(train,test,d1,d2);gc(verbose=FALSE)


char.cols=c('activity_category','people_group_1',
            'char_1','char_2','char_3','char_4','char_5','char_6','char_7','char_8','char_9','char_10',
            'people_char_2','people_char_3','people_char_4','people_char_5','people_char_6','people_char_7','people_char_8','people_char_9')
for (f in char.cols) {
  if (class(D[[f]])=="character") {
    levels <- unique(c(D[[f]]))
    D[[f]] <- as.numeric(factor(D[[f]], levels=levels))
  }
}


cat(Sys.time())
cat("Making sparse data (1)\n")

D.sparse=
  cBind(sparseMatrix(D$i,D$activity_category),
        sparseMatrix(D$i,D$people_group_1),
        sparseMatrix(D$i,D$char_1),
        sparseMatrix(D$i,D$char_2),
        sparseMatrix(D$i,D$char_3),
        sparseMatrix(D$i,D$char_4),
        sparseMatrix(D$i,D$char_5),
        sparseMatrix(D$i,D$char_6),
        sparseMatrix(D$i,D$char_7),
        sparseMatrix(D$i,D$char_8),
        sparseMatrix(D$i,D$char_9),
        sparseMatrix(D$i,D$people_char_2),
        sparseMatrix(D$i,D$people_char_3),
        sparseMatrix(D$i,D$people_char_4),
        sparseMatrix(D$i,D$people_char_5),
        sparseMatrix(D$i,D$people_char_6),
        sparseMatrix(D$i,D$people_char_7),
        sparseMatrix(D$i,D$people_char_8),
        sparseMatrix(D$i,D$people_char_9)
  )


cat(Sys.time())
cat("Making sparse data (2)\n")

D.sparse=
  cBind(D.sparse,
        D$people_char_10,
        D$people_char_11,
        D$people_char_12,
        D$people_char_13,
        D$people_char_14,
        D$people_char_15,
        D$people_char_16,
        D$people_char_17,
        D$people_char_18,
        D$people_char_19,
        D$people_char_20,
        D$people_char_21,
        D$people_char_22,
        D$people_char_23,
        D$people_char_24,
        D$people_char_25,
        D$people_char_26,
        D$people_char_27,
        D$people_char_28,
        D$people_char_29,
        D$people_char_30,
        D$people_char_31,
        D$people_char_32,
        D$people_char_33,
        D$people_char_34,
        D$people_char_35,
        D$people_char_36,
        D$people_char_37,
        D$people_char_38)


cat(Sys.time())
cat("Unmerging train/test sparse data\n")

train.sparse=D.sparse[1:row.train,]
test.sparse=D.sparse[(row.train+1):nrow(D.sparse),]



# Hash train to sparse dmatrix X_train + LibSVM/SVMLight format



cat(Sys.time())
cat("Making data for SVMLight format\n")

# LibSVM format if you use Python / etc. ALWAYS USEFUL


# TOO LONG

cat("Creating SVMLight format\n")
dtrain  <- xgb.DMatrix(train.sparse, label = Y)
gc(verbose=FALSE)
cat("Exporting SVMLight format\n")
xgb.DMatrix.save(dtrain, paste0(prefix,"dtrain.data"))
gc(verbose=FALSE)
#rm(dtrain) #avoid getting through memory limits
#gc(verbose=FALSE)
cat("Zipping SVMLight\n")
zip(paste0(prefix,"dtrain.data.zip"), paste0(prefix,"dtrain.data"), flags = "-m9X", extras = "", zip = Sys.getenv("R_ZIPCMD", "zip"))
#file.remove("dtrain.data")

cat(Sys.time())
cat("File size of train in SVMLight: ", file.size(paste0(prefix,"dtrain.data.zip")), "\n", sep = "")

cat("Creating SVMLight format\n")
dtest  <- xgb.DMatrix(test.sparse)
gc(verbose=FALSE)
cat("Exporting SVMLight format\n")
xgb.DMatrix.save(dtest, paste0(prefix,"dtest.data"))
gc(verbose=FALSE)
cat("Zipping SVMLight\n")
zip(paste0(prefix,"dtest.data.zip"), paste0(prefix,"dtest.data"), flags = "-m9X", extras = "", zip = Sys.getenv("R_ZIPCMD", "zip"))
#file.remove("dtest.data")
cat(Sys.time())
cat("File size of test in SVMLight: ", file.size(paste0(prefix,"dtest.data.zip")), "\n", sep = "")


#cat("Re-creating SVMLight format\n")
#dtrain  <- xgb.DMatrix(train.sparse, label = Y) #recreate train sparse to run under the memory limit of 8589934592 bytes
gc(verbose=FALSE)


param <- list(objective = "binary:logistic", 
              eval_metric = "auc",
              booster = "gbtree", 
              eta = 0.05,
              subsample = 0.86,
              colsample_bytree = 0.92,
              colsample_bylevel = 0.9,
              min_child_weight = 0,
              max_depth = 11)

###uncomment this for CV run
#
#dmodel  <- xgb.DMatrix(train.sparse[model, ], label = Y[model])
#dvalid  <- xgb.DMatrix(train.sparse[valid, ], label = Y[valid])
#
#set.seed(120)
#m1 <- xgb.train(data = dmodel
#                , param
#                , nrounds = 500
#                , watchlist = list(valid = dvalid, model = dmodel)
#                , early.stop.round = 20
#                , nthread=11
#                , print_every_n = 10)

#[300]	valid-auc:0.979167	model-auc:0.990326



cat(Sys.time())
cat("XGBoost\n")

set.seed(120)
m2 <- xgb.train(data = dtrain, 
                param, nrounds = 450,
                watchlist = list(train = dtrain),
                print_every_n = 101)


cat(Sys.time())
cat("Predicting on test data\n")

# Predict
out <- predict(m2, dtest)
sub <- data.frame(activity_id = test_activity_id, outcome = out)
write.csv(sub, file = paste0(prefix,"model_sub.csv"), row.names = F)

#0.98035

cat("Cleaning up...\n")
remove(list = ls())
gc(verbose=FALSE)



## Leak from Loiso (0.987)

cat(Sys.time())
cat("Doing Loiso's magic leak\n")

cat("Working on people\n")
# load and transform people data ------------------------------------------
prefix = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/predicting-red-hat-business-value/'
ppl <- fread(paste0(prefix,"people.csv"))

### Recode logic to numeric
p_logi <- names(ppl)[which(sapply(ppl, is.logical))]

for (col in p_logi) {
  set(ppl, j = col, value = as.numeric(ppl[[col]]))
}
rm(p_logi)

### transform date
ppl[,date := as.Date(as.character(date), format = "%Y-%m-%d")]

# load activities ---------------------------------------------------------

cat("Working on data\n")
# read and combine
activs <- fread(paste0(prefix,"act_train.csv"))
TestActivs <- fread(paste0(prefix,"act_test.csv"))
TestActivs$outcome <- NA
activs <- rbind(activs,TestActivs)
rm(TestActivs)

cat("Merging\n")
# Extract only required variables
activs <- activs[, c("people_id","outcome","activity_id","date"), with = F]

# Merge people data into actvitities
d1 <- merge(activs, ppl, by = "people_id", all.x = T)

# Remember, remember the 5th of November and which is test
testset <- which(ppl$people_id %in% d1$people_id[is.na(d1$outcome)])
d1[, activdate := as.Date(as.character(date.x), format = "%Y-%m-%d")]

rm(activs)

# prepare grid for prediction ---------------------------------------------

cat("Creating interaction\n")
# Create all group_1/day grid
minactivdate <- min(d1$activdate)
maxactivdate <- max(d1$activdate)
alldays <- seq(minactivdate, maxactivdate, "day")
allCompaniesAndDays <- data.table(
  expand.grid(unique(
    d1$group_1[!d1$people_id %in% ppl$people_id[testset]]), alldays
  )
)


## Nicer names
colnames(allCompaniesAndDays) <- c("group_1","date.p")

## sort it
setkey(allCompaniesAndDays,"group_1","date.p")

## What are values on days where we have data?
meanbycomdate <- d1[
  !d1$people_id %in% ppl$people_id[testset],
  mean(outcome),
  by = c("group_1","activdate")
  ]

## Add them to full data grid
allCompaniesAndDays <- merge(
  allCompaniesAndDays,
  meanbycomdate,
  by.x = c("group_1","date.p"), by.y = c("group_1","activdate"),
  all.x = T
)


# design function to interpolate unknown values ---------------------------

interpolateFun <- function(x){
  
  # Find all non-NA indexes, combine them with outside borders
  borders <- c(1, which(!is.na(x)), length(x) + 1)
  # establish forward and backward - looking indexes
  forward_border <- borders[2:length(borders)]
  backward_border <- borders[1:(length(borders) - 1)]
  
  # prepare vectors for filling
  forward_border_x <- x[forward_border]
  forward_border_x[length(forward_border_x)] <- abs(
    forward_border_x[length(forward_border_x) - 1] - 0.2
  ) 
  backward_border_x <- x[backward_border]
  backward_border_x[1] <- abs(forward_border_x[1] - 0.1)
  
  # generate fill vectors
  forward_x_fill <- rep(forward_border_x, forward_border - backward_border)
  backward_x_fill <- rep(backward_border_x, forward_border - backward_border)
  forward_x_fill_2 <- rep(forward_border, forward_border - backward_border) - 
    1:length(forward_x_fill)
  backward_x_fill_2 <- 1:length(forward_x_fill) -
    rep(backward_border, forward_border - backward_border)
  
  #linear interpolation
  vec <- (forward_x_fill + backward_x_fill)/2
  
  x[is.na(x)] <- vec[is.na(x)]
  return(x)
}


# apply and submit --------------------------------------------------------

cat("Applying interaction interpolation\n")
allCompaniesAndDays[, filled := interpolateFun(V1), by = "group_1"]

d1 <- merge(
  d1,
  allCompaniesAndDays,
  all.x = T,all.y = F,
  by.x = c("group_1","activdate"),
  by.y = c("group_1","date.p")
)


cat("Predicting leak\n")
## Create prediction file and write
testsetdt <- d1[
  d1$people_id %in% ppl$people_id[testset],
  c("activity_id","filled"), with = F
  ]
write.csv(testsetdt,paste0(prefix,"Submission.csv"), row.names = FALSE)

cat("Cleaning up...\n")
remove(list = ls())
gc(verbose=FALSE)



## Combo

# Not elegant but fast enough for our taste, could go sqldf one-liner >_>


cat(Sys.time())
cat("Combining two submissions on targeted observations (NAs)\n")

prefix = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/predicting-red-hat-business-value/'
submit1 <- fread(paste0(prefix,"Submission.csv"))
submit2 <- fread(paste0(prefix,"model_sub.csv"))
submit3 <- merge(submit1[is.na(submit1$filled), ], submit2, by = "activity_id", all.x = T)
submit4 <- merge(submit1, submit3, by = "activity_id", all.x = T)
submit4$filled.x[is.na(submit4$filled.x)] <- submit4$outcome[is.na(submit4$filled.x)]
submit5 <- data.frame(activity_id = submit4$activity_id, outcome = submit4$filled.x, stringsAsFactors = FALSE)
write.csv(submit5, file=paste0(prefix,"bech___mod3108Kaggle_01.csv"), row.names=FALSE)