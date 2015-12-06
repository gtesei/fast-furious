library(data.table)
library(fastfurious)
library(Hmisc)
library(Matrix)
library(plyr)

### FUNCS

### CONFIG 

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/deloitte-western-australia-rental-prices/data')
ff.bindPath(type = 'code' , sub_path = 'competitions/deloitte-western-australia-rental-prices')
ff.bindPath(type = 'elab' , sub_path = 'dataset/deloitte-western-australia-rental-prices/elab' ,  createDir = T) 

####
source(paste0(ff.getPath("code"),"fastImpute.R"))

## DATA 
train = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
test = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))

land_admin_areas = as.data.frame( fread(paste(ff.getPath("data") , "land_admin_areas.csv" , sep='') , stringsAsFactors = F))
land_pins = as.data.frame( fread(paste(ff.getPath("data") , "land_pins.csv" , sep='') , stringsAsFactors = F))
land = as.data.frame( fread(paste(ff.getPath("data") , "land.csv" , sep='') , stringsAsFactors = F))
land_restrictions = as.data.frame( fread(paste(ff.getPath("data") , "land_restrictions.csv" , sep='') , stringsAsFactors = F))
land_urban = as.data.frame( fread(paste(ff.getPath("data") , "land_urban.csv" , sep='') , stringsAsFactors = F))
land_valuation_key = as.data.frame( fread(paste(ff.getPath("data") , "land_valuation_key.csv" , sep='') , stringsAsFactors = F))
land_zonings = as.data.frame( fread(paste(ff.getPath("data") , "land_zonings.csv" , sep='') , stringsAsFactors = F))

valuation_entities = as.data.frame( fread(paste(ff.getPath("data") , "valuation_entities.csv" , sep='') , stringsAsFactors = F)) 
valuation_entities_classifications = as.data.frame( fread(paste(ff.getPath("data") , "valuation_entities_classifications.csv" , sep='') , 
                                                          stringsAsFactors = F))  
valuation_entities_details = as.data.frame( fread(paste(ff.getPath("data") , "valuation_entities_details.csv" , sep='') , 
                                                  stringsAsFactors = F))  

## PROCS 

############################################################################
#############   TRAIN / TEST   #############################################
############################################################################

##### FINAL 

# remove NAs 
obsNAs = ff.obsNA(train)

# Ytrain
train = train[-obsNAs,]
Ytrain = train$REN_BASE_RENT

# remove REN_LEASE_LENGTH and REN_BASE_RENT (only in train)
predToDel = "REN_LEASE_LENGTH"
train = train [, -grep(pattern = paste(predToDel , "REN_BASE_RENT" , sep = "|") , x = colnames(train))]
test = test [, -grep(pattern = predToDel  , x = colnames(test))]

# makefeature 
train$REN_DATE_EFF_FROM = as.Date(train$REN_DATE_EFF_FROM)
test$REN_DATE_EFF_FROM = as.Date(test$REN_DATE_EFF_FROM)
l = ff.makeFeatureSet(data.train = train , data.test = test , meta = c("N","D","N"))
train = l$traindata
test = l$testdata
rm(l)

train = cbind(train,Ytrain=Ytrain)

############################################################################
#############   LAND           #############################################
############################################################################

### land
sum(is.na(land))

predNAs = ff.predNA(data = land,asIndex=F)
for (pp in predNAs) {
  perc = sum(is.na(land[,pp]))/nrow(land)
  cat("[",pp,"]:",perc,"-- NAs",sum(is.na(land[,pp])),"\n")
}

predToDel = c("LAN_LDS_NUMBER","LAN_LDS_NUMBER_ID_TYPE3","LAN_DATE_SUBDIVISION_LGA","LAN_DATE_SUBDIVISION_WAPC" , "LAN_SKETCH_ID" , 
              "LAN_ID1_PART_LOT" , "LAN_ID2_LOT" , "LAN_ID2_LOT" , "LAN_ID2_PART_LOT" , "LAN_ID3_PART_LOT" , "LAN_DATE_SURVEY_STRATA", 
              "LAN_DATE_LEASE_EXPIRY" , "LAN_DATE_LEASE_FROM" , "LAN_STR_ID_HAS_CORNER" )

for (pp in predToDel) land[,pp] <- NULL

## impute 
land$LAN_LDS_NUMBER_IS_RURAL[is.na(land$LAN_LDS_NUMBER_IS_RURAL)] <- -1 
land$LAN_ID1_LOT_NO[is.na(land$LAN_ID1_LOT_NO)] <- -1 
land$LLG_DATE_EFF_FROM[is.na(land$LLG_DATE_EFF_FROM)] <- "1849-01-01"
land$SUB_POSTCODE[is.na(land$SUB_POSTCODE)] <- -1 
land$URT_DATE_EFF_FROM[is.na(land$URT_DATE_EFF_FROM)] <- "1849-01-01"

sum(is.na(land))

## date 
land$LAN_DATE_SUBDIVISION_MFP = as.Date(land$LAN_DATE_SUBDIVISION_MFP)
land$LLG_DATE_EFF_FROM = as.Date(land$LLG_DATE_EFF_FROM)
land$URT_DATE_EFF_FROM = as.Date(land$URT_DATE_EFF_FROM)

## categ 
feature.names = colnames(land)
for (f in feature.names) {
  if (class(land[,f])=="character") {
    cat(">>> ",f," is character \n")
    levels <- unique(land[,f])
    land[,f] <- as.integer(factor(land[,f], levels=levels))
  }
}

## encode 
## dates index: 10 , 39 , 50 
meta_land = c(rep("N",9),"D",rep("N",28),"D",rep("N",10),"D","N")
l = ff.makeFeatureSet(data.train = land , data.test = land , meta = meta_land , 
                      scaleNumericFeatures = F , parallelize = F , remove1DummyVarInCatPreds = F)
land = l$traindata
rm(l)

### land_valuation_key
sum(! unique(land_valuation_key$VE_NUMBER) %in% unique(c(train$VE_NUMBER,test$VE_NUMBER)) ) ## 867777 
sum(! unique(c(train$VE_NUMBER)) %in%  unique(land_valuation_key$VE_NUMBER) ) # 3 
sum(! unique(c(test$VE_NUMBER)) %in%  unique(land_valuation_key$VE_NUMBER) ) # 0 

## remove such 3 VEN from train set 
VEN_remove = setdiff(x = unique(train$VE_NUMBER) , y = unique(land_valuation_key$VE_NUMBER))
train = train[! train$VE_NUMBER %in% VEN_remove,]

## remove from land VE_NUMBER not occurring either in train set or test set 
VEN_remove =  setdiff(x = unique(land_valuation_key$VE_NUMBER) , y = c(train$VE_NUMBER,test$VE_NUMBER) )
land_valuation_key = land_valuation_key[! land_valuation_key$VE_NUMBER %in% VEN_remove,,drop=F]

## LAN_ID e' unique in land_valuation_key ? ... no 
length(land_valuation_key$LAN_ID) == length(unique(land_valuation_key$LAN_ID))

## VE_NUMBER e' unique in land_valuation_key ? .. no 
length(land_valuation_key$VE_NUMBER) == length(unique(land_valuation_key$VE_NUMBER))

head(sort(table(land_valuation_key$VE_NUMBER),decreasing = T),100)
#4544602 4670844 1579901 2528154  730167 2728706 3469023 4074201 4403945 ..
#   20       7       6       6       5       5       5       5       5  ...

land_valuation_key[land_valuation_key$VE_NUMBER == 4544602,]
#   LAN_ID VE_NUMBER
#  1082263   4544602
#   684025   4544602
#  2327936   4544602
#  ...         ... 

lvkd = table(land_valuation_key$VE_NUMBER)
sum(lvkd>1) ##1694 
lvkd = lvkd[lvkd>1]
ve_lvkd = as.numeric(names(lvkd))
sum(unique(train$VE_NUMBER) %in% ve_lvkd) #1410
sum(unique(test$VE_NUMBER) %in% ve_lvkd) #688
te_lvdk = unique(test$VE_NUMBER)[unique(test$VE_NUMBER) %in% ve_lvkd]
mean(lvkd[names(lvkd) %in% te_lvdk]) ## 2.321221 

# 7624 19456 25805 39584 72903 77429 
# 2     2     2     2     2     3

ve = 7624
land_valuation_key[land_valuation_key$VE_NUMBER==ve,]

# LAN_ID VE_NUMBER
#  3427801      7624
#  4074461      7624

land[land$LAN_ID %in% land_valuation_key[land_valuation_key$VE_NUMBER==ve,]$LAN_ID,]

## >>> no rule. 100% pure noise. you have to cut without a criterion !!
for (ve in ve_lvkd) {
  li = land_valuation_key[land_valuation_key$VE_NUMBER==ve,]$LAN_ID
  ## maybe, last one is more updated??
  land_valuation_key = land_valuation_key[ ! (land_valuation_key$VE_NUMBER==ve & land_valuation_key$LAN_ID %in% li[1:(length(li)-1)]),]
}

## remove from land LAN_ID not occurring in land_valuation_key
sum(! (unique(land_valuation_key$LAN_ID) %in% unique(land$LAN_ID))  ) ## 0 
sum(! (unique(land$LAN_ID) %in% unique(land_valuation_key$LAN_ID))  ) ## 885370 
LD_remove =  setdiff(x = unique(land$LAN_ID) , y = unique(land_valuation_key$LAN_ID) )
land = land[! land$LAN_ID %in% LD_remove,,drop=F]

## joins 
sum(! (land_valuation_key$LAN_ID %in% land$LAN_ID) ) #0
sum(! (land$LAN_ID %in% land_valuation_key$LAN_ID) ) #0 
land = merge(x = land , y = land_valuation_key , by="LAN_ID" , all = F)

## Xtrain
sum(! (train$VE_NUMBER %in% land$VE_NUMBER) ) #0
Xtrain = merge(x = train , y = land , by = "VE_NUMBER" , all=F)
stopifnot(nrow(Xtrain)==nrow(train))
cat(">>> Xtrain:",dim(Xtrain),"\n")
Xtrain.n_now = nrow(Xtrain)

## Xtest
sum(! (test$VE_NUMBER %in% land$VE_NUMBER) ) #0
Xtest = merge(x = test , y = land , by = "VE_NUMBER" , all = F)
stopifnot(nrow(Xtest)==nrow(test))
cat(">>> Xtest:",dim(Xtest),"\n")
Xtest.n_now = nrow(Xtest)

##############################################  MERGE with Xtrain_NAs4.csv / Ytrain_NAs4.csv / Xtest_NAs4.csv  ######################

## write on disk 
cat(">>> merging from disk  Xtrain_NAs4.csv / Ytrain_NAs4.csv / Xtest_NAs4.csv ... \n")
Xtrain_NAs4 = as.data.frame( fread(paste(ff.getPath("elab") , "Xtrain_NAs4.csv" , sep='') , stringsAsFactors = F))
Xtest_NAs4 = as.data.frame( fread(paste(ff.getPath("elab") , "Xtest_NAs4.csv" , sep='') , stringsAsFactors = F))
Ytrain_NAs4 = as.data.frame( fread(paste(ff.getPath("elab") , "Ytrain_NAs4.csv" , sep='') , stringsAsFactors = F))

## checks 
Xtrain_NAs4$Ytrain_NAs4 = Ytrain_NAs4
mCols = c("REN_ID","VE_NUMBER" , "REN_DATE_EFF_FROM" , "LAN_MULTIPLE_ZONING_FLAG" , "LAN_SURVEY_STRATA_IND" , "LAN_SRD_TAXABLE" , "LAN_ID_TYPE" , "LAN_POWER"
          , "LAN_WATER", "LAN_GAS", "LAN_DRAINAGE", "LAN_DATE_SUBDIVISION_MFP",  "LAN_LST_CODE",  "LAN_LDS_NUMBER_IS_RURAL", "LAN_HOUSE_NO", "LAN_HOUSE_NO_SFX"
          , "LAN_ADDRESS_SITUATION" , "LAN_LOT_NO" , "LAN_UNIT_NO", "LAN_DATE_REDUNDANT_EFF", "LAN_RESERVE_CLASS", "LAN_LOCATION", "LAN_URBAN_MAP_GRID", "LAN_ID1_SURVEY_NO"
          , "LAN_ID1_ALPHA_LOT",  "LAN_ID1_LOT_NO" , "LAN_ID1_LEASE_PART", "LAN_ID1_SECTION",  "LAN_ID1_TYPE", "LAN_ID1_TOWN_LOT", "LAN_ID1_TOWN_LOT_TYPE" , "LAN_ID2_LEASE_PART"
          , "LAN_ID2_TYPE" , "LAN_ID2_ALPHA_PREFIX", "LAN_ID2_ALPHA_SUFFIX", "LAN_ID3_TYPE", "LAN_ID3_LEASE_RESERVE_NO", "LAN_ID3_LEASE_PART", "LAN_PART_LOT_SOURCE"
          , "LAN_STR_ID",  "LLG_DATE_EFF_FROM", "LDS_NAME", "LDS_CODE", "LDS_STATUS", "STR_NAME", "STR_STY_CODE", "CORNER_STR_NAME", "CORNER_STR_STATUS"
          , "CORNER_STR_STY_CODE", "SUB_NAME" , "SUB_POSTCODE",  "URT_DATE_EFF_FROM",  "URT_URBAN_RURAL_IND")

## train 
data = merge(x = Xtrain , y = Xtrain_NAs4 , by = mCols , all = F) 
stopifnot(sum(data$Ytrain_NAs4 != data$Ytrain)==0)
data$Ytrain_NAs4 <- NULL

## test 
data_test = merge(x = Xtest , y = Xtest_NAs4 , by = mCols , all = F) 
stopifnot(nrow(data_test)==nrow(Xtest_NAs4) , nrow(data_test)==nrow(Xtest))

## finally 
Xtrain <- data 
Xtest <- data_test

rm(list = c("data","data_test"))
cat(">>> end of merge\n")
##############################################  End of MERGE 

### land_admin_areas  
cat(">>> processing land_admin_areas ... \n") 
sum(!  (unique(land$LAN_ID) %in% unique(land_admin_areas$LAN_ID)) ) / length(unique(land$LAN_ID)) ## 0.8265898

head(sort(table(land_admin_areas$LAN_ID),decreasing = T))
# 113756 261981 275548 285676 302272 376410 
#   4      4      4      4      4      4 


## make uniq - taking most updated version 
land_admin_areas$LAA_DATE_EFF_FROM <- as.Date(land_admin_areas$LAA_DATE_EFF_FROM)

laa_uniq <- land_admin_areas[1:length(unique(land_admin_areas$LAN_ID)),]
i <- 1 
a = lapply(unique(land_admin_areas$LAN_ID), function(x) {
  laa = land_admin_areas[land_admin_areas$LAN_ID==x,,drop=F]
  idx <- which(laa$LAA_DATE_EFF_FROM == max(laa$LAA_DATE_EFF_FROM))
  laa_uniq[i, ] <<- laa[idx,]
  i <<- i + 1 
})

stopifnot(  length(unique(land_admin_areas$LAN_ID)) == length(unique(laa_uniq$LAN_ID))  )

## merge 
Xtrain = merge(x = Xtrain , y = laa_uniq[,c("LAN_ID","LAA_ADA_LGA_NUMBER","LAA_ADA_CODE")] , by = "LAN_ID" , all.x = T , all.y = F)
Xtest = merge(x = Xtest , y = laa_uniq[,c("LAN_ID","LAA_ADA_LGA_NUMBER","LAA_ADA_CODE")] , by = "LAN_ID" , all.x = T , all.y = F)
cat(">>> Xtrain:",dim(Xtrain),"\n")
cat(">>> Xtest:",dim(Xtest),"\n")
stopifnot(nrow(Xtrain)==Xtrain.n_now)
stopifnot(nrow(Xtest)==Xtest.n_now)

## imputing NAs 
sum(is.na(Xtrain)) # 1465958 
sum(is.na(Xtest))  # 258542

predNAs.test = ff.predNA(data=Xtest,asIndex = F) 
cat(">>> there're ",sum(is.na(Xtest)),"NAs in Xtest @:",predNAs.test," --> imputing (with 0 as min = 1) for XGBoost ...\n")

predNAs.train = ff.predNA(data=Xtrain,asIndex = F) 
cat(">>> there're ",sum(is.na(Xtest)),"NAs in Xtrain @:",predNAs.train," --> imputing (with 0 as min = 1) for XGBoost ...\n")

stopifnot(identical(predNAs.train,predNAs.test))

Xtrain[is.na(Xtrain)] <- 0 
Xtest[is.na(Xtest)] <- 0 

stopifnot(  sum(is.na(Xtrain)) == 0 ,  sum(is.na(Xtest)) == 0 )

### land_pins
cat(">>> processing land_pins ... \n") 
rinit <- sum(!  unique(land$LAN_ID) %in% unique(land_pins$LAN_ID) ) / length(unique(land$LAN_ID)) ## 2.77356e-05
sum(!  unique(land_pins$LAN_ID) %in% unique(land$LAN_ID) ) / length(unique(land_pins$LAN_ID)) ## 0.7544071

## focusing on LAN_ID in land 
land_pins <- land_pins[land_pins$LAN_ID %in% unique(land$LAN_ID) , ]
stopifnot(sum(!  unique(land$LAN_ID) %in% unique(land_pins$LAN_ID) ) / length(unique(land$LAN_ID)) == rinit)

head(sort(table(land_pins$LAN_ID),decreasing = T))
# 2662828 1594367   49074  315484  396688  575447 
#   11       5       4       4       4       4 

describe(land_pins)

predKeep = c("LNP_CENTROID_EASTING", "LNP_CENTROID_NORTHING", "LNP_CENTROID_YLAT", "LNP_CENTROID_ZONE", "LNP_CENTROID_XLONG")
cat(">>> keeping only predictors with low rate of missing values: ",predKeep,"...\n")

## make uniq - taking last one 
lp_uniq <- land_pins[1:length(unique(land_pins$LAN_ID)),]
i <- 1 
a = lapply(unique(land_pins$LAN_ID), function(x) {
  lp = land_pins[land_pins$LAN_ID==x,,drop=F]
  lp_uniq[i, ] <<- lp[nrow(lp),]
  i <<- i + 1 
})

stopifnot(  length(unique(land_pins$LAN_ID)) == length(unique(lp_uniq$LAN_ID))  )

## merge 
Xtrain = merge(x = Xtrain , y = lp_uniq[,c(predKeep,"LAN_ID")] , by = "LAN_ID" , all.x = T , all.y = F)
Xtest = merge(x = Xtest , y = lp_uniq[,c(predKeep,"LAN_ID")] , by = "LAN_ID" , all.x = T , all.y = F)
cat(">>> Xtrain:",dim(Xtrain),"\n")
cat(">>> Xtest:",dim(Xtest),"\n")
stopifnot(nrow(Xtrain)==Xtrain.n_now)
stopifnot(nrow(Xtest)==Xtest.n_now)

## imputing NAs 
sum(is.na(Xtrain)) # 6529 
sum(is.na(Xtest))  # 3155

predNAs.test = ff.predNA(data=Xtest,asIndex = F) 
cat(">>> there're ",sum(is.na(Xtest)),"NAs in Xtest @:",predNAs.test," --> imputing for XGBoost ...\n")

predNAs.train = ff.predNA(data=Xtrain,asIndex = F) 
cat(">>> there're ",sum(is.na(Xtest)),"NAs in Xtrain @:",predNAs.train," --> imputing for XGBoost ...\n")

stopifnot(identical(predNAs.train,predNAs.test))

## LNP_CENTROID_EASTING <- 0 
## LNP_CENTROID_NORTHING <- 0 
## LNP_CENTROID_YLAT <- 0 
## LNP_CENTROID_ZONE <- 0 
## LNP_CENTROID_XLONG <- 0 

Xtrain[is.na(Xtrain)] <- 0 
Xtest[is.na(Xtest)] <- 0 

stopifnot(  sum(is.na(Xtrain)) == 0 ,  sum(is.na(Xtest)) == 0 )


### land_restrictions
cat(">>> processing land_restrictions ... \n") 
sum(!  unique(land$LAN_ID) %in% unique(land_restrictions$LAN_ID) ) / length(unique(land$LAN_ID)) #### 0.999681

head(sort(table(land_restrictions$LAN_ID),decreasing = T))
# 2244747 4130780  211001  273638  326957  400756 
#   3       3       2       2       2       2 

## make uniq - taking most updated version 
land_restrictions$LRS_DATE_START <- as.Date(land_restrictions$LRS_DATE_START)

lr_uniq <- land_restrictions[1:length(unique(land_restrictions$LAN_ID)),]
i <- 1 
a = lapply(unique(land_restrictions$LAN_ID), function(x) {
  lr = land_restrictions[land_restrictions$LAN_ID==x,,drop=F]
  idx <- which(lr$LRS_DATE_START == max(lr$LRS_DATE_START))
  lr_uniq[i, ] <<- lr[idx,]
  i <<- i + 1 
})

stopifnot(  length(unique(land_restrictions$LAN_ID)) == length(unique(lr_uniq$LAN_ID))  )

## encoding 
levels <- unique( lr_uniq$LRS_RST_CODE   )
lr_uniq$LRS_RST_CODE_enc = as.integer(factor(lr_uniq$LRS_RST_CODE, levels=levels))

## keeping only LRS_RST_CODE 
## merge 
Xtrain = merge(x = Xtrain , y = lr_uniq[,c("LRS_RST_CODE_enc","LAN_ID")] , by = "LAN_ID" , all.x = T , all.y = F)
Xtest = merge(x = Xtest , y = lr_uniq[,c("LRS_RST_CODE_enc","LAN_ID")] , by = "LAN_ID" , all.x = T , all.y = F)
cat(">>> Xtrain:",dim(Xtrain),"\n")
cat(">>> Xtest:",dim(Xtest),"\n")
stopifnot(nrow(Xtrain)==Xtrain.n_now)
stopifnot(nrow(Xtest)==Xtest.n_now)

cat(">>> imputing NAs (with 0) for XGBoost ...\n")

Xtrain[is.na(Xtrain)] <- 0 
Xtest[is.na(Xtest)] <- 0 

stopifnot(  sum(is.na(Xtrain)) == 0 ,  sum(is.na(Xtest)) == 0 )


### land_urban
cat(">>> processing land_urban ... \n") 
linit = sum(!  unique(land$LAN_ID) %in% unique(land_urban$LAN_ID) ) / length(unique(land$LAN_ID)) #### 0.3121017
sum(!  unique(land_urban$LAN_ID) %in% unique(land$LAN_ID) ) / length(unique(land_urban$LAN_ID)) ### 0.6945567

## focusing on LAN_ID in land 
land_urban <- land_urban[land_urban$LAN_ID %in% unique(land$LAN_ID) , ]
stopifnot(sum(!  unique(land$LAN_ID) %in% unique(land_urban$LAN_ID) ) / length(unique(land$LAN_ID)) == linit)

head(sort(table(land_urban$LAN_ID),decreasing = T))
# 2694845 3477626 4970725   21682   44699  130246 
#   5       5       5       4       4       4 

## eliminate not valid ocuurencies 
land_urban <- land_urban[! (land_urban$URV_VEN_QUANTITY_IND == 'Y'  & is.na(land_urban$ULV_QUANTITY)) , ]

## 224 (quant) --> 0 or value 
## other (qualit) --> 0 (no) / 1 (yes)
land_urban$ULV_DATE_EFF_FROM <- as.Date(land_urban$ULV_DATE_EFF_FROM)
URV_IDs = sort(unique(land_urban$URV_ID))

dlu = as.data.frame(matrix(data = rep(0, (length(URV_IDs)+1)*length(unique(land_urban$LAN_ID)) ) , 
                           nrow = length(unique(land_urban$LAN_ID)) , 
                           ncol = (length(URV_IDs)+1)))
dlu[,1] = sort(unique(land_urban$LAN_ID))
colnames(dlu) <- c( "LAN_ID" , paste0("LU_URV",URV_IDs) )

a = lapply(sort(unique(land_urban$LAN_ID)),function(x) {
  lu <- land_urban[land_urban$LAN_ID==x,,drop=F]
  uids <- unique(lu$URV_ID)
  for (uid in uids) {
    idx <- which(uid == URV_IDs)
    if (uid == 224) {
      aalu <- land_urban[land_urban$LAN_ID==x & land_urban$URV_ID==224,]$ULV_QUANTITY
      stopifnot(length(aalu)>0)
      dlu[dlu$LAN_ID==x , (idx+1) ] <<- aalu
    } else {
      dlu[dlu$LAN_ID==x , (idx+1) ] <<- 1
    }
  } 
})

## merge 
Xtrain = merge(x = Xtrain , y = dlu , by = "LAN_ID" , all.x = T , all.y = F)
Xtest = merge(x = Xtest , y = dlu , by = "LAN_ID" , all.x = T , all.y = F)
cat(">>> Xtrain:",dim(Xtrain),"\n")
cat(">>> Xtest:",dim(Xtest),"\n")
stopifnot(nrow(Xtrain)==Xtrain.n_now)
stopifnot(nrow(Xtest)==Xtest.n_now)

cat(">>> imputing NAs (with 0) for XGBoost ...\n")
sum(is.na(Xtrain)) ## 4672800
sum(is.na(Xtest)) ## 1626432

Xtrain[is.na(Xtrain)] <- 0 
Xtest[is.na(Xtest)] <- 0 

stopifnot(  sum(is.na(Xtrain)) == 0 ,  sum(is.na(Xtest)) == 0 )

### land_zonings
zinit <- sum(!  unique(land$LAN_ID) %in% unique(land_zonings$LAN_ID) ) / length(unique(land$LAN_ID)) #### 0.1572851
sum(!  unique(land_zonings$LAN_ID) %in% unique(land$LAN_ID) ) / length(unique(land_zonings$LAN_ID)) ### 0.7720676 

## focusing on LAN_ID in land 
land_zonings <- land_zonings[land_zonings$LAN_ID %in% unique(land$LAN_ID) , ]
stopifnot(sum(!  unique(land$LAN_ID) %in% unique(land_zonings$LAN_ID) ) / length(unique(land$LAN_ID)) == zinit)

head(sort(table(land_zonings$LAN_ID),decreasing = T))
# 1222002 4367311 4570300   26384  229085  245604 
# 10      10      10       9       9       9 

## make uniq
land_zonings$LNZ_DATE_EFF_FROM <- as.Date(land_zonings$LNZ_DATE_EFF_FROM)
lz_uniq <- land_zonings[1:length(unique(land_zonings$LAN_ID)) , ]
i <- 1 
a = lapply(sort(unique(land_zonings$LAN_ID)), function(x) {
  lz = land_zonings[land_zonings$LAN_ID==x,,drop=F]
  idx <- which(lz$LNZ_DATE_EFF_FROM == max(lz$LNZ_DATE_EFF_FROM))
  lz_uniq[i, ] <<- lz[idx,]
  i <<- i + 1 
})

stopifnot(  length(unique(land_zonings$LAN_ID)) == length(unique(lz_uniq$LAN_ID))  )

## encode 
levels <- unique( lz_uniq$LNZ_LGZ_ZON_CODE   )
lz_uniq$LNZ_LGZ_ZON_CODE_enc = as.integer(factor(lz_uniq$LNZ_LGZ_ZON_CODE, levels=levels))

## keeping only LNZ_LGZ_ZON_CODE 
## merge 
Xtrain = merge(x = Xtrain , y = lz_uniq[,c("LNZ_LGZ_ZON_CODE_enc","LAN_ID")] , by = "LAN_ID" , all.x = T , all.y = F)
Xtest = merge(x = Xtest , y = lz_uniq[,c("LNZ_LGZ_ZON_CODE_enc","LAN_ID")] , by = "LAN_ID" , all.x = T , all.y = F)
cat(">>> Xtrain:",dim(Xtrain),"\n")
cat(">>> Xtest:",dim(Xtest),"\n")
stopifnot(nrow(Xtrain)==Xtrain.n_now)
stopifnot(nrow(Xtest)==Xtest.n_now)

cat(">>> imputing NAs (with 0) for XGBoost ...\n")
sum(is.na(Xtrain)) 
sum(is.na(Xtest)) 

Xtrain[is.na(Xtrain)] <- 0 
Xtest[is.na(Xtest)] <- 0 

stopifnot(  sum(is.na(Xtrain)) == 0 ,  sum(is.na(Xtest)) == 0 )

########### FINAL 
## MEMO #1: remove REN_ID in train / test set before fitting models 
## MEMO #2: in final dataset remove LAN_ID 

Ytrain = Xtrain$Ytrain

predToDel = c("LAN_ID","Ytrain")
for (pp in predToDel) {
  cat(">>> removing ",pp,"...\n")
  Xtrain[,pp] <- NULL
  Xtest[,pp] <- NULL
}

## write on disk 
cat(">>> writing on disk ... \n")
write.csv(Xtrain,
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"Xtrain_NAs5.csv"),
          row.names=FALSE)
write.csv(data.frame(Ytrain=Ytrain),
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"Ytrain_NAs5.csv"),
          row.names=FALSE)
write.csv(Xtest,
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"Xtest_NAs5.csv"),
          row.names=FALSE)

############################################################################
#############   VALUATION ENTITIES           ###############################
############################################################################
