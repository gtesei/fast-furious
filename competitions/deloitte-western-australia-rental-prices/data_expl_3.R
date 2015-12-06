library(data.table)
library(fastfurious)
library(Hmisc)
library(Matrix)

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

### land_admin_areas  
sum(!  (unique(land$LAN_ID) %in% unique(land_admin_areas$LAN_ID)) ) / length(unique(land$LAN_ID)) ## 0.8265898

### land_pins
sum(!  unique(land$LAN_ID) %in% unique(land_pins$LAN_ID) ) / length(unique(land$LAN_ID)) ## 3.097041e-05 (9)

### land_restrictions
sum(!  unique(land$LAN_ID) %in% unique(land_restrictions$LAN_ID) ) / length(unique(land$LAN_ID)) #### 0.9996593

### land_urban
sum(!  unique(land$LAN_ID) %in% unique(land_urban$LAN_ID) ) / length(unique(land$LAN_ID)) #### 0.3122368

### land_zonings
sum(!  unique(land$LAN_ID) %in% unique(land_zonings$LAN_ID) ) / length(unique(land$LAN_ID)) #### 0.156139


############################################################################
#############   VALUATION ENTITIES           ###############################
############################################################################

### valuation_entities
sum(! (c(train$VE_NUMBER,test$VE_NUMBER) %in% unique(valuation_entities$VE_NUMBER)) ) ## 0 
sum(! (unique(valuation_entities$VE_NUMBER) %in% c(train$VE_NUMBER,test$VE_NUMBER)) ) ## 867851 

## remove VE in valuation_entities not occurring in train / test set 
VE_remove = setdiff(x = unique(valuation_entities$VE_NUMBER) , y = c(train$VE_NUMBER,test$VE_NUMBER) )
valuation_entities = valuation_entities[! valuation_entities$VE_NUMBER %in% VE_remove,,drop=F]

## NAs 
sum(is.na(valuation_entities)) ## 271458 
predNAs = ff.predNA(data = valuation_entities,asIndex=F)
for (pp in predNAs) {
  perc = sum(is.na(valuation_entities[,pp]))/nrow(valuation_entities)
  cat("[",pp,"]:",perc,"-- NAs",sum(is.na(valuation_entities[,pp])),"\n")
}
# [ VE_SUB_NUMBER ]: 0.9411247 -- NAs 271458  

## impute
valuation_entities$VE_SUB_NUMBER[is.na(valuation_entities$VE_SUB_NUMBER)] <- -1 

## feat. sel. 
valuation_entities$VE_DATE_CREATED <- NULL ## when in the system created a VE has been created doesn't really matter 
valuation_entities$VE_DATE_MODIFIED <- NULL ## neither when the system was last updated 
valuation_entities$VE_USE <- NULL ##  R (288438, 100%), V (2, 0%) 
valuation_entities$VE_CRIT_INFRA_IND <- NULL ##  N (288437, 100%), Y (3, 0%) 

## categ 
feature.names = colnames(valuation_entities)
for (f in feature.names) {
  if (class(valuation_entities[,f])=="character") {
    cat(">>> ",f," is character \n")
    levels <- unique(valuation_entities[,f])
    valuation_entities[,f] <- as.integer(factor(valuation_entities[,f], levels=levels))
  }
}

## merge 
Xtrain = merge(x = Xtrain , y = valuation_entities , by = "VE_NUMBER" , all = F)
Xtest = merge(x = Xtest , y = valuation_entities , by = "VE_NUMBER" , all = F)
cat(">>> Xtrain:",dim(Xtrain),"\n")
cat(">>> Xtest:",dim(Xtest),"\n")
stopifnot(nrow(Xtrain)==Xtrain.n_now)
stopifnot(nrow(Xtest)==Xtest.n_now)

### valuation_entities_classifications
sum(! (valuation_entities_classifications$VE_NUMBER %in% valuation_entities$VE_NUMBER) ) ## 1202279 
sum(! ( valuation_entities$VE_NUMBER %in% valuation_entities_classifications$VE_NUMBER) ) ## 4 
VE_NA = setdiff( x = valuation_entities$VE_NUMBER , y = unique(valuation_entities_classifications$VE_NUMBER))
sum( VE_NA %in% train$VE_NUMBER ) ## 0 
sum( VE_NA %in% test$VE_NUMBER  ) ## 4 

## focus on VEN occurring in valuation_entities
VE_remove = setdiff( x = valuation_entities_classifications$VE_NUMBER , y = unique(valuation_entities$VE_NUMBER))
valuation_entities_classifications = valuation_entities_classifications[! valuation_entities_classifications$VE_NUMBER %in% VE_remove,,drop=F]

## NAs 
sum(is.na(valuation_entities_classifications)) ## 369373 
predNAs = ff.predNA(data = valuation_entities_classifications,asIndex=F)
for (pp in predNAs) {
  perc = sum(is.na(valuation_entities_classifications[,pp]))/nrow(valuation_entities_classifications)
  cat("[",pp,"]:",perc,"-- NAs",sum(is.na(valuation_entities_classifications[,pp])),"\n")
}
#[ VEC_DATE_EFF_TO ]: 0.7400995 -- NAs 288436 
#[ CLS_CODE_PLURAL ]: 0.2076767 -- NAs 80937 

## feat. sel. 
valuation_entities_classifications$VEC_DATE_EFF_TO <- NULL
valuation_entities_classifications$VEC_DATE_EFF_FROM <- NULL

## imputing 
valuation_entities_classifications$CLS_CODE_PLURAL[is.na(valuation_entities_classifications$CLS_CODE_PLURAL)] <- -1 

## VE_NUMBER e' unique in valuation_entities_classifications ? .. no 
length(valuation_entities_classifications$VE_NUMBER) == length(unique(valuation_entities_classifications$VE_NUMBER))

head(sort(table(valuation_entities_classifications$VE_NUMBER),decreasing = T),100)
# 3291818 4911209     537   33272   38888  202446  356521  478047  546066
#   6       6       5       5       5       5       5       5       5       

valuation_entities_classifications[valuation_entities_classifications$VE_NUMBER == 3291818,]

# again, there's no rule. Keep the last one and cut the others!
lvkd = table(valuation_entities_classifications$VE_NUMBER)
sum(lvkd>1) ##92613 
lvkd = lvkd[lvkd>1]
ve_lvkd = as.numeric(names(lvkd))
ve_uniq = valuation_entities_classifications[1:length(ve_lvkd),]
i = 1
a = lapply(ve_lvkd , function(ve) {
  ves = valuation_entities_classifications[valuation_entities_classifications$VE_NUMBER == ve,,drop=F]
  ## maybe, last one is more updated??
  ve_uniq[i , ] <<- ves[nrow(ves),,drop=F]
  i <<- i + 1 
})
stopifnot( (i-1) == nrow(ve_uniq) )
valuation_entities_classifications = valuation_entities_classifications[!valuation_entities_classifications$VE_NUMBER %in% ve_lvkd,] 
valuation_entities_classifications = rbind(valuation_entities_classifications,ve_uniq)

## categ 
feature.names = colnames(valuation_entities_classifications)
for (f in feature.names) {
  if (class(valuation_entities_classifications[,f])=="character") {
    cat(">>> ",f," is character \n")
    levels <- unique(valuation_entities_classifications[,f])
    valuation_entities_classifications[,f] <- as.integer(factor(valuation_entities_classifications[,f], levels=levels))
  }
}

## merge 
Xtrain = merge(x = Xtrain , y = valuation_entities_classifications , by = "VE_NUMBER" , all = F)
Xtest = merge(x = Xtest , y = valuation_entities_classifications , by = "VE_NUMBER" , all.x = T , all.y = F)
cat(">>> Xtrain:",dim(Xtrain),"\n")
cat(">>> Xtest:",dim(Xtest),"\n")
stopifnot(nrow(Xtrain)==Xtrain.n_now)
stopifnot(nrow(Xtest)==Xtest.n_now)

## imputing NAs 
predNAs = ff.predNA(data=Xtest) 
obsNAs = ff.obsNA(data=Xtest) 
cat(">>> there're ",sum(is.na(Xtest)),"NAs in Xtest @:",colnames(Xtest)[predNAs]," --> imputing for XGBoost ...\n")
data = rbind(Xtrain[,-grep(pattern = "Ytrain",x = colnames(Xtrain))],Xtest) 
predNAsIndex = which(predNAs)
mins = rep(NA,length(predNAsIndex))
for (i in seq_along(mins)) {
  mins[i] <- min(data[,predNAsIndex[i]],na.rm=T)
  cat(">>> min(",colnames(data)[predNAsIndex[i]],"):",mins[i],"--> imputing NAs with ",min(0,(mins[i]-1)),"\n")
  Xtest[,predNAsIndex[i]][is.na(Xtest[,predNAsIndex[i]])] <- min(0,(mins[i]-1))
}

### valuation_entities_details
sum(! (valuation_entities_details$VE_NUMBER %in% valuation_entities$VE_NUMBER) ) ## 9564212
sum(! ( valuation_entities$VE_NUMBER %in% valuation_entities_details$VE_NUMBER) ) ## 107
VE_NA = setdiff( x = valuation_entities$VE_NUMBER , y = unique(valuation_entities_details$VE_NUMBER))
sum(VE_NA %in% train$VE_NUMBER ) ## 12
sum(VE_NA %in% test$VE_NUMBER ) ## 92

## focus on VEN occurring in valuation_entities
VE_remove = setdiff( x = valuation_entities_details$VE_NUMBER , y = unique(valuation_entities$VE_NUMBER))
valuation_entities_details = valuation_entities_details[! valuation_entities_details$VE_NUMBER %in% VE_remove,,drop=F]

## NAs 
sum(is.na(valuation_entities_details)) ## 5045478 

predNAs = ff.predNA(data = valuation_entities_details,asIndex=F)
for (pp in predNAs) {
  perc = sum(is.na(valuation_entities_details[,pp]))/nrow(valuation_entities_details)
  cat("[",pp,"]:",perc,"-- NAs",sum(is.na(valuation_entities_details[,pp])),"\n")
}
# [ UVV_DATE_EFF_TO ]: 0.9357387 -- NAs 3791152 
# [ UVV_QUANTITY ]: 0.3095949 -- NAs 1254326 


########### FINAL 
## MEMO #1: remove REN_ID in train / test set before fitting models 
## MEMO #2: in final dataset remove LAN_ID 
length(unique(land_valuation_key$LAN_ID))/nrow(land_valuation_key) ## 0.9999931 

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
          file=paste0(ff.getPath("elab"),"Xtrain_NAs_2_3.csv"),
          row.names=FALSE)
write.csv(data.frame(Ytrain=Ytrain),
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"Ytrain_NAs_2_3.csv"),
          row.names=FALSE)
write.csv(Xtest,
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"Xtest_NAs_2_3.csv"),
          row.names=FALSE)



