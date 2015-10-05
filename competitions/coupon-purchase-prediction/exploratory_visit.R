library(fastfurious)
library(data.table)
library(plyr)
library(Hmisc)

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'code' , sub_path = 'competitions/coupon-purchase-prediction')
ff.bindPath(type = 'data' , sub_path = 'dataset/coupon-purchase-prediction/data')

### GLOBAL CONFIG 
debug = F

### DATA & COUPON VECTORS 
coupon_visit = as.data.frame( fread(paste(ff.getPath("data") , "coupon_visit_train.csv" , sep='')))

source(paste0(ff.getPath('code'),'make_coupon_vector.R'))
ff.bindPath(type = 'elab' , sub_path = 'dataset/coupon-purchase-prediction/elab' , createDir = T)
ff.bindPath(type = 'elab_train' , sub_path = 'dataset/coupon-purchase-prediction/elab/train' , createDir = T)
ff.bindPath(type = 'elab_labels' , sub_path = 'dataset/coupon-purchase-prediction/elab/labels' , createDir = T)
ff.bindPath(type = 'elab_pred' , sub_path = 'dataset/coupon-purchase-prediction/elab/pred' , createDir = T)

### PROCESSING 
coupon_visit$I_DATE = as.Date(coupon_visit$I_DATE)
colnames(coupon_visit)[5] = 'COUPON_ID_hash'

cat(">>> how many test coupons are in there? \n") 
coupon_visit_test = merge(x=coupon_visit,y=coupon_list_test.meta[,c('COUPON_ID_hash','DISPFROM')] , by='COUPON_ID_hash' , all=F)
cat(">>> visits: ",nrow(coupon_visit_test), "\n") # 741 (out of 2833180 visits)
cat(">>> coupons: ",length(unique(coupon_visit_test$COUPON_ID_hash)),"\n") ## 39 ~12%

cat(">>> eliminating from coupon_visit visits happened after the first day of coupon activating period ... \n") 
coupon_visit_ext = merge(x=coupon_visit,y=coupon_list_train.meta[,c('COUPON_ID_hash','DISPFROM')] , by='COUPON_ID_hash' , all=F)
coupon_visit_ext$TO_REMOVE = coupon_visit_ext$I_DATE >= coupon_visit_ext$DISPFROM

cat(">>> how many visits occur on average before the activating period out of the total number of visits? \n") 
print(table(coupon_visit_ext$TO_REMOVE)/nrow(coupon_visit_ext)) ## 3.4% 

coupon_visit_ext = coupon_visit_ext[!coupon_visit_ext$TO_REMOVE,]

trans_data_red = trans_data[,c('USER_ID_hash','COUPON_ID_hash')]
coupon_visit_red = ddply(coupon_visit_ext , .(USER_ID_hash,COUPON_ID_hash) , function(x) c(num=nrow(x)) )

## purchased: how many times viewed on average? 
cat(">>> On average, how many times purchased coupons are viewed before the activating period ? \n") 
coupon_visit_red_pur = merge(x = coupon_visit_red , y = trans_data_red , by=c('USER_ID_hash','COUPON_ID_hash') , all.x =  F , all.y = T)
coupon_visit_red_pur[is.na(coupon_visit_red_pur)] = 0
coupon_visit_red_pur_avg = ddply(coupon_visit_red_pur , .(USER_ID_hash) , function(x) c(avg_num=mean(x$num)) ) ## numero medio visite comprati

cat(">> mean:",mean(coupon_visit_red_pur_avg$avg_num),"\n") # 2% <<<<<<<<<<<<<<<<<
cat(">> sd:",sd(coupon_visit_red_pur_avg$avg_num),"\n") # 23% 
print(describe(coupon_visit_red_pur_avg$avg_num))

## viewed: how many times purchased on average? 
cat(">>> On average, how many times viewed coupons before the activating period are purchased? \n") 
trans_data_red = trans_data[,c('USER_ID_hash','COUPON_ID_hash')]
trans_data_red$PUR = 1
coupon_visit_red_view = merge(x = coupon_visit_red , y = trans_data_red , by=c('USER_ID_hash','COUPON_ID_hash') , all.x =  T , all.y = F)
coupon_visit_red_view[is.na(coupon_visit_red_view)] = 0
coupon_visit_red_view_avg = ddply(coupon_visit_red_view , .(USER_ID_hash) , function(x) c(avg_num=mean(x$num)) ) # numero medio acquisti dei visti 

cat(">> mean:",mean(coupon_visit_red_view_avg$avg_num),"\n") # 1.213912 <<<<<<<<<< i coupon visti primi del periodo di ativazioni sono comprati in media 1.2 volte!!!!
cat(">> sd:",sd(coupon_visit_red_view_avg$avg_num),"\n") #  0.5079141
describe(coupon_visit_red_view_avg$avg_num)

hist(coupon_visit_red_view_avg$avg_num)

## >>>>>>>>>>> nella la lista dei 10 coupon x utente, se un utente ha visto uno dei 39 coupon, mettili dentro 



