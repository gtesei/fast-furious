library(fastfurious)
library(data.table)
library(plyr)
library(Hmisc)

### FUNCS 

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'code' , sub_path = 'competitions/coupon-purchase-prediction')
ff.bindPath(type = 'data' , sub_path = 'dataset/coupon-purchase-prediction/data')

### GLOBAL CONFIG 
debug = F

### DATA & COUPON VECTORS 
source(paste0(ff.getPath('code'),'make_coupon_vector.R'))
ff.bindPath(type = 'elab' , sub_path = 'dataset/coupon-purchase-prediction/elab' , createDir = T)
ff.bindPath(type = 'elab_train' , sub_path = 'dataset/coupon-purchase-prediction/elab/train' , createDir = T)
ff.bindPath(type = 'elab_labels' , sub_path = 'dataset/coupon-purchase-prediction/elab/labels' , createDir = T)
ff.bindPath(type = 'elab_pred' , sub_path = 'dataset/coupon-purchase-prediction/elab/pred' , createDir = T)


### PROCESSING 
wn = 1 
coupons = getLabels(week_number = wn)$coupons
coupons_red = coupon_list_train.meta[coupon_list_train.meta$COUPON_ID_hash %in% coupons , ]
trans = coupon_detail_train.meta[coupon_detail_train.meta$COUPON_ID_hash %in% coupons , ]
trans_ext = merge(x = trans , y = coupon_list_train.meta , by='COUPON_ID_hash' , all.y = F , all.x =  T)
stopifnot(sum(is.na(trans_ext))==0)
par(mfrow=c(2,1))
hist(trans_ext$WDISPPERIOD)
hist(coupons_red$WDISPPERIOD)

describe(trans_ext$WDISPPERIOD)
describe(coupons_red$WDISPPERIOD)





