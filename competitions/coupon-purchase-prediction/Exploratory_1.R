library(fastfurious)
library(data.table)
library(plyr)

################# FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/coupon-purchase-prediction/data')
ff.bindPath(type = 'elab' , sub_path = 'dataset/coupon-purchase-prediction/elab' , createDir = T)
ff.bindPath(type = 'process' , sub_path = 'data_process')

################# DATA
users = train_enc = as.data.frame( fread(paste(ff.getPath("data") , "user_list.csv" , sep=''))) 

coupon_list_train = as.data.frame( fread(paste(ff.getPath("data") , "coupon_list_train.csv" , sep='')))
coupon_list_test = as.data.frame( fread(paste(ff.getPath("data") , "coupon_list_test.csv" , sep='')))

coupon_detail_train = as.data.frame( fread(paste(ff.getPath("data") , "coupon_detail_train.csv" , sep='')))

coupon_area_train = as.data.frame( fread(paste(ff.getPath("data") , "coupon_area_train.csv" , sep='')))
coupon_area_test = as.data.frame( fread(paste(ff.getPath("data") , "coupon_area_test.csv" , sep='')))

###############
coupon_sales_train = merge(x = coupon_list_train , y = coupon_detail_train , by = 'COUPON_ID_hash' , all = F)

sum(!coupon_sales_train$I_DATE < coupon_sales_train$DISPEND)
##[1] 4
coupon_sales_train[which(!coupon_sales_train$I_DATE < coupon_sales_train$DISPEND), ]
## fuori per questioni di minuti 
# DISPEND                I_DATE
# 2011-12-03 12:00:00   2011-12-03 12:00:01
# 2011-08-13 13:00:00   2011-08-13 18:26:53 
# 2011-08-13 13:00:00   2011-08-13 13:06:20 
# 2011-07-17 12:00:00   2011-07-17 12:00:01  

## let's check on the test set 
sum( coupon_list_test$DISPEND < as.Date('2012-06-24') )
# [1] 0

sum( coupon_list_test$DISPFROM > as.Date('2012-06-30') )
# [1] 0

#### >>>> quindi tutti i coupon del test set sono attivi nel periodo di test 

sum(!coupon_sales_train$I_DATE > coupon_sales_train$DISPFROM)
##[1] 0

sum(!coupon_sales_train$I_DATE > coupon_sales_train$VALIDFROM , na.rm = T)
# [1] 97123
sum(!coupon_sales_train$I_DATE < coupon_sales_train$VALIDEND , na.rm = T)
# [1] 1

#### >>> ne discende che un coupon viene venduto in un tempo compreso tra [coupon_sales$DISPFROM , coupon_sales$DISPEND] 

#### >>> inoltre, dato che VALIDFROM e VALIDEND hanno circa 65.0000 NAs cadauno --> li butto nel cestino!!


#### how many coupons an average user bougth in a week ??

## The training set spans the dates 2011-07-01 to 2012-06-23
## The test set spans the week after the end of the training set, 2012-06-24 to 2012-06-30

day_1 = as.Date(min(coupon_detail_train$I_DATE))
#"2011-07-01 00:10:42"

day_max_train = as.Date(max(coupon_detail_train$I_DATE))
#"2012-06-23 23:54:47"

day_last = as.Date("2012-06-30")

day_last - day_max_train
#Time difference of 7 days

day_last - day_1
#Time difference of 7 days 

weeks = 1+ ceiling(as.numeric(day_max_train - day_1) / 7)
#[1] 52

a_day = as.Date("2011-07-01 18:32:55")
week_number_of_a_day = 1 + floor(as.numeric(a_day-day_1)/7)
#[1] 32

coupon_sales_train$I_DATE_WN = 1 + floor(as.numeric(as.Date(coupon_sales_train$I_DATE)-day_1)/7)

l = ff.encodeCategoricalFeature(data.train = coupon_sales_train$I_DATE_WN , 
                                data.test = coupon_sales_train$I_DATE_WN , 
                                asNumericSequence=T, 
                                colname.prefix = 'I_DATE_WN')


coupon_sales_train = cbind(coupon_sales_train , l$traindata)

clust = ddply(coupon_sales_train , .(USER_ID_hash) , function(x) {
  ret = rep(NA,52)
  for (i in 1:52) {
    ret[i] = sum( x[,paste0('I_DATE_WN_',i)] )
  }
  setNames(object = ret , nm = paste0('I_DATE_WN_',1:52))
} )

## check 
coupon_sales_train[coupon_sales_train$USER_ID_hash=='0000b53e182165208887ba65c079fc21',] ## ok 
coupon_sales_train[coupon_sales_train$USER_ID_hash=='00035b86e6884589ec8d28fbf2fe7757',] ## ok 
coupon_sales_train[coupon_sales_train$USER_ID_hash=='000cc06982785a19e2a2fdb40b1c9d59',] ## ok 

max(apply(X = clust[2:ncol(clust)] , MARGIN = 2 , FUN = max))
#[1] 31

## Max coupons sales per user/week 
cat(">>> Coupons max sales per user/week ...\n")
as = setNames(apply(X = clust[2:ncol(clust)] , MARGIN = 2 , FUN = max),1:weeks)
barplot(as, legend.text = F,
        main = "Max coupons sales per user by week", 
        col = terrain.colors(weeks),
        beside = TRUE,
        xlab = "Week",
        ylab = "Sales (units)")

text(wn(day = "2011-12-25"), 25, "Christmas" , cex=0.8 , col = 'purple' )
text(wn(day = "2011-11-26"), 28, "Thanksgiving day" , cex=0.8 , col = 'purple')
text(wn(day = "2012-06-30"), 25, "June 30th" , cex=0.8 , col = 'purple')
par(new=TRUE)
plot( x = 1:weeks , y = filter(as, rep(1, 5)),  type='l', pch=27 , col="blue", lty=6, lwd=2 , axes=FALSE , xlab = "",
      ylab = "") 

## Average coupons sales per user/week
cat(">>> Coupons average sales per user/week ...\n")
as = setNames(apply(X = clust[2:ncol(clust)] , MARGIN = 2 , FUN = mean),1:weeks)
barplot(as, legend.text = F,
        main = "Average coupons sales per user (at least 1 purchase) by week", 
        col = terrain.colors(weeks),
        beside = TRUE,
        xlab = "Week",
        ylab = "Sales (units)")
text(wn(day = "2011-12-25"), 0.2, "Christmas" , cex=0.8 , col = 'brown' )
text(wn(day = "2011-11-26"), 0.23, "Thanksgiving day" , cex=0.8 , col = 'brown')
text(wn(day = "2012-06-30"), 0.23, "June 30th" , cex=0.8 , col = 'brown')
par(new=TRUE)
plot( x = 1:weeks , y = filter(as, rep(1, 5)),  type='l', pch=27 , col="blue", lty=6, lwd=2 , axes=FALSE , xlab = "",
      ylab = "") 

########
dim(users)
# [1] 22873     6 

### quasi tutti gli utenti hanno acquistato almeno 1 coupon 
length(unique(coupon_detail_train$USER_ID_hash))
# 22782 

## cluster by user 
c2 = ddply(coupon_detail_train , .(USER_ID_hash) , function(x) c(tot = nrow(x) , uniq_coupon = length(unique(x$COUPON_ID_hash)) ))


## ogni utente ha in media acquistato 7.4 coupon 
mean(c2$tot)
# 7.417962 

## di cui 6.9 (quasi tutti) sono diversi tra loro 
mean(c2$uniq_coupon)
# 6.976253

###### coupon_area_train , coupon_area_test 
c3 = ddply(coupon_area_test , .(COUPON_ID_hash) , function(x) c(num=nrow(x)) )

### in media ci sono 6.9 coupon listing area for each coupon in the test set 
mean(c3$num)
# 6.983871 

# le pref area non sono molte  
length(unique(coupon_area_train$PREF_NAME))
# 47

## e neanche le small area 
length(unique(coupon_area_train$SMALL_AREA_NAME))
# 55 


####
intersect( unique(users$PREF_NAME) , coupon_list_train$ken_name)
# [1] "東京都"   "愛知県"   "神奈川県" "広島県"   "埼玉県"   "奈良県"   "石川県"   "大阪府"   "熊本県"   "福岡県"   "北海道"   "京都府"   "秋田県"  
# [14] "千葉県"   "長崎県"   "兵庫県"   "沖縄県"   "三重県"   "茨城県"   "鹿児島県" "宮城県"   "静岡県"   "和歌山県" "長野県"   "岡山県"   "栃木県"  
# [27] "滋賀県"   "富山県"   "佐賀県"   "宮崎県"   "岩手県"   "新潟県"   "大分県"   "山口県"   "岐阜県"   "群馬県"   "福島県"   "愛媛県"   "香川県"  
# [40] "山梨県"   "高知県"   "島根県"   "徳島県"   "福井県"   "青森県"   "山形県"   "鳥取県"  

intersect( unique(users$PREF_NAME) , coupon_area_train$PREF_NAME)
# [1] "東京都"   "愛知県"   "神奈川県" "広島県"   "埼玉県"   "奈良県"   "石川県"   "大阪府"   "熊本県"   "福岡県"   "北海道"   "京都府"   "秋田県"  
# [14] "千葉県"   "長崎県"   "兵庫県"   "沖縄県"   "三重県"   "茨城県"   "鹿児島県" "宮城県"   "静岡県"   "和歌山県" "長野県"   "岡山県"   "栃木県"  
# [27] "滋賀県"   "富山県"   "佐賀県"   "宮崎県"   "岩手県"   "新潟県"   "大分県"   "山口県"   "岐阜県"   "群馬県"   "福島県"   "愛媛県"   "香川県"  
# [40] "山梨県"   "高知県"   "島根県"   "徳島県"   "福井県"   "青森県"   "山形県"   "鳥取県" 

intersect( unique(coupon_list_train$small_area_name) ,  unique(coupon_detail_train$SMALL_AREA_NAME) )
# [1] "兵庫"                         "銀座・新橋・東京・上野"       "恵比寿・目黒・品川"           "渋谷・青山・自由が丘"        
# [5] "新宿・高田馬場・中野・吉祥寺" "群馬"                         "愛知"                         "山形"                        
# [9] "赤坂・六本木・麻布"           "川崎・湘南・箱根他"           "埼玉"                         "横浜"                        
# [13] "栃木"                         "広島"                         "池袋・神楽坂・赤羽"           "三重"                        
# [17] "岐阜"                         "静岡"                         "キタ"                         "ミナミ他"                    
# [21] "滋賀"                         "京都"                         "北海道"                       "石川"                        
# [25] "長野"                         "千葉"                         "和歌山"                       "鹿児島"                      
# [29] "佐賀"                         "長崎"                         "福岡"                         "大分"                        
# [33] "宮崎"                         "沖縄"                         "立川・町田・八王子他"         "岩手"                        
# [37] "富山"                         "島根"                         "山口"                         "奈良"                        
# [41] "福島"                         "青森"                         "宮城"                         "茨城"                        
# [45] "秋田"                         "岡山"                         "愛媛"                         "熊本"                        
# [49] "香川"                         "徳島"                         "高知"                         "福井"                        
# [53] "新潟"                         "鳥取"                         "山梨"                        

intersect( unique(coupon_area_train$SMALL_AREA_NAME) ,  unique(coupon_detail_train$SMALL_AREA_NAME) )
# [1] "埼玉"                         "千葉"                         "新宿・高田馬場・中野・吉祥寺" "京都"                        
# [5] "恵比寿・目黒・品川"           "銀座・新橋・東京・上野"       "愛知"                         "川崎・湘南・箱根他"          
# [9] "北海道"                       "福岡"                         "栃木"                         "ミナミ他"                    
# [13] "渋谷・青山・自由が丘"         "池袋・神楽坂・赤羽"           "赤坂・六本木・麻布"           "横浜"                        
# [17] "宮城"                         "福島"                         "大分"                         "高知"                        
# [21] "立川・町田・八王子他"         "広島"                         "新潟"                         "岡山"                        
# [25] "愛媛"                         "香川"                         "キタ"                         "徳島"                        
# [29] "兵庫"                         "岐阜"                         "宮崎"                         "長崎"                        
# [33] "山梨"                         "石川"                         "山口"                         "富山"                        
# [37] "山形"                         "秋田"                         "鳥取"                         "奈良"                        
# [41] "鹿児島"                       "三重"                         "熊本"                         "長野"                        
# [45] "滋賀"                         "静岡"                         "青森"                         "茨城"                        
# [49] "群馬"                         "福井"                         "和歌山"                       "沖縄"                        
# [53] "佐賀"                         "島根"                         "岩手"                        


### attenzione che nel 30% dei casi l'informazioni area_pref e' missing in users 
sum(users$PREF_NAME == '') / nrow(users)
##[1] 0.3172299

##### >>> ok, vediamo se gli acquisti fatti da untente (che e' registrato in una certa area_pref coincide con area_pred dell'acquisto)
sales_users = merge(x = users , y = coupon_detail_train , by = 'USER_ID_hash' , all = F)
sales_users_ext = merge(x = sales_users , y = coupon_area_train , by=c('SMALL_AREA_NAME','COUPON_ID_hash') , 
                        all.x = T , all.y = F)

### ci sono 209 casi (=0.1%) di transazioni avvenute al di fuori delle previste aree geografiche 
lapply(sales_users_ext , function(x) sum(is.na(x)))
# $PREF_NAME.y
# [1] 209

clust_user = ddply(sales_users_ext , .(USER_ID_hash) , function(x) c(num_small_area = length(unique(x$SMALL_AREA_NAME)) , 
                                                      num_pref_area = length(unique(x$PREF_NAME.y))))

mean(clust_user$num_small_area , na.rm = T)
# 3.463656

### 
mean(clust_user$num_pref_area , na.rm = T)
# 2.527566 
max(clust_user$num_pref_area , na.rm = T)
# 24 


## nel 68% dei casi area_pref della transazione != area_pref dell'utente 
## puo' significare che i giapponesi viaggiano molto per shopping oppure che la base dati e' sporca o entrambe 
## >> in ogni caso area_pref dell'utente non sembra molto correlato all'area in cui avviene la transazione !! 
sum(sales_users_ext$PREF_NAME.y != sales_users_ext$PREF_NAME.x , na.rm = T)
# 115594 

## vediamo se area_pref di coupon_list e' piu' correlato 
sales_users_ext2 = merge(x = sales_users , y = coupon_list_train , by=c('COUPON_ID_hash') , 
                         all.x = T , all.y = F)


## nel 75% dei casi area_pref dell'utente non ci azzecca una fava con area_pref del coupon (list) laddove sono avvenute transazioni
sum(sales_users_ext2$PREF_NAME != sales_users_ext2$ken_name)
# 128282 

## vediamo se c'e piu' congruenza tra le info legate ai coupon e le transazioni 
sales_coupon = merge(x = coupon_list_train , y = coupon_detail_train , by = 'COUPON_ID_hash' , all = F)

sum(sales_coupon$small_area_name != sales_coupon$SMALL_AREA_NAME) / nrow(sales_coupon)
# 0.4620228


