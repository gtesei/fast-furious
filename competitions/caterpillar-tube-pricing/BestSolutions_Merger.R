library(fastfurious)
library(data.table)

ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'submission' , sub_path = 'dataset/caterpillar-tube-pricing/pred_ensemble_1') ## out 


##### Best solutions 
sub_cubist = as.data.frame( fread(paste(ff.getPath("submission") , 
                                                 "8_cubist_useQty.csv" , sep=''))) 

sub_knn = as.data.frame( fread(paste(ff.getPath("submission") , 
                                        "1_knn_useQty.csv" , sep=''))) 

sub_xgb = as.data.frame( fread(paste(ff.getPath("submission") , 
                                     "4_xgbTree_useQty.csv" , sep=''))) 

sub_best = sub_cubist

sub_best$cost = (1/4) * sub_cubist$cost + (1/2) * sub_knn$cost + (1/4) * sub_xgb$cost
write.csv(sub_best,
          quote=FALSE, 
          file=paste(ff.getPath("submission"), "merge_knn_cubist_xgb.csv" ,sep='') ,
          row.names=FALSE)


##### Other solutions 
