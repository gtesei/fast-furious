############## functions 

writeOnDisk = function (base.path , sub.fn, subI) {
  submit <- gzfile(paste(base.path,sub.fn,sep=""), "wt")
  write.table(data.frame(id=subI$id, target=subI$pred_wmean), submit, sep=",", row.names=F, quote=F)
  close(submit)
} 

##############  loading predictions 

#base.path = "C:/docs/ff/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"
base.path = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"

#pred_vect = c("sub_knn.csv", "sub_pls.csv" , "sub_my_ridge.csv" )
#score_vect = c(0.32078 , 0.32078 , 0.31804 )
pred_vect = c("sub_knn.csv", "sub_my_ridge.csv"  )
score_vect = c(0.32078 , 0.31804  )
score_coeff = score_vect / sum(score_vect)

subs = vector("list", length(pred_vect) )

for (i in 1:length(pred_vect)) {
  subs[[i]] = read.csv(paste0(base.path,pred_vect[i]))
}

## checking same order of ids ... 
for (i in 1:length(pred_vect)) {
  for (j in i:length(pred_vect)) {
    diffIds = sum((subs[[i]]$id  - subs[[j]]$id)^2)
    cat("processed i = ",i," , j = ", j," ---> diffIds = ",diffIds,"\n")
    if (diffIds > 0) stop("different order of ids in prediction data!")
  }
}

# merging predictions .. 
sub_df = data.frame(id = subs[[1]]$id)
for (i in 1:length(pred_vect)){
  sub_df = cbind(sub_df , cname = subs[[i]]$target)
}
names(sub_df) = c("id",pred_vect)

sub_df$pred_wmean = 0
for (i in 1:length(pred_vect)) {
  sub_df$pred_wmean = sub_df$pred_wmean + (   score_coeff[i] * sub_df[,(i+1)]    ) 
}

# serializing boosted prediction 
writeOnDisk (base.path , "sub_boost.csv.gz", sub_df) 






