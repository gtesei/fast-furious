library(quantreg)
library(data.table)
library(glmnet)
library(class)
library(caret)

writeOnDisk = function (base.path , sub.fn, subI) {
  submit <- gzfile(paste(base.path,sub.fn,sep=""), "wt")
  write.table(data.frame(id=subI$id, target=subI$pred), submit, sep=",", row.names=F, quote=F)
  close(submit)
}

doSubmission = function (base.path , sampleSub.fn, octaveSub.fn) {
  sampleSub.tab = fread(paste(base.path,sampleSub.fn,sep="") , header = TRUE , sep=","  )
  octaveSub.tab = fread(paste(base.path,octaveSub.fn,sep="") , header = TRUE , sep=","  )
  octaveSub = as.data.frame(octaveSub.tab)
  colnames(octaveSub) = c("id","pred")
  
  ### merge 
  predI.nona = octaveSub
  mm = mean(octaveSub$pred)
  mg = merge(sampleSub.tab,octaveSub,by=c("id") , all.x = T )
  mg$submission = ifelse( is.na(mg$pred) , mm , mg$pred)  
  sub = as.data.frame(mg)
  sub = sub[,c(1,4)]
  colnames(sub) = c("id","pred")
  writeOnDisk(base.path , paste0(paste0("merged_",octaveSub.fn),".csv.gz") , sub)
}

#base.path = "C:/docs/ff/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"
base.path = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/liberty-mutual-fire-peril/"

sampleSub.fn = "sampleSubmission.csv"
#octaveSubs.fn = c("pred_normal.zat" , "pred_no_skew.zat" );
octaveSubs.fn = c("pred_normal.zat" );

for (i in 1:length(octaveSubs.fn) ) {
  octaveSub.fn = octaveSubs.fn[i]
  cat("processing ",octaveSub.fn," ... \n")
  doSubmission (base.path , sampleSub.fn, octaveSub.fn)
}

  
