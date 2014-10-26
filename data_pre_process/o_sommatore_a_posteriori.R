library(data.table)

############ da editare 

submission.final.path = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/seizure-prediction/merge/sommatora_a_posteriori.zat"

submissions = data.frame(path = c("/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/seizure-prediction/merge/sub66.zat" , 
                                  "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/seizure-prediction/merge/mySub_class.zat" ) , 
                        weigth = c(0.66174 , 0.65629 )) 

sub.col = 2 

######## fine parte da editare 
sub = NULL
sub.df = NULL 
for (i in 1:nrow(submissions)) {
  sub.t = as.data.frame(fread(paste(submissions[i,]$path,sep="") , header = T , sep=","  ))
  if (is.null(sub)) {
    sub = rep(0,nrow(sub.t))
    sub.df = sub.t
  }
  sub = sub + (sub.t[,sub.col] * submissions[i,]$weigth / sum(submissions$weigth) )
}

sub.df[,sub.col] = sub
write.csv(sub.df,quote=FALSE,file=submission.final.path, row.names=FALSE)