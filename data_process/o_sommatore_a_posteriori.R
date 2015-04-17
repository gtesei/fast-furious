library(data.table)

model.average = function (submissions = data.frame(path = 
                                c("/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/seizure-prediction/merge/IN1.zat", 
                                  "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/seizure-prediction/merge/IN2.zat") , 
                                weigth = c(  0.83013 , 0.83542)) , 
                          sub.col = 2 , 
                          submission.final.path = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/seizure-prediction/merge/sommatora_a_posteriori.zat"
                          ) {
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
}