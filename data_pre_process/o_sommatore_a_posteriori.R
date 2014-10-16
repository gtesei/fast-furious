library(data.table)

############ da editare 

submission.final.path = "C:/docs/ff/gitHub/fast-furious/dataset/seizure-prediction/sommatora_a_posteriori.zat"

submissions = data.frame(path = c("C:/docs/ff/gitHub/fast-furious/dataset/seizure-prediction/mySub_bayes_calibrat_class.zat" , 
                                  "C:/docs/ff/gitHub/fast-furious/dataset/seizure-prediction/mySub_class.zat" ) , 
                         weigth = c(0.6 , 0.5 )) 

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