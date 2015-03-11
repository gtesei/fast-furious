library(data.table)

############ da editare 

submission.final.path = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/submission_supervised/ward_svm_avg2.csv"

submissions = data.frame(path = 
                           c(
                             
                             "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/submission_supervised/ward_svm_avg.csv", 
                             "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/axa-driver-telematics-analysis/submission/features_red__ward_average_submission.csv") , 
                         weigth = c(  0.52 , 0.52915)) 

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