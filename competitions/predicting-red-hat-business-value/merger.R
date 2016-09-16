library(data.table)
library(dplyr)
library(Matrix)

cat(Sys.time())
cat("Reading data\n")   

prefix = 'C:/Users/gtesei/Desktop/Deloitte/C_Folder/Cognitive_Technologies/Machine_Learning/git/fast-furious/dataset/predicting-red-hat-business-value/'
sub1 <- fread(paste0(prefix,"mod3108Kaggle_01.csv"), header=TRUE)
sub2 <- fread(paste0(prefix,"bech___mod3108Kaggle_01.csv"), header=TRUE)

sub2$outcome = 0.6*sub1$outcome+0.4*sub2$outcome+0.1

write.csv(sub2, file=paste0(prefix,"merge.csv"), row.names=FALSE)


