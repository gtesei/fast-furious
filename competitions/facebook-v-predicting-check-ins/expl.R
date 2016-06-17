library(data.table)
library(Hmisc)

prefix = "C:/Users/gtesei/Desktop/Deloitte/C_Folder/Cognitive_Technologies/Machine_Learning//git/fast-furious/"
data = paste0(prefix , 'dataset/facebook-v-predicting-check-ins/')


test_data = as.data.frame( fread(paste0(data,"test.csv") , sep=',') , stringsAsFactors = F)
train_data = as.data.frame( fread(paste0(data,"train.csv") , sep=',') , stringsAsFactors = F)
sample_submission = as.data.frame( fread(paste0(data,"sample_submission.csv") , sep=',') , stringsAsFactors = F)


describe(train_data)

