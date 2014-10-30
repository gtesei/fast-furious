## notes_seizure_R
library(caret)
library(Hmisc)
library(data.table)
library(verification)
library(pROC)
library(kernlab)
library(subselect)

getBasePath = function (type = "data" , ds="") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/seizure-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/seizure-prediction/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/data_pre_process"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/data_pre_process/"
  } else {
    stop("unrecognized type.")
  }
  
  if (file.exists(base.path1))  {
    ret = paste0(base.path1,"/")
  } else {
    ret = base.path2
  }
  
  if (ds != "" ) {
    ret = paste0(paste0(ret,ds),"_digest_4gen/")
  }
  ret
} 
  
buildRelativeSpectralPowerFeatures = function(Xtrain_mean_sd,Xtest_mean_sd,Xtrain_quant,Xtest_quant,ytrain,verbose) {
  ########
  #   1- Xtrain_mean_sd(train_index,((i-1)*16+1)) = mu;
  #   2 - Xtrain_mean_sd(train_index,((i-1)*16+2)) = sd;
  #   3 - Xtrain_mean_sd(train_index,((i-1)*16+3)) = fm;
  #   4 - Xtrain_mean_sd(train_index,((i-1)*16+4)) = P;
  
  #   5 - Xtrain_mean_sd(train_index,((i-1)*16+5)) = p0; 
  #   6 - Xtrain_mean_sd(train_index,((i-1)*16+6)) = p1;
  #   7 - Xtrain_mean_sd(train_index,((i-1)*16+7)) = p2;
  #   8 - Xtrain_mean_sd(train_index,((i-1)*16+8)) = p3;
  #   9 - Xtrain_mean_sd(train_index,((i-1)*16+9)) = p4;
  #   10 - Xtrain_mean_sd(train_index,((i-1)*16+10)) = p5;
  #   11 - Xtrain_mean_sd(train_index,((i-1)*16+11)) = p6;
  #   12 - Xtrain_mean_sd(train_index,((i-1)*16+12)) = pTail; 
  
  #   13 - Xtrain_mean_sd(train_index,((i-1)*16+13)) = f_50;
  #   14 - Xtrain_mean_sd(train_index,((i-1)*16+14)) = min_tau;
  #   15 - Xtrain_mean_sd(train_index,((i-1)*16+15)) = skw;
  #   16 - Xtrain_mean_sd(train_index,((i-1)*16+16)) = kur;
  
  P.mean = apply(Xtrain_mean_sd,2,mean)
  idx.preict = (ytrain[,2] == 1)
  P.mean.inter = apply(Xtrain_mean_sd[! idx.preict , ],2,mean)
  P.mean.preict = apply(Xtrain_mean_sd[idx.preict , ],2,mean)
  delta.mean =  (P.mean.inter -  P.mean.preict) 
  idx.delta.mean.asc = order(delta.mean)
  idx.delta.mean.desc = order(delta.mean, decreasing = T)
  
  ## solo le features di interesse 
  idx.filter = idx.delta.mean.asc %% 16 == 5 | idx.delta.mean.asc %% 16 == 6 | idx.delta.mean.asc %% 16 == 7 |
    idx.delta.mean.asc %% 16 == 8 | idx.delta.mean.asc %% 16 == 9 | idx.delta.mean.asc %% 16 == 10 |
    idx.delta.mean.asc %% 16 == 11 | idx.delta.mean.asc %% 16 == 12 
  
  idx.delta.mean.asc = idx.delta.mean.asc[idx.filter]
  idx.delta.mean.desc = idx.delta.mean.desc[idx.filter]
  
  
  
  ## prendo le prime 10 
  NN = 10 
  idx.delta.mean.asc = idx.delta.mean.asc[1:NN]
  idx.delta.mean.desc = idx.delta.mean.desc[1:NN]
  
  cat("************* Indici \n")
  print(idx.delta.mean.desc)
  print(idx.delta.mean.asc)
  cat("************* Delta \n")
  print(delta.mean[idx.delta.mean.desc])
  print(delta.mean[idx.delta.mean.asc])
  
  ## fill train 
  feat.mat = matrix(rep(-1,nrow(Xtrain_mean_sd)*NN*NN),nrow = nrow(Xtrain_mean_sd) , ncol = NN*NN)
  for (idx.num in 1:NN) {
    for (idx.denum in 1:NN) {
      feat.mat[,idx.denum+10*(idx.num-1)] =  (Xtrain_mean_sd[,idx.delta.mean.asc[idx.num]] / mean(Xtrain_mean_sd[,idx.delta.mean.asc[idx.num]])) / (Xtrain_mean_sd[,idx.delta.mean.desc[idx.denum]]/mean(Xtrain_mean_sd[,idx.delta.mean.desc[idx.denum]]))
    }
  }
  
  new.features.train = as.data.frame(feat.mat) 
  colnames(new.features.train) = paste("relspect",rep(1:(ncol(feat.mat))) , sep = "")
  
  if (verbose) {
    for (yy in 1:10) {
      cat("----------------------------------------------------------------- \n")
      cat("-- [train] mean of relspect feature n. ", yy , "=",as.character(mean (new.features.train[,yy]))  , "\n")
      cat("-- [train] mean of relspect feature (preict) n. ", yy  ,"=",as.character(mean (new.features.train[idx.preict,yy])) , "\n")
      cat("-- [train] mean of relspect feature (interict) n. ", yy  ,"=",as.character(mean (new.features.train[!idx.preict,yy])) , "\n")
    }
  }
  
  ## fill test 
  feat.mat = matrix(rep(-1,nrow(Xtest_mean_sd)*NN*NN),nrow = nrow(x = Xtest_mean_sd) , ncol = NN*NN)
  for (idx.num in 1:NN) {
    for (idx.denum in 1:NN) {
      feat.mat[,idx.denum+10*(idx.num-1)] =  (Xtest_mean_sd[,idx.delta.mean.asc[idx.num]]/mean(Xtest_mean_sd[,idx.delta.mean.asc[idx.num]])) / (Xtest_mean_sd[,idx.delta.mean.desc[idx.denum]]/mean(Xtest_mean_sd[,idx.delta.mean.desc[idx.denum]]))
    }
  }
  
  new.features.test = as.data.frame(feat.mat) 
  colnames(new.features.test) = paste("relspect",rep(1:(ncol(feat.mat))) , sep = "")
  
  ###### merge 
  Xtrain_mean_sd = cbind(Xtrain_mean_sd,new.features.train)
  Xtest_mean_sd = cbind(Xtest_mean_sd , new.features.test)
  
  Xtrain_quant = cbind(Xtrain_quant,new.features.train)
  Xtest_quant = cbind(Xtest_quant,new.features.test)
  
  return( list(Xtrain_mean_sd,Xtest_mean_sd,Xtrain_quant,Xtest_quant) )
} 

############# 
verbose = T
doPlot = F 

##dss = c("Dog_1","Dog_2","Dog_3","Dog_4","Dog_5","Patient_1","Patient_2")
dss = c("Patient_1")
cat("|---------------->>> data set to process: <<",dss,">> ..\n")

for (ds in dss) {
  
  cat("|---------------->>> processing data set <<",ds,">> ..\n")
  
  ######### loading data sets ...
  Xtrain_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"Xtrain_mean_sd.zat",sep="") , header = F , sep=","  ))
  Xtrain_quant = as.data.frame( fread(paste(getBasePath(type = "data" , ds=ds),"Xtrain_quant.zat",sep="") , header = F , sep=","  ))
  
  Xtest_mean_sd = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"Xtest_mean_sd.zat",sep="") , header = F , sep=","  ))
  Xtest_quant = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"Xtest_quant.zat",sep="") , header = F , sep=","  ))
  
  ytrain = as.data.frame(fread(paste(getBasePath(type = "data" , ds=ds),"ytrain.zat",sep="") , header = F , sep=","  ))
  
  ## names 
  colnames(Xtrain_mean_sd)  = colnames(Xtest_mean_sd) = paste("fmeansd",rep(1:((dim(Xtrain_mean_sd)[2]))) , sep = "")
  colnames(Xtrain_quant) = colnames(Xtest_quant) = paste("fquant",rep(1:((dim(Xtrain_quant)[2]))) , sep = "")
  
  ytrain.cat = factor(ytrain[,2]) 
  levels(ytrain.cat) = c("interict","preict")
  
#   l = buildRelativeSpectralPowerFeatures (Xtrain_mean_sd,Xtest_mean_sd,Xtrain_quant,Xtest_quant,ytrain,verbose)
#   Xtrain_mean_sd = l[[1]]
#   Xtest_mean_sd = l[[2]]
#   Xtrain_quant = l[[3]]
#   Xtest_quant = l[[4]]
}