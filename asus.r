
encodeSaleTrain = function(x) {
  mods = as.numeric(lapply( strsplit(as.character(saleTrain$module_category) ,"M" )  , function(x) as.numeric(x[2]) ))
  modMatrix = matrix(rep(0,dim(x)[1]*9) , nrow = dim(x)[1] , ncol = 9)  
  for (i in 1:length(mods)) {
     modMatrix[i,mods[i]] = 1
  }
  modDF = data.frame(modMatrix)
  colnames(modDF) = paste("M",1:9,sep='')

  comps = as.numeric(lapply( strsplit(as.character(saleTrain$module_category) ,"P" )  , function(x) as.numeric(x[2]) ))
  comMatrix = matrix(rep(0,dim(x)[1]*31) , nrow = dim(x)[1] , ncol = 31)  
  for (i in 1:length(comps)) {
     comMatrix [i,comps[i]] = 1
  }
  comDF = data.frame(comMatrix) 
  colnames(comDF) = paste("P",1:31,sep='')

  ret = x[,-(1:2)]
  ret = data.frame(modDF,comDF,ret)
}

repTrainFn = "dataset/pakdd-cup-2014/RepairTrain.csv"
saleTrainFn = "dataset/pakdd-cup-2014/SaleTrain.csv"
outTargetIdMapFn = "dataset/pakdd-cup-2014/Output_TargetID_Mapping.csv"
#sampleSubFn = "dataset/pakdd-cup-2014/SampleSubmission.csv"

repTrain = read.csv(repTrainFn) 
saleTrain = read.csv(saleTrainFn)
outTargetIdMap = read.csv(outTargetIdMapFn)
#sampleSub = read.csv(sampleSubFn)
