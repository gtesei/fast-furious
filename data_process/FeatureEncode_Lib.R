


encodeCategoricalFeature = function(data.train , data.test , colname.prefix, asNumeric=T) {
  ### assembling 
  data = c(data.test , data.train)
  
  ###
  fact_min = 1 
  fact_max = -1
  facts = NULL
  if (asNumeric) {
    fact_max = max(unique(data))
    fact_min = min(unique(data))
    facts = fact_min:fact_max
  } else {
    facts = sort(unique(data))
  }
  
  ##
  mm = matrix(rep(0,length(data)),nrow=length(data),ncol=length(facts))
  colnames(mm) = paste(paste(colname.prefix,"_",sep=''),    facts   ,sep='')
  
  ##
  for ( j in 1:length(facts) ) 
    for (i in 1:length(data))
      mm[i,j] = (data[i] == facts[j])
  
  ##
  mm = as.data.frame(mm)
  
  ## reassembling 
  testdata = mm[1:(length(data.test)),]
  traindata = mm[((length(data.test))+1):(dim(mm)[1]),]
  
  list(traindata,testdata)
}