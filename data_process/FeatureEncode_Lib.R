


encodeCategoricalFeature = function(data.train , data.test , colname.prefix, 
                                    asNumericSequence=F , 
                                    replaceWhiteSpaceInLevelsWith=NULL,
                                    levels = NULL) {
  ### assembling 
  data = c(data.test , data.train)
  
  ###
  fact_min = 1 
  fact_max = -1
  facts = NULL
  if (asNumericSequence) {
    if (! is.null(levels))
      stop("levels must bel NULL if you set up asNumericSequence to true.")
    fact_max = max(unique(data))
    fact_min = min(unique(data))
    facts = fact_min:fact_max
  } else {
    facts = NULL
    
    if(is.null(levels)) facts = sort(unique(data))
    else facts = levels 
    
    colns = facts
    
    if (! is.null(replaceWhiteSpaceInLevelsWith) ) 
      colns = gsub(" ", replaceWhiteSpaceInLevelsWith , sort(unique(data)))
  }
  
  ##
#   mm = matrix(rep(0,length(data)),nrow=length(data),ncol=length(facts))
#   colnames(mm) = paste(paste(colname.prefix,"_",sep=''),    colns   ,sep='')
  
  ##
#   for ( j in 1:length(facts) ) 
#     for (i in 1:length(data))
#       mm[i,j] = (data[i] == facts[j])

mm = outer(data,facts,function(x,y) ifelse(x==y,1,0))
colnames(mm) = paste(paste(colname.prefix,"_",sep=''),    colns   ,sep='')  
   
  
  ##
  mm = as.data.frame(mm)
  
  ## reassembling 
  testdata = mm[1:(length(data.test)),]
  traindata = mm[((length(data.test))+1):(dim(mm)[1]),]
  
  list(traindata,testdata)
}