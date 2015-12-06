

ff.predNA = function(data,asIndex=TRUE) {
  stopifnot(identical(class(data),"data.frame"))
  feature.names <- colnames(data)
  predNA = unlist(lapply(1:length(feature.names) , function(i) {
    sum(is.na(data[,i]))>0 
  }))  
  if (asIndex) return(predNA)
  else return(feature.names[predNA])
}

ff.obsNA = function(data) {
  stopifnot(identical(class(data),"data.frame"))
  obsNAs =  which(is.na(data)) %% nrow(data) 
  return(obsNAs)
}