

verify.kfolds = function(k,folds,dataset,event.level) {
  correct = T
  for(j in 1:k) {  
    ## y 
    dataset.train = dataset[folds != j]
    dataset.xval = dataset[folds == j]
    
    if (sum(dataset.train == event.level ) == 0 | 
          sum(dataset.xval == event.level ) == 0  ) {
      correct = F
      break
    }
  }
  return(correct)
}

kfolds = function(k,data.length) {
  k = min(k,data.length)
  folds = rep(NA,data.length)
  labels = 1:data.length
  st = floor(data.length/k)
  al_labels = NULL
  for (s in 1:k) {
    x = NULL
    if (is.null(al_labels))
      x = sample(labels,st)
    else
      x = sample(labels[-al_labels],st)
    
    folds[x] = s
    if (is.null(al_labels))
      al_labels = x
    else
      al_labels = c(al_labels,x)
  }
  ss = 1
  for (s in 1:length(folds)){
    if (is.na(folds[s])) {
      folds[s] = ss
      ss = ss + 1
    } 
  }
  folds
}