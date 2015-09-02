# Analyzing collateral effects of bad resampling  

# library required
library(devtools)
devtools::install_github('gtesei/fast-furious', subdir='R-package')
library(fastfurious)

library(plyr)
library(data.table)
library(lattice)
library(caret)

### FUNCS

RMSLE = function(pred, obs) {
  if (sum(pred<0)>0) {
    pred = ifelse(pred >=0 , pred , 1.5)
  }
  rmsle = sqrt(    sum( (log(pred+1) - log(obs+1))^2 )   / length(pred))
  return (rmsle)
}

RMSLECostSummary <- function (data, lev = NULL, model = NULL) {
  c(postResample(data[, "pred"], data[, "obs"]),
    RMSLE = RMSLE(pred = data[, "pred"], obs = data[, "obs"]))
}

makeResampleIndexSameTubeAssemblyId = function (foldList) {
  
  stopifnot(! is.null(foldList) && length(foldList)>0)
  
  nFolds = unique(sort(foldList[[1]]))
  
  ## par out 
  ## e.g. Fold1.Rep1 - ResFold1.Rep1 / .. /  Fold5.Rep3 - ResFold5.Rep3
  index = list()
  indexOut = list()
  
  ##
  for (i in seq_along(foldList)) {
    for (j in seq_along(nFolds) ) {
      iF = which(foldList[[i]] != j) 
      iR = which(foldList[[i]] == j) 
      
      ## partition checks 
      stopifnot(length(intersect(iF,iR)) == 0)
      stopifnot(identical(seq_along(foldList[[i]]),sort(union(iF,iR))))
      
      index[[paste('Fold',j,'.Rep',i,sep='')]] = iF
      indexOut[[paste('ResFold',j,'.Rep',i,sep='')]]  = iR
    }
  }
  
  ## 
  return(list(
    index = index , 
    indexOut = indexOut
  )) 
}

createFoldsSameTubeAssemblyId = function (data,
                                          nFolds = 8, 
                                          repeats = 3, 
                                          seeds) {  
  ## par out 
  foldList = list()
  for (i in 1:repeats) {
    folds_i_name = paste0('folds.',i)
    foldList[[folds_i_name]] = rep(NA_integer_,nrow(data)) 
  }
  
  ## 
  clusters = ddply(data , .(tube_assembly_id) , function(x) c(num = nrow(x)))
  clusters = clusters[order(clusters$num , decreasing = T),]
  stopifnot(sum(clusters$num)==nrow(data)) 
  
  for (j in 1:repeats) {
    folds = list()
    for (i in 1:nFolds) {
      folds_i_name = paste0('Fold',i)
      folds[[folds_i_name]] = rep(NA_character_,nrow(data)) 
    }
    
    idx = 1 
    
    ##
    if (! is.null(seeds)) set.seed(seeds[j])
    seq = sample(1:nFolds)
    
    while (idx<=nrow(clusters)) {
      ## fw
      for (k in 1:length(seq)) {
        folds_k_name = paste0('Fold',seq[k])
        idx_k = min(which(is.na(folds[[folds_k_name]])))
        folds[[folds_k_name]][idx_k] = clusters[idx,'tube_assembly_id'] 
        idx = idx + 1 
        if (idx > nrow(clusters)) break 
      }
      
      if (idx > nrow(clusters)) break 
      
      ## bw 
      for (k in length(seq):1) {
        folds_k_name = paste0('Fold',seq[k])
        idx_k = min(which(is.na(folds[[folds_k_name]])))
        folds[[folds_k_name]][idx_k] = clusters[idx,'tube_assembly_id'] 
        idx = idx + 1 
        if (idx > nrow(clusters)) break 
      }
    }
    
    ## remove NAs and convert to chars 
    for (k in seq_along(seq)) folds[[k]] = as.character(na.omit(folds[[k]]))
    
    ## union check 
    stopifnot(identical(intersect(clusters$tube_assembly_id , Reduce(union , folds) ) , clusters$tube_assembly_id)) 
    
    ## intersect check 
    samp = sample(1:nFolds,2,replace = F)
    stopifnot(length(intersect( folds[[samp[1]]] , folds[[samp[2]]]))==0) 
    
    ## refill nFolds 
    folds_j_name = paste0('folds.',j)
    for (k in 1:nFolds) {
      idx_k = which( data$tube_assembly_id %in% folds[[k]] )
      foldList[[folds_j_name]] [idx_k] = k
    }
    
    ## checks
    stopifnot(sum( is.na(foldList[[folds_j_name]]) ) == 0) 
    stopifnot( length(foldList[[folds_j_name]]) == nrow(data) ) 
    stopifnot(identical(intersect(unique(sort(foldList[[folds_j_name]])) , 1:nFolds), 1:nFolds))
  }
  
  return(foldList)
}


## Importing data into R
train  <- read.csv("../input/train_set.csv",header = T)


train$tube_assembly_id = as.character(train$tube_assembly_id)
train$supplier = as.character(train$supplier)
train$quote_date = as.Date(train$quote_date)
train$bracket_pricing = as.character(train$bracket_pricing)
y = train[,'cost']
train[,'cost'] = NULL

featureSet = ff.makeFeatureSet (data.train = train[,-1], 
                             data.test = train[,-1], 
                             meta =c('C','D','N','N','C','N'),
                             scaleNumericFeatures = TRUE,
                             parallelize = FALSE)$traindata


## bad model 
controlObject <- trainControl(method = "repeatedcv", repeats = 2, number = 4 , summaryFunction = RMSLECostSummary )
.tuneGrid = expand.grid(.ncomp = 1:10)

model_bad <- train(y = y, x = featureSet ,
               method = "pls",
               tuneGrid = .tuneGrid , 
               trControl = controlObject,
               metric = 'RMSLE' , maximize = F)

cat("***** BAD MODEL *****\n")
print(model_bad)

## good model 
foldList = createFoldsSameTubeAssemblyId(data = train, nFolds = 4, repeats = 2, seeds=c(123,456))
resamples = makeResampleIndexSameTubeAssemblyId(foldList)
controlObject.good <- trainControl(method = "cv", 
                                   ## The method doesn't really matter
                                   ## since we defined the resamples
                                   index = resamples$index, 
                                   indexOut = resamples$indexOut, 
                                   summaryFunction = RMSLECostSummary)

model_good <- train(y = y, x = featureSet ,
                   method = "pls",
                   tuneGrid = .tuneGrid , 
                   trControl = controlObject.good, 
                   metric = 'RMSLE' , maximize = F)

cat("***** GOOD MODEL *****\n")
print(model_good)

allResamples <- resamples(list("Bad Pls" = model_bad, "Good Pls" = model_good)) 
dotplot(allResamples, metric = "RMSLE")

t.test(x = model_bad$results$RMSLE , y = model_good$results$RMSLE , paired = T)

# 95 percent confidence interval:
#   -0.005646021  0.004657906
                               



