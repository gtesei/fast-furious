#' Encode a generic predictor as a categorical features using both observations of train set and test for levels. 
#' It's anyway possible to adopt more levels by using the parameter levels. 
#' Notice that modeling a generic vector, e.g. \code{c(1,2,3,4,5,2,3)} as a categorical predictor xor a numeric predictor is a 
#' modeling choice (eventually to be assessed by cross-validation).
#' 
#' @param data.train the observations of the predictor in train set. 
#' @param data.test the observations of the predictor in test set. 
#' @param colname.prefix the prefix of output data frame. 
#' @param asNumericSequence set \code{T} if the predictor is a numeric sequence filling any possible hole between min and max in observations that could occour both in train set and test set. 
#' @param replaceWhiteSpaceInLevelsWith replace possible spaces in the train/test name of feature. 
#' @param levels the levels of the categorical feature. Must be \code{NULL} if asNumericSequence is \code{T}.  
#' 
#' @examples
#' Xtrain <- data.frame( a = rep(1:3 , each = 2), b = 6:1, c = letters[1:6])
#' Xtest <- data.frame( a = rep(2:4 , each = 2), b = 1:6, c = letters[6:1])
#' print(Xtrain)
#' #   a b c
#' # 1 1 6 a
#' # 2 1 5 b
#' # 3 2 4 c
#' # 4 2 3 d
#' # 5 3 2 e
#' # 6 3 1 f
#' 
#' l = ff.encodeCategoricalFeature (Xtrain$c , Xtest$c , "c")
#' l$traindata
#' #     c_1 c_2 c_3 c_4 c_5 c_6
#' # 7    1   0   0   0   0   0
#' # 8    0   1   0   0   0   0
#' # 9    0   0   1   0   0   0
#' # 10   0   0   0   1   0   0
#' # 11   0   0   0   0   1   0
#' # 12   0   0   0   0   0   1
#' 
#' Xtrain[,'c'] = NULL
#' Xtest[,'c'] = NULL
#' Xtrain = cbind(Xtrain,l$traindata)
#' Xtest = cbind(Xtest,l$testdata)
#' @export
#' 
ff.encodeCategoricalFeature = function(data.train , 
                                    data.test , 
                                    colname.prefix, 
                                    asNumericSequence=F , 
                                    replaceWhiteSpaceInLevelsWith=NULL,
                                    levels = NULL) {
  
  stopifnot(is.atomic(data.train))
  stopifnot(is.atomic(data.test))
  
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
    
    if(is.null(levels)) facts = sort(unique(data))
    else facts = levels 
    
    colns = facts
    
    if (! is.null(replaceWhiteSpaceInLevelsWith) ) 
      colns = gsub(" ", replaceWhiteSpaceInLevelsWith , sort(unique(data)))
  }
  
  mm = outer(data,facts,function(x,y) ifelse(x==y,1,0))
  colnames(mm) = paste(colname.prefix,"_",colns,sep='')  
  
  ##
  mm = as.data.frame(mm)
  
  ## reassembling 
  testdata = mm[1:(length(data.test)),]
  traindata = mm[((length(data.test))+1):(dim(mm)[1]),]
  
  return(list(traindata = traindata ,testdata = testdata))
}

#' Extracts a numerical feature from a date predictor. The feature is built as the difference in days from 
#' the oldest date in bothe train set and test set and any given observation.  
#' 
#' @param data.train the observations of the predictor in train set. 
#' @param data.test the observations of the predictor in test set. 
#' 
#' @examples
#' Xtrain <- data.frame( a = rep(1:3 , each = 2), b = 6:1, 
#'    c = rep(as.Date(c("2007-06-22", "2004-02-13")),3) )
#' Xtest <- data.frame( a = rep(2:4 , each = 2), b = 1:6, 
#'    c = rep(as.Date(c("2007-03-01", "2004-05-23")),3) )
#' l = ff.extractDateFeature(Xtrain$c,Xtest$c)
#' Xtrain[,'c'] = NULL
#' Xtest[,'c'] = NULL
#' Xtrain = cbind(Xtrain,c=l$traindata)
#' Xtest = cbind(Xtest,c=l$testdata)
#' @export
#' 
ff.extractDateFeature = function(data.train , 
                               data.test) {
  
  stopifnot(identical(class(data.train),'Date') )
  stopifnot(identical(class(data.test),'Date'))
  
  all_date = as.Date(c(data.train , data.test))
  all_date = sort(all_date)
  
  all_date_uniq = as.Date(unique(c(data.train , data.test)))
  all_date_uniq = sort(all_date_uniq)
  
  drange = all_date_uniq[length(all_date_uniq)] - all_date_uniq[1]
  
  traindata = as.numeric(as.Date(data.train) - rep(all_date_uniq[1] , length(data.train))) 
  testdata = as.numeric(as.Date(data.test) - rep(all_date_uniq[1] , length(data.test))) 
  
  return(list(traindata = traindata ,testdata = testdata,drange = drange))
}