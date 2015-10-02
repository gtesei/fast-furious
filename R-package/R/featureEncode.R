
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
#' @param remove1DummyVar \code{T} to remove one dummy variable. Why? 
#' First, if you know the values of the first C - 1 dummy variables, you know the last one too and it is more economical to use C - 1. 
#' Secondly, if the model has slopes and intercepts (e.g. linear regression), the sum of all of the dummy variables wil add up to the 
#' intercept (usually encoded as a "1") and that is bad for the math involved. On the other hand, there are models like penalized methods (such as ridge regression) 
#' that seldom penalize the intercept, so a C-1 encoded variable could cause the other category effects to be penalized towards the reference category effect.
#' @references \url{http://appliedpredictivemodeling.com/blog/2013/10/23/the-basics-of-encoding-categorical-data-for-predictive-models}
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
#' @return the list of trainset and testset after applying the specified filters 
#' 
ff.encodeCategoricalFeature = function(data.train , 
                                    data.test , 
                                    colname.prefix, 
                                    asNumericSequence=F , 
                                    replaceWhiteSpaceInLevelsWith=NULL,
                                    levels = NULL, 
                                    remove1DummyVar=FALSE) {
  
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
    
    if (! is.null(replaceWhiteSpaceInLevelsWith) ) 
      colns = gsub(" ", replaceWhiteSpaceInLevelsWith , sort(unique(data)))
  }
  
  colns = facts
  mm = outer(data,facts,function(x,y) ifelse(x==y,1,0))
  colnames(mm) = paste(colname.prefix,"_",colns,sep='')  
  
  ##
  mm = as.data.frame(mm)
  if (remove1DummyVar) {
    mm = mm[,-1,drop=F]
  }
  
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
#' @return the list of trainset and testset after applying the specified encoding and the related date range 
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


#' Encode the feature set according to meta data passed as input.   
#' 
#' @param data.train the observations of the predictor in train set. 
#' @param data.test the observations of the predictor in test set. 
#' @param meta the meata data. It should be a vector of the character \code{'C'} , \code{'N'} , \code{'D'} , 
#'        e.g. \code{c('N','C','D')} of the same length of the train set / test set columns 
#' @param scaleNumericFeatures seto to \code{'TRUE'} to center and scale numeric features 
#' @param parallelize set to \code{'TRUE'} to enable parallelization (require \code{parallel} package)  
#' @param remove1DummyVarInCatPreds \code{T} to remove one dummy variable in encoding categorical predictors. 
#' For further details see \code{\link[fastfurious]{ff.encodeCategoricalFeature}}.
#' 
#' @examples
#' Xtrain <- data.frame( a = rep(1:3 , each = 2), b = 6:1, 
#'    c = rep(as.Date(c("2007-06-22", "2004-02-13")),3) )
#' Xtest <- data.frame( a = rep(2:4 , each = 2), b = 1:6, 
#'    c = rep(as.Date(c("2007-03-01", "2004-05-23")),3) )
#' l = ff.makeFeatureSet(Xtrain,Xtest,c('C','N','D'))
#' Xtrain = l$traindata
#' Xtest = l$testdata
#' @importFrom caret preProcess
#' @importFrom parallel mcMap
#' @export
#' @return the list of trainset and testset after applying the specified encodings 
#' 
ff.makeFeatureSet = function(data.train , 
                             data.test, 
                             meta,
                             scaleNumericFeatures = FALSE,
                             parallelize = FALSE,
                             remove1DummyVarInCatPreds=FALSE) { 

  ##
  stopifnot(  ! (is.null(data.train) || is.null(data.test)) )
  
  stopifnot(  (length(meta) != nrow(data.train)) )
  stopifnot(  (length(meta) != nrow(data.test)) )
  
  stopifnot(  sum(unlist(lapply(unique(meta),function(x) {
    return(!x %in% c("C","N","D"))
  }))) == 0 )
  
  stopifnot(  sum(unlist(lapply(data.train,function(x) {
    return(! (is.atomic(x) || identical(class(x),'Date')) )
  }))) == 0 )
  stopifnot(  sum(unlist(lapply(data.test,function(x) {
    return(! (is.atomic(x) || identical(class(x),'Date')) )
  }))) == 0 )
  
  stopifnot(  sum(unlist(Map(function(x,y) {
   if (identical(y,'D')) return(!identical(class(x),'Date'))
   else return(!is.atomic(x))
  } , data.train , meta)))==0 )
  stopifnot(  sum(unlist(Map(function(x,y) {
    if (identical(y,'D')) return(!identical(class(x),'Date'))
    else return(!is.atomic(x))
  } , data.test , meta)))==0 )
  
  
  ##
  doEncoding = function(x,y,m,nx,ny) {
    if (identical(m,'D')) {
      ll = ff.extractDateFeature(x,y)
      ll['x.name'] = nx 
      ll['y.name'] = ny
      ll['dim'] = 1
      return(ll)
    } else if (identical(m,'C')) {
      ll = ff.encodeCategoricalFeature (data.train = x , data.test = y , colname.prefix = nx, remove1DummyVar = remove1DummyVarInCatPreds)
      ll['dim'] = ncol(ll$traindata)
      return(ll)
    } else if (identical(m,'N')) {
      ll = list(traindata=x,testdata=y)
      ll['x.name'] = nx 
      ll['y.name'] = ny
      ll['dim'] = 1
      return(ll)
    } else stop('unrecognized type of meta-data')
  }
  
  l = NULL
  if (parallelize) { 
    l = parallel::mcMap( doEncoding , data.train , data.test, meta,colnames(data.train),colnames(data.test) , 
               mc.cores = min(colnames(data.test),ff.getMaxCuncurrentThreads()) )
  } else {
    l = Map( doEncoding , data.train , data.test, meta,colnames(data.train),colnames(data.test))
  }
  
  ##
  ncols = sum(unlist(lapply(l,function(x) return(x$dim))))
  traindata = as.data.frame(matrix(rep(NA,ncols*nrow(data.train)),ncol = ncols))
  testdata = as.data.frame(matrix(rep(NA,ncols*nrow(data.test)),ncol = ncols))
  
  #
  currIdx = 1 
  lapply(seq_along(l) , function(i) {
    if (identical(meta[i],'C')) {
      
      traindata[,currIdx:(currIdx+l[[i]]$dim-1)]  <<- l[[i]]$traindata
      colnames(traindata)[currIdx:(currIdx+l[[i]]$dim-1)] <<- colnames(l[[i]]$traindata)
      testdata[,currIdx:(currIdx+l[[i]]$dim-1)]  <<- l[[i]]$testdata
      colnames(testdata)[currIdx:(currIdx+l[[i]]$dim-1)] <<- colnames(l[[i]]$testdata)
      
    } else if ( identical(meta[i],'N') || identical(meta[i],'D') ) {
      trdata = l[[i]]$traindata
      tsdata = l[[i]]$testdata
      
      if (scaleNumericFeatures) {
        data = as.data.frame(c(trdata,tsdata))
        scaler = caret::preProcess(data,method = c("center","scale"))
        data = predict(scaler,data)
        trdata = data[1:length(trdata),]
        tsdata = data[(length(trdata)+1):nrow(data),]
      }
      
      traindata[,currIdx:(currIdx+l[[i]]$dim-1)]  <<- trdata
      colnames(traindata)[currIdx:(currIdx+l[[i]]$dim-1)] <<- l[[i]]$x.name
      testdata[,currIdx:(currIdx+l[[i]]$dim-1)]  <<- tsdata
      colnames(testdata)[currIdx:(currIdx+l[[i]]$dim-1)] <<- l[[i]]$y.name
      
    } else stop('unrecognized type of meta-data')
    
    currIdx <<- currIdx + l[[i]]$dim
  })
  
  #
  stopifnot(sum(is.na(traindata))==0)
  stopifnot(sum(is.na(testdata))==0)
  
  #
  return(list(traindata=traindata,testdata=testdata))
}
