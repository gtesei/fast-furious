% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/featureEncode.R
\name{ff.makeFeatureSet}
\alias{ff.makeFeatureSet}
\title{Encode the feature set according to meta data passed as input.}
\usage{
ff.makeFeatureSet(data.train, data.test, meta, scaleNumericFeatures = FALSE,
  parallelize = FALSE, remove1DummyVarInCatPreds = FALSE)
}
\arguments{
\item{data.train}{the observations of the predictor in train set.}

\item{data.test}{the observations of the predictor in test set.}

\item{meta}{the meata data. It should be a vector of the character \code{'C'} , \code{'N'} , \code{'D'} , 
e.g. \code{c('N','C','D')} of the same length of the train set / test set columns}

\item{scaleNumericFeatures}{seto to \code{'TRUE'} to center and scale numeric features}

\item{parallelize}{set to \code{'TRUE'} to enable parallelization (require \code{parallel} package)}

\item{remove1DummyVarInCatPreds}{\code{T} to remove one dummy variable in encoding categorical predictors. 
For further details see \code{\link[fastfurious]{ff.encodeCategoricalFeature}}.}
}
\value{
the list of trainset and testset after applying the specified encodings
}
\description{
Encode the feature set according to meta data passed as input.
}
\examples{
Xtrain <- data.frame( a = rep(1:3 , each = 2), b = 6:1, 
   c = rep(as.Date(c("2007-06-22", "2004-02-13")),3) )
Xtest <- data.frame( a = rep(2:4 , each = 2), b = 1:6, 
   c = rep(as.Date(c("2007-03-01", "2004-05-23")),3) )
l = ff.makeFeatureSet(Xtrain,Xtest,c('C','N','D'))
Xtrain = l$traindata
Xtest = l$testdata
}

