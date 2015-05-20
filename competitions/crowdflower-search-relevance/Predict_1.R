library(readr)
library(tm)

library(NLP)

require(xgboost)
require(methods)

######################################################
## TODO 
## 1) lsa before tf-idf 
## 2) trying other models e.g. SVC (cap. 17 APM)
######################################################

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/crowdflower-search-relevance"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/crowdflower-search-relevance/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/crowdflower-search-relevance"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/ocrowdflower-search-relevance/"
  } else if (type == "process") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/data_process"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/data_process/"
  } else {
    stop("unrecognized type.")
  }
  
  if (file.exists(base.path1))  {
    ret = paste0(base.path1,"/")
  } else {
    ret = base.path2
  }
  
  ret
}

BigramTokenizer = function(x) {
    unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)
}

TrigramTokenizer = function(x) {
  unlist(lapply(ngrams(words(x), 3), paste, collapse = " "), use.names = FALSE)
}

ScoreQuadraticWeightedKappa = function (rater.a, rater.b, min.rating, max.rating) {
  if (missing(min.rating)) {
    min.rating <- min(min(rater.a), min(rater.b))
  }
  if (missing(max.rating)) {
    max.rating <- max(max(rater.a), max(rater.b))
  }
  rater.a <- factor(rater.a, levels <- min.rating:max.rating)
  rater.b <- factor(rater.b, levels <- min.rating:max.rating)
  confusion.mat <- table(data.frame(rater.a, rater.b))
  confusion.mat <- confusion.mat/sum(confusion.mat)
  histogram.a <- table(rater.a)/length(table(rater.a))
  histogram.b <- table(rater.b)/length(table(rater.b))
  expected.mat <- histogram.a %*% t(histogram.b)
  expected.mat <- expected.mat/sum(expected.mat)
  labels <- as.numeric(as.vector(names(table(rater.a))))
  weights <- outer(labels, labels, FUN <- function(x, y) (x - y)^2)
  kappa <- 1 - sum(weights * confusion.mat)/sum(weights * expected.mat)
  kappa
}

myCleanFun <- function(htmlString) {
  ## unicode 
  htmlString = gsub("w*\u0092s"," ",htmlString) ##steve\u0092s
  
  ## entities 
  htmlString = gsub("(\\&\\w+\\;|\\&#\\d+\\;)"," ",htmlString)
  
  ## tab 
  htmlString = gsub("(\\\t|\\\n)"," ",htmlString)
  
  ## custom tags 
  htmlString = gsub("(\\*|h2|dd|li|.Pa_\\w+(\\s+(table|td|table\\s+img|table\\s+td\\s+img))?)\\s*\\{[^\\}]*\\}*", "", htmlString)
  htmlString = gsub(".Pa_\\w+", "", htmlString)
  
  ## html tags 
  pattern <- "</?\\w+((\\s+\\w+(\\s*=\\s*(?:\".*?\"|'.*?'|[^'\">\\s]+))?)+\\s*|\\s*)/?>"
  htmlString = gsub(pattern, "\\1", htmlString)
  
  ## inches 
  htmlString = gsub("\\\""," inch",htmlString) ##Asus VS238H-P 23 LED LCD Monitor - 16:9 - 2 ms\""
  
  return(htmlString)
}

myCorpus = function(documents , allow_numbers = F , do_stemming = F) {
  cp <- Corpus(VectorSource(  myCleanFun(documents)   ) )
  cp <- tm_map( cp, content_transformer(tolower)) 
  cp <- tm_map( cp, content_transformer(removePunctuation))
  
  cp <- tm_map( cp, removeWords, stopwords('english')) 
  cp <- tm_map( cp, removeWords, c("ul","dl","dt","li") ) ## residuals html tags 
  
  cp <- tm_map( cp, stripWhitespace)
  
  if (do_stemming) {
    cat(">> performing stemming on making corpora ... \n")
    ctrpd = tm_map(ctrpd, stemDocument) 
  }
  
  if (! allow_numbers) {
    cat(">> discarding numbers on making corpora ... \n") 
    cp <- tm_map(cp, removeNumbers)
  }
  
  return (cp)
}

#### Settings 
cat("***************** VARIANT SETTINGS *****************\n")
settings = data.frame(
  process_step = c("parsing","text_processing","text_processing","text_processing","text_processing","text_processing","text_processing","modeling") ,
  variant =      c("allow_numbers","bigrams","trigrams","1gr_th","2gr_th","3gr_th","use_desc","lossfunc") , 
  value =        c(F,T,T,2,2,2,F,"qwk")
  ##value =        c(F,T,T,2,2,2,F,"logloss")
  )

print(settings)
fn = paste("sub__",paste(settings$variant,settings$value,sep='',collapse = "_"),".csv",sep='')
cat(">> saving prediction on",fn,"...\n")

#### Data 
sampleSubmission <- read_csv(paste(getBasePath("data") , "sampleSubmission.csv" , sep=''))
train <- read_csv(paste(getBasePath("data") , "train.csv" , sep=''))
test  <- read_csv(paste(getBasePath("data") , "test.csv" , sep=''))

## Make corpora  
if( ! as.logical(settings[settings$variant == "use_desc" , ]$value) ) {
  cat(">> discarding product description on making corpora ... \n")
  train$merge = apply(X = train , 1 , function(x) paste(x[2] , x[3]  , sep= ' ') )
  test$merge = apply(X = test , 1 , function(x) paste(x[2] , x[3]  , sep= ' ') )
} else {
  cat(">> using product description on making corpora ... \n")
  train$merge = apply(X = train , 1 , function(x) paste(x[2] , x[3] , x[4] , sep= ' ') )
  test$merge = apply(X = test , 1 , function(x) paste(x[2] , x[3] , x[4] , sep= ' ') )
}

cp = myCorpus( c(train$merge , test$merge), 
               allow_numbers = as.logical(settings[settings$variant == "allow_numbers" , ]$value) , 
               do_stemming = F
)

st = sample(length(cp),10)
st = c(826,6180,4585,st)
for (i in st) {
  cat("\n****************************************[",i,"]*************************************\n")
  cat(">>> Raw: \n")
  if (i <= nrow(train) ) print(train[i,]$merge)
  else print(test[i-nrow(train),]$merge)
  cat(">>> Processed: \n")
  print(cp[[i]]$content)
}

cat (">>> corpora length: ",length(cp),"\n")

########## N-Grams 
dtm.tfidf.1 = NULL
dtm.tfidf.2 = NULL
dtm.tfidf.3 = NULL

### 1-grams 
onegr_th = as.numeric(as.character(settings[settings$variant == "1gr_th" , ]$value)) 
cat(">>> 1-grams lowfreq:",onegr_th,"\n")
dtm.tfidf.1 <- DocumentTermMatrix( cp, control = list( weighting = weightTfIdf , 
                                                     bounds = list( global = c( onegr_th, Inf)) ) ) 

cat ("dtm.tfidf.1 dim: ",dim(dtm.tfidf.1),"\n")
print(dtm.tfidf.1)

for (i in seq(0.1,30,by = 4) ) {
  cat(".lowfreq > ",i,"\n")
  print(summary(findFreqTerms(dtm.tfidf.1 , i) ))
}

### 2-grams 
if ( as.logical(settings[settings$variant == "bigrams" , ]$value) ) { 
  lowfreq = as.numeric(as.character(settings[settings$variant == "2gr_th" , ]$value)) 
  cat(">>> 2-grams lowfreq:",lowfreq,"\n")
  
  ptm <- proc.time()
  dtm.tfidf.2 <- DocumentTermMatrix( cp, control = list( weighting = weightTfIdf , 
                                                           bounds = list( global = c( lowfreq , Inf)) , 
                                                           tokenize = BigramTokenizer)) 
  cat(">Time elapsed:",(proc.time() - ptm),"\n")
  
  cat ("dtm.tfidf.2 dim: ",dim(dtm.tfidf.2),"\n")
  print(dtm.tfidf.2)
  
  for (i in seq(0.1,30,by = 4) ) {
    cat(".lowfreq > ",i,"\n")
    print(summary(findFreqTerms(dtm.tfidf.2 , i) ))
  }
  
} else {
  cat(">>> no 2-grams ...\n") 
}

### 3-grams 
if ( as.logical(settings[settings$variant == "trigrams" , ]$value) ) { 
  lowfreq = as.numeric(as.character(settings[settings$variant == "3gr_th" , ]$value)) 
  cat(">>> 3-grams lowfreq:",lowfreq,"\n")
  
  ptm <- proc.time()
  dtm.tfidf.3 <- DocumentTermMatrix( cp, control = list( weighting = weightTfIdf , 
                                                               bounds = list( global = c( lowfreq , Inf)) , 
                                                               tokenize = TrigramTokenizer)) 
  cat(">Time elapsed:",(proc.time() - ptm),"\n")
  
  cat ("dtm.tfidf.2 dim: ",dim(dtm.tfidf.2),"\n")
  print(dtm.tfidf.2)
  
  for (i in seq(0.1,30,by = 4) ) {
    cat(".lowfreq > ",i,"\n")
    print(summary(findFreqTerms(dtm.tfidf.2 , i) ))
  }
  
} else {
  cat(">>> no 3-grams ...\n") 
}

##### Convert matrices 
cat ("converting dtm.tfidf.1 ... \n")
dtm.tfidf.1.df <- as.data.frame(inspect( dtm.tfidf.1 ))
cat ("dtm.tfidf.1.df - dim: ",dim(dtm.tfidf.1.df),"\n")
print(dtm.tfidf.1.df[1:5,1:5])

dtm.tfidf.df = dtm.tfidf.1.df

if ( as.logical(settings[settings$variant == "bigrams" , ]$value) ) { 
  cat ("converting dtm.tfidf.2 ... \n")
  dtm.tfidf.2.df <- as.data.frame(inspect( dtm.tfidf.2 ))
  cat ("dtm.tfidf.2.df - dim: ",dim(dtm.tfidf.2.df),"\n")
  print(dtm.tfidf.2.df[1:5,1:5])
  
  dtm.tfidf.df = cbind(dtm.tfidf.df , dtm.tfidf.2.df)
}

if ( as.logical(settings[settings$variant == "trigrams" , ]$value) ) { 
  cat ("converting dtm.tfidf.3 ... \n")
  dtm.tfidf.3.df <- as.data.frame(inspect( dtm.tfidf.3 ))
  cat ("dtm.tfidf.3.df - dim: ",dim(dtm.tfidf.3.df),"\n")
  print(dtm.tfidf.3.df[1:5,1:5])
  
  dtm.tfidf.df = cbind(dtm.tfidf.df , dtm.tfidf.3.df)
}

cat ("dtm.tfidf.df dim: ",dim(dtm.tfidf.df),"\n")

#### preparing xboost 
x = as.matrix(dtm.tfidf.df)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:nrow(train)
teind = (nrow(train)+1):nrow(x)

y = train$median_relevance-1 

# Set necessary parameter
param <- list("objective" = "multi:softmax",
              "num_class" = 4,
              "eta" = 0.05,  ## suggested in ESLII
              "gamma" = 0.5,  
              "max_depth" = 25, 
              "subsample" = 0.5 , ## suggested in ESLII
              "nthread" = 10, 
              
              "min_child_weight" = 1 , 
              "colsample_bytree" = 0.5, 
              "max_delta_step" = 1
)

cat(">>Params:\n")
print(param)

cat(">> loss function:",as.character(settings[settings$variant == "lossfunc" , ]$value),"\n") 

### Cross-validation 
cat(">>Cross Validation ... \n")
inCV = T
early.stop = cv.nround = 300
bst.cv = NULL

while (inCV) {
  
  cat(">> cv.nround: ",cv.nround,"\n") 
  if (as.character(settings[settings$variant == "lossfunc" , ]$value) == "qwk") {
    cat(">>> maximizing ScoreQuadraticWeightedKappa ...\n")
    bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                    nfold = 5, nrounds=cv.nround , 
                    feval = ScoreQuadraticWeightedKappa , maximize = T)
  } else {
    cat(">>> minimizing mlogloss ...\n")
    bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                    nfold = 5, nrounds=cv.nround , 
                    feval = mlogloss , maximize = F)  
  }
  
  print(bst.cv)
  early.stop = which(bst.cv$test.mlogloss.mean == min(bst.cv$test.mlogloss.mean) )
  cat(">> early.stop: ",early.stop," [test.mlogloss.mean:",bst.cv[early.stop,]$test.mlogloss.mean,"]\n") 
  if (early.stop < cv.nround) {
    inCV = F
    cat(">> stopping [early.stop < cv.nround=",cv.nround,"] ... \n") 
  } else {
    cat(">> redo-cv [early.stop == cv.nround=",cv.nround,"] with 2 * cv.nround ... \n") 
    cv.nround = cv.nround * 2 
  }
  gc()
}

### Prediction 
bst = NULL

if (as.character(settings[settings$variant == "lossfunc" , ]$value) == "qwk") {
  cat(">>> maximizing ScoreQuadraticWeightedKappa ...\n")
  bst = xgboost(param = param, data = x[trind,], label = y, 
                nrounds = early.stop,
                feval = ScoreQuadraticWeightedKappa , maximize = T) 
} else {
  cat(">>> minimizing mlogloss ...\n")
  bst = xgboost(param = param, data = x[trind,], label = y, 
                nrounds = early.stop,
                feval = mlogloss , maximize = F) 
}

cat(">> Making prediction ... \n")
pred = predict(bst,x[teind,])
pred = pred + 1 

print(">> prediction << \n")
print(table(pred))

print(">> train set labels << \n")
print(table(y+1))

print(">> writing prediction on disk ... \n")
write_csv(data.frame(id = sampleSubmission$id , prediction = pred) , paste(getBasePath("data") , fn , sep=''))
