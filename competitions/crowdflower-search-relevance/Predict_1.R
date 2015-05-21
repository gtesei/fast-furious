library(readr)
library(tm)

library(NLP)

require(xgboost)
require(methods)

######################################################
## TODO 
## 1) extending with qwk 
##    http://rpackages.ianhowson.com/cran/xgboost/man/xgb.cv.html 
##    http://rpackages.ianhowson.com/cran/xgboost/man/xgb.train.html



## 2) come si ditribuisce il numero delle parole nella descrizione / titolo / query nelle varie categorie di rilevanza 1/2/3/4? 
# > ddply(train , .(median_relevance)  , function(x) c(q.w = mean(x$q.w )  ))
# median_relevance      q.w
# 1                1 2.541344
# 2                2 2.476965
# 3                3 2.369603
# 4                4 2.309512
# > ddply(train , .(median_relevance)  , function(x) c(pd.w = mean(x$pd.w )  ))
# median_relevance     pd.w
# 1                1 46.17959
# 2                2 42.55691
# 3                3 42.76166
# 4                4 41.70977
# > ddply(train , .(median_relevance)  , function(x) c(pt.w = mean(x$pt.w )  ))
# median_relevance     pt.w
# 1                1 7.624031
# 2                2 7.787263
# 3                3 7.858952
# 4                4 7.983957
# > 


## 3) median_relevance relevance_variance
#  1                1          0.3828928
#  2                2          0.6183625  <<<
#  3                3          0.6191744  <<<
#  4                4          0.2517856

## 4) lsa before tf-idf 
## 5) trying other models e.g. SVC (cap. 17 APM)
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

ScoreQuadraticWeightedKappa = function (preds, dtrain) {
  
  obs <- getinfo(dtrain, "label")
  
  min.rating=1
  max.rating=4
  
  if (missing(min.rating)) {
    min.rating <- min(min(obs), min(preds))
  }
  if (missing(max.rating)) {
    max.rating <- max(max(obs), max(preds))
  }
  obs <- factor(obs, levels <- min.rating:max.rating)
  preds <- factor(preds, levels <- min.rating:max.rating)
  confusion.mat <- table(data.frame(obs, preds))
  confusion.mat <- confusion.mat/sum(confusion.mat)
  histogram.a <- table(obs)/length(table(obs))
  histogram.b <- table(preds)/length(table(preds))
  expected.mat <- histogram.a %*% t(histogram.b)
  expected.mat <- expected.mat/sum(expected.mat)
  labels <- as.numeric(as.vector(names(table(obs))))
  weights <- outer(labels, labels, FUN <- function(x, y) (x - y)^2)
  kappa <- 1 - sum(weights * confusion.mat)/sum(weights * expected.mat)
  
  return(list(metric = "qwk", value = kappa))
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

myCorpus = function(documents , allow_numbers = F , do_stemming = F , clean = T) {
  cp = NULL 
  
  if (clean)
    cp <- Corpus(VectorSource(myCleanFun(documents)))
  else 
    cp <- Corpus(VectorSource(documents))
  
  cp <- tm_map( cp, content_transformer(tolower)) 
  cp <- tm_map( cp, content_transformer(removePunctuation))
  
  cp <- tm_map( cp, removeWords, stopwords('english')) 
  cp <- tm_map( cp, removeWords, c("ul","dl","dt","li", "a") ) ## residuals html tags 
  
  cp <- tm_map( cp, stripWhitespace)
  
  if (do_stemming) {
    cat(">> performing stemming on making corpora ... \n")
    cp = tm_map(cp, stemDocument, language = "english" )
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
  process_step = c("parsing","text_processing","text_processing","text_processing","text_processing","text_processing","text_processing","text_processing","modeling") ,
  variant =      c("allow_numbers","bigrams","trigrams","1gr_th","2gr_th","3gr_th","use_desc","stem","lossfunc") , 
  ##value =        c(F,T,T,2,3,3,F,"logloss") #0.46335
  ##value =        c(F,T,T,2,3,3,F,"qwk") ##0.51
  ##value =        c(F,T,T,2,4,4,F,T,"qwk") ##0.461 in xval 
  value =        c(F,T,T,2,4,4,F,T,"qwk")  
  )

print(settings)
fn = paste("sub2__",paste(settings$variant,settings$value,sep='',collapse = "_"),".csv",sep='')
cat(">> saving prediction on",fn,"...\n")

#### Data 
sampleSubmission <- read_csv(paste(getBasePath("data") , "sampleSubmission.csv" , sep=''))
train <- read_csv(paste(getBasePath("data") , "train.csv" , sep=''))
test  <- read_csv(paste(getBasePath("data") , "test.csv" , sep=''))

nouns <-  read.table(paste(getBasePath("data") , "nouns91K.txt" , sep=''), header=F , sep="" , colClasses = 'character') 
nouns = as.character(nouns[,1])

# adjectives <-  read.table(paste(getBasePath("data") , "adjectives28K.txt" , sep=''), header=F , sep="" , colClasses = 'character') 
# adjectives = as.character(adjectives[,1])

## Compute number of words per class 
cat(">> computing number of words in query / product title / product description  ... \n")

qcorp = myCorpus(c(train$query,test$query) , allow_numbers = T , do_stemming =  F )
ptcorp = myCorpus(c(train$product_title,test$product_title) , allow_numbers = T , do_stemming =  F )
pdcorp = myCorpus(c(train$product_description,test$product_description) , allow_numbers = T , do_stemming =  F )

qlen = sapply( qcorp , function(x)    sapply(gregexpr("\\S+", (x$content) ) , length))
ptlen = sapply( ptcorp , function(x)    sapply(gregexpr("\\S+", (x$content) ) , length))
pdlen = sapply( pdcorp , function(x)    sapply(gregexpr("\\S+", (x$content) ) , length))

rm(qcorp)
rm(ptcorp)
rm(pdcorp)

## Spell matching 
cat(">> spell matching query/description ... \n")

qcorp = myCorpus(c(train$query,test$query) , allow_numbers = T , do_stemming =  F )
ptcorp = myCorpus(c(train$product_title,test$product_title) , allow_numbers = T , do_stemming =  F )

sm = rep(NA,(nrow(train)+nrow(test)))
for (i in 1:length(sm)) {
  l = which(unlist(strsplit(qcorp[[i]]$content, " "))   %in% nouns)  
  if (length(l) > 1) l = max(l) 
  query.sost = unlist(strsplit(qcorp[[i]]$content, " "))[l]
  if (length(query.sost) == 0)  query.sost = ''
  
  l = which(unlist(strsplit(ptcorp[[i]]$content, " "))   %in% query.sost)  
  pt.sost = unlist(strsplit(ptcorp[[i]]$content, " "))[l]
  if (length(pt.sost) == 0 & substr(query.sost,nchar(query.sost),nchar(query.sost)) == 's' )  {
    ## fai la ricerca sul termine singolare 
    query.sost = substr(query.sost,1,nchar(query.sost)-1)
    l = which(unlist(strsplit(ptcorp[[i]]$content, " "))   %in% query.sost)  
    pt.sost = unlist(strsplit(ptcorp[[i]]$content, " "))[l]
  } else if (length(pt.sost) == 0) {
    ## fai la ricerca sul termine plurale 
    query.sost = paste0(query.sost,'s')
    l = which(unlist(strsplit(ptcorp[[i]]$content, " "))   %in% query.sost) 
    pt.sost = unlist(strsplit(ptcorp[[i]]$content, " "))[l]
  }
  
  sm[i] = (length(pt.sost)>0)
  
  #   cat(">>>>>>>> i:",i,"\n")
  #   cat(">> query:",qcorp[[i]]$content,"\n")
  #   cat(">> query sostantive:",query.sost,"\n")
  #   
  #   cat(">> pt:",ptcorp[[i]]$content,"\n")
  #   cat(">> pt sostantive:",pt.sost,"\n")
  #   
  #   cat(">> match:",(length(pt.sost)>0),"\n")
  
  if (i %% 1000 == 0) cat(i,"/",length(sm),"..\n")
} 

if ( sum(is.na(sm)) ) stop("something wrong with spell matching")

rm(qcorp)
rm(ptcorp)

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
               do_stemming = as.logical(settings[settings$variant == "stem" , ]$value)
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
  
  cat ("dtm.tfidf.3 dim: ",dim(dtm.tfidf.3),"\n")
  print(dtm.tfidf.3)
  
  for (i in seq(0.1,30,by = 4) ) {
    cat(".lowfreq > ",i,"\n")
    print(summary(findFreqTerms(dtm.tfidf.3 , i) ))
  }
  
} else {
  cat(">>> no 3-grams ...\n") 
}

## corpra are not necessary any more 
rm(cp)

##### Convert matrices 
cat ("converting dtm.tfidf.1 ... \n")
dtm.tfidf.1.df <- as.data.frame(inspect( dtm.tfidf.1 ))
cat ("dtm.tfidf.1.df - dim: ",dim(dtm.tfidf.1.df),"\n")
print(dtm.tfidf.1.df[1:5,1:5])

dtm.tfidf.df = dtm.tfidf.1.df
rm(dtm.tfidf.1.df)

if ( as.logical(settings[settings$variant == "bigrams" , ]$value) ) { 
  cat ("converting dtm.tfidf.2 ... \n")
  dtm.tfidf.2.df <- as.data.frame(inspect( dtm.tfidf.2 ))
  cat ("dtm.tfidf.2.df - dim: ",dim(dtm.tfidf.2.df),"\n")
  print(dtm.tfidf.2.df[1:5,1:5])
  
  dtm.tfidf.df = cbind(dtm.tfidf.df , dtm.tfidf.2.df)
  rm(dtm.tfidf.2.df)
}

if ( as.logical(settings[settings$variant == "trigrams" , ]$value) ) { 
  cat ("converting dtm.tfidf.3 ... \n")
  dtm.tfidf.3.df <- as.data.frame(inspect( dtm.tfidf.3 ))
  cat ("dtm.tfidf.3.df - dim: ",dim(dtm.tfidf.3.df),"\n")
  print(dtm.tfidf.3.df[1:5,1:5])
  
  dtm.tfidf.df = cbind(dtm.tfidf.df , dtm.tfidf.3.df)
  rm(dtm.tfidf.3.df)
}

cat (">>> dtm.tfidf.df dim: ",dim(dtm.tfidf.df),"\n")

### binding other features 
cat ("> binding other features ... \n")

dtm.tfidf.df = cbind(dtm.tfidf.df , qlen)
dtm.tfidf.df = cbind(dtm.tfidf.df , ptlen)
dtm.tfidf.df = cbind(dtm.tfidf.df , pdlen)

dtm.tfidf.df = cbind(dtm.tfidf.df , sm)

rm(qlen)
rm(ptlen)
rm(pdlen)
rm(sm)

cat (">>> dtm.tfidf.df dim: ",dim(dtm.tfidf.df),"\n")

#### preparing xboost 
x = as.matrix(dtm.tfidf.df)
x = matrix(as.numeric(x),nrow(x),ncol(x))
rm(dtm.tfidf.df)

trind = 1:nrow(train)
teind = (nrow(train)+1):nrow(x)

y = train$median_relevance-1 

# Set necessary parameter
param <- list("objective" = "multi:softmax",
                      "num_class" = 4,
                      "eta" = 0.05,  ## suggested in ESLII
                      "gamma" = 0.7,  
                      "max_depth" = 25, 
                      "subsample" = 0.5 , ## suggested in ESLII
                      "nthread" = 10, 
                      
                      "min_child_weight" = 1 , 
                      "colsample_bytree" = 0.5, 
                      "max_delta_step" = 1)

if ( ! as.character(settings[settings$variant == "lossfunc" , ]$value) == "qwk") {
  param['eval_metric'] = 'mlogloss'
} 

cat(">>Params:\n")
print(param)

cat(">> loss function:",as.character(settings[settings$variant == "lossfunc" , ]$value),"\n") 

### Cross-validation 
cat(">>Cross Validation ... \n")
inCV = T
early.stop = cv.nround = xval.perf = -1
bst.cv = NULL

if (as.character(settings[settings$variant == "lossfunc" , ]$value) == "qwk") { 
  early.stop = cv.nround = 3000
} else {
  early.stop = cv.nround = 300
}
cat(">> cv.nround: ",cv.nround,"\n") 

while (inCV) {

  if (as.character(settings[settings$variant == "lossfunc" , ]$value) == "qwk") {
    cat(">>> maximizing ScoreQuadraticWeightedKappa ...\n")
    
    bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                    nfold = 5, nrounds=cv.nround , 
                    feval = ScoreQuadraticWeightedKappa , maximize = T)
    
    print(bst.cv)
    early.stop = which(bst.cv$test.qwk.mean == max(bst.cv$test.qwk.mean) )
    xval.perf = bst.cv[early.stop,]$test.qwk.mean
    cat(">> early.stop: ",early.stop," [xval.perf:",xval.perf,"]\n") 
    
  } else {
    cat(">>> minimizing mlogloss ...\n")
    bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                    nfold = 5, nrounds=cv.nround )  
    
    print(bst.cv)
    early.stop = which(bst.cv$test.mlogloss.mean == min(bst.cv$test.mlogloss.mean) )
    xval.perf = bst.cv[early.stop,]$test.mlogloss.mean
    cat(">> early.stop: ",early.stop," [xval.perf:",xval.perf,"]\n") 
  }
  
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
#   bst = xgboost(param = param, data = x[trind,], label = y, 
#                 nrounds = early.stop,
#                 feval = ScoreQuadraticWeightedKappa , maximize = T) 

  dtrain <- xgb.DMatrix(x[trind,], label = y)
  watchlist <- list(train = dtrain)
  bst = xgb.train(param = param, dtrain , 
              nrounds = early.stop, watchlist = watchlist , 
              feval = ScoreQuadraticWeightedKappa , maximize = T , verbose = 1)
} else {
  cat(">>> minimizing mlogloss ...\n")
  bst = xgboost(param = param, data = x[trind,], label = y, 
                nrounds = early.stop) 
}

cat(">> Making prediction ... \n")
pred = predict(bst,x[teind,])
pred = pred + 1 

print(">> prediction << \n")
print(table(pred))

print(">> train set labels << \n")
print(table(y+1))

fn = paste("sub__",paste(settings$variant,settings$value,sep='',collapse = "_"),"_xval",xval.perf,".csv",sep='')
cat(">> writing prediction on disk [",fn,"]... \n")
write_csv(data.frame(id = sampleSubmission$id , prediction = pred) , paste(getBasePath("data") , fn , sep=''))
