library(readr)
library(tm)

library(NLP)

require(xgboost)
require(methods)

require(plyr)

#library(caret)

######################################################
## TODO 

## 0) provare con maggori valori di 2-gram / 3-gram 

## 1) estrarre le seguenti 2 features: 
##          # di volte che i 2-gram della query occorrono nel product title  
##          # di volte che i 2-gram della query occorrono nel product description 
##          --> fatto(prtm,pdm). in xval ho 0.478 vs 0.473 e sulla leaderboard 0.587 vs 0.589 ... la lascio?

## 2) usa dtm <- removeSparseTerms(dtm,0.99) - 
##       come nel post https://www.kaggle.com/users/191033/luis-argerich/crowdflower-search-relevance/r-vector-space-model


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
  
  min.rating = 1
  max.rating = 4
  
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
  ##value =        c(F,T,T,2,4,4,F,T,"qwk") ##0.58 
  ##value =        c(F,T,T,2,5,5,F,T,"qwk")##0.589 
  value =        c(F,T,T,2,6,6,F,T,"qwk") ##0.597
  )

print(settings)
fn = paste("sub3__",paste(settings$variant,settings$value,sep='',collapse = "_"),".csv",sep='')
cat(">> saving prediction on",fn,"...\n")

#### Data 
sampleSubmission <- read_csv(paste(getBasePath("data") , "sampleSubmission.csv" , sep=''))
train <- read_csv(paste(getBasePath("data") , "train.csv" , sep=''))
test  <- read_csv(paste(getBasePath("data") , "test.csv" , sep=''))

nouns <-  read.table(paste(getBasePath("data") , "nouns91K.txt" , sep=''), header=F , sep="" , colClasses = 'character') 
nouns = as.character(nouns[,1])

adjectives <-  read.table(paste(getBasePath("data") , "adjectives28K.txt" , sep=''), header=F , sep="" , colClasses = 'character') 
adjectives = as.character(adjectives[,1])

## Number of words in query / product title / product description 
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
qcorp = myCorpus(c(train$query,test$query) , allow_numbers = F , do_stemming =  F )
ptcorp = myCorpus(c(train$product_title,test$product_title) , allow_numbers = F , do_stemming =  F )
pdcorp = myCorpus(c(train$product_description,test$product_description) , allow_numbers = F , do_stemming =  F )

cat(">> spell matching query/title as a substantive .. matching query/title  .. query/description .. ufos ..")
ufos.q = ufos.pt = ufos.same = amd = am = sm = prtm = pdm = notprtm = notpdm = rep(NA,(nrow(train)+nrow(test)))
for (i in 1:length(sm)) {
  
  ## ufos
  l = which(   (! unlist(strsplit(qcorp[[i]]$content, " "))   %in% adjectives) && (! unlist(strsplit(qcorp[[i]]$content, " "))   %in% nouns)    )  
  ufos.q[i] = length(unlist(strsplit(qcorp[[i]]$content, " "))[l])
  
  ll = which(   (! unlist(strsplit(ptcorp[[i]]$content, " "))   %in% adjectives) && (! unlist(strsplit(ptcorp[[i]]$content, " "))   %in% nouns)    )  
  ufos.pt[i] = length(unlist(strsplit(ptcorp[[i]]$content, " "))[ll])
  
  if (ufos.q[i] == 0) ufos.same[i] = F 
  else ufos.same[i] = (unlist(strsplit(qcorp[[i]]$content, " "))[l] %in% unlist(strsplit(ptcorp[[i]]$content, " "))[ll])
    
  ## amd
  l = which(unlist(strsplit(qcorp[[i]]$content, " "))   %in% adjectives)  
  query.adj = unlist(strsplit(qcorp[[i]]$content, " "))[l]
  if (length(query.adj) == 0)  {
    amd[i] = F
  } else {
    l = which(unlist(strsplit(pdcorp[[i]]$content, " "))   %in% query.adj)  
    pt.adj = unlist(strsplit(pdcorp[[i]]$content, " "))[l]   
    
    amd[i] = (length(pt.adj)>0)
  }
  
  ## am 
  l = which(unlist(strsplit(qcorp[[i]]$content, " "))   %in% adjectives)  
  query.adj = unlist(strsplit(qcorp[[i]]$content, " "))[l]
  if (length(query.adj) == 0)  {
    am[i] = F
  } else {
    l = which(unlist(strsplit(ptcorp[[i]]$content, " "))   %in% query.adj)  
    pt.adj = unlist(strsplit(ptcorp[[i]]$content, " "))[l]   
    
    am[i] = (length(pt.adj)>0)
  }
  
  ## sm 
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
  
  ## prtm pdm 
  q = unlist(strsplit(qcorp[[i]]$content, " "))
  
  l = which(unlist(strsplit(ptcorp[[i]]$content, " "))   %in% q)  
  #prtm[i] = (length(l)>0) 
  prtm[i] = length(l)
  
  l = which(unlist(strsplit(pdcorp[[i]]$content, " "))   %in% q)  
  #pdm[i] = (length(l)>0)
  pdm[i] = length(l)
  
  l = which(! (q %in% unlist(strsplit(ptcorp[[i]]$content, " "))) )  
  notprtm[i] = length(l)
  
  l = which(! (q %in% unlist(strsplit(pdcorp[[i]]$content, " "))) )  
  notpdm[i] = length(l)
  
  if (i %% 1000 == 0) cat(" [",i,"/",length(sm),"].. ")
} 

if ( sum(is.na(ufos.q)) ) stop("something wrong with ufos.q")
if ( sum(is.na(ufos.pt)) ) stop("something wrong with ufos.pt")
if ( sum(is.na(ufos.same)) ) stop("something wrong with ufos.same")
if ( sum(is.na(sm)) ) stop("something wrong with sm")
if ( sum(is.na(am)) ) stop("something wrong with am")
if ( sum(is.na(amd)) ) stop("something wrong with amd")
if ( sum(is.na(prtm)) ) stop("something wrong with prtm")
if ( sum(is.na(pdm)) ) stop("something wrong with pdm")
if ( sum(is.na(notprtm)) ) stop("something wrong with pdm")
if ( sum(is.na(notpdm)) ) stop("something wrong with pdm")

train$ufos.q = ufos.q[1:nrow(train)]
train$ufos.pt = ufos.pt[1:nrow(train)]
train$ufos.same = ufos.same[1:nrow(train)]
train$sm = sm[1:nrow(train)]
train$am = am[1:nrow(train)]
train$amd = amd[1:nrow(train)]
train$prtm = prtm[1:nrow(train)]
train$pdm = pdm[1:nrow(train)]
train$notprtm = notprtm[1:nrow(train)]
train$notpdm = notpdm[1:nrow(train)]

cat("\n")
print(ddply(train, .(median_relevance) , function(x) c( ufos.q.mean=mean(x$ufos.q) , ufos.pt.mean=mean(x$ufos.pt) , ufos.same.mean=mean(x$ufos.same) , sm.mean = mean(x$sm), am.mean = mean(x$am) , amd.mean = mean(x$amd) , prtm.mean = mean(x$prtm) , pdm.mean = mean(x$pdm) )  ))
print(ddply(train, .(median_relevance) , function(x) c( ufos.q.sd = sd(x$ufos.q), ufos.pt.sd = sd(x$ufos.pt), ufos.same.sd = sd(x$ufos.same), sm.sd = sd(x$sm), am.sd = sd(x$am) , amd.sd = sd(x$amd) , prtm.sd = sd(x$prtm) , pdm.sd = sd(x$pdm) )  ))

#rm(qcorp)
#rm(ptcorp)
#rm(pdcorp)

rm(nouns)
rm(adjectives)

## Make corpora  
if( ! as.logical(settings[settings$variant == "use_desc" , ]$value) ) {
  cat("\n>> discarding product description on making corpora ... \n")
  train$merge = apply(X = train , 1 , function(x) paste(x[2] , x[3]  , sep= ' ') )
  test$merge = apply(X = test , 1 , function(x) paste(x[2] , x[3]  , sep= ' ') )
} else {
  cat("\n>> using product description on making corpora ... \n")
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
# dtm.tfidf.1 <- DocumentTermMatrix( cp, control = list( weighting = weightTfIdf , 
#                                                      bounds = list( global = c( onegr_th, Inf)) ) ) 

dtm.tfidf.1 <- DocumentTermMatrix( cp, control = list( weighting = function(x) weightSMART(x, spec = "ntc") , 
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
#   dtm.tfidf.2 <- DocumentTermMatrix( cp, control = list( weighting = weightTfIdf , 
#                                                            bounds = list( global = c( lowfreq , Inf)) , 
#                                                            tokenize = BigramTokenizer)) 
  
  dtm.tfidf.2 <- DocumentTermMatrix( cp, control = list( weighting = function(x) weightSMART(x, spec = "ntc") , 
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
#   dtm.tfidf.3 <- DocumentTermMatrix( cp, control = list( weighting = weightTfIdf , 
#                                                                bounds = list( global = c( lowfreq , Inf)) , 
#                                                                tokenize = TrigramTokenizer)) 
  
  dtm.tfidf.3 <- DocumentTermMatrix( cp, control = list( weighting = function(x) weightSMART(x, spec = "ntc") , 
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
#rm(cp)

##### Convert matrices 
cat ("converting dtm.tfidf.1 ... \n")
dtm.tfidf.1.df <- as.data.frame(inspect( dtm.tfidf.1 ))
cat ("dtm.tfidf.1.df - dim: ",dim(dtm.tfidf.1.df),"\n")
print(dtm.tfidf.1.df[1:5,1:5])

dtm.tfidf.df = dtm.tfidf.1.df

rm(dtm.tfidf.1)
rm(dtm.tfidf.1.df)

if ( as.logical(settings[settings$variant == "bigrams" , ]$value) ) { 
  cat ("converting dtm.tfidf.2 ... \n")
  dtm.tfidf.2.df <- as.data.frame(inspect( dtm.tfidf.2 ))
  cat ("dtm.tfidf.2.df - dim: ",dim(dtm.tfidf.2.df),"\n")
  print(dtm.tfidf.2.df[1:5,1:5])
  
  dtm.tfidf.df = cbind(dtm.tfidf.df , dtm.tfidf.2.df)
  
  rm(dtm.tfidf.2)
  rm(dtm.tfidf.2.df)
}

if ( as.logical(settings[settings$variant == "trigrams" , ]$value) ) { 
  cat ("converting dtm.tfidf.3 ... \n")
  dtm.tfidf.3.df <- as.data.frame(inspect( dtm.tfidf.3 ))
  cat ("dtm.tfidf.3.df - dim: ",dim(dtm.tfidf.3.df),"\n")
  print(dtm.tfidf.3.df[1:5,1:5])
  
  dtm.tfidf.df = cbind(dtm.tfidf.df , dtm.tfidf.3.df)
  
  rm(dtm.tfidf.3)
  rm(dtm.tfidf.3.df)
}

cat (">>> dtm.tfidf.df dim: ",dim(dtm.tfidf.df),"\n")

### binding other features 
cat ("> binding other features ... \n")

dtm.tfidf.df = cbind(dtm.tfidf.df , qlen)
dtm.tfidf.df = cbind(dtm.tfidf.df , ptlen)
dtm.tfidf.df = cbind(dtm.tfidf.df , pdlen)

dtm.tfidf.df = cbind(dtm.tfidf.df , ufos.q)
dtm.tfidf.df = cbind(dtm.tfidf.df , ufos.pt)
dtm.tfidf.df = cbind(dtm.tfidf.df , ufos.same)

dtm.tfidf.df = cbind(dtm.tfidf.df , sm)
dtm.tfidf.df = cbind(dtm.tfidf.df , am)
dtm.tfidf.df = cbind(dtm.tfidf.df , amd)

dtm.tfidf.df = cbind(dtm.tfidf.df , prtm)
dtm.tfidf.df = cbind(dtm.tfidf.df , pdm)

dtm.tfidf.df = cbind(dtm.tfidf.df , notprtm)
dtm.tfidf.df = cbind(dtm.tfidf.df , notpdm)

rm(qlen)
rm(ptlen)
rm(pdlen)

rm(ufos.q)
rm(ufos.pt)
rm(ufos.same)

rm(sm)
rm(am)
rm(amd)

rm(prtm)
rm(pdm)

rm(notprtm)
rm(notpdm)

cat (">>> dtm.tfidf.df dim: ",dim(dtm.tfidf.df),"\n")

#### preparing xboost 
x = as.matrix(dtm.tfidf.df)
x = matrix(as.numeric(x),nrow(x),ncol(x))
#rm(dtm.tfidf.df)

trind = 1:nrow(train)
teind = (nrow(train)+1):nrow(x)

y = train$median_relevance-1 

rm(train)
rm(test)

### some data transformation  
# cat(">>> Applying some transformation to data ... \n")
# ptm <- proc.time()
# trans <- preProcess(x, method = c("center", "scale") )
# # trans <- preProcess(x, method = c("BoxCox","center", "scale","pca") )
# # trans <- preProcess(x, method = c("center", "scale","pca") )
# # trans <- preProcess(x, method = c("BoxCox","center", "scale","ica") )
# # trans <- preProcess(x, method = c("center", "scale","ica") )
# # trans <- preProcess(x, method = c("center", "scale", "spatialSign") )
# print(trans)
# x = predict(trans,x)
# cat(">Time elapsed:",(proc.time() - ptm),"\n")
# rm(trans)

##### xgboost --> set necessary parameter
param <- list("objective" = "multi:softmax",
                      "num_class" = 4,
                      "eta" = 0.05,  
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

print(">> prediction <<")
print(table(pred))

print(">> train set labels <<")
print(table(y+1))

fn = paste("sub5__",paste(settings$variant,settings$value,sep='',collapse = "_"),"_xval",xval.perf,".csv",sep='')
cat(">> writing prediction on disk [",fn,"]... \n")
write_csv(data.frame(id = sampleSubmission$id , prediction = pred) , paste(getBasePath("data") , fn , sep=''))
