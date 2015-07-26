library(readr)
library(tm)

library(NLP)

require(xgboost)
require(methods)

require(plyr)


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

#### Data 

## word_vect_map 
#word_vect_lines <- read_lines(paste(getBasePath("data") , "./glove/glove.6B.50d.txt" , sep=''))
#label = 'cosine_6B.50d' 

word_vect_lines <- read_lines(paste(getBasePath("data") , "./glove/glove.6B.300d.txt" , sep=''))
label = 'cosine_6B.300d'

# word_vect_lines <- read_lines(paste(getBasePath("data") , "./glove/glove.840B.300d.txt" , sep=''))
# label = 'ext_cosine_840B.300d'

word_vect_list = lapply(word_vect_lines , function(x) {
  stuff_vect_list = strsplit(x, split=' ', fixed = T)
  stuff_vect = stuff_vect_list[[1]]
  return(list(term = stuff_vect[1] ,
              vect = as.numeric(stuff_vect[2:length(stuff_vect)])))
})
word_vect_map <- new.env(hash = T, parent = emptyenv())
lapply(word_vect_list , function(x) {
  word_vect_map[[x$term]] = x$vect
})

## sampleSubmission , train , test  
sampleSubmission <- read_csv(paste(getBasePath("data") , "sampleSubmission.csv" , sep=''))
train <- read_csv(paste(getBasePath("data") , "train.csv" , sep=''))
test  <- read_csv(paste(getBasePath("data") , "test.csv" , sep=''))

qcorp = myCorpus(c(train$query,test$query) , allow_numbers = T , do_stemming =  F )
#ptcorp = myCorpus(c(train$product_title,test$product_title) , allow_numbers = T , do_stemming =  F )
#pdcorp = myCorpus(c(train$product_description,test$product_description) , allow_numbers = T , do_stemming =  F )


### study similar questions 
q_map <- new.env(hash = T, parent = emptyenv())
lapply(qcorp , function(x) {
  q_map[[x$content]] = x$content
})

queries = ls(q_map)
for (i in 1:length(queries))
  q_map[[queries[i]]] = i

query_vect = matrix(rep(NA,300*length(queries)),nrow = length(queries))

for (i in 1:length(queries)) {
  ##qavg
  qavg = NULL
  q = qcorp[[i]]$content
  
  if (q == "harleydavidson") q = c("harley","davidson")
  if (q == "refrigirator") q = c("refrigerator")
  if (q == "blendtec") q = c("blendtec" , "blenders")
  
  qq = unlist(strsplit(q, split=' ', fixed = T))
  
  if ('edgerouter' %in% qq) qq =c(qq,'edge','router') 
  
  l = lapply( qq[qq != ""] , function(x) word_vect_map[[x]])
  for (j in 1:length(l)) {
    if (is.null(l[[j]])) next  
    if (is.null(qavg)) qavg = l[[j]]
    else qavg = qavg + l[[j]]
  }
  qavg = qavg / length(l)
  
  query_vect[i,] = qavg
}
  

#cat(">> serializing on disk ... \n")
save(query_vect,file=paste(getBasePath("data") , "query_vect" , sep=''))
save(queries,file=paste(getBasePath("data") , "queries" , sep=''))
