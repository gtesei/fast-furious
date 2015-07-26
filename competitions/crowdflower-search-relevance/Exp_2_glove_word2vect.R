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
ptcorp = myCorpus(c(train$product_title,test$product_title) , allow_numbers = T , do_stemming =  F )
pdcorp = myCorpus(c(train$product_description,test$product_description) , allow_numbers = T , do_stemming =  F )


### study similar questions 
q_map <- new.env(hash = T, parent = emptyenv())
lapply(qcorp , function(x) {
  q_map[[x$content]] = x$content
})

queries = ls(q_map)
for (i in 1:length(queries))
  q_map[[queries[i]]] = i

## q_class 
train$q_class = NA
for (i in 1:nrow(train)) {
  train[i,]$q_class = q_map[[qcorp[[i]]$content]]
}

## cosine 
cosine_vect = rep(NA, (nrow(train)+nrow(test)) )  
#q_vect = matrix(rep(NA, 300*(nrow(train)+nrow(test)) )  , ncol = 300)
#pt_vect = matrix(rep(NA, 300*(nrow(train)+nrow(test)) )  , ncol = 300)
train$cosine = NA
test$cosine = NA
for (i in 1:(nrow(train)+nrow(test))) {
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
  
  ##pavg
  pavg = NULL
  #p = c(ptcorp[[i]]$content,pdcorp[[i]]$content)
  p = c(ptcorp[[i]]$content)
  
  pp = unlist(strsplit(p, split=' ', fixed = T))
  
  if ('nespreso' %in% pp) pp =c(pp,'nespresso','coffee') 
  if ('nespressoäî' %in% pp) pp =c(pp,'nespresso','coffee')
  
  if ('silverplated' %in% pp) pp =c(pp,"silver","plated")
  if ('zippohandwarmerrealtree' %in% pp) pp =c(pp,"zippo","hand","warmer","real","tree")
  if ('batteryrechargeablegn' %in% pp) pp =c(pp,"battery","rechargeable")
  if ('daisylane' %in% pp) pp =c(pp,"daisy","lane")
  if ('zippochromehandwarmer' %in% pp) pp =c(pp,"zippo","chrome", "handwarmer")
  
  l = lapply(pp[pp != ""] , function(x) word_vect_map[[x]])
  for (j in 1:length(l)) {
    if (is.null(l[[j]])) next  
    if (is.null(pavg)) pavg = l[[j]]
    else pavg = pavg + l[[j]]
  }
  pavg = pavg / length(l)
  
  ##update 
  #q_vect[i,] = qavg 
  #pt_vect[i,] = pavg 
  
  if (i <= nrow(train)) {
    if (length(pavg) > 1 & length(qavg) > 1) {
      train[i,]$cosine = (as.numeric((qavg %*% pavg))  / (sqrt(sum(qavg^2)) * sqrt(sum(pavg^2)) ) )
      cosine_vect[i] = train[i,]$cosine
    }
    else {
      cat(i,") q=",q," -- p=",p,"\n")
      train[i,]$cosine = 0 
      cosine_vect[i] = 0
    }
  } else {
    idx = i-nrow(train)
    if (length(pavg) > 1 & length(qavg) > 1) {
      test[idx,]$cosine = (as.numeric((qavg %*% pavg))  / (sqrt(sum(qavg^2)) * sqrt(sum(pavg^2)) ) )
      cosine_vect[i] = test[idx,]$cosine
    }
    else {
      cat(i,") q=",q," -- p=",p,"\n")
      test[idx,]$cosine = 0 
      cosine_vect[i] = 0
    }
  }
}
  
train[train$q_class == 44 , c("query","product_title","median_relevance","cosine") ]
ddply(train , .(median_relevance) , function(x) c(mean = mean(x$cosine)  , sd = sd(x$cosine)) )  

### glove.6B.50d.txt 
# median_relevance      mean        sd
# 1                1 0.5691602 0.2309199
# 2                2 0.6804498 0.1774544
# 3                3 0.6923356 0.1814901
# 4                4 0.7386879 0.1625147

### glove.6B.300d.txt
# median_relevance      mean        sd
# 1                1 0.4169997 0.2230864
# 2                2 0.5630648 0.1852858
# 3                3 0.5964809 0.1838824
# 4                4 0.6563371 0.1631150

#cat(">> serializing cosine_vect on disk ... \n")
write_csv(data.frame(id = 1:(nrow(train)+nrow(test)) , cosine_vect = cosine_vect , query_vect = q_vect , pt_vect = pt_vect) , 
          paste(getBasePath("data") , paste0(label,'.csv') , sep=''))
