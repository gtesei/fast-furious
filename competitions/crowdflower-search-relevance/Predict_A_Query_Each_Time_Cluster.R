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

makeSyntacticFeatures <- function(train,test,nouns,adjectives) {
  
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
  ufos.q = ufos.pt = ufos.same = amd = am = sm = prtm = pdm = rep(NA,(nrow(train)+nrow(test)))
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
  
  #   cat("\n")
  #   print(ddply(train, .(median_relevance) , function(x) c( ufos.q.mean=mean(x$ufos.q) , ufos.pt.mean=mean(x$ufos.pt) , ufos.same.mean=mean(x$ufos.same) , sm.mean = mean(x$sm), am.mean = mean(x$am) , amd.mean = mean(x$amd) , prtm.mean = mean(x$prtm) , pdm.mean = mean(x$pdm) )  ))
  #   print(ddply(train, .(median_relevance) , function(x) c( ufos.q.sd = sd(x$ufos.q), ufos.pt.sd = sd(x$ufos.pt), ufos.same.sd = sd(x$ufos.same), sm.sd = sd(x$sm), am.sd = sd(x$am) , amd.sd = sd(x$amd) , prtm.sd = sd(x$prtm) , pdm.sd = sd(x$pdm) )  ))
  #   
  rm(qcorp)
  rm(ptcorp)
  rm(pdcorp)
  
  rm(nouns)
  rm(adjectives)
  
  return(list(qlen=qlen,ptlen=ptlen,pdlen=pdlen,ufos.q=ufos.q,ufos.pt=ufos.pt,ufos.same=ufos.same,sm=sm,am=am,amd=amd,prtm=prtm,pdm=pdm))
}

makeGrams <- function(train,test,settings) {
  
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
  
#   cat("\n>> using product title and product description on making corpora ... \n")
#   train$merge = apply(X = train , 1 , function(x) paste(x[2] , x[3] , x[4] , sep= ' ') )
#   test$merge = apply(X = test , 1 , function(x) paste(x[2] , x[3] , x[4] , sep= ' ') )
  
  cp = myCorpus( c(train$merge , test$merge), 
                 allow_numbers = as.logical(settings[settings$variant == "allow_numbers" , ]$value) , 
                 do_stemming = as.logical(settings[settings$variant == "stem" , ]$value)
  )
  
#   st = sample(length(cp),10)
#   st = c(826,6180,4585,st)
#   for (i in st) {
#     cat("\n****************************************[",i,"]*************************************\n")
#     cat(">>> Raw: \n")
#     if (i <= nrow(train) ) print(train[i,]$merge)
#     else print(test[i-nrow(train),]$merge)
#     cat(">>> Processed: \n")
#     print(cp[[i]]$content)
#   }
  
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
  
#   for (i in seq(0.1,30,by = 4) ) {
#     cat(".lowfreq > ",i,"\n")
#     print(summary(findFreqTerms(dtm.tfidf.1 , i) ))
#   }
  
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
    
#     for (i in seq(0.1,30,by = 4) ) {
#       cat(".lowfreq > ",i,"\n")
#       print(summary(findFreqTerms(dtm.tfidf.2 , i) ))
#     }
    
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
    
#     for (i in seq(0.1,30,by = 4) ) {
#       cat(".lowfreq > ",i,"\n")
#       print(summary(findFreqTerms(dtm.tfidf.3 , i) ))
#     }
    
  } else {
    cat(">>> no 3-grams ...\n") 
  }
  
  ## corpra are not necessary any more 
  rm(cp)
  
  ##### Convert matrices 
  cat ("converting dtm.tfidf.1 ... \n")
  dtm.tfidf.1.df <- as.data.frame(inspect( dtm.tfidf.1 ))
  cat ("dtm.tfidf.1.df - dim: ",dim(dtm.tfidf.1.df),"\n")
  #print(dtm.tfidf.1.df[1:5,1:5])
  
  dtm.tfidf.df = dtm.tfidf.1.df
  
  rm(dtm.tfidf.1)
  rm(dtm.tfidf.1.df)
  
  if ( as.logical(settings[settings$variant == "bigrams" , ]$value) ) { 
    cat ("converting dtm.tfidf.2 ... \n")
    dtm.tfidf.2.df <- as.data.frame(inspect( dtm.tfidf.2 ))
    cat ("dtm.tfidf.2.df - dim: ",dim(dtm.tfidf.2.df),"\n")
    #print(dtm.tfidf.2.df[1:5,1:5])
    
    dtm.tfidf.df = cbind(dtm.tfidf.df , dtm.tfidf.2.df)
    
    rm(dtm.tfidf.2)
    rm(dtm.tfidf.2.df)
  }
  
  if ( as.logical(settings[settings$variant == "trigrams" , ]$value) ) { 
    cat ("converting dtm.tfidf.3 ... \n")
    dtm.tfidf.3.df <- as.data.frame(inspect( dtm.tfidf.3 ))
    cat ("dtm.tfidf.3.df - dim: ",dim(dtm.tfidf.3.df),"\n")
    #print(dtm.tfidf.3.df[1:5,1:5])
    
    dtm.tfidf.df = cbind(dtm.tfidf.df , dtm.tfidf.3.df)
    
    rm(dtm.tfidf.3)
    rm(dtm.tfidf.3.df)
  }
  
  cat (">>> dtm.tfidf.df dim: ",dim(dtm.tfidf.df),"\n")
  
  return(dtm.tfidf.df)
}

makeFeatures <- function(train,test,nouns,adjectives,settings) {
  #### make syntactic features 
  sf = makeSyntacticFeatures (train,test,nouns,adjectives)
  
  #### make grams 
  dtm.tfidf.df = makeGrams (train,test,settings)
  
  ### binding other features 
  cat ("> binding other features ... \n")
  
  dtm.tfidf.df = cbind(dtm.tfidf.df , qlen = sf$qlen)
  dtm.tfidf.df = cbind(dtm.tfidf.df , ptlen = sf$ptlen)
  dtm.tfidf.df = cbind(dtm.tfidf.df , pdlen = sf$pdlen)
  
  dtm.tfidf.df = cbind(dtm.tfidf.df , ufos.q = sf$ufos.q)
  dtm.tfidf.df = cbind(dtm.tfidf.df , ufos.pt = sf$ufos.pt)
  dtm.tfidf.df = cbind(dtm.tfidf.df , ufos.same = sf$ufos.same)
  
  dtm.tfidf.df = cbind(dtm.tfidf.df , sm = sf$sm)
  dtm.tfidf.df = cbind(dtm.tfidf.df , am = sf$am)
  dtm.tfidf.df = cbind(dtm.tfidf.df , amd = sf$amd)
  
  dtm.tfidf.df = cbind(dtm.tfidf.df , prtm = sf$prtm)
  dtm.tfidf.df = cbind(dtm.tfidf.df , pdm = sf$pdm)
  
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
  
  cat (">>> dtm.tfidf.df dim: ",dim(dtm.tfidf.df)," ... \n")
  #write_csv(dtm.tfidf.df , paste(getBasePath("data") , "base_matrix.csv" , sep=''))
  
  return(dtm.tfidf.df)
} 

get_labels = function(label_j, std , bootrap_factor) {
  if (bootrap_factor != 8) stop("only bootrap_factor == 8 supported")
  grid = expand.grid(x1 = 1:4 , x2 = 1:4 , x3 = 1:4 , x4 = 1:4 , x5 = 1:4 , x6 = 1:4 , x7 = 1:4 , x8 = 1:4)
  
  mean_j = apply(grid,1,function(x) mean(x))
  std_j = apply(grid,1,function(x) sd(x))
  
  grid$mean = mean_j
  grid$std = std_j
  
  grid$delta = abs(grid$std - std)
  grid_m = grid[(grid$mean > label_j-0.5 & grid$mean < label_j+0.5),]
  idx = which(grid_m$delta == min(grid_m$delta))  
  if ( length(idx) > 1 ) idx = idx[1]
  
  list(x1=grid_m[idx,]$x1, x2=grid_m[idx,]$x2, x3=grid_m[idx,]$x3, x4=grid_m[idx,]$x4 , 
       x5=grid_m[idx,]$x5, x6=grid_m[idx,]$x6, x7=grid_m[idx,]$x7, x8=grid_m[idx,]$x8)
}

makeBoostrap = function(x,trind,teind,ytrain,bootrap_factor) {
  x.train = x[trind,]
  cat(">>> old train set:",dim(x.train),"..\n")
  for (k in 1:(bootrap_factor-1)) {
    x.train = rbind(x.train,x[trind,])
  }
  cat(">>> new train set:",dim(x.train),"..\n")
  
  x = rbind(x.train,x[teind,])
  trind = 1:nrow(x.train)
  teind = (nrow(x.train)+1):nrow(x)
  cat(">>> new data set:",dim(x),"..\n")
  
  ## boostrap - labels 
  cat(">>> computing labels ... \n") 
  cat(">>> old label set:",length(ytrain),"..\n")
  ytrain = rep(ytrain,bootrap_factor)
  cat(">>> new label set:",length(ytrain),"..\n")
  
  for (j in 1:length_init_train) {
    labels_j = get_labels(Xtrain[j,]$median_relevance , Xtrain[j,]$relevance_variance ,  bootrap_factor)
    ytrain[j] = labels_j$x1 
    ytrain[j+length_init_train] = labels_j$x2
    ytrain[j+2*length_init_train] = labels_j$x3
    ytrain[j+3*length_init_train] = labels_j$x4
    ytrain[j+4*length_init_train] = labels_j$x5
    ytrain[j+5*length_init_train] = labels_j$x6
    ytrain[j+6*length_init_train] = labels_j$x7
    ytrain[j+7*length_init_train] = labels_j$x8
  }
  
  return(list(
    x = x ,
    x.train = x.train , 
    trind = trind , 
    teind = teind , 
    ytrain = ytrain
    ))
  
}

get_early_stop = function(param,x,trind,ytrain,nfold=5,cv.nround=3000) {
  ### Cross-validation 
  cat(">>Cross Validation ... \n")
  inCV = T
  xval.perf = -1
  bst.cv = NULL
  early.stop = -1 
  
  cat(">> cv.nround: ",cv.nround,"\n") 
  
  while (inCV) {
    cat(">>> maximizing ScoreQuadraticWeightedKappa ...\n")
    
    bst.cv = xgb.cv(param=param, data = x[trind,], label = ytrain, 
                    nfold = nfold, nrounds=cv.nround , 
                    feval = ScoreQuadraticWeightedKappa , maximize = T)
    
    print(bst.cv)
    max_perf = max(bst.cv[!is.na(bst.cv$test.qwk.mean),]$test.qwk.mean)
    early.stop = which(bst.cv$test.qwk.mean == max_perf )
    if (length(early.stop) > 0) {
      early.stop = early.stop[1]
    }
    xval.perf = bst.cv[early.stop,]$test.qwk.mean
    cat(">> early.stop: ",early.stop," [xval.perf:",xval.perf,"]\n") 
    
    if (early.stop < cv.nround) {
      inCV = F
      cat(">> stopping [early.stop < cv.nround=",cv.nround,"] ... \n") 
    } else {
      cat(">> redo-cv [early.stop == cv.nround=",cv.nround,"] with 2 * cv.nround ... \n") 
      cv.nround = cv.nround * 2 
    }
    
    gc()
  }
  
  return( list(early_stop = early.stop , xval_qwk = xval.perf) )
}

predict_xgb = function(param,x,trind,ytrain,early.stop) {
  ### Prediction 
  bst = NULL
  
  cat(">>> maximizing ScoreQuadraticWeightedKappa ...\n")
  dtrain <- xgb.DMatrix(x[trind,], label = ytrain)
  watchlist <- list(train = dtrain)
  bst = xgb.train(param = param, dtrain , 
                  nrounds = early.stop, watchlist = watchlist , 
                  feval = ScoreQuadraticWeightedKappa , maximize = T , verbose = 1)
  
  cat(">> Making prediction ... \n")
  pred = predict(bst,x[teind,])
  pred = pred + 1 
  
  print(">> prediction <<")
  print(table(pred))
  
  print(">> train set labels <<")
  print(table(ytrain+1))
  
  return(pred)
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

########################################################################## Settings 
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
fn = paste("sub_boot_each_",paste(settings$variant,settings$value,sep='',collapse = "_"),".csv",sep='')
cat(">> saving prediction on",fn,"...\n")

#### Data 
sampleSubmission <- read_csv(paste(getBasePath("data") , "sampleSubmission.csv" , sep=''))
train <- read_csv(paste(getBasePath("data") , "train.csv" , sep=''))
test  <- read_csv(paste(getBasePath("data") , "test.csv" , sep=''))
#digest_df = read_csv(paste(getBasePath("data") , "base_matrix059.csv" , sep=''))

## nouns , adjectives
nouns <-  read.table(paste(getBasePath("data") , "nouns91K.txt" , sep=''), header=F , sep="" , colClasses = 'character') 
nouns = as.character(nouns[,1])
adjectives <-  read.table(paste(getBasePath("data") , "adjectives28K.txt" , sep=''), header=F , sep="" , colClasses = 'character') 
adjectives = as.character(adjectives[,1])

### parts to assemble here 
add_cosine = T 
cosine_df = read_csv(paste(getBasePath("data") , "cosine_6B.300d.csv" , sep=''))

##### xgboost 
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


#### 
cat(">>Params:\n")
print(param)

##################################
qcorp = myCorpus(c(train$query,test$query) , allow_numbers = T , do_stemming =  F )
q_map <- new.env(hash = T, parent = emptyenv())
 lapply(qcorp , function(x) {
   q_map[[x$content]] = x$content
 })

load(file=paste(getBasePath("data") , "queries" , sep=''))
load(file=paste(getBasePath("data") , "my_cluster_knn" , sep=''))

queries_2 = ls(q_map)
### check 
for (i in 1:length(queries_2)) 
  if (queries_2[i] != queries[i] ) 
    stop("queries_2[i] != queries[i]") 
####

for (i in 1:length(queries))
  q_map[[queries[i]]] = i

## q_class 
q_class = rep(NA,nrow(train)+nrow(test))
train$q_class = NA
test$q_class = NA
for (i in 1:(nrow(train)+nrow(test))) {
  if (i <= nrow(train)) train[i,]$q_class = q_map[[qcorp[[i]]$content]]
  else test[(i-nrow(train)),]$q_class = q_map[[qcorp[[i]]$content]]
  q_class[i] = q_map[[qcorp[[i]]$content]]
}

train_qcard = ddply(train , .(q_class) , function(x) c(num=nrow(x)) )
test_qcard = ddply(test , .(q_class) , function(x) c(num=nrow(x)) )
if (sum(train_qcard$num) != nrow(train)) stop("something wrong")
if (sum(test_qcard$num) != nrow(test)) stop("something wrong")

#### cluter 
cluster_card = max(unique(my_cluster))

##set.seed(1234)
#set.seed(3456)
#my_cluster = kfolds(cluster_card,261)
#cat(">> saving on disk clusters ... \n")
#save(my_cluster,file=paste(getBasePath("data") , "my_cluster.csv" , sep=''))

##### do the work  
do_boostap = F 
bootrap_factor = 8

### state objects 
pred = rep(NA,nrow(test))
cv_grid = data.frame(cluter=1:cluster_card, train_obs = NA , early_stop = NA , xval_qwk = NA )

### do the job  
for (i in 1:cluster_card) {
  cat(">>> processing cluster ",i,"/",cluster_card,"...\n")
  cluster_idx = which(my_cluster == i)
  train_idx = NULL 
  test_idx = NULL 
  for (qr in cluster_idx) {
    train_idx = c(train_idx,which(train$q_class == qr)) 
    test_idx = c(test_idx,which(test$q_class == qr))
  }
  Xtrain = train[train_idx,]
  ytrain = train[train_idx,]$median_relevance
  Xtest = test[test_idx,]
  
  #### make features 
  features.df = makeFeatures (Xtrain,Xtest,nouns,adjectives,settings)
  if (add_cosine) {
    cat(">>> adding cosine similarity ... \n")
    features.df = cbind(features.df,cosine_similar=cosine_df[c(train_idx,test_idx),]$cosine_vect)  
  }
  cat("***** head featureset ***** \n")
  print(head(features.df))
  cat("***** tail featureset ***** \n")
  print(tail(features.df))
  
  ##
  x = as.matrix(features.df)
  x = matrix(as.numeric(x),nrow(x),ncol(x))
  rm(features.df)
  trind = 1:nrow(Xtrain)
  teind = (nrow(Xtrain)+1):nrow(x)
  length_init_train = max(trind)
  
  #### boostrap - train/test matrix 
  if (do_boostap) {
    cat(">>>> making bootstrap [bootrap_factor=",bootrap_factor,"] ... \n")
    boot = makeBoostrap(x,trind,teind,ytrain,bootrap_factor)
    ytrain = (boot$ytrain)-1 
    x = boot$x 
    trind = boot$trind 
    teind = boot$teind 
  } else {
    ytrain = ytrain-1 
  }
  
  #### cross-validation 
  cross_val = NULL
  early_stop = xval_qwk = -1
  cross_val = tryCatch({
    get_early_stop (param,x,trind,ytrain,nfold=5,cv.nround=1800)
  } , error = function(err) { 
    print(paste("ERROR:  ",err))
    NULL
  })
  if (is.null(cross_val) ) {
    early_stop = 800
    xval_qwk = -1 
  } else {
    early_stop = cross_val$early_stop  
    xval_qwk = cross_val$xval_qwk
  }
  
  #### predict 
  pred_j = predict_xgb (param,x,trind,ytrain,early_stop)
  
  #### assemble 
  pred[test_idx] = pred_j
  
  ### update cv_grid 
  cv_grid[cv_grid$cluter == i , ]$train_obs = length(train_idx)
  cv_grid[cv_grid$cluter == i , ]$early_stop = early_stop
  cv_grid[cv_grid$cluter == i , ]$xval_qwk = xval_qwk
  
  #### dumping state for XGB errors like R(15152,0x7fff7b60b300) malloc: *** error for object 0x7f844b94d800: incorrect checksum for freed object - object was probably modified after being freed.
  cat(">> writing TEMP prediction on disk [",fn,"]... \n")
  write_csv(data.frame(id = sampleSubmission$id , prediction = pred) , paste(getBasePath("data") , "sub_dump.csv" , sep=''))
  
  ## cv_grid
  cat(">> writing TEMP cv_grid on disk ... \n")
  write_csv(cv_grid , paste(getBasePath("data") , "cv_grid_dump.csv" , sep=''))
}

if ( sum(is.na(pred)) > 0 ) stop("something wrong on pred (NAs)")

### write on disk 
#fn = paste("sub_2gen___xval",xval.perf,".csv",sep='')
cat(">> writing prediction on disk [",fn,"]... \n")
write_csv(data.frame(id = sampleSubmission$id , prediction = pred) , paste(getBasePath("data") , fn , sep=''))

## cv_grid
cat(">> writing cv_grid on disk ... \n")
write_csv(cv_grid , paste(getBasePath("data") , "cv_grid.csv" , sep=''))

cv_mean_qwk = (cv_grid[cv_grid$xval_qwk != 1 ,]$train_obs %*% cv_grid[cv_grid$xval_qwk != 1 ,]$xval_qwk)/sum(cv_grid[cv_grid$xval_qwk != 1 ,]$train_obs)
cat(">> mean qwk on cross validation:",cv_mean_qwk,"\n")
