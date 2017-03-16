######################
library(Hmisc)
library(plyr)
library(ggplot2)
library(corrplot)

library(ggfortify)
library(reshape2)
library(DT)


####
library(tm)

library(NLP)

require(xgboost)
require(methods)

require(plyr)
library(dplyr)

library(topicmodels)
library(tidytext)

## CONFIG
PREF = "C:/Users/gtesei/Desktop/Deloitte/Projects/Catalyst_Fund/Audit_Core/2_round/Rollover/QA_dump_with_significant_digits/"
PREF_OUT = "C:/Users/gtesei/Desktop/Deloitte/Projects/Catalyst_Fund/Audit_Core/2_round/Rollover/lda/"

.files = list.files( PREF )
files = list()
data_sets = list()
i = 1 
for (file in .files) {
  if (  endsWith(file, ".csv") ) {
    files[[i]] = file
    data_sets[[i]] = list(name = file, data = read.csv(paste0(PREF,file),stringsAsFactors=F))  
    i = i + 1
  }
}


## f_num = 1 , file = "MSFT-2015-10K.csv"
f_num = 1 
ds = data_sets[[f_num]]$data 
fn = data_sets[[f_num]]$name 
fn$clean_label = NA

# # clean_txt 
# clean_txt <- function(txt){
#   txt.lower <- tolower(txt)
#   txt.lower.l <- strsplit(txt.lower, "\\W")
#   txt.word.v <- unlist(txt.lower.l)
#   
#   ## remove blanks 
#   not.blanks.v  <-  which(txt.word.v!="")
#   txt.word.v <-  txt.word.v[not.blanks.v]
#   
#   return(paste(txt.word.v,collapse = " " ))
# }

clean_txt <- function(txt){ 
  txt.lower <- tolower(txt)
  txt.lower.l = gsub("\\W"," ",txt.lower) 
  
  txt.lower.m = gsub("\\b(jan(uary)?|feb(ruary)?|mar(ch)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?)\\b","",txt.lower.l); 
  
  txt.lower.n = gsub("\\b(million(s)?|billion(s)?|thousands(s)?|hundred(s)?|ten(s)?)\\b","",txt.lower.m); 
  
  txt.lower.s = gsub("\\s+"," ",txt.lower.n) 
  
  return(txt.lower.s)
}



myCorpus = function(documents , allow_numbers = F , do_stemming = F , clean = T) {
  cp = NULL 
  
  if (clean)
    cp <- Corpus(VectorSource(clean_txt(documents)))
  else 
    cp <- Corpus(VectorSource(documents))
  
  cp <- tm_map( cp, content_transformer(tolower)) 
  cp <- tm_map( cp, content_transformer(removePunctuation))
  
  cp <- tm_map( cp, removeWords, stopwords('english'))
  
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


### remove -- Unable to recognize label -- 
ds <- ds[ds$USER_LABEL!="Unable to recognize label" , ]


cp <- myCorpus(ds$USER_LABEL)

print(cp)

for (i in sample(x = 1:length(cp),size = 10) ) {
  cat("\n****************************************[",i,"]*************************************\n")
  cat(">>> Raw: \n")
  print(ds$USER_LABEL[i])
  cat(">>> Processed: \n")
  print(cp[[i]]$content)
}



############################################
######## PARAMETERS 
############################################



do_iter <- function(K_LDA,K_KMEANS) {
  ################## topic models 
  dtm <- DocumentTermMatrix(cp, control = list(stemming = F, stopwords = F, removeNumbers = F, removePunctuation = F))
  cat(">>> document-term matrix dim:",dim(dtm),"\n")
  rowTotals <- apply(dtm , 1, sum) #Find the sum of words in each Document
  cat(">> Removing 0-rows in term-matrix: ", which(rowTotals == 0),"\n")
  dtm.new   <- dtm[rowTotals> 0, ] #remove all docs without words
  cat(">>> document-term matrix dim:",dim(dtm.new),"\n")
  
  lb_lda <- LDA(dtm.new, k = K_LDA, control = list(seed = 1234))
  lb_lda
  
  ### per-topic-per-word probabilities.
  label_lda_td <- tidy(lb_lda)
  label_lda_td
  
  library(ggplot2)
  library(dplyr)
  
  ## top words per topic 
  ap_top_terms <- label_lda_td %>%
    group_by(topic) %>%
    top_n(10, beta) %>%
    ungroup() %>%
    arrange(topic, -beta) 
  
  ## top words per topic as data.frame 
  ap_top_terms.df <- label_lda_td %>%
    group_by(topic) %>%
    top_n(10, beta) %>%
    ungroup() %>%
    arrange(topic, -beta) %>%
    as.data.frame()
  
  
  ###  Document-topic probabilities
  ap_gamma <- tidy(lb_lda, matrix = "gamma")
  ap_gamma
  
  ## top topic per document 
  ap_gamma_top <- ap_gamma %>%
    group_by(document) %>%
    top_n(2, gamma) %>%        #### todo better: if n.2 == n.1 set up NA 
    ungroup() %>%
    arrange(document, -gamma , topic) %>%
    as.data.frame()
  
  ap_gamma_top$document <- as.numeric(ap_gamma_top$document)
  ap_gamma_top <- ap_gamma_top[order(ap_gamma_top$document , decreasing = F),]
  
  ap_gamma_top_elab <- ddply( ap_gamma_top , .(document) , function(x) c( 
    gamma = max(x$gamma), topic = x[which.max(x$gamma),]$topic ,maxs = length(which.max(x$gamma)) , n = nrow(x) )) 
  
  ap_gamma_top_elab$ok = ap_gamma_top_elab$maxs == 1 
  
  final_ds = ap_gamma_top_elab
  final_ds <- cbind(final_ds , ds[-which(rowTotals == 0),])
  
  
  #final_ds$label <- ds$USER_LABEL[-which(rowTotals == 0)]
  
  
  ## serialize 
  #fn <- paste(data_sets[[f_num]]$name,"___lda_2000.csv",sep = "")
  #write.csv(x = final_ds,file = paste0(PREF_OUT,fn))
  
  ############# clustering 
  dtm.new.mat <- dtm.new %>% as.matrix()
  
  cl <- kmeans(x = dtm.new.mat , centers = K_KMEANS)
  
  final_ds$clid = cl$cluster
  
  ## serialize 
  fn <- paste(data_sets[[f_num]]$name,"___lda_2000__cluster.csv",sep = "")
  write.csv(x = final_ds,file = paste0(PREF_OUT,fn))
  
  #### evaluate accuracy 
  final_ds$clust_topic = as.numeric(paste0(final_ds$topic,final_ds$clid))
  topic_ntags = ddply(final_ds , .(topic) , function(x) c (ntags=length(unique(x$TAG))) )
  cluster_ntags = ddply(final_ds , .(clid) , function(x) c (ntags=length(unique(x$TAG))) )
  topic_cluste_ntags = ddply(final_ds , .(clust_topic) , function(x) c (ntags=length(unique(x$TAG))) )
  
  final_ds$ok_topic_clust = NA
  final_ds$ok_topic = NA
  final_ds$ok_cluster = NA
  for (i in 1:nrow(final_ds)) {
    ntags_topic = topic_ntags[topic_ntags$topic==final_ds[i,]$topic,]$ntags
    ntags_cluster = cluster_ntags[cluster_ntags$clid==final_ds[i,]$clid,]$ntags
    ntags_topic_cluster = topic_cluste_ntags[topic_cluste_ntags$clust_topic==final_ds[i,]$clust_topic,]$ntags
    
    if (ntags_topic > 1) {
      final_ds[i,]$ok_topic = 0
    } else {
      final_ds[i,]$ok_topic = 1
    }
    
    if (ntags_cluster > 1) {
      final_ds[i,]$ok_cluster = 0
    } else {
      final_ds[i,]$ok_cluster = 1
    }
    
    if (ntags_topic_cluster > 1) {
      final_ds[i,]$ok_topic_clust = 0
    } else {
      final_ds[i,]$ok_topic_clust = 1
    }
  }
  
  ##
  acc_topic = sum(final_ds$ok_topic)/nrow(final_ds)
  acc_cluster = sum(final_ds$ok_cluster)/nrow(final_ds)
  acc_topic_cluster = sum(final_ds$ok_topic_clust)/nrow(final_ds)
  
  cat(">>> acc_topic: ",acc_topic,"\n")
  cat(">>> acc_cluster: ",acc_cluster,"\n")
  cat(">>> acc_topic_cluster: ",acc_topic_cluster,"\n")
  
  return (list(acc_topic=acc_topic,acc_cluster=acc_cluster,acc_topic_cluster=acc_topic_cluster))
  
}

################################ TUNING 

#K_LDA = 2000
#K_KMEANS = 800

k_topic =  seq(500,3000,by=100)
k_cluster =  seq(500,3000,by=100)

grid = expand.grid(k_topic=k_topic,k_cluster=k_cluster)

grid$acc_topic = NA 
grid$acc_cluster = NA 
grid$acc_topic_cluster = NA 


for (i in 1:norw(grid)) {
  k_topic = grid[i,]$k_topic
  k_cluster = grid[i,]$k_cluster
  cat(">>>>>>>>>>>>>>>>>>>>>>>>>>>> [",i,"]  --- k_topic=",k_topic,"  k_cluster=",k_cluster,"\n")
  
  l <- do_iter(k_topic,k_cluster)
  
  grid[i,]$acc_topic <- l$acc_topic
  grid[i,]$acc_cluster <- l$acc_cluster 
  grid[i,]$acc_topic_cluster <- l$acc_topic_cluster
  
}


## serialize 
fn <- paste("tuning.csv",sep = "")
write.csv(x = grid,file = paste0(PREF_OUT,fn))








