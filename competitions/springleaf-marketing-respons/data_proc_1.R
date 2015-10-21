library(data.table)
library(xgboost)
library(fastfurious)

### FUNCS

### CONFIG 
DEBUG = F

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/springleaf-marketing-respons')
ff.bindPath(type = 'code' , sub_path = 'competitions/springleaf-marketing-respons')

ff.bindPath(type = 'elab' , sub_path = 'dataset/springleaf-marketing-respons/elab',createDir = T) ## out 


### generate predictors' meta-table   
train = as.data.frame( fread(paste(ff.getPath("data") , "train.csv" , sep='') , stringsAsFactors = F))
test = as.data.frame( fread(paste(ff.getPath("data") , "test.csv" , sep='') , stringsAsFactors = F))

train = train[,-c(1,208,214,839,846,1427)]
test = test[,-c(1,208,214,839,846,1427)]
feature.names <- names(train)[2:ncol(train)-1]

meta.data = data.frame(feature.name = feature.names , 
                       num=NA, 
                       char=F, 
                       missing=NA, 
                       mean=NA,
                       sd=NA,
                       lowest_1 = NA, lowest_2=NA, lowest_3=NA, 
                       highest_1=NA, highest_2=NA, highest_3=NA, 
                       isCateg=NA, isDate=NA)

for (f in feature.names) {
  levels <- sort(unique(c(train[[f]], test[[f]])))
  
  if (class(train[[f]])=="character") {
    cat(">>> ",f," is character \n")
    meta.data[meta.data$feature.name==f,'char'] <- T
    train_f <- as.integer(factor(train[[f]], levels=levels))
    test_f  <- as.integer(factor(test[[f]],  levels=levels))
  }
  
  meta.data[meta.data$feature.name==f,'num'] <- length(levels)
  meta.data[meta.data$feature.name==f,'missing'] <- sum(is.na(c(train[[f]], test[[f]])))
  meta.data[meta.data$feature.name==f,'mean'] <- if (class(train[[f]])=="character") NA else mean(c(train[[f]], test[[f]]),na.rm = T)
  meta.data[meta.data$feature.name==f,'sd'] <- if (class(train[[f]])=="character") NA else sd(c(train[[f]], test[[f]]),na.rm = T)
  
  meta.data[meta.data$feature.name==f,'lowest_1'] <- levels[1]
  meta.data[meta.data$feature.name==f,'lowest_2'] <- levels[ if (length(levels)>=2) 2 else length(levels) ]
  meta.data[meta.data$feature.name==f,'lowest_3'] <- levels[ if (length(levels)>=3) 3 else length(levels) ]
  
  meta.data[meta.data$feature.name==f,'highest_1'] <- levels[ length(levels) ]
  meta.data[meta.data$feature.name==f,'highest_2'] <- levels[ if (length(levels)>=2) length(levels)-1 else length(levels) ]
  meta.data[meta.data$feature.name==f,'highest_3'] <- levels[ max(length(levels)-2,1) ]
  
}

cat(">>> writing on disk ... \n")
write.csv(meta.data,
          quote=FALSE, 
          file=paste0(ff.getPath("elab"),"meta.data.csv"),
          row.names=FALSE)








