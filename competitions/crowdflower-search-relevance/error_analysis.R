

trainIndex <- createDataPartition(y, p = .7,
                                  list = FALSE,
                                  times = 1)

test.idx = which(! (1:nrow(train) %in% trainIndex)) 

xx = x[trind,]
x.train <- xx[trainIndex,]
x.test  <- xx[test.idx,]


y.train = y[trainIndex]
y.test = y[test.idx]

###
cat(">>> maximizing ScoreQuadraticWeightedKappa ...\n")
dtrain <- xgb.DMatrix(x.train, label = y.train)
watchlist <- list(train = dtrain)
bst = xgb.train(param = param, dtrain , 
                nrounds = early.stop, watchlist = watchlist , 
                feval = ScoreQuadraticWeightedKappa , maximize = T , verbose = 1)

cat(">> Making prediction ... \n")
pred = predict(bst,x.test)
pred = pred + 1 

labels_ok = which(pred == (y.test+1))
labels_nok = which(pred != (y.test+1))

length(labels_ok) / length(y.test)  #0.6629472
 

classified = train[ test.idx , c('query','product_title','median_relevance')]
classified$pred = pred

classified$ok = (classified$pred == classified$median_relevance)

sum(classified$pred == classified$median_relevance) / nrow(classified)

ddply ( classified , .(median_relevance) , function(x) c( perc_ok =  table(x$ok)/nrow(x) )  ) 

# median_relevance perc_ok.FALSE perc_ok.TRUE
# 1                1     0.5425101    0.4574899
# 2                2     0.6479482    0.3520518
# 3                3     0.7839506    0.2160494
# 4                4     0.1145327    0.8854673

ddply ( classified[classified$median_relevance ==1 , ] , .(median_relevance) , function(x) c( perc_ok =  table(x$ok) , perc_2 = sum(x$pred==2) , perc_3 = sum(x$pred==3) , perc_4 = sum(x$pred==4) )  )

# median_relevance perc_ok.FALSE perc_ok.TRUE perc_2 perc_3 perc_4
# 1                1           134          113     55     25     54


ddply ( classified[classified$median_relevance ==2 , ] , .(median_relevance) , function(x) c( perc_ok =  table(x$ok) , perc_1 = sum(x$pred==1) , perc_3 = sum(x$pred==3) , perc_4 = sum(x$pred==4) )  )

# median_relevance perc_ok.FALSE perc_ok.TRUE perc_1 perc_3 perc_4
# 1                2           300          163     42     89    169

ddply ( classified[classified$median_relevance ==3 , ] , .(median_relevance) , function(x) c( perc_ok =  table(x$ok) , perc_1 = sum(x$pred==1) , perc_2 = sum(x$pred==2) , perc_4 = sum(x$pred==4) )  )

# median_relevance perc_ok.FALSE perc_ok.TRUE perc_1 perc_2 perc_4
# 1                3           381          105     15     60    306

ddply ( classified[classified$median_relevance ==4 , ] , .(median_relevance) , function(x) c( perc_ok =  table(x$ok) , perc_1 = sum(x$pred==1) , perc_2 = sum(x$pred==2) , perc_3 = sum(x$pred==3) )  )

# median_relevance perc_ok.FALSE perc_ok.TRUE perc_1 perc_2 perc_3
# 1                4           212         1639     16     51    145






