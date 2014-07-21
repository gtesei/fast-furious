
## by yr@Kaggle

# This demo uses a toy dataset. If you want to play with Triskelion's features in VW format, 
# first convert/save them in csv format, and then use read.csv or fread (I would recommend
# the latter) to import the data. After that, everything should work in a similar manner.
# 
# Thanks to @tks, the conversion could be easily done. See the
# following link:
# http://www.kaggle.com/c/acquire-valued-shoppers-challenge/forums/t/7917/
#  performance-of-logistic-regression-on-features-described-by-triskelion/48837#post48837
#


##############
## Quantreg ##
##############

# install the quantreg package if necessary
if(!suppressWarnings(require('quantreg', character.only=TRUE, quietly=TRUE))){
  install.packages('quantreg')
}

# load quantreg library
require(quantreg)
# check the help doc in R for usage (I would suggest you use RStudio)
?rq

# load demo data
data(engel)
# engel is just a simple data.frame with two variables income and foodexp
# you can easily use your own data in a similar way
head(engel, 10)
edit(engel)

# fit a quantreg model
rq.tau <- 0.5 # you can use other tau like 0.6
rq.model <- rq(foodexp~., data=engel, tau=rq.tau)
# show some summary info
summary(rq.model)

# make prediction on new data (we here simply use the training data)
foodexp.rq.pred <- predict(rq.model, new.data=engel)


#########
## GBM ##
#########

# install the gbm package if necessary
if(!suppressWarnings(require('gbm', character.only=TRUE, quietly=TRUE))){
  install.packages('gbm')
}

# load gbm library
require(gbm)
# check the help doc in R for usage (I would suggest you use RStudio)
?gbm

# fit a gbm model with quantile regression loss
gbm.tau <- 0.6 # you can use other tau like 0.6
gbm.n.trees <- 200
gbm.model <- gbm(foodexp ~ income ,  data=engel, distribution=list(name='quantile', alpha=gbm.tau), n.trees = gbm.n.trees, interaction.depth = 1, n.minobsinnode = 10, shrinkage = 0.1, bag.fraction = 0.5, train.fraction = 1.0, verbose=FALSE)
# show some summary info
summary(gbm.model)
# plot oob estimated error of the fit
suppressWarnings(gbm.perf(gbm.model))

# make prediction on new data (we here simply use the training data)
foodexp.gbm.pred <- predict(gbm.model, new.data=engel, n.trees=gbm.n.trees)


####################
## QuantregForest ##
####################

# install the quantregForest package if necessary
if(!suppressWarnings(require('quantregForest', character.only=TRUE, quietly=TRUE))){
  install.packages('quantregForest')
}

# load quantregForest library
require(quantregForest)
# check the help doc in R for usage (I would suggest you use RStudio)
?quantregForest

# fit a quantregForest model with quantile regression loss
quanregForest.tau <- 0.6 # you can use other tau like 0.6
quanregForest.ntree <- 100
nTrain <- nrow(engel)
x <- matrix(engel$income, nTrain, 1)
y <- engel$foodexp
quantregForest.model <- quantregForest(x=x, y=y, mtry = 1, nodesize = 10, ntree = quanregForest.ntree)
# show some summary info
summary(quantregForest.model)

# make prediction on new data (we here simply use the training data)
foodexp.quantregForest.pred <- predict(quantregForest.model, new.data=engel, quantiles=quanregForest.tau )


###################
## Visualization ##
###################

par(mfrow=c(1,1))

# the ground truth
plot(engel$income,  engel$foodexp, lwd=2,
     xlab='income', ylab='foodexp', main='Quantile Regression', type = "n", cex=.5)
points(engel$income, engel$foodexp, lwd=1, col='black')

# quantreg
lines(engel$income, foodexp.rq.pred, lwd=1, col='blue')

# gbm
points(engel$income, foodexp.gbm.pred, lwd=1, col='red')
# or using lines for gbm too (a bit messy...)
#lines(engel$income, foodexp.gbm.pred, lwd=1, col='red')

# quantregForest
points(engel$income, foodexp.quantregForest.pred, lwd=1, col='green')

# legend
legend(2500, 450, c('Observation',
                    paste('Quantreg with tau = ', rq.tau, sep=''),
                    paste('GBM (Quantile Regression with tau = ', gbm.tau, ')',sep=''),
                    paste('QuantregForest with tau = ', quanregForest.tau,sep='')),
       col=c('black', 'blue','red','green'), lty = c(0,1,0,0), lwd=c(1,2,1,1), pch=c(1,NA_integer_, 1, 1))

