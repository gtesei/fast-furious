Polynomial 
========================================================

This is an R Markdown document. 

### Pure Polynomial - Data set 

Let's generate a pure (= no noise added) data set. 


```r
yfun <- function(x) {
    ret <- vector("numeric", length = dim(x)[1])
    for (i in 1:dim(x)[1]) {
        ret[i] <- 4 + 3 * (x[i, 1]^3) - x[i, 2]^4 + 1.4 * (x[i, 3]^3) - 3.2 * 
            (x[i, 4]^4) - 0.5 * (x[i, 5]^3)
    }
    ret
}
s <- seq(1, 10, length = 3500)
s <- sample(s, replace = FALSE)
X <- matrix(s, nrow = 700, ncol = 5)
y <- yfun(X)

Xtrain <- X[1:500, ]
Xval <- X[501:700, ]
ytrain <- y[1:500]
yval <- y[501:700]

write.table(Xtrain, file = "poly_pure_Xtrain.zat", eol = "\n", quote = FALSE, 
    col.names = FALSE, row.names = FALSE, sep = ",")
write.table(ytrain, file = "poly_pure_ytrain.zat", eol = "\n", quote = FALSE, 
    col.names = FALSE, row.names = FALSE, sep = ",")
write.table(Xval, file = "poly_pure_Xval.zat", eol = "\n", quote = FALSE, col.names = FALSE, 
    row.names = FALSE, sep = ",")
write.table(yval, file = "poly_pure_yval.zat", eol = "\n", quote = FALSE, col.names = FALSE, 
    row.names = FALSE, sep = ",")
```


### Pure Polynomial - Fitting 

Let's fit with a linear model. 


```r
train <- data.frame(y = ytrain, x1 = Xtrain[, 1], x2 = Xtrain[, 2], x3 = Xtrain[, 
    3], x4 = Xtrain[, 4], x5 = Xtrain[, 5])
val <- data.frame(y = yval, x1 = Xval[, 1], x2 = Xval[, 2], x3 = Xval[, 3], 
    x4 = Xval[, 4], x5 = Xval[, 5])
lm1 <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = train)
summary(lm1)
```

```
## 
## Call:
## lm(formula = y ~ x1 + x2 + x3 + x4 + x5, data = train)
## 
## Residuals:
##    Min     1Q Median     3Q    Max 
## -12335  -2751    740   3278   6377 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  10767.6      927.7   11.61  < 2e-16 ***
## x1             349.4       71.0    4.92  1.2e-06 ***
## x2           -1083.8       71.0  -15.27  < 2e-16 ***
## x3             185.8       69.9    2.66   0.0081 ** 
## x4           -2838.4       71.1  -39.92  < 2e-16 ***
## x5             -37.7       72.7   -0.52   0.6047    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 4090 on 494 degrees of freedom
## Multiple R-squared:  0.791,	Adjusted R-squared:  0.789 
## F-statistic:  375 on 5 and 494 DF,  p-value: <2e-16
```

```r
ypred <- predict(lm1, newdata = val)
write.table(ypred, file = "poly_pure_ypred.zat", eol = "\n", quote = FALSE, 
    col.names = FALSE, row.names = FALSE, sep = ",")
RMSEtrain <- sqrt(sum((lm1$fitted - train$y)^2))/length(train$y)
RMSEval <- sqrt(sum((predict(lm1, newdata = val) - val$y)^2))/length(val$y)

RMSEtrain
```

```
## [1] 181.6
```

```r
RMSEval
```

```
## [1] 310.1
```


