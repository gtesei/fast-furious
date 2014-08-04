
######### Q -> C 
x = rnorm(n = 1000 , mean = 100 , sd =  20000)
y = factor(ifelse(x > 100 , "Yes" , "No" ))
train.df = data.frame(y = y , x = x)
mod = glm(y ~ x , family = binomial , data = train.df )
summary(mod)
x.test = rnorm(n = 500 , mean = 100 , sd =  20000)
test.df = data.frame(x = x.test)
pred = predict(mod , test.df , type = "response")
label0 = rownames(contrasts(y))[1]
label1 = rownames(contrasts(y))[2]
pred.test = ifelse(pred > 0.5 , label1 , label0)
y.test = ifelse(x.test > 100 , "Yes" , "No" )
table(y.test , pred.test)

test.anova = aov(x~y)
pvalue = summary(test.anova)[[1]][["Pr(>F)"]][1]
pvalue

## quadratic 
x = rnorm(n = 1000 , mean = 100 , sd =  50)
z = rnorm(n = 1000 , mean = 100 , sd =  50)
y = factor(ifelse( (x^2 + z^2) > 5000 , "Yes" , "No" ))
summary(y)
train.df = data.frame(y = y , x = x , z = z)

mod = glm(y ~ I(x^2) + I(z^2), family = binomial , data = train.df )
summary(mod)

x.test = rnorm(n = 500 , mean = 100 , sd =  50)
z.test = rnorm(n = 500 , mean = 100 , sd =  50)
test.df = data.frame(x = x.test , z = z.test)
pred = predict(mod , test.df , type = "response")
label0 = rownames(contrasts(y))[1]
label1 = rownames(contrasts(y))[2]
pred.test = ifelse(pred > 0.5 , label1 , label0)
y.test = factor(ifelse((x.test^2 + z.test^2) > 5000 , "Yes" , "No" ))
summary(y.test)
table(y.test , pred.test)

test.anova = aov(x~y)
pvalue = summary(test.anova)[[1]][["Pr(>F)"]][1]
pvalue

## interaction terms  
x = rnorm(n = 1000 , mean = 100 , sd =  50)
z = rnorm(n = 1000 , mean = 100 , sd =  50)
y = factor(ifelse( (x * z) > 5000 , "Yes" , "No" ))
summary(y)
train.df = data.frame(y = y , x = x , z = z)

mod = glm(y ~ x:z, family = binomial , data = train.df )
summary(mod)

x.test = rnorm(n = 500 , mean = 100 , sd =  50)
z.test = rnorm(n = 500 , mean = 100 , sd =  50)
test.df = data.frame(x = x.test , z = z.test)
pred = predict(mod , test.df , type = "response")
label0 = rownames(contrasts(y))[1]
label1 = rownames(contrasts(y))[2]
pred.test = ifelse(pred > 0.5 , label1 , label0)
y.test = factor(ifelse((x.test * z.test) > 5000 , "Yes" , "No" ))
summary(y.test)
table(y.test , pred.test)

test.anova = aov(x~y)
pvalue = summary(test.anova)[[1]][["Pr(>F)"]][1]
pvalue



######### Q -> Q
x = rnorm(n = 1000 , mean = 100 , sd =  20)
y = 3 * x + 3 
train.df = data.frame(y = y , x = x)
mod = lm(y ~ x ,  data = train.df )
summary(mod)
x.test = rnorm(n = 500 , mean = 100 , sd =  20)
y.test = 3 * x.test + 3 
test.df = data.frame(x = x.test)
pred = predict(mod , test.df)
mean(1/length(x.test)*(y.test - pred)^2)

test.corr = cor.test(x =  x , y =  y)
pvalue = test.corr$p.value
pvalue


### quadratic 
x = rnorm(n = 1000 , mean = 100 , sd =  20)
y = 3 * x^2 + 3 
train.df = data.frame(y = y , x = x)
mod = lm(y ~ x ,  data = train.df )
summary(mod)
x.test = rnorm(n = 500 , mean = 100 , sd =  20)
y.test = 3 * x.test^2 + 3 
test.df = data.frame(x = x.test)
pred = predict(mod , test.df)
mean(abs(y.test - pred))

test.corr = cor.test(x =  x , y =  y)
pvalue = test.corr$p.value
pvalue

### interaction terms  
x = rnorm(n = 1000 , mean = 100 , sd =  20)
z = rnorm(n = 1000 , mean = 100 , sd =  20)
y = 3 * x*z + 3 
train.df = data.frame(y = y , x = x , z = z)
mod = lm(y ~ x + z,  data = train.df )
summary(mod)
x.test = rnorm(n = 500 , mean = 100 , sd =  20)
z.test = rnorm(n = 500 , mean = 100 , sd =  20)
y.test = 3 * x.test*z.test + 3
test.df = data.frame(x = x.test , z = z.test)
pred = predict(mod , test.df)
mean(abs(y.test - pred))

test.corr = cor.test(x =  x , y =  y)
pvalue = test.corr$p.value
pvalue

test.corr = cor.test(x =  z , y =  y)
pvalue = test.corr$p.value
pvalue


######### C -> C 
x = factor(rep(1000,x=c("A","B","C") , 1000))
y = factor(ifelse(x == "A" , "Yes" , "No"))
train.df = data.frame(y = y , x = x)
mod = glm(y ~ x , data = train.df , family = binomial )
summary(mod)
x.test = factor(rep(1000,x=c("A","B","C") , 500))
y.test = factor(ifelse(x.test == "A" , "Yes" , "No"))
test.df = data.frame(y = y.test , x = x.test)
pred.probs = predict(mod , test.df , type = "response")
pred = ifelse(pred.probs > 0.5 , "Yes" , "No")
table(pred,y.test)
mean(y.test == pred)

test.chisq = chisq.test( x = x , y = y)
pvalue = test.corr$p.value
pvalue

######### C -> Q
x = factor(rep(1000,x=c("A","B","C") , 1000))
y = ifelse(x == "A" , 20 , 2)
train.df = data.frame(y = y , x = x)
mod = lm(y ~ x , data = train.df  )
summary(mod)
x.test = factor(rep(1000,x=c("A","B","C") , 500))
y.test = ifelse(x.test == "A" , 20 , 2)
test.df = data.frame(y = y.test , x = x.test)
pred = predict(mod , test.df )
mean(1/length(x.test)*(y.test - pred)^2)

test.anova = aov(y~x)
pvalue = summary(test.anova)[[1]][["Pr(>F)"]][1]
pvalue

