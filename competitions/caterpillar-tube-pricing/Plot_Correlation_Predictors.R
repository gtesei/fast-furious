
library(data.table)
library(lattice)

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/caterpillar-tube-pricing/competition_data"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/caterpillar-tube-pricing/competition_data/"
  } else if(type == "submission") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/caterpillar-tube-pricing"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/caterpillar-tube-pricing/"
  } else if(type == "elab") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/caterpillar-tube-pricing/elab"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/caterpillar-tube-pricing/elab/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/caterpillar-tube-pricing"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/caterpillar-tube-pricing/"
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


################# DATA IN 

train_set = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train_set.csv" , sep=''))) 

tube = as.data.frame( fread(paste(getBasePath("data") , 
                                  "tube.csv" , sep='')))


#################

data = train_set[,c('tube_assembly_id' , 'annual_usage' , 'min_order_quantity' , 'quantity' , 'bracket_pricing' , 'cost')]
data = merge(x = data , y = tube , by='tube_assembly_id' , all = F)

lev_th = c(2 , 5 , 9 , 14 , 24 , 48 , 100 , 2501 ) 
data$qty_lev = NA 
data = data[order(data$quantity , decreasing = F),]
curr_lev = 1 
for (i in  1:nrow(data)) {
  if (data[i,]$quantity < lev_th[curr_lev] ){
    data[i,]$qty_lev = curr_lev 
  } else {
    curr_lev = curr_lev + 1 
    data[i,]$qty_lev = curr_lev 
  }
}

data$qty_lev = factor(data$qty_lev)
levels(data$qty_lev) = c( 'qty [1,2]'    , 'qty (2,5]'   , 'qty (5,9]'   , 
                          'qty (9,14]'   , 'qty (14,24]' , 'qty (24,48]' , 
                          'qty (48,100]' , 'qty (100,2500]' )

xyplot(data$cost ~ data$diameter | data$qty_lev, panel = function(x, y, ...) {
  panel.xyplot(x, y, ...)
  lm1 <- lm(y ~ x)
  lm1sum <- summary(lm1)
  r2 <- lm1sum$adj.r.squared
  p <- lm1sum$coefficients[2, 4]
  panel.abline(lm1 , col.line = 'green', lty=1, lwd=1.5 )
  
  xt <- x[x==min(x)] 
  if (length(xt)>1) xt=xt[1]
  yt <- y[x==max(x)] 
  yt = yt + 200
  if (length(yt)>1) yt=yt[1]
  yyt = yt + 400
  
  panel.text(labels = bquote(italic(R)^2 == .(format(r2, digits = 3)))  , 
             x = xt, y = yt, pos=4  )
  panel.text(labels = bquote(italic(p) == .(format(p, digits = 3)))   ,  
             x = xt, y = yyt, pos=4  )
}, data = data, 
as.table = TRUE, 
xlab = "Diamter", 
ylab = "Cost", 
main = "Fig. 3 - Cost vs. Diameter for each Quantity Level")
