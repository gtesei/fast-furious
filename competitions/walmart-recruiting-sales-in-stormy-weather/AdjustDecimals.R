library(Hmisc)

sub = as.data.frame( 
  fread("/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/sub_fix_37_5 2.csv")
  #fread("/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/sub_adjust_TSS.csv")
  )

th = 0.05

cat(">>> decimal01:",sum(sub$units>0 & sub$units<1),"\n")
cat(">>> decimal:",sum(sub$units %% 1 > 0),"\n")
cat(">>> decimal01:",sum(sub$units %% 1 > 0 & sub$units %% 1 < th & sub$units>0 & sub$units<1),"\n")

ii = sort(which(sub$units %% 1 > 0 & sub$units %% 1 < th & sub$units>0 & sub$units<1 ) , decreasing = T)

describe(sub[ii,]$units)
sub[ii,]$units = 0 
describe(sub[ii,]$units)

for(i in ii){
  before = sub[i,]$units
  
  fl = floor(sub[i,]$units)
  
  dec = sub[i,]$units %% 1 
  
  after = ifelse(dec < 0.5 & fl<1 , fl , fl+1)
  
  sub[i,]$units = after
  
  cat("before:",before," - after:",after,"\n")
}

cat(">>> decimal01:",sum(sub$units>0 & sub$units<1),"\n")
cat(">>> decimal:",sum(sub$units %% 1 > 0),"\n")


write.csv(sub,quote=FALSE, 
          file=paste("/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/walmart-recruiting-sales-in-stormy-weather/",
                     "sub_adjust_decimals_th_",th,".csv",sep='') ,
          row.names=FALSE)
