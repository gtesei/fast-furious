library(data.table)

test <- fread("./Inputs/test.csv")

scaleZeroOne <- function(x){
	(x - min(x)) / (max(x) - min(x))
}

var13 <- test[, var13]
var15 <- test[, var15]
var15[is.na(var15)] <- 0.0
target <- scaleZeroOne(-var13) + scaleZeroOne(var15)

sfile <- fread("./Inputs/sampleSubmission.csv")
submit <- gzfile("./Submits/submit.csv.gz", "wt")
write.table(data.frame(id=sfile$id, target=target), submit, sep=",", row.names=F, quote=F)
close(submit)

