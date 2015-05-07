transfrom4Skewness = function (traindata,
                               testdata,
                               verbose = T) {
  require(e1071)
  
  ## assembling train and test data 
  data = rbind(testdata,traindata)
  
  no = which(as.numeric(lapply(data,is.factor)) == 1)
  
  data.no = data 
  if (length(no) > 0) {
    data.no = data[,-no]
  }
  
  skewValuesBefore <- apply(data.no, 2, skewness)
  if (verbose) {
    cat("----> skewValues  before transformation  \n\n")
    print(skewValuesBefore)
  }
  
  idx = (1:(dim(data)[2]))
  if (length(no) > 0) {
    idx = (1:(dim(data)[2]))[-no]
  }
  
  for (i in idx) {
    varname = colnames(data)[i]
    tr = BoxCoxTrans(data[,i])
    if (verbose) { 
      cat("processing ",varname," ... \n")
      print(tr)
    }
    if (! is.na(tr$lambda) ) {
      if (verbose) cat("tranforming data ... \n")
      newVal = predict(tr,data[,i])
      data[,i] = newVal
      if (verbose) cat("skewness after transformation: " , skewness(data[,i]), "  \n")
    }
  } 
  
  if (verbose) { 
    cat("---->  skewValues  before transformation \n")
    print(skewValuesBefore)
    skewValues <- apply(data.no, 2, skewness)
    cat("\n---->  skewValues  after transformation:  \n")
    print(skewValues)
  }
  
  ## disassembling data 
  testdata = data[1:(dim(testdata)[1]),]
  traindata = data[((dim(testdata)[1])+1):(dim(data)[1]),]
  
  list(traindata,testdata)
}