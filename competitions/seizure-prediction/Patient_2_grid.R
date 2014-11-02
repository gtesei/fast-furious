

build.label = function (grid.item) {
  
  label = ""
  
  ## attributes 
  model.id = as.numeric(grid.item[1])
  data.source.gen = as.character(grid.item[2])
  pca.feature = as.logical(grid.item[3])
  superFeature = as.logical(grid.item[4])
  
  ##building 
  if (model.id >= 1 && model.id <= 6) { ## logistic reg 
    label = paste0(label,"LOG_")
  } else if (model.id >= 7 && model.id <= 12) { ## lda 
    label = paste0(label,"LDA_")
  } else if (model.id >= 13 && model.id <= 18) { ## plsda 
    label = paste0(label,"PLSDA_")
  } else if (model.id >= 19 && model.id <= 24) { ## pm 
    label = paste0(label,"PM_")
  } else if (model.id >= 25 && model.id <= 30) { ## nsc 
    label = paste0(label,"NSC_")
  } else if (model.id >= 31 && model.id <= 36) { # neural networks 
    label = paste0(label,"NN_")
  } else if (model.id >= 37 && model.id <= 42) { ## svm 
    label = paste0(label,"SVM_")
  } else if (model.id >= 43 && model.id <= 49) { ## knn 
    label = paste0(label,"KNN_")
  } else if (model.id >= 49 && model.id <= 54) { ## class trees 
    label = paste0(label,"CT_")
  } else if (model.id >= 55 && model.id <= 60) { ## boosted trees 
    label = paste0(label,"BOT_")
  } else if (model.id >= 61 && model.id <= 66) { ## bagging trees 
    label = paste0(label,"BAT_")
  } else if (model.id == 0) { ## null prediction, i.e. pred = (0,0,0,...,0)
  } else {
    stop("ma che modello e' (modello) !! ")
  }
  
  if (model.id %% 6 == 0) { ##QUANTILES_REDUCED
    label = paste0(label,"QR_")
  } else if (model.id %% 6 == 1) { ## MEAN_SD
    label = paste0(label,"MS_")
  } else if (model.id %% 6 == 2) { ## QUANTILES
    label = paste0(label,"Q_")
  } else if (model.id %% 6 == 3) { ## MEAN_SD_SCALED
    label = paste0(label,"MSS_")
  } else if (model.id %% 6 == 4) { ## QUANTILES_SCALED
    label = paste0(label,"QS_")
  } else if (model.id %% 6 == 5) { ## MEAN_SD_REDUCED
    label = paste0(label,"MSR_")
  } else {
    stop("ma che modello e' (data set) !! ")
  }
  
  label = paste(label,"ds-",data.source.gen,"_",sep="")
  label = paste(label,"pca-", ifelse(pca.feature,"T","F") ,"_",sep="")
  label = paste(label,"supr-", ifelse(superFeature,"T","F") ,sep="")

  return(label)
}


build.grid = function (model.vect , ds=c(7,5,4) , pca=c(T,F) , super=c(T,F) ) {
  
  grid.t = expand.grid( model.id=model.vect,
                        data.source.gen=ds,
                        pca.feature=pca, 
                        superFeature=super ) 
  
  grid.t$data.source.gen = ifelse(grid.t$data.source.gen == 7 , "7gen" , 
                                  ifelse(grid.t$data.source.gen == 5 , "5gen" , "4gen")
  ) 
  grid.t$model.label = "TODO"
  for (i in 1:nrow(grid.t) )  
    grid.t[i,]$model.label = build.label(grid.t[i,])
  
  grid.t = grid.t[,c(5,1,2,3,4)]
  
  return(grid.t)
}


######################################################## CONSTANTS 
NULL_MODEL = 0

LOGISTIC_REG_MEAN_SD = 1 
LOGISTIC_REG_QUANTILES = 2
LOGISTIC_REG_MEAN_SD_SCALED = 3
LOGISTIC_REG_QUANTILES_SCALED = 4
LOGISTIC_REG_MEAN_SD_REDUCED = 5
LOGISTIC_REG_QUANTILES_REDUCED = 6

LDA_MEAN_SD = 7
LDA_QUANTILES = 8
LDA_MEAN_SD_SCALED = 9 
LDA_QUANTILES_SCALED= 10 
LDA_MEAN_SD_REDUCED = 11
LDA_REG_QUANTILES_REDUCED = 12 

PLSDA_MEAN_SD = 13
PLSDA_QUANTILES = 14
PLSDA_MEAN_SD_SCALED = 15 
PLSDA_QUANTILES_SCALED= 16 
PLSDA_MEAN_SD_REDUCED = 17
PLSDA_REG_QUANTILES_REDUCED = 18 

PM_MEAN_SD = 19
PM_QUANTILES = 20
PM_MEAN_SD_SCALED = 21 
PM_QUANTILES_SCALED= 22 
PM_MEAN_SD_REDUCED = 23
PM_REG_QUANTILES_REDUCED = 24

NSC_MEAN_SD = 25
NSC_QUANTILES = 26
NSC_MEAN_SD_SCALED = 27 
NSC_QUANTILES_SCALED= 28 
NSC_MEAN_SD_REDUCED = 29
NSC_REG_QUANTILES_REDUCED = 30

NN_MEAN_SD = 31
NN_QUANTILES = 32
NN_MEAN_SD_SCALED = 33 
NN_QUANTILES_SCALED= 34 
NN_MEAN_SD_REDUCED = 35
NN_QUANTILES_REDUCED = 36

SVM_MEAN_SD = 37
SVM_QUANTILES = 38
SVM_MEAN_SD_SCALED = 39 
SVM_QUANTILES_SCALED= 40 
SVM_MEAN_SD_REDUCED = 41
SVM_QUANTILES_REDUCED = 42

KNN_MEAN_SD = 43
KNN_QUANTILES = 44
KNN_MEAN_SD_SCALED = 45 
KNN_QUANTILES_SCALED= 46 
KNN_MEAN_SD_REDUCED = 47
KNN_QUANTILES_REDUCED = 48

CLASS_TREE_MEAN_SD = 49
CLASS_TREE_QUANTILES = 50
CLASS_TREE_MEAN_SD_SCALED = 51 
CLASS_TREE_QUANTILES_SCALED= 52 
CLASS_TREE_MEAN_SD_REDUCED = 53
CLASS_TREE_QUANTILES_REDUCED = 54

BOOSTED_TREE_MEAN_SD = 55
BOOSTED_TREE_QUANTILES = 56
BOOSTED_TREE_MEAN_SD_SCALED = 57 
BOOSTED_TREE_QUANTILES_SCALED= 58 
BOOSTED_TREE_MEAN_SD_REDUCED = 59
BOOSTED_TREE_QUANTILES_REDUCED = 60

BAGGING_TREE_MEAN_SD = 61
BAGGING_TREE_QUANTILES = 62
BAGGING_TREE_MEAN_SD_SCALED = 63 
BAGGING_TREE_QUANTILES_SCALED= 64 
BAGGING_TREE_MEAN_SD_REDUCED = 65
BAGGING_TREE_QUANTILES_REDUCED = 66

######################################################## 

model.grid.pat2 = NULL

## ## ## ## ## SVMs 
model.grid.pat2 = build.grid(c(SVM_MEAN_SD_SCALED, SVM_QUANTILES_SCALED, SVM_MEAN_SD_REDUCED, SVM_QUANTILES_REDUCED))  

## ## ## ## ## LOG
model.grid.pat2 = rbind ( model.grid.pat2 , build.grid(c(LOGISTIC_REG_QUANTILES, LOGISTIC_REG_MEAN_SD_REDUCED)) )

## ## ## ## ## LDA
model.grid.pat2 = rbind ( model.grid.pat2 , build.grid(c(LDA_MEAN_SD_SCALED, LDA_MEAN_SD_REDUCED)) )

## ## ## ## ## PLSDA
model.grid.pat2 = rbind ( model.grid.pat2 , build.grid(c(PLSDA_MEAN_SD_SCALED)) )

## ## ## ## ## NSC
model.grid.pat2 = rbind ( model.grid.pat2 , build.grid(c(NSC_MEAN_SD_SCALED)) )

## ## ## ## ## NSC
model.grid.pat2 = rbind ( model.grid.pat2 , build.grid(c(NSC_MEAN_SD_SCALED) , ds=c(7,4)) )

## ## ## ## ## NN
model.grid.pat2 = rbind ( model.grid.pat2 , build.grid(c(NN_MEAN_SD_REDUCED) , ds=c(5)) )

## ## ## ## ## KNN
model.grid.pat2 = rbind ( model.grid.pat2 , build.grid(c(KNN_MEAN_SD_REDUCED) , ds=c(5)) )

## ## ## ## ## CT
model.grid.pat2 = rbind ( model.grid.pat2 , build.grid(c(CLASS_TREE_MEAN_SD) , ds=c(5)) )

## ## ## ## ## BOOST
model.grid.pat2 = rbind ( model.grid.pat2 , build.grid(c(BOOSTED_TREE_QUANTILES,BOOSTED_TREE_MEAN_SD_SCALED) , ds=c(5)) )

## ## ## ## ## BAGGING
model.grid.pat2 = rbind ( model.grid.pat2 , build.grid(c(BAGGING_TREE_MEAN_SD) , ds=c(4)) )

## ## ## ## ## PM
model.grid.pat2 = rbind ( model.grid.pat2 , build.grid(c(PM_MEAN_SD_SCALED) , ds=c(4)) )

cat("*************************************** MODEL GRID *************************************** \n") 
print(model.grid.pat2)
cat("****************************************************************************** \n")

