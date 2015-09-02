################################
# The main difference of this v2 version respect baseline 
# is that here we extract also weigth and orientation from bom for each 
# tube component 
################################

library(binhf)
library(fBasics)
library(lattice)
require(xgboost)
require(methods)
library(data.table)
library(plyr)

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

################# FAST-FURIOUS SOURCES
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))

################# DATA IN 
sample_submission = as.data.frame( fread(paste(getBasePath("data") , 
                                               "sample_submission.csv" , sep=''))) 

train_set = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train_set.csv" , sep=''))) 

test_set = as.data.frame( fread(paste(getBasePath("data") , 
                                      "test_set.csv" , sep=''))) 

tube = as.data.frame( fread(paste(getBasePath("data") , 
                                  "tube.csv" , sep='')))

# bill_of_materials
bill_of_materials = as.data.frame( fread(paste(getBasePath("data") , 
                                               "bill_of_materials.csv" , sep='')))

components = as.data.frame( fread(paste(getBasePath("data") , 
                                        "components.csv" , sep='')))
# specs 
specs = as.data.frame( fread(paste(getBasePath("data") , 
                                               "specs.csv" , sep='')))
## compents details 
comp_adaptor = as.data.frame( fread(paste(getBasePath("data") , 
                                        "comp_adaptor.csv" , sep='')))
comp_boss = as.data.frame( fread(paste(getBasePath("data") , 
                                        "comp_boss.csv" , sep='')))
comp_elbow = as.data.frame( fread(paste(getBasePath("data") , 
                                        "comp_elbow.csv" , sep='')))
comp_float = as.data.frame( fread(paste(getBasePath("data") , 
                                        "comp_float.csv" , sep='')))
comp_hfl = as.data.frame( fread(paste(getBasePath("data") , 
                                        "comp_hfl.csv" , sep='')))
comp_nut = as.data.frame( fread(paste(getBasePath("data") , 
                                        "comp_nut.csv" , sep='')))
comp_other = as.data.frame( fread(paste(getBasePath("data") , 
                                      "comp_other.csv" , sep='')))
comp_sleeve = as.data.frame( fread(paste(getBasePath("data") , 
                                      "comp_sleeve.csv" , sep='')))
comp_straight = as.data.frame( fread(paste(getBasePath("data") , 
                                         "comp_straight.csv" , sep='')))
comp_tee = as.data.frame( fread(paste(getBasePath("data") , 
                                           "comp_tee.csv" , sep='')))
comp_threaded = as.data.frame( fread(paste(getBasePath("data") , 
                                      "comp_threaded.csv" , sep='')))

################# DATA OUT 
tube_base = NULL 
bom_base = NULL
spec_enc = NULL 

################# PROCESSING 
# fix NAs material_id 
tube[is.na(tube$material_id) , 'material_id'] = 'UNKNOWN' 

# map tube2material for having tube_assembly_id --> material_id
tube2material = ddply(tube , .(tube_assembly_id,material_id) , function(x) c(num=length(x$tube_assembly_id)) )
if (sum(tube2material$num>1)) stop('something wrong')
tube2material$num = NULL

# e.g. the material_id of tube_assembly_id TA-00001 is SP-0035 
tube2material[tube2material$tube_assembly_id=='TA-00001','material_id']
#[1] "SP-0035"

# categorize end_a_1x end_a_2x end_x_1x end_x_2x 
tube$end_a_1x = ifelse(tube$end_a_1x == 'N' , 0 , 1)
tube$end_a_2x = ifelse(tube$end_a_2x == 'N' , 0 , 1)
tube$end_x_1x = ifelse(tube$end_x_1x == 'N' , 0 , 1)
tube$end_x_2x = ifelse(tube$end_x_2x == 'N' , 0 , 1)

# categorize end_a end_x 
l = encodeCategoricalFeature (tube$end_a , NULL , colname.prefix = "end_a" , asNumeric=F)
tube = cbind(tube,l$traindata)
tube[,'end_a'] = NULL  

l = encodeCategoricalFeature (tube$end_x , NULL , colname.prefix = "end_x" , asNumeric=F)
tube = cbind(tube,l$traindata)
tube[,'end_x'] = NULL  

## valorize tube_base  
tube_base = tube 
if (sum(is.na(tube_base)) > 0)  stop('NAs in tube_base')

##### bill_of_materials
# we build a first model where we extract from the bill of materials only the 
#  - type of component for each component 
#  - and the number of components for each component  

## components is a map for component_id --> component_type_id
components[components$component_id == 'C-0002' , 'component_type_id']
#[1] "CP-024"

## check 
comp_detail = comp_adaptor[,c('component_id','component_type_id','weight','orientation')]
comp_detail = rbind(comp_detail , comp_boss[,c('component_id','component_type_id','weight','orientation')])
comp_detail = rbind(comp_detail , comp_elbow[,c('component_id','component_type_id','weight','orientation')])
comp_detail = rbind(comp_detail , comp_float[,c('component_id','component_type_id','weight','orientation')])
comp_detail = rbind(comp_detail , comp_hfl[,c('component_id','component_type_id','weight','orientation')])
comp_detail = rbind(comp_detail , comp_nut[,c('component_id','component_type_id','weight','orientation')])
comp_other$orientation = 'No'
comp_other$component_type_id = 'OTHER'
comp_detail = rbind(comp_detail , comp_other[,c('component_id','component_type_id','weight','orientation')])
comp_detail = rbind(comp_detail , comp_sleeve[,c('component_id','component_type_id','weight','orientation')])
comp_detail = rbind(comp_detail , comp_straight[,c('component_id','component_type_id','weight','orientation')])
comp_detail = rbind(comp_detail , comp_tee[,c('component_id','component_type_id','weight','orientation')])
comp_detail = rbind(comp_detail , comp_threaded[,c('component_id','component_type_id','weight','orientation')])

length(intersect(comp_detail$component_id , components$component_id))
length(components$component_id)

## create matrix 
comp_lev = sort(unique(components$component_type_id))
mm = matrix(rep(0,nrow(tube)*3*length(comp_lev)),nrow=nrow(tube),ncol=3*length(comp_lev))
colns = gsub("-", '_' , comp_lev )
colns = as.vector(unlist(outer(colns , c('_num','_orientation','_weight') , paste0)))
colnames(mm) = colns

## populate matrix 
for ( j in 1:nrow(tube) ) {
  if (j %% 1000 == 0) cat(">> [bill of materials] processing tube ",j,"/",nrow(tube)," ... \n")
  comps = as.vector(na.omit(unlist((bill_of_materials[ bill_of_materials$tube_assembly_id == tube[j,'tube_assembly_id'] ,
                                                       c(2,4,6,8,10,12,14,16) ]))))
  comps = comps[comps != '']
  
  nums = as.vector(na.omit(unlist((bill_of_materials[ bill_of_materials$tube_assembly_id == tube[j,'tube_assembly_id'] ,
                                                      c(3,5,7,9,11,13,15,17) ]))))
  
  # check 
  if ( length(comps) > length(nums) ) stop('number of components length bigger than components length')
  # notice that somethime length(comps) < length(nums) 
  
  for (k in 1:length(comps)) {
    comp_type = components[components$component_id == comps[k] , 'component_type_id']
    
    orientation = comp_detail[comp_detail$component_id == comps[k] , 'orientation']
    orientation = ifelse(orientation == 'Yes' , 1 , 0)
    if (length(orientation) == 0) {
      cat(">> [bill of materials]---- tube ",j,"/",nrow(tube)," setting orientation to 0 [from undefined] ... \n") 
      orientation = 0
    }
    if (is.na(orientation)) {
      cat(">> [bill of materials]---- tube ",j,"/",nrow(tube)," setting orientation to 0 [from NA] ... \n") 
      orientation = 0
    }
    
    weight = comp_detail[comp_detail$component_id == comps[k] , 'weight']
    if (length(weight) == 0) {
      cat(">> [bill of materials]---- tube ",j,"/",nrow(tube)," setting weight to 0 [from undefined] ... \n") 
      weight = 0
    }
    if (is.na(weight)) {
      cat(">> [bill of materials]---- tube ",j,"/",nrow(tube)," setting weight to 0 [from NA] ... \n") 
      weight = 0
    }
    
    #mm[j , which(comp_lev == comp_type)] = nums[k]
    mm[j , 3*(which(comp_lev == comp_type)-1)+1] = nums[k]
    mm[j , 3*(which(comp_lev == comp_type)-1)+2] = orientation 
    mm[j , 3*(which(comp_lev == comp_type)-1)+3] = weight 
  }
}

## valorize bom_base 
bom_base = as.data.frame(mm)
if (sum(is.na(bom_base)) > 0)  stop('NAs in bom_base')
  
## how to bind matrix to tube 
tube = cbind(tube,bom_base)

##### specs 
specs_uniq = as.vector(na.omit(unique(c(specs$spec1,specs$spec2,specs$spec3,specs$spec4,specs$spec5,
         specs$spec6,specs$spec7,specs$spec8,specs$spec9,specs$spec10))))
specs_uniq = specs_uniq[specs_uniq != '']
specs_uniq = sort(specs_uniq)

length(specs_uniq) # 85
# we have 85 different specs 

## create matrix 
comp_lev = specs_uniq
mm = matrix(rep(0,nrow(tube)*length(comp_lev)),nrow=nrow(tube),ncol=length(comp_lev))
colns = gsub("-", '_' , comp_lev )
colnames(mm) = colns

## populate matrix 
for ( j in 1:nrow(tube) ) {
  if (j %% 1000 == 0) cat(">> [specs] processing tube ",j,"/",nrow(tube)," ... \n")
  sp = as.vector(na.omit(unlist(as.vector(specs[ specs$tube_assembly_id == tube[j,'tube_assembly_id'] , -1 ]))))
  sp = sp[sp != '']
  
  for (k in 1:length(sp)) {
    mm[j , which(comp_lev == sp[k])] = 1
  }
}

## valorize bom_base 
spec_enc = as.data.frame(mm)
if (sum(is.na(spec_enc)) > 0)  stop('NAs in spec_enc')

## how to bind matrix to tube 
tube = cbind(tube,spec_enc)

## so we have a base feature set of 180 (encoded) technical features 
dim(tube)
#[1] 21198   180

################# SAVE DATA OUT on disk 
cat(">> saving on disk ...\n")

# tube_base
write.csv(tube_base,
          quote=FALSE, 
          file=paste(getBasePath("elab"),'tube_base.csv',sep='') ,
          row.names=FALSE)

#bom_base 
write.csv(bom_base,
          quote=FALSE, 
          file=paste(getBasePath("elab"),'bom_base.csv',sep='') ,
          row.names=FALSE)

#spec_enc  
write.csv(spec_enc,
          quote=FALSE, 
          file=paste(getBasePath("elab"),'spec_enc.csv',sep='') ,
          row.names=FALSE)
