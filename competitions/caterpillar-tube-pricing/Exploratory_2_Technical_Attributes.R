library(caret)
library(Hmisc)
library(data.table)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)
library(fBasics)
library(lattice)
require(xgboost)
require(methods)

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

################# DATA 
sample_submission = as.data.frame( fread(paste(getBasePath("data") , 
                                               "sample_submission.csv" , sep=''))) 

train_set = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train_set.csv" , sep=''))) 

test_set = as.data.frame( fread(paste(getBasePath("data") , 
                                      "test_set.csv" , sep=''))) 

tube = as.data.frame( fread(paste(getBasePath("data") , 
                                  "tube.csv" , sep='')))

bill_of_materials = as.data.frame( fread(paste(getBasePath("data") , 
                                               "bill_of_materials.csv" , sep='')))

## specific component attributes 
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

comp_sleeve = as.data.frame( fread(paste(getBasePath("data") , 
                                      "comp_sleeve.csv" , sep='')))

comp_straight = as.data.frame( fread(paste(getBasePath("data") , 
                                         "comp_straight.csv" , sep='')))

comp_tee = as.data.frame( fread(paste(getBasePath("data") , 
                                           "comp_tee.csv" , sep='')))

comp_threaded = as.data.frame( fread(paste(getBasePath("data") , 
                                      "comp_threaded.csv" , sep='')))

## list of components and component type 
components = as.data.frame( fread(paste(getBasePath("data") , 
                                           "components.csv" , sep='')))

type_component = as.data.frame( fread(paste(getBasePath("data") , 
                                        "type_component.csv" , sep='')))

################# PROCESSING 

## find similar attributes 
common_attr = intersect( colnames(comp_adaptor) , colnames(comp_boss) )
common_attr = intersect( common_attr , colnames(comp_elbow) )
common_attr = intersect( common_attr , colnames(comp_float) )
common_attr = intersect( common_attr , colnames(comp_hfl) )
common_attr = intersect( common_attr , colnames(comp_sleeve) )
common_attr = intersect( common_attr , colnames(comp_straight) )
common_attr = intersect( common_attr , colnames(comp_tee) )
common_attr = intersect( common_attr , colnames(comp_threaded) )
common_attr
# [1] "component_id"      "component_type_id" "orientation"       "weight"      

## check that each component occour in components 
comps = unique(union(comp_adaptor$component_id,comp_boss$component_id))
comps = unique(union(comps,comp_elbow$component_id))
comps = unique(union(comps,comp_float$component_id))
comps = unique(union(comps,comp_hfl$component_id))
comps = unique(union(comps,comp_sleeve$component_id))
comps = unique(union(comps,comp_straight$component_id))
comps = unique(union(comps,comp_tee$component_id))
comps = unique(union(comps,comp_threaded$component_id))

if (length(unique(union(components$component_id , comps))) != length(components$component_id)) ## 2048 components 
  stop('something wrong')

## components is a map for component_id --> component_type_id
components[components$component_id == 'C-0002' , 'component_type_id']
#[1] "CP-024"

if ( length(unique(components$component_type_id)) != length(unique(type_component$component_type_id)) ) ## 29 component_type_id
  stop('something wrong')

















