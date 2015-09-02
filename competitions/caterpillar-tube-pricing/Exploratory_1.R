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

################ ANALYSIS 

####### train_set , test_set 
sum(is.na(train_set)) ## 0
sum(is.na(test_set))  ## 0 

## >> quindi nel trainset and testset non ci sono NAs

tube_assembly_id.quantity = ddply(train_set , .(tube_assembly_id , quantity) , function(x) c(num=length(x$tube_assembly_id)  ))
sum( tube_assembly_id.quantity$num > 1) ## 378

## >> che significa che (tube_assembly_id , quantity) e' una quasi-chiave primaria di train_set and test_set 

tube_assembly_id.quantity = ddply(train_set , .(tube_assembly_id , quantity,annual_usage) , function(x) c(num=length(x$tube_assembly_id)  ))
sum( tube_assembly_id.quantity$num > 1) ## 230

## >> perche' non esiste una chiave primaria del train set, 
#     e.g. due quotazioni fatte dalle stesso fornitore in date diverse (e costi !=)
train_set[train_set$tube_assembly_id=='TA-00178',]
# tube_assembly_id supplier quote_date annual_usage min_order_quantity bracket_pricing quantity     cost
# 425         TA-00178   S-0072 2014-07-15          300                  1             Yes        1 6.245723
# 426         TA-00178   S-0072 2014-07-12          300                  1             Yes        1 6.663031

## nel test set idem con patate ... 
tube_assembly_id.quantity = ddply(test_set , .(tube_assembly_id , quantity,annual_usage) , function(x) c(num=length(x$tube_assembly_id)  ))
sum( tube_assembly_id.quantity$num > 1) 
head(tube_assembly_id.quantity[tube_assembly_id.quantity$num>1 , ])
head(test_set[test_set$tube_assembly_id == 'TA-00340' , ])


length(unique(train_set[train_set$bracket_pricing == 'No' , ]$tube_assembly_id)) / length(train_set[train_set$bracket_pricing == 'No' , ]$tube_assembly_id) ##1

## >> che significa che per ogni tubo Non-Bracket esiste un e un solo caso da quotare 
#     nel qual caso nel campo <min_order_quantity> e' riportato la quantita' minima di pezzi da ordinare 
#     e nel campo <quantity> la quantia' minima per ordine 
# 
#     NOTA che <quantity> puo' essere diverso da <min_order_quantity> 
#         es. tube_assembly_id      = TA-00048 
#             min_order_quantity    = 20
#             quantity              = 1
# 
# >>>>>> ne deriva che anche per i tubi Non-Bracket il campo significativo e' <quantity> e non <min_order_quantity> 

####### tube 
sum(is.na(tube)) ##279
sum(is.na(tube$material_id)) ##279 

tube[is.na(tube$material_id) , ]

length(unique(intersect(tube[is.na(tube$material_id) , ]$tube_assembly_id , train_set$tube_assembly_id))) ## 101 NAs nel train_set
length(unique(intersect(tube[is.na(tube$material_id) , ]$tube_assembly_id , test_set$tube_assembly_id))) ## 97 NAs nel test_set

## >> in tube ci sono 279 NAs tutti relativi al campo material_id associati a 101 tube nel train set (su ~30.000) 
#     e 97 tubi nel test set (su ~30.000)
## >> cambiamo il valore di NA dei material_id NA da NA a UNKNOWN
tube[is.na(tube$material_id) , 'material_id'] = 'UNKNOWN'

##
tube.mat.tubid = ddply(tube , .(tube_assembly_id  ) ,  function(x) c(num=length(x$tube_assembly_id)))
sum(tube.mat.tubid$num>1) # 0 

## >> il che signiifica che (tube_assembly_id) e' una chiave primaria di tube <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

## >> a questo punto i prezzi associati allo stessa coppia (material_id, quantity) 
#    dovrebbero essere abbastanza uniformi (zeta >> 1)

tube_train = merge(x = train_set , y = tube , by = 'tube_assembly_id' , all = F)
tube_test = merge(x = test_set , y = tube , by = 'tube_assembly_id' , all = F)


cost.material_id = ddply(tube_train , .(material_id , quantity ) ,  
                         function(x) c(  cost.mean = mean(x$cost) 
                                       , cost.var = sd(x$cost)
                                       , num = length(x$cost)))
cost.material_id$zeta = cost.material_id$cost.mean / cost.material_id$cost.var
cost.material_id = cost.material_id[order(cost.material_id$zeta , decreasing = T) , ]

## tranne le dovute eccezzioni , es. material_id = SP-0037 
#  dove pero' si nota una differenze di specifiche tecniche, i.e. diameter wall length num_bends ...
tube_train[tube_train$material_id == 'SP-0037' & tube_train$bracket_pricing == 'Yes' , ] 

########>>>>>>>>>>>>>>> ci chiediamo a questo punto che performance puo' avere un semplice modello in 
#  cui il costo del tubo e' dato semplicemente dalla media del cluster di appartenenza in cost.material_id

cluter_test = merge(x = tube_test , y = cost.material_id , by = c('material_id','quantity') , all.x = T , all.y = F)
if (dim(cluter_test)[1] != dim(test_set)[1]) stop('something wrong')

# abbiamo 141 missing (cluter_test$cost.mean)  
# per questi 141 lavoreremo in seguito. Al momento li imputiamo con la media dei prezzi in prediction (~13.52663)
cluter_test[is.na(cluter_test$cost.mean) , ]$cost.mean = mean(cluter_test[! is.na(cluter_test$cost.mean) , ]$cost.mean)

# estraiamo la nostra prediction e la salviamo su disco ... 
sub = cluter_test[ , c('id' , 'cost.mean')]
sub = sub[order(sub$id,decreasing = F) , ]
colnames(sub) = colnames(sample_submission)
cat(">> writing prediction on disk ... \n")
write.csv(sub,quote=FALSE, 
          file=paste(getBasePath("submission"),'sub_base.csv',sep='') ,
          row.names=FALSE)

## >>>>>> otteniamo 0.625195 (769/869) dove il miglior modello performa come 0.210458 

id_qty = ddply(tube_train , .(tube_assembly_id,quantity,material_id)   , function(x)  c(num = length(unique(x$material_id))) )
sum(id_qty$num>1) # 0 

## >>  per ogni coppia (tube_assembly_id,quantity) e' associato al piu' un e un solo <material_id> 
#  tuttavia per ogni tripletta (tube_assembly_id,quantity,material_id) possono esistere diversi costi 
#  correlati essenzialmente a diversi <annual_usage> , <date_quotation>, etc.
id_qty.var = ddply(tube_train , .(tube_assembly_id,quantity,material_id)   , function(x)  c(num = length(x$material_id)) )
describe(id_qty.var$num)
# n         missing  unique    Info    Mean 
# 29815       0          5     0.04    1.013 
# 
#               1   2  3 4 6
# Frequency 29437 363 12 2 1
# %            99   1  0 0 0

####################################################################
## CONCLUSIONI 
####################################################################
#
# 1) <tube_assembly_id> e' chiave primaria di <tube>, per cui conviene creare un dataset di caratteristiche tecniche 
#    di ogni tubo <tube_tech>, la cui chiave primaria e' <tube_assembly_id>   
#    
# 2) <tube_assembly_id , quantity> e' una quasi-chiave primaria di train_set and test_set, i.e. a meno di circa 1% 
#    dei casi (sia nel train set che nel test set) si comporta come chiave primaria. Nell'1% in cui non lo e' succede
#    che ci possono essere diverse quotazioni a parita' di <tube_assembly_id , quantity> a seconda del <supplier>,
#    <quote_date>, <annual_usage>, etc. 
#    Per cui conviene creare un dataset finale <tube_info> che sia la join tra <tube_tech> e <train_set>/<test_set>
# 
# 3) Il campo <tube_assembly_id> e' sempre associato ad un e un solo <material_id> nel dataset <tube> tranne 1% dei casi. 
#    Tale campo sembra proprio essere il SAP material_id, per cui puo' essere usato per clusterizzare i dati, i.e. 
#    Se un tubo con un certo <tube_assembly_id> di cui occorre fare la quotazione nel <test_set> e' associato 
#    ad un certo <material_id> e lo zeta-score del <material_id> e' superiore ad una certa soglia, 
#    , allora il training del modello conviene farlo su tutti i tubi associati allo stesso 
#    <material_id> e la stessa <quantity> nel dataset <tube_info>, escludendo gli altri. 
#    Se invece, un tubo con un certo <tube_assembly_id> 
#    di cui occorre fare la quotazione nel <test_set> non e' associato ad un <material_id> (1% dei casi)
#    oppure lo zeta score e' inferiore ad una certa soglia, allora 
#    il training del modello conviene farlo su tutti i tubi nel dataset <tube_info>. 
#
#    In sostanza, si clusterizza il train set / test set per ogni coppia <material_id,quantity>, i.e. 
#    all'interno dello stesso cluster <material_id,quantity> sono univoci. 
#
# 4) il punto (3) ci porta anche ad un'architettura a 3 livelli in cui il primo livello e' costituito da un certo numero di modelli. 
#    Il secondo livello ha come feature le prediction del primo livello il <material_id> e la <quantity> e utilizza un altro set 
#    di modelli. L'eventuale terzo livello puo' essere costituito dalle medie pesate delle prediction del secondo livello. 




