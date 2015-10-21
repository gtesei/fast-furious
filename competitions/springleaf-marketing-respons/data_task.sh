#!/bin/bash 

#echo ">> running data_proc_2@cor  ... "
#Rscript -e 'TASK = "cor";source("data_proc_2.R")'

#echo ">> running data_proc_2@polycut  ... "
#Rscript -e 'TASK = "polycut";source("data_proc_2.R")'

#echo ">> running data_proc_2@pca  ... "
#Rscript -e 'TASK = "pca";source("data_proc_2.R")'

echo ">> running ffOctNNet.m  ... "
cd /Users/gino/kaggle/fast-furious/gitHub/fast-furious
/usr/local/octave/3.8.0/bin/octave-3.8.0 "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/springleaf-marketing-respons/ffOctNNet.m"

#echo ">> running data_proc_2@K-MEANS  ... " 
#Rscript -e 'TASK = "K-MEANS";source("data_proc_2.R")' 