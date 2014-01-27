#!/bin/bash

let COUNT=0
for f in $(ls train-dogs-vs-cats/ ) 
do  
if [ $COUNT -gt 3000 ]
then
 break 
fi
 echo $COUNT
if [[ $f == *cat* ]]; 
then 
 echo $f
 cp "train-dogs-vs-cats/$f" small_train-dogs-cats
 let COUNT=COUNT+1
fi 
done


let COUNT=0
for f in $(ls train-dogs-vs-cats/ ) 
do  
if [ $COUNT -gt 3000 ]
then
 break 
fi
 echo $COUNT
if [[ $f == *dog* ]]; 
then 
 echo $f
 cp "train-dogs-vs-cats/$f" small_train-dogs-cats
 let COUNT=COUNT+1
fi 
done


