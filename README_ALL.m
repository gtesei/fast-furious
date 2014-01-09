#! /opt/local/bin/octave -qf 

##setting enviroment 
clear ; close all;
menv;

##Test Cases
source linear_reg/README_LIN_REG.m;
var1_doBasicUseCase();
var1_doFindOptPAndLambdaUseCase();


