#! /opt/local/bin/octave -qf 

##setting enviroment 
clear ; close all;
global curr_dir = pwd;

addpath(curr_dir);
addpath([curr_dir "/util"]);
addpath([curr_dir "/linear_reg"]);
addpath([curr_dir "/logistic_reg"]);
addpath([curr_dir "/neural"]);

addpath([curr_dir "/dataset/poly/"]);


##Test Cases
%source "./linear_reg/README.m";
%go();

source "./logistic_reg/README.m";
go();



