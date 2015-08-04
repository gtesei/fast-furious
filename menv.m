
%% setting enviroment 
clear ; close all;
global curr_dir 
curr_dir= pwd;

addpath(curr_dir);
addpath([curr_dir '/util']);
addpath([curr_dir '/linear_reg']);
addpath([curr_dir '/logistic_reg']);
addpath([curr_dir '/neural']);
addpath([curr_dir '/SVM']);

addpath([curr_dir '/competitions/liberty-mutual-fire-peril']);
addpath([curr_dir '/competitions/seizure-prediction']);
addpath([curr_dir '/competitions/digit-recognizer']);
addpath([curr_dir '/competitions/restaurant-revenue-prediction']);
addpath([curr_dir '/competitions/otto-group-product-classification-challenge']);

addpath([curr_dir '/dataset/poly/']);

%% addpath('/Users/gino/kaggle/libsvm/matlab');

if exist('OCTAVE_VERSION', 'builtin') > 0 
  setenv('GNUTERM','qt')
end 
