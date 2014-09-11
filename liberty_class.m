#! /opt/local/bin/octave -qf 

%%%% setting enviroment 
menv;

#trainFile = "train_class.csv"
#testFile = "test_class.csv"
trainFile = "train_class_pca.csv"
testFile = "test_class_pca.csv"

printf("|--> loading Xtrain, ytrain files ...\n");
train = dlmread([curr_dir "/dataset/liberty-mutual-fire-peril/" trainFile]); 
X_test = dlmread([curr_dir "/dataset/liberty-mutual-fire-peril/" testFile]); 

train = train(2:end,:); ## elimina le intestazioni del csv 
X_test = X_test(2:end,:); ## elimina le intestazioni del csv

#y = train(:,1); ## la prima colonna e' target mentra l'ultima e' tragetPos che va scartata ...
y = train(:,end); ##for classification problem
X = train(:,2:(end-1));

idTest = X_test(:,end); ## l'ultima colonna e' id
X_test = X_test(:,1:(end-1));

clear train;


### cv ...
perc_train = 0.8;
[m,n] = size(X);
rand_indices = randperm(m);
[_Xtrain,ytrain,_Xval,yval] = splitTrainValidation(X(rand_indices,:),y(rand_indices,:),perc_train);

### feature scaling and cenering ...
[Xtrain,mu,sigma] = treatContFeatures(_Xtrain,1);
[Xval,mu,sigma] = treatContFeatures(_Xval,1,1,mu,sigma);

### find C,gamma SVM 
[C_opt_recall,g_opt_recall,C_opt_accuracy,g_opt_accuracy,C_opt_precision,g_opt_precision,C_opt_F1,g_opt_F1,grid] = ...
  findOptCAndGammaSVM(Xtrain, ytrain, Xval, yval, featureScaled = 0 , 
  C_vec = [2^-5 2^-3 2^-1 2^1 2^3 2^5 2^7 2^11 2^15]' , 
  g_vec = [2^-15 2^-11 2^-7 2^-3 2^-1 2^1 2^2 2^3 2^5 2^7]' ,
  verbose = 1);







