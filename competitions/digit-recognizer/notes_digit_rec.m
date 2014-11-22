#! /opt/local/bin/octave -qf 

%%%% making enviroment 
menv;

printf("|--> loading data <<digit recognizer>> ...\n");

train = dlmread([curr_dir "/dataset/digit-recognizer/train.csv"]); 
test = dlmread([curr_dir "/dataset/digit-recognizer/test.csv"]); 
%%sampleSub = dlmread([curr_dir "/dataset/digit-recognizer/test.csv"]); 

## elimina le intestazioni del csv
train = train(2:end,:);
labels = train(:,1:1);  
train = train(:,2:end);
test = test(2:end,:);

############################################################## Normalizing naming
cross_size = 0.7;
printf("|--> performing feature scaling and normalization on train dataset and cross validation dataset ...\n");
printf("|--> splitting dataset into train set (%f) and cross validation set ...\n",cross_size);
rand_indices = randperm(size(train,1));
[Xtrain,ytrain,Xval,yval] = splitTrainValidation(train(rand_indices,:),labels(rand_indices),0.70);
#[Xtrain,mu,sigma] = treatContFeatures(Xtrain,1);
#[Xval,mu_val,sigma_val] = treatContFeatures(Xval,1,1,mu,sigma);

#Xtrain = Xtrain & Xtrain;
#Xval = Xval & Xval;

%%% Model training 
n = size(train,2);
num_label = length(unique(labels));
NNMeta = buildNNMeta([n n num_label]);disp(NNMeta);
lambda = 0;


[p_opt_RMSE,h_opt_RMSE,lambda_opt_RMSE,RMSE_opt,grid] = findOptPAndHAndLambda(Xtrain, ytrain, Xval, yval, ...
  featureScaled = 0 , scaleFeatures = 0 ,  ...
  p_vec = [] , ...
  h_vec = [1 2 3 4 5 6 7 8 9 10] , ...
  lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10] , ...
  verbose = 1, doPlot=1 , ...
  initGrid = [] , initStart = -1 ,  ...
  iter = 200 , ...
  regression = 0 , num_labels = num_label )

## fprintf("|--> Training Neural Network   (lambda=%f) ... \n",lambda);
## [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda , iter = 200, featureScaled = 0);
## pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 0);
## acc_train = mean(double(pred_train == ytrain)) * 100;
## fprintf("|-> Training Set Accuracy with feature normalization (lambda=%f): %f\n",lambda,acc_train);
## pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 0);
## acc_val = mean(double(pred_val == yval)) * 100;
## fprintf("Cross Validation Set Accuracy with feature normalization (lambda=%f): %f\n",lambda,acc_val);

