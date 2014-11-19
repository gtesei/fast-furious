#! /opt/local/bin/octave -qf 

%%%% making enviroment 
menv;

printf("|--> loading data <<digit recognizer>> ...\n");

train = dlmread([curr_dir "/dataset/digit-recognizer/train_sample.csv"]); 
test = dlmread([curr_dir "/dataset/digit-recognizer/train_sample.csv"]); 
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
rand_indices = randperm(m);
[Xtrain,ytrain,Xval,yval] = splitTrainValidation(train(rand_indices,:),labels(rand_indices),0.70);
[Xtrain,mu,sigma] = treatContFeatures(Xtrain,1);
[Xval,mu_val,sigma_val] = treatContFeatures(Xval,1,1,mu,sigma);

fprintf("|--> Neural Network Training  (lambda=%f) ... \n",lambda);
[Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda , iter = 200, featureScaled = 1);
pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
acc_train = mean(double(pred_train == ytrain)) * 100;
fprintf("|-> Training Set Accuracy with feature normalization (lambda=%f): %f\n",lambda,acc_train);
pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
acc_val = mean(double(pred_val == yval)) * 100;
fprintf("Cross Validation Set Accuracy with feature normalization (lambda=%f): %f\n",lambda,acc_val);