#! /opt/local/bin/octave -qf 

##setting enviroment 
menv;

%%% Loading Features & scaling 
printf("|--> Loading trainset features ...\n");

Xtrain1 = dlmread([curr_dir "/dataset/images2/featuresDogsCatsE.zat"]); %13 features 
Xtrain2 = dlmread([curr_dir "/dataset/images2/featuresDogsCats_sobelsE.zat"]); %14 features
Xtrain3 = dlmread([curr_dir "/dataset/images2/featuresDogsCatsSURF.zat"]); %14 features

Xtrain = [Xtrain1 , Xtrain2 , Xtrain3];
ytrain = dlmread([curr_dir "/dataset/images2/labelsDogsCats.zat"]);
m = size(Xtrain, 1);
n = size(Xtrain,2);
ytrain = ones(m,1) + ytrain; 

printf("|-> performing feature scaling ...\n");
[Xtrain,mu,sigma] = treatContFeatures(Xtrain,1);


%%% Model training 
NNMeta = buildNNMeta([n n 2]);disp(NNMeta);
lambda = 10;
fprintf("|--> Neural Network Training  (lambda=%f) ... \n",lambda);
[Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda , iter = 10, featureScaled = 1);
pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
acc_train = mean(double(pred_train == ytrain)) * 100;
fprintf("|-> Training Set Accuracy with feature normalization (lambda=%f): %f\n",lambda,acc_train);
[_dir] = serializeNNTheta(Theta);
fprintf("|-> Serialized Thetas into %s directory.\n",_dir); 

%%% Predicting on Testset
Xtest1 = dlmread([curr_dir "/dataset/images2/??????"]); 
Xtest2 = dlmread([curr_dir "/dataset/images2/?????"]); 
Xtest3 = dlmread([curr_dir "/dataset/images2/?????"]); 

Xtest = [Xtest1 , Xtest2 , Xtest3];
[Xest,mu_test,sigma_test] = treatContFeatures(Xest,1,1,mu,sigma);
pred_test = NNPredictMulticlass(NNMeta, Theta , Xtest , featureScaled = 1);
pred = [(1:length(pred_test))' pred_test];
fn = "predictions.zat";
dlmwrite(fn,pred);
printf("|-> Predictions serialized into %s \n",fn);


function findOptParams()

 global curr_dir;

 _X = dlmread([curr_dir "/dataset/images2/featuresDogsCatsE.zat"]); %13 features 
 X_sobels = dlmread([curr_dir "/dataset/images2/featuresDogsCats_sobelsE.zat"]); %14 features
 X_surf = dlmread([curr_dir "/dataset/images2/featuresDogsCatsSURF.zat"]); %14 features
 m = size(X_sobels, 1);
 _y = dlmread([curr_dir "/dataset/images2/labelsDogsCats.zat"]);
 y = ones(m,1) + _y; 

 X = [_X , X_sobels , X_surf];

 n = size(X,2);

 printf("|--> performing feature scaling and normalization on train dataset and cross validation dataset ...\n");
 printf("\n|--> splitting dataset into train set and cross validation set ...\n");
 rand_indices = randperm(m);
 [Xtrain,ytrain,Xval,yval] = splitTrainValidation(X(rand_indices,:),y(rand_indices),0.70); 
 [Xtrain,mu,sigma] = treatContFeatures(Xtrain,1);
 [Xval,mu_val,sigma_val] = treatContFeatures(Xval,1,1,mu,sigma);

 NNMeta = buildNNMeta([n n 2]);disp(NNMeta);
 lambda = 0.5;


 %% finding optimal number of hidden layers
 lambda = 0;
 printf("|--> finding optimal number of neurons per layer ... \n");
 %tic(); [h_opt,J_opt] = findOptHiddenLayers(Xtrain, ytrain, Xval, yval,lambda); toc();
 %pause;

%% finding optimal number of neurons per layer 
printf("|--> finding optimal number of neurons per layer ... \n");
%tic(); [s_opt,J_opt] = findOptNeuronsPerLayer(Xtrain, ytrain, Xval, yval,lambda,start_neurons=10,end_neurons=100,step_fw=10); toc();
%pause;

%% finding optimal lambda 
printf("|--> finding optimal number of neurons per layer ... \n");
tic(); [l_opt,J_opt] = findOptLambda(NNMeta, Xtrain, ytrain, Xval, yval); toc();
%pause;

 
 [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda , iter = 400, featureScaled = 1);
 pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
 pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
 acc_train = mean(double(pred_train == ytrain)) * 100;
 acc_val = mean(double(pred_val == yval)) * 100;
 fprintf("Training Set Accuracy with feature normalization (lambda=%f): %f\n",lambda,acc_train);
 fprintf("Cross Validation Set Accuracy with feature normalization (lambda=%f): %f\n",lambda,acc_val);


  
endfunction 





############################## MAIN ###############
#findOptParams();


