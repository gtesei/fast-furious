#! /opt/local/bin/octave -qf 

##setting enviroment 
menv;

%%% Loading Features & scaling 
printf("|--> Loading trainset features ...\n");

Xtrain1 = dlmread([curr_dir "/dataset/images2/features_64/Xtrain1.zat"]);  
Xtrain2 = dlmread([curr_dir "/dataset/images2/features_64/Xtrain2.zat"]); 
Xtrain3 = dlmread([curr_dir "/dataset/images2/features_64/Xtrain3.zat"]); 

Xtrain = [Xtrain1 , Xtrain2 , Xtrain3];
%Xtrain = [Xtrain1 , Xtrain2 ];
ytrain = dlmread([curr_dir "/dataset/images2/features_64/Ytrain.zat"]);
m = size(Xtrain, 1);
n = size(Xtrain,2);
ytrain = ones(m,1) + ytrain; 

printf("|-> performing feature scaling ...\n");
[Xtrain,mu,sigma] = treatContFeatures(Xtrain,1);


%%% Model training
NNMeta = buildNNMeta([n n 2]);
lambda = 15; 
Theta = cell(0,0);
while (1 == 1) 
  if (isempty(Theta))
    fprintf("|--> Neural Network Training from scratch (lambda=%f) ... \n",lambda);disp(NNMeta);
    [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda , iter = 1000, featureScaled = 1);
  else
    fprintf("|--> Neural Network Training using last trained Theta (lambda=%f) ... \n",lambda);disp(NNMeta);
    [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda , iter = 2000, ... 
         featureScaled = 1 , initialTheta = Theta); 
  endif 
  pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
  acc_train = mean(double(pred_train == ytrain)) * 100;
  fprintf("|-> Training Set Accuracy with feature normalization (lambda=%f): %f\n",lambda,acc_train);
  [_dir] = serializeNNTheta(Theta,rDir="tmp");
  fprintf("|-> Serialized Thetas into %s directory.\n",_dir); 

  %%% Predicting on Testset
  Xtest1 = dlmread([curr_dir "/dataset/images2/features_64/Xtest1.zat"]); 
  Xtest2 = dlmread([curr_dir "/dataset/images2/features_64/Xtest2.zat"]); 
  Xtest3 = dlmread([curr_dir "/dataset/images2/features_64/Xtest3.zat"]);

  Xtest = [Xtest1 , Xtest2 , Xtest3];
				%Xtest = [Xtest1 , Xtest2 ];
  [Xtest,mu_test,sigma_test] = treatContFeatures(Xtest,1,1,mu,sigma);
  pred_test = NNPredictMulticlass(NNMeta, Theta , Xtest , featureScaled = 1);
  pred_test = pred_test - ones(size(pred_test,1),1);
  pred = [(1:length(pred_test))' pred_test];
  ts = strftime ("%d_%m_%Y-%H%M", localtime (time ()));
  fn = ["tmp/predictions_" ts ".zat"];
  dlmwrite(fn,pred);
  printf("|-> Predictions serialized into %s \n",fn);
endwhile 

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


