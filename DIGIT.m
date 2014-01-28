#! /opt/local/bin/octave -qf 

##setting enviroment 
menv;

%%% Loading Features & scaling 
printf("|--> Loading trainset features ...\n");

data = dlmread([curr_dir "/dataset/digit/train.csv"]); 
m = size(data,1);
n = size(data,2) - 1;  
rand_indices = randperm(m);
data = data(rand_indices,:);
data = data(1:m/2,:);
m = size(data,1);
X = data(:,2:end);
y = data(:,1);
num_label = length(unique(y));
data = [];

printf("|--> loaded X(%i,%i) and y(%i)\n",m,n,m);

train_mode = 0;

if (train_mode)

	%%% Model training 
	NNMeta = buildNNMeta([n n num_label]);disp(NNMeta);
	lambda = 0;

	printf("|--> performing feature scaling and normalization on train dataset and cross validation dataset ...\n");
	printf("\n|--> splitting dataset into train set and cross validation set ...\n");
	rand_indices = randperm(m);
	[Xtrain,ytrain,Xval,yval] = splitTrainValidation(X(rand_indices,:),y(rand_indices),0.70);
	X = []; y = [];
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
	printf("|--> finding optimal lambda ... \n");
	%tic(); [l_opt,J_opt] = findOptLambda(NNMeta, Xtrain, ytrain, Xval, yval); toc();
	%pause;

else 

	%%%%%%% production 
	NNMeta = buildNNMeta([n n num_label]);disp(NNMeta);
	lambda = 3;
	[X,mu,sigma] = treatContFeatures(X,1);
	fprintf("|--> Neural Network Training  (lambda=%f) ... \n",lambda);
	%[Theta] = deserializeNNTheta(NNMeta,'Theta28_01_2014-1129');
	%[Theta] = trainNeuralNetwork(NNMeta, X, y, lambda , iter = 200, featureScaled = 1 , initialTheta = Theta);
	[Theta] = trainNeuralNetwork(NNMeta, X, y, lambda , iter = 400, featureScaled = 1);
	pred_train = NNPredictMulticlass(NNMeta, Theta , X , featureScaled = 1);
	acc_train = mean(double(pred_train == y)) * 100;
	fprintf("|-> Training Set Accuracy with feature normalization (lambda=%f): %f\n",lambda,acc_train);
	[_dir] = serializeNNTheta(Theta);
	fprintf("|-> Serialized Thetas into %s directory.\n",_dir); 

	%%% Predicting on Testset
	fprintf("|--> loadinf test set ... \n");
	X = []; y = [];
	X = dlmread([curr_dir "/dataset/digit/test.csv"]); 
	[X,mu_test,sigma_test] = treatContFeatures(X,1,1,mu,sigma);
	pred_test = NNPredictMulticlass(NNMeta, Theta , X , featureScaled = 1);
	pred = [(1:length(pred_test))' pred_test];
	fn = "digit_prediction.zat";
	dlmwrite(fn,pred);
	printf("|-> Predictions serialized into %s \n",fn);


endif	







