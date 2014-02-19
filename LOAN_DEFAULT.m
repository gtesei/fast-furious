#! /opt/local/bin/octave -qf 

##setting enviroment 
menv;

%%% 1) FEATURES ENGINEERING 
printf("|--> FEATURES ENGINEERING ...\n");

data = dlmread([curr_dir "/dataset/loan_default/train_100.csv"]); %%NA filled in R
y_loss = data(:,end);
y_def = (y_loss > 0);

Xcat = [data(:,2) data(:,3)]; %% ... TODO 
Xcont = [data(:,4) data(:,5)]; %% ... TODO 

data = [];

[XcatE,map] = encodeCategoricalFeatures(Xcat);
[Xcont,mu,sigma] = treatContFeatures(Xcont,1);

X = [XcatE Xcont];
y = [y_def y_loss];

[m,n] = size(X);
rand_indices = randperm(m);
[Xtrain,ytrain,Xval,yval] = splitTrainValidation(X(rand_indices,:),y(rand_indices,:),0.70); 

ytrain_def = ytrain(:,1);
ytrain_loss = ytrain(:,2);

yval_def = yval(:,1);
yval_loss = yval(:,2);


%%% 2) DEFAULT CLASSIFIER 
printf("|--> DEFAULT CLASSIFIER  ...\n");
num_label = length(unique(ytrain_def));
[n_opt,J_opt] = findOptNeuronsPerLayer(Xtrain, ytrain_def, Xval, yval_def , lambda=0 ,start_neurons=-1,end_neurons=-1,step_fw=-1,hidden_layers=1);
[h_opt,J_opt] = findOptHiddenLayers(Xtrain, ytrain_def, Xval, yval_def , lambda=0);
NNMeta = buildNNMeta([(n_opt) repmat(n_opt,1,h_opt) num_label]);disp(NNMeta);
[lambda_opt,J_opt] = findOptLambda(NNMeta, Xtrain, ytrain_def, Xval, yval_def , lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]');

%% --> model parameters: n_opt , h_opt , lambda_opt
NNpars = [n_opt h_opt lambda_opt];
dlmwrite ('NNpars.zat', NNpars);


%%% 3) LOSS REGRESSOR 
printf("|--> LOSS REGRESSOR  ...\n");
[p_opt,J_opt] = findOptP_RegLin(Xtrain, ytrain_loss, Xval, yval_loss, p_vec = [1 2 3 4 5 6 7 8 9 10 12 20 60]' , lambda=0);
[reg_lambda_opt,J_opt] = findOptLambda_RegLin(Xtrain, ytrain_loss, Xval, yval_loss, lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' , p=p_opt);

%% --> model parameters: p_opt , reg_lambda_opt
REGpars = [p_opt reg_lambda_opt];
dlmwrite ('REGpars.zat', REGpars);

%%% 4) TRAINING CLASSIFIER / REGRESSOR 
printf("|--> TRAINING CLASSIFIERS  ...\n");

NNMeta = buildNNMeta([(n_opt) repmat(n_opt,1,h_opt) num_label]);disp(NNMeta);
[Theta] = trainNeuralNetwork(NNMeta, X, y_def, lambda_opt , iter = 200, featureScaled = 1);
pred_def = NNPredictMulticlass(NNMeta, Theta , X , featureScaled = 1);
acc = mean(double(pred_def == y_def)) * 100;
printf("|-> trained Neural Network default classifier. Accuracy on training set = %f  \n",acc);
[_dir] = serializeNNTheta(Theta);

rtheta = trainLinearReg(X, y_loss, , 400);
pred_loss = predictLinearReg(X,theta);
rerr = linearRegCostFunction(X, y_loss, rtheta, 0);
printf("|-> trained loss regressor. Error on training set = %f  \n",rerr);
dlmwrite ('rtheta.zat', rtheta);

%%% 5) PREDICTION ON TEST SET




	



