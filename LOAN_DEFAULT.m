#! /opt/local/bin/octave -qf 

%%%% setting enviroment 
menv;

find_par_mode = 1;


%%% 1) FEATURES ENGINEERING 
printf("|--> FEATURES BUILDING ...\n");

data = dlmread([curr_dir "/dataset/loan_default/train_impute_mean.zat"]); %%NA filled in R
if (find_par_mode)
  [m,n] = size(data);
  rand_indices = randperm(m);
  data = data(rand_indices,:);
  data = data(1:10000,:);
endif

y_loss = data(:,end);
y_def = (y_loss > 0);

Xcat = [data(:,3) data(:,6) data(:,768) data(:,769)]; 
Xcont = [data(:,2) data(:,4:5) data(:,7:767) data(:,770:771)];  

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

if (find_par_mode) 
  tic();
  
  %%% 2) FINDING BEST PARAMS DEFAULT CLASSIFIER 
  printf("|--> FINDING BEST PARAMS DEFAULT CLASSIFIER   ...\n");
  num_label = length(unique(ytrain_def));
  [n_opt,J_opt] = findOptNeuronsPerLayer(Xtrain, ytrain_def, Xval, yval_def , lambda=0 ,start_neurons=-1,end_neurons=-1,step_fw=-1,hidden_layers=1,verbose=0);
  [h_opt,J_opt] = findOptHiddenLayers(Xtrain, ytrain_def, Xval, yval_def , lambda=0,neurons_hidden_layers=n_opt,verbose=0);
  NNMeta = buildNNMeta([(n - 1) repmat(n_opt,1,h_opt) num_label]);disp(NNMeta);
  [lambda_opt,J_opt] = findOptLambda(NNMeta, Xtrain, ytrain_def, Xval, yval_def , lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]');

  %% --> model parameters: n_opt , h_opt , lambda_opt
  NNpars = [n_opt h_opt lambda_opt];
  printf("|--> OPTIMAL PARAMETERS NEURAL NETWORK DEFAULT CLASSIFIER  --> opt. number of hidden layers(h_opt) = %i , opt. number of neurons for hidden layers(n_opt) = % i , opt. lambda = %f\n",h_opt,n_opt,lamda_opt);
  dlmwrite ('NNpars.zat', NNpars);

  %% --> performance
  NNMeta = buildNNMeta([(n - 1) repmat(n_opt,1,h_opt) num_label]);disp(NNMeta);
  [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain_def, lambda_opt , iter = 200, featureScaled = 1);
  pred_def = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
  acc = mean(double(pred_def == yval_def)) * 100;
  printf("|-> trained Neural Network default classifier. Accuracy on cross validantion set = %f  \n",acc);

  %%% 2.5) FINDING BEST PARAMS NEURAL NETWORK LOSS CLASSIFIER  
  printf("|--> FINDING BEST PARAMS NEURAL NETWORK LOSS CLASSIFIER  ...\n");
  num_label_loss = 101;
  [n_opt_loss,J_opt] = findOptNeuronsPerLayer(Xtrain, ytrain_loss, Xval, yval_loss , lambda=0 ,start_neurons=-1,end_neurons=-1,step_fw=-1,hidden_layers=1,verbose=0);
  [h_opt_loss,J_opt] = findOptHiddenLayers(Xtrain, ytrain_loss, Xval, yval_def , lambda=0 , neurons_hidden_layers=n_opt_loss,verbose=0);
  NNMeta = buildNNMeta([(n - 1) repmat(n_opt_loss,1,h_opt_loss) num_label_loss]);disp(NNMeta);
  [lambda_opt_loss,J_opt] = findOptLambda(NNMeta, Xtrain, ytrain_loss, Xval, yval_loss , lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]');

  %% --> model parameters: n_opt_loss , h_opt_loss , lambda_opt_loss
  NNpars_loss = [n_opt_loss h_opt_loss lambda_opt_loss];
  printf("|--> OPTIMAL PARAMETERS NEURAL NETWORK LOSS CLASSIFIER  --> opt. number of hidden layers(h_opt_loss) = %i , opt. number of neurons for hidden layers(n_opt_loss) = % i , opt. lambda = %f\n",h_opt_loss,n_opt_loss,lamda_opt_loss);
  dlmwrite ('NNpars_loss.zat', NNpars_loss);

  %% --> performance
  NNMeta = buildNNMeta([(n - 1) repmat(n_opt_loss,1,h_opt_loss) num_label_loss]);disp(NNMeta);
  [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain_loss, lambda_opt_loss , iter = 200, featureScaled = 1);
  pred_loss_nn = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
  [mae] = MSE(pred_loss_nn, yval_loss);
  printf("|-> trained Neural Network --- LOSS --- classifier. MAE on cross validantion set = %f  \n",mae);


  %%% 3) FINDING BEST PARAMS LOSS REGRESSOR 
  printf("|--> FINDING BEST PARAMS LOSS REGRESSOR  ...\n");
  [p_opt,J_opt] = findOptP_RegLin(Xtrain, ytrain_loss, Xval, yval_loss, p_vec = [1 2 3 4 5 6 7 8 9 10 12 20 60]' , lambda=0);
  [reg_lambda_opt,J_opt] = findOptLambda_RegLin(Xtrain, ytrain_loss, Xval, yval_loss, lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' , p=p_opt);

  %% --> model parameters: p_opt , reg_lambda_opt
  REGpars = [p_opt reg_lambda_opt];
  printf("|--> OPTIMAL LINERA REGRESSOR PARAMS -->  opt. number of poliinomial degree (p_opt) = %i , opt. lambda = %f\n",p_opt,reg_lamda_opt);
  dlmwrite ('REGpars.zat', REGpars);

  %% --> performance 
  rtheta = trainLinearReg(Xtrain, ytrain_loss, 0, 400);
  pred_loss = predictLinearReg(Xtrain,rtheta);
  [mae] = MSE(pred_loss, yval_loss);
  printf("|-> trained loss regressor. MAE on cross validation set = %f  \n",mae);

  %% combining predictions 
  pred_comb = (pred_def == 0) .* 0 + (pred_def == 1) .* pred_loss;
  [mae] = MSE(pred_comb, yval_loss);
  printf("|-> COMBINED PREDICTION --> MAE on cross validation set = %f  \n",mae);
  
  toc();
else 
  tic();

  %%% NN default classifier params 
  n_opt = 0;
  h_opt = 0;
  lambda_opt = 0;

  %%% NN Loss classifier params
  n_opt_loss = 0;
  h_opt_loss = 0;
  lambda_opt_loss = 0;

  %%% Linear Regressor params 
  p_opt = 0;
  reg_lambda_opt = 0;

  %%% 4) TRAINING CLASSIFIER / REGRESSOR 
  printf("|--> TRAINING CLASSIFIERS  ...\n");

  NNMeta = buildNNMeta([(n - 1) repmat(n_opt,1,h_opt) num_label]);disp(NNMeta);
  [Theta] = trainNeuralNetwork(NNMeta, X, y_def, lambda_opt , iter = 200, featureScaled = 1);
  pred_def = NNPredictMulticlass(NNMeta, Theta , X , featureScaled = 1);
  acc = mean(double(pred_def == y_def)) * 100;
  printf("|-> trained Neural Network default classifier. Accuracy on training set = %f  \n",acc);
  [_dir] = serializeNNTheta(Theta);

  rtheta = trainLinearReg(X, y_loss, 0, 400);
  pred_loss = predictLinearReg(X,rtheta);
  rerr = linearRegCostFunction(X, y_loss, rtheta, 0); 
  printf("|-> trained loss regressor. Error on training set = %f  \n",rerr);
  dlmwrite ('rtheta.zat', rtheta);

  %%% 5) PREDICTION ON TEST SET


  toc();
endif 
	



