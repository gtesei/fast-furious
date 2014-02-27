#! /opt/local/bin/octave -qf 

%%%% setting enviroment 
menv;

find_par_mode = 1;

trainFile = "train_NO_NA_oct.zat"; 
testFile = "test_v2_NA_CI_oct.zat";   


%%% 1) FEATURES ENGINEERING 
printf("|--> FEATURES BUILDING ...\n");

data = dlmread([curr_dir "/dataset/loan_default/" trainFile]); %%NA clean in R
if (find_par_mode)
  [m,n] = size(data);
  rand_indices = randperm(m);
  data = data(rand_indices,:);
  data = data(1:10000,:);
endif

y_loss = data(:,end);
y_def = (y_loss > 0) * 1 + (y_loss == 0)*2; %%% il default e' stato mappato con 1, mentre il caso loss == 0 con 2 

Xcat = [data(:,3) data(:,6) data(:,768) data(:,769)]; 
Xcont = [data(:,2) data(:,4:5) data(:,7:767) data(:,770)];  

%%%%%%%%%%%%% merge test set for encoding categorical features 
data = [];
data = dlmread([curr_dir "/dataset/loan_default/" testFile]); %%NA clean in R
Xcat_test = [data(:,3) data(:,6) data(:,768) data(:,769)]; 
Xcat_tot = [Xcat;Xcat_test]; 
data = [];
[Xcat_totE,map,offset] = encodeCategoricalFeatures(Xcat_tot);
Xcat_totE = []; Xcat_tot = [];
%%%%%%%%%%%%%%

[XcatE,map,offset] = encodeCategoricalFeatures(Xcat,map,offset);
[Xcont,mu,sigma] = featureNormalize(Xcont);

X = [XcatE Xcont];
y = [y_def y_loss];

X = [ones(size(X,1),1) X];
[m,n] = size(X)
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
  [n_opt,J_opt] = findOptNeuronsPerLayer(Xtrain, ytrain_def, Xval, yval_def , lambda=0 ,start_neurons=-1,end_neurons=-1,step_fw=-1,hidden_layers=1,_num_label=-1, verbose=0);
  [h_opt,J_opt] = findOptHiddenLayers(Xtrain, ytrain_def, Xval, yval_def , lambda=0,neurons_hidden_layers=n_opt,_num_label=-1, verbose=0);
  NNMeta = buildNNMeta([(n - 1) repmat(n_opt,1,h_opt) num_label]);disp(NNMeta);
  [lambda_opt,J_opt] = findOptLambda(NNMeta, Xtrain, ytrain_def, Xval, yval_def , lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]');

  %% --> model parameters: n_opt , h_opt , lambda_opt
  NNpars = [n_opt h_opt lambda_opt];
  printf("|--> OPTIMAL PARAMETERS NEURAL NETWORK DEFAULT CLASSIFIER  --> opt. number of hidden layers(h_opt) = %i , opt. number of neurons for hidden layers(n_opt) = % i , opt. lambda = %f\n",h_opt,n_opt,lambda_opt);
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
  ytrain_closs = (ytrain_loss != 0) .* ytrain_loss + (ytrain_loss == 0) * 101; %%% il caso loss == 0 e' stato mappato con 101
  yval_closs = (yval_loss != 0) .* yval_loss + (yval_loss == 0 ) * 101; 
  [n_opt_loss,J_opt] = findOptNeuronsPerLayer(Xtrain, ytrain_loss, Xval, yval_closs , lambda=0 ,start_neurons=-1,end_neurons=-1,step_fw=-1,hidden_layers=1,_num_label=101,verbose=0);
  [h_opt_loss,J_opt] = findOptHiddenLayers(Xtrain, ytrain_loss, Xval, yval_closs , lambda=0 , neurons_hidden_layers=n_opt_loss,_num_label=101,verbose=0);
  NNMeta = buildNNMeta([(n - 1) repmat(n_opt_loss,1,h_opt_loss) num_label_loss]);disp(NNMeta);
  [lambda_opt_loss,J_opt] = findOptLambda(NNMeta, Xtrain, ytrain_closs, Xval, yval_loss , lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]',_num_label=101);

  %% --> model parameters: n_opt_loss , h_opt_loss , lambda_opt_loss
  NNpars_loss = [n_opt_loss h_opt_loss lambda_opt_loss];
  printf("|--> OPTIMAL PARAMETERS NEURAL NETWORK LOSS CLASSIFIER  --> opt. number of hidden layers(h_opt_loss) = %i , opt. number of neurons for hidden layers(n_opt_loss) = % i , opt. lambda = %f\n",h_opt_loss,n_opt_loss,lambda_opt_loss);
  dlmwrite ('NNpars_loss.zat', NNpars_loss);

  %% --> performance
  NNMeta = buildNNMeta([(n - 1) repmat(n_opt_loss,1,h_opt_loss) num_label_loss]);disp(NNMeta);
  [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain_closs, lambda_opt_loss , iter = 200, featureScaled = 1);
  pred_loss_nn = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
  pred_loss_nn = (pred_loss_nn != 101) .* pred_loss_nn + (pred_loss_nn == 101 ) * 0; %%% il caso loss == 0 e' stato mappato con 101 
  [mae] = MAE(pred_loss_nn, yval_closs);
  printf("|-> trained Neural Network --- LOSS --- classifier. MAE on cross validantion set = %f  \n",mae);


  %%% 3) FINDING BEST PARAMS LOSS REGRESSOR 
  printf("|--> FINDING BEST PARAMS LOSS REGRESSOR  ...\n");
  [p_opt,J_opt] = findOptP_RegLin(Xtrain, ytrain_loss, Xval, yval_loss, p_vec = [1 2 3 4 5 6 7 8 9 10 12 20 60]' , lambda=0);
  [reg_lambda_opt,J_opt] = findOptLambda_RegLin(Xtrain, ytrain_loss, Xval, yval_loss, lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' , p=p_opt);

  %% --> model parameters: p_opt , reg_lambda_opt
  REGpars = [p_opt reg_lambda_opt];
  printf("|--> OPTIMAL LINEAR REGRESSOR PARAMS -->  opt. number of polinomial degree (p_opt) = %i , opt. lambda = %f\n",p_opt,reg_lambda_opt);
  dlmwrite ('REGpars.zat', REGpars);

  %% --> performance
  [Xtrain_poly,mu,sigma] = treatContFeatures(Xtrain,p_opt); 
  rtheta = trainLinearReg(Xtrain_poly, ytrain_loss, reg_lambda_opt, 400);
  [Xval_poly,mu,sigma] = treatContFeatures(Xval,p_opt,1,mu,sigma);
  pred_loss = predictLinearReg(Xval_poly,rtheta);
  [mae] = MAE(pred_loss, yval_loss);
  printf("|-> trained loss regressor. MAE on cross validation set = %f  \n",mae);

  %% combining predictions 
  pred_comb = (pred_def == 2) .* 0 + (pred_def == 1) .* pred_loss;
  [mae] = MAE(pred_comb, yval_loss);
  printf("|-> COMBINED PREDICTION --> MAE on cross validation set = %f  \n",mae);
  
  toc();
else 
  tic();

  %%% NN default classifier params 
  n_opt = 790;
  h_opt = 1;
  lambda_opt = 0;

  %%% NN Loss classifier params
  n_opt_loss = 830; 
  h_opt_loss = 1;  %% TODO better 
  lambda_opt_loss = 0.01; 

  %%% Linear Regressor params 
  p_opt = 1;
  reg_lambda_opt = 0.001000;

  %%% 4) TRAINING CLASSIFIERS / REGRESSOR 
  printf("|--> TRAINING CLASSIFIERS  ...\n");

  printf("|-> training default classifier...  \n");
  NNMeta_def = buildNNMeta([(n - 1) repmat(n_opt,1,h_opt) 2]);disp(NNMeta_def);
  [Theta_def] = trainNeuralNetwork(NNMeta_def, X, y_def, lambda_opt , iter = 500, featureScaled = 1);
  pred_def = NNPredictMulticlass(NNMeta_def, Theta_def , X , featureScaled = 1);
  acc = mean(double(pred_def == y_def)) * 100;
  printf("|-> trained Neural Network default classifier. Accuracy on training set = %f  \n",acc);
  [_dir] = serializeNNTheta(Theta_def,dPrefix="Theta-def-class");
  
  printf("|-> training  loss classifier...  \n");
  NNMeta_loss = buildNNMeta([(n - 1) repmat(n_opt_loss,1,h_opt_loss) 101]);disp(NNMeta_loss);
  y_closs = (y_loss != 0) .* y_loss + (y_loss == 0) * 101;                                      %%% il caso loss == 0 e' stato mappato con 101
  [Theta_loss] = trainNeuralNetwork(NNMeta_loss, X, y_closs, lambda_opt_loss , iter = 500, featureScaled = 1);
  pred_closs = NNPredictMulticlass(NNMeta_loss, Theta_loss , X , featureScaled = 1);
  [mae_closs] = MAE(pred_closs, y_closs);
  printf("|-> LOSS CLASSIFIER --> MAE train set = %f  \n",mae_closs);
  [_dir] = serializeNNTheta(Theta_loss,dPrefix="Theta-loss-class");

  printf("|-> training  linear regressor ...  \n");
  X = treatContFeatures(X,p_opt); %% cambia X 
  rtheta = trainLinearReg(X, y_loss, 0, 500);
  pred_loss = predictLinearReg(X,rtheta);
  rerr = linearRegCostFunction(X, y_loss, rtheta, 0); 
  printf("|-> trained loss regressor. Error on training set = %f  \n",rerr);
  [mae] = MAE(pred_loss, y_loss);
  printf("|-> LINEAR REGRESSOR PREDICTION --> MAE train set = %f  \n",mae);
  dlmwrite ('rtheta.zat', rtheta);
  
  printf("|-> combining predictions ...  \n");
  pred_comb = (pred_def == 0) .* 0 + (pred_def == 1) .* pred_loss;
  [mae] = MAE(pred_comb, y_loss);
  printf("|-> COMBINED PREDICTION --> MAE training set = %f  \n",mae);

  %%% 5) PREDICTION ON TEST SET
  X = [];
  data = dlmread([curr_dir "/dataset/loan_default/" testFile]); %%NA clean in R
  Xcat = [data(:,3) data(:,6) data(:,768) data(:,769)]; 
  Xcont = [data(:,2) data(:,4:5) data(:,7:767) data(:,770)];  
  data = [];
  
  [XcatE,map,offset] = encodeCategoricalFeatures(Xcat,map,offset);
  [Xcont,mu,sigma] = featureNormalize(Xcont);
  
  X = [XcatE Xcont];
  X = [ones(size(X,1),1) X];
  [m,n] = size(X)
  
  pred_def  = NNPredictMulticlass(NNMeta_def, Theta_def , X , featureScaled = 1);
  pred_closs = NNPredictMulticlass(NNMeta_loss, Theta_loss , X , featureScaled = 1);
  
  %%% linear regressor 
  X = treatContFeatures(X,p_opt); %% cambia X
  pred_loss = predictLinearReg(X,rtheta);
  pred_comb = (pred_def == 0) .* 0 + (pred_def == 1) .* pred_loss;
  
  sub_comb = [(1:m)' pred_comb];
  sub_closs = [(1:m)' pred_closs];
  sub_loss = [(1:m)' pred_loss];
  
  dlmwrite ('sub_comb.csv', sub_comb,",");
  dlmwrite ('sub_closs.csv', sub_closs,",");
  dlmwrite ('sub_loss.csv', sub_loss,",");

  toc();
endif 
	



