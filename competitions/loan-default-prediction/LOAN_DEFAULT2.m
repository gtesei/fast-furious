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
  %%data = data(1:10000,:);
endif

y_loss = data(:,end);
y_def = (y_loss > 0) * 1 + (y_loss == 0)*2; %%% il default e' stato mappato con 1, mentre il caso loss == 0 con 2 

%%Xcat = [data(:,3) data(:,6) data(:,768) data(:,769)]; 
Xcont = [data(:,522) data(:,523) data(:,270)];  

%%%%%%%%%%%%% merge test set for encoding categorical features 
%%data = [];
# data = dlmread([curr_dir "/dataset/loan_default/" testFile]); %%NA clean in R
# Xcat_test = [data(:,3) data(:,6) data(:,768) data(:,769)]; 
# Xcat_tot = [Xcat;Xcat_test]; 
# data = [];
# [Xcat_totE,map,offset] = encodeCategoricalFeatures(Xcat_tot);
# Xcat_totE = []; Xcat_tot = [];
%%%%%%%%%%%%%%

%[XcatE,map,offset] = encodeCategoricalFeatures(Xcat,map,offset);
[Xcont,mu,sigma] = featureNormalize(Xcont);

X = [Xcont];
y = [y_def y_loss];

X = [ones(size(X,1),1) X];
[m,n] = size(X)
rand_indices = randperm(m);
[Xtrain,ytrain,Xval,yval] = splitTrainValidation(X(rand_indices,:),y(rand_indices,:),0.90); 

ytrain_def = ytrain(:,1);
ytrain_loss = ytrain(:,2);

yval_def = yval(:,1);
yval_loss = yval(:,2);


%%%%%%%%%%%%% BOOTSTRAP %%%%%%%%%
%%%% making training set 50% 0' & 50% 1'
# id0 = find(ytrain_loss == 0);
# id1 = find(ytrain_loss > 0);
# printf("|--> found id0 - %i , id1- %i ... making equal length ...\n",length(id0),length(id1));
# id0 = id0(1:length(id1),:);
# printf("|--> found id0 - %i , id1- %i ...\n",length(id0),length(id1));
# Xtrain_boot = [Xtrain(id1,:); Xtrain(id0,:)];
# ytrain_boot = [ytrain(id1,:); ytrain(id0,:)];

# %%% shuffle 
# rand_indices = randperm(size(Xtrain_boot,1));
# Xtrain_boot = Xtrain_boot(rand_indices,:);
# ytrain_boot = ytrain_boot(rand_indices,:);

# Xtrain_no_boot = Xtrain;
# ytrain_no_boot = ytrain;
# ytrain_def_no_boot = ytrain_def;
# ytrain_loss_no_boot = ytrain_loss;

# Xtrain = Xtrain_boot;
# ytrain = ytrain_boot;

# ytrain_def = ytrain(:,1);
# ytrain_loss = ytrain(:,2);   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
  
  %%logistic classifier 
  [all_theta] = oneVsAll(Xtrain, ytrain_def, 2, 0,800);
  pred_log = predictOneVsAll(all_theta, Xval);
  acc_log = mean(double(pred_log == yval_def)) * 100;
  fprintf("\n Logistic classifier - training set accuracy (p=%i,lambda=%f): %f\n", 1,0.1,acc_log);
  
  pred_comb_log = (pred_log == 2) .* 0 + (pred_log == 1) .* pred_loss;
  [mae_log] = MAE(pred_comb_log, yval_loss);
  printf("|-> COMBINED PREDICTION --> MAE on cross validation set = %f  \n",mae_log);

	



