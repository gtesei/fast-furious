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
y_def = (y_loss > 15) * 1 + (y_loss >= 0& y_loss <= 15)*2; %%% il default e' stato mappato con 1, mentre il caso loss == 0 con 2 

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

  tic();
  
  %% --> performance
  NNMeta = buildNNMeta([(n - 1)  (n -1) (n-1) 2]);disp(NNMeta);
  [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain_def, 3 , iter = 400, featureScaled = 1);
  pred_def = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
  acc = mean(double(pred_def == yval_def)) * 100;
  printf("|-> trained Neural Network default classifier. Accuracy on cross validantion set = %f  \n",acc);

  %% --> performance
  [Xtrain_poly,mu,sigma] = treatContFeatures(Xtrain,1); 
  rtheta = trainLinearReg(Xtrain_poly, ytrain_loss, 3, 500);
  [Xval_poly,mu,sigma] = treatContFeatures(Xval,1,1,mu,sigma);
  pred_loss = predictLinearReg(Xval_poly,rtheta);
  [mae] = MAE(pred_loss, yval_loss);
  printf("|-> trained loss regressor. MAE on cross validation set = %f  \n",mae);

  %% combining predictions 
  pred_comb = (pred_def == 2) .* 0 + (pred_def == 1) .* pred_loss;
  [mae] = MAE(pred_comb, yval_loss);
  printf("|-> COMBINED PREDICTION --> MAE on cross validation set = %f  \n",mae);
  
  toc();

	



