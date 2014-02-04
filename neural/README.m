%% Fast-Furious 
%  Linear Regession 
%  =================
%  Variant #1: Continuous Features 
%              In this case, by default, it's performed features scaling and normalization 
%               - [X]treatContFeatures(X,p)  where p is the polinomial degree of the model (p=1 means linear model)          
%  ==========
%
%  Variant #2: Categarical Features 
%            In this case, by default, it's performed features encoding    
%               - [X]treatCatFeatures(X)
%  ==========
%
1;
%
%  Variant #1 / Use Cases .
%  ======================== 

function [is_ok] = go()
  is_ok = 1;
#   is_ok &= var1_doBasicUseCase();
#   is_ok &= var1_doFindOptParamsUseCase();
#   is_ok &= var1_doComparisonPurePolyDatasetUseCase();
   is_ok &= var1_doBufferedUseCase();
endfunction 

function [is_ok] = var1_doBasicUseCase()
  
 is_ok = 0; % return as 1 if ok  
 printf("Running var1_doBasicUseCase ... \n"); 

 %% load images 
 load ('dataset/images/digits.mat'); %load X and y
 printf("|-> Loading and Visualizing Data ...\n"); 
 m = size(X, 1);
 rand_indices = randperm(m);
 sel = X(rand_indices(1:100), :);
 displayData(sel);
 pause;

 %% check cost function 
 fprintf("\n|-> Loading Saved Neural Network Parameters ...\n");
 NNMeta = buildNNMeta([400 25 10]);disp(NNMeta);
 load('ex4weights.mat');
 nn_params = [Theta1(:) ; Theta2(:)]; 
 fprintf("|-> Feedforward Using Neural Network ...\n");
 lambda = 0;
 J = nnCostFunction(nn_params, NNMeta , X, y, lambda);
 fprintf(["Cost at parameters (loaded from ex4weights): %f "...
         "\n(this value should be about 0.287629)\n"], J);
 if ( J - 0.287629 > 0.01 )
   error("J - 0.287629 > 0.01");
 else 
   printf("OK");
 endif 

 %% check cost function regularized 
 fprintf('\nChecking Cost Function (w/ Regularization) ... \n')
 lambda = 1;
 J = nnCostFunction(nn_params, NNMeta , X, y, lambda);
 fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], J); 
 if ( J - 0.383770 > 0.01 )
   error("J - 0.383770 > 0.01");
 else 
   printf("OK");
 endif 
 
 %% sigmoid gradient
 fprintf('\nEvaluating sigmoid gradient...\n')
 g = sigmoidGradient([1 -0.5 0 0.5 1]);
 fprintf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
 fprintf('%f ', g);
 fprintf('\n\n');

 %% Initializing Pameters
 fprintf('\nInitializing Neural Network Parameters ...\n');
 L = length(NNMeta.NNArchVect); 
 Theta = cell(L-1,1);
 for i = 1:(L-1)
  Theta(i,1) = randInitializeWeights(NNMeta.NNArchVect(i),NNMeta.NNArchVect(i+1));
 endfor  
 initial_nn_params = [];
 for i = fliplr(1:L-1)
  initial_nn_params =  [ cell2mat(Theta(i))(:) ;  initial_nn_params(:) ];
 endfor

 %%Checking Backpropagation (lambda = 0)
 lambda = 0;
 fprintf('\nChecking Backpropagation (lambda = %f)... \n',lambda);
 checkNNGradients(lambda);
 
 % Checking Backpropagation (lambda != 0).
 lambda = 3;
 fprintf('\nChecking Backpropagation (lambda = %f)... \n',lambda);
 checkNNGradients(lambda);
 
 % Also output the costFunction debugging values
 debug_J  = nnCostFunction(nn_params, NNMeta, X, y, lambda);
 fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f ' ...
         '\n(this value should be about 0.576051)\n\n'], debug_J);
 
 % Training Neural Network
 fprintf('\nTraining Neural Network... \n');
 lambda = 1;
 [Theta] = trainNeuralNetwork(NNMeta, X, y, lambda , iter = 50 );
 
 % Visualizing Neural Network
 fprintf('\nVisualizing Neural Network... \n')
 displayData(cell2mat(Theta(1,1))(:, 2:end));
 
 % Predict
 [pred] = NNPredictMulticlass(NNMeta, Theta , X);
 
 accuracy = mean(double(pred == y)) * 100; 
 fprintf('\nTraining Set Accuracy: %f\n', accuracy);

 %% -- splitting data set into training set and cross validation set  
 printf("\n|--> splitting dataset into train set and cross validation set ...\n");
 m = size(X, 1);
 rand_indices = randperm(m);
 [Xtrain,ytrain,Xval,yval] = splitTrainValidation(X(rand_indices,:),y(rand_indices),0.70);
 printf("|--> performing feature scaling and normalization on train dataset and cross validation dataset ...\n" );

 %% -- feature normalization
 p = 1; lambda = 0;
 printf("|-> Repeating with feature normalization ... \n "); 
 [Xtrain,mu,sigma] = treatContFeatures(Xtrain,p);
 [Xval,mu_val,sigma_val] = treatContFeatures(Xval,p,1,mu,sigma);
 [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda , iter = 100, featureScaled = 1);
 pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
 pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
 acc_train = mean(double(pred_train == ytrain)) * 100;
 acc_val = mean(double(pred_val == yval)) * 100;
 fprintf("Training Set Accuracy with feature normalization (p=%i,lambda=%f): %f\n", p,lambda,acc_train);
 fprintf("Cross Validation Set Accuracy with feature normalization (p=%i,lambda=%f): %f\n", p,lambda,acc_val);
 
 %% -- testing initialTheta 
 printf("|-> Repeating with starting from trained Theta ... \n "); 
 [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda , iter = 100, featureScaled = 1 , initialTheta = Theta);
 pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
 pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
 acc_train = mean(double(pred_train == ytrain)) * 100;
 acc_val = mean(double(pred_val == yval)) * 100;
 fprintf("Training Set Accuracy with feature normalization (p=%i,lambda=%f): %f\n", p,lambda,acc_train);
 fprintf("Cross Validation Set Accuracy with feature normalization (p=%i,lambda=%f): %f\n", p,lambda,acc_val);
 
 %% -- testing serializing / deserializing Theta 
 printf("|-> Testing serializing Theta ... \n"); 
 [_dir] = serializeNNTheta(Theta);
 printf("|-> created directory %s and serialized Thetas inside \n",_dir);
 printf("|-> deserializing Thetas from %s  ... \n",_dir);
 [_Theta] = deserializeNNTheta(NNMeta,_dir);
 L = length(NNMeta.NNArchVect); 
 diff = cell(L-1,1);
 sq = 0;
 for i = 1:(L-1)
     diff(i,1) = cell2mat(Theta(i,1)) - cell2mat(_Theta(i,1));
     s = cell2mat(diff(i,1));
     s = s .* s;
     sq += sum(s(:));
 endfor  
 printf("|-> DIFFERENCE: \n"); %disp(cell2mat(diff(1,1))(1,:));printf("\n");
 printf("|-> sum squares diffs:%f \n",sq);
 
 %% -- Testing and serializing prediction   
 pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
 pred = [(1:length(pred_val))' pred_val];
 fn = "prediction.zat";
 dlmwrite(fn,pred);
 printf("|-> prediction serialized into %s \n",fn);
 
 if ( accuracy > 94 )  % put correctness tests here
   is_ok = 1;
   printf("Test case passed.\n");
 else 
   is_ok = 0;             
   printf("Accuracy = %f < 94   \n",accuracy);
   error("Test case NOT passed.\n"); 
 endif 

endfunction

function [is_ok] = var1_doFindOptParamsUseCase()
 
is_ok = 0; % return as 1 if ok  
 
printf("Running var1_doFindOptParamsUseCase ... \n"); 

%% load images 
load ('dataset/images/digits.mat'); %load X and y
printf("|-> Loading and Visualizing Data ...\n"); 
m = size(X, 1);
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);
pause;

printf("|--> performing feature scaling and normalization on train dataset and cross validation dataset ...\n");
printf("\n|--> splitting dataset into train set and cross validation set ...\n");
m = size(X, 1);
rand_indices = randperm(m);
[Xtrain,ytrain,Xval,yval] = splitTrainValidation(X(rand_indices,:),y(rand_indices),0.70);
 
[Xtrain,mu,sigma] = treatContFeatures(Xtrain,1);
[Xval,mu_val,sigma_val] = treatContFeatures(Xval,1,1,mu,sigma);
if (mu != mu_val | sigma != sigma_val) 
     disp(mu); disp(mu_val); disp(sigma); disp(sigma_val);
     error("error in function treatContFeatures: mu != mu_val or sigma != sigma_val - displayed mu,mu_val,sigma,sigma_val .. ");
endif


%% finding optimal number of hidden layers
lambda = 0;
printf("|--> finding optimal number of neurons per layer ... \n");
%tic(); [h_opt,J_opt] = findOptHiddenLayers(Xtrain, ytrain, Xval, yval,lambda); toc();
pause;

%% finding optimal number of neurons per layer 
printf("|--> finding optimal number of neurons per layer ... \n");
%tic(); [s_opt,J_opt] = findOptNeuronsPerLayer(Xtrain, ytrain, Xval, yval,lambda,start_neurons=10,end_neurons=100,step_fw=10); toc();
pause;

%% finding optimal lambda 
printf("|--> finding optimal number of neurons per layer ... \n");
NNMeta = buildNNMeta([400 85 10]);disp(NNMeta);
tic(); [l_opt,J_opt] = findOptLambda(NNMeta, Xtrain, ytrain, Xval, yval); toc();
pause;













 

 
#  printf("|--> finding optimal regularization parameter ... \n");
#  tic(); [lambda_opt,J_opt] = findOptLambda_RegLin(Xtrain, ytrain, Xval, yval); toc();
#  pause;
 
#  printf("|--> computing learning curve ... \n");
#   tic(); [error_train,error_val] = learningCurve_RegLin(Xtrain, ytrain, Xval, yval); toc();
#  pause;
 
#  if ( p_opt == 3 )  % put correctness tests here 
#    is_ok = 1;
#    printf("Test case passed.\n");
#  else 
#    is_ok = 0;    
#    error("Test case NOT passed.\n"); 
#  endif 

endfunction

# function [is_ok] = var1_doComparisonPurePolyDatasetUseCase()
#  global curr_dir; 

#  is_ok = 0; % return as 1 if ok  
 
#  printf("Running var1_doComparisonPurePolyDatasetUseCase ... \n"); 
 
#  _Xtrain = dlmread([curr_dir "/dataset/poly/poly_pure_Xtrain.zat"]);
#  ytrain =dlmread([curr_dir "/dataset/poly/poly_pure_ytrain.zat"]);
#  _Xval =dlmread([curr_dir "/dataset/poly/poly_pure_Xval.zat"]);
#  yval = dlmread([curr_dir "/dataset/poly/poly_pure_yval.zat"]);
 
#  printf("|--> performing feature scaling and normalization on train dataset and cross validation dataset ...\n");
#  [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,1);
#  [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,1,1,mu,sigma);
#  if (mu != mu_val | sigma != sigma_val) 
#     disp(mu); disp(mu_val); disp(sigma); disp(sigma_val);
#     error("error in function treatContFeatures: mu != mu_val or sigma != sigma_val - displayed mu,mu_val,sigma,sigma_val .. ");
#  endif
 
#  printf("|--> finding optimal polinomial degree ... \n");
#  tic(); [p_opt,J_opt] = findOptP_RegLin(Xtrain, ytrain, Xval, yval); toc();
#  pause;
 
#  printf("|--> finding optimal regularization parameter ... \n");
#  tic(); [lambda_opt,J_opt] = findOptLambda_RegLin(Xtrain, ytrain, Xval, yval); toc();
#  pause;
 
#  printf("|--> computing learning curve ... \n");
#   tic(); [error_train,error_val] = learningCurve_RegLin(Xtrain, ytrain, Xval, yval); toc();
#  pause;
 
#  ##normal equation p = 1, lambda = 0
#  p = 1; lambda = 0;
#  printf("|--> finding optimal solution with normal equation p = %i and lambda = %f \n",p,lambda);
#  tic();
#  [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
#  [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
#  [theta] = normalEqn_RegLin(Xtrain,ytrain,lambda);
#  y_pred = predictLinearReg(Xval,theta);
#  y_train_pred = predictLinearReg(Xtrain,theta);
#  cost_val_ne1 = MSE(y_pred, yval);
#  cost_train = MSE(y_train_pred, ytrain);
#  toc();
#  printf("MSE on training set = %f \n",cost_train);
#  printf("MSE on cross validation set = %f \n",cost_val_ne1);

#  ## p = 1 , lambda = 0                                                                                                                                                                   
#  p = 1;lambda=0;
#  printf("|--> trying gradient descent (optimized) with p = %i and lambda = %f \n",p,lambda);
#  tic();
#  [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
#  [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
#  [theta] = trainLinearReg(Xtrain, ytrain, lambda , 200 );
#  y_pred = predictLinearReg(Xval,theta);
#  y_train_pred =predictLinearReg(Xtrain,theta);
#  cost_val_gd1 = MSE(y_pred, yval);
#  cost_train_gd1 = MSE(y_train_pred, ytrain);
#  toc();
#  printf("MSE on training set = %f \n",cost_train_gd1);
#  printf("MSE on cross validation set = %f \n",cost_val_gd1);

#  ##normal equation p = 4, lambda = 0                                                                                                                                                          
#  p = 4; lambda = 0;
#  printf("|--> finding optimal solution with normal equation p = %i and lambda = %f \n",p,lambda);
#  tic();
#  [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
#  [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
#  [theta] = normalEqn_RegLin(Xtrain,ytrain,lambda);
#  y_pred = predictLinearReg(Xval,theta);
#  y_train_pred =predictLinearReg(Xtrain,theta);
#  cost_val = MSE(y_pred, yval);
#  cost_train = MSE(y_train_pred, ytrain); 
#  toc();
#  printf("MSE on training set = %f \n",cost_train);
#  printf("MSE on cross validation set = %f \n",cost_val);

#  ## p = 5 , lambda = 1                                                                                                                                                                                    
#  p = 5;lambda=1;
#  printf("|--> trying gradient descent (optimized) with p = %i and lambda = %f \n",p,lambda);
#  tic();
#  [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
#  [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
#  [theta] = trainLinearReg(Xtrain, ytrain, lambda , 200 );
#  y_pred = predictLinearReg(Xval,theta);
#  y_train_pred =predictLinearReg(Xtrain,theta);
#  cost_val = MSE(y_pred, yval);
#  cost_train = MSE(y_train_pred, ytrain);
#  toc();
#  printf("MSE on training set = %f \n",cost_train);
#  printf("MSE on cross validation set = %f \n",cost_val);


#  ## p = 4 , lambda = 0 , MaxIter = 200
#  p = 4;lambda=0;MaxIter = 200;
#  printf("|--> trying gradient descent (optimized) with p = %i , lambda = %f , MaxIter = %i \n",p,lambda,MaxIter);
#  tic();  
#  [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
#  [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
#  [theta] = trainLinearReg(Xtrain, ytrain, lambda , MaxIter );
#  y_pred = predictLinearReg(Xval,theta);
#  y_train_pred =predictLinearReg(Xtrain,theta);
#  cost_val = MSE(y_pred, yval);
#  cost_train = MSE(y_train_pred, ytrain);
#  toc(); 
#  printf("MSE on training set = %f \n",cost_train); 
#  printf("MSE on cross validation set = %f \n",cost_val);

#  ## p = 4 , lambda = 0 , MaxIter = 400                                                                                                                                                                   
#  p = 4;lambda=0;MaxIter= 400;
#  printf("|--> trying gradient descent (optimized) with p = %i , lambda = %f , MaxIter = %i \n",p,lambda,MaxIter);
#  tic();
#  [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
#  [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
#  [theta] = trainLinearReg(Xtrain, ytrain, lambda , MaxIter );
#  y_pred = predictLinearReg(Xval,theta);
#  y_train_pred =predictLinearReg(Xtrain,theta);
#  cost_val = MSE(y_pred, yval);
#  cost_train = MSE(y_train_pred, ytrain);
#  toc();
#  printf("MSE on training set = %f \n",cost_train);
#  printf("MSE on cross validation set = %f \n",cost_val);

#  ## p = 4 , lambda = 0 , MaxIter = 600                                                                                                                                                                    
#  p = 4;lambda=0;MaxIter= 600;
#  printf("|--> trying gradient descent (optimized) with p = %i , lambda = %f , MaxIter = %i \n",p,lambda,MaxIter);
#  tic();
#  [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
#  [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
#  [theta] = trainLinearReg(Xtrain, ytrain, lambda , MaxIter );
#  y_pred = predictLinearReg(Xval,theta);
#  y_train_pred =predictLinearReg(Xtrain,theta);
#  cost_val = MSE(y_pred, yval);
#  cost_train = MSE(y_train_pred, ytrain);
#  toc();
#  printf("MSE on training set = %f \n",cost_train);
#  printf("MSE on cross validation set = %f \n",cost_val);

#  ## comparing R lm solution
#  printf("|--> comparing with R(lm) solution ...\n"); 
#  R_y_pred = dlmread([curr_dir "/dataset/poly/poly_pure_ypred.zat"]);
#  R_cost_val = MSE(R_y_pred, yval);
#  printf("MSE on cross validation set = %f \n",R_cost_val);
#  printf("MSE on xval R(lm1) / normal equation(p=1,lambda=0) = %f \n",(R_cost_val/cost_val_ne1));
#  printf("MSE on xval R(lm1) / opt. gradient descent (p=1,lambda=0) = %f \n",(R_cost_val/cost_val_gd1));


#  if ( cost_val < 100 )  % put correctness tests here 
#    is_ok = 1;
#    printf("Test case passed.\n");
#  else 
#    is_ok = 0;    
#    error("Test case NOT passed.\n"); 
#  endif 

# endfunction

 function [is_ok] = var1_doBufferedUseCase()
  
  global curr_dir;

  is_ok = 0; % return as 1 if ok  
  p = 1;
  lambda = 0.1; 
  printf("Running var1_doBufferedUseCase ... \n"); 
 
  ## path 
  fiXtrain = "dataset/poly/poly_pure_Xtrain.zat";
  foXtrain = "dataset/poly/poly_pure_Xtrain_buff.zat";
  fiXval = "dataset/poly/poly_pure_Xval.zat";
  foXval = "dataset/poly/poly_pure_Xval_buff.zat"; 

  ## column index
  fytrain = "dataset/poly/poly_pure_ytrain.zat";
  fyval = "dataset/poly/poly_pure_yval.zat";                                                                                                                                                                 
  ciX = 0;
  ceX = 4;
  ciy = 0;
  cey = 0;

  ## reading datasets 
  _Xtrain = dlmread(fiXtrain);
  _Xval   = dlmread(fiXval);    
  ytrain  = dlmread(fytrain);
  yval    = dlmread(fyval);
 
  num_label = length(unique(ytrain));
  [m,n] = size(_Xtrain);
  
  ## p = 1 , lambda = 0                                                                                                                                                                                    
  p = 1;lambda=0;
  printf("|--> comparing performances of Buffered/Batch gradient descent (optimized) with p = %i and lambda = %f \n",p,lambda);
  tic();

  ## feature normalization 
  printf("|-> comparing Xtrain vs Xtrain_buff ... \n");
  [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
  [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
 
  [foXtrain,mu_b,sigma_b] = treatContFeatures_Buff(fiXtrain,foXtrain,p);
  [foXval,mu_b,sigma_b] = treatContFeatures_Buff(fiXval,foXval,p,1,mu_b,sigma_b);

  printf("|-> sum( (mu - mu_b) .^2) = %f \n" , sum( (mu - mu_b) .^2 ) );
  printf("|-> sum( (sigma - sigma_b) .^2) = %f \n" , sum( (sigma - sigma_b) .^2 ) ); 
  
  Xtrain_buff = dlmread(foXtrain); 
  Xval_buff = dlmread(foXval);

  diff = Xtrain - Xtrain_buff;
  disp(diff(1:5,:));
  printf("|-> sum diff squares:%f \n", sum(sum(diff .^ 2))  );
 
  ## training and predicting  
  printf("|-> comparing training and predicting ...  \n"); 
  NNMeta = buildNNMeta([n n num_label]);disp(NNMeta);
  
  [Theta_Buff] = trainNeuralNetwork_Buff(NNMeta, foXtrain,ciX,ceX,fytrain,ciy,cey,_sep=',',b=10000,lambda, ...
                     iter = 50 , featureScaled = 0 , initialTheta = cell(0,0) );
  pred_val_bf = NNPredictMulticlass_Buff(NNMeta,foXval,ciX,ceX,Theta_Buff,10000,',',0);
  pred_train_bf = NNPredictMulticlass_Buff(NNMeta,foXtrain,ciX,ceX,Theta_Buff,10000,',',0);
  ##(NNMeta,fX,ciX,ceX,Theta,b=10000,_sep=',',featureScaled = 0)
  cost_val_bf = MSE(pred_val_bf, yval);
  cost_train_bf = MSE(pred_train_bf, ytrain);
  [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda , iter = 100, ... 
      featureScaled = 1);
  pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
  pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
  cost_val = MSE(pred_val, yval);
  cost_train = MSE(pred_train, ytrain);
  toc();
  
  acc_train = mean(double(pred_train == ytrain)) * 100;
  acc_val = mean(double(pred_val == yval)) * 100;
  fprintf("Training Set Accuracy with feature normalization (p=%i,lambda=%f): %f\n", p,lambda,acc_train);
  fprintf("Cross Validation Set Accuracy with feature normalization (p=%i,lambda=%f): %f\n", p,lambda,acc_val);
 
  acc_train_buff = mean(double(pred_train_bf == ytrain)) * 100;
  acc_val_buff = mean(double(pred_val_bf == yval)) * 100;
  fprintf("Training Set Accuracy with feature normalization BUFFERED (p=%i,lambda=%f): %f\n", p,lambda,acc_train);
  fprintf("Cross Validation Set Accuracy with feature normalization BUFFERED (p=%i,lambda=%f): %f\n", p,lambda,acc_val);
  
  printf("|-> BATCH - MSE on training set = %f \n",cost_train);
  printf("|-> BATCH - MSE on cross validation set = %f \n",cost_val);

  printf("|-> BUFFERED - MSE on training set = %f  -   MSE(buffered_train) / MSE(batch_train) = %f  \n",cost_train_bf , (cost_train_bf / cost_train));
  printf("|-> BUFFERED - MSE on cross validation set = %f  -   MSE(buffered_val) / MSE(batch_val) = %f  \n",cost_val_bf , (cost_val_bf / cost_val) );

  ## cheking NNPredictMulticlass_Buff
  y_pred_mb_10 = NNPredictMulticlass_Buff(NNMeta,foXval,ciX,ceX,Theta_Buff,10,',',0);
  y_pred_mb_100 = NNPredictMulticlass_Buff(NNMeta,foXval,ciX,ceX,Theta_Buff,100,',',0);
  printf("|->  cheking predictLinearReg_Buff: (y_pred_mb_10/y_pred_mb_100) = %f   \n" , mean(y_pred_mb_10 ./ y_pred_mb_100) );
  
  
  ######### checking with other dataset #########################################
  load ('dataset/images/digits.mat'); %load X and y
  printf("\n|--> loading digits dataset and splitting into train set and cross validation set ...\n");
  [m,n] = size(X);
  rand_indices = randperm(m);
  [Xtrain,ytrain,Xval,yval] = splitTrainValidation(X(rand_indices,:),y(rand_indices),0.70);   
  [Xtrain,mu,sigma] = treatContFeatures(Xtrain,1);
  [Xval,mu_val,sigma_val] = treatContFeatures(Xval,1,1,mu,sigma);
  
  ## path 
  fiXtrain = "dataset/images/digits_Xtrain.zat";
  foXtrain = "dataset/images/digits_Xtrain_buff.zat";
  fiXval = "dataset/images/digits_Xval.zat";
  foXval = "dataset/images/digits_Xval_buff.zat"; 
  
  ## column index
  fytrain = "dataset/images/digits_ytrain.zat";
  fyval = "dataset/images/digits_yval.zat";                                                                                                                                                                 
  ciX = 0;
  ceX = n;
  ciy = 0;
  cey = 0;
  
  ## writing input files
  dlmwrite(fiXtrain,Xtrain);
  dlmwrite(fiXval,Xval);
  dlmwrite(fytrain,ytrain);
  dlmwrite(fyval,yval);
  
  ## reading datasets 
  _Xtrain = dlmread(fiXtrain);
  _Xval   = dlmread(fiXval);    
  ytrain  = dlmread(fytrain);
  yval    = dlmread(fyval);

  num_label = length(unique(ytrain));
  [m,n] = size(_Xtrain);
    
  ## p = 1 , lambda = 0                                                                                                                                                                                    
  p = 1;lambda=0;
  printf("|--> comparing performances of Buffered/Batch gradient descent (optimized) with p = %i and lambda = %f \n",p,lambda);
  tic();
   
  ## training and predicting  
  printf("|-> comparing training and predicting ...  \n"); 
  printf("|-> training buffering ... ");
  NNMeta = buildNNMeta([n-1 n-1 num_label]);disp(NNMeta);fflush(stdout);
    
  [Theta_Buff] = trainNeuralNetwork_Buff(NNMeta, fiXtrain,ciX,ceX,fytrain,ciy,cey,',',10000,lambda, ...
                       100 , 1 , cell(0,0) );
  pred_val_bf = NNPredictMulticlass_Buff(NNMeta,fiXval,ciX,ceX,Theta_Buff,10000,',',1);
  pred_train_bf = NNPredictMulticlass_Buff(NNMeta,fiXtrain,ciX,ceX,Theta_Buff,10000,',',1);
  ##(NNMeta,fX,ciX,ceX,Theta,b=10000,_sep=',',featureScaled = 0)
  cost_val_bf = MSE(pred_val_bf, yval);
  cost_train_bf = MSE(pred_train_bf, ytrain);
   
  printf("|-> training batch ... ");
  [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda , iter = 100, ... 
        featureScaled = 1);
  pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
  pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
  cost_val = MSE(pred_val, yval);
  cost_train = MSE(pred_train, ytrain);
   
  toc();
  acc_train = mean(double(pred_train == ytrain)) * 100;
  acc_val = mean(double(pred_val == yval)) * 100;
  acc_train_bf = mean(double(pred_train_bf == ytrain)) * 100;
  acc_val_bf = mean(double(pred_val_bf == yval)) * 100;
  fprintf("Training Set Accuracy with feature normalization (p=%i,lambda=%f): %f\n", p,lambda,acc_train);
  fprintf("Cross Validation Set Accuracy with feature normalization (p=%i,lambda=%f): %f\n", p,lambda,acc_val);  
  acc_train_buff = mean(double(pred_train_bf == ytrain)) * 100;
  acc_val_buff = mean(double(pred_val_bf == yval)) * 100;
  fprintf("Training Set Accuracy with feature normalization BUFFERED (p=%i,lambda=%f): %f\n", p,lambda,acc_train_bf);
  fprintf("Cross Validation Set Accuracy with feature normalization BUFFERED (p=%i,lambda=%f): %f\n", p,lambda,acc_val_bf);
  printf("|-> BATCH - MSE on training set = %f \n",cost_train);
  printf("|-> BATCH - MSE on cross validation set = %f \n",cost_val);
  
  printf("|-> BUFFERED - MSE on training set = %f  -   MSE(buffered_train) / MSE(batch_train) = %f  \n",cost_train_bf , (cost_train_bf / cost_train));
  printf("|-> BUFFERED - MSE on cross validation set = %f  -   MSE(buffered_val) / MSE(batch_val) = %f  \n",cost_val_bf , (cost_val_bf / cost_val) );
  
  ## cheking NNPredictMulticlass_Buff
  y_pred_mb_10 = NNPredictMulticlass_Buff(NNMeta,fiXval,ciX,ceX,Theta_Buff,10,',',1);
  y_pred_mb_100 = NNPredictMulticlass_Buff(NNMeta,fiXval,ciX,ceX,Theta_Buff,100,',',1);
  printf("|->  cheking predictLinearReg_Buff: (y_pred_mb_10/y_pred_mb_100) = %f   \n" , mean(y_pred_mb_10 ./ y_pred_mb_100) );
  
  ##buffering curve
  train_mb = 10:10:size(Xtrain,1);
  val_mb = 10:10:size(Xtrain,1);

  train_bf = 10:10:size(Xtrain,1);
  val_bf = 10:10:size(Xval,1);

  idx = 1;
#  for i = 10:10:size(Xtrain,1)
#   [theta_mb] = trainLinearReg_MiniBatch(foXtrain,ciX,ceX,fytrain,ciy,cey,lambda, b=i, _sep=',' , iter=200);
#   y_pred_mb = predictLinearReg_Buff(foXval,ciX,ceX,theta_mb,b=10000,_sep=',');
#   y_train_pred_mb = predictLinearReg_Buff(foXtrain,ciX,ceX,theta_mb,b=10000,_sep=',');
#   cost_val_mb = MSE(y_pred_mb, yval);
#   cost_train_mb = MSE(y_train_pred_mb, ytrain);

#   train_mb(idx) = cost_train_mb / cost_train;
#   val_mb(idx) = cost_val_mb / cost_val;

#   [theta_bf] = trainLinearReg_Buff(foXtrain,ciX,ceX,fytrain,ciy,cey,lambda, b=i, _sep=',' , iter=200);
#   y_pred_bf = predictLinearReg_Buff(foXval,ciX,ceX,theta_bf,b=10000,_sep=',');
#   y_train_pred_bf = predictLinearReg_Buff(foXtrain,ciX,ceX,theta_bf,b=10000,_sep=',');
#   cost_val_bf = MSE(y_pred_bf, yval);
#   cost_train_bf = MSE(y_train_pred_bf, ytrain);

#   train_bf(idx) = cost_train_bf / cost_train;
#   val_bf(idx) = cost_val_bf / cost_val;

#   idx += 1;
#  endfor

#  %%plot
#  max_X = size(Xtrain,1);
#  max_Y = 4;
#  ##min_Y = min(min(train_mb) , min(val_mb) );
#  plot(10:10:size(Xtrain,1), train_mb, 10:10:size(Xtrain,1), val_mb , 10:10:size(Xtrain,1), train_bf, 10:10:size(Xtrain,1), val_bf);
#  title(sprintf('Buffering Curve (lambda = %f, p = %i)', lambda,p))
#  xlabel('Buffer size')
#  ylabel('MSE ratio (vs BATCH)')
#  axis([0 max_X 0 max_Y])
#  legend('Mini-Batch Train', 'Mini-Batch Xval' , 'Buff Train', 'Buff Xval' )
#  pause;

#  disp("## val_mb(1:10) ##");disp(val_mb(1:10));
#  disp("## val_bf(1:10) ##");disp(val_bf(1:10));
 
 
#  ##buffering curve (time) 
#  time_mb = 10:10:size(Xtrain,1);
#  time_bf = 10:10:size(Xtrain,1);
#  time_b = 10:10:size(Xtrain,1);
 
#  _iter = 200,
 
#  idx = 1;
#  for i = 10:10:size(Xtrain,1)
#    tic();[theta_mb] = trainLinearReg_MiniBatch(foXtrain,ciX,ceX,fytrain,ciy,cey,lambda, b=i, _sep=',' , iter=_iter);t = toc();
#    time_mb(idx) = t;
 
#    tic(); [theta_bf] = trainLinearReg_Buff(foXtrain,ciX,ceX,fytrain,ciy,cey,lambda, b=i, _sep=',' , iter=_iter); t = toc(); 
#    time_bf(idx) = t;
   
#    tic();[theta] = trainLinearReg(Xtrain, ytrain, lambda , _iter ); t = toc();
#    time_b(idx) = t;
 
#    idx += 1; 
#  end
 
#  %%plot
#  max_X = size(Xtrain,1);
#  max_Y = max(max(time_mb) , max(max(time_mb) ,max(time_b) ) );
#  plot(10:10:size(Xtrain,1), time_mb, 10:10:size(Xtrain,1), time_bf, 10:10:size(Xtrain,1), time_b);
#  title(sprintf('Buffering Curve (lambda = %f, p = %i , iter = %i)', lambda,p,_iter))
#  xlabel('Buffer size')
#  ylabel('Training time')
#  axis([0 max_X 0 max_Y])
#  legend('Mini-Batch' , 'Buffered', 'Batch' )
#  pause;

#  if ( 1 )  % put correctness tests here
#    is_ok = 1;
#    printf("Test case passed.\n");
#  else 
#    is_ok = 0;             
#    error("Test case NOT passed.\n"); 
#  endif 

endfunction
