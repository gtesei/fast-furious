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
%  Variant #1 / Use Cases 
%  ========================

function [is_ok] = go()
  %ok1 = var1_doBasicUseCase();
  %ok2 = var1_doFindOptPAndLambdaUseCase();
  %ok3 = var1_doComparisonPurePolyDatasetUseCase();
  %is_ok = ok1 & ok2 & ok3;
  is_ok = var1_doBufferedUseCase();
endfunction 

function [is_ok] = var1_doBasicUseCase()
  
 is_ok = 0; % return as 1 if ok  
 p = 10;
 lambda = 0.1; 
 printf("Running var1_doBasicUseCase ... \n"); 

 load ('ex5data1.mat');
 
 printf("|--> splitting dataset into train set and cross validation set ...\n");
 [Xtrain,ytrain,Xval,yval] = splitTrainValidation(X,y,0.70);

 printf("|--> performing feature scaling and normalization on train dataset and cross validation dataset ...\n");
 [Xtrain,mu,sigma] = treatContFeatures(Xtrain,p);
 
 printf("|--> training with gradient descent (optmized) ...\n");
 [theta] = trainLinearReg(Xtrain, ytrain, lambda , 200 );
 disp("theta - gradient descent");disp(theta);disp("-----------");
 printf("|--> finding solution with normal equation ...\n");
 [thetaNormEqn] = normalEqn_RegLin(Xtrain,ytrain,lambda); 
 disp("theta - normal equation");disp(thetaNormEqn);disp("-----------");
 disp("computing difference");disp(thetaNormEqn - theta);disp("-----------");

 [Xval,mu_val,sigma_val] = treatContFeatures(Xval,p,1,mu,sigma);
 if (mu != mu_val | sigma != sigma_val) 
   disp(mu); disp(mu_val); disp(sigma); disp(sigma_val);
   error("error in function treatContFeatures: mu != mu_val or sigma != sigma_val - displayed mu,mu_val,sigma,sigma_val .. ");
 endif  
 
 printf("|--> predicting with trained model on cross validation data set ...");
 y_pred = predictLinearReg(Xval,theta);
 cost = linearRegCostFunction(Xval, yval, theta, lambda); 

 if ( cost < 5.5 )  % put correctness tests here 
   is_ok = 1;
   printf("Test case passed.\n");
 else 
   is_ok = 0;             
   printf("Cost; %f\n",cost);
   error("Test case NOT passed.\n"); 
 endif 

endfunction

function [is_ok] = var1_doFindOptPAndLambdaUseCase()
 
 is_ok = 0; % return as 1 if ok  
 
 printf("Running var1_doFindOptPAndLambdaUseCase ... \n"); 
 
 load ('ex5data1.mat');
 Xtrain = Xtest; ytrain = ytest;

 printf("|--> performing feature scaling and normalization on train dataset and cross validation dataset ...\n");
 [Xtrain,mu,sigma] = treatContFeatures(Xtrain,1);
 [Xval,mu_val,sigma_val] = treatContFeatures(Xval,1,1,mu,sigma);
 if (mu != mu_val | sigma != sigma_val) 
    disp(mu); disp(mu_val); disp(sigma); disp(sigma_val);
    error("error in function treatContFeatures: mu != mu_val or sigma != sigma_val - displayed mu,mu_val,sigma,sigma_val .. ");
 endif
 
 printf("|--> finding optimal polinomial degree ... \n");
 tic(); [p_opt,J_opt] = findOptP_RegLin(Xtrain, ytrain, Xval, yval); toc();
 pause;
 
 printf("|--> finding optimal regularization parameter ... \n");
 tic(); [lambda_opt,J_opt] = findOptLambda_RegLin(Xtrain, ytrain, Xval, yval); toc();
 pause;
 
 printf("|--> computing learning curve ... \n");
  tic(); [error_train,error_val] = learningCurve_RegLin(Xtrain, ytrain, Xval, yval); toc();
 pause;
 
 if ( p_opt == 3 )  % put correctness tests here 
   is_ok = 1;
   printf("Test case passed.\n");
 else 
   is_ok = 0;    
   error("Test case NOT passed.\n"); 
 endif 

endfunction

function [is_ok] = var1_doComparisonPurePolyDatasetUseCase()
 global curr_dir; 

 is_ok = 0; % return as 1 if ok  
 
 printf("Running var1_doComparisonPurePolyDatasetUseCase ... \n"); 
 
 _Xtrain = dlmread([curr_dir "/dataset/poly/poly_pure_Xtrain.zat"]);
 ytrain =dlmread([curr_dir "/dataset/poly/poly_pure_ytrain.zat"]);
 _Xval =dlmread([curr_dir "/dataset/poly/poly_pure_Xval.zat"]);
 yval = dlmread([curr_dir "/dataset/poly/poly_pure_yval.zat"]);
 
 printf("|--> performing feature scaling and normalization on train dataset and cross validation dataset ...\n");
 [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,1);
 [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,1,1,mu,sigma);
 if (mu != mu_val | sigma != sigma_val) 
    disp(mu); disp(mu_val); disp(sigma); disp(sigma_val);
    error("error in function treatContFeatures: mu != mu_val or sigma != sigma_val - displayed mu,mu_val,sigma,sigma_val .. ");
 endif
 
 printf("|--> finding optimal polinomial degree ... \n");
 tic(); [p_opt,J_opt] = findOptP_RegLin(Xtrain, ytrain, Xval, yval); toc();
 pause;
 
 printf("|--> finding optimal regularization parameter ... \n");
 tic(); [lambda_opt,J_opt] = findOptLambda_RegLin(Xtrain, ytrain, Xval, yval); toc();
 pause;
 
 printf("|--> computing learning curve ... \n");
  tic(); [error_train,error_val] = learningCurve_RegLin(Xtrain, ytrain, Xval, yval); toc();
 pause;
 
 ##normal equation p = 1, lambda = 0
 p = 1; lambda = 0;
 printf("|--> finding optimal solution with normal equation p = %i and lambda = %f \n",p,lambda);
 tic();
 [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
 [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
 [theta] = normalEqn_RegLin(Xtrain,ytrain,lambda);
 y_pred = predictLinearReg(Xval,theta);
 y_train_pred = predictLinearReg(Xtrain,theta);
 cost_val_ne1 = MSE(y_pred, yval);
 cost_train = MSE(y_train_pred, ytrain);
 toc();
 printf("MSE on training set = %f \n",cost_train);
 printf("MSE on cross validation set = %f \n",cost_val_ne1);

 ## p = 1 , lambda = 0                                                                                                                                                                   
 p = 1;lambda=0;
 printf("|--> trying gradient descent (optimized) with p = %i and lambda = %f \n",p,lambda);
 tic();
 [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
 [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
 [theta] = trainLinearReg(Xtrain, ytrain, lambda , 200 );
 y_pred = predictLinearReg(Xval,theta);
 y_train_pred =predictLinearReg(Xtrain,theta);
 cost_val_gd1 = MSE(y_pred, yval);
 cost_train_gd1 = MSE(y_train_pred, ytrain);
 toc();
 printf("MSE on training set = %f \n",cost_train_gd1);
 printf("MSE on cross validation set = %f \n",cost_val_gd1);

 ##normal equation p = 4, lambda = 0                                                                                                                                                          
 p = 4; lambda = 0;
 printf("|--> finding optimal solution with normal equation p = %i and lambda = %f \n",p,lambda);
 tic();
 [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
 [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
 [theta] = normalEqn_RegLin(Xtrain,ytrain,lambda);
 y_pred = predictLinearReg(Xval,theta);
 y_train_pred =predictLinearReg(Xtrain,theta);
 cost_val = MSE(y_pred, yval);
 cost_train = MSE(y_train_pred, ytrain); 
 toc();
 printf("MSE on training set = %f \n",cost_train);
 printf("MSE on cross validation set = %f \n",cost_val);

 ## p = 5 , lambda = 1                                                                                                                                                                                    
 p = 5;lambda=1;
 printf("|--> trying gradient descent (optimized) with p = %i and lambda = %f \n",p,lambda);
 tic();
 [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
 [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
 [theta] = trainLinearReg(Xtrain, ytrain, lambda , 200 );
 y_pred = predictLinearReg(Xval,theta);
 y_train_pred =predictLinearReg(Xtrain,theta);
 cost_val = MSE(y_pred, yval);
 cost_train = MSE(y_train_pred, ytrain);
 toc();
 printf("MSE on training set = %f \n",cost_train);
 printf("MSE on cross validation set = %f \n",cost_val);


 ## p = 4 , lambda = 0 , MaxIter = 200
 p = 4;lambda=0;MaxIter = 200;
 printf("|--> trying gradient descent (optimized) with p = %i , lambda = %f , MaxIter = %i \n",p,lambda,MaxIter);
 tic();  
 [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
 [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
 [theta] = trainLinearReg(Xtrain, ytrain, lambda , MaxIter );
 y_pred = predictLinearReg(Xval,theta);
 y_train_pred =predictLinearReg(Xtrain,theta);
 cost_val = MSE(y_pred, yval);
 cost_train = MSE(y_train_pred, ytrain);
 toc(); 
 printf("MSE on training set = %f \n",cost_train); 
 printf("MSE on cross validation set = %f \n",cost_val);

 ## p = 4 , lambda = 0 , MaxIter = 400                                                                                                                                                                   
 p = 4;lambda=0;MaxIter= 400;
 printf("|--> trying gradient descent (optimized) with p = %i , lambda = %f , MaxIter = %i \n",p,lambda,MaxIter);
 tic();
 [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
 [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
 [theta] = trainLinearReg(Xtrain, ytrain, lambda , MaxIter );
 y_pred = predictLinearReg(Xval,theta);
 y_train_pred =predictLinearReg(Xtrain,theta);
 cost_val = MSE(y_pred, yval);
 cost_train = MSE(y_train_pred, ytrain);
 toc();
 printf("MSE on training set = %f \n",cost_train);
 printf("MSE on cross validation set = %f \n",cost_val);

 ## p = 4 , lambda = 0 , MaxIter = 600                                                                                                                                                                    
 p = 4;lambda=0;MaxIter= 600;
 printf("|--> trying gradient descent (optimized) with p = %i , lambda = %f , MaxIter = %i \n",p,lambda,MaxIter);
 tic();
 [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
 [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
 [theta] = trainLinearReg(Xtrain, ytrain, lambda , MaxIter );
 y_pred = predictLinearReg(Xval,theta);
 y_train_pred =predictLinearReg(Xtrain,theta);
 cost_val = MSE(y_pred, yval);
 cost_train = MSE(y_train_pred, ytrain);
 toc();
 printf("MSE on training set = %f \n",cost_train);
 printf("MSE on cross validation set = %f \n",cost_val);

 ## comparing R lm solution
 printf("|--> comparing with R(lm) solution ...\n"); 
 R_y_pred = dlmread([curr_dir "/dataset/poly/poly_pure_ypred.zat"]);
 R_cost_val = MSE(R_y_pred, yval);
 printf("MSE on cross validation set = %f \n",R_cost_val);
 printf("MSE on xval R(lm1) / normal equation(p=1,lambda=0) = %f \n",(R_cost_val/cost_val_ne1));
 printf("MSE on xval R(lm1) / opt. gradient descent (p=1,lambda=0) = %f \n",(R_cost_val/cost_val_gd1));


 if ( cost_val < 100 )  % put correctness tests here 
   is_ok = 1;
   printf("Test case passed.\n");
 else 
   is_ok = 0;    
   error("Test case NOT passed.\n"); 
 endif 

endfunction

function [is_ok] = var1_doBufferedUseCase()
  
 is_ok = 0; % return as 1 if ok  
 p = 10;
 lambda = 0.1; 
 printf("Running var1_doBufferedUseCase ... \n"); 

 _Xtrain = dlmread([curr_dir "/dataset/poly/poly_pure_Xtrain.zat"]);
 ytrain =dlmread([curr_dir "/dataset/poly/poly_pure_ytrain.zat"]);
 
 fX = "/dataset/poly/poly_pure_Xtrain.zat";
 fy = "/dataset/poly/poly_pure_ytrain.zat";
 ciX = 0;
 ceX = 4;
 ciy = 0;
 cey = 0;
 
 TODO  

 if ( cost < 5.5 )  % put correctness tests here 
   is_ok = 1;
   printf("Test case passed.\n");
 else 
   is_ok = 0;             
   printf("Cost; %f\n",cost);
   error("Test case NOT passed.\n"); 
 endif 

endfunction




  