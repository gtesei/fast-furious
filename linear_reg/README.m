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
  ok1 = var1_doBasicUseCase();
  ok2 = var1_doFindOptPAndLambdaUseCase();
  ok3 = var1_doComparisonPurePolyDatasetUseCase();
  is_ok = ok1 & ok2 & ok3;
endfunction 

function [is_ok] = var1_doBasicUseCase()
  
 is_ok = 0; % return as 1 if ok  
 p = 10;
 lambda = 0.1; 
 printf("Running var1_doBasicUseCase ... \n"); 

 load ('ex5data1.mat');
 
 printf("|--> splitting dataset into train set and cross validation set ...");
 [Xtrain,ytrain,Xval,yval] = splitTrainValidation(X,y,0.70);

 printf("|--> performing feature scaling and normalization on train dataset and cross validation dataset ...\n");
 [Xtrain,mu,sigma] = treatContFeatures(Xtrain,p);
 
 printf("|--> training ...");
 [theta] = trainLinearReg(Xtrain, ytrain, lambda , 200 );
 
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
 
 _Xtrain = dlmread([curr_dir "/dataset/poly/poly_pure_Xtrain.mat"]);
 ytrain =dlmread([curr_dir "/dataset/poly/poly_pure_ytrain.mat"]);
 _Xval =dlmread([curr_dir "/dataset/poly/poly_pure_Xval.mat"]);
 yval = dlmread([curr_dir "/dataset/poly/poly_pure_yval.mat"]);
 
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
 
 
 ## p = 5 , lambda = 1                                                                                                                                                                                    
 p = 5;lambda=1;
 printf("|--> trying with p = %i and lambda = %f \n",p,lambda);
 tic();
 [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
 [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
 [theta] = trainLinearReg(Xtrain, ytrain, lambda , 200 );
 y_pred = predictLinearReg(Xval,theta);
 cost_val = linearRegCostFunction(Xval, yval, theta, 0);
 cost_train = linearRegCostFunction(Xtrain, ytrain, theta, 0);
 toc();
 printf("RMSE on training set = %f \n",cost_train);
 printf("RMSE on cross validation set = %f \n",cost_val);


 ## p = 4 , lambda = 0
 p = 4;lambda=0;
 printf("|--> trying with p = %i and lambda = %f \n",p,lambda);
 tic();  
 [Xtrain,mu,sigma] = treatContFeatures(_Xtrain,p);
 [Xval,mu_val,sigma_val] = treatContFeatures(_Xval,p,1,mu,sigma);
 [theta] = trainLinearReg(Xtrain, ytrain, lambda , 200 );
 y_pred = predictLinearReg(Xval,theta);
 cost_val = linearRegCostFunction(Xval, yval, theta, 0); 
 cost_train = linearRegCostFunction(Xtrain, ytrain, theta, 0);
 toc(); 
 printf("RMSE on training set = %f \n",cost_train); 
 printf("RMSE on cross validation set = %f \n",cost_val);

 if ( cost_val < 100 )  % put correctness tests here 
   is_ok = 1;
   printf("Test case passed.\n");
 else 
   is_ok = 0;    
   error("Test case NOT passed.\n"); 
 endif 

endfunction





  