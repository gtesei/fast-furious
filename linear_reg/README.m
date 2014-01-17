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
  
 global curr_dir;

 is_ok = 0; % return as 1 if ok  
 p = 10;
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
 
 ## p = 1 , lambda = 0                                                                                                                                                                                    
 p = 1;lambda=0;
 printf("|--> comparing performances of Mini-Batch/Buffered/Batch gradient descent (optimized) with p = %i and lambda = %f \n",p,lambda);
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
 
 [theta] = trainLinearReg(Xtrain, ytrain, lambda , 200 );
 y_pred = predictLinearReg(Xval,theta);
 y_train_pred = predictLinearReg(Xtrain,theta);
 cost_val = MSE(y_pred, yval);
 cost_train = MSE(y_train_pred, ytrain);

 [theta_mb] = trainLinearReg_MiniBatch(foXtrain,ciX,ceX,fytrain,ciy,cey,lambda, b=100, _sep=',' , iter=200);
 y_pred_mb = predictLinearReg_Buff(foXval,ciX,ceX,theta_mb,b=10000,_sep=',');
 y_train_pred_mb = predictLinearReg_Buff(foXtrain,ciX,ceX,theta_mb,b=10000,_sep=',');
 cost_val_mb = MSE(y_pred_mb, yval);
 cost_train_mb = MSE(y_train_pred_mb, ytrain);

 [theta_bf] = trainLinearReg_Buff(foXtrain,ciX,ceX,fytrain,ciy,cey,lambda, b=100, _sep=',' , iter=200);
 y_pred_bf = predictLinearReg_Buff(foXval,ciX,ceX,theta_bf,b=10000,_sep=',');
 y_train_pred_bf = predictLinearReg_Buff(foXtrain,ciX,ceX,theta_bf,b=10000,_sep=',');
 cost_val_bf = MSE(y_pred_bf, yval);
 cost_train_bf = MSE(y_train_pred_bf, ytrain);

 toc();
 printf("|-> BATCH - MSE on training set = %f \n",cost_train);
 printf("|-> BATCH - MSE on cross validation set = %f \n",cost_val);

 printf("|-> MINI-BATCH - MSE on training set = %f   -   MSE(mini-batch_train) / MSE(batch_train) = %f   \n" , cost_train_mb , (cost_train_mb / cost_train) );
 printf("|-> MINI-BATCH - MSE on cross validation set = %f  -   MSE(mini-batch_val) / MSE(batch_val) = %f  \n", cost_val_mb , (cost_val_mb / cost_val) );

 printf("|-> BUFFERED - MSE on training set = %f  -   MSE(buffered_train) / MSE(batch_train) = %f  \n",cost_train_bf , (cost_train_bf / cost_train));
 printf("|-> BUFFERED - MSE on cross validation set = %f  -   MSE(buffered_val) / MSE(batch_val) = %f  \n",cost_val_bf , (cost_val_bf / cost_val) );

 ## cheking predictLinearReg_Buff
 y_pred_mb_10 = predictLinearReg_Buff(foXval,ciX,ceX,theta_mb,b=10,_sep=',');
 y_pred_mb_100 = predictLinearReg_Buff(foXval,ciX,ceX,theta_mb,b=100,_sep=',');
 printf("|->  cheking predictLinearReg_Buff: (y_pred_mb_10/y_pred_mb_100) = %f   \n" , mean(y_pred_mb_10 ./ y_pred_mb_100) );

 ##buffering curve
 train_mb = 1:10:size(Xtrain,1);
 val_mb = 1:10:size(Xtrain,1);

 train_bf = 1:10:size(Xtrain,1);
 val_bf = 1:10:size(Xtrain,1);

 idx = 1;
 for i = 1:10:size(Xtrain,1)
  [theta_mb] = trainLinearReg_MiniBatch(foXtrain,ciX,ceX,fytrain,ciy,cey,lambda, b=i, _sep=',' , iter=200);
  y_pred_mb = predictLinearReg_Buff(foXval,ciX,ceX,theta_mb,b=10000,_sep=',');
  y_train_pred_mb = predictLinearReg_Buff(foXtrain,ciX,ceX,theta_mb,b=10000,_sep=',');
  cost_val_mb = MSE(y_pred_mb, yval);
  cost_train_mb = MSE(y_train_pred_mb, ytrain);

  train_mb(idx) = cost_train_mb / cost_train;
  val_mb(idx) = cost_val_mb / cost_val;

  [theta_bf] = trainLinearReg_Buff(foXtrain,ciX,ceX,fytrain,ciy,cey,lambda, b=i, _sep=',' , iter=200);
  y_pred_bf = predictLinearReg_Buff(foXval,ciX,ceX,theta_bf,b=10000,_sep=',');
  y_train_pred_bf = predictLinearReg_Buff(foXtrain,ciX,ceX,theta_bf,b=10000,_sep=',');
  cost_val_bf = MSE(y_pred_bf, yval);
  cost_train_bf = MSE(y_train_pred_bf, ytrain);

  train_bf(idx) = cost_train_bf / cost_train;
  val_bf(idx) = cost_val_bf / cost_val;

  idx += 1;
end

 %%plot
 max_X = size(Xtrain,1);
 max_Y = 4;
 ##min_Y = min(min(train_mb) , min(val_mb) );
 plot(1:10:size(Xtrain,1), train_mb, 1:10:size(Xtrain,1), val_mb , 1:10:size(Xtrain,1), train_bf, 1:10:size(Xtrain,1), val_bf);
 title(sprintf('Buffering Curve (lambda = %f, p = %i)', lambda,p))
 xlabel('Buffer size')
 ylabel('MSE ratio (vs BATCH)')
 axis([0 max_X 0 max_Y])
 legend('Mini-Batch Train', 'Mini-Batch Xval' , 'Buff Train', 'Buff Xval' )
 pause;

 disp("## val_mb(1:10) ##");disp(val_mb(1:10));
 disp("## val_bf(1:10) ##");disp(val_bf(1:10));

 if ( cost_val < 5.5 )  % put correctness tests here
   is_ok = 1;
   printf("Test case passed.\n");
 else 
   is_ok = 0;             
   error("Test case NOT passed.\n"); 
 endif 

endfunction




  