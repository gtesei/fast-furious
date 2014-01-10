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
function [is_ok] = var1_doBasicUseCase()
 clear ; close all; 
 is_ok = 0; % return as 1 if ok  
 p = 10;
 lambda = 0.1; 
 printf("Running var1_doBasicUseCase ... \n"); 

 load ('ex5data1.mat');
 
 [Xtrain,ytrain,Xval,yval] = splitTrainValidation(X,y,0.70);
 [Xtrain,mu,sigma] = treatContFeatures(Xtrain,p);
 [theta] = trainLinearReg(Xtrain, ytrain, lambda , 200 );
 
 [Xval,mu_val,sigma_val] = treatContFeatures(Xval,p,1,mu,sigma);
 if (mu != mu_val | sigma != sigma_val) 
   disp(mu);
   disp(mu_val);
   disp(sigma);
   disp(sigma_val);
   error("error in function treatContFeatures: mu != mu_val or sigma != sigma_val - displayed mu,mu_val,sigma,sigma_val .. ");
 endif  
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
 clear ; close all;
 is_ok = 0; % return as 1 if ok  
 
 printf("Running var1_doFindOptPAndLambdaUseCase ... \n"); 
 
 load ('ex5data1.mat');
 Xtrain = Xtest; ytrain = ytest;

 [Xtrain,mu,sigma] = treatContFeatures(Xtrain,1);
 [Xval,mu_val,sigma_val] = treatContFeatures(Xval,1,1,mu,sigma);
 if (mu != mu_val | sigma != sigma_val) 
    disp(mu);
    disp(mu_val);
    disp(sigma);
    disp(sigma_val);
    error("error in function treatContFeatures: mu != mu_val or sigma != sigma_val - displayed mu,mu_val,sigma,sigma_val .. ");
 endif
 
 [p_opt,J_opt] = findOptP_RegLin(Xtrain, ytrain, Xval, yval);
 
 [lambda_opt,J_opt] = findOptLambda_RegLin(Xtrain, ytrain, Xval, yval);
 
 if ( p_opt == 3 )  % put correctness tests here 
   is_ok = 1;
   printf("Test case passed.\n");
 else 
   is_ok = 0;    
   error("Test case NOT passed.\n"); 
 endif 

endfunction





  