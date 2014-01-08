%% Fast-Furious 
%  Linear Regression 
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
function [is_ok] = var1_doUseCaseBasic()
 
 is_ok = 0; % return as 1 if ok  
 p = 10;
 lambda = 0.1; 
 
 load ('ex5data1.mat');
 
 X = treatContFeatures(X,p);
 [Xtrain,ytrain,Xval,yval] = splitTrainValidation(X,y,0.70);
 [theta] = trainLinearReg(Xtrain, ytrain, lambda , 200 );
 
 y_pred = predictLinearReg(Xval,theta);
 cost = linearRegCostFunction(Xval, yval, theta, lambda);
 
 if ( cost < 5.5 )  % put correctness tests here 
   is_ok = 1;
 endif 

endfunction





  