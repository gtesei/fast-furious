#! /opt/local/bin/octave -qf 

%%%% setting enviroment 
menv;

trainFile = "train_nn.csv";
testFile = "test_nn.csv";

#trainFile = "train_reg_pca.csv";
#testFile = "test_reg_pca.csv";

%trainFile = "train_nn_no_skewness.csv";
%testFile = "test_nn_no_skewness.csv";

printf("|--> loading Xtrain, ytrain files ...\n");
train = dlmread([curr_dir "/dataset/liberty-mutual-fire-peril/" trainFile]); 
X_test = dlmread([curr_dir "/dataset/liberty-mutual-fire-peril/" testFile]); 

train = train(2:end,:); ## elimina le intestazioni del csv 
X_test = X_test(2:end,:); ## elimina le intestazioni del csv

y = train(:,1); ## la prima colonna e' target mentra l'ultima e' tragetPos che va scartata ...
#y = train(:,end); ##for classification problem
X = train(:,2:(end-1));

idTest = X_test(:,end); ## l'ultima colonna e' id
X_test = X_test(:,1:(end-1));

clear train;


### cv ...
perc_train = 0.8;
[m,n] = size(X);
rand_indices = randperm(m);
[_Xtrain,ytrain,_Xval,yval] = splitTrainValidation(X(rand_indices,:),y(rand_indices,:),perc_train);

### feature scaling and cenering ...
[Xtrain,mu,sigma] = treatContFeatures(_Xtrain,1);
[Xval,mu,sigma] = treatContFeatures(_Xval,1,1,mu,sigma);

####### Linear Regression
p_opt = 5;
lambda_opt = 0.1;
p_opt = 7; ## no skew
lambda_opt = 0.001; ## no skew

printf("|--> finding optimal polinomial degree ... \n");
%tic(); [p_opt,J_opt] = findOptP_RegLin(Xtrain, ytrain, Xval, yval , p_vec = [1 2 3 4 5 6 7 8 9 10]'); toc();
tic(); [p_opt,J_opt] = findOptP_RegLinLiberty4NormReg(Xtrain, ytrain, Xval, yval, p_vec = [1 2 6 8 9 10 12 15 20]' , lambda=0 , _Xtrain(:,43),_Xval(:,43)); toc();

                                                      
printf("|--> finding optimal regularization parameter ... \n");
%tic(); [lambda_opt,J_opt] = findOptLambda_RegLinLiberty(_Xtrain, ytrain, _Xval, yval , lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' , p=1); toc();
tic(); [lambda_opt,J_opt] = findOptLambda_RegLinLiberty4NormReg(_Xtrain, ytrain, _Xval, yval , lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' , p=p_opt , _Xtrain(:,43),_Xval(:,43)); toc();

                                                 
## predicting setting p = p_opt and lambda = lambda_opt
lambda_liberty = 1e11;
[X_poly_train,mu,sigma] = treatContFeatures(Xtrain,p_opt);
[X_poly_val,mu_val,sigma_val] = treatContFeatures(Xval,p_opt,1,mu,sigma);
                                                 
[thetaIn] = trainLinearReg(X_poly_train, ytrain, lambda_opt , 300 );
#[theta] = trainLinearRegLiberty(X_poly_train, ytrain, lambda_liberty , 8 , _theta= [] , var11 = _Xtrain(:,43));
[theta] = trainLinearRegLiberty(X_poly_train, ytrain, lambda_liberty , 8 , _theta= [] , var11 = _Xtrain(:,1) , simulatedAnealing = 1);
pred_val = predictLinearReg(X_poly_val,theta);
pred_train =predictLinearReg(X_poly_train,theta);
cost_val_gd1 = MSE(pred_val, yval);
cost_train_gd1 = MSE(pred_train, ytrain);
printf("MSE on training set = %f \n",cost_train_gd1);
printf("MSE on cross validation set = %f \n",cost_val_gd1);
                                                 
#gini_train = NormalizedWeightedGini(ytrain,_Xtrain(:,1),pred_train);
gini_train = NormalizedWeightedGini(ytrain,_Xtrain(:,1),pred_train);
printf("LR - NormalizedWeightedGini on train = %f \n", gini_train );
                                                 
#gini_xval = NormalizedWeightedGini(yval,_Xval(:,43),pred_val);
gini_xval = NormalizedWeightedGini(yval,_Xval(:,1),pred_val);
printf("LR - NormalizedWeightedGini on cv = %f \n", gini_xval );
                                                        
                                                 
####### Neural Networks 
[m,n] = size(Xtrain);
num_label = 1;
NNMeta = buildNNMeta([(n-1) (n-1) num_label]);disp(NNMeta);

[Theta] = trainNeuralNetworkReg(NNMeta, Xtrain, ytrain, 0 , iter = 300, featureScaled = 1 );

pred_train = NNPredictReg(NNMeta, Theta , Xtrain , featureScaled = 1);
pred_val = NNPredictReg(NNMeta, Theta , Xval , featureScaled = 1);
cost_val_gd1 = MSE(pred_val, yval);
cost_train_gd1 = MSE(pred_train, ytrain);
                                                 
printf("NN - MSE on training set = %f \n",cost_train_gd1);
printf("NN - MSE on cross validation set = %f \n",cost_val_gd1);
                                                 
gini_train = NormalizedWeightedGini(ytrain,_Xtrain(:,2),pred_train);
printf("NN - NormalizedWeightedGini on train = %f \n", gini_train );
                                                 
gini_xval = NormalizedWeightedGini(yval,_Xval(:,2),pred_val);
printf("NN - NormalizedWeightedGini on cv = %f \n", gini_xval );
                                                 
### NN EGS
num_label = 1;
NNMeta = buildNNMeta([(n-1) (n-1)  num_label]);disp(NNMeta);
[Theta] = trainNeuralNetworkRegEGS(NNMeta, Xtrain, ytrain, 0 , iter = 300, featureScaled = 1 );

pred_train = NNPredictRegEGS(NNMeta, Theta , Xtrain , featureScaled = 1);
pred_val = NNPredictRegEGS(NNMeta, Theta , Xval , featureScaled = 1);
cost_val_gd1 = MSE(pred_val, yval);
cost_train_gd1 = MSE(pred_train, ytrain);

printf("NN - MSE on training set = %f \n",cost_train_gd1);
printf("NN - MSE on cross validation set = %f \n",cost_val_gd1);

gini_train = NormalizedWeightedGini(ytrain,_Xtrain(:,43),pred_train);
printf("NN - NormalizedWeightedGini on train = %f \n", gini_train );

gini_xval = NormalizedWeightedGini(yval,_Xval(:,43),pred_val);
printf("NN - NormalizedWeightedGini on cv = %f \n", gini_xval );
                                                                
                                                                
##### SVM 
model = svmtrain(ytrain, Xtrain, '-t 2 -c 100 -g 1');
[pred_train, accuracy, dec_values] = svmpredict(ytrain, Xtrain, model);
[pred_val, accuracy, dec_values] = svmpredict(yval, Xval, model);
printf("|--> *** TRAIN STATS ***\n");
printClassMetrics(pred_train,ytrain);
printf("|--> *** CV STATS ***\n");
printClassMetrics(pred_val,yval);

                                                                
                                                                
####### Predicting Linear Regression
lambda_liberty = 1e14;
[X_poly,mu,sigma] = treatContFeatures(X,p_opt);
[X_poly_test,mu,sigma] = treatContFeatures(X_test,p_opt,1,mu,sigma);

[thetaIn] = trainLinearReg(X_poly, y, lambda_opt , 900 );
                                                                
#tic(); [theta] = trainLinearRegLiberty(X_poly, y, lambda_liberty , 8 , _theta= [] , var11 = X(:,43)); toc();
tic(); [theta] = trainLinearRegLiberty(X_poly, y, lambda_liberty , 10 , _theta= [] , var11 = X(:,1) , simulatedAnealing = 1); toc();


## pred
pred = predictLinearReg(X_poly_test,theta);

sub = [idTest pred];
dlmwrite ([curr_dir "/dataset/liberty-mutual-fire-peril/pred_no_skew.zat"] , sub);
dlmwrite ([curr_dir "/dataset/liberty-mutual-fire-peril/pred_normal.zat"] , sub);





