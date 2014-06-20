#! /opt/local/bin/octave -qf 

%%%% setting enviroment 
menv;

XtrainFile = "Xtrain.zat"; 
XtestFile = "Xtest.zat"; 
ytrainFile = "ytrain.zat";
sampleSub = "id.zat";

printf("|--> loading Xtrain, ytrain files ...\n");
X = dlmread([curr_dir "/dataset/acquire-valued-shoppers-challenge/" XtrainFile]); 
y = dlmread([curr_dir "/dataset/acquire-valued-shoppers-challenge/" ytrainFile]); 

X = X(1:20000,:);
y = y(1:20000,:);
y = (y ==0)*1 + (y==1)*2 ;

num_label = 2; 
lambda = 0;
p = 1;

[m,n] = size(X);
rand_indices = randperm(m);
[Xtrain,ytrain,Xval,yval] = splitTrainValidation(X(rand_indices,:),y(rand_indices,:),0.70); 
[Xtrain,mu,sigma] = treatContFeatures(Xtrain,p); 
[Xval,mu,sigma] = treatContFeatures(Xval,p,1,mu,sigma);



%%% 2) FINDING BEST PARAMS DEFAULT CLASSIFIER 
printf("|--> FINDING BEST PARAMS   ...\n");
#[n_opt,J_opt] = findOptNeuronsPerLayer(Xtrain, ytrain, Xval, yval , lambda=0 ,start_neurons=-1,end_neurons=-1,step_fw=-1,hidden_layers=1,_num_label=-1, verbose=0);
#[h_opt,J_opt] = findOptHiddenLayers(Xtrain, ytrain, Xval, yval , lambda=0,neurons_hidden_layers=n_opt,_num_label=-1, verbose=0);

NNMeta = buildNNMeta([n n num_label]);disp(NNMeta);
#[lambda_opt,J_opt] = findOptLambda(NNMeta, Xtrain, ytrain, Xval, yval , lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]');


#[Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, 0 , iter = 300, featureScaled = 1);
# pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
# pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
# acc_train = mean(double(pred_train == ytrain)) * 100;
# acc_val = mean(double(pred_val == yval)) * 100;
#printf("|-> acc_train=%f , acc_val=%f \n",acc_train,acc_val);

printf("|-> trying logistic regression ... \n");
[all_theta] = oneVsAll(Xtrain, ytrain, num_label, lambda);
 pred_train = predictOneVsAll(all_theta, Xtrain);
 pred_val = predictOneVsAll(all_theta, Xval);
 acc_train = mean(double(pred_train == ytrain)) * 100;
 acc_val = mean(double(pred_val == yval)) * 100;
 printf("|-> acc_train=%f , acc_val=%f \n",acc_train,acc_val);
 
 clear X;
 clear Xtrain;
 clear Xval; 
 Xtest = dlmread([curr_dir "/dataset/acquire-valued-shoppers-challenge/" XtestFile]); 
 [Xtest_n,mu,sigma] = treatContFeatures(Xtest,p,1,mu,sigma);
 pred_test = predictOneVsAll(all_theta, Xtest_n);
 
 pred_sub = (pred_test == 1) * 0 + (pred_test == 2) * 1;
 
 ids = dlmread([curr_dir "/dataset/acquire-valued-shoppers-challenge/" sampleSub]); 
 sub = [ids pred_sub];
 fn = "pred_ac.zat";
 dlmwrite(fn,sub);
 
