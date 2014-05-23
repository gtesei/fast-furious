#! /opt/local/bin/octave -qf 

%%%% setting enviroment 
menv;

XtrainFile = "Xtrain.zat"; 
ytrainFile = "ytrain.zat";

printf("|--> loading Xtrain, ytrain files ...\n");
X = dlmread([curr_dir "/dataset/acquire-valued-shoppers-challenge/" XtrainFile]); 
y = dlmread([curr_dir "/dataset/acquire-valued-shoppers-challenge/" ytrainFile]); 

[m,n] = size(X)
rand_indices = randperm(m);
[Xtrain,ytrain,Xval,yval] = splitTrainValidation(X(rand_indices,:),y(rand_indices,:),0.70); 

%%% 2) FINDING BEST PARAMS DEFAULT CLASSIFIER 
printf("|--> FINDING BEST PARAMS   ...\n");
[n_opt,J_opt] = findOptNeuronsPerLayer(Xtrain, ytrain, Xval, yval , lambda=0 ,start_neurons=-1,end_neurons=-1,step_fw=-1,hidden_layers=1,_num_label=-1, verbose=0);
[h_opt,J_opt] = findOptHiddenLayers(Xtrain, ytrain, Xval, yval , lambda=0,neurons_hidden_layers=n_opt,_num_label=-1, verbose=0);
NNMeta = buildNNMeta([(n - 1) repmat(n_opt,1,h_opt) num_label]);disp(NNMeta);
[lambda_opt,J_opt] = findOptLambda(NNMeta, Xtrain, ytrain, Xval, yval , lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]');
