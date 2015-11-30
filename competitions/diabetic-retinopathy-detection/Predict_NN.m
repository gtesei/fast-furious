#! /opt/local/bin/octave -qf 

%%%% making enviroment 
menv;

printf("|--> loading data ...\n");
%train_data = dlmread([curr_dir "/dataset/diabetic-retinopathy-detection/img_2000.csv"]); 
train_data = dlmread([curr_dir "/dataset/diabetic-retinopathy-detection/img_2000.csv"]); 

train = train_data(:,2:end);
y_train = train_data(:,1);
y_train = y_train + ones(length(y_train),1); %% fast-furious NN 1-based 

[Xtrain,ytrain,Xval,yval] = splitTrainValidation(train,y_train,0.80,shuffle=1);

%% to remove 
NNMeta = buildNNMeta([(size(train,2)) (ones(2,1) .* size(train,2))' 5]');disp(NNMeta);
[Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, 0.0001 , iter = 200, featureScaled = 0 );
pred = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 0);
acc = mean(double(pred == yval)) * 100;
printf(">>> Accuracy on train set: %f ..  \n", acc );
printf(">>> Accuracy all 0s : %f ..  \n", mean(double(ones(length(pred),1) == yval)) * 100 );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = size(Xval, 1);
p = zeros(size(Xval, 1), 1);

L = length(NNMeta.NNArchVect);
h = cell(L-1,1);

for i = 1:L-1
  if (featureScaled & i == 1)
    h(i,1) = sigmoid(Xval * cell2mat(Theta(i,1))');
  elseif (i == 1)
    h(i,1) = sigmoid([ones(m, 1) Xval] * cell2mat(Theta(i,1))');
  else
    h(i,1) = sigmoid([ones(m, 1) cell2mat(h(i-1,1))] * cell2mat(Theta(i,1))');
  endif
endfor

[ mm , idx] = max(hx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%find best pars                                                                                                                                                               
printf("|--> NN finding best parameters (p,h,lambda) ...\n");
fflush(stdout);
  
[p_opt_acc,h_opt_acc,lambda_opt_acc,acc_opt,grid_j] = findOptPAndHAndLambda(Xtrain, ytrain, Xval, yval, ...
									    featureScaled = 0 , scaleFeatures = 0 , ...
									    p_vec = [] , ...
									    h_vec = [1 2 3 4] , ...
									    lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10] , ...
									    #lambda_vec = [0 0.001] , ... 
									    verbose = 1, doPlot=1 , ...
									    initGrid = [] , initStart = -1 , ...   
									    iter = 200 , ...
									    #iter = 2 , ...
									    regression = 0 , num_labels = 1 );
  
  grid(j,1) = j;
  grid(j,2) = p_opt_acc;
  grid(j,3) = h_opt_acc;
  grid(j,4) = lambda_opt_acc;
  grid(j,5) = acc_opt;
  dlmwrite('NN__main_grid.mat',grid);  

  ## training on full train set 
  NNMeta = buildNNMeta([(size(train_data_bal,2)-1) (ones(h_opt_acc,1) .* p_opt_acc)' 1]');disp(NNMeta);
  printf("|--> NN training on the full train set with best parameters found (p=%f, h=%f, lambda=%f, accuracy_xval=%f) ...\n", p_opt_acc,h_opt_acc,lambda_opt_acc,acc_opt);
  [Theta] = trainNeuralNetwork(NNMeta, train_data_bal, ytrain_j_bal, lambda_opt_acc , iter = 2000, featureScaled = 1);
  
  ## predicting on train set 
  probs_train = NNPredictMulticlass(NNMeta, Theta , train_data_bal , featureScaled = 1);
  pred_train = (probs_train > 0.5);
  acc_train = mean(double(pred_train == ytrain_j_bal)) * 100;
  printf(">>> Accuracy on train set: %f ..  \n", acc_train );

  ## predicting on test set 
  printf("|--> predicting on test set ...\n");
  probs_test = NNPredictMulticlass(NNMeta, Theta , test_data , featureScaled = 1); 
  probs_vect(:,j) = probs_test;

endfor 
tm = toc();

dlmwrite([curr_dir "/dataset/otto-group-product-classification-challenge/pred_NN.csv"],probs_vect); 
