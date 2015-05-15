#! /opt/local/bin/octave -qf 

%%%% making enviroment 
menv;

printf("|--> loading data ...\n");
train_data = dlmread([curr_dir "/dataset/otto-group-product-classification-challenge/oct_train_reduced_xg.csv"]); 
test_data = dlmread([curr_dir "/dataset/otto-group-product-classification-challenge/oct_test_reduced_xg.csv"]); 
Y = dlmread([curr_dir "/dataset/otto-group-product-classification-challenge/oct_y.csv"]);

printf("|--> feature scaling ...\n");
[train_data,mu,sigma] = treatContFeatures(train_data,1);
[test_data,mu_val,sigma_val] = treatContFeatures(test_data,1,1,mu,sigma);


## processing multiclass 
y_class = unique (Y); 
printf(">>>> found %i classes .. proceeding One vs. All ..  \n", length(y_class) );

## probability vector 
probs_vect = zeros(size(test_data,1) , length(y_class) );

## best params grid
grid = zeros(length(y_class),5); 

tic();
for j = 1:length(y_class) 
  printf(">>> processing Class%i  ..  \n", j );
  ytrain_j = ( Y == j  );

  ## handling class imbalance 
  ytrain_j_size = sum(ytrain_j == 1);
  printf(">>> Class%i occurs %i times on trainset of %i [%f] ..  \n", j , ytrain_j_size , length(ytrain_j) , (ytrain_j_size/length(ytrain_j)) );
  idx_1 = find( ytrain_j == 1 );
  idx_0 = find( ytrain_j == 0 );
  rand_ind = randperm(length(idx_0));
  idx_0_bal = idx_0(rand_ind);
  idx_0_bal = idx_0_bal(1:length(idx_1)); 
  idx = [idx_1 ; idx_0_bal ]; 
  train_data_bal = train_data(idx,:);
  ytrain_j_bal = ytrain_j(idx);

  [Xtrain,ytrain,Xval,yval] = splitTrainValidation(train_data_bal,ytrain_j_bal,0.80,shuffle=1);


  ## find best pars                                                                                                                                                               
  printf("|--> NN finding best parameters (p,h,lambda) ...\n");
  fflush(stdout);
  
  [p_opt_acc,h_opt_acc,lambda_opt_acc,acc_opt,grid_j] = findOptPAndHAndLambda(Xtrain, ytrain, Xval, yval, ...
									    featureScaled = 1 , scaleFeatures = 0 , ...
									    p_vec = [] , ...
									    h_vec = [1 2 3] , ...
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
  dlmwrite('NN__main_grid_reduced.mat',grid);  

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

dlmwrite([curr_dir "/dataset/otto-group-product-classification-challenge/pred_NN_reduced.csv"],probs_vect); 
