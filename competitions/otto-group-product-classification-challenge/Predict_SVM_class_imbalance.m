#! /opt/local/bin/octave -qf 

%%%% making enviroment 
menv;

printf("|--> loading data ...\n");
train_data = dlmread([curr_dir "/dataset/otto-group-product-classification-challenge/oct_train.csv"]); 
test_data = dlmread([curr_dir "/dataset/otto-group-product-classification-challenge/oct_test.csv"]); 

Y = dlmread([curr_dir "/dataset/otto-group-product-classification-challenge/oct_y.csv"]);

#[Xtrain,ytrain,Xval,yval] = splitTrainValidation(train_data,Y,0.8,shuffle=1);

printf("|--> feature scaling ...\n");
[train_data,mu,sigma] = treatContFeatures(train_data,1);
[test_data,mu_val,sigma_val] = treatContFeatures(test_data,1,1,mu,sigma);


## processing multiclass 
y_class = unique (Y); 
printf(">>>> found %i classes .. proceeding One vs. All ..  \n", length(y_class) );

probs_vect = zeros(size(test_data,1) , length(y_class) );

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

  ## gid                                                                                                                                                                             
  printf("|--> SVM ...\n");

  [C gamma, epsilon, nu] = getBestParamSVM(type = 1 , train_data_bal , ytrain_j_bal); ### errato ... devi usare xvalidation ... TODO 
  printf(">>  [label = %i] found analitically C=%f , gamma=%f , ... \n" , j , C, gamma(2) );
  
  fflush(stdout);

  ## training and prediction
  model = svmtrain( double(ytrain_j_bal) , double(train_data_bal), sprintf('-t 2 -g %g -c %g -b 1', gamma(2),C)  );

  [pred_j, accuracy_j, probs_j] = svmpredict( test_data(:,1) , test_data , model , '-b 1 ');
  
  probs_vect(:,j) = probs_j(:,1);
endfor 
tm = toc();

dlmwrite([curr_dir "/dataset/otto-group-product-classification-challenge/pred_SVM_class_balanced.csv"],probs_vect); 
