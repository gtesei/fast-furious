#! /opt/local/bin/octave -qf 

%%%% making enviroment 
menv;

printf("|--> loading data ...\n");
train_data = dlmread([curr_dir "/dataset/otto-group-product-classification-challenge/oct_train.csv"]); 
test_data = dlmread([curr_dir "/dataset/otto-group-product-classification-challenge/oct_test.csv"]); 

Y = dlmread([curr_dir "/dataset/otto-group-product-classification-challenge/oct_y.csv"]);

#[Xtrain,ytrain,Xval,yval] = splitTrainValidation(train_data,Y,0.8,shuffle=1);

## processing multiclass 
y_class = unique (Y); 
printf(">>>> found %i classes .. proceeding One vs. All ..  \n", length(y_class) );

probs_vect = zeros(size(test_data,1) , length(y_class) );

tic();
for j = 1:length(y_class) 
  printf(">>> processing Class%i  ..  \n", j );
  ytrain_j = ( Y == j  );

  ## gid                                                                                                                                                                             
  printf("|--> SVM ...\n");

  [C gamma, epsilon, nu] = getBestParamSVM(type = 1 , train_data , ytrain_j); ### errato ... devi usare xvalidation ... TODO 
  printf(">>  [label = %i] found analitically C=%f , gamma=%f , ... \n" , j , C, gamma(2) );
  
  fflush(stdout);

  ## training and prediction
  model = svmtrain( double(ytrain_j) , double(train_data), sprintf('-t 2 -g %g -c %g -b 1', gamma(2),C)  );
  #model2 = svmtrain( double(ytrain_j) , double(Xtrain), sprintf(' -g %g -c %g -b 1', gamma(2),C)  );  

  [pred_j, accuracy_j, probs_j] = svmpredict( test_data(:,1) , test_data , model , '-b 1 ');
  #[pred_val2, accuracy_xval2, probs_val2] = svmpredict( yval_j , Xval, model2 , '-b 1 ');
  
  probs_vect(:,j) = probs_j(:,1);
endfor 
tm = toc();

dlmwrite([curr_dir "/dataset/otto-group-product-classification-challenge/pred_SVM.csv"],probs_vect); 
