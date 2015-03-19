#! /opt/local/bin/octave -qf 

%%%% making enviroment 
menv;

printf("|--> loading data <<digit recognizer>> ...\n");

train = dlmread([curr_dir "/dataset/digit-recognizer/train.csv"]); 
test = dlmread([curr_dir "/dataset/digit-recognizer/test.csv"]); 
%%sampleSub = dlmread([curr_dir "/dataset/digit-recognizer/test.csv"]); 

## elimina le intestazioni del csv
train = train(2:end,:);
labels = train(:,1:1);  
train = train(:,2:end);
test = test(2:end,:);

############################################################## Normalizing naming
cross_size = 0.7;
verbose = 1; 

printf("|--> performing feature scaling and normalization on train dataset and cross validation dataset ...\n");
printf("|--> splitting dataset into train set (%f) and cross validation set ...\n",cross_size);
rand_indices = randperm(size(train,1));
[Xtrain,ytrain,Xval,yval] = splitTrainValidation(train(rand_indices,:),labels(rand_indices),0.70);


[Xtrain,mu,sigma] = treatContFeatures(Xtrain,1);
[Xval,mu_val,sigma_val] = treatContFeatures(Xval,1,1,mu,sigma);

%%% TODO BETTER il feature scaling deve essere fatto unificando training set e cross validation set 

##[C , gamma , epsilon , nu] = getBestParamSVM(type = 1 , solTrainXtrans , solTrainY);

probs_matrix_train = zeros(size(Xtrain,1),10); 
probs_matrix_xval = zeros(size(Xval,1),10); 

for label = 0:9 
  [C , gamma , epsilon , nu] = getBestParamSVM(type = 1 , Xtrain , (ytrain == label) );
  if (verbose)
    fprintf("|---------------------->  [label = %i] found analitically C=%f , gamma=%f , ... \n" , label,C,gamma(2));
    fflush(stdout);
  endif

  ## training and prediction
  model = svmtrain( double(ytrain == label) , double(Xtrain), sprintf('-g %g -c %g -b 1',gamma(2),C)  );

  ##model = svmtrain(ytrain, Xtrain, ['-s 4 -t 2 -g' gamma ' -c ' C ' -n ' nu ] );
  [pred_train, accuracy_train, probs_train] = svmpredict( double(ytrain == label) , Xtrain, model , '-b 1');
  [pred_val, accuracy_xval, probs_val] = svmpredict( double(yval == label) , Xval, model , '-b 1 ');

  probs_matrix_train(:,(label+1)) = probs_train(:,2);
  probs_matrix_xval(:,(label+1)) = probs_val(:,2);

endfor 

[prob_min_train , ind_train_min] = min(probs_matrix_train')';
[prob_min_xval , ind_xval_min] = min(probs_matrix_xval')';

pred_train = (ind_train_min - 1)';
pred_val = (ind_xval_min - 1)';

acc_train = mean(double(pred_train == ytrain)) * 100;
acc_val = mean(double(pred_val == yval)) * 100;
if (verbose)
  printf("*** TRAIN STATS ***\n");
  printf("|-->  Accuracy == %f \n",acc_train);
endif

if (verbose)
  printf("*** CV STATS ***\n");
  printf("|-->  Accuracy == %f \n",acc_val);
endif
