#! /opt/local/bin/octave -qf 

%%%% making enviroment 
menv;

printf("|--> loading data ...\n");
#train_red = dlmread([curr_dir "/dataset/restaurant-revenue-prediction/oct_train_reduced.csv"]); 
#test_red = dlmread([curr_dir "/dataset/restaurant-revenue-prediction/oct_test_reduced.csv"]); 

train_red = dlmread([curr_dir "/dataset/restaurant-revenue-prediction/oct_train.csv"]); 
test_red = dlmread([curr_dir "/dataset/restaurant-revenue-prediction/oct_test.csv"]); 

y_train = dlmread([curr_dir "/dataset/restaurant-revenue-prediction/oct_y.csv"]); 
 
[Xtrain,ytrain,Xval,yval] = splitTrainValidation(train_red,y_train,0.8);

############################################################## Regularized Polynomial Regression 
## grid
perf_grid = zeros(2,5);

## find best p, lambda 
printf("|--> Regularized Polynomial Regression data sets ...\n");
tic();
[p_opt_RMSE,lambda_opt_RMSE,RMSE_opt_pr,grid]  = ... 
findOptPAndLambdaRegLin(Xtrain, ytrain, Xval, yval, ...
  p_vec = [1 2 3 4 5 6 7 8 9 10 12 20]' , ...
  lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' , ...
  verbose = 0, initGrid = [] , initStart = -1 , iter=1000);
tm = toc();
printf(">>>>> [Regularized Polynomial Regression] found min RMSE=%f  with p=%i and lambda=%f \n", RMSE_opt_pr , p_opt_RMSE , lambda_opt_RMSE );
i = 1;
perf_grid(i,1) = i; perf_grid(i,2) = p_opt_RMSE; perf_grid(i,3) = lambda_opt_RMSE; perf_grid(i,4) = RMSE_opt_pr; perf_grid(i,5) = tm;

############################################################## epsilon-SVR
## grid
perf_grid = zeros(1,6);

printf("|--> epsilon-SVR ...\n");

[C gamma, epsilon, nu] = getBestParamSVM(type = 1 , Xtrain , ytrain);

tic();
[C_opt_RMSE_e,gamma_opt_RMSE_e,epsilon_opt_RMSE,RMSE_opt_e,grid] = findOptCAndGammaAndEpsilon_SVR(Xtrain, ytrain, Xval, yval, ...
                                 featureScaled = 0 , ...
                                 C_vec = [C]' , ...
                                 g_vec = [gamma]' , ...
                                 e_vec = [0 0.2 0.6 0.8 1 2 3 5 10] , ...
                                 verbose = 0, initGrid = [] , initStart = -1 , doPlot=1);
tm = toc();
printf(">>>>> [epsilon-SVR] found min. RMSE=%f  with C=%i , gamma=%f , epsilon=%f\n", RMSE_opt_e , C_opt_RMSE_e , gamma_opt_RMSE_e ,epsilon_opt_RMSE);
i = 1;
perf_grid(i,1) = i; perf_grid(i,2) = C_opt_RMSE_e; perf_grid(i,3) = gamma_opt_RMSE_e; perf_grid(i,4) = epsilon_opt_RMSE; perf_grid(i,5) = RMSE_opt_e; perf_grid(i,6) = tm;

disp(perf_grid);

printf("|--> predicting ...\n");
model = svmtrain(y_train, train_red, sprintf('-s 3 -t 2 -g %g -c %g -p %g',gamma_opt_RMSE_e,C_opt_RMSE_e,epsilon_opt_RMSE));
[pred_epsilon_SVR, accuracy, decision_values] = svmpredict(test_red(:,1), test_red, model);

dlmwrite([curr_dir "/dataset/restaurant-revenue-prediction/pred_epsilon_SVR.csv"],pred_epsilon_SVR);

############################################################## nu-SVR
## grid
perf_grid = zeros(1,6);

printf("|--> nu-SVR ...\n");

## C and gamma are computed analytically
[C , gamma , epsilon , nu] = getBestParamSVM(type = 1 , Xtrain , ytrain);

tic();
[C_opt_RMSE_n,gamma_opt_RMSE_n,nu_opt_RMSE,RMSE_opt_n,grid] = findOptCAndGammaAndNu_SVR(Xtrain, ytrain, Xval, yval, ...
                                                                                              featureScaled = 0 , ...
                                                                                              C_vec = [C]' , ...
                                                                                              g_vec = [gamma]' , ...
                                                                                              n_vec = (0.1:0.05:1)' , ...
                                                                                              verbose = 0, initGrid = [] , initStart = -1 , doPlot=1);
tm = toc();
fflush(stdout);
i = 1;
perf_grid(i,1) = i; perf_grid(i,2) = C_opt_RMSE_n; perf_grid(i,3) = gamma_opt_RMSE_n; perf_grid(i,4) = nu_opt_RMSE; perf_grid(i,5) = RMSE_opt_n; perf_grid(i,6) = tm;

printf(">>>>> [nu-SVR] found min. RMSE=%f  with C=%i , gamma=%f , nu=%f\n", RMSE_opt_n , C_opt_RMSE_n , gamma_opt_RMSE_n ,nu_opt_RMSE);
i = 1;

disp(perf_grid);

printf("|--> predicting ...\n");
model = svmtrain( y_train, train_red, sprintf('-s 4 -t 2 -g %g -c %g -n %g',gamma_opt_RMSE_n,C_opt_RMSE_n,nu_opt_RMSE) );
[pred_nu_SVR, accuracy, decision_values] = svmpredict(test_red(:,1), test_red, model);

dlmwrite([curr_dir "/dataset/restaurant-revenue-prediction/pred_nu_SVR.csv"],pred_nu_SVR);

disp("\n\n***************************************** SUMMARY *****************************************");
printf(">>>>> [Regularized Polynomial Regression] found min RMSE=%f  with p=%i and lambda=%f \n", RMSE_opt_pr , p_opt_RMSE , lambda_opt_RMSE );
printf(">>>>> [epsilon-SVR] found min. RMSE=%f  with C=%i , gamma=%f , epsilon=%f\n", RMSE_opt_e , C_opt_RMSE_e , gamma_opt_RMSE_e ,epsilon_opt_RMSE);
printf(">>>>> [nu-SVR] found min. RMSE=%f  with C=%i , gamma=%f , nu=%f\n", RMSE_opt_n , C_opt_RMSE_n , gamma_opt_RMSE_n ,nu_opt_RMSE);
disp("*******************************************************************************************");
