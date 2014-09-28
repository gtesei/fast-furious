#! /opt/local/bin/octave -qf 

%%%% making enviroment 
menv;

printf("|--> loading data ...\n");
solTrainX = dlmread([curr_dir "/linear_reg/__solTrainX.zat"]); 
solTestX = dlmread([curr_dir "/linear_reg/__solTestX.zat"]); 

solTestXtrans = dlmread([curr_dir "/linear_reg/__solTestXtrans.zat"]); 
solTrainXtrans = dlmread([curr_dir "/linear_reg/__solTrainXtrans.zat"]); 

solTestY = dlmread([curr_dir "/linear_reg/__solTestY.zat"]); 
solTrainY = dlmread([curr_dir "/linear_reg/__solTrainY.zat"]); 

## elimina le intestazioni del csv
solTrainX = solTrainX(2:end,:);  
solTestX = solTestX(2:end,:);  

solTestXtrans = solTestXtrans(2:end,:);  
solTrainXtrans = solTrainXtrans(2:end,:);  

solTestY = solTestY(2:end,:);  
solTrainY = solTrainY(2:end,:);

############################################################## Regularized Polynomial Regression 
## grid
perf_grid = zeros(2,5);

## without transformations , find best p, lambda 
printf("|--> data sets  WITHOUT transformations ...\n");
tic();
[p_opt_RMSE,lambda_opt_RMSE,RMSE_opt,grid]  = ... 
findOptPAndLambdaRegLin(solTrainX, solTrainY, solTestX, solTestY, ...
  p_vec = [1 2 3 4 5 6 7 8 9 10 12 20]' , ...
  lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' , ...
  verbose = 1, initGrid = [] , initStart = -1 , iter=1000);
tm = toc();
printf(">>>>> [predictors not transformed] found min RMSE=%f  with p=%i and lambda=%f \n", RMSE_opt , p_opt_RMSE , lambda_opt_RMSE );
i = 1;
perf_grid(i,1) = i; perf_grid(i,2) = p_opt_RMSE; perf_grid(i,3) = lambda_opt_RMSE; perf_grid(i,4) = RMSE_opt; perf_grid(i,5) = tm;

## with transformations , find best p, lambda 
printf("|--> data sets  WITH transformations ...\n");
tic();
[p_opt_RMSE,lambda_opt_RMSE,RMSE_opt,grid]  = ... 
findOptPAndLambdaRegLin(solTrainXtrans, solTrainY, solTestXtrans, solTestY, ...
  p_vec = [1 2 3 4 5 6 7 8 9 10 12 20]' , ...
  lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' , ...
  verbose = 1, initGrid = [] , initStart = -1, iter=1000);
tm = toc();
printf(">>>>> [predictors transformed] found min RMSE=%f  with p=%i and lambda=%f \n", RMSE_opt , p_opt_RMSE , lambda_opt_RMSE );
i = 2;
perf_grid(i,1) = i; perf_grid(i,2) = p_opt_RMSE; perf_grid(i,3) = lambda_opt_RMSE; perf_grid(i,4) = RMSE_opt; perf_grid(i,5) = tm;

disp(perf_grid);

############################################################## epsilon-SVR
## grid
perf_grid = zeros(1,6);

## without transformations
printf("|--> epsilon-SVR ...\n");

[C gamma, epsilon, nu] = getBestParamSVM(type = 1 , solTrainXtrans , solTrainY);

tic();
[C_opt_RMSE,gamma_opt_RMSE,epsilon_opt_RMSE,RMSE_opt,grid] = findOptCAndGammaAndEpsilon_SVR(solTrainXtrans, solTrainY, solTestXtrans, solTestY, ...
                                 featureScaled = 0 , ...
                                 C_vec = [C]' , ...
                                 g_vec = [gamma]' , ...
                                 e_vec = [0 0.2 0.6 0.8 1 2 3 5 10] , ...
                                 verbose = 1, initGrid = [] , initStart = -1 , doPlot=1);
tm = toc();
printf(">>>>> [nu-SVR, predictors transformed] found min. RMSE=%f  with C=%i , gamma=%f , epsilon=%f\n", RMSE_opt , C_opt_RMSE , gamma_opt_RMSE ,epsilon_opt_RMSE);
i = 1;
perf_grid(i,1) = i; perf_grid(i,2) = C_opt_RMSE; perf_grid(i,3) = gamma_opt_RMSE; perf_grid(i,4) = epsilon_opt_RMSE; perf_grid(i,5) = RMSE_opt; perf_grid(i,6) = tm;

disp(perf_grid);

############################################################## nu-SVR
## grid
perf_grid = zeros(1,6);

## without transformations
printf("|--> nu-SVR ...\n");

## C and gamma are computed analytically
[C , gamma , epsilon , nu] = getBestParamSVM(type = 1 , solTrainXtrans , solTrainY);

tic();
[C_opt_RMSE,gamma_opt_RMSE,nu_opt_RMSE,RMSE_opt,grid] = findOptCAndGammaAndNu_SVR(solTrainXtrans, solTrainY, solTestXtrans, solTestY, ...
                                                                                              featureScaled = 0 , ...
                                                                                              C_vec = [C]' , ...
                                                                                              g_vec = [gamma]' , ...
                                                                                              n_vec = (0.1:0.05:1)' , ...
                                                                                              verbose = 1, initGrid = [] , initStart = -1 , doPlot=1);
tm = toc();
i = 1;
perf_grid(i,1) = i; perf_grid(i,2) = C_opt_RMSE; perf_grid(i,3) = gamma_opt_RMSE; perf_grid(i,4) = epsilon_opt_RMSE; perf_grid(i,5) = RMSE_opt; perf_grid(i,6) = tm;

disp(perf_grid);

############################################################## (fast-furiuos) Neural Networks
## grid
perf_grid = zeros(1,6);

## without transformations
printf("|--> Training Neural Networks ...\n");





