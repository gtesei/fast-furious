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

## grid 
perf_grid = zeros(10,5);

############################################################## Regularized Polynomial Regression 
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

############################################################## Neural Networks   
## without transformations , find best p, lambda 
printf("|--> loading data sets  without transformations ...\n");
tic();


