function [p_opt_RMSE,lambda_opt_RMSE,RMSE_opt,grid] = ...
  findOptPAndLambdaRegLin(Xtrain, ytrain, Xval, yval,  
  p_vec = [1 2 3 4 5 6 7 8 9 10 12 20 60]' , 
  lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' ,
  verbose = 1, initGrid = [] , initStart = -1, iter=60 , doPlot = 1)
  
  grid = [];
  gLen = 0;
  if (size(initGrid,1) == 0 | size(initGrid,2) == 0 | initStart < 0) 
    gLen = length(p_vec)*length(lambda_vec);
    grid = zeros(gLen,5);
  else 
    grid = initGrid;
    gLen = size(grid,1)
  endif 

  %% Finding ...
    i = 1; 
    for pIdx = 1:length(p_vec)
      for lambdaIdx = 1:length(lambda_vec)

        if (size(initGrid,1) > 0 & i < initStart)
          i = i + 1;
          continue;
        endif

	p = p_vec(pIdx);
	lambda = lambda_vec(lambdaIdx);    

	if (verbose)
  	  fprintf("|---------------------->  trying p=%f , lambda=%f ... \n" , p,lambda);
          fflush(stdout); 
	endif
	
	[X_poly_train,mu,sigma] = treatContFeatures(Xtrain,p);
        [X_poly_val,mu_val,sigma_val] = treatContFeatures(Xval,p,1,mu,sigma); 
        
	theta = trainLinearReg(X_poly_train, ytrain, lambda, iter=iter); 
	pred_train = predictLinearReg(X_poly_train,theta);
	pred_val = predictLinearReg(X_poly_val,theta);
        
	grid(i,1) = i;
	grid(i,2) = p;
	grid(i,3) = lambda;
	
	RMSE = sqrt(MSE(pred_train, ytrain));
	
	if (verbose)      
	  printf("*** TRAIN STATS ***\n");
	  printf("|-->  RMSE == %f \n",RMSE);
	endif 
	
	grid(i,4) = RMSE;
	
	RMSE = sqrt(MSE(pred_val, yval));
	
	if (verbose)
	  printf("*** CV STATS ***\n");
	  printf("|-->  RMSE == %f \n",RMSE);
	endif 
	
	grid(i,5) = RMSE;
	
	i = i + 1;
	dlmwrite('____lin_reg__grid_tmp.mat',grid);
	fflush(stdout); 
      endfor
    endfor


  [RMSE_opt,RMSE_opt_idx] = min(grid(:,5));
  
  p_opt_RMSE = grid(RMSE_opt_idx,2);
  lambda_opt_RMSE = grid(RMSE_opt_idx,3);
  
  ### print grid 
  if (verbose)
    printf("*** GRID ***\n");
    fprintf('i \tp \t\tlambda \t\tRMSE(Train) \tRMSE(Val) \n');
    for i = 1:gLen
      fprintf('%i\t%f\t%f\t%f\t%f\ \n',
              i, grid(i,2), grid(i,3),grid(i,4),grid(i,5) );
    endfor

    fprintf('>>>> found min RMSE=%f  with p=%i and lambda=%f \n', RMSE_opt , p_opt_RMSE , lambda_opt_RMSE );
  endif  
  
  if (doPlot)
	  %%plot RMSE
	  %subplot (1, 1, 1);
	  plot(1:gLen, grid(:,4), 1:gLen, grid(:,5));
	  title(sprintf('Validation Curve -- min RMSE=%f  with p=%i and lambda=%f', RMSE_opt , p_opt_RMSE , lambda_opt_RMSE ));
	  xlabel('i')
	  ylabel('RMSE')
	  max_X = gLen;
	  max_Y = max(max(max(grid(:,4) , grid(:,5)))) * 1.1;
	  min_Y = min(min(min(grid(:,4) , grid(:,5))));
	  axis([1 max_X min_Y max_Y]);
	  legend('Train', 'Cross Validation');
  endif 

endfunction 
