function [C_opt_RMSE,gamma_opt_RMSE,epsilon_opt_RMSE,RMSE_opt,grid] = ...
  findOptCAndGammaAndNu_SVR(Xtrain, ytrain, Xval, yval, featureScaled = 0 ,
  C_vec = [2^-5 2^-3 2^-1 2^1 2^3 2^5 2^7 2^11 2^15]' , 
  g_vec = [2^-7 2^-3 2^-1 2^1 2^2 2^3 2^5 2^7]' ,
  n_vec = [0.1 0.2 0.4 0.5 0.8 1] ,
  verbose = 1, initGrid = [] , initStart = -1 , doPlot=1)
    
  if (! featureScaled) 
    [Xtrain,mu,sigma] = treatContFeatures(Xtrain,1);
    [Xval,mu,sigma] = treatContFeatures(Xval,1,1,mu,sigma);
  endif 
  
  grid = [];
  gLen = 0;
  if (size(initGrid,1) == 0 | size(initGrid,2) == 0 | initStart < 0) 
    gLen = length(C_vec)*length(g_vec)*length(n_vec);
    grid = zeros(gLen,6);
  else 
    grid = initGrid;
    gLen = size(grid,1)
  endif 

  %% Finding ...
    i = 1; 
    for gIdx = 1:length(g_vec)
      for CIdx = 1:length(C_vec)
        for nIdx = 1:length(n_vec)

         if (size(initGrid,1) > 0 & i < initStart)
           i = i + 1;
           continue;
          endif

	      C = C_vec(CIdx);
	      gamma = g_vec(gIdx);
          nu = n_vec(nIdx);

          if (verbose)
            fprintf("|---------------------->  trying C=%f , gamma=%f , nu=%f... \n" , C,gamma,nu);
            fflush(stdout);
          endif

          ## training and prediction
          model = svmtrain( ytrain, Xtrain, sprintf('-s 4 -t 2 -g %g -c %g -n %g',gamma,C,nu) );

          ##model = svmtrain(ytrain, Xtrain, ['-s 4 -t 2 -g' gamma ' -c ' C ' -n ' nu ] );
         [pred_train, accuracy, decision_values] = svmpredict(ytrain, Xtrain, model);
         [pred_val, accuracy, decision_values] = svmpredict(yval, Xval, model);
        
	     grid(i,1) = i;
	     grid(i,2) = C;
	     grid(i,3) = gamma;
         grid(i,4) = nu;

	     RMSE = sqrt(MSE(pred_train, ytrain));
	     if (verbose)
	       printf("*** TRAIN STATS ***\n");
           printf("|-->  RMSE == %f \n",RMSE);
         endif
	     grid(i,5) = RMSE;

         RMSE = sqrt(MSE(pred_val, yval));
	     if (verbose)
	       printf("*** CV STATS ***\n");
           printf("|-->  RMSE == %f \n",RMSE);
	     endif
         grid(i,6) = RMSE;
	
	     i = i + 1;
         dlmwrite('_____nuSVR__grid_tmp.mat',grid);
	     fflush(stdout);
       endfor
     endfor
   endfor

   [RMSE_opt,RMSE_opt_idx] = min(grid(:,6));
   C_opt_RMSE = grid(RMSE_opt_idx,2);
   gamma_opt_RMSE = grid(RMSE_opt_idx,3);
   nu_opt_RMSE = grid(RMSE_opt_idx,4);

  ### print grid
  if (verbose)
    printf("*** GRID ***\n");
    fprintf('i \tC \t\tgamma \t\tnu \t\tRMSE(Train) \tRMSE(Val) \n');
    for i = 1:gLen
      fprintf('%i\t%f\t%f\t%f\t%f\t%f \n',
      i, grid(i,2), grid(i,3),grid(i,4),grid(i,5),grid(i,6) );
    endfor
    fprintf('>>>> found min RMSE=%f  with C=%i , gamma=%f , nu=%f \n', RMSE_opt , C_opt_RMSE , gamma_opt_RMSE , nu_opt_RMSE );
  endif
  
  if (doPlot)
    %%plot RMSE
    %subplot (1, 1, 1);
    plot(1:gLen, grid(:,5), 1:gLen, grid(:,6));
    title(sprintf('Validation Curve -- min RMSE=%f  with C=%i,gamma=%f,nu=%f', RMSE_opt ,...
                  C_opt_RMSE , gamma_opt_RMSE , nu_opt_RMSE));
    xlabel('i')
    ylabel('RMSE')
    max_X = gLen;
    max_Y = max( max(grid(:,6))  ,  max(grid(:,5)) ) * 1.1;
    min_Y = min( min(grid(:,6))  ,  min(grid(:,5)) ) * 0.9;
    axis([1 max_X min_Y max_Y]);
    legend('Train', 'Cross Validation');
  endif

endfunction
