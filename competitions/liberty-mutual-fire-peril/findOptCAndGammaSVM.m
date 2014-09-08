function [C_opt_recall,g_opt_recall,C_opt_accuracy,g_opt_accuracy,C_opt_precision,g_opt_precision,C_opt_F1,g_opt_F1,grid] = ...
  findOptCAndGammaSVM(Xtrain, ytrain, Xval, yval, featureScaled = 0 , 
  C_vec = [2^-5 2^-3 2^-1 2^1 2^3 2^5 2^7 2^11 2^15]' , 
  g_vec = [2^-15 2^-11 2^-7 2^-3 2^-1 2^1 2^2 2^3 2^5 2^7]')
  
  if (! featureScaled) 
    [X_poly_train,mu,sigma] = treatContFeatures(Xtrain,1);
    [X_poly_val,mu_val,sigma_val] = treatContFeatures(Xval,1,1,mu,sigma);
  endif 
  
  grid = zeros(length(C_vec)*length(g_vec),11);

  %% Finding ...
  i = 1; 
  for CIdx = 1:length(C_vec)
  for gIdx = 1:length(g_vec)
  	fprintf("|---------------------->  trying C=%f , gamma=%f ... \n" , C_vec(CIdx),g_vec(gIdx));
  	
        fprintf(' do training \n' );
        fprintf('  do prediction \n');
    	
    	%%pred_train =predictLinearReg(X_poly_train,theta);
        %%pred_val = predictLinearReg(X_poly_val,theta);
        
        pred_train = ytrain;
        pred_val = yval;
        
        grid(i,1) = i;
        grid(i,2) = C_vec(CIdx);
        grid(i,3) = g_vec(gIdx);
        
        printf("*** TRAIN STATS ***\n");
	[F1,precision,recall,accuracy] = printClassMetrics(pred_train,ytrain);
	grid(i,4) = recall;
	grid(i,5) = accuracy;
	grid(i,6) = precision;
	grid(i,7) = F1;
  
  
	printf("*** CV STATS ***\n");
	[F1,precision,recall,accuracy] = printClassMetrics(pred_val,yval);
        grid(i,8) = recall;
	grid(i,9) = accuracy;
	grid(i,10) = precision;
	grid(i,11) = F1;
	
  	i = i + 1; 
  endfor
  endfor

  [recall_val_opt,recall_val_opt_idx] = min(grid(:,8));
  [accuracy_val_opt,accuracy_val_opt_idx] = min(grid(:,9));
  [precision_val_opt,precision_val_opt_idx] = min(grid(:,10));
  [F1_val_opt,F1_val_opt_idx] = min(grid(:,11));
  
  C_opt_recall = grid(recall_val_opt_idx,2);
  g_opt_recall = grid(recall_val_opt_idx,3);
  
  C_opt_accuracy = grid(accuracy_val_opt_idx,2);
  g_opt_accuracy = grid(accuracy_val_opt_idx,3);
  
  C_opt_precision = grid(precision_val_opt_idx,2);
  g_opt_precision = grid(precision_val_opt_idx,3);
  
  C_opt_F1 = grid(F1_val_opt_idx,2);
  g_opt_F1 = grid(F1_val_opt_idx,3);
  
  ### print grid 
  fprintf('i \tC \tgamma \tRecall(Train) \tAccuracy(Train) \tPrecision(Train) \tF1(Train) \tRecall(Val) \tAccuracy(Val) \tPrecision(Val) \tF1(Val)\n');
  for i = 1:length(C_vec)*length(g_vec)
        fprintf('\t%i\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n', 
        	i, grid(i,2), grid(i,3),grid(i,4),grid(i,5),grid(i,6),grid(i,7),grid(i,8),grid(i,9),grid(i,10),grid(i,11) );
  endfor

  fprintf('C_opt_recall ==  %f , g_opt_recall == %f \n', C_opt_recall , g_opt_recall);
  fprintf('C_opt_accuracy ==  %f , g_opt_accuracy == %f \n', C_opt_accuracy , g_opt_accuracy);
  fprintf('C_opt_precision ==  %f , g_opt_precision == %f \n', C_opt_precision , g_opt_precision);
  fprintf('C_opt_F1 ==  %f , g_opt_F1 == %f \n', C_opt_F1 , g_opt_F1);

  %%plot 
  plot(lambda_vec, error_train, lambda_vec, error_val);
  title(sprintf('Validation Curve (p = %f)', p));
  xlabel('Regression Parameter lambda')
  ylabel('Error')
  max_X = max(lambda_vec);
  max_Y = max(max(error_train) , max(error_val));
  axis([0 max_X 0 max_Y]);
  legend('Train', 'Cross Validation')

endfunction 