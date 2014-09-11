function [C_opt_recall,g_opt_recall,C_opt_accuracy,g_opt_accuracy,C_opt_precision,g_opt_precision,C_opt_F1,g_opt_F1,grid] = ...
  findOptCAndGammaSVM(Xtrain, ytrain, Xval, yval, featureScaled = 0 , 
  C_vec = [2^-5 2^-3 2^-1 2^1 2^3 2^5 2^7 2^11 2^15]' , 
  g_vec = [2^-15 2^-11 2^-7 2^-3 2^-1 2^1 2^2 2^3 2^5 2^7]' ,
  verbose = 1, initGrid = [] , initStart = -1)
    
  if (! featureScaled) 
    [Xtrain,mu,sigma] = treatContFeatures(Xtrain,1);
    [Xval,mu,sigma] = treatContFeatures(Xval,1,1,mu,sigma);
  endif 
  
  grid = [];
  gLen = 0;
  if (size(initGrid,1) == 0 | size(initGrid,2) == 0 | initStart < 0) 
    gLen = length(C_vec)*length(g_vec);
    grid = zeros(gLen,11);
  else 
    grid = initGrid;
    gLen = size(grid,1)
  endif 

  %% Finding ...
  if (size(initGrid,1) == 0 | size(initGrid,2) == 0 | initStart < 0) 
    i = 1; 
    for gIdx = 1:length(g_vec)
      for CIdx = 1:length(C_vec)

	C = C_vec(CIdx);
	gamma = g_vec(gIdx);    

	if (verbose)
  	  fprintf("|---------------------->  trying C=%f , gamma=%f ... \n" , C,gamma);
	endif

	## training and prediction 
	model = svmtrain(ytrain, Xtrain, sprintf('-s 0 -t 2 -g %g -c %g',gamma,C));    	
	[pred_train, accuracy, decision_values] = svmpredict(ytrain, Xtrain, model);
	[pred_val, accuracy, decision_values] = svmpredict(yval, Xval, model);
        
	grid(i,1) = i;
	grid(i,2) = C;
	grid(i,3) = gamma;
	
	if (verbose)      
	  printf("*** TRAIN STATS ***\n");
	endif 
	[F1,precision,recall,accuracy] = printClassMetrics(pred_train,ytrain,verbose);
	grid(i,4) = recall;
	grid(i,5) = accuracy;
	grid(i,6) = precision;
	grid(i,7) = F1;
	
	if (verbose)
	  printf("*** CV STATS ***\n");
	endif 
	[F1,precision,recall,accuracy] = printClassMetrics(pred_val,yval,verbose);
	grid(i,8) = recall;
	grid(i,9) = accuracy;
	grid(i,10) = precision;
	grid(i,11) = F1;
	
	i = i + 1;
	dlmwrite('grid_tmp.mat',grid);
	fflush(stdout); 
      endfor
    endfor
  else
      for i = initStart:gLen

	C = grid(i,2);
	gamma = grid(i,3);    

	if (verbose)
  	  fprintf("|---------------------->  trying C=%f , gamma=%f ... \n" , C,gamma);
	endif

	## training and prediction 
	model = svmtrain(ytrain, Xtrain, sprintf('-s 0 -t 2 -g %g -c %g',gamma,C));    	
	[pred_train, accuracy, decision_values] = svmpredict(ytrain, Xtrain, model);
	[pred_val, accuracy, decision_values] = svmpredict(yval, Xval, model);
        
	if (verbose)      
	  printf("*** TRAIN STATS ***\n");
	endif 
	[F1,precision,recall,accuracy] = printClassMetrics(pred_train,ytrain,verbose);
	grid(i,4) = recall;
	grid(i,5) = accuracy;
	grid(i,6) = precision;
	grid(i,7) = F1;
	
	if (verbose)
	  printf("*** CV STATS ***\n");
	endif 
	[F1,precision,recall,accuracy] = printClassMetrics(pred_val,yval,verbose);
	grid(i,8) = recall;
	grid(i,9) = accuracy;
	grid(i,10) = precision;
	grid(i,11) = F1;
	
	i = i + 1;
	dlmwrite('grid_tmp.mat',grid);
	fflush(stdout); 
      endfor
  endif 

  [recall_val_opt,recall_val_opt_idx] = max(grid(:,8));
  [accuracy_val_opt,accuracy_val_opt_idx] = max(grid(:,9));
  [precision_val_opt,precision_val_opt_idx] = max(grid(:,10));
  [F1_val_opt,F1_val_opt_idx] = max(grid(:,11));
  
  C_opt_recall = grid(recall_val_opt_idx,2);
  g_opt_recall = grid(recall_val_opt_idx,3);
  
  C_opt_accuracy = grid(accuracy_val_opt_idx,2);
  g_opt_accuracy = grid(accuracy_val_opt_idx,3);
  
  C_opt_precision = grid(precision_val_opt_idx,2);
  g_opt_precision = grid(precision_val_opt_idx,3);
  
  C_opt_F1 = grid(F1_val_opt_idx,2);
  g_opt_F1 = grid(F1_val_opt_idx,3);
  
  ### print grid 
  if (verbose)
    printf("*** GRID ***\n");
    fprintf('i \tC \t\tgamma \t\tRecall(Train) \tAcc.(Train) \tPrec.(Train) \tF1(Train) \tRecall(Val) \tAcc.(Val) \tPrec.(Val) \tF1(Val)\n');
    for i = 1:gLen
      fprintf('%i\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n',
              i, grid(i,2), grid(i,3),grid(i,4),grid(i,5),grid(i,6),grid(i,7),grid(i,8),grid(i,9),grid(i,10),grid(i,11) );
    endfor

    fprintf('C_opt_recall ==  %f , g_opt_recall == %f , i ==%i \n', C_opt_recall , g_opt_recall,recall_val_opt_idx);
    fprintf('C_opt_accuracy ==  %f , g_opt_accuracy == %f , i ==%i  \n', C_opt_accuracy , g_opt_accuracy,accuracy_val_opt_idx);
    fprintf('C_opt_precision ==  %f , g_opt_precision == %f , i ==%i  \n', C_opt_precision , g_opt_precision,precision_val_opt_idx);
    fprintf('C_opt_F1 ==  %f , g_opt_F1 == %f , i ==%i  \n', C_opt_F1 , g_opt_F1,F1_val_opt_idx);
  endif  
  
  %%plot recall
  subplot (2, 2, 1);
  plot(1:gLen, grid(:,4), 1:gLen, grid(:,8));
  title(sprintf('Validation Curve -- Recall ' ));
  xlabel('i')
  ylabel('Recall')
  max_X = gLen;
  max_Y = max(max(max(grid(:,4) , grid(:,8)))) * 1.1;
  min_Y = min(min(min(grid(:,4) , grid(:,8))));
  axis([1 max_X min_Y max_Y]);
  legend('Train', 'Cross Validation')
  
  %%plot accuracy
  subplot (2, 2, 2);
  plot(1:gLen, grid(:,5), 1:gLen, grid(:,9));
  title(sprintf('Validation Curve -- Accuracy ' ));
  xlabel('i');
  ylabel('Accuracy');
  max_X = gLen;
  max_Y = max(max(max(grid(:,5) , grid(:,9)))) * 1.1;
  min_Y = min(min(min(grid(:,5) , grid(:,9))));
  axis([0 max_X min_Y max_Y]);
  legend('Train', 'Cross Validation');
  
  %%plot precision
  subplot (2, 2, 3);
  plot(1:gLen, grid(:,6), 1:gLen, grid(:,10));
  title(sprintf('Validation Curve -- Precision ' ));
  xlabel('i');
  ylabel('Precision');
  max_X = gLen;
  max_Y = max(max(max(grid(:,6) , grid(:,10)))) * 1.1;
  min_Y = min(min(min(grid(:,6) , grid(:,10))));
  axis([0 max_X min_Y max_Y]);
  legend('Train', 'Cross Validation')
  
  %%plot F1
  subplot (2, 2, 4);
  plot(1:gLen, grid(:,7), 1:gLen, grid(:,11));
  title(sprintf('Validation Curve -- F1 ' ));
  xlabel('i');
  ylabel('F1');
  max_X = gLen;
  max_Y = max(max(max(grid(:,7) , grid(:,11)))) * 1.1;
  min_Y = min(min(min(grid(:,7) , grid(:,11))));
  axis([0 max_X min_Y max_Y]);
  legend('Train', 'Cross Validation');

endfunction 
