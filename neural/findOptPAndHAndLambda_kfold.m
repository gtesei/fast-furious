function [p_opt_RMSE,h_opt_RMSE,lambda_opt_RMSE,RMSE_opt,grid] = ...
	 findOptPAndHAndLambda(Xtrain, ytrain, ...
			       featureScaled = 0 , scaleFeatures = 0 , ...
			       p_vec = [] , ...
			       h_vec = [1 2 3 4 5 6 7 8 9 10] , ...
			       lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10] , ...
			       verbose = 1, doPlot=1 , ...
			       initGrid = [] , initStart = -1 , ...   
			       iter = 200 , ...
			       regression = 1 , num_labels = 0 , k = 4) 
	 
  if (! featureScaled & scaleFeatures) 
    [Xtrain,mu,sigma] = treatContFeatures(Xtrain,1);
    [Xval,mu,sigma] = treatContFeatures(Xval,1,1,mu,sigma);
  elseif (! featureScaled & ! scaleFeatures) 
    Xtrain = [ones(size(Xtrain,1), 1), Xtrain]; % Add Ones
    Xval = [ones(size(Xval,1), 1), Xval]; % Add Ones
  end

  %% p_vec
  n = size(Xtrain,2);
  s0 = n-1;
  p_vec = s0:(floor(s0/2)):(2*s0);

  grid = [];
  gLen = 0;
  if (size(initGrid,1) == 0 | size(initGrid,2) == 0 | initStart < 0) 
    gLen = length(p_vec)*length(h_vec)*length(lambda_vec);
    grid = zeros(gLen,6);
  else 
    grid = initGrid;
    gLen = size(grid,1)
  end

  %% k-folds 
  folds = kfold_bclass(k=k,y=ytrain,seed=123); 

  %% Finding ...
  i = 1; 
  for pIdx = 1:length(p_vec)
    for hIdx = 1:length(h_vec)
      for lambdaIdx = 1:length(lambda_vec)
        if (size(initGrid,1) > 0 & i < initStart)
          i = i + 1;
          continue;
        end

	p = p_vec(pIdx);
	h = h_vec(hIdx);
        lambda = lambda_vec(lambdaIdx);

        if (verbose)
          fprintf("|---------------------->  trying p=%f , h=%f , lambda=%f... \n" , p,h,lambda);
          fflush(stdout);
        endif
        
        grid(i,1) = i;
        grid(i,2) = p;
        grid(i,3) = h;
        grid(i,4) = lambda;

        %% training and prediction
        if (regression) 
          error("TODO");
          
        else 
          roc_tes = zeros(1,k);
          roc_trs = zeros(1,k);
          for kf = 1:k
            xtr = Xtrain(folds != kf,);
            ytr = ytrain(folds != kf);
            xte = Xtrain(folds == kf);
            yte = ytrain(folds == kf);
 
            NNMeta = buildNNMeta([s0 (ones(h,1) .* p)' num_labels]');disp(NNMeta);
            [Theta] = trainNeuralNetwork(NNMeta, xtr, ytr, lambda , iter = iter, featureScaled = 1);
            pred_train = NNPredictMulticlass(NNMeta, Theta , xtr , featureScaled = 1);
            pred_val = NNPredictMulticlass(NNMeta, Theta , xte , featureScaled = 1);
            
            roc_trs = auc(probs=pred_train,labels=ytr);
            roc_tes(kf) = auc(probs=pred_val , labels=yte); 
          end 
          grid(i,5) = mean(roc_trs);
          grid(i,6) = mean(roc_tes);
        end 

        ## TODO ensemble: i ensembles (one for each tune grid point) , selectiong best tune option ...
	
	i = i + 1;
        dlmwrite('_____NN__grid_tmp.mat',grid);
	fflush(stdout);
      end
    end
  end

  [RMSE_opt,RMSE_opt_idx] = min(grid(:,6));
  p_opt_RMSE = grid(RMSE_opt_idx,2);
  h_opt_RMSE = grid(RMSE_opt_idx,3);
  lambda_opt_RMSE = grid(RMSE_opt_idx,4);

  if (! regression)
    [RMSE_opt,RMSE_opt_idx] = max(grid(:,6));
    p_opt_RMSE = grid(RMSE_opt_idx,2);
    h_opt_RMSE = grid(RMSE_opt_idx,3);
    lambda_opt_RMSE = grid(RMSE_opt_idx,4); 
  endif 

  ### print grid
  if (verbose)
    printf("*** GRID ***\n");
    if (regression)
      fprintf('i \tp \t\th \t\tlambda \t\tRMSE(Train) \tRMSE(Val) \n');
    else 
      fprintf('i \tp \t\th \t\tlambda \t\tAccuracy(Train) \tAccuracy(Val) \n');
    endif 
    for i = 1:gLen
      fprintf('%i\t%f\t%f\t%f\t%f\t%f \n',
	      i, grid(i,2), grid(i,3),grid(i,4),grid(i,5),grid(i,6) );
    endfor
    if (regression)
      fprintf('>>>> found min RMSE=%f  with p=%i , h=%f , lambda=%f \n', RMSE_opt , p_opt_RMSE , h_opt_RMSE , lambda_opt_RMSE );
    else 
      fprintf('>>>> found max AUC=%f  with p=%i , h=%f , lambda=%f \n', RMSE_opt , p_opt_RMSE , h_opt_RMSE , lambda_opt_RMSE );
    endif 
  endif
  
  if (doPlot)
    %subplot (1, 1, 1);
    plot(1:gLen, grid(:,5), 1:gLen, grid(:,6));
    if (regression)
      title(sprintf('Validation Curve -- min RMSE=%f  with p=%i,h=%f,lambda=%f', RMSE_opt ,...
                    p_opt_RMSE , h_opt_RMSE , lambda_opt_RMSE));
    else 
      title(sprintf('Validation Curve -- max AOC=%f  with p=%i,h=%f,lambda=%f', RMSE_opt ,...
                    p_opt_RMSE , h_opt_RMSE , lambda_opt_RMSE));
    endif 
    xlabel('i')
    if (regression)
      ylabel('RMSE')
    else 
      ylabel('AUC')
    endif 
    max_X = gLen;
    max_Y = max( max(grid(:,6))  ,  max(grid(:,5)) ) * 1.1;
    min_Y = min( min(grid(:,6))  ,  min(grid(:,5)) ) * 0.9;
    axis([1 max_X min_Y max_Y]);
    legend('Train', 'Cross Validation');
  endif

endfunction