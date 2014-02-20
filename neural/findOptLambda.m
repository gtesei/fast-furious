function [lambda_opt,J_opt] = findOptLambda(NNMeta, Xtrain, ytrain, Xval, yval , lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]')

  [m_train,n] = size(Xtrain);
  num_label = length(unique(ytrain));
  if (length(ytrain) != m_train) error("m_train error") endif;
  s1 = n-1; 
  printf("|-> findOptLambda: detected %i features and %i classes  (m_train=%i) ... \n",s1,num_label,m_train);
  printf("|-> findOptLambda: setting  %i  neurons per layers... \n",s1);
   
  error_train = zeros(length(lambda_vec), 1);
  error_val = zeros(length(lambda_vec), 1);

  %% Finding ...
  for lambdaIdx = 1:length(lambda_vec)
    %arch = [s1; ones(hl(i),1) .* s1 ;num_label]';
    %NNMeta = buildNNMeta(arch);disp(NNMeta);
            
    [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda_vec(lambdaIdx) , iter = 200, featureScaled = 1);
     pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
     pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
     acc_train = mean(double(pred_train == ytrain)) * 100;
     acc_val = mean(double(pred_val == yval)) * 100;
        
     error_train(lambdaIdx) = 100 - acc_train;
     error_val(lambdaIdx)   = 100 - acc_val;
  endfor

  [J_opt, lambda_opt] = min(error_val); 
  
  fprintf('\tLambda \tTrain Error\tCross Validation Error\n');
  for lambdaIdx = 1:length(lambda_vec)
          fprintf('  \t%f\t\t%f\t%f\n', lambda_vec(lambdaIdx), error_train(lambdaIdx), error_val(lambdaIdx));
  endfor

  fprintf('Optimal Number of lambda ==  %f , Minimum Cost == %f \n', lambda_vec(lambda_opt) , J_opt);

  %%plot 
  plot(lambda_vec, error_train, lambda_vec, error_val);
  %text(h_opt+1,J_opt+6,"Optimal Number of Hidden Layers","fontsize",10);
  %line([h_opt,J_opt],[h_opt+1,J_opt+5],"linewidth",1);
  title(sprintf('Finding optimal Lambda'));
  xlabel('Lambda')
  ylabel('Error')
  max_X = max(lambda_vec);
  max_Y = max(max(error_train) , max(error_val));
  axis([0 max_X 0 max_Y]);
  legend('Train', 'Cross Validation')

endfunction 