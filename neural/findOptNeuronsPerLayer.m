function [s_opt,J_opt] = findOptNeuronsPerLayer(Xtrain, ytrain, Xval, yval , lambda=1)

  [m_train,n] = size(Xtrain);
  num_label = unique(ytrain);
  hl = 3; step = 40;
  s = n:step:2*n; s1 = n-1;
  printf("|-> findOptNeuronsPerLayer: detected %i features and %i classes ... \n",s1,length(num_label));
  printf("|-> findOptNeuronsPerLayer: setting  %i  hidden layers... \n",hl);
  printf("|-> findOptNeuronsPerLayer: setting  s2...sL-1 = %i,%i ... %i  neurons per layers... \n",min(s),min(s)+step,max(s));
  
  error_train = zeros(length(s), 1);
  error_val = zeros(length(s), 1);

  %% Finding ...
  for i = 1:length(s)
        
        NNMeta = buildNNMeta([s1; ones(hl,1)*i ;num_label]); 
            
        [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda , iter = 60, featureScaled = 1);
	pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
	pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
	acc_train = mean(double(pred_train == ytrain)) * 100;
        acc_val = mean(double(pred_val == yval)) * 100;
        
        error_train(i) = 100 - acc_train;
        error_val(i)   = 100 - acc_val;
  endfor

  [J_opt, s_opt] = min(error_val); 
  
  fprintf('\tNeurons \tTrain Error\tCross Validation Error\n');
    for i = 1:length(s)
          fprintf('  \t%d\t\t%f\t%f\n', s(i), error_train(i), error_val(i));
  endfor

  fprintf('Optimal Number of neurons s ==  %i , Minimum Cost == %f \n', s_opt , J_opt);

  %%plot 
  plot(s, error_train, s, error_val);
  text(s_opt+1,J_opt+6,"Optimal Number of Neurons","fontsize",10);
  line([s_opt,J_opt],[s_opt+1,J_opt+5],"linewidth",1);
  title(sprintf('Finding optimal number of Neurons per Hidden Layer (lambda = %f , hidden layers = %i)', lambda, hl));
  xlabel('Number of Neurons per Hidden Layer')
  ylabel('Error')
  max_X = max(s);
  max_Y = max(max(error_train) , max(error_val));
  axis([0 max_X 0 max_Y]);
  legend('Train', 'Cross Validation')

endfunction 