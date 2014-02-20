function [h_opt,J_opt] = findOptHiddenLayers(Xtrain, ytrain, Xval, yval , lambda=1,neurons_hidden_layers=-1,verbose=1)

  [m_train,n] = size(Xtrain);
  num_label = length(unique(ytrain));
  if (length(ytrain) != m_train) error("m_train error") endif;
  s1 = n-1;
  if (neurons_hidden_layers > 0)
    s1 = neurons_hidden_layers;
  endif 
  step = 1;
  hl = 1:step:5; 
  printf("|-> findOptHiddenLayers: detected %i features and %i classes (lambda=%f) (m_train=%i) ... \n",s1,num_label,lambda,m_train);
  printf("|-> findOptHiddenLayers: setting  %i  neurons per layers... \n",s1);
  printf("|-> findOptHiddenLayers: setting  hidden layers = %i,%i ... %i ... \n",min(hl),min(hl)+step,max(hl));
  
  error_train = zeros(length(hl), 1);
  error_val = zeros(length(hl), 1);

  %% Finding ...
  for i = 1:length(hl)
    arch = [s1; ones(hl(i),1) .* s1 ;num_label]';
    NNMeta = buildNNMeta(arch);
    if (verbose)
      disp(NNMeta);
    endif 
            
    [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda , iter = 200, featureScaled = 1);
	pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
	pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
	acc_train = mean(double(pred_train == ytrain)) * 100;
    acc_val = mean(double(pred_val == yval)) * 100;
        
    error_train(i) = 100 - acc_train;
    error_val(i)   = 100 - acc_val;
  endfor

  [J_opt, h_opt] = min(error_val); 
  
  fprintf('\tHidden Layers \tTrain Error\tCross Validation Error\n');
    for i = 1:length(hl)
          fprintf('  \t%d\t\t%f\t%f\n', hl(i), error_train(i), error_val(i));
  endfor

  fprintf('Optimal Number of hidden layers ==  %i , Minimum Cost == %f \n', hl(h_opt) , J_opt);

  %%plot 
  plot(hl, error_train, hl, error_val);
  text(h_opt+1,J_opt+6,"Optimal Number of Hidden Layers","fontsize",10);
  line([h_opt,J_opt],[h_opt+1,J_opt+5],"linewidth",1);
  title(sprintf('Finding optimal number of Hidden Layers (lambda = %f , number of neurons per layer = %i)', lambda, s1));
  xlabel('Number of Neurons per Hidden Layer')
  ylabel('Error')
  max_X = max(hl);
  max_Y = max(max(error_train) , max(error_val));
  axis([0 max_X 0 max_Y]);
  legend('Train', 'Cross Validation')

endfunction 