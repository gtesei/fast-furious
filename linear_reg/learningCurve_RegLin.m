function [error_train, error_val] = ...
    learningCurve_RegLin(Xtrain, ytrain, Xval, yval, lambda=0,p=1)

  m = size(Xtrain, 1);
  error_train = zeros(m, 1);
  error_val   = zeros(m, 1);
  
  [X_poly_train,mu,sigma] = treatContFeatures(Xtrain,p);
  [X_poly_val,mu_val,sigma_val] = treatContFeatures(Xval,p,1,mu,sigma);

  %% computing error_train, err_val
  for i = 1:m	
  	[theta] = trainLinearReg(X_poly_train(1:i,:), ytrain(1:i,:), lambda);
  	[error_train(i), grad_train] = linearRegCostFunction(X_poly_train(1:i,:), ytrain(1:i,:), theta, 0);
  	[error_val(i), grad_cv] = linearRegCostFunction(X_poly_val,   yval,   theta, 0);
  
  endfor 
  
  fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
  for i = 1:m
      fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
  endfor

  %%plot   
  max_X = m;
  max_Y = max(max(error_train) , max(error_val));
  min_Y = min(min(error_train) , min(error_val));
  plot(1:m, error_train, 1:m, error_val);
  title(sprintf('Learning Curve (lambda = %f, p = %i)', lambda,p))
  xlabel('Train data set size')
  ylabel('Error')
  axis([0 max_X 0 max_Y])
  legend('Train', 'Cross Validation')

endfunction 