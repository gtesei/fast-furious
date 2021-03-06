function [error_train, error_val] = ...
    learningCurve_RegLin(Xtrain, ytrain, Xval, yval, lambda=0,p=1)

  m = size(Xtrain, 1);
  b = ceil(m/500);
  samples = ceil(m/b);
  error_train = zeros(samples, 1);
  error_val   = zeros(samples, 1);
  
  [X_poly_train,mu,sigma] = treatContFeatures(Xtrain,p);
  [X_poly_val,mu_val,sigma_val] = treatContFeatures(Xval,p,1,mu,sigma);

  %% computing error_train, err_val
  for i = b:b:m
  	[theta] = trainLinearReg(X_poly_train(1:i,:), ytrain(1:i,:), lambda);
  	[error_train(i), grad_train] = linearRegCostFunction(X_poly_train(1:i,:), ytrain(1:i,:), theta, 0);
  	[error_val(i), grad_cv] = linearRegCostFunction(X_poly_val,   yval,   theta, 0);
  
  endfor 

  if (length(error_train) < 100)
    fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
    for i = b:b:m
      fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
    endfor
  endif

  %%plot   
  max_X = m;
  max_Y = max(max(error_train) , max(error_val));
  min_Y = min(min(error_train) , min(error_val));
  plot(b:b:m, error_train, b:b:m, error_val);
  title(sprintf('Learning Curve (lambda = %f, p = %i)', lambda,p))
  xlabel('Train data set size')
  ylabel('Error')
  axis([0 max_X 0 max_Y])
  legend('Train', 'Cross Validation')

endfunction 