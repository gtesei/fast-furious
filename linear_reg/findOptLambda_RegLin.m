function [lambda_opt,J_opt] = findOptLambda_RegLin(Xtrain, ytrain, Xval, yval, lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' , p=5)

  error_train = zeros(length(lambda_vec), 1);
  error_val = zeros(length(lambda_vec), 1);

  %% Finding ...
  for lambdaIdx = 1:length(lambda_vec)
        [X_poly_train,mu,sigma] = treatContFeatures(Xtrain,p);
        [X_poly_val,mu_val,sigma_val] = treatContFeatures(Xval,p,1,mu,sigma);
        
        theta = trainLinearReg(X_poly_train, ytrain, lambda_vec(lambdaIdx), 400);
        [error_train(lambdaIdx), grad_train] = linearRegCostFunction(X_poly_train, ytrain, theta, 0);
        [error_val(lambdaIdx), grad_cv] =      linearRegCostFunction(X_poly_val,   yval,   theta, 0);
  endfor

  lambda_opt = -1;
  J_opt = 1000000;
  for lambdaIdx = 1:length(lambda_vec)
        if (lambdaIdx == 1 | error_val(lambdaIdx) < J_opt) 
                J_opt = error_val(lambdaIdx);
                lambda_opt = lambda_vec(lambdaIdx);
        endif
  endfor 

  fprintf('Optimal Regression Parameter lambda ==  %f , Minimum Cost == %f \n', lambda_opt , J_opt);

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