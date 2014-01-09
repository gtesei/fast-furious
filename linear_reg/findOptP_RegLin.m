function [p_opt,J_opt] = findOptP_RegLin(Xtrain, ytrain, Xval, yval, p_vec = [1 2 3 4 5 6 7 8 9 10 12 20 60]' , lambda=0)

  error_train = zeros(length(p_vec), 1);
  error_val = zeros(length(p_vec), 1);

  %% Finding ...
  for p = 1:length(p_vec)
        [X_poly_train,mu,sigma] = treatContFeatures(Xtrain,p);
        [X_poly_val,mu_val,sigma_val] = treatContFeatures(Xval,p,1,mu,sigma);
        
        theta = trainLinearReg(X_poly_train, ytrain, lambda);        
        [error_train(p), grad_train] = linearRegCostFunction(X_poly_train, ytrain, theta, lambda);
        [error_val(p), grad_cv] = linearRegCostFunction(X_poly_val, yval, theta, lambda);
  endfor

  p_opt = -1;
  J_opt = 1000000;
  for p = 1:length(p_vec)
        if (p == 1 | error_val(p) < J_opt) 
                J_opt = error_val(p);
                p_opt = p_vec(p);
        endif
  endfor 

  fprintf('Optimal Polynomial Degree p ==  %f , Minimum Cost == %f \n', p_opt , J_opt);

  %%plot 
  plot(1:length(p_vec), error_train, 1:length(p_vec), error_val);
  title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
  xlabel('Polynomial degree')
  ylabel('Error')
  max_X = length(p_vec);
  max_Y = max(max(error_train) , max(error_val));
  axis([0 max_X 0 max_Y]);
  legend('Train', 'Cross Validation')

endfunction 