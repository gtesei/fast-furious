function [p_opt,J_opt] = findOptP_RegLinLiberty4NormReg(Xtrain, ytrain, Xval, yval, p_vec = [1 2 6 8 9 10 12 15 20]' , lambda=0 , var11Train,var11Val)

  error_train = zeros(length(p_vec), 1);
  error_val = zeros(length(p_vec), 1);

  %% Finding ...
  for p = 1:length(p_vec)
        [X_poly_train,mu,sigma] = treatContFeatures(Xtrain,p_vec(p));
        [X_poly_val,mu_val,sigma_val] = treatContFeatures(Xval,p_vec(p),1,mu,sigma);
        
        theta = trainLinearReg(X_poly_train, ytrain, lambda);
        
        pred_train =predictLinearReg(X_poly_train,theta);
        pred_val = predictLinearReg(X_poly_val,theta);
                                         
        error_train(p) = NormalizedWeightedGini(ytrain,var11Train,pred_train);
        error_val(p)   = NormalizedWeightedGini(yval,var11Val,pred_val);
                                         
  endfor

  [J_opt, p_opt] = max(error_train);
  
  fprintf('Polynomial Degree \tTrain Error\tCross Validation Error\n');
    for i = 1:length(p_vec)
          fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
  endfor

  fprintf('Optimal Polynomial Degree p ==  %f , Max NormalizedWeightedGini  == %f \n', p_opt , J_opt);

  %%plot 
  plot(p_vec, error_train, p_vec, error_val);
  text(p_opt+1,J_opt+6,"Optimal Polynomial Degree","fontsize",10);
  line([p_opt,J_opt],[p_opt+1,J_opt+5],"linewidth",1);
  title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
  xlabel('Polynomial degree')
  ylabel('Error')
  max_X = max(p_vec);
  max_Y = max(max(error_train) , max(error_val));
  axis([0 max_X 0 max_Y]);
  legend('Train', 'Cross Validation')

endfunction 