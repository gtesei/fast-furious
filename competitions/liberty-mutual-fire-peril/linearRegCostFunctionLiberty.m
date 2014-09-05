function [J, grad] = linearRegCostFunctionLiberty(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
J = 0; % cost
grad = zeros(size(theta)); % gradient 

% ====================== 
tai = [0;theta(2:end,1)];
%J = 1/(2*m) * sum((X * theta - y) .^ 2) + (lambda/(2*m)) * (tai' * tai);
%grad =   (1/m)*(X' * (X * theta -y)) + (lambda/m)*tai;
                 
hx = X * theta;
gini = NormalizedWeightedGini(y,X(:,43),hx);
if (gini < 0)
  gini = 0.01;
elseif (gini > 1.5)
  gini = 1;
end
J = (1 - gini)^2 + (lambda/(2*m)) * (tai' * tai);

                                   
                                   
################# calcola grad numericamente
numgrad = zeros(size(theta));
perturb = zeros(size(theta));
e = 1e-4;
for p = 1:size(theta,1)
  % Set perturbation vector
                                   
  perturb(p) = e;
                                   
  theta1 = theta - perturb;
  hx1 = X * theta1;
  gini1 = NormalizedWeightedGini(y,X(:,43),hx1);
  if (gini1 < 0)
    gini1 = 0.01;
  elseif gini1 > 1.5
    gini1 = 1;
  end
  loss1 = (1 - gini1)^2 + (lambda/(2*m)) * (tai' * tai);
  
  theta2 = theta + perturb;
  hx2 = X * theta2;
  gini2 = NormalizedWeightedGini(y,X(:,43),hx2);
  if (gini2 < 0)
    gini2 = 0.01;
  elseif gini2 > 1.5
    gini2 = 1;
  end
                                            
  loss2 = (1 - gini2)^2 + (lambda/(2*m)) * (tai' * tai);
                                                                                  
                                   
  % Compute Numerical Gradient
  numgrad(p) = (loss2 - loss1) / (2*e);
  perturb(p) = 0;
end
                    disp(p); disp(J); fflush(stdout);
if (J == 0)
    disp(theta);
end

% =========================================================================

grad = numgrad(:);

endfunction 
