function [J, grad] = lrCostFunction(theta, X, y, lambda)
%   y can be 1 xor 0

 m = length(y);

 J = 0;
 grad = zeros(size(theta));

 hx = sigmoid(X * theta);
 tai = [0;theta(2:end,1)];

 J = (1/m) *  (   -1 * innerProduct_Y_1_X(y, log(hx))     -1 * innerProduct_Y_1_X( (1 .- y) , log(1 .- hx) )     + (lambda/2) * (tai' * tai)   );

 grad = 1/m * (X' * (hx -y)) + (lambda/m)*tai;
 grad = grad(:);

endfunction
