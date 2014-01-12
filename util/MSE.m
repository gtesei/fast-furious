function [J] = MSE(ypred, y)

m = length(y); % number of training examples

J = 1/(m) * sum((ypred - y) .^ 2) ;


endfunction 
