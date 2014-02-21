function [J] = MAE(ypred, y)

m = length(y); % number of training examples

J = 1/(m) * sum(abs(ypred - y)) ;


endfunction 
