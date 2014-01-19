function p = predictOneVsAll(all_theta, X)
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

 m = size(X, 1);
 num_labels = size(all_theta, 1);

 p = zeros(size(X, 1), 1);

 %X = [ones(m, 1) X] X is passsed already that way 

 score = sigmoid(X * all_theta');

 [max , _p] = max(score');
 p = _p';

endfunction 
